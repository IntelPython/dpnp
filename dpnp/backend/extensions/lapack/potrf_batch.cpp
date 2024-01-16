//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/type_utils.hpp"

#include "linalg_exceptions.hpp"
#include "potrf.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*potrf_batch_impl_fn_ptr_t)(
    sycl::queue,
    oneapi::mkl::uplo,
    std::int64_t,
    char *,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

static potrf_batch_impl_fn_ptr_t
    potrf_batch_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event potrf_batch_impl(sycl::queue exec_q,
                                    oneapi::mkl::uplo upper_lower,
                                    std::int64_t n,
                                    char *in_a,
                                    std::int64_t lda,
                                    std::int64_t stride_a,
                                    std::int64_t batch_size,
                                    std::vector<sycl::event> &host_task_events,
                                    const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);

    const std::int64_t scratchpad_size =
        oneapi::mkl::lapack::potrf_batch_scratchpad_size<T>(
            exec_q, upper_lower, n, lda, stride_a, batch_size);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool is_exception_caught = false;

    sycl::event potrf_batch_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        potrf_batch_event = oneapi::mkl::lapack::potrf_batch(
            exec_q,
            upper_lower, // An enumeration value of type oneapi::mkl::uplo:
                         // oneapi::mkl::uplo::upper for the upper triangular
                         // part; oneapi::mkl::uplo::lower for the lower
                         // triangular part.
            n,           // Order of each square matrix in the batch; (0 ≤ n).
            a,           // Pointer to the batch of matrices.
            lda,         // The leading dimension of `a`.
            stride_a,    // Stride between matrices: Element spacing between
                         // matrices in `a`.
            batch_size,  // Total number of matrices in the batch.
            scratchpad,  // Pointer to scratchpad memory to be used by MKL
                         // routine for storing intermediate results.
            scratchpad_size, depends);
    } catch (mkl_lapack::batch_error const &be) {
        // Get the indices of matrices within the batch that encountered an
        // error
        auto error_matrices_ids = be.ids();

        error_msg
            << "Matrix is not positive definite. Errors in matrices with IDs: ";
        for (size_t i = 0; i < error_matrices_ids.size(); ++i) {
            error_msg << error_matrices_ids[i];
            if (i < error_matrices_ids.size() - 1) {
                error_msg << ", ";
            }
        }
        error_msg << ".";

        sycl::free(scratchpad, exec_q);
        throw LinAlgError(error_msg.str().c_str());
    } catch (mkl_lapack::exception const &e) {
        is_exception_caught = true;
        info = e.info();

        if (info < 0) {
            error_msg << "Parameter number " << -info
                      << " had an illegal value.";
        }
        else if (info == scratchpad_size && e.detail() != 0) {
            error_msg
                << "Insufficient scratchpad size. Required size is at least "
                << e.detail();
        }
        else if (info != 0 && e.detail() == 0) {
            error_msg << "Error in batch processing. "
                         "Number of failed calculations: "
                      << info;
        }
        else {
            error_msg << "Unexpected MKL exception caught during potrf_batch() "
                         "call:\nreason: "
                      << e.what() << "\ninfo: " << e.info();
        }
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg
            << "Unexpected SYCL exception caught during potrf_batch() call:\n"
            << e.what();
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        if (scratchpad != nullptr) {
            sycl::free(scratchpad, exec_q);
        }
        throw std::runtime_error(error_msg.str());
    }

    sycl::event clean_up_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(potrf_batch_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);
    return potrf_batch_event;
}

std::pair<sycl::event, sycl::event>
    potrf_batch(sycl::queue q,
                dpctl::tensor::usm_ndarray a_array,
                std::int64_t n,
                std::int64_t stride_a,
                std::int64_t batch_size,
                const std::vector<sycl::event> &depends)
{
    const int a_array_nd = a_array.get_ndim();

    if (a_array_nd < 3) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(a_array_nd) +
            ", but a 3-dimensional or higher array is expected.");
    }

    const py::ssize_t *a_array_shape = a_array.get_shape_raw();

    if (a_array_shape[a_array_nd - 1] != a_array_shape[a_array_nd - 2]) {
        throw py::value_error(
            "The last two dimensions of the input array must be square,"
            " but got a shape of (" +
            std::to_string(a_array_shape[a_array_nd - 1]) + ", " +
            std::to_string(a_array_shape[a_array_nd - 2]) + ").");
    }

    bool is_a_array_c_contig = a_array.is_c_contiguous();
    if (!is_a_array_c_contig) {
        throw py::value_error("The input array "
                              "must be C-contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());

    potrf_batch_impl_fn_ptr_t potrf_batch_fn =
        potrf_batch_dispatch_vector[a_array_type_id];
    if (potrf_batch_fn == nullptr) {
        throw py::value_error(
            "No potrf_batch implementation defined for the provided type "
            "of the input matrix.");
    }

    char *a_array_data = a_array.get_data();
    const std::int64_t lda = std::max<size_t>(1UL, n);
    oneapi::mkl::uplo upper_lower = oneapi::mkl::uplo::upper;

    std::vector<sycl::event> host_task_events;
    sycl::event potrf_batch_ev =
        potrf_batch_fn(q, upper_lower, n, a_array_data, lda, stride_a,
                       batch_size, host_task_events, depends);

    sycl::event args_ev =
        dpctl::utils::keep_args_alive(q, {a_array}, host_task_events);

    return std::make_pair(args_ev, potrf_batch_ev);
}

template <typename fnT, typename T>
struct PotrfBatchContigFactory
{
    fnT get()
    {
        if constexpr (types::PotrfBatchTypePairSupportFactory<T>::is_defined) {
            return potrf_batch_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_potrf_batch_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<potrf_batch_impl_fn_ptr_t,
                                       PotrfBatchContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(potrf_batch_dispatch_vector);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
