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

#include "getrf.hpp"
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

typedef sycl::event (*getrf_batch_impl_fn_ptr_t)(
    sycl::queue,
    std::int64_t,
    char *,
    std::int64_t,
    std::int64_t,
    std::int64_t *,
    std::int64_t,
    std::int64_t,
    py::list,
    std::vector<sycl::event> &,
    const std::vector<sycl::event> &);

static getrf_batch_impl_fn_ptr_t
    getrf_batch_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event getrf_batch_impl(sycl::queue exec_q,
                                    std::int64_t n,
                                    char *in_a,
                                    std::int64_t lda,
                                    std::int64_t stride_a,
                                    std::int64_t *ipiv,
                                    std::int64_t stride_ipiv,
                                    std::int64_t batch_size,
                                    py::list dev_info,
                                    std::vector<sycl::event> &host_task_events,
                                    const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);

    const std::int64_t scratchpad_size =
        mkl_lapack::getrf_batch_scratchpad_size<T>(exec_q, n, n, lda, stride_a,
                                                   stride_ipiv, batch_size);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool is_exception_caught = false;

    sycl::event getrf_batch_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        getrf_batch_event = mkl_lapack::getrf_batch(
            exec_q,
            n, // The order of each square matrix in the batch; (0 ≤ n).
               // It must be a non-negative integer.
            n, // The number of columns in each matrix in the batch; (0 ≤ n).
               // It must be a non-negative integer.
            a, // Pointer to the batch of square matrices, each of size (n x n).
            lda,      // The leading dimension of each matrix in the batch.
            stride_a, // Stride between consecutive matrices in the batch.
            ipiv, // Pointer to the array of pivot indices for each matrix in
                  // the batch.
            stride_ipiv, // Stride between pivot indices: Spacing between pivot
                         // arrays in 'ipiv'.
            batch_size,  // Stride between pivot index arrays in the batch.
            scratchpad,  // Pointer to scratchpad memory to be used by MKL
                         // routine for storing intermediate results.
            scratchpad_size, depends);
    } catch (mkl_lapack::batch_error const &be) {
        // Get the indices of matrices within the batch that encountered an
        // error
        auto error_matrices_ids = be.ids();
        // Get the indices of the first zero diagonal elements of these matrices
        auto error_info = be.exceptions();

        for (size_t i = 0; i < error_matrices_ids.size(); ++i) {
            // Assign the index of the first zero diagonal element in each
            // error matrix to the corresponding index in 'dev_info'
            dev_info[error_matrices_ids[i]] = error_info[i];
        }
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
        else {
            error_msg << "Unexpected MKL exception caught during getrf_batch() "
                         "call:\nreason: "
                      << e.what() << "\ninfo: " << e.info();
        }
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg
            << "Unexpected SYCL exception caught during getrf_batch() call:\n"
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
        cgh.depends_on(getrf_batch_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);
    return getrf_batch_event;
}

std::pair<sycl::event, sycl::event>
    getrf_batch(sycl::queue exec_q,
                dpctl::tensor::usm_ndarray a_array,
                dpctl::tensor::usm_ndarray ipiv_array,
                py::list dev_info,
                std::int64_t n,
                std::int64_t stride_a,
                std::int64_t stride_ipiv,
                std::int64_t batch_size,
                const std::vector<sycl::event> &depends)
{
    const int a_array_nd = a_array.get_ndim();
    const int ipiv_array_nd = ipiv_array.get_ndim();

    if (a_array_nd < 3) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(a_array_nd) +
            ", but an array with ndim >= 3 is expected.");
    }

    if (ipiv_array_nd != 2) {
        throw py::value_error("The array of pivot indices has ndim=" +
                              std::to_string(ipiv_array_nd) +
                              ", but a 2-dimensional array is expected.");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q, {a_array, ipiv_array})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    bool is_a_array_c_contig = a_array.is_c_contiguous();
    if (!is_a_array_c_contig) {
        throw py::value_error("The input array "
                              "must be C-contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());

    getrf_batch_impl_fn_ptr_t getrf_batch_fn =
        getrf_batch_dispatch_vector[a_array_type_id];
    if (getrf_batch_fn == nullptr) {
        throw py::value_error(
            "No getrf_batch implementation defined for the provided type "
            "of the input matrix.");
    }

    auto ipiv_types = dpctl_td_ns::usm_ndarray_types();
    int ipiv_array_type_id =
        ipiv_types.typenum_to_lookup_id(ipiv_array.get_typenum());

    if (ipiv_array_type_id != static_cast<int>(dpctl_td_ns::typenum_t::INT64)) {
        throw py::value_error("The type of 'ipiv_array' must be int64.");
    }

    char *a_array_data = a_array.get_data();
    const std::int64_t lda = std::max<size_t>(1UL, n);

    char *ipiv_array_data = ipiv_array.get_data();
    std::int64_t *d_ipiv = reinterpret_cast<std::int64_t *>(ipiv_array_data);

    std::vector<sycl::event> host_task_events;
    sycl::event getrf_batch_ev = getrf_batch_fn(
        exec_q, n, a_array_data, lda, stride_a, d_ipiv, stride_ipiv, batch_size,
        dev_info, host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {a_array, ipiv_array}, host_task_events);

    return std::make_pair(args_ev, getrf_batch_ev);
}

template <typename fnT, typename T>
struct GetrfBatchContigFactory
{
    fnT get()
    {
        if constexpr (types::GetrfBatchTypePairSupportFactory<T>::is_defined) {
            return getrf_batch_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_getrf_batch_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<getrf_batch_impl_fn_ptr_t,
                                       GetrfBatchContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(getrf_batch_dispatch_vector);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
