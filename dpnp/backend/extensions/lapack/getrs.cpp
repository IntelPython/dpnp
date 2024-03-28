//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
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

#include "getrs.hpp"
#include "linalg_exceptions.hpp"
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

typedef sycl::event (*getrs_impl_fn_ptr_t)(sycl::queue,
                                           oneapi::mkl::transpose,
                                           const std::int64_t,
                                           const std::int64_t,
                                           char *,
                                           std::int64_t,
                                           std::int64_t *,
                                           char *,
                                           std::int64_t,
                                           std::vector<sycl::event> &,
                                           const std::vector<sycl::event> &);

static getrs_impl_fn_ptr_t getrs_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event getrs_impl(sycl::queue exec_q,
                              oneapi::mkl::transpose trans,
                              const std::int64_t n,
                              const std::int64_t nrhs,
                              char *in_a,
                              std::int64_t lda,
                              std::int64_t *ipiv,
                              char *in_b,
                              std::int64_t ldb,
                              std::vector<sycl::event> &host_task_events,
                              const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    T *b = reinterpret_cast<T *>(in_b);

    const std::int64_t scratchpad_size =
        mkl_lapack::getrs_scratchpad_size<T>(exec_q, trans, n, nrhs, lda, ldb);
    T *scratchpad = nullptr;

    std::stringstream error_msg;
    std::int64_t info = 0;
    bool is_exception_caught = false;

    sycl::event getrs_event;
    try {
        scratchpad = sycl::malloc_device<T>(scratchpad_size, exec_q);

        getrs_event = mkl_lapack::getrs(
            exec_q,
            trans, // Specifies the operation: whether or not to transpose
                   // matrix A. Can be 'N' for no transpose, 'T' for transpose,
                   // and 'C' for conjugate transpose.
            n,     // The order of the square matrix A
                   // and the number of rows in matrix B (0 ≤ n).
                   // It must be a non-negative integer.
            nrhs,  // The number of right-hand sides,
                   // i.e., the number of columns in matrix B (0 ≤ nrhs).
            a,     // Pointer to the square matrix A (n x n).
            lda,   // The leading dimension of matrix A, must be at least max(1,
                   // n). It must be at least max(1, n).
            ipiv, // Pointer to the output array of pivot indices that were used
                  // during factorization (n, ).
            b,    // Pointer to the matrix B of right-hand sides (ldb, nrhs).
            ldb,  // The leading dimension of matrix B, must be at least max(1,
                  // n).
            scratchpad, // Pointer to scratchpad memory to be used by MKL
                        // routine for storing intermediate results.
            scratchpad_size, depends);
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
        else if (info > 0) {
            is_exception_caught = false;
            if (scratchpad != nullptr) {
                sycl::free(scratchpad, exec_q);
            }
            throw LinAlgError("The solve could not be completed.");
        }
        else {
            error_msg << "Unexpected MKL exception caught during getrs() "
                         "call:\nreason: "
                      << e.what() << "\ninfo: " << e.info();
        }
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg << "Unexpected SYCL exception caught during getrs() call:\n"
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
        cgh.depends_on(getrs_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad]() { sycl::free(scratchpad, ctx); });
    });
    host_task_events.push_back(clean_up_event);
    return getrs_event;
}

std::pair<sycl::event, sycl::event>
    getrs(sycl::queue exec_q,
          dpctl::tensor::usm_ndarray a_array,
          dpctl::tensor::usm_ndarray ipiv_array,
          dpctl::tensor::usm_ndarray b_array,
          const std::vector<sycl::event> &depends)
{
    const int a_array_nd = a_array.get_ndim();
    const int b_array_nd = b_array.get_ndim();
    const int ipiv_array_nd = ipiv_array.get_ndim();

    if (a_array_nd != 2) {
        throw py::value_error(
            "The LU-factorized array has ndim=" + std::to_string(a_array_nd) +
            ", but a 2-dimensional array is expected.");
    }
    if (b_array_nd > 2) {
        throw py::value_error(
            "The right-hand sides array has ndim=" +
            std::to_string(b_array_nd) +
            ", but a 1-dimensional or a 2-dimensional array is expected.");
    }
    if (ipiv_array_nd != 1) {
        throw py::value_error("The array of pivot indices has ndim=" +
                              std::to_string(ipiv_array_nd) +
                              ", but a 1-dimensional array is expected.");
    }

    const py::ssize_t *a_array_shape = a_array.get_shape_raw();
    const py::ssize_t *b_array_shape = b_array.get_shape_raw();

    if (a_array_shape[0] != a_array_shape[1]) {
        throw py::value_error("The LU-factorized array must be square,"
                              " but got a shape of (" +
                              std::to_string(a_array_shape[0]) + ", " +
                              std::to_string(a_array_shape[1]) + ").");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q,
                                             {a_array, b_array, ipiv_array}))
    {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(a_array, b_array)) {
        throw py::value_error("The LU-factorized and right-hand sides arrays "
                              "are overlapping segments of memory");
    }

    bool is_a_array_c_contig = a_array.is_c_contiguous();
    bool is_b_array_f_contig = b_array.is_f_contiguous();
    bool is_ipiv_array_c_contig = ipiv_array.is_c_contiguous();
    if (!is_a_array_c_contig) {
        throw py::value_error("The LU-factorized array "
                              "must be C-contiguous");
    }
    if (!is_b_array_f_contig) {
        throw py::value_error("The right-hand sides array "
                              "must be F-contiguous");
    }
    if (!is_ipiv_array_c_contig) {
        throw py::value_error("The array of pivot indices "
                              "must be C-contiguous");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int a_array_type_id =
        array_types.typenum_to_lookup_id(a_array.get_typenum());
    int b_array_type_id =
        array_types.typenum_to_lookup_id(b_array.get_typenum());

    if (a_array_type_id != b_array_type_id) {
        throw py::value_error("The types of the LU-factorized and "
                              "right-hand sides arrays are mismatched");
    }

    getrs_impl_fn_ptr_t getrs_fn = getrs_dispatch_vector[a_array_type_id];
    if (getrs_fn == nullptr) {
        throw py::value_error(
            "No getrs implementation defined for the provided type "
            "of the input matrix.");
    }

    auto ipiv_types = dpctl_td_ns::usm_ndarray_types();
    int ipiv_array_type_id =
        ipiv_types.typenum_to_lookup_id(ipiv_array.get_typenum());

    if (ipiv_array_type_id != static_cast<int>(dpctl_td_ns::typenum_t::INT64)) {
        throw py::value_error("The type of 'ipiv_array' must be int64.");
    }

    const std::int64_t n = b_array_shape[0];
    const std::int64_t nrhs = (b_array_nd > 1) ? b_array_shape[1] : 1;

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t ldb = std::max<size_t>(1UL, n);

    // Use transpose::T since the LU-factorized array is passed as C-contiguous.
    // For F-contiguous we would use transpose::N.
    oneapi::mkl::transpose trans = oneapi::mkl::transpose::T;

    char *a_array_data = a_array.get_data();
    char *b_array_data = b_array.get_data();
    char *ipiv_array_data = ipiv_array.get_data();

    std::int64_t *ipiv = reinterpret_cast<std::int64_t *>(ipiv_array_data);

    std::vector<sycl::event> host_task_events;
    sycl::event getrs_ev =
        getrs_fn(exec_q, trans, n, nrhs, a_array_data, lda, ipiv, b_array_data,
                 ldb, host_task_events, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {a_array, b_array, ipiv_array}, host_task_events);

    return std::make_pair(args_ev, getrs_ev);
}

template <typename fnT, typename T>
struct GetrsContigFactory
{
    fnT get()
    {
        if constexpr (types::GetrsTypePairSupportFactory<T>::is_defined) {
            return getrs_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_getrs_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<getrs_impl_fn_ptr_t, GetrsContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(getrs_dispatch_vector);
}
} // namespace lapack
} // namespace ext
} // namespace backend
} // namespace dpnp
