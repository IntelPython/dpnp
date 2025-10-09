//*****************************************************************************
// Copyright (c) 2023-2025, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
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

#include <exception>
#include <stdexcept>

#include <pybind11/pybind11.h>

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "utils/type_utils.hpp"

#include "common_helpers.hpp"
#include "gesv.hpp"
#include "gesv_common_utils.hpp"
#include "types_matrix.hpp"

namespace dpnp::extensions::lapack
{
namespace mkl_lapack = oneapi::mkl::lapack;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

using dpctl::tensor::alloc_utils::sycl_free_noexcept;
using ext::common::init_dispatch_vector;

typedef sycl::event (*gesv_impl_fn_ptr_t)(sycl::queue &,
                                          const std::int64_t,
                                          const std::int64_t,
                                          char *,
                                          char *,
                                          const std::vector<sycl::event> &);

static gesv_impl_fn_ptr_t gesv_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event gesv_impl(sycl::queue &exec_q,
                             const std::int64_t n,
                             const std::int64_t nrhs,
                             char *in_a,
                             char *in_b,
                             const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *a = reinterpret_cast<T *>(in_a);
    T *b = reinterpret_cast<T *>(in_b);

    const std::int64_t lda = std::max<size_t>(1UL, n);
    const std::int64_t ldb = std::max<size_t>(1UL, n);

    std::int64_t scratchpad_size = 0;
    sycl::event comp_event;
    std::int64_t *ipiv = nullptr;

    std::stringstream error_msg;
    bool is_exception_caught = false;

#if defined(USE_ONEMATH)
    // Use transpose::T if the LU-factorized array is passed as C-contiguous.
    // For F-contiguous we use transpose::N.
    // Since gesv takes F-contiguous as input, we use transpose::N.
    oneapi::mkl::transpose trans = oneapi::mkl::transpose::N;

    scratchpad_size = std::max(
        mkl_lapack::getrf_scratchpad_size<T>(exec_q, n, n, lda),
        mkl_lapack::getrs_scratchpad_size<T>(exec_q, trans, n, nrhs, lda, ldb));

#else
    scratchpad_size =
        mkl_lapack::gesv_scratchpad_size<T>(exec_q, n, nrhs, lda, ldb);

#endif // USE_ONEMATH

    T *scratchpad = helper::alloc_scratchpad<T>(scratchpad_size, exec_q);

    try {
        ipiv = helper::alloc_ipiv(n, exec_q);
    } catch (const std::exception &e) {
        if (scratchpad != nullptr)
            sycl_free_noexcept(scratchpad, exec_q);
        throw;
    }

#if defined(USE_ONEMATH)
    sycl::event getrf_event;
    try {
        getrf_event = mkl_lapack::getrf(
            exec_q,
            n,    // The order of the square matrix A (0 ≤ n).
                  // It must be a non-negative integer.
            n,    // The number of columns in the square matrix A (0 ≤ n).
                  // It must be a non-negative integer.
            a,    // Pointer to the square matrix A (n x n).
            lda,  // The leading dimension of matrix A.
                  // It must be at least max(1, n).
            ipiv, // Pointer to the output array of pivot indices.
            scratchpad, // Pointer to scratchpad memory to be used by MKL
                        // routine for storing intermediate results.
            scratchpad_size, depends);

        comp_event = mkl_lapack::getrs(
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
            scratchpad_size, {getrf_event});
    } catch (mkl_lapack::exception const &e) {
        is_exception_caught = true;
        gesv_utils::handle_lapack_exc(exec_q, lda, a, scratchpad_size,
                                      scratchpad, ipiv, e, error_msg);
    } catch (oneapi::mkl::computation_error const &e) {
        // TODO: remove this catch when gh-642(oneMath) is fixed
        // Workaround for oneMath
        // oneapi::mkl::computation_error is thrown instead of
        // oneapi::mkl::lapack::computation_error.
        if (scratchpad != nullptr)
            sycl_free_noexcept(scratchpad, exec_q);
        if (ipiv != nullptr)
            sycl_free_noexcept(ipiv, exec_q);
        throw LinAlgError("The input coefficient matrix is singular.");
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg << "Unexpected SYCL exception caught during getrf() or "
                     "getrs() call:\n"
                  << e.what();
    }
#else
    try {
        comp_event = mkl_lapack::gesv(
            exec_q,
            n,    // The order of the square matrix A
                  // and the number of rows in matrix B (0 ≤ n).
            nrhs, // The number of right-hand sides,
                  // i.e., the number of columns in matrix B (0 ≤ nrhs).
            a,    // Pointer to the square coefficient matrix A (n x n).
            lda,  // The leading dimension of a, must be at least max(1, n).
            ipiv, // The pivot indices that define the permutation matrix P;
                  // row i of the matrix was interchanged with row ipiv(i),
                  // must be at least max(1, n).
            b,    // Pointer to the right hand side matrix B (n x nrhs).
            ldb,  // The leading dimension of matrix B,
                  // must be at least max(1, n).
            scratchpad, // Pointer to scratchpad memory to be used by MKL
                        // routine for storing intermediate results.
            scratchpad_size, depends);
    } catch (mkl_lapack::exception const &e) {
        is_exception_caught = true;
        gesv_utils::handle_lapack_exc(exec_q, lda, a, scratchpad_size,
                                      scratchpad, ipiv, e, error_msg);
    } catch (sycl::exception const &e) {
        is_exception_caught = true;
        error_msg << "Unexpected SYCL exception caught during gesv() call:\n"
                  << e.what();
    }
#endif // USE_ONEMATH

    if (is_exception_caught) // an unexpected error occurs
    {
        if (scratchpad != nullptr)
            sycl_free_noexcept(scratchpad, exec_q);
        if (ipiv != nullptr)
            sycl_free_noexcept(ipiv, exec_q);
        throw std::runtime_error(error_msg.str());
    }

    sycl::event ht_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(comp_event);
        auto ctx = exec_q.get_context();
        cgh.host_task([ctx, scratchpad, ipiv]() {
            sycl_free_noexcept(scratchpad, ctx);
            sycl_free_noexcept(ipiv, ctx);
        });
    });

    return ht_ev;
}

std::pair<sycl::event, sycl::event>
    gesv(sycl::queue &exec_q,
         const dpctl::tensor::usm_ndarray &coeff_matrix,
         const dpctl::tensor::usm_ndarray &dependent_vals,
         const std::vector<sycl::event> &depends)
{
    const int coeff_matrix_nd = coeff_matrix.get_ndim();
    const int dependent_vals_nd = dependent_vals.get_ndim();

    const py::ssize_t *coeff_matrix_shape = coeff_matrix.get_shape_raw();
    const py::ssize_t *dependent_vals_shape = dependent_vals.get_shape_raw();

    constexpr int expected_coeff_matrix_ndim = 2;
    constexpr int min_dependent_vals_ndim = 1;
    constexpr int max_dependent_vals_ndim = 2;

    gesv_utils::common_gesv_checks(
        exec_q, coeff_matrix, dependent_vals, coeff_matrix_shape,
        dependent_vals_shape, expected_coeff_matrix_ndim,
        min_dependent_vals_ndim, max_dependent_vals_ndim);

    // Ensure `batch_size`, `n` and 'nrhs' are non-zero, otherwise return empty
    // events
    if (helper::check_zeros_shape(coeff_matrix_nd, coeff_matrix_shape) ||
        helper::check_zeros_shape(dependent_vals_nd, dependent_vals_shape))
    {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int coeff_matrix_type_id =
        array_types.typenum_to_lookup_id(coeff_matrix.get_typenum());

    gesv_impl_fn_ptr_t gesv_fn = gesv_dispatch_vector[coeff_matrix_type_id];
    if (gesv_fn == nullptr) {
        throw py::value_error(
            "No gesv implementation defined for the provided type "
            "of the coefficient matrix.");
    }

    char *coeff_matrix_data = coeff_matrix.get_data();
    char *dependent_vals_data = dependent_vals.get_data();

    const std::int64_t n = dependent_vals_shape[0];
    const std::int64_t nrhs =
        (dependent_vals_nd > 1) ? dependent_vals_shape[1] : 1;

    sycl::event gesv_ev = gesv_fn(exec_q, n, nrhs, coeff_matrix_data,
                                  dependent_vals_data, depends);

    sycl::event ht_ev = dpctl::utils::keep_args_alive(
        exec_q, {coeff_matrix, dependent_vals}, {gesv_ev});

    return std::make_pair(ht_ev, gesv_ev);
}

template <typename fnT, typename T>
struct GesvContigFactory
{
    fnT get()
    {
        if constexpr (types::GesvTypePairSupportFactory<T>::is_defined) {
            return gesv_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_gesv_dispatch_vector(void)
{
    init_dispatch_vector<gesv_impl_fn_ptr_t, GesvContigFactory>(
        gesv_dispatch_vector);
}
} // namespace dpnp::extensions::lapack
