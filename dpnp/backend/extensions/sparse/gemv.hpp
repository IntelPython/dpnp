//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include <dpctl4pybind11.hpp>

namespace dpnp::extensions::sparse
{

/**
 * sparse_gemv_init -- ONE-TIME setup per sparse matrix operator.
 *
 * Calls init_matrix_handle + set_csr_data + optimize_gemv.
 *
 * Returns a tuple of:
 *   - handle_ptr:   opaque matrix_handle_t cast to uintptr_t for safe
 *                   Python round-tripping.
 *   - val_type_id:  the dpctl typenum lookup id of the value dtype Tv.
 *                   Python MUST pass this back to sparse_gemv_compute so
 *                   the C++ layer can verify that x and y dtype match the
 *                   handle's value type.
 *   - event:        dependency event from optimize_gemv; the caller must
 *                   wait on it (or chain via depends) before the first
 *                   sparse_gemv_compute call.
 *
 * LIFETIME CONTRACT -- IMPORTANT:
 * The handle owns NO copies of the CSR arrays. The caller MUST keep
 * row_ptr, col_ind, and values USM allocations alive until
 * sparse_gemv_release has been called AND its returned event has
 * completed. Dropping any of them earlier is undefined behavior and
 * will produce silent memory corruption -- there is no runtime check.
 *
 * The Python wrapper (_CachedSpMV) enforces this contract by holding
 * a reference to the CSR matrix for the lifetime of the handle.
 */
extern std::tuple<std::uintptr_t, int, sycl::event>
sparse_gemv_init(sycl::queue &exec_q,
                 const int trans,
                 const dpctl::tensor::usm_ndarray &row_ptr,
                 const dpctl::tensor::usm_ndarray &col_ind,
                 const dpctl::tensor::usm_ndarray &values,
                 const std::int64_t num_rows,
                 const std::int64_t num_cols,
                 const std::int64_t nnz,
                 const std::vector<sycl::event> &depends);

/**
 * sparse_gemv_compute -- PER-ITERATION SpMV.
 *
 * Calls only oneapi::mkl::sparse::gemv using the pre-built handle.
 * Verifies that:
 *   - x and y are 1-D usm_ndarrays on a queue compatible with exec_q
 *   - x and y dtype match val_type_id (the handle's value type)
 *   - x and y shapes match op(A) dimensions, taking trans into account
 *     (op(A) is num_rows x num_cols for trans=N, num_cols x num_rows
 *     for trans={T,C})
 *   - y is writable and does not overlap x
 *
 * alpha and beta are passed as double and cast inside gemv_compute_impl
 * to the matrix value type. For complex Tv the cast drops the imaginary
 * part; callers needing complex scalars should keep alpha=1, beta=0
 * (the solver use case).
 *
 * Returns the gemv event. The caller is responsible for sequencing
 * subsequent work on the same queue; no host-side wait or host_task
 * keep-alive is performed.
 */
extern sycl::event
sparse_gemv_compute(sycl::queue &exec_q,
                    const std::uintptr_t handle_ptr,
                    const int val_type_id,
                    const int trans,
                    const double alpha,
                    const dpctl::tensor::usm_ndarray &x,
                    const double beta,
                    const dpctl::tensor::usm_ndarray &y,
                    const std::int64_t num_rows,
                    const std::int64_t num_cols,
                    const std::vector<sycl::event> &depends);

/**
 * sparse_gemv_release -- free the matrix_handle created by sparse_gemv_init.
 *
 * Must be called exactly once per handle, after all compute calls that
 * depend on it have completed. The returned event depends on the release,
 * so the caller can chain CSR buffer deallocation on it safely.
 */
extern sycl::event
sparse_gemv_release(sycl::queue &exec_q,
                    const std::uintptr_t handle_ptr,
                    const std::vector<sycl::event> &depends);

/**
 * Register the init (2-D on Tv x Ti) and compute (1-D on Tv) dispatch
 * tables. Called exactly once from PYBIND11_MODULE.
 */
extern void init_sparse_gemv_dispatch_tables(void);

} // namespace dpnp::extensions::sparse
