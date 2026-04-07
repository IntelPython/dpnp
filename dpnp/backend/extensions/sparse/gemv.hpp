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

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include <dpctl4pybind11.hpp>

namespace dpnp::extensions::sparse
{

/**
 * sparse_gemv_init -- ONE-TIME setup per sparse matrix operator.
 *
 * Calls init_matrix_handle + set_csr_data + optimize_gemv.
 * Returns the opaque matrix_handle_t cast to uintptr_t for safe
 * Python round-tripping, plus the dependency event from optimize_gemv
 * (caller must wait on it before calling sparse_gemv_compute).
 *
 * Lifetime: the handle owns NO data copies; all CSR arrays must remain
 * alive (in USM) until sparse_gemv_release is called.
 */
extern std::pair<std::uintptr_t, sycl::event>
sparse_gemv_init(sycl::queue                           &exec_q,
                 const int                              trans,
                 const dpctl::tensor::usm_ndarray      &row_ptr,
                 const dpctl::tensor::usm_ndarray      &col_ind,
                 const dpctl::tensor::usm_ndarray      &values,
                 const std::int64_t                     num_rows,
                 const std::int64_t                     num_cols,
                 const std::int64_t                     nnz,
                 const std::vector<sycl::event>        &depends);

/**
 * sparse_gemv_compute -- PER-ITERATION SpMV.
 *
 * Calls only oneapi::mkl::sparse::gemv using the pre-built handle.
 * alpha and beta are passed as double and cast inside gemv_compute_impl
 * to the matrix value type.
 */
extern std::pair<sycl::event, sycl::event>
sparse_gemv_compute(sycl::queue                           &exec_q,
                    const std::uintptr_t                   handle_ptr,
                    const int                              trans,
                    const double                           alpha,
                    const dpctl::tensor::usm_ndarray      &x,
                    const double                           beta,
                    const dpctl::tensor::usm_ndarray      &y,
                    const std::int64_t                     num_rows,
                    const std::int64_t                     num_cols,
                    const std::vector<sycl::event>        &depends);

/**
 * sparse_gemv_release -- free the matrix_handle created by sparse_gemv_init.
 *
 * Must be called exactly once per handle, after all compute calls
 * that depend on it have completed.
 */
extern sycl::event
sparse_gemv_release(sycl::queue                     &exec_q,
                    const std::uintptr_t             handle_ptr,
                    const std::vector<sycl::event>  &depends);

extern void init_sparse_gemv_dispatch_table(void);

} // namespace dpnp::extensions::sparse
