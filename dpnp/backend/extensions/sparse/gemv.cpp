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
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#include <stdexcept>
#include <vector>

#include "gemv.hpp"

// oneMKL sparse BLAS
namespace mkl_sparse = oneapi::mkl::sparse;

namespace dpnp::extensions::sparse
{

// ---------------------------------------------------------------------------
// Type-dispatched implementation: y = alpha * op(A) * x + beta * y
// ---------------------------------------------------------------------------

template <typename T, typename intType>
static sycl::event
sparse_gemv_impl(sycl::queue &exec_q,
                 oneapi::mkl::transpose mkl_trans,
                 T alpha,
                 intType *row_ptr_ptr,
                 intType *col_ind_ptr,
                 T *values_ptr,
                 std::int64_t num_rows,
                 std::int64_t num_cols,
                 std::int64_t nnz,
                 T *x_ptr,
                 T beta,
                 T *y_ptr,
                 const std::vector<sycl::event> &depends)
{
    mkl_sparse::matrix_handle_t handle = nullptr;
    mkl_sparse::init_matrix_handle(&handle);

    auto ev_set = mkl_sparse::set_csr_data(
        exec_q, handle,
        num_rows, num_cols,
        oneapi::mkl::index_base::zero,
        row_ptr_ptr, col_ind_ptr, values_ptr,
        depends);

    // optimize_gemv performs internal analysis — amortises over repeated SpMV
    auto ev_opt = mkl_sparse::optimize_gemv(
        exec_q, mkl_trans, handle, {ev_set});

    auto ev_gemv = mkl_sparse::gemv(
        exec_q, mkl_trans,
        alpha, handle,
        x_ptr, beta, y_ptr,
        {ev_opt});

    // async release — waits for ev_gemv internally
    mkl_sparse::release_matrix_handle(exec_q, &handle, {ev_gemv});

    return ev_gemv;
}


// ---------------------------------------------------------------------------
// Python-facing function
// ---------------------------------------------------------------------------

std::pair<sycl::event, sycl::event>
sparse_gemv(sycl::queue &exec_q,
            const int trans,
            const double alpha,
            const dpctl::tensor::usm_ndarray &row_ptr,
            const dpctl::tensor::usm_ndarray &col_ind,
            const dpctl::tensor::usm_ndarray &values,
            const dpctl::tensor::usm_ndarray &x,
            const double beta,
            const dpctl::tensor::usm_ndarray &y,
            const std::int64_t num_rows,
            const std::int64_t num_cols,
            const std::int64_t nnz,
            const std::vector<sycl::event> &depends)
{
    // Map trans integer to oneMKL enum
    oneapi::mkl::transpose mkl_trans;
    switch (trans) {
        case 0: mkl_trans = oneapi::mkl::transpose::nontrans; break;
        case 1: mkl_trans = oneapi::mkl::transpose::trans; break;
        case 2: mkl_trans = oneapi::mkl::transpose::conjtrans; break;
        default:
            throw std::invalid_argument(
                "sparse_gemv: trans must be 0 (N), 1 (T), or 2 (C)");
    }

    int val_typenum = values.get_typenum();
    int idx_typenum = row_ptr.get_typenum();

    sycl::event gemv_ev;

    // Dispatch on value type x index type
    // oneMKL sparse BLAS supports float32, float64 (no complex yet)
    if (val_typenum == UAR_FLOAT) {
        auto alpha_f = static_cast<float>(alpha);
        auto beta_f  = static_cast<float>(beta);

        if (idx_typenum == UAR_INT32) {
            gemv_ev = sparse_gemv_impl<float, std::int32_t>(
                exec_q, mkl_trans, alpha_f,
                row_ptr.get_data<std::int32_t>(),
                col_ind.get_data<std::int32_t>(),
                values.get_data<float>(),
                num_rows, num_cols, nnz,
                x.get_data<float>(), beta_f,
                y.get_data<float>(), depends);
        }
        else if (idx_typenum == UAR_INT64) {
            gemv_ev = sparse_gemv_impl<float, std::int64_t>(
                exec_q, mkl_trans, alpha_f,
                row_ptr.get_data<std::int64_t>(),
                col_ind.get_data<std::int64_t>(),
                values.get_data<float>(),
                num_rows, num_cols, nnz,
                x.get_data<float>(), beta_f,
                y.get_data<float>(), depends);
        }
        else {
            throw std::runtime_error(
                "sparse_gemv: index dtype must be int32 or int64");
        }
    }
    else if (val_typenum == UAR_DOUBLE) {
        if (idx_typenum == UAR_INT32) {
            gemv_ev = sparse_gemv_impl<double, std::int32_t>(
                exec_q, mkl_trans, alpha,
                row_ptr.get_data<std::int32_t>(),
                col_ind.get_data<std::int32_t>(),
                values.get_data<double>(),
                num_rows, num_cols, nnz,
                x.get_data<double>(), beta,
                y.get_data<double>(), depends);
        }
        else if (idx_typenum == UAR_INT64) {
            gemv_ev = sparse_gemv_impl<double, std::int64_t>(
                exec_q, mkl_trans, alpha,
                row_ptr.get_data<std::int64_t>(),
                col_ind.get_data<std::int64_t>(),
                values.get_data<double>(),
                num_rows, num_cols, nnz,
                x.get_data<double>(), beta,
                y.get_data<double>(), depends);
        }
        else {
            throw std::runtime_error(
                "sparse_gemv: index dtype must be int32 or int64");
        }
    }
    else {
        throw std::runtime_error(
            "sparse_gemv: value dtype must be float32 or float64");
    }

    return std::make_pair(sycl::event{}, gemv_ev);
}


// ---------------------------------------------------------------------------
// Dispatch vector init (placeholder — matches blas convention)
// ---------------------------------------------------------------------------

void init_sparse_gemv_dispatch_vector(void)
{
    // No dispatch table needed for sparse_gemv since we do explicit
    // type switching in the function body (oneMKL sparse API uses
    // opaque handles, not templated dispatch tables).
    // This function exists to match the dpnp extension convention.
}

} // namespace dpnp::extensions::sparse
