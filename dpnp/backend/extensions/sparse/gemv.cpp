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

#include <sstream>
#include <stdexcept>
#include <vector>

#include <pybind11/pybind11.h>

// ext/common.hpp — dpctl_td_ns; mirrors every other dpnp extension
#include "ext/common.hpp"

// dpctl tensor validation and utility headers — same set as blas/gemm.cpp
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "gemv.hpp"

// oneMKL sparse BLAS
namespace mkl_sparse = oneapi::mkl::sparse;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

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
    // Validate that T is supported on this device (mirrors gemm_impl pattern)
    type_utils::validate_type_for_device<T>(exec_q);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    mkl_sparse::matrix_handle_t handle = nullptr;
    sycl::event gemv_ev;

    try {
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

        gemv_ev = mkl_sparse::gemv(
            exec_q, mkl_trans,
            alpha, handle,
            x_ptr, beta, y_ptr,
            {ev_opt});

        // async release — waits for gemv_ev internally
        mkl_sparse::release_matrix_handle(exec_q, &handle, {gemv_ev});

    } catch (oneapi::mkl::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during sparse_gemv() call:"
               "\nreason: "
            << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg
            << "Unexpected SYCL exception caught during sparse_gemv() call:\n"
            << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) {
        // Best-effort handle cleanup before re-raising
        if (handle != nullptr) {
            mkl_sparse::release_matrix_handle(exec_q, &handle, {});
        }
        throw std::runtime_error(error_msg.str());
    }

    return gemv_ev;
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
    // --- 1. ndim checks ---
    if (x.get_ndim() != 1) {
        throw py::value_error("sparse_gemv: x must be a 1-D array.");
    }
    if (y.get_ndim() != 1) {
        throw py::value_error("sparse_gemv: y must be a 1-D array.");
    }

    // --- 2. Queue compatibility (all USM arrays must share the same queue) ---
    if (!dpctl::utils::queues_are_compatible(
            exec_q,
            {row_ptr.get_queue(), col_ind.get_queue(),
             values.get_queue(),  x.get_queue(), y.get_queue()})) {
        throw py::value_error(
            "sparse_gemv: USM allocations are not compatible with the "
            "execution queue.");
    }

    // --- 3. Memory overlap: x and y must not alias ---
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(x, y)) {
        throw py::value_error(
            "sparse_gemv: input array x and output array y are overlapping "
            "segments of memory.");
    }

    // --- 4. Output writability and size ---
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(y);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        y, static_cast<std::size_t>(num_rows));

    // --- 5. Map trans integer to oneMKL enum ---
    oneapi::mkl::transpose mkl_trans;
    switch (trans) {
        case 0: mkl_trans = oneapi::mkl::transpose::nontrans;  break;
        case 1: mkl_trans = oneapi::mkl::transpose::trans;     break;
        case 2: mkl_trans = oneapi::mkl::transpose::conjtrans; break;
        default:
            throw std::invalid_argument(
                "sparse_gemv: trans must be 0 (N), 1 (T), or 2 (C)");
    }

    // --- 6. Type dispatch (value type x index type) ---
    // oneMKL sparse BLAS supports float32 and float64 (no complex yet)
    int val_typenum = values.get_typenum();
    int idx_typenum = row_ptr.get_typenum();

    sycl::event gemv_ev;

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

    // Keep all input/output USM arrays alive until gemv_ev completes
    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {row_ptr, col_ind, values, x, y}, {gemv_ev});

    return std::make_pair(args_ev, gemv_ev);
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
