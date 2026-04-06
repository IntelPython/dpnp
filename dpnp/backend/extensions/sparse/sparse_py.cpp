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
//
// Defines the dpnp.backend._sparse_impl pybind11 extension module.
// Provides oneMKL sparse BLAS operations on CSR matrices over dpctl USM arrays.
// Equivalent role to _cusparse for the SYCL/oneMKL backend.
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gemv.hpp"

namespace sparse_ns = dpnp::extensions::sparse;
namespace py = pybind11;

static void init_dispatch_vectors_tables(void)
{
    sparse_ns::init_sparse_gemv_dispatch_vector();
}

PYBIND11_MODULE(_sparse_impl, m)
{
    init_dispatch_vectors_tables();

    using arrayT     = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    // ------------------------------------------------------------------
    // _sparse_gemv — CSR SpMV:  y = alpha * op(A) * x + beta * y
    //
    // Equivalent to _cusparse.spMV_make_fast_matvec for the SYCL stack.
    // Backed by oneMKL sparse::gemv with set_csr_data + optimize_gemv so
    // matrix-handle analysis is amortised across repeated calls.
    // ------------------------------------------------------------------
    {
        m.def(
            "_sparse_gemv",
            [](sycl::queue &exec_q,
               const int trans,
               const double alpha,
               const arrayT &row_ptr,
               const arrayT &col_ind,
               const arrayT &values,
               const arrayT &x,
               const double beta,
               const arrayT &y,
               const std::int64_t num_rows,
               const std::int64_t num_cols,
               const std::int64_t nnz,
               const event_vecT &depends) {
                return sparse_ns::sparse_gemv(
                    exec_q, trans, alpha,
                    row_ptr, col_ind, values,
                    x, beta, y,
                    num_rows, num_cols, nnz, depends);
            },
            "CSR sparse matrix-vector product y = alpha*op(A)*x + beta*y "
            "via oneMKL sparse::gemv.\n\n"
            "Parameters\n"
            "----------\n"
            "sycl_queue : dpctl.SyclQueue\n"
            "trans      : int  0=N, 1=T, 2=C\n"
            "alpha      : float\n"
            "row_ptr    : usm_ndarray  CSR row offsets (int32 or int64)\n"
            "col_ind    : usm_ndarray  CSR column indices (int32 or int64)\n"
            "values     : usm_ndarray  CSR non-zeros (float32 or float64)\n"
            "x          : usm_ndarray  input vector\n"
            "beta       : float\n"
            "y          : usm_ndarray  output vector (in/out)\n"
            "num_rows, num_cols, nnz : int64\n"
            "depends    : list[sycl.Event]\n"
            "\nReturns\n-------\n"
            "(host_task_event, compute_event) : pair of sycl.Event",
            py::arg("sycl_queue"),
            py::arg("trans"),
            py::arg("alpha"),
            py::arg("row_ptr"),
            py::arg("col_ind"),
            py::arg("values"),
            py::arg("x"),
            py::arg("beta"),
            py::arg("y"),
            py::arg("num_rows"),
            py::arg("num_cols"),
            py::arg("nnz"),
            py::arg("depends") = py::list());
    }

    // ------------------------------------------------------------------
    // Runtime query: which sparse library backend is active
    // ------------------------------------------------------------------
    {
        m.def(
            "_using_onemath",
            []() {
#ifdef USE_ONEMATH
                return true;
#else
                return false;
#endif
            },
            "Return True if built against OneMath portable backend, "
            "False if built directly against oneMKL.");
    }
}
