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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sycl/sycl.hpp>
#include <dpctl4pybind11.hpp>

#include "gemv.hpp"

namespace py = pybind11;

using dpnp::extensions::sparse::init_sparse_gemv_dispatch_table;
using dpnp::extensions::sparse::sparse_gemv_init;
using dpnp::extensions::sparse::sparse_gemv_compute;
using dpnp::extensions::sparse::sparse_gemv_release;

PYBIND11_MODULE(_sparse_impl, m)
{
    init_sparse_gemv_dispatch_table();

    // ------------------------------------------------------------------
    // _using_onemath()
    // Reports whether the module was compiled against the portable OneMath
    // interface (USE_ONEMATH) rather than direct oneMKL.
    // ------------------------------------------------------------------
    m.def("_using_onemath", []() -> bool {
#ifdef USE_ONEMATH
        return true;
#else
        return false;
#endif
    });

    // ------------------------------------------------------------------
    // _sparse_gemv_init(exec_q, trans, row_ptr, col_ind, values,
    //                   num_rows, num_cols, nnz, depends)
    //     -> (handle: int, event)
    //
    // Calls init_matrix_handle + set_csr_data + optimize_gemv ONCE.
    // The returned handle is an opaque uintptr_t; pass it back to
    // _sparse_gemv_compute and _sparse_gemv_release.
    // ------------------------------------------------------------------
    m.def(
        "_sparse_gemv_init",
        [](sycl::queue                                &exec_q,
           const int                                   trans,
           const dpctl::tensor::usm_ndarray           &row_ptr,
           const dpctl::tensor::usm_ndarray           &col_ind,
           const dpctl::tensor::usm_ndarray           &values,
           const std::int64_t                          num_rows,
           const std::int64_t                          num_cols,
           const std::int64_t                          nnz,
           const std::vector<sycl::event>             &depends)
            -> std::pair<std::uintptr_t, sycl::event>
        {
            return sparse_gemv_init(
                exec_q, trans,
                row_ptr, col_ind, values,
                num_rows, num_cols, nnz,
                depends);
        },
        py::arg("exec_q"),
        py::arg("trans"),
        py::arg("row_ptr"),
        py::arg("col_ind"),
        py::arg("values"),
        py::arg("num_rows"),
        py::arg("num_cols"),
        py::arg("nnz"),
        py::arg("depends"),
        "Initialise oneMKL sparse matrix handle (set_csr_data + optimize_gemv). "
        "Returns (handle_ptr: int, event). Call once per operator."
    );

    // ------------------------------------------------------------------
    // _sparse_gemv_compute(exec_q, handle, trans, alpha, x, beta, y,
    //                      num_rows, num_cols, depends)
    //     -> (args_event, gemv_event)
    //
    // Fires sparse::gemv using the pre-built handle.
    // Only the cheap kernel is dispatched; no analysis overhead.
    // ------------------------------------------------------------------
    m.def(
        "_sparse_gemv_compute",
        [](sycl::queue                                &exec_q,
           const std::uintptr_t                        handle_ptr,
           const int                                   trans,
           const double                                alpha,
           const dpctl::tensor::usm_ndarray           &x,
           const double                                beta,
           const dpctl::tensor::usm_ndarray           &y,
           const std::int64_t                          num_rows,
           const std::int64_t                          num_cols,
           const std::vector<sycl::event>             &depends)
            -> std::pair<sycl::event, sycl::event>
        {
            return sparse_gemv_compute(
                exec_q, handle_ptr, trans, alpha,
                x, beta, y,
                num_rows, num_cols,
                depends);
        },
        py::arg("exec_q"),
        py::arg("handle"),
        py::arg("trans"),
        py::arg("alpha"),
        py::arg("x"),
        py::arg("beta"),
        py::arg("y"),
        py::arg("num_rows"),
        py::arg("num_cols"),
        py::arg("depends"),
        "Execute sparse::gemv using a pre-built handle. "
        "Returns (args_event, gemv_event)."
    );

    // ------------------------------------------------------------------
    // _sparse_gemv_release(exec_q, handle, depends) -> event
    //
    // Releases the matrix_handle allocated by _sparse_gemv_init.
    // Must be called exactly once per handle after all compute calls
    // referencing it are complete.
    // ------------------------------------------------------------------
    m.def(
        "_sparse_gemv_release",
        [](sycl::queue                                &exec_q,
           const std::uintptr_t                        handle_ptr,
           const std::vector<sycl::event>             &depends)
            -> sycl::event
        {
            return sparse_gemv_release(exec_q, handle_ptr, depends);
        },
        py::arg("exec_q"),
        py::arg("handle"),
        py::arg("depends"),
        "Release the oneMKL matrix_handle created by _sparse_gemv_init."
    );
}
