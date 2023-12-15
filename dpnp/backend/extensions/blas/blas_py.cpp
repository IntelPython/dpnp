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
//
// This file defines functions of dpnp.backend._lapack_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gemm.hpp"

namespace blas_ext = dpnp::backend::ext::blas;
namespace py = pybind11;

// populate dispatch tables
void init_dispatch_tables(void)
{
    blas_ext::init_gemm_batch_dispatch_table();
    blas_ext::init_gemm_dispatch_table();
}

PYBIND11_MODULE(_blas_impl, m)
{
    init_dispatch_tables();

    {
        m.def("_gemm", &blas_ext::gemm,
              "Call `gemm` from OneMKL LAPACK library to return "
              "the matrix-matrix product with 2-D matrices.",
              py::arg("sycl_queue"), py::arg("matrixA"), py::arg("matrixB"),
              py::arg("result"), py::arg("depends") = py::list());
    }

    {
        m.def("_gemm_batch", &blas_ext::gemm_batch,
              "Call `gemm_batch` from OneMKL LAPACK library to return "
              "the matrix-matrix product for a batch of 2-D matrices.",
              py::arg("sycl_queue"), py::arg("matrixA"), py::arg("matrixB"),
              py::arg("result"), py::arg("m"), py::arg("n"), py::arg("k"),
              py::arg("batch_size"), py::arg("ld_array_1"),
              py::arg("ld_array_2"), py::arg("ld_result"), py::arg("stridea"),
              py::arg("strideb"), py::arg("stridec"), py::arg("transA_int"),
              py::arg("transB_int"), py::arg("depends") = py::list());
    }
}
