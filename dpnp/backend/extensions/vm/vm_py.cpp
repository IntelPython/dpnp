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

#include "div.hpp"
#include "sqrt.hpp"

namespace vm_ext = dpnp::backend::ext::vm;
namespace py = pybind11;

// populate dispatch vectors
void init_dispatch_vectors(void)
{
    vm_ext::init_div_dispatch_vector();
    vm_ext::init_sqrt_dispatch_vector();
}

// populate dispatch tables
void init_dispatch_tables(void) {}

PYBIND11_MODULE(_vm_impl, m)
{
    init_dispatch_vectors();
    init_dispatch_tables();

    m.def("_div", &vm_ext::div,
          "Call `div` from OneMKL VM library to performs element by element "
          "division "
          "of vector `src1` by vector `src2` to resulting vector `dst`",
          py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
          py::arg("dst"), py::arg("depends") = py::list());

    m.def("_can_call_div", &vm_ext::can_call_div,
          "Check input arrays to answer if `div` function from OneMKL VM "
          "library can be used",
          py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
          py::arg("dst"));

    m.def("_sqrt", &vm_ext::sqrt,
          "Call `sqrt` from OneMKL VM library to performs element by element "
          "operation of extracting the square root "
          "of vector `src` to resulting vector `dst`",
          py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
          py::arg("depends") = py::list());

    m.def("_can_call_sqrt", &vm_ext::can_call_sqrt,
          "Check input arrays to answer if `sqrt` function from OneMKL VM "
          "library can be used",
          py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
}
