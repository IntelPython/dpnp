//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines basic pybind11 extension example using
/// dpnp::tensor::usm_ndarray.
//===----------------------------------------------------------------------===//

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

int get_ndim(const dpnp::tensor::usm_ndarray &arr) { return arr.get_ndim(); }

std::vector<py::ssize_t> get_shape(const dpnp::tensor::usm_ndarray &arr)
{
    return arr.get_shape_vector();
}

py::ssize_t get_size(const dpnp::tensor::usm_ndarray &arr)
{
    return arr.get_size();
}

int get_typenum(const dpnp::tensor::usm_ndarray &arr)
{
    return arr.get_typenum();
}

int get_elemsize(const dpnp::tensor::usm_ndarray &arr)
{
    return arr.get_elemsize();
}

bool is_c_contiguous(const dpnp::tensor::usm_ndarray &arr)
{
    return arr.is_c_contiguous();
}

bool is_f_contiguous(const dpnp::tensor::usm_ndarray &arr)
{
    return arr.is_f_contiguous();
}

bool is_writable(const dpnp::tensor::usm_ndarray &arr)
{
    return arr.is_writable();
}

PYBIND11_MODULE(_use_dpnp_array, m)
{
    m.def("get_ndim", &get_ndim);
    m.def("get_shape", &get_shape);
    m.def("get_size", &get_size);
    m.def("get_typenum", &get_typenum);
    m.def("get_elemsize", &get_elemsize);
    m.def("is_c_contiguous", &is_c_contiguous);
    m.def("is_f_contiguous", &is_f_contiguous);
    m.def("is_writable", &is_writable);
}
