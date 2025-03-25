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

#include "common.hpp"
#include "utils/type_dispatch.hpp"
#include <pybind11/pybind11.h>

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

namespace math
{
namespace common
{
pybind11::dtype dtype_from_typenum(int dst_typenum)
{
    dpctl_td_ns::typenum_t dst_typenum_t =
        static_cast<dpctl_td_ns::typenum_t>(dst_typenum);
    switch (dst_typenum_t) {
    case dpctl_td_ns::typenum_t::BOOL:
        return py::dtype("?");
    case dpctl_td_ns::typenum_t::INT8:
        return py::dtype("i1");
    case dpctl_td_ns::typenum_t::UINT8:
        return py::dtype("u1");
    case dpctl_td_ns::typenum_t::INT16:
        return py::dtype("i2");
    case dpctl_td_ns::typenum_t::UINT16:
        return py::dtype("u2");
    case dpctl_td_ns::typenum_t::INT32:
        return py::dtype("i4");
    case dpctl_td_ns::typenum_t::UINT32:
        return py::dtype("u4");
    case dpctl_td_ns::typenum_t::INT64:
        return py::dtype("i8");
    case dpctl_td_ns::typenum_t::UINT64:
        return py::dtype("u8");
    case dpctl_td_ns::typenum_t::HALF:
        return py::dtype("f2");
    case dpctl_td_ns::typenum_t::FLOAT:
        return py::dtype("f4");
    case dpctl_td_ns::typenum_t::DOUBLE:
        return py::dtype("f8");
    case dpctl_td_ns::typenum_t::CFLOAT:
        return py::dtype("c8");
    case dpctl_td_ns::typenum_t::CDOUBLE:
        return py::dtype("c16");
    default:
        throw py::value_error("Unrecognized dst_typeid");
    }
}

} // namespace common
} // namespace math
