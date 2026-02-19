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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include <string>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sycl/sycl.hpp>

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;

namespace
{

std::string _default_device_fp_type(const sycl::device &d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "f8";
    }
    else {
        return "f4";
    }
}

int get_numpy_major_version()
{

    py::module_ numpy = py::module_::import("numpy");
    py::str version_string = numpy.attr("__version__");
    py::module_ numpy_lib = py::module_::import("numpy.lib");

    py::object numpy_version = numpy_lib.attr("NumpyVersion")(version_string);
    int major_version = numpy_version.attr("major").cast<int>();

    return major_version;
}

std::string _default_device_int_type(const sycl::device &)
{
    const int np_ver = get_numpy_major_version();

    if (np_ver >= 2) {
        return "i8";
    }
    else {
        // code for numpy.dtype('long') to be consistent
        // with NumPy's default integer type across
        // platforms.
        return "l";
    }
}

std::string _default_device_uint_type(const sycl::device &)
{
    const int np_ver = get_numpy_major_version();

    if (np_ver >= 2) {
        return "u8";
    }
    else {
        // code for numpy.dtype('long') to be consistent
        // with NumPy's default integer type across
        // platforms.
        return "L";
    }
}

std::string _default_device_complex_type(const sycl::device &d)
{
    if (d.has(sycl::aspect::fp64)) {
        return "c16";
    }
    else {
        return "c8";
    }
}

std::string _default_device_bool_type(const sycl::device &)
{
    return "b1";
}

std::string _default_device_index_type(const sycl::device &)
{
    return "i8";
}

sycl::device _extract_device(const py::object &arg)
{
    auto const &api = dpctl::detail::dpctl_capi::get();

    PyObject *source = arg.ptr();
    if (api.PySyclQueue_Check_(source)) {
        const sycl::queue &q = py::cast<sycl::queue>(arg);
        return q.get_device();
    }
    else if (api.PySyclDevice_Check_(source)) {
        return py::cast<sycl::device>(arg);
    }
    else {
        throw py::type_error(
            "Expected type `dpctl.SyclQueue` or `dpctl.SyclDevice`.");
    }
}

} // namespace

std::string default_device_fp_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_fp_type(d);
}

std::string default_device_int_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_int_type(d);
}

std::string default_device_uint_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_uint_type(d);
}

std::string default_device_bool_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_bool_type(d);
}

std::string default_device_complex_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_complex_type(d);
}

std::string default_device_index_type(const py::object &arg)
{
    const sycl::device &d = _extract_device(arg);
    return _default_device_index_type(d);
}

} // namespace dpctl::tensor::py_internal
