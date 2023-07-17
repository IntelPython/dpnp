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
// This file defines functions of dpnp.backend._sycl_ext_impl extensions
//
//*****************************************************************************

#include <memory>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dpctl4pybind11.hpp"
#include "sum_mean.hpp"

namespace sycl_ext = dpnp::backend::ext::sycl_ext;
namespace py = pybind11;

using dpctl::tensor::usm_ndarray;

std::unique_ptr<sycl_ext::SumOverAxisContigDispatcher> sum_dispatcher;
std::unique_ptr<sycl_ext::MeanOverAxisContigDispatcher> mean_dispatcher;

void init_dispatchers(void)
{
    sum_dispatcher.reset(new sycl_ext::SumOverAxisContigDispatcher());
    mean_dispatcher.reset(new sycl_ext::MeanOverAxisContigDispatcher());
}

using SumMeanFnT = sycl::event (*)(const usm_ndarray &,
                                   usm_ndarray &,
                                   const std::vector<sycl::event> &);

sycl::event sum_over_axis(usm_ndarray input,
                          usm_ndarray output,
                          std::vector<sycl::event> depends)
{
    auto sum_fn = (*sum_dispatcher)({input, output});

    if (sum_fn == nullptr)
        throw py::value_error("No suitable implementation found");

    return sum_fn(input, output, depends);
}

sycl::event mean_over_axis(usm_ndarray input,
                           usm_ndarray output,
                           std::vector<sycl::event> depends)
{
    auto mean_fn = (*mean_dispatcher)({input, output});

    if (mean_fn == nullptr)
        throw py::value_error("No suitable implementation found");

    return mean_fn(input, output, depends);
}

py::cpp_function get_sum_over_axis(usm_ndarray input, usm_ndarray output)
{
    if (not sycl_ext::check_limitations(input, output))
        return nullptr;

    return (*sum_dispatcher)({input, output});
}

py::cpp_function get_mean_over_axis(usm_ndarray input, usm_ndarray output)
{
    if (not sycl_ext::check_limitations(input, output))
        return nullptr;

    return (*mean_dispatcher)({input, output});
}

PYBIND11_MODULE(_sycl_ext_impl, m)
{
    import_dpctl();
    init_dispatchers();

    m.def("_sum_over_axis_0", &sum_over_axis, "Sum over axis 0",
          py::arg("input"), py::arg("output"), py::arg("depends") = py::list());

    m.def("_mean_over_axis_0", &mean_over_axis, "Mean over axis 0",
          py::arg("input"), py::arg("output"), py::arg("depends") = py::list());

    m.def("_get_sum_over_axis_0", &get_sum_over_axis, "Sum over axis 0",
          py::arg("input"), py::arg("output"));

    m.def("_get_mean_over_axis_0", &get_mean_over_axis, "Mean over axis 0",
          py::arg("input"), py::arg("output"));
}
