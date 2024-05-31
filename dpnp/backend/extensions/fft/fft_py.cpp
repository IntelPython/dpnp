//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
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
// This file defines functions of dpnp.backend._fft_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "c2c.hpp"

namespace fft_ext = dpnp::extensions::fft;
namespace mkl_dft = oneapi::mkl::dft;
namespace py = pybind11;

template <mkl_dft::precision prec>
void register_complex_descriptor(py::module &m, const char *name)
{
    using DwT = fft_ext::ComplexDescriptorWrapper<prec>;
    py::class_<DwT>(m, name)
        .def(py::init<std::int64_t>())
        .def(py::init<std::vector<std::int64_t>>())
        .def_property("fwd_strides",
                      &DwT::template get_fwd_strides<std::vector<std::int64_t>>,
                      &DwT::template set_fwd_strides<std::vector<std::int64_t>>)
        .def_property("bwd_strides",
                      &DwT::template get_bwd_strides<std::vector<std::int64_t>>,
                      &DwT::template set_bwd_strides<std::vector<std::int64_t>>)
        .def_property("fwd_distance",
                      &DwT::template get_fwd_distance<std::int64_t>,
                      &DwT::template set_fwd_distance<std::int64_t>)
        .def_property("bwd_distance",
                      &DwT::template get_bwd_distance<std::int64_t>,
                      &DwT::template set_bwd_distance<std::int64_t>)
        .def_property("number_of_transforms",
                      &DwT::template get_number_of_transforms<std::int64_t>,
                      &DwT::template set_number_of_transforms<std::int64_t>)
        .def_property("transform_in_place", &DwT::get_in_place,
                      &DwT::set_in_place)
        .def("commit", &DwT::commit);
}

PYBIND11_MODULE(_fft_impl, m)
{
    constexpr mkl_dft::precision single_prec = mkl_dft::precision::SINGLE;
    register_complex_descriptor<single_prec>(m, "Complex64Descriptor");
    m.def("compute_fft", &fft_ext::compute_fft<single_prec>,
          "Compute forward/backward fft using OneMKL DFT library for complex "
          "float data types.",
          py::arg("descriptor"), py::arg("input"), py::arg("output"),
          py::arg("is_forward"), py::arg("depends") = py::list());

    constexpr mkl_dft::precision double_prec = mkl_dft::precision::DOUBLE;
    register_complex_descriptor<double_prec>(m, "Complex128Descriptor");
    m.def("compute_fft", &fft_ext::compute_fft<double_prec>,
          "Compute forward/backward fft using OneMKL DFT library for complex "
          "double data types.",
          py::arg("descriptor"), py::arg("input"), py::arg("output"),
          py::arg("is_forward"), py::arg("depends") = py::list());
}
