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
// This file defines functions of dpnp.backend._fft_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "in_place.hpp"
#include "out_of_place.hpp"

namespace fft_ns = dpnp::extensions::fft;
namespace mkl_dft = oneapi::mkl::dft;
namespace py = pybind11;

template <mkl_dft::precision prec, mkl_dft::domain dom>
void register_descriptor(py::module &m, const char *name)
{
    using DwT = fft_ns::DescriptorWrapper<prec, dom>;
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
    constexpr mkl_dft::domain complex_dom = mkl_dft::domain::COMPLEX;
    constexpr mkl_dft::domain real_dom = mkl_dft::domain::REAL;

    constexpr mkl_dft::precision single_prec = mkl_dft::precision::SINGLE;
    constexpr mkl_dft::precision double_prec = mkl_dft::precision::DOUBLE;

    register_descriptor<single_prec, complex_dom>(m, "Complex64Descriptor");
    register_descriptor<double_prec, complex_dom>(m, "Complex128Descriptor");
    register_descriptor<single_prec, real_dom>(m, "Real32Descriptor");
    register_descriptor<double_prec, real_dom>(m, "Real64Descriptor");

    // out-of-place FFT, all possible combination (single/double precisions and
    // real/complex domains) are supported with overloading of
    // "_fft_out_of_place" function on python side
    m.def("_fft_out_of_place", // single precision c2c out-of-place FFT
          &fft_ns::compute_fft_out_of_place<single_prec, complex_dom>,
          "Compute out-of-place complex-to-complex fft using OneMKL DFT "
          "library for complex64 data types.",
          py::arg("descriptor"), py::arg("input"), py::arg("output"),
          py::arg("is_forward"), py::arg("depends") = py::list());

    m.def("_fft_out_of_place", // double precision c2c out-of-place FFT
          &fft_ns::compute_fft_out_of_place<double_prec, complex_dom>,
          "Compute out-of-place complex-to-complex fft using OneMKL DFT "
          "library for complex128 data types.",
          py::arg("descriptor"), py::arg("input"), py::arg("output"),
          py::arg("is_forward"), py::arg("depends") = py::list());

    m.def("_fft_out_of_place", // single precision r2c/c2r out-of-place FFT
          &fft_ns::compute_fft_out_of_place<single_prec, real_dom>,
          "Compute out-of-place real-to-complex fft using OneMKL DFT library "
          "for float32 data types.",
          py::arg("descriptor"), py::arg("input"), py::arg("output"),
          py::arg("is_forward"), py::arg("depends") = py::list());

    m.def("_fft_out_of_place", // double precision r2c/c2r out-of-place FFT
          &fft_ns::compute_fft_out_of_place<double_prec, real_dom>,
          "Compute out-of-place real-to-complex fft using OneMKL DFT library "
          "for float64 data types.",
          py::arg("descriptor"), py::arg("input"), py::arg("output"),
          py::arg("is_forward"), py::arg("depends") = py::list());

    // in-place c2c FFT, both single and double precisions are supported with
    // overloading of "_fft_in_place" function on python side
    m.def("_fft_in_place", // single precision c2c in-place FFT
          &fft_ns::compute_fft_in_place<single_prec, complex_dom>,
          "Compute in-place complex-to-complex fft using OneMKL DFT library "
          "for complex64 data types.",
          py::arg("descriptor"), py::arg("input-output"), py::arg("is_forward"),
          py::arg("depends") = py::list());

    m.def("_fft_in_place", // double precision c2c in-place FFT
          &fft_ns::compute_fft_in_place<double_prec, complex_dom>,
          "Compute in-place complex-to-complex fft using OneMKL DFT library "
          "for complex128 data types.",
          py::arg("descriptor"), py::arg("input-output"), py::arg("is_forward"),
          py::arg("depends") = py::list());
}
