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
// This file defines functions of dpnp.backend._rng_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dpctl4pybind11.hpp>

#include <oneapi/mkl/rng.hpp>

#include "gaussian.hpp"

namespace mkl_rng = oneapi::mkl::rng;
namespace rng_dev_ext = dpnp::backend::ext::rng::device;
namespace py = pybind11;

// populate dispatch vectors
void init_dispatch_vectors(void)
{
    rng_dev_ext::init_gaussian_dispatch_vector();
}

// populate dispatch tables
void init_dispatch_tables(void)
{
    // lapack_ext::init_heevd_dispatch_table();
}


PYBIND11_MODULE(_rng_dev_impl, m)
{
    // using engine_base_t = rng_ext::EngineBase;
    // py::class_<engine_base_t> engine_base(m, "EngineBase");
    // engine_base.def(py::init<sycl::queue>())
    //            .def("get_queue", &engine_base_t::get_queue);

    // using mt19937_engine_t = rng_ext::EngineProxy<mkl_rng::mt19937, std::uint32_t>;
    // py::class_<mt19937_engine_t>(m, "mt19937", engine_base)
    //     .def(py::init<sycl::queue, std::uint32_t>())
    //     .def(py::init<sycl::queue, std::vector<std::uint32_t>>());

    // using mcg59_engine_t = rng_ext::EngineProxy<mkl_rng::mcg59, std::uint64_t>;
    // py::class_<mcg59_engine_t>(m, "mcg59", engine_base)
    //     .def(py::init<sycl::queue, std::uint64_t>());

    init_dispatch_vectors();
    init_dispatch_tables();

    // m.def("_heevd", &lapack_ext::heevd,
    //       "Call `heevd` from OneMKL LAPACK library to return "
    //       "the eigenvalues and eigenvectors of a complex Hermitian matrix",
    //       py::arg("sycl_queue"), py::arg("jobz"), py::arg("upper_lower"),
    //       py::arg("eig_vecs"), py::arg("eig_vals"),
    //       py::arg("depends") = py::list());

    m.def("_gaussian", &rng_dev_ext::gaussian,
          "",
          py::arg("sycl_queue"), py::arg("seed"), py::arg("mean"), py::arg("stddev"),
          py::arg("n"), py::arg("res"),
          py::arg("depends") = py::list());
}
