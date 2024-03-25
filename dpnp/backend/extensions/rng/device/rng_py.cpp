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

#include "engine/mcg31m1_engine.hpp"
#include "engine/mcg59_engine.hpp"
#include "engine/mrg32k3a_engine.hpp"
#include "engine/philox4x32x10_engine.hpp"

namespace mkl_rng = oneapi::mkl::rng;
namespace rng_dev_ext = dpnp::backend::ext::rng::device;
namespace rng_dev_engine = dpnp::backend::ext::rng::device::engine;
namespace py = pybind11;

// populate dispatch 3-D tables
void init_dispatch_3d_tables(void)
{
    rng_dev_ext::init_gaussian_dispatch_3d_table();
}

class PyEngineBase : public rng_dev_engine::EngineBase
{
public:
    // inherit the constructor
    using EngineBase::EngineBase;

    // trampoline (need one for each virtual function)
    // sycl::queue &get_queue() {
    //     PYBIND11_OVERRIDE_PURE(
    //         sycl::queue&, /* Return type */
    //         EngineBase,   /* Parent class */
    //         get_queue,    /* Name of function in C++ (must match Python name)
    //         */
    //     );
    // }
};

PYBIND11_MODULE(_rng_dev_impl, m)
{
    init_dispatch_3d_tables();

    py::class_<rng_dev_engine::EngineBase, PyEngineBase /* <--- trampoline */>(
        m, "EngineBase")
        .def(py::init<>())
        .def("get_queue", &rng_dev_engine::EngineBase::get_queue);

    py::class_<rng_dev_engine::MRG32k3a, rng_dev_engine::EngineBase>(m,
                                                                     "MRG32k3a")
        .def(py::init<sycl::queue &, std::uint32_t, std::uint64_t>(),
             py::arg("sycl_queue"), py::arg("seed"), py::arg("offset") = 0)
        .def(py::init<sycl::queue &, std::vector<std::uint32_t> &,
                      std::uint64_t>(),
             py::arg("sycl_queue"), py::arg("seed"), py::arg("offset") = 0)
        .def(py::init<sycl::queue &, std::uint32_t,
                      std::vector<std::uint64_t> &>(),
             py::arg("sycl_queue"), py::arg("seed"),
             py::arg("offset") = py::list())
        .def(py::init<sycl::queue &, std::vector<std::uint32_t> &,
                      std::vector<std::uint64_t> &>(),
             py::arg("sycl_queue"), py::arg("seed"),
             py::arg("offset") = py::list());

    py::class_<rng_dev_engine::PHILOX4x32x10, rng_dev_engine::EngineBase>(
        m, "PHILOX4x32x10")
        .def(py::init<sycl::queue &, std::uint64_t, std::uint64_t>(),
             py::arg("sycl_queue"), py::arg("seed"), py::arg("offset") = 0)
        .def(py::init<sycl::queue &, std::vector<std::uint64_t> &,
                      std::uint64_t>(),
             py::arg("sycl_queue"), py::arg("seed"), py::arg("offset") = 0)
        .def(py::init<sycl::queue &, std::uint64_t,
                      std::vector<std::uint64_t> &>(),
             py::arg("sycl_queue"), py::arg("seed"),
             py::arg("offset") = py::list())
        .def(py::init<sycl::queue &, std::vector<std::uint64_t> &,
                      std::vector<std::uint64_t> &>(),
             py::arg("sycl_queue"), py::arg("seed"),
             py::arg("offset") = py::list());

    py::class_<rng_dev_engine::MCG31M1, rng_dev_engine::EngineBase>(m,
                                                                    "MCG31M1")
        .def(py::init<sycl::queue &, std::uint32_t, std::uint64_t>(),
             py::arg("sycl_queue"), py::arg("seed"), py::arg("offset") = 0)
        .def(py::init<sycl::queue &, std::vector<std::uint32_t> &,
                      std::uint64_t>(),
             py::arg("sycl_queue"), py::arg("seed"), py::arg("offset") = 0);

    py::class_<rng_dev_engine::MCG59, rng_dev_engine::EngineBase>(m, "MCG59")
        .def(py::init<sycl::queue &, std::uint32_t, std::uint64_t>(),
             py::arg("sycl_queue"), py::arg("seed"), py::arg("offset") = 0)
        .def(py::init<sycl::queue &, std::vector<std::uint32_t> &,
                      std::uint64_t>(),
             py::arg("sycl_queue"), py::arg("seed"), py::arg("offset") = 0);

    m.def("_gaussian", &rng_dev_ext::gaussian, "", py::arg("engine"),
          py::arg("method_id"), py::arg("vec_size"), py::arg("mean"),
          py::arg("stddev"), py::arg("n"), py::arg("res"),
          py::arg("depends") = py::list());
}
