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
//
// This file defines functions of dpnp.backend._window_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bartlett.hpp"
#include "blackman.hpp"
#include "common.hpp"
#include "hamming.hpp"
#include "hanning.hpp"

namespace window_ns = dpnp::extensions::window;
namespace py = pybind11;
using window_ns::window_fn_ptr_t;

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

static window_fn_ptr_t bartlett_dispatch_vector[dpctl_td_ns::num_types];
static window_fn_ptr_t blackman_dispatch_vector[dpctl_td_ns::num_types];
static window_fn_ptr_t hamming_dispatch_vector[dpctl_td_ns::num_types];
static window_fn_ptr_t hanning_dispatch_vector[dpctl_td_ns::num_types];

PYBIND11_MODULE(_window_impl, m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    {
        window_ns::init_window_dispatch_vectors<
            window_ns::kernels::BartlettFactory>(bartlett_dispatch_vector);

        auto bartlett_pyapi = [&](sycl::queue &exec_q, const arrayT &result,
                                  const event_vecT &depends = {}) {
            return window_ns::py_window(exec_q, result, depends,
                                        bartlett_dispatch_vector);
        };

        m.def("_bartlett", bartlett_pyapi, "Call Bartlett kernel",
              py::arg("sycl_queue"), py::arg("result"),
              py::arg("depends") = py::list());
    }

    {
        window_ns::init_window_dispatch_vectors<
            window_ns::kernels::BlackmanFactory>(blackman_dispatch_vector);

        auto blackman_pyapi = [&](sycl::queue &exec_q, const arrayT &result,
                                  const event_vecT &depends = {}) {
            return window_ns::py_window(exec_q, result, depends,
                                        blackman_dispatch_vector);
        };

        m.def("_blackman", blackman_pyapi, "Call Blackman kernel",
              py::arg("sycl_queue"), py::arg("result"),
              py::arg("depends") = py::list());
    }

    {
        window_ns::init_window_dispatch_vectors<
            window_ns::kernels::HammingFactory>(hamming_dispatch_vector);

        auto hamming_pyapi = [&](sycl::queue &exec_q, const arrayT &result,
                                 const event_vecT &depends = {}) {
            return window_ns::py_window(exec_q, result, depends,
                                        hamming_dispatch_vector);
        };

        m.def("_hamming", hamming_pyapi, "Call Hamming kernel",
              py::arg("sycl_queue"), py::arg("result"),
              py::arg("depends") = py::list());
    }

    {
        window_ns::init_window_dispatch_vectors<
            window_ns::kernels::HanningFactory>(hanning_dispatch_vector);

        auto hanning_pyapi = [&](sycl::queue &exec_q, const arrayT &result,
                                 const event_vecT &depends = {}) {
            return window_ns::py_window(exec_q, result, depends,
                                        hanning_dispatch_vector);
        };

        m.def("_hanning", hanning_pyapi, "Call Hanning kernel",
              py::arg("sycl_queue"), py::arg("result"),
              py::arg("depends") = py::list());
    }
}
