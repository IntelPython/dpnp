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
// This file defines functions of dpnp.backend._lapack_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dot.hpp"
#include "dot_common.hpp"
#include "dotc.hpp"
#include "dotu.hpp"
#include "gemm.hpp"

namespace blas_ext = dpnp::backend::ext::blas;
namespace py = pybind11;
namespace dot_ext = blas_ext::dot;
using dot_ext::dot_impl_fn_ptr_t;

// populate dispatch tables
void init_dispatch_tables(void)
{
    blas_ext::init_gemm_batch_dispatch_table();
    blas_ext::init_gemm_dispatch_table();
}

static dot_impl_fn_ptr_t dot_dispatch_vector[dpctl_td_ns::num_types];
static dot_impl_fn_ptr_t dotc_dispatch_vector[dpctl_td_ns::num_types];
static dot_impl_fn_ptr_t dotu_dispatch_vector[dpctl_td_ns::num_types];

PYBIND11_MODULE(_blas_impl, m)
{
    init_dispatch_tables();

    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    {
        dot_ext::init_dot_dispatch_vector<dot_impl_fn_ptr_t,
                                          blas_ext::DotContigFactory>(
            dot_dispatch_vector);

        auto dot_pypi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                            arrayT dst, const event_vecT &depends = {}) {
            return dot_ext::dot_func(exec_q, src1, src2, dst, depends,
                                     dot_dispatch_vector);
        };

        m.def("_dot", dot_pypi,
              "Call `dot` from OneMKL BLAS library to return "
              "the dot product of two real-valued vectors.",
              py::arg("sycl_queue"), py::arg("vectorA"), py::arg("vectorB"),
              py::arg("result"), py::arg("depends") = py::list());
    }

    {
        dot_ext::init_dot_dispatch_vector<dot_impl_fn_ptr_t,
                                          blas_ext::DotcContigFactory>(
            dotc_dispatch_vector);

        auto dotc_pypi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                             arrayT dst, const event_vecT &depends = {}) {
            return dot_ext::dot_func(exec_q, src1, src2, dst, depends,
                                     dotc_dispatch_vector);
        };

        m.def("_dotc", dotc_pypi,
              "Call `dotc` from OneMKL BLAS library to return "
              "the dot product of two complex vectors, "
              "conjugating the first vector.",
              py::arg("sycl_queue"), py::arg("vectorA"), py::arg("vectorB"),
              py::arg("result"), py::arg("depends") = py::list());
    }

    {
        dot_ext::init_dot_dispatch_vector<dot_impl_fn_ptr_t,
                                          blas_ext::DotuContigFactory>(
            dotu_dispatch_vector);

        auto dotu_pypi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                             arrayT dst, const event_vecT &depends = {}) {
            return dot_ext::dot_func(exec_q, src1, src2, dst, depends,
                                     dotu_dispatch_vector);
        };

        m.def("_dotu", dotu_pypi,
              "Call `dotu` from OneMKL BLAS library to return "
              "the dot product of two complex vectors.",
              py::arg("sycl_queue"), py::arg("vectorA"), py::arg("vectorB"),
              py::arg("result"), py::arg("depends") = py::list());
    }

    {
        m.def("_gemm", &blas_ext::gemm,
              "Call `gemm` from OneMKL BLAS library to return "
              "the matrix-matrix product with 2-D matrices.",
              py::arg("sycl_queue"), py::arg("matrixA"), py::arg("matrixB"),
              py::arg("result"), py::arg("depends") = py::list());
    }

    {
        m.def("_gemm_batch", &blas_ext::gemm_batch,
              "Call `gemm_batch` from OneMKL BLAS library to return "
              "the matrix-matrix product for a batch of 2-D matrices.",
              py::arg("sycl_queue"), py::arg("matrixA"), py::arg("matrixB"),
              py::arg("result"), py::arg("batch_size"), py::arg("stridea"),
              py::arg("strideb"), py::arg("stridec"),
              py::arg("depends") = py::list());
    }
}
