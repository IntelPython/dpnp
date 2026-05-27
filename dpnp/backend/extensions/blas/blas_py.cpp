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
// This file defines functions of dpnp.backend._blas_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// utils extension header
#include "ext/common.hpp"

#include "dot.hpp"
#include "dot_common.hpp"
#include "dotc.hpp"
#include "dotu.hpp"
#include "gemm.hpp"
#include "gemv.hpp"
#include "syrk.hpp"

namespace blas_ns = dpnp::extensions::blas;
namespace py = pybind11;
namespace dot_ns = blas_ns::dot;

using dot_ns::dot_impl_fn_ptr_t;
using ext::common::init_dispatch_vector;

// populate dispatch vectors and tables
void init_dispatch_vectors_tables(void)
{
    blas_ns::init_gemm_batch_dispatch_table();
    blas_ns::init_gemm_dispatch_table();
    blas_ns::init_gemv_dispatch_vector();
    blas_ns::init_syrk_dispatch_vector();
}

static dot_impl_fn_ptr_t dot_dispatch_vector[dpnp_td_ns::num_types];
static dot_impl_fn_ptr_t dotc_dispatch_vector[dpnp_td_ns::num_types];
static dot_impl_fn_ptr_t dotu_dispatch_vector[dpnp_td_ns::num_types];

PYBIND11_MODULE(_blas_impl, m)
{
    init_dispatch_vectors_tables();

    using arrayT = dpnp::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    {
        init_dispatch_vector<dot_impl_fn_ptr_t, blas_ns::DotContigFactory>(
            dot_dispatch_vector);

        auto dot_pyapi = [&](sycl::queue &exec_q, const arrayT &src1,
                             const arrayT &src2, const arrayT &dst,
                             const event_vecT &depends = {}) {
            return dot_ns::dot_func(exec_q, src1, src2, dst, depends,
                                    dot_dispatch_vector);
        };

        m.def("_dot", dot_pyapi,
              "Call `dot` from oneMKL BLAS library to compute "
              "the dot product of two real-valued vectors.",
              py::arg("sycl_queue"), py::arg("vectorA"), py::arg("vectorB"),
              py::arg("result"), py::arg("depends") = py::list());
    }

    {
        init_dispatch_vector<dot_impl_fn_ptr_t, blas_ns::DotcContigFactory>(
            dotc_dispatch_vector);

        auto dotc_pyapi = [&](sycl::queue &exec_q, const arrayT &src1,
                              const arrayT &src2, const arrayT &dst,
                              const event_vecT &depends = {}) {
            return dot_ns::dot_func(exec_q, src1, src2, dst, depends,
                                    dotc_dispatch_vector);
        };

        m.def("_dotc", dotc_pyapi,
              "Call `dotc` from oneMKL BLAS library to compute "
              "the dot product of two complex vectors, "
              "conjugating the first vector.",
              py::arg("sycl_queue"), py::arg("vectorA"), py::arg("vectorB"),
              py::arg("result"), py::arg("depends") = py::list());
    }

    {
        init_dispatch_vector<dot_impl_fn_ptr_t, blas_ns::DotuContigFactory>(
            dotu_dispatch_vector);

        auto dotu_pyapi = [&](sycl::queue &exec_q, const arrayT &src1,
                              const arrayT &src2, const arrayT &dst,
                              const event_vecT &depends = {}) {
            return dot_ns::dot_func(exec_q, src1, src2, dst, depends,
                                    dotu_dispatch_vector);
        };

        m.def("_dotu", dotu_pyapi,
              "Call `dotu` from oneMKL BLAS library to compute "
              "the dot product of two complex vectors.",
              py::arg("sycl_queue"), py::arg("vectorA"), py::arg("vectorB"),
              py::arg("result"), py::arg("depends") = py::list());
    }

    {
        m.def("_gemm", &blas_ns::gemm,
              "Call `gemm` from oneMKL BLAS library to compute "
              "the matrix-matrix product with 2-D matrices.",
              py::arg("sycl_queue"), py::arg("matrixA"), py::arg("matrixB"),
              py::arg("resultC"), py::arg("depends") = py::list());
    }

    {
        m.def("_gemm_batch", &blas_ns::gemm_batch,
              "Call `gemm_batch` from oneMKL BLAS library to compute "
              "the matrix-matrix product for a batch of 2-D matrices.",
              py::arg("sycl_queue"), py::arg("matrixA"), py::arg("matrixB"),
              py::arg("resultC"), py::arg("depends") = py::list());
    }

    {
        m.def("_gemv", &blas_ns::gemv,
              "Call `gemv` from oneMKL BLAS library to compute "
              "the matrix-vector product with a general matrix.",
              py::arg("sycl_queue"), py::arg("matrixA"), py::arg("vectorX"),
              py::arg("vectorY"), py::arg("transpose"),
              py::arg("depends") = py::list());
    }

    {
        // y = alpha * op(A) * x + beta * y, the full BLAS gemv form.
        // Used by dpnp.scipy.sparse.linalg.gmres to fuse the Arnoldi
        // orthogonalisation (u -= V @ h) into a single oneMKL call
        // (alpha=-1, beta=1) and to write the Hessenberg column
        // h = V^H @ u directly into the Hessenberg matrix slice
        // (alpha=1, beta=0). For complex matrices the scalars must be
        // exactly representable as their real form: callers pass
        // {-1, 0, 1}; fractional or complex scalars would lose the
        // imaginary component on the C++ cast.
        m.def("_gemv_alpha_beta", &blas_ns::gemv_alpha_beta,
              "Call `gemv` from oneMKL BLAS library with explicit "
              "alpha and beta scalars: y = alpha * op(A) * x + beta * y.",
              py::arg("sycl_queue"), py::arg("matrixA"), py::arg("vectorX"),
              py::arg("vectorY"), py::arg("transpose"), py::arg("alpha"),
              py::arg("beta"), py::arg("depends") = py::list());
    }

    {
        m.def("_syrk", &blas_ns::syrk,
              "Call `syrk` from oneMKL BLAS library to compute "
              "the matrix-vector product with a general matrix.",
              py::arg("sycl_queue"), py::arg("matrixA"), py::arg("resultC"),
              py::arg("depends") = py::list());
    }

    {
        m.def(
            "_using_onemath",
            []() {
#ifdef USE_ONEMATH
                return true;
#else
                return false;
#endif
            },
            "Check if OneMath is being used.");
    }
}
