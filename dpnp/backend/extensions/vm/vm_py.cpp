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
// This file defines functions of dpnp.backend._lapack_impl extensions
//
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "cos.hpp"
#include "div.hpp"
#include "ln.hpp"
#include "sin.hpp"
#include "sqrt.hpp"
#include "types_matrix.hpp"

namespace py = pybind11;
namespace vm_ext = dpnp::backend::ext::vm;

using vm_ext::binary_impl_fn_ptr_t;
using vm_ext::unary_impl_fn_ptr_t;

static binary_impl_fn_ptr_t div_dispatch_vector[dpctl_td_ns::num_types];

static unary_impl_fn_ptr_t cos_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t ln_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t sin_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t sqrt_dispatch_vector[dpctl_td_ns::num_types];

PYBIND11_MODULE(_vm_impl, m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    // BinaryUfunc: ==== Div(x1, x2) ====
    {
        vm_ext::init_ufunc_dispatch_vector<binary_impl_fn_ptr_t,
                                           vm_ext::DivContigFactory>(
            div_dispatch_vector);

        auto div_pyapi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                             arrayT dst, const event_vecT &depends = {}) {
            return vm_ext::binary_ufunc(exec_q, src1, src2, dst, depends,
                                        div_dispatch_vector);
        };
        m.def("_div", div_pyapi,
              "Call `div` function from OneMKL VM library to performs element "
              "by element division of vector `src1` by vector `src2` "
              "to resulting vector `dst`",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("depends") = py::list());

        auto div_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src1,
                                          arrayT src2, arrayT dst) {
            return vm_ext::need_to_call_binary_ufunc(exec_q, src1, src2, dst,
                                                     div_dispatch_vector);
        };
        m.def("_mkl_div_to_call", div_need_to_call_pyapi,
              "Check input arguments to answer if `div` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"));
    }

    // UnaryUfunc: ==== Cos(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::CosContigFactory>(
            cos_dispatch_vector);

        auto cos_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                             const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       cos_dispatch_vector);
        };
        m.def("_cos", cos_pyapi,
              "Call `cos` function from OneMKL VM library to compute "
              "cosine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto cos_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                          arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    cos_dispatch_vector);
        };
        m.def("_mkl_cos_to_call", cos_need_to_call_pyapi,
              "Check input arguments to answer if `cos` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

    // UnaryUfunc: ==== Ln(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::LnContigFactory>(
            ln_dispatch_vector);

        auto ln_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                            const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       ln_dispatch_vector);
        };
        m.def("_ln", ln_pyapi,
              "Call `ln` function from OneMKL VM library to compute "
              "natural logarithm of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto ln_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                         arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    ln_dispatch_vector);
        };
        m.def("_mkl_ln_to_call", ln_need_to_call_pyapi,
              "Check input arguments to answer if `ln` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

    // UnaryUfunc: ==== Sin(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::SinContigFactory>(
            sin_dispatch_vector);

        auto sin_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                             const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       sin_dispatch_vector);
        };
        m.def("_sin", sin_pyapi,
              "Call `sin` function from OneMKL VM library to compute "
              "sine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto sin_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                          arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    sin_dispatch_vector);
        };
        m.def("_mkl_sin_to_call", sin_need_to_call_pyapi,
              "Check input arguments to answer if `sin` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

    // UnaryUfunc: ==== Sqrt(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::SqrtContigFactory>(
            sqrt_dispatch_vector);

        auto sqrt_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                              const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       sqrt_dispatch_vector);
        };
        m.def(
            "_sqrt", sqrt_pyapi,
            "Call `sqrt` from OneMKL VM library to performs element by element "
            "operation of extracting the square root "
            "of vector `src` to resulting vector `dst`",
            py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
            py::arg("depends") = py::list());

        auto sqrt_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                           arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    sqrt_dispatch_vector);
        };
        m.def("_mkl_sqrt_to_call", sqrt_need_to_call_pyapi,
              "Check input arguments to answer if `sqrt` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }
}
