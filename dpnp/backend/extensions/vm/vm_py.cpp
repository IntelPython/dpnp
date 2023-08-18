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

#include "add.hpp"
#include "ceil.hpp"
#include "common.hpp"
#include "cos.hpp"
#include "div.hpp"
#include "floor.hpp"
#include "ln.hpp"
#include "mul.hpp"
#include "sin.hpp"
#include "sqr.hpp"
#include "sqrt.hpp"
#include "sub.hpp"
#include "trunc.hpp"
#include "types_matrix.hpp"

namespace py = pybind11;
namespace vm_ext = dpnp::backend::ext::vm;

using vm_ext::binary_impl_fn_ptr_t;
using vm_ext::unary_impl_fn_ptr_t;

static binary_impl_fn_ptr_t add_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t ceil_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t cos_dispatch_vector[dpctl_td_ns::num_types];
static binary_impl_fn_ptr_t div_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t floor_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t ln_dispatch_vector[dpctl_td_ns::num_types];
static binary_impl_fn_ptr_t mul_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t sin_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t sqr_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t sqrt_dispatch_vector[dpctl_td_ns::num_types];
static binary_impl_fn_ptr_t sub_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t trunc_dispatch_vector[dpctl_td_ns::num_types];

PYBIND11_MODULE(_vm_impl, m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    // BinaryUfunc: ==== Add(x1, x2) ====
    {
        vm_ext::init_ufunc_dispatch_vector<binary_impl_fn_ptr_t,
                                           vm_ext::AddContigFactory>(
            add_dispatch_vector);

        auto add_pyapi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                             arrayT dst, const event_vecT &depends = {}) {
            return vm_ext::binary_ufunc(exec_q, src1, src2, dst, depends,
                                        add_dispatch_vector);
        };
        m.def("_add", add_pyapi,
              "Call `add` function from OneMKL VM library to performs element "
              "by element addition of vector `src1` by vector `src2` "
              "to resulting vector `dst`",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("depends") = py::list());

        auto add_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src1,
                                          arrayT src2, arrayT dst) {
            return vm_ext::need_to_call_binary_ufunc(exec_q, src1, src2, dst,
                                                     add_dispatch_vector);
        };
        m.def("_mkl_add_to_call", add_need_to_call_pyapi,
              "Check input arguments to answer if `add` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"));
    }

    // UnaryUfunc: ==== Ceil(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::CeilContigFactory>(
            ceil_dispatch_vector);

        auto ceil_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                              const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       ceil_dispatch_vector);
        };
        m.def("_ceil", ceil_pyapi,
              "Call `ceil` function from OneMKL VM library to compute "
              "ceiling of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto ceil_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                           arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    ceil_dispatch_vector);
        };
        m.def("_mkl_ceil_to_call", ceil_need_to_call_pyapi,
              "Check input arguments to answer if `ceil` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
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

    // UnaryUfunc: ==== Floor(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::FloorContigFactory>(
            floor_dispatch_vector);

        auto floor_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                               const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       floor_dispatch_vector);
        };
        m.def("_floor", floor_pyapi,
              "Call `floor` function from OneMKL VM library to compute "
              "floor of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto floor_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                            arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    floor_dispatch_vector);
        };
        m.def("_mkl_floor_to_call", floor_need_to_call_pyapi,
              "Check input arguments to answer if `floor` function from "
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

    // BinaryUfunc: ==== Mul(x1, x2) ====
    {
        vm_ext::init_ufunc_dispatch_vector<binary_impl_fn_ptr_t,
                                           vm_ext::MulContigFactory>(
            mul_dispatch_vector);

        auto mul_pyapi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                             arrayT dst, const event_vecT &depends = {}) {
            return vm_ext::binary_ufunc(exec_q, src1, src2, dst, depends,
                                        mul_dispatch_vector);
        };
        m.def("_mul", mul_pyapi,
              "Call `mul` function from OneMKL VM library to performs element "
              "by element multiplication of vector `src1` by vector `src2` "
              "to resulting vector `dst`",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("depends") = py::list());

        auto mul_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src1,
                                          arrayT src2, arrayT dst) {
            return vm_ext::need_to_call_binary_ufunc(exec_q, src1, src2, dst,
                                                     mul_dispatch_vector);
        };
        m.def("_mkl_mul_to_call", mul_need_to_call_pyapi,
              "Check input arguments to answer if `mul` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"));
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

    // UnaryUfunc: ==== Sqr(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::SqrContigFactory>(
            sqr_dispatch_vector);

        auto sqr_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                             const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       sqr_dispatch_vector);
        };
        m.def(
            "_sqr", sqr_pyapi,
            "Call `sqr` from OneMKL VM library to performs element by element "
            "operation of squaring of vector `src` to resulting vector `dst`",
            py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
            py::arg("depends") = py::list());

        auto sqr_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                          arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    sqr_dispatch_vector);
        };
        m.def("_mkl_sqr_to_call", sqr_need_to_call_pyapi,
              "Check input arguments to answer if `sqr` function from "
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

    // BinaryUfunc: ==== Sub(x1, x2) ====
    {
        vm_ext::init_ufunc_dispatch_vector<binary_impl_fn_ptr_t,
                                           vm_ext::SubContigFactory>(
            sub_dispatch_vector);

        auto sub_pyapi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                             arrayT dst, const event_vecT &depends = {}) {
            return vm_ext::binary_ufunc(exec_q, src1, src2, dst, depends,
                                        sub_dispatch_vector);
        };
        m.def("_sub", sub_pyapi,
              "Call `sub` function from OneMKL VM library to performs element "
              "by element subtraction of vector `src1` by vector `src2` "
              "to resulting vector `dst`",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("depends") = py::list());

        auto sub_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src1,
                                          arrayT src2, arrayT dst) {
            return vm_ext::need_to_call_binary_ufunc(exec_q, src1, src2, dst,
                                                     sub_dispatch_vector);
        };
        m.def("_mkl_sub_to_call", sub_need_to_call_pyapi,
              "Check input arguments to answer if `sub` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"));
    }

    // UnaryUfunc: ==== Trunc(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::TruncContigFactory>(
            trunc_dispatch_vector);

        auto trunc_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                               const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       trunc_dispatch_vector);
        };
        m.def("_trunc", trunc_pyapi,
              "Call `trunc` function from OneMKL VM library to compute "
              "the truncated value of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto trunc_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                            arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    trunc_dispatch_vector);
        };
        m.def("_mkl_trunc_to_call", trunc_need_to_call_pyapi,
              "Check input arguments to answer if `trunc` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }
}
