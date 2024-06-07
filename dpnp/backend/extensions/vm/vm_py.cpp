//*****************************************************************************
// Copyright (c) 2023-2024, Intel Corporation
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

#include "abs.hpp"
#include "acos.hpp"
#include "acosh.hpp"
#include "add.hpp"
#include "asin.hpp"
#include "asinh.hpp"
#include "atan.hpp"
#include "atan2.hpp"
#include "atanh.hpp"
#include "cbrt.hpp"
#include "ceil.hpp"
#include "common.hpp"
#include "conj.hpp"
#include "cos.hpp"
#include "cosh.hpp"
#include "div.hpp"
#include "exp.hpp"
#include "exp2.hpp"
#include "expm1.hpp"
#include "floor.hpp"
#include "hypot.hpp"
#include "ln.hpp"
#include "log10.hpp"
#include "log1p.hpp"
#include "log2.hpp"
#include "mul.hpp"
#include "pow.hpp"
#include "round.hpp"
#include "sin.hpp"
#include "sinh.hpp"
#include "sqr.hpp"
#include "sqrt.hpp"
#include "sub.hpp"
#include "tan.hpp"
#include "tanh.hpp"
#include "trunc.hpp"
#include "types_matrix.hpp"

namespace py = pybind11;
namespace vm_ext = dpnp::backend::ext::vm;
namespace vm_ns = dpnp::extensions::vm;

using vm_ext::binary_impl_fn_ptr_t;
using vm_ext::unary_impl_fn_ptr_t;

static binary_impl_fn_ptr_t mul_dispatch_vector[dpctl_td_ns::num_types];
static binary_impl_fn_ptr_t pow_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t round_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t sin_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t sinh_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t sqr_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t sqrt_dispatch_vector[dpctl_td_ns::num_types];
static binary_impl_fn_ptr_t sub_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t tan_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t tanh_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t trunc_dispatch_vector[dpctl_td_ns::num_types];

PYBIND11_MODULE(_vm_impl, m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    vm_ns::init_abs(m);
    vm_ns::init_acos(m);
    vm_ns::init_acosh(m);
    vm_ns::init_add(m);
    vm_ns::init_asin(m);
    vm_ns::init_asinh(m);
    vm_ns::init_atan(m);
    vm_ns::init_atan2(m);
    vm_ns::init_atanh(m);
    vm_ns::init_cbrt(m);
    vm_ns::init_ceil(m);
    vm_ns::init_conj(m);
    vm_ns::init_cos(m);
    vm_ns::init_cosh(m);
    vm_ns::init_div(m);
    vm_ns::init_exp(m);
    vm_ns::init_exp2(m);
    vm_ns::init_expm1(m);
    vm_ns::init_floor(m);
    vm_ns::init_hypot(m);
    vm_ns::init_ln(m);
    vm_ns::init_log10(m);
    vm_ns::init_log1p(m);
    vm_ns::init_log2(m);

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

    // BinaryUfunc: ==== Pow(x1, x2) ====
    {
        vm_ext::init_ufunc_dispatch_vector<binary_impl_fn_ptr_t,
                                           vm_ext::PowContigFactory>(
            pow_dispatch_vector);

        auto pow_pyapi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                             arrayT dst, const event_vecT &depends = {}) {
            return vm_ext::binary_ufunc(exec_q, src1, src2, dst, depends,
                                        pow_dispatch_vector);
        };
        m.def("_pow", pow_pyapi,
              "Call `pow` function from OneMKL VM library to performs element "
              "by element exponentiation of vector `src1` raised to the power "
              "of vector `src2` to resulting vector `dst`",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("depends") = py::list());

        auto pow_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src1,
                                          arrayT src2, arrayT dst) {
            return vm_ext::need_to_call_binary_ufunc(exec_q, src1, src2, dst,
                                                     pow_dispatch_vector);
        };
        m.def("_mkl_pow_to_call", pow_need_to_call_pyapi,
              "Check input arguments to answer if `pow` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"));
    }

    // UnaryUfunc: ==== Round(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::RoundContigFactory>(
            round_dispatch_vector);

        auto round_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                               const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       round_dispatch_vector);
        };
        m.def("_round", round_pyapi,
              "Call `rint` function from OneMKL VM library to compute "
              "the rounded value of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto round_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                            arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    round_dispatch_vector);
        };
        m.def("_mkl_round_to_call", round_need_to_call_pyapi,
              "Check input arguments to answer if `rint` function from "
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

    // UnaryUfunc: ==== Sinh(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::SinhContigFactory>(
            sinh_dispatch_vector);

        auto sinh_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                              const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       sinh_dispatch_vector);
        };
        m.def("_sinh", sinh_pyapi,
              "Call `sinh` function from OneMKL VM library to compute "
              "inverse cosine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto sinh_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                           arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    sinh_dispatch_vector);
        };
        m.def("_mkl_sinh_to_call", sinh_need_to_call_pyapi,
              "Check input arguments to answer if `sinh` function from "
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

    // UnaryUfunc: ==== Tan(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::TanContigFactory>(
            tan_dispatch_vector);

        auto tan_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                             const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       tan_dispatch_vector);
        };
        m.def("_tan", tan_pyapi,
              "Call `tan` function from OneMKL VM library to compute "
              "tangent of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto tan_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                          arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    tan_dispatch_vector);
        };
        m.def("_mkl_tan_to_call", tan_need_to_call_pyapi,
              "Check input arguments to answer if `tan` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

    // UnaryUfunc: ==== Tanh(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::TanhContigFactory>(
            tanh_dispatch_vector);

        auto tanh_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                              const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       tanh_dispatch_vector);
        };
        m.def("_tanh", tanh_pyapi,
              "Call `tanh` function from OneMKL VM library to compute "
              "inverse cosine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto tanh_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                           arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    tanh_dispatch_vector);
        };
        m.def("_mkl_tanh_to_call", tanh_need_to_call_pyapi,
              "Check input arguments to answer if `tanh` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
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
