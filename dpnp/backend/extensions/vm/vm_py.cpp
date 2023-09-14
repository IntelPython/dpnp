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

#include "abs.hpp"
#include "acos.hpp"
#include "acosh.hpp"
#include "add.hpp"
#include "asin.hpp"
#include "asinh.hpp"
#include "atan.hpp"
#include "atan2.hpp"
#include "atanh.hpp"
#include "ceil.hpp"
#include "common.hpp"
#include "conj.hpp"
#include "cos.hpp"
#include "cosh.hpp"
#include "div.hpp"
#include "floor.hpp"
#include "hypot.hpp"
#include "ln.hpp"
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

using vm_ext::binary_impl_fn_ptr_t;
using vm_ext::unary_impl_fn_ptr_t;

static unary_impl_fn_ptr_t abs_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t acos_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t acosh_dispatch_vector[dpctl_td_ns::num_types];
static binary_impl_fn_ptr_t add_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t asin_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t asinh_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t atan_dispatch_vector[dpctl_td_ns::num_types];
static binary_impl_fn_ptr_t atan2_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t atanh_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t ceil_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t conj_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t cos_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t cosh_dispatch_vector[dpctl_td_ns::num_types];
static binary_impl_fn_ptr_t div_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t floor_dispatch_vector[dpctl_td_ns::num_types];
static binary_impl_fn_ptr_t hypot_dispatch_vector[dpctl_td_ns::num_types];
static unary_impl_fn_ptr_t ln_dispatch_vector[dpctl_td_ns::num_types];
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

    // UnaryUfunc: ==== Abs(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::AbsContigFactory>(
            abs_dispatch_vector);

        auto abs_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                             const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       abs_dispatch_vector);
        };
        m.def("_abs", abs_pyapi,
              "Call `abs` function from OneMKL VM library to compute "
              "the absolute of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto abs_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                          arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    abs_dispatch_vector);
        };
        m.def("_mkl_abs_to_call", abs_need_to_call_pyapi,
              "Check input arguments to answer if `abs` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

    // UnaryUfunc: ==== Acos(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::AcosContigFactory>(
            acos_dispatch_vector);

        auto acos_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                              const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       acos_dispatch_vector);
        };
        m.def("_acos", acos_pyapi,
              "Call `acos` function from OneMKL VM library to compute "
              "inverse cosine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto acos_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                           arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    acos_dispatch_vector);
        };
        m.def("_mkl_acos_to_call", acos_need_to_call_pyapi,
              "Check input arguments to answer if `acos` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

    // UnaryUfunc: ==== Acosh(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::AcoshContigFactory>(
            acosh_dispatch_vector);

        auto acosh_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                               const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       acosh_dispatch_vector);
        };
        m.def("_acosh", acosh_pyapi,
              "Call `acosh` function from OneMKL VM library to compute "
              "inverse cosine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto acosh_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                            arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    acosh_dispatch_vector);
        };
        m.def("_mkl_acosh_to_call", acosh_need_to_call_pyapi,
              "Check input arguments to answer if `acosh` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

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

    // UnaryUfunc: ==== Asin(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::AsinContigFactory>(
            asin_dispatch_vector);

        auto asin_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                              const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       asin_dispatch_vector);
        };
        m.def("_asin", asin_pyapi,
              "Call `asin` function from OneMKL VM library to compute "
              "inverse sine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto asin_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                           arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    asin_dispatch_vector);
        };
        m.def("_mkl_asin_to_call", asin_need_to_call_pyapi,
              "Check input arguments to answer if `asin` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

    // UnaryUfunc: ==== Asinh(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::AsinhContigFactory>(
            asinh_dispatch_vector);

        auto asinh_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                               const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       asinh_dispatch_vector);
        };
        m.def("_asinh", asinh_pyapi,
              "Call `asinh` function from OneMKL VM library to compute "
              "inverse cosine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto asinh_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                            arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    asinh_dispatch_vector);
        };
        m.def("_mkl_asinh_to_call", asinh_need_to_call_pyapi,
              "Check input arguments to answer if `asinh` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

    // UnaryUfunc: ==== Atan(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::AtanContigFactory>(
            atan_dispatch_vector);

        auto atan_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                              const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       atan_dispatch_vector);
        };
        m.def("_atan", atan_pyapi,
              "Call `atan` function from OneMKL VM library to compute "
              "inverse tangent of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto atan_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                           arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    atan_dispatch_vector);
        };
        m.def("_mkl_atan_to_call", atan_need_to_call_pyapi,
              "Check input arguments to answer if `atan` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
    }

    // BinaryUfunc: ==== Atan2(x1, x2) ====
    {
        vm_ext::init_ufunc_dispatch_vector<binary_impl_fn_ptr_t,
                                           vm_ext::Atan2ContigFactory>(
            atan2_dispatch_vector);

        auto atan2_pyapi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                               arrayT dst, const event_vecT &depends = {}) {
            return vm_ext::binary_ufunc(exec_q, src1, src2, dst, depends,
                                        atan2_dispatch_vector);
        };
        m.def("_atan2", atan2_pyapi,
              "Call `atan2` function from OneMKL VM library to compute element "
              "by element inverse tangent of `x1/x2`",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("depends") = py::list());

        auto atan2_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src1,
                                            arrayT src2, arrayT dst) {
            return vm_ext::need_to_call_binary_ufunc(exec_q, src1, src2, dst,
                                                     atan2_dispatch_vector);
        };
        m.def("_mkl_atan2_to_call", atan2_need_to_call_pyapi,
              "Check input arguments to answer if `atan2` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"));
    }

    // UnaryUfunc: ==== Atanh(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::AtanhContigFactory>(
            atanh_dispatch_vector);

        auto atanh_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                               const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       atanh_dispatch_vector);
        };
        m.def("_atanh", atanh_pyapi,
              "Call `atanh` function from OneMKL VM library to compute "
              "inverse cosine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto atanh_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                            arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    atanh_dispatch_vector);
        };
        m.def("_mkl_atanh_to_call", atanh_need_to_call_pyapi,
              "Check input arguments to answer if `atanh` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
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

    // UnaryUfunc: ==== Conj(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::ConjContigFactory>(
            conj_dispatch_vector);

        auto conj_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                              const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       conj_dispatch_vector);
        };
        m.def("_conj", conj_pyapi,
              "Call `conj` function from OneMKL VM library to compute "
              "conjugate of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto conj_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                           arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    conj_dispatch_vector);
        };
        m.def("_mkl_conj_to_call", conj_need_to_call_pyapi,
              "Check input arguments to answer if `conj` function from "
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

    // UnaryUfunc: ==== Cosh(x) ====
    {
        vm_ext::init_ufunc_dispatch_vector<unary_impl_fn_ptr_t,
                                           vm_ext::CoshContigFactory>(
            cosh_dispatch_vector);

        auto cosh_pyapi = [&](sycl::queue exec_q, arrayT src, arrayT dst,
                              const event_vecT &depends = {}) {
            return vm_ext::unary_ufunc(exec_q, src, dst, depends,
                                       cosh_dispatch_vector);
        };
        m.def("_cosh", cosh_pyapi,
              "Call `cosh` function from OneMKL VM library to compute "
              "inverse cosine of vector elements",
              py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
              py::arg("depends") = py::list());

        auto cosh_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src,
                                           arrayT dst) {
            return vm_ext::need_to_call_unary_ufunc(exec_q, src, dst,
                                                    cosh_dispatch_vector);
        };
        m.def("_mkl_cosh_to_call", cosh_need_to_call_pyapi,
              "Check input arguments to answer if `cosh` function from "
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

    // BinaryUfunc: ==== Hypot(x1, x2) ====
    {
        vm_ext::init_ufunc_dispatch_vector<binary_impl_fn_ptr_t,
                                           vm_ext::HypotContigFactory>(
            hypot_dispatch_vector);

        auto hypot_pyapi = [&](sycl::queue exec_q, arrayT src1, arrayT src2,
                               arrayT dst, const event_vecT &depends = {}) {
            return vm_ext::binary_ufunc(exec_q, src1, src2, dst, depends,
                                        hypot_dispatch_vector);
        };
        m.def("_hypot", hypot_pyapi,
              "Call `hypot` function from OneMKL VM library to compute element "
              "by element hypotenuse of `x`",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"), py::arg("depends") = py::list());

        auto hypot_need_to_call_pyapi = [&](sycl::queue exec_q, arrayT src1,
                                            arrayT src2, arrayT dst) {
            return vm_ext::need_to_call_binary_ufunc(exec_q, src1, src2, dst,
                                                     hypot_dispatch_vector);
        };
        m.def("_mkl_hypot_to_call", hypot_need_to_call_pyapi,
              "Check input arguments to answer if `hypot` function from "
              "OneMKL VM library can be used",
              py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
              py::arg("dst"));
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
