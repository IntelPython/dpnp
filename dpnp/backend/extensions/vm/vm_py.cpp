//*****************************************************************************
// Copyright (c) 2023-2025, Intel Corporation
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
// This file defines functions of dpnp.backend._vm_impl extensions
//
//*****************************************************************************

#if not defined(USE_ONEMKL_INTERFACES)
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
#include "conj.hpp"
#include "cos.hpp"
#include "cosh.hpp"
#include "div.hpp"
#include "exp.hpp"
#include "exp2.hpp"
#include "expm1.hpp"
#include "floor.hpp"
#include "fmax.hpp"
#include "fmin.hpp"
#include "fmod.hpp"
#include "hypot.hpp"
#include "ln.hpp"
#include "log10.hpp"
#include "log1p.hpp"
#include "log2.hpp"
#include "mul.hpp"
#include "nextafter.hpp"
#include "pow.hpp"
#include "rint.hpp"
#include "sin.hpp"
#include "sinh.hpp"
#include "sqr.hpp"
#include "sqrt.hpp"
#include "sub.hpp"
#include "tan.hpp"
#include "tanh.hpp"
#include "trunc.hpp"

namespace vm_ns = dpnp::extensions::vm;
#endif // USE_ONEMKL_INTERFACES

#include <pybind11/pybind11.h>

PYBIND11_MODULE(_vm_impl, m)
{
#if not defined(USE_ONEMKL_INTERFACES)
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
    vm_ns::init_fmax(m);
    vm_ns::init_fmin(m);
    vm_ns::init_fmod(m);
    vm_ns::init_hypot(m);
    vm_ns::init_ln(m);
    vm_ns::init_log10(m);
    vm_ns::init_log1p(m);
    vm_ns::init_log2(m);
    vm_ns::init_mul(m);
    vm_ns::init_nextafter(m);
    vm_ns::init_pow(m);
    vm_ns::init_rint(m);
    vm_ns::init_sin(m);
    vm_ns::init_sinh(m);
    vm_ns::init_sqr(m);
    vm_ns::init_sqrt(m);
    vm_ns::init_sub(m);
    vm_ns::init_tan(m);
    vm_ns::init_tanh(m);
    vm_ns::init_trunc(m);
#endif // USE_ONEMKL_INTERFACES
    m.def(
        "_is_available",
        [](void) {
#if defined(USE_ONEMKL_INTERFACES)
            return false;
#else
            return true;
#endif // USE_ONEMKL_INTERFACES
        },
        "Check if the OneMKL VM library can be used.");
}
