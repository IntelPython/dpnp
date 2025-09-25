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

#pragma once

#include <type_traits>

#include <sycl/sycl.hpp>

/**
 * Include <sycl/ext/intel/math.hpp> only when targeting to Intel devices.
 */
#if defined(__INTEL_LLVM_COMPILER)
#define __SYCL_EXT_INTEL_MATH_SUPPORT
#endif

#if defined(__SYCL_EXT_INTEL_MATH_SUPPORT)
#include <sycl/ext/intel/math.hpp>
#else
#include <cmath>
#endif

namespace dpnp::kernels::erfs
{
template <typename OpT, typename ArgT, typename ResT>
struct BaseFunctor
{
    // is function constant for given ArgT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr ResT constant_value = ResT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both ArgT and ResT support subgroup store/load operation
    using supports_sg_loadstore = typename std::true_type;

    ResT operator()(const ArgT &x) const
    {
        if constexpr (std::is_same_v<ArgT, sycl::half> &&
                      std::is_same_v<ResT, float>) {
            // cast sycl::half to float for accuracy reasons
            return OpT::apply(float(x));
        }
        else {
            return OpT::apply(x);
        }
    }
};

#define MACRO_DEFINE_FUNCTOR(__name__, __f_name__)                             \
    struct __f_name__##Op                                                      \
    {                                                                          \
        template <typename Tp>                                                 \
        static Tp apply(const Tp &x)                                           \
        {                                                                      \
            return __name__(x);                                                \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <typename ArgT, typename ResT>                                    \
    using __f_name__##Functor = BaseFunctor<__f_name__##Op, ArgT, ResT>;

MACRO_DEFINE_FUNCTOR(sycl::erf, Erf);
MACRO_DEFINE_FUNCTOR(sycl::erfc, Erfc);
MACRO_DEFINE_FUNCTOR(
#if defined(__SYCL_EXT_INTEL_MATH_SUPPORT)
    sycl::ext::intel::math::erfcx,
#else
    std::erfc,
#endif
    Erfcx);
} // namespace dpnp::kernels::erfs
