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

#pragma once

#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>

namespace dpnp::kernels::lcm
{
template <typename _Result, typename _Source>
_Result
__abs_impl(_Source __t, std::true_type)
{
    if (__t >= 0)
        return __t;
    // if (__t == ::std::numeric_limits<_Source>::min())
    //     return -static_cast<_Result>(__t);
    return -__t;
};

template <typename _Result, typename _Source>
_Result
__abs_impl(_Source __t, std::false_type)
{
    if (__t >= 0)
        return __t;
    // if (__t == ::std::numeric_limits<_Source>::min())
    //     return -static_cast<_Result>(__t);
    return -__t;
};

template <typename _Result, typename _Source>
constexpr _Result
__get_abs(_Source __t)
{
    return __abs_impl<_Result>(__t, std::is_signed<_Source>{});
}
template <typename argT1, typename argT2, typename resT>
struct LcmFunctor
{
    using supports_sg_loadstore = typename std::true_type;
    using supports_vec = typename std::false_type;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if (in1 == 0 || in2 == 0)
            return 0;

        using _Rp = std::common_type_t<argT1, argT2>;
        static_assert((std::is_same_v<_Rp, resT>), "Result type must be common type");
        if constexpr (std::is_same_v<_Rp, std::int8_t>) {
            static_assert((std::is_signed_v<_Rp>), "Result type int8 must be signed for sign type");
        }
        else if constexpr (std::is_same_v<_Rp, std::int32_t>) {
            static_assert((std::is_signed_v<_Rp>), "Result type int16 must be signed for sign type");
        }
        else if constexpr (std::is_same_v<_Rp, std::int32_t>) {
            static_assert((std::is_signed_v<_Rp>), "Result type int32 must be signed for sign type");
        }
        else if constexpr (std::is_same_v<_Rp, std::int64_t>) {
            static_assert((std::is_signed_v<_Rp>), "Result type int64 must be signed for sign type");
        }

        // resT val1 = sycl::abs(in1) / oneapi::dpl::gcd(in1, in2);
        // resT val2 = sycl::abs(in2);
        // _Rp val1 = __get_abs<_Rp>(in1) / oneapi::dpl::gcd(in1, in2); // does not work
        // _Rp val1 = sycl::abs<_Rp>(in1) / oneapi::dpl::gcd(in1, in2); // this works
        _Rp _v = __get_abs<_Rp>(in1);
        _Rp val1 = _v / oneapi::dpl::gcd(in1, in2);
        _Rp val2 = __get_abs<_Rp>(in2);

        return val1 * val2;

        // return oneapi::dpl::lcm(in1, in2);
    }
};
} // namespace dpnp::kernels::lcm
