//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

#include <sycl/sycl.hpp>

// dpctl tensor headers
#include "utils/math_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::kernels::fmin
{
namespace mu_ns = dpctl::tensor::math_utils;
namespace tu_ns = dpctl::tensor::type_utils;

template <typename argT1, typename argT2, typename resT>
struct FminFunctor
{
    using supports_sg_loadstore = std::negation<
        std::disjunction<tu_ns::is_complex<argT1>, tu_ns::is_complex<argT2>>>;
    using supports_vec =
        std::conjunction<std::is_same<argT1, argT2>,
                         std::disjunction<std::is_floating_point<argT1>,
                                          std::is_same<argT1, sycl::half>>>;

    resT operator()(const argT1 &in1, const argT2 &in2) const
    {
        if constexpr (std::is_integral_v<argT1> && std::is_integral_v<argT2>) {
            return in1 <= in2 ? in1 : in2;
        }
        else if constexpr (tu_ns::is_complex<argT1>::value &&
                           tu_ns::is_complex<argT2>::value)
        {
            static_assert(std::is_same_v<argT1, argT2>);

            using realT = typename argT1::value_type;
            const realT in2r = std::real(in2);
            const realT in2i = std::imag(in2);

            if (sycl::isnan(in2r) || sycl::isnan(in2i) ||
                mu_ns::less_equal_complex<argT1>(in1, in2))
            {
                return in1;
            }
            return in2;
        }
        else {
            return sycl::fmin(in1, in2);
        }
    }

    template <int vec_sz>
    sycl::vec<resT, vec_sz>
        operator()(const sycl::vec<argT1, vec_sz> &in1,
                   const sycl::vec<argT2, vec_sz> &in2) const
    {
        return sycl::fmin(in1, in2);
    }
};
} // namespace dpnp::kernels::fmin
