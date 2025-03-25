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

#include <complex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// clang-format off
// math_utils.hpp doesn't include sycl header but uses sycl types
// so sycl.hpp must be included before math_utils.hpp
#include <sycl/sycl.hpp>
#include "utils/math_utils.hpp"
// clang-format on

namespace math
{
namespace common
{
template <typename T>
struct Less
{
    bool operator()(const T &lhs, const T &rhs) const
    {
        return std::less{}(lhs, rhs);
    }
};

template <typename T>
struct Less<std::complex<T>>
{
    bool operator()(const std::complex<T> &lhs,
                    const std::complex<T> &rhs) const
    {
        return dpctl::tensor::math_utils::less_complex(lhs, rhs);
    }
};

template <typename T>
struct IsNan
{
    static bool isnan(const T &v)
    {
        if constexpr (std::is_floating_point_v<T> ||
                      std::is_same_v<T, sycl::half>) {
            return sycl::isnan(v);
        }

        return false;
    }
};

template <typename T>
struct IsNan<std::complex<T>>
{
    static bool isnan(const std::complex<T> &v)
    {
        T real1 = std::real(v);
        T imag1 = std::imag(v);
        return sycl::isnan(real1) || sycl::isnan(imag1);
    }
};


// This function is a copy from dpctl because it is not available in the public
// headers of dpctl.
pybind11::dtype dtype_from_typenum(int dst_typenum);

} // namespace common
} // namespace math
