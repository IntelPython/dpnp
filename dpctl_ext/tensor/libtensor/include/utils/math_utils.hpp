//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
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
///
/// \file
/// This file defines math utility functions.
//===----------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <complex>
#include <limits>

#include <sycl/sycl.hpp>

namespace dpctl::tensor::math_utils
{
template <typename T>
bool less_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2)
               ? (imag1 < imag2)
               : (real1 < real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T>
bool greater_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2)
               ? (imag1 > imag2)
               : (real1 > real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T>
bool less_equal_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2)
               ? (imag1 <= imag2)
               : (real1 < real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T>
bool greater_equal_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    return (real1 == real2)
               ? (imag1 >= imag2)
               : (real1 > real2 && !std::isnan(imag1) && !std::isnan(imag2));
}

template <typename T>
T max_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    bool isnan_imag1 = std::isnan(imag1);
    bool gt = (real1 == real2)
                  ? (imag1 > imag2)
                  : (real1 > real2 && !isnan_imag1 && !std::isnan(imag2));
    return (std::isnan(real1) || isnan_imag1 || gt) ? x1 : x2;
}

template <typename T>
T min_complex(const T &x1, const T &x2)
{
    using realT = typename T::value_type;
    realT real1 = std::real(x1);
    realT real2 = std::real(x2);
    realT imag1 = std::imag(x1);
    realT imag2 = std::imag(x2);

    bool isnan_imag1 = std::isnan(imag1);
    bool lt = (real1 == real2)
                  ? (imag1 < imag2)
                  : (real1 < real2 && !isnan_imag1 && !std::isnan(imag2));
    return (std::isnan(real1) || isnan_imag1 || lt) ? x1 : x2;
}

template <typename T>
T logaddexp(T x, T y)
{
    if (x == y) { // handle signed infinities
        const T log2 = sycl::log(T(2));
        return x + log2;
    }
    else {
        const T tmp = x - y;
        static constexpr T zero(0);

        return (tmp > zero)
                   ? (x + sycl::log1p(sycl::exp(-tmp)))
                   : ((tmp <= zero) ? y + sycl::log1p(sycl::exp(tmp))
                                    : std::numeric_limits<T>::quiet_NaN());
    }
}
} // namespace dpctl::tensor::math_utils
