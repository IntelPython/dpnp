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
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines an implementation of the complex absolute value.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
#include <limits>

#include "sycl_complex.hpp"

namespace dpctl::tensor::kernels::detail
{

template <typename realT>
realT cabs(std::complex<realT> const &z)
{
    // Special values for cabs( x + y * 1j):
    //   * If x is either +infinity or -infinity and y is any value
    //   (including NaN), the result is +infinity.
    //   * If x is any value (including NaN) and y is either +infinity or
    //   -infinity, the result is +infinity.
    //   * If x is either +0 or -0, the result is equal to abs(y).
    //   * If y is either +0 or -0, the result is equal to abs(x).
    //   * If x is NaN and y is a finite number, the result is NaN.
    //   * If x is a finite number and y is NaN, the result is NaN.
    //   * If x is NaN and y is NaN, the result is NaN.

    const realT x = std::real(z);
    const realT y = std::imag(z);

    static constexpr realT q_nan = std::numeric_limits<realT>::quiet_NaN();
    static constexpr realT p_inf = std::numeric_limits<realT>::infinity();

    const realT res =
        std::isinf(x)
            ? p_inf
            : ((std::isinf(y)
                    ? p_inf
                    : ((std::isnan(x)
                            ? q_nan
                            : exprm_ns::abs(exprm_ns::complex<realT>(z))))));

    return res;
}

} // namespace dpctl::tensor::kernels::detail
