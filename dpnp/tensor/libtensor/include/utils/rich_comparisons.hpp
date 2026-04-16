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
/// This file defines functions of dpctl.tensor._tensor_sorting_impl
/// extension.
//===---------------------------------------------------------------------===//

#pragma once

#include <cmath>
#include <complex>
#include <functional>
#include <type_traits>

#include "sycl/sycl.hpp"

namespace dpctl::tensor::rich_comparisons
{

namespace detail
{
template <typename fpT>
struct ExtendedRealFPLess
{
    /* [R, nan] */
    bool operator()(const fpT v1, const fpT v2) const
    {
        return (!std::isnan(v1) && (std::isnan(v2) || (v1 < v2)));
    }
};

template <typename fpT>
struct ExtendedRealFPGreater
{
    bool operator()(const fpT v1, const fpT v2) const
    {
        return (!std::isnan(v2) && (std::isnan(v1) || (v2 < v1)));
    }
};

template <typename cT>
struct ExtendedComplexFPLess
{
    /* [(R, R), (R, nan), (nan, R), (nan, nan)] */

    bool operator()(const cT &v1, const cT &v2) const
    {
        using realT = typename cT::value_type;

        const realT real1 = std::real(v1);
        const realT real2 = std::real(v2);

        const bool r1_nan = std::isnan(real1);
        const bool r2_nan = std::isnan(real2);

        const realT imag1 = std::imag(v1);
        const realT imag2 = std::imag(v2);

        const bool i1_nan = std::isnan(imag1);
        const bool i2_nan = std::isnan(imag2);

        const int idx1 = ((r1_nan) ? 2 : 0) + ((i1_nan) ? 1 : 0);
        const int idx2 = ((r2_nan) ? 2 : 0) + ((i2_nan) ? 1 : 0);

        const bool res =
            !(r1_nan && i1_nan) &&
            ((idx1 < idx2) ||
             ((idx1 == idx2) &&
              ((r1_nan && !i1_nan && (imag1 < imag2)) ||
               (!r1_nan && i1_nan && (real1 < real2)) ||
               (!r1_nan && !i1_nan &&
                ((real1 < real2) || (!(real2 < real1) && (imag1 < imag2)))))));

        return res;
    }
};

template <typename cT>
struct ExtendedComplexFPGreater
{
    bool operator()(const cT &v1, const cT &v2) const
    {
        auto less_ = ExtendedComplexFPLess<cT>{};
        return less_(v2, v1);
    }
};

template <typename T>
inline constexpr bool is_fp_v =
    (std::is_same_v<T, sycl::half> || std::is_same_v<T, float> ||
     std::is_same_v<T, double>);

} // namespace detail

template <typename argTy>
struct AscendingSorter
{
    using type = std::conditional_t<detail::is_fp_v<argTy>,
                                    detail::ExtendedRealFPLess<argTy>,
                                    std::less<argTy>>;
};

template <typename T>
struct AscendingSorter<std::complex<T>>
{
    using type = detail::ExtendedComplexFPLess<std::complex<T>>;
};

template <typename argTy>
struct DescendingSorter
{
    using type = std::conditional_t<detail::is_fp_v<argTy>,
                                    detail::ExtendedRealFPGreater<argTy>,
                                    std::greater<argTy>>;
};

template <typename T>
struct DescendingSorter<std::complex<T>>
{
    using type = detail::ExtendedComplexFPGreater<std::complex<T>>;
};

} // namespace dpctl::tensor::rich_comparisons
