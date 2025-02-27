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

#include <oneapi/mkl/rng/device.hpp>

#include "utils/type_dispatch.hpp"

namespace dpnp::backend::ext::rng::host::dispatch
{
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace mkl_rng = oneapi::mkl::rng;

template <typename Ty, typename ArgTy, typename Method, typename argMethod>
struct TypePairDefinedEntry
    : std::bool_constant<std::is_same_v<Ty, ArgTy> &&
                         std::is_same_v<Method, argMethod>>
{
    static constexpr bool is_defined = true;
};

template <typename T, typename M>
struct GaussianTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        TypePairDefinedEntry<T,
                             double,
                             M,
                             mkl_rng::gaussian_method::by_default>,
        TypePairDefinedEntry<T,
                             double,
                             M,
                             mkl_rng::gaussian_method::box_muller2>,
        TypePairDefinedEntry<T, float, M, mkl_rng::gaussian_method::by_default>,
        TypePairDefinedEntry<T,
                             float,
                             M,
                             mkl_rng::gaussian_method::box_muller2>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};
} // namespace dpnp::backend::ext::rng::host::dispatch
