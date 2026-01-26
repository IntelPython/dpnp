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
/// This file defines utilities for selection of hyperparameters for kernels
/// implementing unary and binary elementwise functions for contiguous inputs
//===---------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <type_traits>

namespace dpctl::tensor::kernels::vec_size_utils
{
template <typename Ty1,
          typename ArgTy1,
          typename Ty2,
          typename ArgTy2,
          std::uint8_t vec_sz_v,
          std::uint8_t n_vecs_v>
struct BinaryContigHyperparameterSetEntry
    : std::conjunction<std::is_same<Ty1, ArgTy1>, std::is_same<Ty2, ArgTy2>>
{
    static constexpr std::uint8_t vec_sz = vec_sz_v;
    static constexpr std::uint8_t n_vecs = n_vecs_v;
};

template <typename Ty,
          typename ArgTy,
          std::uint8_t vec_sz_v,
          std::uint8_t n_vecs_v>
struct UnaryContigHyperparameterSetEntry : std::is_same<Ty, ArgTy>
{
    static constexpr std::uint8_t vec_sz = vec_sz_v;
    static constexpr std::uint8_t n_vecs = n_vecs_v;
};

template <std::uint8_t vec_sz_v, std::uint8_t n_vecs_v>
struct ContigHyperparameterSetDefault : std::true_type
{
    static constexpr std::uint8_t vec_sz = vec_sz_v;
    static constexpr std::uint8_t n_vecs = n_vecs_v;
};
} // namespace dpctl::tensor::kernels::vec_size_utils
