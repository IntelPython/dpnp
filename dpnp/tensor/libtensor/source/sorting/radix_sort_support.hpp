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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpnp.tensor._tensor_sorting_impl
/// extension.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

namespace dpctl::tensor::py_internal
{

template <typename Ty, typename ArgTy>
struct TypeDefinedEntry : std::bool_constant<std::is_same_v<Ty, ArgTy>>
{
    static constexpr bool is_defined = true;
};

struct NotDefinedEntry : std::true_type
{
    static constexpr bool is_defined = false;
};

template <typename T>
struct RadixSortSupportVector
{
    using resolver_t =
        typename std::disjunction<TypeDefinedEntry<T, bool>,
                                  TypeDefinedEntry<T, std::int8_t>,
                                  TypeDefinedEntry<T, std::uint8_t>,
                                  TypeDefinedEntry<T, std::int16_t>,
                                  TypeDefinedEntry<T, std::uint16_t>,
                                  TypeDefinedEntry<T, std::int32_t>,
                                  TypeDefinedEntry<T, std::uint32_t>,
                                  TypeDefinedEntry<T, std::int64_t>,
                                  TypeDefinedEntry<T, std::uint64_t>,
                                  TypeDefinedEntry<T, sycl::half>,
                                  TypeDefinedEntry<T, float>,
                                  TypeDefinedEntry<T, double>,
                                  NotDefinedEntry>;

    static constexpr bool is_defined = resolver_t::is_defined;
};

} // namespace dpctl::tensor::py_internal
