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

#include "utils/type_dispatch.hpp"

namespace dpnp::extensions::py_internal::type_dispatch
{
/**
 * Extends dpctl::tensor::type_dispatch::TypeMapResultEntry helper structure
 * with support of the two result types.
 */
template <typename Ty,
          typename ArgTy,
          typename ResTy1 = ArgTy,
          typename ResTy2 = ArgTy>
struct TypeMapTwoResultsEntry : std::bool_constant<std::is_same_v<Ty, ArgTy>>
{
    using result_type1 = ResTy1;
    using result_type2 = ResTy2;
};

/**
 * Extends dpctl::tensor::type_dispatch::BinaryTypeMapResultEntry helper
 * structure with support of the two result types.
 */
template <typename Ty1,
          typename ArgTy1,
          typename Ty2,
          typename ArgTy2,
          typename ResTy1 = ArgTy1,
          typename ResTy2 = ArgTy2>
struct BinaryTypeMapTwoResultsEntry
    : std::bool_constant<std::conjunction_v<std::is_same<Ty1, ArgTy1>,
                                            std::is_same<Ty2, ArgTy2>>>
{
    using result_type1 = ResTy1;
    using result_type2 = ResTy2;
};

/**
 * Extends dpctl::tensor::type_dispatch::DefaultResultEntry helper structure
 * with support of the two result types.
 */
template <typename Ty = void>
struct DefaultTwoResultsEntry : std::true_type
{
    using result_type1 = Ty;
    using result_type2 = Ty;
};
} // namespace dpnp::extensions::py_internal::type_dispatch
