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
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

// dpctl tensor headers
#include "utils/type_dispatch.hpp"

// dpctl namespace alias for type dispatch utilities
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;

namespace dpnp::extensions::sparse::types
{

/**
 * @brief Factory encoding the supported (value type, index type) combinations
 * for oneapi::mkl::sparse::gemv.
 *
 * oneMKL sparse BLAS supports:
 *   - float32              with int32 indices
 *   - float32              with int64 indices
 *   - float64              with int32 indices
 *   - float64              with int64 indices
 *   - complex<float>  (c64) with int32 indices
 *   - complex<float>  (c64) with int64 indices
 *   - complex<double> (c128) with int32 indices
 *   - complex<double> (c128) with int64 indices
 *
 * Complex support requires oneMKL >= 2023.x (sparse BLAS complex USM API).
 * The dispatch table entry is non-null only when the pair is registered here;
 * the Python layer falls back to A.dot(x) when the entry is nullptr.
 *
 * @tparam Tv  Value type of the sparse matrix and dense vectors.
 * @tparam Ti  Index type of the sparse matrix (row_ptr / col_ind arrays).
 */
template <typename Tv, typename Ti>
struct SparseGemvTypePairSupportFactory
{
    static constexpr bool is_defined = std::disjunction<
        // real single precision
        dpctl_td_ns::TypePairDefinedEntry<Tv, float,                   Ti, std::int32_t>,
        dpctl_td_ns::TypePairDefinedEntry<Tv, float,                   Ti, std::int64_t>,
        // real double precision
        dpctl_td_ns::TypePairDefinedEntry<Tv, double,                  Ti, std::int32_t>,
        dpctl_td_ns::TypePairDefinedEntry<Tv, double,                  Ti, std::int64_t>,
        // complex single precision
        dpctl_td_ns::TypePairDefinedEntry<Tv, std::complex<float>,     Ti, std::int32_t>,
        dpctl_td_ns::TypePairDefinedEntry<Tv, std::complex<float>,     Ti, std::int64_t>,
        // complex double precision
        dpctl_td_ns::TypePairDefinedEntry<Tv, std::complex<double>,    Ti, std::int32_t>,
        dpctl_td_ns::TypePairDefinedEntry<Tv, std::complex<double>,    Ti, std::int64_t>,
        // fall-through
        dpctl_td_ns::NotDefinedEntry>::is_defined;
};

} // namespace dpnp::extensions::sparse::types
