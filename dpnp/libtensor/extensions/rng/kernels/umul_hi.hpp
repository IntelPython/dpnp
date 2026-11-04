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

#pragma once

#include <cstdint>

namespace dpnp::kernels::rng
{
// The high 64 bits of the 128-bit product a * b, from 32-bit halves.
// sycl::mul_hi is not used: optimized DPC++ 2026.2 builds get it wrong on
// Intel GPUs when one 64-bit operand is a compile-time constant (all results
// wrong at -O2, right at -O0). oneMKL's device RNG uses 32-bit halves too.
inline std::uint64_t umul_hi(const std::uint64_t a, const std::uint64_t b)
{
    const std::uint64_t a_lo = a & 0xffffffffULL;
    const std::uint64_t a_hi = a >> 32;
    const std::uint64_t b_lo = b & 0xffffffffULL;
    const std::uint64_t b_hi = b >> 32;

    const std::uint64_t lo_lo = a_lo * b_lo;
    const std::uint64_t hi_lo = a_hi * b_lo;
    const std::uint64_t lo_hi = a_lo * b_hi;

    // the carry out of the low 64 bits
    const std::uint64_t mid =
        (lo_lo >> 32) + (hi_lo & 0xffffffffULL) + (lo_hi & 0xffffffffULL);
    return a_hi * b_hi + (hi_lo >> 32) + (lo_hi >> 32) + (mid >> 32);
}
} // namespace dpnp::kernels::rng
