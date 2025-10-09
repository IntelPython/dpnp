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

#include <oneapi/mkl.hpp>

namespace dpnp::extensions::fft
{
namespace mkl_dft = oneapi::mkl::dft;

// Structure to map MKL precision to float/double types
template <mkl_dft::precision prec>
struct PrecisionType;

template <>
struct PrecisionType<mkl_dft::precision::SINGLE>
{
    using type = float;
};

template <>
struct PrecisionType<mkl_dft::precision::DOUBLE>
{
    using type = double;
};

// Structure to map combination of precision, domain, and is_forward flag to
// in/out types
template <mkl_dft::precision prec, mkl_dft::domain dom, bool is_forward>
struct ScaleType
{
    using type_in = void;
    using type_out = void;
};

// for r2c FFT, type_in is real and type_out is complex
// is_forward is true
template <mkl_dft::precision prec>
struct ScaleType<prec, mkl_dft::domain::REAL, true>
{
    using prec_type = typename PrecisionType<prec>::type;
    using type_in = prec_type;
    using type_out = std::complex<prec_type>;
};

// for c2r FFT, type_in is complex and type_out is real
// is_forward is false
template <mkl_dft::precision prec>
struct ScaleType<prec, mkl_dft::domain::REAL, false>
{
    using prec_type = typename PrecisionType<prec>::type;
    using type_in = std::complex<prec_type>;
    using type_out = prec_type;
};

// for c2c FFT, both type_in and type_out are complex
// regardless of is_fwd
template <mkl_dft::precision prec, bool is_fwd>
struct ScaleType<prec, mkl_dft::domain::COMPLEX, is_fwd>
{
    using prec_type = typename PrecisionType<prec>::type;
    using type_in = std::complex<prec_type>;
    using type_out = std::complex<prec_type>;
};
} // namespace dpnp::extensions::fft
