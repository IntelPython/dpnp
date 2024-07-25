//*****************************************************************************
// Copyright (c) 2023-2024, Intel Corporation
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
#include <complex>
#include <cstring>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace dpnp::extensions::lapack::helper
{
namespace py = pybind11;

template <typename T>
struct value_type_of
{
    using type = T;
};

template <typename T>
struct value_type_of<std::complex<T>>
{
    using type = T;
};

// Rounds up the number `value` to the nearest multiple of `mult`.
template <typename intT>
intT round_up_mult(intT value, intT mult)
{
    intT q = (value + (mult - 1)) / mult;
    return q * mult;
}

// Checks if the shape array has any non-zero dimension.
inline bool check_zeros_shape(int ndim, const py::ssize_t *shape)
{
    size_t src_nelems(1);

    for (int i = 0; i < ndim; ++i) {
        src_nelems *= static_cast<size_t>(shape[i]);
    }
    return src_nelems == 0;
}
} // namespace dpnp::extensions::lapack::helper
