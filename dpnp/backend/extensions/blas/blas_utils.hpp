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

#include <pybind11/pybind11.h>

namespace dpnp::extensions::blas
{
inline void standardize_strides_to_nonzero(std::vector<py::ssize_t> &strides,
                                           const py::ssize_t *shape)
{
    // When shape of an array along any particular dimension is 1, the stride
    // along that dimension is undefined. This function standardize the strides
    // by calculating the non-zero value of the strides.
    const std::size_t ndim = strides.size();
    const bool has_zero_stride =
        std::accumulate(strides.begin(), strides.end(), 1,
                        std::multiplies<py::ssize_t>{}) == 0;

    if (has_zero_stride) {
        for (std::size_t i = 0; i < ndim - 1; ++i) {
            strides[i] = strides[i] == 0
                             ? std::accumulate(shape + i + 1, shape + ndim, 1,
                                               std::multiplies<py::ssize_t>{})
                             : strides[i];
        }
        strides[ndim - 1] = strides[ndim - 1] == 0 ? 1 : strides[ndim - 1];
    }
}

inline void standardize_strides_to_zero(std::vector<py::ssize_t> &strides,
                                        const py::ssize_t *shape)
{
    // When shape of an array along any particular dimension is 1, the stride
    // along that dimension is undefined. This function standardize the strides
    // by defining such a stride as zero. This is because for these cases,
    // instead of copying the array into the additional dimension for batch
    // multiplication, we choose to use zero as the stride between different
    // matrices.  Therefore, the same array is used repeatedly.
    const std::size_t ndim = strides.size();

    for (std::size_t i = 0; i < ndim; ++i) {
        if (shape[i] <= 1) {
            strides[i] = 0;
        }
    }
}
} // namespace dpnp::extensions::blas
