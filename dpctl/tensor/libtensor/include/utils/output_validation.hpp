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
/// This file defines utilities for determining if an array is a valid output
/// array.
//===----------------------------------------------------------------------===//

#pragma once

#include <stdexcept>

#include <pybind11/pybind11.h>

#include "dpctl4pybind11.hpp"

namespace dpctl::tensor::validation
{
namespace py = pybind11;

/*! @brief Raises a value error if an array is read-only.

    This should be called with an array before writing.*/
struct CheckWritable
{
    static void throw_if_not_writable(const dpctl::tensor::usm_ndarray &arr)
    {
        if (!arr.is_writable()) {
            throw py::value_error("output array is read-only.");
        }
        return;
    }
};

/*! @brief Raises a value error if an array's memory is not sufficiently ample
    to accommodate an input number of elements.

    This should be called with an array before writing.*/
struct AmpleMemory
{
    template <typename T>
    static void throw_if_not_ample(const dpctl::tensor::usm_ndarray &arr,
                                   T nelems)
    {
        auto arr_offsets = arr.get_minmax_offsets();
        T range = static_cast<T>(arr_offsets.second - arr_offsets.first);
        if (range + 1 < nelems) {
            throw py::value_error("Memory addressed by the output array is not "
                                  "sufficiently ample.");
        }
        return;
    }
};
} // namespace dpctl::tensor::validation
