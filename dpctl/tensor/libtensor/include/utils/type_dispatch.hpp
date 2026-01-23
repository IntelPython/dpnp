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
/// This file defines class to implement dispatch tables for pair of types
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#if __has_include(<dpnp4pybind11.hpp>)
#include "dpnp4pybind11.hpp"
#else
#include "dpctl4pybind11.hpp"
#endif

#include "type_dispatch_building.hpp"

namespace dpctl::tensor::type_dispatch
{
struct usm_ndarray_types
{
    int typenum_to_lookup_id(int typenum) const
    {
        using typenum_t = ::dpctl::tensor::type_dispatch::typenum_t;
        auto const &api = ::dpctl::detail::dpctl_capi::get();

        if (typenum == api.UAR_DOUBLE_) {
            return static_cast<int>(typenum_t::DOUBLE);
        }
        else if (typenum == api.UAR_INT64_) {
            return static_cast<int>(typenum_t::INT64);
        }
        else if (typenum == api.UAR_INT32_) {
            return static_cast<int>(typenum_t::INT32);
        }
        else if (typenum == api.UAR_BOOL_) {
            return static_cast<int>(typenum_t::BOOL);
        }
        else if (typenum == api.UAR_CDOUBLE_) {
            return static_cast<int>(typenum_t::CDOUBLE);
        }
        else if (typenum == api.UAR_FLOAT_) {
            return static_cast<int>(typenum_t::FLOAT);
        }
        else if (typenum == api.UAR_INT16_) {
            return static_cast<int>(typenum_t::INT16);
        }
        else if (typenum == api.UAR_INT8_) {
            return static_cast<int>(typenum_t::INT8);
        }
        else if (typenum == api.UAR_UINT64_) {
            return static_cast<int>(typenum_t::UINT64);
        }
        else if (typenum == api.UAR_UINT32_) {
            return static_cast<int>(typenum_t::UINT32);
        }
        else if (typenum == api.UAR_UINT16_) {
            return static_cast<int>(typenum_t::UINT16);
        }
        else if (typenum == api.UAR_UINT8_) {
            return static_cast<int>(typenum_t::UINT8);
        }
        else if (typenum == api.UAR_CFLOAT_) {
            return static_cast<int>(typenum_t::CFLOAT);
        }
        else if (typenum == api.UAR_HALF_) {
            return static_cast<int>(typenum_t::HALF);
        }
        else if (typenum == api.UAR_INT_ || typenum == api.UAR_UINT_) {
            switch (sizeof(int)) {
            case sizeof(std::int32_t):
                return ((typenum == api.UAR_INT_)
                            ? static_cast<int>(typenum_t::INT32)
                            : static_cast<int>(typenum_t::UINT32));
            case sizeof(std::int64_t):
                return ((typenum == api.UAR_INT_)
                            ? static_cast<int>(typenum_t::INT64)
                            : static_cast<int>(typenum_t::UINT64));
            default:
                throw_unrecognized_typenum_error(typenum);
            }
        }
        else if (typenum == api.UAR_LONGLONG_ || typenum == api.UAR_ULONGLONG_)
        {
            switch (sizeof(long long)) {
            case sizeof(std::int64_t):
                return ((typenum == api.UAR_LONGLONG_)
                            ? static_cast<int>(typenum_t::INT64)
                            : static_cast<int>(typenum_t::UINT64));
            default:
                throw_unrecognized_typenum_error(typenum);
            }
        }
        else {
            throw_unrecognized_typenum_error(typenum);
        }
        // return code signalling error, should never be reached
        assert(false);
        return -1;
    }

private:
    void throw_unrecognized_typenum_error(int typenum) const
    {
        throw std::runtime_error("Unrecognized typenum " +
                                 std::to_string(typenum) + " encountered.");
    }
};
} // namespace dpctl::tensor::type_dispatch
