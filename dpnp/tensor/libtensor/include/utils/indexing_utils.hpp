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
/// This file defines utilities for handling out-of-bounds integer indices in
/// kernels that involve indexing operations, such as take, put, or advanced
/// tensor integer indexing.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "kernels/dpctl_tensor_types.hpp"

namespace dpctl::tensor::indexing_utils
{
using dpctl::tensor::ssize_t;

/*
 * ssize_t for indices is a design choice, dpctl::tensor::usm_ndarray
 * uses py::ssize_t for shapes and strides internally and Python uses
 * py_ssize_t for sizes of e.g. lists.
 */

template <typename IndT>
struct WrapIndex
{
    static_assert(std::is_integral_v<IndT>);

    ssize_t operator()(ssize_t max_item, IndT ind) const
    {
        ssize_t projected;
        static constexpr ssize_t unit(1);
        max_item = sycl::max(max_item, unit);

        static constexpr std::uintmax_t ind_max =
            std::numeric_limits<IndT>::max();
        static constexpr std::uintmax_t ssize_max =
            std::numeric_limits<ssize_t>::max();

        if constexpr (std::is_signed_v<IndT>) {
            static constexpr std::intmax_t ind_min =
                std::numeric_limits<IndT>::min();
            static constexpr std::intmax_t ssize_min =
                std::numeric_limits<ssize_t>::min();

            if constexpr (ind_max <= ssize_max && ind_min >= ssize_min) {
                const ssize_t ind_ = static_cast<ssize_t>(ind);
                const ssize_t lb = -max_item;
                const ssize_t ub = max_item - 1;
                projected = sycl::clamp(ind_, lb, ub);
            }
            else {
                const IndT lb = static_cast<IndT>(-max_item);
                const IndT ub = static_cast<IndT>(max_item - 1);
                projected = static_cast<ssize_t>(sycl::clamp(ind, lb, ub));
            }
            return (projected < 0) ? projected + max_item : projected;
        }
        else {
            if constexpr (ind_max <= ssize_max) {
                const ssize_t ind_ = static_cast<ssize_t>(ind);
                const ssize_t ub = max_item - 1;
                projected = sycl::min(ind_, ub);
            }
            else {
                const IndT ub = static_cast<IndT>(max_item - 1);
                projected = static_cast<ssize_t>(sycl::min(ind, ub));
            }
            return projected;
        }
    }
};

template <typename IndT>
struct ClipIndex
{
    static_assert(std::is_integral_v<IndT>);

    ssize_t operator()(ssize_t max_item, IndT ind) const
    {
        ssize_t projected;
        static constexpr ssize_t unit(1);
        max_item = sycl::max<ssize_t>(max_item, unit);

        static constexpr std::uintmax_t ind_max =
            std::numeric_limits<IndT>::max();
        static constexpr std::uintmax_t ssize_max =
            std::numeric_limits<ssize_t>::max();
        if constexpr (std::is_signed_v<IndT>) {
            static constexpr std::intmax_t ind_min =
                std::numeric_limits<IndT>::min();
            static constexpr std::intmax_t ssize_min =
                std::numeric_limits<ssize_t>::min();

            if constexpr (ind_max <= ssize_max && ind_min >= ssize_min) {
                const ssize_t ind_ = static_cast<ssize_t>(ind);
                static constexpr ssize_t lb(0);
                const ssize_t ub = max_item - 1;
                projected = sycl::clamp(ind_, lb, ub);
            }
            else {
                static constexpr IndT lb(0);
                const IndT ub = static_cast<IndT>(max_item - 1);
                projected = static_cast<std::size_t>(sycl::clamp(ind, lb, ub));
            }
        }
        else {
            if constexpr (ind_max <= ssize_max) {
                const ssize_t ind_ = static_cast<ssize_t>(ind);
                const ssize_t ub = max_item - 1;
                projected = sycl::min(ind_, ub);
            }
            else {
                const IndT ub = static_cast<IndT>(max_item - 1);
                projected = static_cast<ssize_t>(sycl::min(ind, ub));
            }
        }
        return projected;
    }
};
} // namespace dpctl::tensor::indexing_utils
