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
/// This file defines kernels for tensor sort/argsort operations.
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>

namespace dpctl::tensor::kernels::search_sorted_detail
{

template <typename T>
T quotient_ceil(T n, T m)
{
    return (n + m - 1) / m;
}

template <typename Acc, typename Value, typename Compare>
std::size_t lower_bound_impl(const Acc acc,
                             const std::size_t first,
                             const std::size_t last,
                             const Value &value,
                             const Compare &comp)
{
    std::size_t n = last - first;
    std::size_t cur = n, start = first;
    std::size_t it;
    while (n > 0) {
        it = start;
        cur = n / 2;
        it += cur;
        if (comp(acc[it], value)) {
            n -= cur + 1, start = ++it;
        }
        else
            n = cur;
    }
    return start;
}

template <typename Acc, typename Value, typename Compare>
std::size_t upper_bound_impl(const Acc acc,
                             const std::size_t first,
                             const std::size_t last,
                             const Value &value,
                             const Compare &comp)
{
    const auto &op_comp = [comp](auto x, auto y) { return !comp(y, x); };
    return lower_bound_impl(acc, first, last, value, op_comp);
}

template <typename Acc, typename Value, typename Compare, typename IndexerT>
std::size_t lower_bound_indexed_impl(const Acc acc,
                                     std::size_t first,
                                     std::size_t last,
                                     const Value &value,
                                     const Compare &comp,
                                     const IndexerT &acc_indexer)
{
    std::size_t n = last - first;
    std::size_t cur = n, start = first;
    std::size_t it;
    while (n > 0) {
        it = start;
        cur = n / 2;
        it += cur;
        if (comp(acc[acc_indexer(it)], value)) {
            n -= cur + 1, start = ++it;
        }
        else
            n = cur;
    }
    return start;
}

template <typename Acc, typename Value, typename Compare, typename IndexerT>
std::size_t upper_bound_indexed_impl(const Acc acc,
                                     const std::size_t first,
                                     const std::size_t last,
                                     const Value &value,
                                     const Compare &comp,
                                     const IndexerT &acc_indexer)
{
    const auto &op_comp = [comp](auto x, auto y) { return !comp(y, x); };
    return lower_bound_indexed_impl(acc, first, last, value, op_comp,
                                    acc_indexer);
}

} // namespace dpctl::tensor::kernels::search_sorted_detail
