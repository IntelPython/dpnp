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

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

namespace dpctl::tensor::kernels::sort_utils_detail
{

namespace syclexp = sycl::ext::oneapi::experimental;

template <class KernelName, typename T>
sycl::event iota_impl(sycl::queue &exec_q,
                      T *data,
                      std::size_t nelems,
                      const std::vector<sycl::event> &dependent_events)
{
    static constexpr std::uint32_t lws = 256;
    static constexpr std::uint32_t n_wi = 4;
    const std::size_t n_groups = (nelems + n_wi * lws - 1) / (n_wi * lws);

    sycl::range<1> gRange{n_groups * lws};
    sycl::range<1> lRange{lws};
    sycl::nd_range<1> ndRange{gRange, lRange};

    sycl::event e = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependent_events);
        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> it) {
            const std::size_t gid = it.get_global_linear_id();
            const auto &sg = it.get_sub_group();
            const std::uint32_t lane_id = sg.get_local_id()[0];

            const std::size_t offset = (gid - lane_id) * n_wi;
            const std::uint32_t max_sgSize = sg.get_max_local_range()[0];

            std::array<T, n_wi> stripe{};
#pragma unroll
            for (std::uint32_t i = 0; i < n_wi; ++i) {
                stripe[i] = T(offset + lane_id + i * max_sgSize);
            }

            if (offset + n_wi * max_sgSize < nelems) {
                static constexpr auto group_ls_props =
                    syclexp::properties{syclexp::data_placement_striped};

                auto out_multi_ptr = sycl::address_space_cast<
                    sycl::access::address_space::global_space,
                    sycl::access::decorated::yes>(&data[offset]);

                syclexp::group_store(sg, sycl::span<T, n_wi>{&stripe[0], n_wi},
                                     out_multi_ptr, group_ls_props);
            }
            else {
                for (std::size_t idx = offset + lane_id; idx < nelems;
                     idx += max_sgSize)
                {
                    data[idx] = T(idx);
                }
            }
        });
    });

    return e;
}

template <class KernelName, typename IndexTy>
sycl::event map_back_impl(sycl::queue &exec_q,
                          std::size_t nelems,
                          const IndexTy *flat_index_data,
                          IndexTy *reduced_index_data,
                          std::size_t row_size,
                          const std::vector<sycl::event> &dependent_events)
{
    static constexpr std::uint32_t lws = 64;
    static constexpr std::uint32_t n_wi = 4;
    const std::size_t n_groups = (nelems + lws * n_wi - 1) / (n_wi * lws);

    sycl::range<1> lRange{lws};
    sycl::range<1> gRange{n_groups * lws};
    sycl::nd_range<1> ndRange{gRange, lRange};

    sycl::event map_back_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependent_events);

        cgh.parallel_for<KernelName>(ndRange, [=](sycl::nd_item<1> it) {
            const std::size_t gid = it.get_global_linear_id();
            const auto &sg = it.get_sub_group();
            const std::uint32_t lane_id = sg.get_local_id()[0];
            const std::uint32_t sg_size = sg.get_max_local_range()[0];

            const std::size_t start_id = (gid - lane_id) * n_wi + lane_id;

#pragma unroll
            for (std::uint32_t i = 0; i < n_wi; ++i) {
                const std::size_t data_id = start_id + i * sg_size;

                if (data_id < nelems) {
                    const IndexTy linear_index = flat_index_data[data_id];
                    reduced_index_data[data_id] = (linear_index % row_size);
                }
            }
        });
    });

    return map_back_ev;
}

} // namespace dpctl::tensor::kernels::sort_utils_detail
