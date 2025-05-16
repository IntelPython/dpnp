//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

#include "utils/math_utils.hpp"
#include <sycl/sycl.hpp>
#include <type_traits>

#include <stdio.h>

#include "ext/common.hpp"

#include "partitioning.hpp"

using dpctl::tensor::usm_ndarray;

using ext::common::AtomicOp;
using ext::common::IsNan;
using ext::common::Less;
using ext::common::make_ndrange;

namespace statistics::partitioning
{

template <typename T>
struct partition_one_pivot_kernel_gpu;

template <typename T>
auto partition_one_pivot_func_gpu(sycl::handler &cgh,
                                  T *in,
                                  T *out,
                                  PartitionState<T> &state,
                                  uint32_t group_size,
                                  uint32_t WorkPI)
{
    auto loc_counters =
        sycl::local_accessor<uint32_t, 1>(sycl::range<1>(4), cgh);
    auto loc_global_counters =
        sycl::local_accessor<uint32_t, 1>(sycl::range<1>(2), cgh);
    auto loc_items =
        sycl::local_accessor<T, 1>(sycl::range<1>(WorkPI * group_size), cgh);

    return [=](sycl::nd_item<1> item) {
        if (state.stop[0])
            return;

        auto group = item.get_group();
        auto group_range = group.get_local_range(0);
        auto llid = item.get_local_linear_id();
        uint64_t items_per_group = group.get_local_range(0) * WorkPI;
        uint64_t num_elems = state.num_elems[0];

        if (group.get_group_id(0) * items_per_group >= num_elems)
            return;

        T *_in = nullptr;
        if (state.left[0]) {
            _in = in;
        }
        else {
            _in = in + state.n - num_elems;
        }

        auto value = state.pivot[0];

        auto sbg = item.get_sub_group();

        uint32_t sbg_size = sbg.get_max_local_range()[0];
        uint32_t sbg_work_size = sbg_size * WorkPI;
        uint32_t sbg_llid = sbg.get_local_linear_id();
        uint64_t i_base = (item.get_global_linear_id() - sbg_llid) * WorkPI;

        if (group.leader()) {
            loc_counters[0] = 0;
            loc_counters[1] = 0;
            loc_counters[2] = 0;
        }

        sycl::group_barrier(group);

        for (uint32_t _i = 0; _i < WorkPI; ++_i) {
            uint32_t less_count = 0;
            uint32_t equal_count = 0;
            uint32_t greater_equal_count = 0;

            uint32_t actual_count = 0;
            auto i = i_base + _i * sbg_size + sbg_llid;
            uint32_t valid = i < num_elems;
            auto val = valid ? _in[i] : 0;
            uint32_t less = (val < value) && valid;
            uint32_t equal = (val == value) && valid;

            auto le_pos =
                sycl::exclusive_scan_over_group(sbg, less, sycl::plus<>());
            auto ge_pos = sbg.get_local_linear_id() - le_pos;
            auto sbg_less_equal =
                sycl::reduce_over_group(sbg, less, sycl::plus<>());
            auto sbg_equal =
                sycl::reduce_over_group(sbg, equal, sycl::plus<>());
            auto tot_valid =
                sycl::reduce_over_group(sbg, valid, sycl::plus<>());
            auto sbg_greater = tot_valid - sbg_less_equal;

            uint32_t local_less_offset = 0;
            uint32_t local_gr_offset = 0;

            if (sbg.leader()) {
                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group>
                    gr_less_eq(loc_counters[0]);
                local_less_offset = gr_less_eq.fetch_add(sbg_less_equal);

                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group>
                    gr_eq(loc_counters[1]);
                gr_eq += sbg_equal;

                sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                 sycl::memory_scope::work_group>
                    gr_greater(loc_counters[2]);
                local_gr_offset = gr_greater.fetch_add(sbg_greater);
            }

            uint32_t local_less_offset_ =
                sycl::select_from_group(sbg, local_less_offset, 0);
            uint32_t local_gr_offset_ =
                sycl::select_from_group(sbg, local_gr_offset, 0);

            if (valid) {
                if (less) {
                    uint32_t ll_offset = local_less_offset_ + le_pos;
                    loc_items[ll_offset] = val;
                }
                else {
                    auto loc_gr_offset = group_range * WorkPI -
                                         local_gr_offset_ - sbg_greater +
                                         ge_pos;
                    loc_items[loc_gr_offset] = val;
                }
            }
        }

        sycl::group_barrier(group);

        if (group.leader()) {
            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                glbl_less_eq(state.iteration_counters.less_count[0]);
            auto global_less_eq_offset =
                glbl_less_eq.fetch_add(loc_counters[0]);

            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                glbl_eq(state.iteration_counters.equal_count[0]);
            glbl_eq += loc_counters[1];

            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed,
                             sycl::memory_scope::device>
                glbl_greater(state.iteration_counters.greater_equal_count[0]);
            auto global_gr_offset = glbl_greater.fetch_add(loc_counters[2]);

            loc_global_counters[0] = global_less_eq_offset;
            loc_global_counters[1] = global_gr_offset + loc_counters[2];
        }

        sycl::group_barrier(group);

        auto global_less_eq_offset = loc_global_counters[0];
        auto global_gr_offset = state.n - loc_global_counters[1];

        uint32_t sbg_id = sbg.get_group_id();
        for (uint32_t _i = 0; _i < WorkPI; ++_i) {
            uint32_t i = sbg_id * sbg_size * WorkPI + _i * sbg_size + sbg_llid;
            if (i < loc_counters[0]) {
                out[global_less_eq_offset + i] = loc_items[i];
            }
            else if (i < loc_counters[0] + loc_counters[2]) {
                auto global_gr_offset_ = global_gr_offset + i - loc_counters[0];
                uint32_t local_buff_offset = WorkPI * group_range -
                                             loc_counters[2] + i -
                                             loc_counters[0];

                out[global_gr_offset_] = loc_items[local_buff_offset];
            }
        }
    };
}

template <typename T>
sycl::event run_partition_one_pivot_gpu(sycl::queue &exec_q,
                                        T *in,
                                        T *out,
                                        PartitionState<T> &state,
                                        const std::vector<sycl::event> &deps,
                                        uint32_t group_size,
                                        uint32_t WorkPI)
{
    auto e = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(deps);

        auto work_range = make_ndrange(state.n, group_size, WorkPI);

        cgh.parallel_for<partition_one_pivot_kernel_gpu<T>>(
            work_range, partition_one_pivot_func_gpu<T>(cgh, in, out, state,
                                                        group_size, WorkPI));
    });

    return e;
}

} // namespace statistics::partitioning
