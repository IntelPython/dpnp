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

using dpctl::tensor::usm_ndarray;

using ext::common::AtomicOp;
using ext::common::IsNan;
using ext::common::Less;
using ext::common::make_ndrange;

namespace statistics::partitioning
{

struct Counters
{
    uint64_t *less_count;
    uint64_t *equal_count;
    uint64_t *greater_equal_count;
    uint64_t *nan_count;

    Counters(sycl::queue &queue)
    {
        less_count = sycl::malloc_device<uint64_t>(1, queue);
        equal_count = sycl::malloc_device<uint64_t>(1, queue);
        greater_equal_count = sycl::malloc_device<uint64_t>(1, queue);
        nan_count = sycl::malloc_device<uint64_t>(1, queue);
    };

    void cleanup(sycl::queue &queue)
    {
        sycl::free(less_count, queue);
        sycl::free(equal_count, queue);
        sycl::free(greater_equal_count, queue);
        sycl::free(nan_count, queue);
    }
};

template <typename T>
struct State
{
    Counters counters;
    Counters iteration_counters;

    bool *stop;
    bool *target_found;
    bool *left;

    T *pivot;
    T *values;

    size_t *num_elems;

    size_t n;

    State(sycl::queue &queue, size_t _n, T *values_buff)
        : counters(queue), iteration_counters(queue)
    {
        stop = sycl::malloc_device<bool>(1, queue);
        target_found = sycl::malloc_device<bool>(1, queue);
        left = sycl::malloc_device<bool>(1, queue);

        pivot = sycl::malloc_device<T>(1, queue);
        values = values_buff;

        num_elems = sycl::malloc_device<size_t>(1, queue);

        n = _n;
    }

    sycl::event init(sycl::queue &queue, const std::vector<sycl::event> &deps)
    {
        sycl::event fill_e =
            queue.fill<uint64_t>(counters.less_count, 0, 1, deps);
        fill_e = queue.fill<uint64_t>(counters.equal_count, 0, 1, {fill_e});
        fill_e =
            queue.fill<uint64_t>(counters.greater_equal_count, n, 1, {fill_e});
        fill_e = queue.fill<uint64_t>(counters.nan_count, 0, 1, {fill_e});
        fill_e = queue.fill<uint64_t>(num_elems, 0, 1, {fill_e});
        fill_e = queue.fill<bool>(stop, false, 1, {fill_e});
        fill_e = queue.fill<bool>(target_found, false, 1, {fill_e});
        fill_e = queue.fill<bool>(left, false, 1, {fill_e});
        fill_e = queue.fill<T>(pivot, 0, 1, {fill_e});

        return fill_e;
    }

    void update_counters() const
    {
        if (*left) {
            counters.less_count[0] -= iteration_counters.greater_equal_count[0];
            counters.greater_equal_count[0] +=
                iteration_counters.greater_equal_count[0];
        }
        else {
            counters.less_count[0] += iteration_counters.less_count[0];
            counters.greater_equal_count[0] -= iteration_counters.less_count[0];
        }
        counters.equal_count[0] = iteration_counters.equal_count[0];
        counters.nan_count[0] += iteration_counters.nan_count[0];
    }

    void reset_iteration_counters() const
    {
        iteration_counters.less_count[0] = 0;
        iteration_counters.equal_count[0] = 0;
        iteration_counters.greater_equal_count[0] = 0;
        iteration_counters.nan_count[0] = 0;
    }

    void cleanup(sycl::queue &queue)
    {
        counters.cleanup(queue);
        iteration_counters.cleanup(queue);

        sycl::free(stop, queue);
        sycl::free(target_found, queue);
        sycl::free(left, queue);

        sycl::free(num_elems, queue);
        sycl::free(pivot, queue);
    }
};

template <typename T>
struct PartitionState
{
    Counters iteration_counters;

    bool *stop;
    bool *left;

    T *pivot;

    size_t n;
    size_t *num_elems;

    PartitionState(State<T> &state)
        : iteration_counters(state.iteration_counters)
    {
        stop = state.stop;
        left = state.left;

        num_elems = state.num_elems;
        pivot = state.pivot;

        n = state.n;
    }

    sycl::event init(sycl::queue &queue, const std::vector<sycl::event> &deps)
    {
        sycl::event fill_e =
            queue.fill<uint64_t>(iteration_counters.less_count, n, 1, deps);
        fill_e = queue.fill<uint64_t>(iteration_counters.equal_count, 0, 1,
                                      {fill_e});
        fill_e = queue.fill<uint64_t>(iteration_counters.greater_equal_count, 0,
                                      1, {fill_e});
        fill_e =
            queue.fill<uint64_t>(iteration_counters.nan_count, 0, 1, {fill_e});

        return fill_e;
    }
};

template <typename T>
class partition_one_pivot_kernel;

template <typename T, uint32_t WorkPI>
void submit_partition_one_pivot(sycl::handler &cgh,
                                sycl::nd_range<1> work_sz,
                                T *in,
                                T *out,
                                PartitionState<T> &state)
{
    auto loc_counters =
        sycl::local_accessor<uint32_t, 1>(sycl::range<1>(4), cgh);
    cgh.parallel_for<partition_one_pivot_kernel<T>>(
        work_sz, [=](sycl::nd_item<1> item) {
            if (state.stop[0])
                return;

            auto group = item.get_group();
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

            uint64_t i_base =
                (item.get_global_linear_id() - sbg.get_local_linear_id()) *
                WorkPI;

            if (group.leader()) {
                loc_counters[0] = 0;
                loc_counters[1] = 0;
                loc_counters[2] = 0;
            }

            sycl::group_barrier(group);

            uint32_t less_count = 0;
            uint32_t equal_count = 0;
            uint32_t greater_equal_count = 0;
            uint32_t nan_count = 0;

            T values[WorkPI];
            uint32_t actual_count = 0;
            uint64_t local_i_base = i_base + sbg.get_local_linear_id();

            for (uint32_t _i = 0; _i < WorkPI; ++_i) {
                auto i = local_i_base + _i * sbg_size;
                if (i < num_elems) {
                    values[_i] = _in[i];
                    less_count += Less<T>{}(values[_i], value);
                    equal_count += values[_i] == value;
                    nan_count += IsNan<T>::isnan(values[_i]);
                    actual_count++;
                }
            }

            greater_equal_count = actual_count - less_count;

            auto sbg_less_equal =
                sycl::reduce_over_group(sbg, less_count, sycl::plus<>());
            auto sbg_equal =
                sycl::reduce_over_group(sbg, equal_count, sycl::plus<>());
            auto sbg_greater = sycl::reduce_over_group(sbg, greater_equal_count,
                                                       sycl::plus<>());

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

            local_less_offset =
                sycl::select_from_group(sbg, local_less_offset, 0);
            local_gr_offset = sycl::select_from_group(sbg, local_gr_offset, 0);

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
                    glbl_greater(
                        state.iteration_counters.greater_equal_count[0]);
                auto global_gr_offset = glbl_greater.fetch_add(loc_counters[2]);

                loc_counters[0] = global_less_eq_offset;
                loc_counters[2] = global_gr_offset;
            }

            sycl::group_barrier(group);

            auto sbg_less_offset = loc_counters[0] + local_less_offset;
            auto sbg_gr_offset =
                state.n - (loc_counters[2] + local_gr_offset + sbg_greater);

            uint32_t le_item_offset = 0;
            uint32_t gr_item_offset = 0;

            for (uint32_t _i = 0; _i < WorkPI; ++_i) {
                uint32_t less = values[_i] < value;
                auto le_pos =
                    sycl::exclusive_scan_over_group(sbg, less, sycl::plus<>());
                auto ge_pos = sbg.get_local_linear_id() - le_pos;

                auto total_le =
                    sycl::reduce_over_group(sbg, less, sycl::plus<>());
                auto total_gr = sbg_size - total_le;

                if (_i < actual_count) {
                    if (less) {
                        out[sbg_less_offset + le_item_offset + le_pos] =
                            values[_i];
                    }
                    else {
                        out[sbg_gr_offset + gr_item_offset + ge_pos] =
                            values[_i];
                    }
                    le_item_offset += total_le;
                    gr_item_offset += total_gr;
                }
            }
        });
}

} // namespace statistics::partitioning
