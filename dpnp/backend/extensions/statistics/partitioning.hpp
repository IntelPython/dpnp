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

} // namespace statistics::partitioning

#include "partitioning_one_pivot_kernel_cpu.hpp"
#include "partitioning_one_pivot_kernel_gpu.hpp"

namespace statistics::partitioning
{
template <typename T>
sycl::event run_partition_one_pivot(sycl::queue &exec_q,
                                    T *in,
                                    T *out,
                                    PartitionState<T> &state,
                                    const std::vector<sycl::event> &deps)
{
    auto device = exec_q.get_device();

    if (device.is_gpu()) {
        constexpr uint32_t WorkPI = 8;
        constexpr uint32_t group_size = 128;

        return run_partition_one_pivot_gpu<T>(exec_q, in, out, state, deps,
                                              group_size, WorkPI);
    }
    else {
        constexpr uint32_t WorkPI = 4;
        constexpr uint32_t group_size = 128;

        return run_partition_one_pivot_cpu<T, WorkPI>(exec_q, in, out, state,
                                                      deps, group_size);
    }
}

void validate(const usm_ndarray &a,
              const usm_ndarray &partitioned,
              const size_t k);
} // namespace statistics::partitioning
