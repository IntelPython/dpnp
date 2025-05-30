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

#include <cmath>
#include <complex>
#include <memory>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// dpctl tensor headers
#include "dpctl4pybind11.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

#include "ext/common.hpp"
#include "kth_element1d.hpp"
#include "partitioning.hpp"

#include <chrono>
#include <iostream>

namespace sycl_exp = sycl::ext::oneapi::experimental;
namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace dpctl_utils = dpctl::tensor::alloc_utils;

using dpctl::tensor::usm_ndarray;

using namespace statistics::partitioning;
using namespace ext::common;

namespace
{

template <typename T>
T NextAfter(T x)
{
    if constexpr (std::is_floating_point<T>::value) {
        return sycl::nextafter(x, std::numeric_limits<T>::infinity());
    }
    else if constexpr (std::is_integral<T>::value) {
        if (x < std::numeric_limits<T>::max())
            return x + 1;
        else
            return x;
    }
    else if constexpr (type_utils::is_complex_v<T>) {
        if (x.imag() != std::numeric_limits<T>::infinity()) {
            return T{x.real(), NextAfter(x.imag())};
        }
        else if (x.real() != std::numeric_limits<T>::infinity()) {
            return T{NextAfter(x.real()), -x.imag()};
        }
        else {
            return x;
        }
    }
}

template <typename T>
struct pick_pivot_kernel;

template <typename T>
struct kth_sorter_kernel;

template <typename T>
struct KthElementF
{
    static std::tuple<bool, sycl::event>
        run_kth_sort(sycl::queue &exec_q,
                     const T *in,
                     const size_t k,
                     State<T> &state,
                     const std::vector<sycl::event> &depends)
    {
        auto device = exec_q.get_device();
        size_t local_mem_size = get_local_mem_size_in_bytes(device);
        size_t temp_memory_size =
            sycl_exp::default_sorters::joint_sorter<>::memory_required<T>(
                sycl::memory_scope::work_group, state.n);
        size_t loc_items_mem = sizeof(T) * state.n;

        if ((temp_memory_size + loc_items_mem) > local_mem_size)
            return {false, sycl::event{}};

        auto e = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            const uint32_t local_size = get_max_local_size(exec_q);
            const uint32_t WorkPI = CeilDiv(state.n, local_size);
            auto work_sz = make_ndrange(state.n, local_size, WorkPI);
            auto loc_items =
                sycl::local_accessor<T, 1>(sycl::range<1>(state.n), cgh);
            auto scratch = sycl::local_accessor<std::byte, 1>(
                sycl::range<1>(temp_memory_size), cgh);

            cgh.parallel_for<kth_sorter_kernel<T>>(
                work_sz, [=](sycl::nd_item<1> item) {
                    auto group = item.get_group();
                    auto sbg = item.get_sub_group();

                    if (state.stop[0])
                        return;

                    auto llid = item.get_local_linear_id();
                    uint32_t sbg_size = sbg.get_max_local_range()[0];
                    uint32_t sbg_llid = sbg.get_local_linear_id();
                    auto local_size = item.get_group_range(0);
                    uint32_t nan_count = 0;

                    uint32_t i_base =
                        sbg.get_group_id() * WorkPI * sbg_size + sbg_llid;
                    for (uint32_t i = 0; i < WorkPI; i++) {
                        uint32_t idx = i_base + i * sbg_size;
                        if (idx < state.n) {
                            loc_items[idx] = in[idx];
                            if (IsNan<T>::isnan(in[idx])) {
                                nan_count++;
                            }
                        }
                    }

                    nan_count = sycl::reduce_over_group(group, nan_count,
                                                        sycl::plus<>());
                    sycl::group_barrier(group);

                    auto gh = sycl_exp::group_with_scratchpad(
                        group, sycl::span{&scratch[0], temp_memory_size});
                    sycl_exp::joint_sort(gh, &loc_items[0],
                                         &loc_items[0] + state.n, Less<T>{});

                    sycl::group_barrier(group);

                    if (group.leader()) {
                        state.values[0] = loc_items[k];
                        state.values[1] = loc_items[k + 1];
                        state.target_found[0] = true;
                        state.counters.nan_count[0] = nan_count;
                    }
                });
        });

        return {true, e};
    }

    static sycl::event run_pick_pivot(sycl::queue &queue,
                                      T *in,
                                      T *out,
                                      uint64_t target,
                                      State<T> &state,
                                      uint64_t items_to_sort,
                                      uint64_t limit,
                                      const std::vector<sycl::event> &deps)
    {
        auto e = queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(deps);
            constexpr uint64_t group_size = 128;

            auto work_sz = make_ndrange(group_size, group_size, 1);

            size_t temp_memory_size =
                sycl_exp::default_sorters::joint_sorter<>::memory_required<T>(
                    sycl::memory_scope::work_group, limit);

            auto loc_items =
                sycl::local_accessor<T, 1>(sycl::range<1>(items_to_sort), cgh);
            auto scratch = sycl::local_accessor<std::byte, 1>(
                sycl::range<1>(temp_memory_size), cgh);

            cgh.parallel_for<pick_pivot_kernel<T>>(work_sz, [=](sycl::nd_item<1>
                                                                    item) {
                auto group = item.get_group();

                if (state.stop[0])
                    return;

                auto llid = item.get_local_linear_id();
                auto local_size = item.get_group_range(0);

                uint64_t num_elems = 0;
                bool target_found = false;

                T *_in = nullptr;
                if (group.leader()) {
                    state.update_counters();
                    auto less_count = state.counters.less_count[0];
                    bool left = target < less_count;
                    state.left[0] = left;

                    if (left) {
                        _in = in;
                        num_elems = state.iteration_counters.less_count[0];
                        if (target + 1 == less_count) {
                            _in[num_elems] = state.pivot[0];
                            state.counters.less_count[0] += 1;
                            num_elems += 1;
                        }
                    }
                    else {
                        num_elems =
                            state.iteration_counters.greater_equal_count[0];
                        _in = in + state.n - num_elems;

                        if (target + 1 <
                            less_count +
                                state.iteration_counters.equal_count[0]) {
                            state.values[0] = state.pivot[0];
                            state.values[1] = state.pivot[0];

                            state.stop[0] = true;
                            state.target_found[0] = true;
                            target_found = true;
                        }
                    }
                    state.reset_iteration_counters();
                }

                target_found = sycl::group_broadcast(group, target_found, 0);
                _in = sycl::group_broadcast(group, _in, 0);
                num_elems = sycl::group_broadcast(group, num_elems, 0);

                if (target_found) {
                    return;
                }

                if (num_elems <= limit) {
                    auto gh = sycl_exp::group_with_scratchpad(
                        group, sycl::span{&scratch[0], temp_memory_size});
                    if (num_elems > 0)
                        sycl_exp::joint_sort(gh, &_in[0], &_in[num_elems],
                                             Less<T>{});

                    if (group.leader()) {
                        uint64_t offset = state.counters.less_count[0];
                        if (state.left[0]) {
                            offset = state.counters.less_count[0] - num_elems;
                        }

                        int64_t idx = target - offset;

                        state.values[0] = _in[idx];
                        state.values[1] = _in[idx + 1];

                        state.stop[0] = true;
                        state.target_found[0] = true;
                    }

                    return;
                }

                uint64_t step = num_elems / items_to_sort;
                for (uint32_t i = llid; i < items_to_sort; i += local_size) {
                    loc_items[i] = std::numeric_limits<T>::max();
                    uint32_t idx = i * step;
                    if (idx < num_elems) {
                        loc_items[i] = _in[idx];
                    }
                }

                sycl::group_barrier(group);

                auto gh = sycl_exp::group_with_scratchpad(
                    group, sycl::span{&scratch[0], temp_memory_size});
                sycl_exp::joint_sort(gh, &loc_items[0],
                                     &loc_items[0] + items_to_sort, Less<T>{});

                state.num_elems[0] = num_elems;

                T new_pivot = loc_items[items_to_sort / 2];
                if (new_pivot != state.pivot[0] && !IsNan<T>::isnan(new_pivot))
                {
                    if (group.leader()) {
                        state.pivot[0] = new_pivot;
                    }
                    return;
                }

                auto start = llid + items_to_sort / 2 + 1;
                uint32_t index = start;
                for (uint32_t i = start; i < items_to_sort; i += local_size) {
                    if (loc_items[i] != new_pivot &&
                        !IsNan<T>::isnan(loc_items[i])) {
                        index = i;
                        break;
                    }
                }

                index =
                    sycl::reduce_over_group(group, index, sycl::minimum<>());
                if (group.leader()) {
                    if (loc_items[index] != new_pivot ||
                        !IsNan<T>::isnan(loc_items[index])) {
                        // if all values are Nan just use it as pivot
                        // to filter out all the Nans
                        state.pivot[0] = loc_items[index];
                    }
                    else {
                        // we are going to filter out new_pivot
                        // but we need to keep at least one since it
                        // could be our target (but not target + 1)
                        out[state.n - 1] = new_pivot;
                        state.iteration_counters.greater_equal_count[0] += 1;
                        state.counters.less_count[0] -= 1;
                        new_pivot = NextAfter(new_pivot);
                        state.pivot[0] = new_pivot;
                    }
                }
            });
        });

        return e;
    }

    static sycl::event run_partition(sycl::queue &exec_q,
                                     T *in,
                                     T *out,
                                     PartitionState<T> &state,
                                     const std::vector<sycl::event> &deps)
    {

        uint32_t group_size = 128;
        constexpr uint32_t WorkPI = 4;
        return run_partition_one_pivot_cpu<T, WorkPI>(exec_q, in, out, state,
                                                      deps, group_size);
    }

    static sycl::event run_kth_element(sycl::queue &exec_q,
                                       const T *in,
                                       T *partitioned,
                                       T *temp_buff,
                                       const size_t k,
                                       State<T> &state,
                                       PartitionState<T> &pstate,
                                       const std::vector<sycl::event> &depends)
    {
        auto [success, evt] = run_kth_sort(exec_q, in, k, state, depends);
        if (success) {
            return evt;
        }

        uint32_t items_to_sort = 127;
        uint32_t limit = 4 * (items_to_sort + 1);

        uint32_t iterations = 1;

        if (state.n > limit) {
            iterations = std::ceil(-std::log(double(state.n) / limit) /
                                   std::log(0.536)) +
                         1;

            // Ensure iterations are odd so the final result is always stored in
            // 'partitioned'
            iterations += 1 - iterations % 2;
        }

        auto prev = run_pick_pivot(exec_q, const_cast<T *>(in), partitioned, k,
                                   state, items_to_sort, limit, depends);
        prev = run_partition(exec_q, const_cast<T *>(in), partitioned, pstate,
                             {prev});

        T *_in = partitioned;
        T *_out = temp_buff;
        for (uint32_t i = 0; i < iterations - 1; ++i) {
            prev = run_pick_pivot(exec_q, _in, _out, k, state, items_to_sort,
                                  limit, {prev});
            prev = run_partition(exec_q, _in, _out, pstate, {prev});
            std::swap(_in, _out);
        }
        prev = run_pick_pivot(exec_q, _in, _out, k, state, items_to_sort, limit,
                              {prev});

        return prev;
    }

    static KthElement1d::RetT impl(sycl::queue &exec_queue,
                                   const void *v_ain,
                                   void *v_partitioned,
                                   const size_t a_size,
                                   const size_t k,
                                   const std::vector<sycl::event> &depends)
    {
        const T *ain = static_cast<const T *>(v_ain);
        T *partitioned = static_cast<T *>(v_partitioned);

        State<T> state(exec_queue, a_size, partitioned);
        PartitionState<T> pstate(state);

        exec_queue.wait();
        auto init_e = state.init(exec_queue, depends);
        init_e = pstate.init(exec_queue, {init_e});

        auto temp_buff = dpctl_utils::smart_malloc<T>(state.n, exec_queue,
                                                      sycl::usm::alloc::device);
        auto evt = run_kth_element(exec_queue, ain, partitioned,
                                   temp_buff.get(), k, state, pstate, {init_e});

        bool found = false;
        bool left = false;
        uint64_t less_count = 0;
        uint64_t greater_equal_count = 0;
        uint64_t num_elems = 0;
        uint64_t nan_count = 0;
        auto copy_evt = exec_queue.copy(state.target_found, &found, 1, evt);
        copy_evt = exec_queue.copy(state.left, &left, 1, copy_evt);
        copy_evt = exec_queue.copy(state.counters.less_count, &less_count, 1,
                                   copy_evt);
        copy_evt = exec_queue.copy(state.counters.greater_equal_count,
                                   &greater_equal_count, 1, copy_evt);
        copy_evt = exec_queue.copy(state.num_elems, &num_elems, 1, copy_evt);
        copy_evt =
            exec_queue.copy(state.counters.nan_count, &nan_count, 1, copy_evt);

        copy_evt.wait();

        uint64_t buff_offset = 0;
        uint64_t elems_offset = less_count;

        if (!found) {
            if (left) {
                elems_offset = less_count - num_elems;
            }
            else {
                buff_offset = a_size - num_elems;
            }
        }
        else {
            num_elems = 2;
            elems_offset = k;
        }

        state.cleanup(exec_queue);

        return {found, buff_offset, elems_offset, num_elems, nan_count};
    }
};

using SupportedTypes = std::tuple<uint32_t,
                                  int32_t,
                                  uint64_t,
                                  int64_t,
                                  float,
                                  double,
                                  std::complex<float>,
                                  std::complex<double>>;
} // namespace

KthElement1d::KthElement1d() : dispatch_table("a")
{
    dispatch_table.populate_dispatch_table<SupportedTypes, KthElementF>();
}

KthElement1d::RetT KthElement1d::call(const dpctl::tensor::usm_ndarray &a,
                                      dpctl::tensor::usm_ndarray &partitioned,
                                      const size_t k,
                                      const std::vector<sycl::event> &depends)
{
    validate(a, partitioned, k);

    const int a_typenum = a.get_typenum();
    auto kth_elem_func = dispatch_table.get(a_typenum);

    auto exec_q = a.get_queue();
    auto result = kth_elem_func(exec_q, a.get_data(), partitioned.get_data(),
                                a.get_shape(0), k, depends);

    return result;
}

std::unique_ptr<KthElement1d> kth;

void statistics::partitioning::populate_kth_element1d(py::module_ m)
{
    using namespace std::placeholders;

    kth.reset(new KthElement1d());

    auto kth_func = [kthp = kth.get()](
                        const dpctl::tensor::usm_ndarray &a,
                        dpctl::tensor::usm_ndarray &partitioned, const size_t k,
                        const std::vector<sycl::event> &depends) {
        return kthp->call(a, partitioned, k, depends);
    };

    m.def("kth_element", kth_func, "finding k and k+1 elements.", py::arg("a"),
          py::arg("partitioned"), py::arg("k"),
          py::arg("depends") = py::list());

    auto kth_dtypes = [kthp = kth.get()]() {
        return kthp->dispatch_table.get_all_supported_types();
    };

    m.def("kth_element_dtypes", kth_dtypes,
          "Get the supported data types for kth_element.");
}
