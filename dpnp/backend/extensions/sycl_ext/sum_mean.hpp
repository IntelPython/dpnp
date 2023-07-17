//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
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

#include "dispatcher_utils.hpp"
#include <CL/sycl.hpp>
#include <tuple>

#include "utils/memory_overlap.hpp"
#include "utils/type_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace sycl_ext
{

namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

bool check_limitations(const dpctl::tensor::usm_ndarray &in,
                       const dpctl::tensor::usm_ndarray &out,
                       bool throw_on_fail = false)
{
    if (not in.is_c_contiguous() or not out.is_c_contiguous()) {
        if (throw_on_fail)
            throw py::value_error(
                "Input and output arrays must be c-contiguos");

        return false;
    }

    auto device = in.get_queue().get_device();
    auto local_mem_size = device.get_info<sycl::info::device::local_mem_size>();
    size_t out_full_size = out.get_size() * out.get_elemsize();
    if (out_full_size > local_mem_size) {
        if (throw_on_fail)
            throw py::value_error("Resulting array exceeds local memroy size" +
                                  std::to_string(local_mem_size));

        return false;
    }

    if (out.get_elemsize() == 64 and not device.has(sycl::aspect::atomic64)) {
        if (throw_on_fail)
            throw py::value_error("64-bit atomics are not supported");

        return false;
    }

    return true;
}

using SumMeanFuncT = sycl::event (*)(const dpctl::tensor::usm_ndarray &,
                                     dpctl::tensor::usm_ndarray &,
                                     const std::vector<sycl::event> &);

template <typename InT, typename OutT, bool Mean>
struct sum_mean
{
    static sycl::event call(sycl::queue exec_q,
                            const InT *in,
                            OutT *out,
                            const size_t height,
                            const size_t width,
                            const std::vector<sycl::event> &depends)
    {
        int local_size = std::min(
            1024,
            int(exec_q.get_device()
                    .get_info<sycl::info::device::max_work_group_size>()));

        int WorkPI = 32; // empirically found number
        auto _total_size = height * width;
        auto global_size = Align(DivUp(_total_size, WorkPI), local_size);

        auto fill_e = exec_q.fill(out, OutT(0), width, depends);

        return exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(fill_e);
            auto local_result =
                sycl::local_accessor<OutT>(sycl::range<1>(width), cgh);
            cgh.parallel_for<sum_mean<InT, OutT, Mean>>(
                sycl::nd_range<1>(global_size, local_size),
                [=](sycl::nd_item<1> item) {
                    auto grid = item.get_group_linear_id();
                    auto group = item.get_group();
                    auto llid = item.get_local_linear_id();

                    for (size_t i = llid; i < width; i += local_size) {
                        local_result[i] = 0;
                    }

                    sycl::group_barrier(group);

                    for (int i = 0; i < WorkPI; ++i) {
                        auto id =
                            grid * WorkPI * local_size + i * local_size + llid;
                        if (id < _total_size) {
                            sycl::atomic_ref<OutT, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group>
                                aresult(local_result[id % width]);
                            aresult += in[id];
                        }
                    }

                    sycl::group_barrier(group);

                    for (size_t i = llid; i < width; i += local_size) {
                        sycl::atomic_ref<OutT, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device>
                            autput(out[i]);

                        auto r = local_result[i];
                        autput += Mean ? r / height : r;
                    }
                });
        });
    }

    static sycl::event call(const dpctl::tensor::usm_ndarray &in,
                            dpctl::tensor::usm_ndarray &out,
                            const std::vector<sycl::event> &depends)
    {
        validate_params(in, out);
        return call(in.get_queue(), in.get_data<InT>(), out.get_data<OutT>(),
                    in.get_shape(0), in.get_shape(1), depends);
    }

    static void validate_params(const dpctl::tensor::usm_ndarray &in,
                                dpctl::tensor::usm_ndarray &out)
    {
        auto exec_q = in.get_queue();

        type_utils::validate_type_for_device<InT>(exec_q);
        type_utils::validate_type_for_device<OutT>(exec_q);

        // check compatibility of execution queue and allocation queue
        if (!dpctl::utils::queues_are_compatible(exec_q, {in, out})) {
            throw py::value_error(
                "Execution queue is not compatible with allocation queues");
        }

        if (in.get_shape(1) != out.get_shape(0))
            throw py::value_error(
                "Input array axis 1 size must match output array size");

        check_limitations(in, out, true);
    }
};

template <typename InT, typename OutT>
using SumOverAxisContig = sum_mean<InT, OutT, false>;
using SumInputTypes = std::tuple<int8_t,
                                 uint8_t,
                                 int16_t,
                                 uint16_t,
                                 int32_t,
                                 uint32_t,
                                 int64_t,
                                 uint64_t,
                                 float,
                                 double>;
using SumOutputTypes =
    std::tuple<int32_t, uint32_t, int64_t, uint64_t, float, double>;

template <typename InT, typename OutT>
using MeanOverAxisContig = sum_mean<InT, OutT, true>;
using MeanInputTypes = SumInputTypes;
using MeanOutputTypes = std::tuple<float, double>;

using SumOverAxisContigDispatcher =
    CartesianDispatcher<SumOverAxisContig,
                        SumMeanFuncT,
                        dpctl::tensor::usm_ndarray,
                        UsmArrayMatcher,
                        SumInputTypes,
                        SumOutputTypes>;

using MeanOverAxisContigDispatcher =
    CartesianDispatcher<MeanOverAxisContig,
                        SumMeanFuncT,
                        dpctl::tensor::usm_ndarray,
                        UsmArrayMatcher,
                        SumInputTypes,
                        SumOutputTypes>;
} // namespace sycl_ext
} // namespace ext
} // namespace backend
} // namespace dpnp
