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

#pragma once

#include <algorithm>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "ext/common.hpp"

namespace dpnp::kernels::sliding_window1d
{
using ext::common::CeilDiv;

namespace detail
{
template <typename SizeT>
SizeT get_global_linear_id(const uint32_t wpi, const sycl::nd_item<1> &item)
{
    auto sbgroup = item.get_sub_group();
    const auto sg_loc_id = sbgroup.get_local_linear_id();

    const SizeT sg_base_id = wpi * (item.get_global_linear_id() - sg_loc_id);
    const SizeT id = sg_base_id + sg_loc_id;

    return id;
}

template <typename SizeT>
uint32_t get_results_num(const uint32_t wpi,
                         const SizeT size,
                         const SizeT global_id,
                         const sycl::nd_item<1> &item)
{
    auto sbgroup = item.get_sub_group();

    const auto sbg_size = sbgroup.get_max_local_range()[0];
    const auto size_ = sycl::sub_sat(size, global_id);
    return std::min(SizeT(wpi), CeilDiv(size_, sbg_size));
}

template <typename Results,
          typename AData,
          typename VData,
          typename Op,
          typename Red>
void process_block(Results &results,
                   uint32_t r_size,
                   AData &a_data,
                   VData &v_data,
                   uint32_t block_size,
                   Op op,
                   Red red)
{
    for (uint32_t i = 0; i < block_size; ++i) {
        auto v_val = v_data.broadcast(i);
        for (uint32_t r = 0; r < r_size; ++r) {
            results[r] = red(results[r], op(a_data[r], v_val));
        }
        a_data.advance_left();
    }
}
} // namespace detail

template <uint32_t WorkPI,
          typename SpanT,
          typename KernelT,
          typename OpT,
          typename RedT,
          typename ResultT,
          template <typename, uint32_t>
          class RegistryDataT,
          template <typename, uint32_t>
          class RegistryWindowT>
class SlidingWindow1dFunctor
{
private:
    const SpanT a;
    const KernelT v;
    const OpT op;
    const RedT red;
    ResultT out;

    static constexpr std::uint32_t default_reg_data_size = 1;
    using SizeT = typename SpanT::size_type;

public:
    SlidingWindow1dFunctor(const SpanT &a_,
                           const KernelT &v_,
                           const OpT &op_,
                           const RedT &red_,
                           ResultT &out_)
        : a(a_), v(v_), op(op_), red(red_), out(out_)
    {
    }

    void operator()(sycl::nd_item<1> item) const
    {
        auto glid = detail::get_global_linear_id<SizeT>(WorkPI, item);

        auto results =
            RegistryDataT<typename ResultT::value_type, WorkPI>(item);
        results.fill(0);

        auto results_num =
            detail::get_results_num<SizeT>(WorkPI, out.size(), glid, item);

        const auto *a_begin = a.begin();
        const auto *a_end = a.end();

        auto sbgroup = item.get_sub_group();

        const auto chunks_count =
            CeilDiv(v.size(), sbgroup.get_max_local_range()[0]);

        const auto *a_ptr = &a.padded_begin()[glid];

        auto _a_load_cond = [a_begin, a_end](auto &&ptr) {
            return ptr >= a_begin && ptr < a_end;
        };

        auto a_data =
            RegistryWindowT<typename SpanT::value_type, WorkPI + 1>(item);
        a_ptr = a_data.load(a_ptr, _a_load_cond, 0);

        const auto *v_ptr = &v.begin()[sbgroup.get_local_linear_id()];
        auto v_size = v.size();

        for (uint32_t b = 0; b < chunks_count; ++b) {
            auto v_data = RegistryDataT<typename KernelT::value_type,
                                        default_reg_data_size>(item);
            v_ptr = v_data.load(v_ptr, v_data.x() < v_size, 0);

            uint32_t chunk_size_ = std::min(v_size, SizeT(v_data.total_size()));
            detail::process_block(results, results_num, a_data, v_data,
                                  chunk_size_, op, red);

            if (b != chunks_count - 1) {
                a_ptr = a_data.load_lane(a_data.size_y() - 1, a_ptr,
                                         _a_load_cond, 0);
                v_size -= v_data.total_size();
            }
        }

        auto *const out_ptr = out.begin();
        // auto *const out_end = out.end();

        auto y_start = glid;
        auto y_stop = std::min(y_start + WorkPI * results.size_x(), out.size());
        uint32_t i = 0;
        for (uint32_t y = y_start; y < y_stop; y += results.size_x()) {
            out_ptr[y] = results[i++];
        }
        // while the code itself seems to be valid, inside correlate
        // kernel it results in memory corruption. Further investigation
        // is needed. SAT-7693
        // corruption results.store(&out_ptr[glid],
        //               [out_end](auto &&ptr) { return ptr < out_end; });
    }
};

template <uint32_t WorkPI,
          typename SpanT,
          typename KernelT,
          typename OpT,
          typename RedT,
          typename ResultT,
          template <typename, uint32_t>
          class RegistryDataT,
          template <typename, uint32_t>
          class RegistryWindowT>
class SlidingWindow1dSmallFunctor
{
private:
    const SpanT a;
    const KernelT v;
    const OpT op;
    const RedT red;
    ResultT out;

    static constexpr std::uint32_t default_reg_data_size = 1;
    using SizeT = typename SpanT::size_type;

public:
    SlidingWindow1dSmallFunctor(const SpanT &a_,
                                const KernelT &v_,
                                const OpT &op_,
                                const RedT &red_,
                                ResultT &out_)
        : a(a_), v(v_), op(op_), red(red_), out(out_)
    {
    }

    void operator()(sycl::nd_item<1> item) const
    {
        auto glid = detail::get_global_linear_id<SizeT>(WorkPI, item);

        auto results =
            RegistryDataT<typename ResultT::value_type, WorkPI>(item);
        results.fill(0);

        auto sbgroup = item.get_sub_group();
        auto sg_size = sbgroup.get_max_local_range()[0];

        const uint32_t to_read = WorkPI * sg_size + v.size();
        const auto *a_begin = a.begin();

        const auto *a_ptr = &a.padded_begin()[glid];
        const auto *a_end = std::min(a_ptr + to_read, a.end());

        auto _a_load_cond = [a_begin, a_end](auto &&ptr) {
            return ptr >= a_begin && ptr < a_end;
        };

        auto a_data =
            RegistryWindowT<typename SpanT::value_type, WorkPI + 1>(item);
        a_data.load(a_ptr, _a_load_cond, 0);

        const auto *v_ptr = &v.begin()[sbgroup.get_local_linear_id()];
        auto v_size = v.size();

        auto v_data =
            RegistryDataT<typename KernelT::value_type, default_reg_data_size>(
                item);
        v_ptr = v_data.load(v_ptr, v_data.x() < v_size, 0);

        auto results_num =
            detail::get_results_num<SizeT>(WorkPI, out.size(), glid, item);

        detail::process_block(results, results_num, a_data, v_data, v_size, op,
                              red);

        auto *const out_ptr = out.begin();
        // auto *const out_end = out.end();

        auto y_start = glid;
        auto y_stop = std::min(y_start + WorkPI * results.size_x(), out.size());
        uint32_t i = 0;
        for (uint32_t y = y_start; y < y_stop; y += results.size_x()) {
            out_ptr[y] = results[i++];
        }
        // while the code itself seems to be valid, inside correlate
        // kernel it results in memory corruption. Further investigation
        // is needed. SAT-7693
        // corruption results.store(&out_ptr[glid],
        //               [out_end](auto &&ptr) { return ptr < out_end; });
    }
};
} // namespace dpnp::kernels::sliding_window1d
