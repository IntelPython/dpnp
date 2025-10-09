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

#include "utils/math_utils.hpp"
#include <sycl/sycl.hpp>
#include <type_traits>

#include <stdio.h>

#include "ext/common.hpp"

using dpctl::tensor::usm_ndarray;

using ext::common::Align;
using ext::common::CeilDiv;

namespace statistics::sliding_window1d
{

template <typename T, uint32_t Size>
class _RegistryDataStorage
{
public:
    using ncT = typename std::remove_const_t<T>;
    using SizeT = decltype(Size);
    static constexpr SizeT _size = Size;

    _RegistryDataStorage(const sycl::nd_item<1> &item)
        : sbgroup(item.get_sub_group())
    {
    }

    template <typename yT>
    T &operator[](const yT &idx)
    {
        static_assert(std::is_integral_v<yT>,
                      "idx must be of an integral type");
        return data[idx];
    }

    template <typename yT>
    const T &operator[](const yT &idx) const
    {
        static_assert(std::is_integral_v<yT>,
                      "idx must be of an integral type");
        return data[idx];
    }

    T &value()
    {
        static_assert(Size == 1,
                      "Size is not equal to 1. Use value(idx) instead");
        return data[0];
    }

    const T &value() const
    {
        static_assert(Size == 1,
                      "Size is not equal to 1. Use value(idx) instead");
        return data[0];
    }

    template <typename yT, typename xT>
    T broadcast(const yT &y, const xT &x) const
    {
        static_assert(std::is_integral_v<std::remove_reference_t<yT>>,
                      "y must be of an integral type");
        static_assert(std::is_integral_v<std::remove_reference_t<xT>>,
                      "x must be of an integral type");

        return sycl::select_from_group(sbgroup, data[y], x);
    }

    template <typename iT>
    T broadcast(const iT &idx) const
    {
        if constexpr (Size == 1) {
            return broadcast(0, idx);
        }
        else {
            return broadcast(idx / size_x(), idx % size_x());
        }
    }

    template <typename yT, typename xT>
    T shift_left(const yT &y, const xT &x) const
    {
        static_assert(std::is_integral_v<yT>, "y must be of an integral type");
        static_assert(std::is_integral_v<xT>, "x must be of an integral type");

        return sycl::shift_group_left(sbgroup, data[y], x);
    }

    template <typename yT, typename xT>
    T shift_right(const yT &y, const xT &x) const
    {
        static_assert(std::is_integral_v<yT>, "y must be of an integral type");
        static_assert(std::is_integral_v<xT>, "x must be of an integral type");

        return sycl::shift_group_right(sbgroup, data[y], x);
    }

    constexpr SizeT size_y() const
    {
        return _size;
    }

    SizeT size_x() const
    {
        return sbgroup.get_max_local_range()[0];
    }

    SizeT total_size() const
    {
        return size_x() * size_y();
    }

    ncT *ptr()
    {
        return data;
    }

    SizeT x() const
    {
        return sbgroup.get_local_linear_id();
    }

protected:
    const sycl::sub_group sbgroup;
    ncT data[Size];
};

template <typename T, uint32_t Size = 1>
struct RegistryData : public _RegistryDataStorage<T, Size>
{
    using SizeT = typename _RegistryDataStorage<T, Size>::SizeT;

    using _RegistryDataStorage<T, Size>::_RegistryDataStorage;

    template <typename LaneIdT,
              typename Condition,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<bool, Condition, SizeT>>>
    void fill_lane(const LaneIdT &lane_id, const T &value, Condition &&mask)
    {
        static_assert(std::is_integral_v<LaneIdT>,
                      "lane_id must be of an integral type");
        if (mask(this->x())) {
            this->data[lane_id] = value;
        }
    }

    template <typename LaneIdT>
    void fill_lane(const LaneIdT &lane_id, const T &value, const bool &mask)
    {
        fill_lane(lane_id, value, [mask](auto &&) { return mask; });
    }

    template <typename LaneIdT>
    void fill_lane(const LaneIdT &lane_id, const T &value)
    {
        fill_lane(lane_id, value, true);
    }

    template <typename Condition,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<bool, Condition, SizeT, SizeT>>>
    void fill(const T &value, Condition &&mask)
    {
        for (SizeT i = 0; i < Size; ++i) {
            fill_lane(i, value, mask(i, this->x()));
        }
    }

    void fill(const T &value)
    {
        fill(value, [](auto &&, auto &&) { return true; });
    }

    template <typename LaneIdT,
              typename Condition,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<bool, Condition, const T *const>>>
    T *load_lane(const LaneIdT &lane_id,
                 const T *const data,
                 Condition &&mask,
                 const T &default_v)
    {
        static_assert(std::is_integral_v<LaneIdT>,
                      "lane_id must be of an integral type");
        this->data[lane_id] = mask(data) ? data[0] : default_v;

        return data + this->size_x();
    }

    template <typename LaneIdT>
    T *load_lane(const LaneIdT &laned_id,
                 const T *const data,
                 const bool &mask,
                 const T &default_v)
    {
        return load_lane(
            laned_id, data, [mask](auto &&) { return mask; }, default_v);
    }

    template <typename LaneIdT>
    T *load_lane(const LaneIdT &laned_id, const T *const data)
    {
        constexpr T default_v = 0;
        return load_lane(laned_id, data, true, default_v);
    }

    template <typename yStrideT,
              typename Condition,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<bool, Condition, const T *const>>>
    T *load(const T *const data,
            const yStrideT &y_stride,
            Condition &&mask,
            const T &default_v)
    {
        auto *it = data;
        for (SizeT i = 0; i < Size; ++i) {
            load_lane(i, it, mask, default_v);
            it += y_stride;
        }

        return it;
    }

    template <typename yStrideT>
    T *load(const T *const data,
            const yStrideT &y_stride,
            const bool &mask,
            const T &default_v)
    {
        return load(
            data, y_stride, [mask](auto &&) { return mask; }, default_v);
    }

    template <typename Condition,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<bool, Condition, const T *const>>>
    T *load(const T *const data, Condition &&mask, const T &default_v)
    {
        return load(data, this->size_x(), mask, default_v);
    }

    T *load(const T *const data, const bool &mask, const T &default_v)
    {
        return load(
            data, [mask](auto &&) { return mask; }, default_v);
    }

    T *load(const T *const data)
    {
        constexpr T default_v = 0;
        return load(data, true, default_v);
    }

    template <typename LaneIdT,
              typename Condition,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<bool, Condition, const T *const>>>
    T *store_lane(const LaneIdT &lane_id, T *const data, Condition &&mask)
    {
        static_assert(std::is_integral_v<LaneIdT>,
                      "lane_id must be of an integral type");

        if (mask(data)) {
            data[0] = this->data[lane_id];
        }

        return data + this->size_x();
    }

    template <typename LaneIdT>
    T *store_lane(const LaneIdT &lane_id, T *const data, const bool &mask)
    {
        return store_lane(lane_id, data, [mask](auto &&) { return mask; });
    }

    template <typename LaneIdT>
    T *store_lane(const LaneIdT &lane_id, T *const data)
    {
        return store_lane(lane_id, data, true);
    }

    template <typename yStrideT,
              typename Condition,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<bool, Condition, const T *const>>>
    T *store(T *const data, const yStrideT &y_stride, Condition &&condition)
    {
        auto *it = data;
        for (SizeT i = 0; i < Size; ++i) {
            store_lane(i, it, condition);
            it += y_stride;
        }

        return it;
    }

    template <typename yStrideT>
    T *store(T *const data, const yStrideT &y_stride, const bool &mask)
    {
        return store(data, y_stride, [mask](auto &&) { return mask; });
    }

    template <typename Condition,
              typename = std::enable_if_t<
                  std::is_invocable_r_v<bool, Condition, const T *const>>>
    T *store(T *const data, Condition &&condition)
    {
        return store(data, this->size_x(), condition);
    }

    T *store(T *const data, const bool &mask)
    {
        return store(data, [mask](auto &&) { return mask; });
    }

    T *store(T *const data)
    {
        return store(data, true);
    }
};

template <typename T, uint32_t Size>
struct RegistryWindow : public RegistryData<T, Size>
{
    using SizeT = typename RegistryData<T, Size>::SizeT;

    using RegistryData<T, Size>::RegistryData;

    template <typename shT>
    void advance_left(const shT &shift, const T &fill_value)
    {
        static_assert(std::is_integral_v<shT>,
                      "shift must be of an integral type");

        uint32_t shift_r = this->size_x() - shift;
        for (SizeT i = 0; i < Size; ++i) {
            this->data[i] = this->shift_left(i, shift);
            auto border =
                i < Size - 1 ? this->shift_right(i + 1, shift_r) : fill_value;
            if (this->x() >= shift_r) {
                this->data[i] = border;
            }
        }
    }

    void advance_left(const T &fill_value)
    {
        advance_left(1, fill_value);
    }

    void advance_left()
    {
        constexpr T fill_value = 0;
        advance_left(fill_value);
    }
};

template <typename T, typename SizeT = size_t>
class Span
{
public:
    using value_type = T;
    using size_type = SizeT;

    Span(T *const data, const SizeT size) : data_(data), size_(size) {}

    T *begin() const
    {
        return data();
    }

    T *end() const
    {
        return data() + size();
    }

    SizeT size() const
    {
        return size_;
    }

    T *data() const
    {
        return data_;
    }

protected:
    T *const data_;
    const SizeT size_;
};

template <typename T, typename SizeT = size_t>
Span<T, SizeT> make_span(T *const data, const SizeT size)
{
    return Span<T, SizeT>(data, size);
}

template <typename T, typename SizeT = size_t>
class PaddedSpan : public Span<T, SizeT>
{
public:
    using value_type = T;
    using size_type = SizeT;

    PaddedSpan(T *const data, const SizeT size, const SizeT pad)
        : Span<T, SizeT>(data, size), pad_(pad)
    {
    }

    T *padded_begin() const
    {
        return this->begin() - pad();
    }

    SizeT pad() const
    {
        return pad_;
    }

protected:
    const SizeT pad_;
};

template <typename T, typename SizeT = size_t>
PaddedSpan<T, SizeT>
    make_padded_span(T *const data, const SizeT size, const SizeT offset)
{
    return PaddedSpan<T, SizeT>(data, size, offset);
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

template <uint32_t WorkPI,
          typename T,
          typename SizeT,
          typename Op,
          typename Red>
class sliding_window1d_kernel;

template <uint32_t WorkPI,
          typename T,
          typename SizeT,
          typename Op,
          typename Red>
void submit_sliding_window1d(const PaddedSpan<const T, SizeT> &a,
                             const Span<const T, SizeT> &v,
                             const Op &op,
                             const Red &red,
                             Span<T, SizeT> &out,
                             sycl::nd_range<1> nd_range,
                             sycl::handler &cgh)
{
    cgh.parallel_for<sliding_window1d_kernel<WorkPI, T, SizeT, Op, Red>>(
        nd_range, [=](sycl::nd_item<1> item) {
            auto glid = get_global_linear_id<SizeT>(WorkPI, item);

            auto results = RegistryData<T, WorkPI>(item);
            results.fill(0);

            auto results_num = get_results_num(WorkPI, out.size(), glid, item);

            const auto *a_begin = a.begin();
            const auto *a_end = a.end();

            auto sbgroup = item.get_sub_group();

            const auto chunks_count =
                CeilDiv(v.size(), sbgroup.get_max_local_range()[0]);

            const auto *a_ptr = &a.padded_begin()[glid];

            auto _a_load_cond = [a_begin, a_end](auto &&ptr) {
                return ptr >= a_begin && ptr < a_end;
            };

            auto a_data = RegistryWindow<const T, WorkPI + 1>(item);
            a_ptr = a_data.load(a_ptr, _a_load_cond, 0);

            const auto *v_ptr = &v.begin()[sbgroup.get_local_linear_id()];
            auto v_size = v.size();

            for (uint32_t b = 0; b < chunks_count; ++b) {
                auto v_data = RegistryData<const T>(item);
                v_ptr = v_data.load(v_ptr, v_data.x() < v_size, 0);

                uint32_t chunk_size_ =
                    std::min(v_size, SizeT(v_data.total_size()));
                process_block(results, results_num, a_data, v_data, chunk_size_,
                              op, red);

                if (b != chunks_count - 1) {
                    a_ptr = a_data.load_lane(a_data.size_y() - 1, a_ptr,
                                             _a_load_cond, 0);
                    v_size -= v_data.total_size();
                }
            }

            auto *const out_ptr = out.begin();
            // auto *const out_end = out.end();

            auto y_start = glid;
            auto y_stop =
                std::min(y_start + WorkPI * results.size_x(), out.size());
            uint32_t i = 0;
            for (uint32_t y = y_start; y < y_stop; y += results.size_x()) {
                out_ptr[y] = results[i++];
            }
            // while the code itself seems to be valid, inside correlate
            // kernel it results in memory corruption. Further investigation
            // is needed. SAT-7693
            // corruption results.store(&out_ptr[glid],
            //               [out_end](auto &&ptr) { return ptr < out_end; });
        });
}

template <uint32_t WorkPI,
          typename T,
          typename SizeT,
          typename Op,
          typename Red>
class sliding_window1d_small_kernel;

template <uint32_t WorkPI,
          typename T,
          typename SizeT,
          typename Op,
          typename Red>
void submit_sliding_window1d_small_kernel(const PaddedSpan<const T, SizeT> &a,
                                          const Span<const T, SizeT> &v,
                                          const Op &op,
                                          const Red &red,
                                          Span<T, SizeT> &out,
                                          sycl::nd_range<1> nd_range,
                                          sycl::handler &cgh)
{
    cgh.parallel_for<sliding_window1d_small_kernel<WorkPI, T, SizeT, Op, Red>>(
        nd_range, [=](sycl::nd_item<1> item) {
            auto glid = get_global_linear_id<SizeT>(WorkPI, item);

            auto results = RegistryData<T, WorkPI>(item);
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

            auto a_data = RegistryWindow<const T, WorkPI + 1>(item);
            a_data.load(a_ptr, _a_load_cond, 0);

            const auto *v_ptr = &v.begin()[sbgroup.get_local_linear_id()];
            auto v_size = v.size();

            auto v_data = RegistryData<const T>(item);
            v_ptr = v_data.load(v_ptr, v_data.x() < v_size, 0);

            auto results_num = get_results_num(WorkPI, out.size(), glid, item);

            process_block(results, results_num, a_data, v_data, v_size, op,
                          red);

            auto *const out_ptr = out.begin();
            // auto *const out_end = out.end();

            auto y_start = glid;
            auto y_stop =
                std::min(y_start + WorkPI * results.size_x(), out.size());
            uint32_t i = 0;
            for (uint32_t y = y_start; y < y_stop; y += results.size_x()) {
                out_ptr[y] = results[i++];
            }
            // while the code itself seems to be valid, inside correlate
            // kernel it results in memory corruption. Further investigation
            // is needed. SAT-7693
            // corruption results.store(&out_ptr[glid],
            //               [out_end](auto &&ptr) { return ptr < out_end; });
        });
}

void validate(const usm_ndarray &a,
              const usm_ndarray &v,
              const usm_ndarray &out,
              const size_t l_pad,
              const size_t r_pad);
} // namespace statistics::sliding_window1d
