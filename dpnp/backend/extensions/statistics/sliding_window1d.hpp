//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"

#include "kernels/statistics/sliding_window1d.hpp"

namespace statistics::sliding_window1d
{
using dpctl::tensor::usm_ndarray;

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

    constexpr SizeT size_y() const { return _size; }

    SizeT size_x() const { return sbgroup.get_max_local_range()[0]; }

    SizeT total_size() const { return size_x() * size_y(); }

    ncT *ptr() { return data; }

    SizeT x() const { return sbgroup.get_local_linear_id(); }

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
        return load(data, [mask](auto &&) { return mask; }, default_v);
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

    T *store(T *const data) { return store(data, true); }
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

    void advance_left(const T &fill_value) { advance_left(1, fill_value); }

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

    T *begin() const { return data(); }

    T *end() const { return data() + size(); }

    SizeT size() const { return size_; }

    T *data() const { return data_; }

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

    T *padded_begin() const { return this->begin() - pad(); }

    SizeT pad() const { return pad_; }

protected:
    const SizeT pad_;
};

template <typename T, typename SizeT = size_t>
PaddedSpan<T, SizeT>
    make_padded_span(T *const data, const SizeT size, const SizeT offset)
{
    return PaddedSpan<T, SizeT>(data, size, offset);
}

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
    using SlidingWindow1dKernel =
        dpnp::kernels::sliding_window1d::SlidingWindow1dFunctor<
            WorkPI, PaddedSpan<const T, SizeT>, Span<const T, SizeT>, Op, Red,
            Span<T, SizeT>, RegistryData, RegistryWindow>;

    cgh.parallel_for<SlidingWindow1dKernel>(
        nd_range, SlidingWindow1dKernel(a, v, op, red, out));
}

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
    using SlidingWindow1dSmallKernel =
        dpnp::kernels::sliding_window1d::SlidingWindow1dSmallFunctor<
            WorkPI, PaddedSpan<const T, SizeT>, Span<const T, SizeT>, Op, Red,
            Span<T, SizeT>, RegistryData, RegistryWindow>;

    cgh.parallel_for<SlidingWindow1dSmallKernel>(
        nd_range, SlidingWindow1dSmallKernel(a, v, op, red, out));
}

void validate(const usm_ndarray &a,
              const usm_ndarray &v,
              const usm_ndarray &out,
              const size_t l_pad,
              const size_t r_pad);
} // namespace statistics::sliding_window1d
