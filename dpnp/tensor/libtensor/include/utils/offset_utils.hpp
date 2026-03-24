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
///
/// \file
/// This file defines Indexer callable operator to compute element offset in
/// an array addressed by gloabl_id.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "kernels/dpctl_tensor_types.hpp"
#include "utils/strided_iters.hpp"
#include "utils/sycl_alloc_utils.hpp"

namespace dpctl::tensor::offset_utils
{
namespace detail
{
struct sink_t
{
    sink_t(){};
    template <class T>
    sink_t(T &&){};
};

template <class V>
std::size_t __accumulate_size(std::size_t &s, V &&v)
{
    return s += v.size();
}

template <class V, class U>
sink_t __appender(V &lhs, U &&rhs)
{
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return {};
}

template <typename T, typename A, typename... Vs>
std::vector<T, A> concat(std::vector<T, A> lhs, Vs &&...vs)
{
    std::size_t s = lhs.size();
    {
        // limited scope ensures array is freed
        [[maybe_unused]] sink_t tmp[] = {__accumulate_size(s, vs)..., 0};
    }
    lhs.reserve(s);
    {
        // array of no-data objects ensures ordering of calls to the appender
        [[maybe_unused]] sink_t tmp[] = {
            __appender(lhs, std::forward<Vs>(vs))..., 0};
    }

    return std::move(lhs); // prevent return-value optimization
}
} // namespace detail

template <typename indT, typename... Vs>
std::tuple<std::unique_ptr<indT, dpctl::tensor::alloc_utils::USMDeleter>,
           std::size_t,
           sycl::event>
    device_allocate_and_pack(sycl::queue &q,
                             std::vector<sycl::event> &host_task_events,
                             Vs &&...vs)
{

    using dpctl::tensor::alloc_utils::usm_host_allocator;

    // memory transfer optimization, use USM-host for temporary speeds up
    // transfer to device, especially on dGPUs
    using usm_host_allocatorT = usm_host_allocator<indT>;
    using shT = std::vector<indT, usm_host_allocatorT>;

    usm_host_allocatorT usm_host_alloc(q);
    shT empty{0, usm_host_alloc};
    shT packed_shape_strides = detail::concat(std::move(empty), vs...);

    auto packed_shape_strides_owner =
        std::make_shared<shT>(std::move(packed_shape_strides));

    auto sz = packed_shape_strides_owner->size();
    auto shape_strides_owner =
        dpctl::tensor::alloc_utils::smart_malloc_device<indT>(sz, q);
    indT *shape_strides = shape_strides_owner.get();

    sycl::event copy_ev =
        q.copy<indT>(packed_shape_strides_owner->data(), shape_strides, sz);

    sycl::event cleanup_host_task_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(copy_ev);
        cgh.host_task([packed_shape_strides_owner =
                           std::move(packed_shape_strides_owner)] {
            // increment shared pointer ref-count to keep it alive
            // till copy operation completes;
        });
    });
    host_task_events.push_back(cleanup_host_task_ev);

    return std::make_tuple(std::move(shape_strides_owner), sz, copy_ev);
}

struct NoOpIndexer
{
    constexpr NoOpIndexer() {}
    constexpr std::size_t operator()(std::size_t gid) const
    {
        return gid;
    }
};

using dpctl::tensor::ssize_t;

/* @brief Indexer with shape and strides arrays of same size are packed */
struct StridedIndexer
{
    StridedIndexer(int _nd,
                   ssize_t _offset,
                   ssize_t const *_packed_shape_strides)
        : nd(_nd), starting_offset(_offset),
          shape_strides(_packed_shape_strides)
    {
    }

    ssize_t operator()(ssize_t gid) const
    {
        return compute_offset(gid);
    }

    ssize_t operator()(std::size_t gid) const
    {
        return compute_offset(static_cast<ssize_t>(gid));
    }

private:
    int nd;
    ssize_t starting_offset;
    ssize_t const *shape_strides;

    ssize_t compute_offset(ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        ssize_t relative_offset(0);
        _ind.get_displacement<const ssize_t *, const ssize_t *>(
            gid,
            shape_strides,      // shape ptr
            shape_strides + nd, // strides ptr
            relative_offset);
        return starting_offset + relative_offset;
    }
};

// ensure that indexer is device copyable
static_assert(sycl::is_device_copyable_v<StridedIndexer>);

/* @brief Indexer with shape, strides provided separately */
struct UnpackedStridedIndexer
{
    UnpackedStridedIndexer(int _nd,
                           ssize_t _offset,
                           ssize_t const *_shape,
                           ssize_t const *_strides)
        : nd(_nd), starting_offset(_offset), shape(_shape), strides(_strides)
    {
    }

    ssize_t operator()(ssize_t gid) const
    {
        return compute_offset(gid);
    }

    ssize_t operator()(std::size_t gid) const
    {
        return compute_offset(static_cast<ssize_t>(gid));
    }

private:
    int nd;
    ssize_t starting_offset;
    ssize_t const *shape;
    ssize_t const *strides;

    ssize_t compute_offset(ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        ssize_t relative_offset(0);
        _ind.get_displacement<const ssize_t *, const ssize_t *>(
            gid,
            shape,   // shape ptr
            strides, // strides ptr
            relative_offset);
        return starting_offset + relative_offset;
    }
};

// ensure that indexer is device copyable
static_assert(sycl::is_device_copyable_v<UnpackedStridedIndexer>);

struct Strided1DIndexer
{
    Strided1DIndexer(std::size_t _size) : offset{}, size(_size), step(1) {}
    Strided1DIndexer(ssize_t _size)
        : offset{}, size(static_cast<std::size_t>(_size)), step(1)
    {
    }
    Strided1DIndexer(std::size_t _size, ssize_t _step)
        : offset{}, size(_size), step(_step)
    {
    }
    Strided1DIndexer(std::size_t _size, std::size_t _step)
        : offset{}, size(_size), step(static_cast<ssize_t>(_step))
    {
    }
    Strided1DIndexer(ssize_t _size, ssize_t _step)
        : offset{}, size(static_cast<std::size_t>(_size)), step(_step)
    {
    }
    Strided1DIndexer(ssize_t _offset, std::size_t _size, ssize_t _step)
        : offset(_offset), size(_size), step(_step)
    {
    }
    Strided1DIndexer(ssize_t _offset, std::size_t _size, std::size_t _step)
        : offset(_offset), size(_size), step(static_cast<ssize_t>(_step))
    {
    }
    Strided1DIndexer(ssize_t _offset, ssize_t _size, ssize_t _step)
        : offset(_offset), size(static_cast<std::size_t>(_size)), step(_step)
    {
    }

    ssize_t operator()(std::size_t gid) const
    {
        // ensure 0 <= gid < size
        return offset + std::min<std::size_t>(gid, size - 1) * step;
    }

private:
    ssize_t offset = 0;
    std::size_t size = 1;
    ssize_t step = 1;
};

static_assert(sycl::is_device_copyable_v<Strided1DIndexer>);

struct Strided1DCyclicIndexer
{
    Strided1DCyclicIndexer(ssize_t _offset, ssize_t _size, ssize_t _step)
        : offset(_offset), size(static_cast<std::size_t>(_size)), step(_step)
    {
    }

    ssize_t operator()(std::size_t gid) const
    {
        return offset + (gid % size) * step;
    }

private:
    ssize_t offset = 0;
    std::size_t size = 1;
    ssize_t step = 1;
};

static_assert(sycl::is_device_copyable_v<Strided1DCyclicIndexer>);

template <typename displacementT>
struct TwoOffsets
{
    constexpr TwoOffsets() : first_offset(0), second_offset(0) {}
    constexpr TwoOffsets(const displacementT &first_offset_,
                         const displacementT &second_offset_)
        : first_offset(first_offset_), second_offset(second_offset_)
    {
    }

    constexpr displacementT get_first_offset() const
    {
        return first_offset;
    }
    constexpr displacementT get_second_offset() const
    {
        return second_offset;
    }

private:
    displacementT first_offset = 0;
    displacementT second_offset = 0;
};

struct TwoOffsets_StridedIndexer
{
    TwoOffsets_StridedIndexer(int common_nd,
                              ssize_t first_offset_,
                              ssize_t second_offset_,
                              ssize_t const *_packed_shape_strides)
        : nd(common_nd), starting_first_offset(first_offset_),
          starting_second_offset(second_offset_),
          shape_strides(_packed_shape_strides)
    {
    }

    TwoOffsets<ssize_t> operator()(ssize_t gid) const
    {
        return compute_offsets(gid);
    }

    TwoOffsets<ssize_t> operator()(std::size_t gid) const
    {
        return compute_offsets(static_cast<ssize_t>(gid));
    }

private:
    int nd;
    ssize_t starting_first_offset;
    ssize_t starting_second_offset;
    ssize_t const *shape_strides;

    TwoOffsets<ssize_t> compute_offsets(ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        ssize_t relative_first_offset(0);
        ssize_t relative_second_offset(0);
        _ind.get_displacement<const ssize_t *, const ssize_t *>(
            gid,
            shape_strides,          // shape ptr
            shape_strides + nd,     // strides ptr
            shape_strides + 2 * nd, // strides ptr
            relative_first_offset, relative_second_offset);
        return TwoOffsets<ssize_t>(
            starting_first_offset + relative_first_offset,
            starting_second_offset + relative_second_offset);
    }
};

struct TwoZeroOffsets_Indexer
{
    constexpr TwoZeroOffsets_Indexer() {}

    constexpr TwoOffsets<ssize_t> operator()(ssize_t) const
    {
        return TwoOffsets<ssize_t>();
    }
};

static_assert(sycl::is_device_copyable_v<TwoZeroOffsets_Indexer>);

template <typename FirstIndexerT, typename SecondIndexerT>
struct TwoOffsets_CombinedIndexer
{
private:
    FirstIndexerT first_indexer_;
    SecondIndexerT second_indexer_;

public:
    constexpr TwoOffsets_CombinedIndexer(const FirstIndexerT &first_indexer,
                                         const SecondIndexerT &second_indexer)
        : first_indexer_(first_indexer), second_indexer_(second_indexer)
    {
    }

    constexpr TwoOffsets<ssize_t> operator()(ssize_t gid) const
    {
        return TwoOffsets<ssize_t>(first_indexer_(gid), second_indexer_(gid));
    }
};

template <typename displacementT>
struct ThreeOffsets
{
    constexpr ThreeOffsets()
        : first_offset(0), second_offset(0), third_offset(0)
    {
    }
    constexpr ThreeOffsets(const displacementT &first_offset_,
                           const displacementT &second_offset_,
                           const displacementT &third_offset_)
        : first_offset(first_offset_), second_offset(second_offset_),
          third_offset(third_offset_)
    {
    }

    constexpr displacementT get_first_offset() const
    {
        return first_offset;
    }
    constexpr displacementT get_second_offset() const
    {
        return second_offset;
    }
    constexpr displacementT get_third_offset() const
    {
        return third_offset;
    }

private:
    displacementT first_offset = 0;
    displacementT second_offset = 0;
    displacementT third_offset = 0;
};

struct ThreeOffsets_StridedIndexer
{
    ThreeOffsets_StridedIndexer(int common_nd,
                                ssize_t first_offset_,
                                ssize_t second_offset_,
                                ssize_t third_offset_,
                                ssize_t const *_packed_shape_strides)
        : nd(common_nd), starting_first_offset(first_offset_),
          starting_second_offset(second_offset_),
          starting_third_offset(third_offset_),
          shape_strides(_packed_shape_strides)
    {
    }

    ThreeOffsets<ssize_t> operator()(ssize_t gid) const
    {
        return compute_offsets(gid);
    }

    ThreeOffsets<ssize_t> operator()(std::size_t gid) const
    {
        return compute_offsets(static_cast<ssize_t>(gid));
    }

private:
    int nd;
    ssize_t starting_first_offset;
    ssize_t starting_second_offset;
    ssize_t starting_third_offset;
    ssize_t const *shape_strides;

    ThreeOffsets<ssize_t> compute_offsets(ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        ssize_t relative_first_offset(0);
        ssize_t relative_second_offset(0);
        ssize_t relative_third_offset(0);
        _ind.get_displacement<const ssize_t *, const ssize_t *>(
            gid,
            shape_strides,          // shape ptr
            shape_strides + nd,     // strides ptr
            shape_strides + 2 * nd, // strides ptr
            shape_strides + 3 * nd, // strides ptr
            relative_first_offset, relative_second_offset,
            relative_third_offset);
        return ThreeOffsets<ssize_t>(
            starting_first_offset + relative_first_offset,
            starting_second_offset + relative_second_offset,
            starting_third_offset + relative_third_offset);
    }
};

static_assert(sycl::is_device_copyable_v<ThreeOffsets_StridedIndexer>);

struct ThreeZeroOffsets_Indexer
{
    constexpr ThreeZeroOffsets_Indexer() {}

    constexpr ThreeOffsets<ssize_t> operator()(ssize_t) const
    {
        return ThreeOffsets<ssize_t>();
    }

    constexpr ThreeOffsets<ssize_t> operator()(std::size_t) const
    {
        return ThreeOffsets<ssize_t>();
    }
};

static_assert(sycl::is_device_copyable_v<ThreeZeroOffsets_Indexer>);

template <typename FirstIndexerT,
          typename SecondIndexerT,
          typename ThirdIndexerT>
struct ThreeOffsets_CombinedIndexer
{
private:
    FirstIndexerT first_indexer_;
    SecondIndexerT second_indexer_;
    ThirdIndexerT third_indexer_;

public:
    constexpr ThreeOffsets_CombinedIndexer(const FirstIndexerT &first_indexer,
                                           const SecondIndexerT &second_indexer,
                                           const ThirdIndexerT &third_indexer)
        : first_indexer_(first_indexer), second_indexer_(second_indexer),
          third_indexer_(third_indexer)
    {
    }

    constexpr ThreeOffsets<ssize_t> operator()(ssize_t gid) const
    {
        return ThreeOffsets<ssize_t>(first_indexer_(gid), second_indexer_(gid),
                                     third_indexer_(gid));
    }
};

template <typename displacementT>
struct FourOffsets
{
    constexpr FourOffsets()
        : first_offset(0), second_offset(0), third_offset(0), fourth_offset(0)
    {
    }
    constexpr FourOffsets(const displacementT &first_offset_,
                          const displacementT &second_offset_,
                          const displacementT &third_offset_,
                          const displacementT &fourth_offset_)
        : first_offset(first_offset_), second_offset(second_offset_),
          third_offset(third_offset_), fourth_offset(fourth_offset_)
    {
    }

    constexpr displacementT get_first_offset() const
    {
        return first_offset;
    }
    constexpr displacementT get_second_offset() const
    {
        return second_offset;
    }
    constexpr displacementT get_third_offset() const
    {
        return third_offset;
    }
    constexpr displacementT get_fourth_offset() const
    {
        return fourth_offset;
    }

private:
    displacementT first_offset = 0;
    displacementT second_offset = 0;
    displacementT third_offset = 0;
    displacementT fourth_offset = 0;
};

struct FourOffsets_StridedIndexer
{
    constexpr FourOffsets_StridedIndexer(int common_nd,
                                         ssize_t first_offset_,
                                         ssize_t second_offset_,
                                         ssize_t third_offset_,
                                         ssize_t fourth_offset_,
                                         ssize_t const *_packed_shape_strides)
        : nd(common_nd), starting_first_offset(first_offset_),
          starting_second_offset(second_offset_),
          starting_third_offset(third_offset_),
          starting_fourth_offset(fourth_offset_),
          shape_strides(_packed_shape_strides)
    {
    }

    constexpr FourOffsets<ssize_t> operator()(ssize_t gid) const
    {
        return compute_offsets(gid);
    }

    constexpr FourOffsets<ssize_t> operator()(std::size_t gid) const
    {
        return compute_offsets(static_cast<ssize_t>(gid));
    }

private:
    int nd;
    ssize_t starting_first_offset;
    ssize_t starting_second_offset;
    ssize_t starting_third_offset;
    ssize_t starting_fourth_offset;
    ssize_t const *shape_strides;

    FourOffsets<ssize_t> compute_offsets(ssize_t gid) const
    {
        using dpctl::tensor::strides::CIndexer_vector;

        CIndexer_vector _ind(nd);
        ssize_t relative_first_offset(0);
        ssize_t relative_second_offset(0);
        ssize_t relative_third_offset(0);
        ssize_t relative_fourth_offset(0);
        _ind.get_displacement<const ssize_t *, const ssize_t *>(
            gid,
            shape_strides,          // shape ptr
            shape_strides + nd,     // strides ptr
            shape_strides + 2 * nd, // strides ptr
            shape_strides + 3 * nd, // strides ptr
            shape_strides + 4 * nd, // strides ptr
            relative_first_offset, relative_second_offset,
            relative_third_offset, relative_fourth_offset);
        return FourOffsets<ssize_t>(
            starting_first_offset + relative_first_offset,
            starting_second_offset + relative_second_offset,
            starting_third_offset + relative_third_offset,
            starting_fourth_offset + relative_fourth_offset);
    }
};

static_assert(sycl::is_device_copyable_v<FourOffsets_StridedIndexer>);

struct FourZeroOffsets_Indexer
{
    constexpr FourZeroOffsets_Indexer() {}

    constexpr FourOffsets<ssize_t> operator()(ssize_t) const
    {
        return FourOffsets<ssize_t>();
    }
};

static_assert(sycl::is_device_copyable_v<FourZeroOffsets_Indexer>);

struct NthStrideOffset
{
    NthStrideOffset(int common_nd,
                    ssize_t const *_offsets,
                    ssize_t const *_packed_shape_strides)
        : _ind(common_nd), nd(common_nd), offsets(_offsets),
          shape_strides(_packed_shape_strides)
    {
    }

    std::size_t operator()(ssize_t gid, int n) const
    {
        ssize_t relative_offset(0);
        _ind.get_displacement<const ssize_t *, const ssize_t *>(
            gid, shape_strides, shape_strides + ((n + 1) * nd),
            relative_offset);

        return relative_offset + offsets[n];
    }

private:
    dpctl::tensor::strides::CIndexer_vector<ssize_t> _ind;

    int nd;
    ssize_t const *offsets;
    ssize_t const *shape_strides;
};

static_assert(sycl::is_device_copyable_v<NthStrideOffset>);

template <int nd>
struct FixedDimStridedIndexer
{
    FixedDimStridedIndexer(const std::array<ssize_t, nd> &_shape,
                           const std::array<ssize_t, nd> &_strides,
                           ssize_t _offset)
        : _ind(_shape), strides(_strides), starting_offset(_offset)
    {
    }
    std::size_t operator()(std::size_t gid) const
    {
        dpctl::tensor::strides::CIndexer_array<nd, ssize_t> local_indexer(
            std::move(_ind));
        local_indexer.set(gid);
        auto mi = local_indexer.get();

        ssize_t relative_offset = 0;

#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset += mi[i] * strides[i];
        }
        return starting_offset + relative_offset;
    }

private:
    dpctl::tensor::strides::CIndexer_array<nd, ssize_t> _ind;

    std::array<ssize_t, nd> strides;
    ssize_t starting_offset;
};

static_assert(sycl::is_device_copyable_v<FixedDimStridedIndexer<1>>);

template <int nd>
struct TwoOffsets_FixedDimStridedIndexer
{
    TwoOffsets_FixedDimStridedIndexer(const std::array<ssize_t, nd> &_shape,
                                      const std::array<ssize_t, nd> &_strides1,
                                      const std::array<ssize_t, nd> &_strides2,
                                      ssize_t _offset1,
                                      ssize_t _offset2)
        : _ind(_shape), strides1(_strides1), strides2(_strides2),
          starting_offset1(_offset1), starting_offset2(_offset2)
    {
    }

    TwoOffsets<ssize_t> operator()(std::size_t gid) const
    {
        dpctl::tensor::strides::CIndexer_array<nd, ssize_t> local_indexer(
            std::move(_ind));
        local_indexer.set(gid);
        auto mi = local_indexer.get();

        ssize_t relative_offset1 = 0;
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset1 += mi[i] * strides1[i];
        }

        ssize_t relative_offset2 = 0;
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset2 += mi[i] * strides2[i];
        }

        return TwoOffsets<ssize_t>(starting_offset1 + relative_offset1,
                                   starting_offset2 + relative_offset2);
    }

private:
    dpctl::tensor::strides::CIndexer_array<nd, ssize_t> _ind;

    std::array<ssize_t, nd> strides1;
    std::array<ssize_t, nd> strides2;
    ssize_t starting_offset1;
    ssize_t starting_offset2;
};

static_assert(sycl::is_device_copyable_v<TwoOffsets_FixedDimStridedIndexer<1>>);

template <int nd>
struct ThreeOffsets_FixedDimStridedIndexer
{
    ThreeOffsets_FixedDimStridedIndexer(
        const std::array<ssize_t, nd> &_shape,
        const std::array<ssize_t, nd> &_strides1,
        const std::array<ssize_t, nd> &_strides2,
        const std::array<ssize_t, nd> &_strides3,
        ssize_t _offset1,
        ssize_t _offset2,
        ssize_t _offset3)
        : _ind(_shape), strides1(_strides1), strides2(_strides2),
          strides3(_strides3), starting_offset1(_offset1),
          starting_offset2(_offset2), starting_offset3(_offset3)
    {
    }

    ThreeOffsets<ssize_t> operator()(std::size_t gid) const
    {
        dpctl::tensor::strides::CIndexer_array<nd, ssize_t> local_indexer(
            std::move(_ind));
        local_indexer.set(gid);
        auto mi = local_indexer.get();

        ssize_t relative_offset1 = 0;
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset1 += mi[i] * strides1[i];
        }

        ssize_t relative_offset2 = 0;
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset2 += mi[i] * strides2[i];
        }

        ssize_t relative_offset3 = 0;
#pragma unroll
        for (int i = 0; i < nd; ++i) {
            relative_offset3 += mi[i] * strides3[i];
        }

        return ThreeOffsets<ssize_t>(starting_offset1 + relative_offset1,
                                     starting_offset2 + relative_offset2,
                                     starting_offset3 + relative_offset3);
    }

private:
    dpctl::tensor::strides::CIndexer_array<nd, ssize_t> _ind;

    std::array<ssize_t, nd> strides1;
    std::array<ssize_t, nd> strides2;
    std::array<ssize_t, nd> strides3;
    ssize_t starting_offset1;
    ssize_t starting_offset2;
    ssize_t starting_offset3;
};

static_assert(
    sycl::is_device_copyable_v<ThreeOffsets_FixedDimStridedIndexer<1>>);
} // namespace dpctl::tensor::offset_utils
