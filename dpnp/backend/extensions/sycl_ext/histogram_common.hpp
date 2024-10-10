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

#include "utils/math_utils.hpp"
#include <complex>
#include <sycl/sycl.hpp>
#include <tuple>
#include <type_traits>

namespace dpctl
{
namespace tensor
{
class usm_ndarray;
}
} // namespace dpctl

using dpctl::tensor::usm_ndarray;

namespace histogram
{

template <typename N, typename D>
N CeilDiv(N n, D d)
{
    return (n + d - 1) / d;
}

template <typename N, typename D>
N Align(N n, D d)
{
    return CeilDiv(n, d) * d;
}

template <typename T, int Dims>
struct CachedData
{
    static constexpr bool const sync_after_init = true;
    using pointer_type = T *;

    using ncT = typename std::remove_const<T>::type;
    using LocalData = sycl::local_accessor<ncT, Dims>;

    CachedData(T *global_data, sycl::range<Dims> shape, sycl::handler &cgh)
    {
        this->global_data = global_data;
        local_data = LocalData(shape, cgh);
    }

    T *get_ptr() const
    {
        return &local_data[0];
    }

    template <int _Dims>
    void init(const sycl::nd_item<_Dims> &item) const
    {
        int32_t llid = item.get_local_linear_id();
        auto local_ptr = &local_data[0];
        int32_t size = local_data.size();
        auto group = item.get_group();
        int32_t local_size = group.get_local_linear_range();

        for (int32_t i = llid; i < size; i += local_size) {
            local_ptr[i] = global_data[i];
        }
    }

    size_t size() const
    {
        return local_data.size();
    }

private:
    LocalData local_data;
    T *global_data = nullptr;
};

template <typename T, int Dims>
struct UncachedData
{
    static constexpr bool const sync_after_init = false;
    using Shape = sycl::range<Dims>;
    using pointer_type = T *;

    UncachedData(T *global_data, const Shape &shape, sycl::handler &)
    {
        this->global_data = global_data;
        _shape = shape;
    }

    T *get_ptr() const
    {
        return global_data;
    }

    template <int _Dims>
    void init(const sycl::nd_item<_Dims> &) const
    {
    }

    size_t size() const
    {
        return _shape.size();
    }

private:
    T *global_data = nullptr;
    Shape _shape;
};

template <typename T>
struct HistLocalType
{
    using type = T;
};

template <>
struct HistLocalType<uint64_t>
{
    using type = uint32_t;
};

template <>
struct HistLocalType<int64_t>
{
    using type = int32_t;
};

template <typename T, sycl::memory_order Order, sycl::memory_scope Scope>
struct AtomicOp
{
    static void add(T &lhs, const T value)
    {
        sycl::atomic_ref<T, Order, Scope> lh(lhs);
        lh += value;
    }
};

template <typename T, sycl::memory_order Order, sycl::memory_scope Scope>
struct AtomicOp<std::complex<T>, Order, Scope>
{
    static void add(std::complex<T> &lhs, const std::complex<T> value)
    {
        T *_lhs = reinterpret_cast<T(&)[2]>(lhs);
        const T *_val = reinterpret_cast<const T(&)[2]>(value);
        sycl::atomic_ref<T, Order, Scope> lh0(_lhs[0]);
        lh0 += _val[0];
        sycl::atomic_ref<T, Order, Scope> lh1(_lhs[1]);
        lh1 += _val[1];
    }
};

template <typename T, typename localT = typename HistLocalType<T>::type>
struct HistWithLocalCopies
{
    static constexpr bool const sync_after_init = true;
    static constexpr bool const sync_before_finalize = true;

    using LocalHist = sycl::local_accessor<localT, 2>;

    HistWithLocalCopies(T *global_data,
                        size_t bins_count,
                        int32_t copies_count,
                        sycl::handler &cgh)
    {
        local_hist = LocalHist(sycl::range<2>(copies_count, bins_count), cgh);
        global_hist = global_data;
    }

    template <int _Dims>
    void init(const sycl::nd_item<_Dims> &item, localT val = 0) const
    {
        uint32_t llid = item.get_local_linear_id();
        auto *local_ptr = &local_hist[0][0];
        uint32_t size = local_hist.size();
        auto group = item.get_group();
        uint32_t local_size = group.get_local_linear_range();

        for (uint32_t i = llid; i < size; i += local_size) {
            local_ptr[i] = val;
        }
    }

    template <int _Dims>
    void add(const sycl::nd_item<_Dims> &item, int32_t bin, localT value) const
    {
        int32_t llid = item.get_local_linear_id();
        int32_t local_hist_count = local_hist.get_range().get(0);
        int32_t local_copy_id =
            local_hist_count == 1 ? 0 : llid % local_hist_count;

        AtomicOp<localT, sycl::memory_order::relaxed,
                 sycl::memory_scope::work_group>::add(local_hist[local_copy_id]
                                                                [bin],
                                                      value);
    }

    template <int _Dims>
    void finalize(const sycl::nd_item<_Dims> &item) const
    {
        int32_t llid = item.get_local_linear_id();
        int32_t bins_count = local_hist.get_range().get(1);
        int32_t local_hist_count = local_hist.get_range().get(0);
        auto group = item.get_group();
        int32_t local_size = group.get_local_linear_range();

        for (int32_t i = llid; i < bins_count; i += local_size) {
            auto value = local_hist[0][i];
            for (int32_t lhc = 1; lhc < local_hist_count; ++lhc) {
                value += local_hist[lhc][i];
            }
            if (value != T(0)) {
                AtomicOp<T, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>::add(global_hist[i],
                                                          value);
            }
        }
    }

    uint32_t size() const
    {
        return local_hist.size();
    }

private:
    LocalHist local_hist;
    T *global_hist = nullptr;
};

template <typename T>
struct HistGlobalMemory
{
    static constexpr bool const sync_after_init = false;
    static constexpr bool const sync_before_finalize = false;

    HistGlobalMemory(T *global_data)
    {
        global_hist = global_data;
    }

    template <int _Dims>
    void init(const sycl::nd_item<_Dims> &) const
    {
    }

    template <int _Dims>
    void add(const sycl::nd_item<_Dims> &, int32_t bin, T value) const
    {
        AtomicOp<T, sycl::memory_order::relaxed,
                 sycl::memory_scope::device>::add(global_hist[bin], value);
    }

    template <int _Dims>
    void finalize(const sycl::nd_item<_Dims> &) const
    {
    }

private:
    T *global_hist = nullptr;
};

template <typename T = uint32_t>
struct NoWeights
{
    constexpr T get(size_t) const
    {
        return 1;
    }
};

template <typename T>
struct Weights
{
    Weights(T *weights)
    {
        data = weights;
    }

    T get(size_t id) const
    {
        return data[id];
    }

private:
    T *data = nullptr;
};

template <typename T>
struct less
{
    bool operator()(const T &lhs, const T &rhs) const
    {
        return std::less{}(lhs, rhs);
    }
};

template <typename T>
struct less<std::complex<T>>
{
    bool operator()(const std::complex<T> &lhs,
                    const std::complex<T> &rhs) const
    {
        return dpctl::tensor::math_utils::less_complex(lhs, rhs);
    }
};

template <typename T>
struct IsNan
{
    static bool isnan(const T &v)
    {
        if constexpr (std::is_floating_point<T>::value) {
            return sycl::isnan(v);
        }

        return false;
    }
};

template <typename T>
struct IsNan<std::complex<T>>
{
    static bool isnan(const std::complex<T> &v)
    {
        T real1 = std::real(v);
        T imag1 = std::imag(v);
        return sycl::isnan(real1) || sycl::isnan(imag1);
    }
};

template <typename T, typename DataStorage>
struct Edges
{
    static constexpr bool const sync_after_init = DataStorage::sync_after_init;
    using boundsT = std::tuple<T, T>;

    Edges(const T *global_data, size_t size, sycl::handler &cgh)
        : data(global_data, sycl::range<1>(size), cgh)
    {
    }

    template <int _Dims>
    void init(const sycl::nd_item<_Dims> &item) const
    {
        data.init(item);
    }

    boundsT get_bounds() const
    {
        auto min = data.get_ptr()[0];
        auto max = data.get_ptr()[data.size() - 1];
        return {min, max};
    }

    template <int _Dims, typename dT>
    size_t get_bin(const sycl::nd_item<_Dims> &,
                   const dT *val,
                   const boundsT &) const
    {
        uint32_t edges_count = data.size();
        uint32_t bins_count = edges_count - 1;
        const auto *bins = data.get_ptr();

        uint32_t bin =
            std::upper_bound(bins, bins + edges_count, val[0], less<dT>{}) -
            bins - 1;
        bin = std::min(bin, bins_count - 1);

        return bin;
    }

    template <typename dT>
    bool in_bounds(const dT *val, const boundsT &bounds) const
    {
        less<dT> _less;
        return !_less(val[0], std::get<0>(bounds)) &&
               !_less(std::get<1>(bounds), val[0]) && !IsNan<dT>::isnan(val[0]);
    }

private:
    DataStorage data;
};

template <typename T>
using CachedEdges = Edges<T, CachedData<const T, 1>>;

template <typename T>
using UncachedEdges = Edges<T, UncachedData<const T, 1>>;

template <typename T, typename HistImpl, typename Edges, typename Weights>
class histogram_kernel;

template <typename T, typename HistImpl, typename Edges, typename Weights>
void submit_histogram(const T *in,
                      size_t size,
                      size_t dims,
                      uint32_t WorkPI,
                      const HistImpl &hist,
                      const Edges &edges,
                      const Weights &weights,
                      sycl::nd_range<1> nd_range,
                      sycl::handler &cgh)
{
    cgh.parallel_for<histogram_kernel<T, HistImpl, Edges, Weights>>(
        nd_range, [=](sycl::nd_item<1> item) {
            auto id = item.get_group_linear_id();
            auto lid = item.get_local_linear_id();
            auto group = item.get_group();
            auto local_size = item.get_local_range(0);

            hist.init(item);
            edges.init(item);

            if constexpr (HistImpl::sync_after_init || Edges::sync_after_init) {
                sycl::group_barrier(group, sycl::memory_scope::work_group);
            }

            auto bounds = edges.get_bounds();

            for (uint32_t i = 0; i < WorkPI; ++i) {
                auto data_idx = id * WorkPI * local_size + i * local_size + lid;
                if (data_idx < size) {
                    auto *d = &in[data_idx * dims];

                    if (edges.in_bounds(d, bounds)) {
                        auto bin = edges.get_bin(item, d, bounds);
                        auto weight = weights.get(data_idx);
                        hist.add(item, bin, weight);
                    }
                }
            }

            if constexpr (HistImpl::sync_before_finalize) {
                sycl::group_barrier(group, sycl::memory_scope::work_group);
            }

            hist.finalize(item);
        });
}

void validate(const usm_ndarray &sample,
              const usm_ndarray &bins,
              std::optional<const dpctl::tensor::usm_ndarray> &weights,
              const usm_ndarray &histogram);
} // namespace histogram
