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
/// This file defines utilities used for kernel submission.
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "math_utils.hpp"

namespace dpctl::tensor::sycl_utils
{
namespace detail
{
template <typename...>
struct TypeList;

template <typename Head, typename... Tail>
struct TypeList<Head, Tail...>
{
    using head = Head;
    using tail = TypeList<Tail...>;
};

using NullTypeList = TypeList<>;
template <typename T>
struct IsNullTypeList : std::conditional_t<std::is_same_v<T, NullTypeList>,
                                           std::true_type,
                                           std::false_type>
{
};

// recursively check if type is contained in given TypeList
template <typename T, typename TList>
struct IsContained
    : std::conditional_t<
          std::is_same_v<typename TList::head, std::remove_cv_t<T>>,
          std::true_type,
          IsContained<T, typename TList::tail>>
{
};

template <>
struct TypeList<>
{
};

// std::false_type when last case has been checked for membership
template <typename T>
struct IsContained<T, NullTypeList> : std::false_type
{
};

template <class T>
struct IsComplex : std::false_type
{
};
template <class T>
struct IsComplex<std::complex<T>> : std::true_type
{
};
} // namespace detail

template <typename T>
using sycl_ops = detail::TypeList<sycl::plus<T>,
                                  sycl::bit_or<T>,
                                  sycl::bit_xor<T>,
                                  sycl::bit_and<T>,
                                  sycl::maximum<T>,
                                  sycl::minimum<T>,
                                  sycl::multiplies<T>>;

template <typename T, typename Op>
struct IsSyclOp
{
    static constexpr bool value =
        detail::IsContained<Op, sycl_ops<std::remove_const_t<T>>>::value ||
        detail::IsContained<Op, sycl_ops<std::add_const_t<T>>>::value;
};

/*! @brief Find the smallest multiple of supported sub-group size larger than
 * nelems */
template <std::size_t f = 4>
std::size_t choose_workgroup_size(const std::size_t nelems,
                                  const std::vector<std::size_t> &sg_sizes)
{
    std::vector<std::size_t> wg_choices;
    wg_choices.reserve(f * sg_sizes.size());

    for (const auto &sg_size : sg_sizes) {
#pragma unroll
        for (std::size_t i = 1; i <= f; ++i) {
            wg_choices.push_back(sg_size * i);
        }
    }
    std::sort(std::begin(wg_choices), std::end(wg_choices));

    std::size_t wg = 1;
    for (std::size_t i = 0; i < wg_choices.size(); ++i) {
        if (wg_choices[i] == wg) {
            continue;
        }
        wg = wg_choices[i];
        std::size_t n_groups = ((nelems + wg - 1) / wg);
        if (n_groups == 1)
            break;
    }

    return wg;
}

namespace detail
{

template <typename LocAccT, typename OpT>
void _fold(LocAccT &local_mem_acc,
           const std::uint32_t lid,
           const std::uint32_t cutoff,
           const std::uint32_t step,
           const OpT &op)
{
    if (lid < cutoff) {
        local_mem_acc[lid] = op(local_mem_acc[lid], local_mem_acc[step + lid]);
    }
}

template <typename LocAccT, typename OpT>
void _fold(LocAccT &local_mem_acc,
           const std::uint32_t lid,
           const std::uint32_t step,
           const OpT &op)
{
    if (lid < step) {
        local_mem_acc[lid] = op(local_mem_acc[lid], local_mem_acc[step + lid]);
    }
}

} // end of namespace detail

template <typename T, typename GroupT, typename LocAccT, typename OpT>
T custom_reduce_over_group(const GroupT &wg,
                           LocAccT local_mem_acc,
                           const T &local_val,
                           const OpT &op)
{
    // value experimentally tuned to achieve best runtime on Iris Xe,
    // Arc A140V integrated Intel GPUs, and discrete Intel Max GPU.
    static constexpr std::uint32_t low_sz = 8u;
    // maximal work-group size
    static constexpr std::uint32_t high_sz = 1024u;
    const std::uint32_t wgs = wg.get_local_linear_range();
    const std::uint32_t lid = wg.get_local_linear_id();

    local_mem_acc[lid] = local_val;
    sycl::group_barrier(wg, sycl::memory_scope::work_group);

    std::uint32_t n_witems = wgs;
    if (wgs & (wgs - 1)) {
        // wgs is not a power of 2
#pragma unroll
        for (std::uint32_t sz = high_sz; sz >= low_sz; sz >>= 1) {
            if (n_witems >= sz) {
                const std::uint32_t n_witems_ = (n_witems + 1) >> 1;
                detail::_fold(local_mem_acc, lid, n_witems - n_witems_,
                              n_witems_, op);
                sycl::group_barrier(wg, sycl::memory_scope::work_group);
                n_witems = n_witems_;
            }
        }
    }
    else {
        // wgs is a power of 2
#pragma unroll
        for (std::uint32_t sz = high_sz; sz >= low_sz; sz >>= 1) {
            if (n_witems >= sz) {
                n_witems >>= 1;
                detail::_fold(local_mem_acc, lid, n_witems, op);
                sycl::group_barrier(wg, sycl::memory_scope::work_group);
            }
        }
    }

    T red_val_over_wg = local_mem_acc[0];
    if (wg.leader()) {
        for (std::uint32_t i = 1; i < n_witems; ++i) {
            red_val_over_wg = op(red_val_over_wg, local_mem_acc[i]);
        }
    }

    return sycl::group_broadcast(wg, red_val_over_wg, 0);
}

template <typename GroupT,
          typename SubGroupT,
          typename LocAccT,
          typename T,
          typename OpT>
T custom_inclusive_scan_over_group(GroupT &&wg,
                                   SubGroupT &&sg,
                                   LocAccT &&local_mem_acc,
                                   const T &local_val,
                                   const T &identity,
                                   OpT &&op)
{
    const std::uint32_t local_id = wg.get_local_id(0);
    const std::uint32_t wgs = wg.get_local_range(0);

    const std::uint32_t lane_id = sg.get_local_id()[0];
    const std::uint32_t sgSize = sg.get_local_range()[0];

    T scan_val = local_val;
    for (std::uint32_t step = 1; step < sgSize; step *= 2) {
        const bool advanced_lane = (lane_id >= step);
        const std::uint32_t src_lane_id =
            (advanced_lane ? lane_id - step : lane_id);
        const T modifier = sycl::select_from_group(sg, scan_val, src_lane_id);
        if (advanced_lane) {
            scan_val = op(scan_val, modifier);
        }
    }

    local_mem_acc[local_id] = scan_val;
    sycl::group_barrier(wg, sycl::memory_scope::work_group);

    const std::uint32_t max_sgSize = sg.get_max_local_range()[0];
    const std::uint32_t sgr_id = sg.get_group_id()[0];

    // now scan
    const std::uint32_t n_aggregates = 1 + ((wgs - 1) / max_sgSize);
    const bool large_wg = (n_aggregates > max_sgSize);
    if (large_wg) {
        if (wg.leader()) {
            T _scan_val = identity;
            for (std::uint32_t i = 1; i <= n_aggregates - max_sgSize; ++i) {
                _scan_val = op(local_mem_acc[i * max_sgSize - 1], _scan_val);
                local_mem_acc[i * max_sgSize - 1] = _scan_val;
            }
        }
        sycl::group_barrier(wg, sycl::memory_scope::work_group);
    }

    if (sgr_id == 0) {
        const std::uint32_t offset =
            (large_wg) ? n_aggregates - max_sgSize : 0u;
        const bool in_range = (lane_id < n_aggregates);
        const bool in_bounds = in_range && (lane_id > 0 || large_wg);

        T __scan_val = (in_bounds)
                           ? local_mem_acc[(offset + lane_id) * max_sgSize - 1]
                           : identity;
        for (std::uint32_t step = 1; step < sgSize; step *= 2) {
            const bool advanced_lane = (lane_id >= step);
            const std::uint32_t src_lane_id =
                (advanced_lane ? lane_id - step : lane_id);
            const T modifier =
                sycl::select_from_group(sg, __scan_val, src_lane_id);
            if (advanced_lane && in_range) {
                __scan_val = op(__scan_val, modifier);
            }
        }
        if (in_bounds) {
            local_mem_acc[(offset + lane_id) * max_sgSize - 1] = __scan_val;
        }
    }
    sycl::group_barrier(wg, sycl::memory_scope::work_group);

    if (sgr_id > 0) {
        const T modifier = local_mem_acc[sgr_id * max_sgSize - 1];
        scan_val = op(scan_val, modifier);
    }

    // ensure all work-items finished reading from SLM
    sycl::group_barrier(wg, sycl::memory_scope::work_group);

    return scan_val;
}

// Reduction functors

// Maximum

template <typename T>
struct Maximum
{
    T operator()(const T &x, const T &y) const
    {
        if constexpr (detail::IsComplex<T>::value) {
            using dpctl::tensor::math_utils::max_complex;
            return max_complex<T>(x, y);
        }
        else if constexpr (std::is_floating_point_v<T> ||
                           std::is_same_v<T, sycl::half>) {
            return (std::isnan(x) || x > y) ? x : y;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return x || y;
        }
        else {
            return (x > y) ? x : y;
        }
    }
};

// Minimum

template <typename T>
struct Minimum
{
    T operator()(const T &x, const T &y) const
    {
        if constexpr (detail::IsComplex<T>::value) {
            using dpctl::tensor::math_utils::min_complex;
            return min_complex<T>(x, y);
        }
        else if constexpr (std::is_floating_point_v<T> ||
                           std::is_same_v<T, sycl::half>) {
            return (std::isnan(x) || x < y) ? x : y;
        }
        else if constexpr (std::is_same_v<T, bool>) {
            return x && y;
        }
        else {
            return (x < y) ? x : y;
        }
    }
};

// Define identities and operator checking structs

template <typename Op, typename T, typename = void>
struct GetIdentity
{
};

// Maximum

template <typename T, class Op>
using IsMaximum = std::bool_constant<std::is_same_v<Op, sycl::maximum<T>> ||
                                     std::is_same_v<Op, Maximum<T>>>;

template <typename T, class Op>
using IsSyclMaximum = std::bool_constant<std::is_same_v<Op, sycl::maximum<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsMaximum<T, Op>::value>>
{
    static constexpr T value =
        static_cast<T>(std::numeric_limits<T>::has_infinity
                           ? static_cast<T>(-std::numeric_limits<T>::infinity())
                           : std::numeric_limits<T>::lowest());
};

template <typename Op>
struct GetIdentity<Op, bool, std::enable_if_t<IsMaximum<bool, Op>::value>>
{
    static constexpr bool value = false;
};

template <typename Op, typename T>
struct GetIdentity<Op,
                   std::complex<T>,
                   std::enable_if_t<IsMaximum<std::complex<T>, Op>::value>>
{
    static constexpr std::complex<T> value{-std::numeric_limits<T>::infinity(),
                                           -std::numeric_limits<T>::infinity()};
};

// Minimum

template <typename T, class Op>
using IsMinimum = std::bool_constant<std::is_same_v<Op, sycl::minimum<T>> ||
                                     std::is_same_v<Op, Minimum<T>>>;

template <typename T, class Op>
using IsSyclMinimum = std::bool_constant<std::is_same_v<Op, sycl::minimum<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsMinimum<T, Op>::value>>
{
    static constexpr T value =
        static_cast<T>(std::numeric_limits<T>::has_infinity
                           ? static_cast<T>(std::numeric_limits<T>::infinity())
                           : std::numeric_limits<T>::max());
};

template <typename Op>
struct GetIdentity<Op, bool, std::enable_if_t<IsMinimum<bool, Op>::value>>
{
    static constexpr bool value = true;
};

template <typename Op, typename T>
struct GetIdentity<Op,
                   std::complex<T>,
                   std::enable_if_t<IsMinimum<std::complex<T>, Op>::value>>
{
    static constexpr std::complex<T> value{std::numeric_limits<T>::infinity(),
                                           std::numeric_limits<T>::infinity()};
};

// Plus

template <typename T, class Op>
using IsPlus = std::bool_constant<std::is_same_v<Op, sycl::plus<T>> ||
                                  std::is_same_v<Op, std::plus<T>>>;

template <typename T, class Op>
using IsSyclPlus = std::bool_constant<std::is_same_v<Op, sycl::plus<T>>>;

// Multiplies

template <typename T, class Op>
using IsMultiplies =
    std::bool_constant<std::is_same_v<Op, sycl::multiplies<T>> ||
                       std::is_same_v<Op, std::multiplies<T>>>;

template <typename T, class Op>
using IsSyclMultiplies =
    std::bool_constant<std::is_same_v<Op, sycl::multiplies<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsMultiplies<T, Op>::value>>
{
    static constexpr T value = static_cast<T>(1);
};

// LogSumExp

template <typename T>
struct LogSumExp
{
    T operator()(const T &x, const T &y) const
    {
        using dpctl::tensor::math_utils::logaddexp;
        return logaddexp<T>(x, y);
    }
};

template <typename T, class Op>
using IsLogSumExp = std::bool_constant<std::is_same_v<Op, LogSumExp<T>>>;

// only defined for types with infinity
template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsLogSumExp<T, Op>::value>>
{
    static constexpr T value = -std::numeric_limits<T>::infinity();
};

// Hypot

template <typename T>
struct Hypot
{
    T operator()(const T &x, const T &y) const
    {
        return sycl::hypot(x, y);
    }
};

template <typename T, class Op>
using IsHypot = std::bool_constant<std::is_same_v<Op, Hypot<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsHypot<T, Op>::value>>
{
    static constexpr T value = 0;
};

// Logical_And

template <typename T, class Op>
using IsLogicalAnd =
    std::bool_constant<std::is_same_v<Op, sycl::logical_and<T>> ||
                       std::is_same_v<Op, std::logical_and<T>>>;

template <typename T, class Op>
using IsSyclLogicalAnd =
    std::bool_constant<std::is_same_v<Op, sycl::logical_and<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsLogicalAnd<T, Op>::value>>
{
    static constexpr T value = static_cast<T>(1);
};

// Logical_Or

template <typename T, class Op>
using IsLogicalOr =
    std::bool_constant<std::is_same_v<Op, sycl::logical_or<T>> ||
                       std::is_same_v<Op, std::logical_or<T>>>;

template <typename T, class Op>
using IsSyclLogicalOr =
    std::bool_constant<std::is_same_v<Op, sycl::logical_or<T>>>;

template <typename Op, typename T>
struct GetIdentity<Op, T, std::enable_if_t<IsLogicalOr<T, Op>::value>>
{
    static constexpr T value = static_cast<T>(0);
};

// Identity

template <typename Op, typename T, typename = void>
struct Identity
{
};

template <typename Op, typename T>
using UseBuiltInIdentity =
    std::conjunction<IsSyclOp<T, Op>, sycl::has_known_identity<Op, T>>;

template <typename Op, typename T>
struct Identity<Op, T, std::enable_if_t<!UseBuiltInIdentity<Op, T>::value>>
{
    static constexpr T value = GetIdentity<Op, T>::value;
};

template <typename Op, typename T>
struct Identity<Op, T, std::enable_if_t<UseBuiltInIdentity<Op, T>::value>>
{
    static constexpr T value = sycl::known_identity<Op, T>::value;
};

// Sub-group load/store

#ifndef USE_GROUP_LOAD_STORE
#if defined(SYCL_EXT_ONEAPI_GROUP_LOAD_STORE) &&                               \
    SYCL_EXT_ONEAPI_GROUP_LOAD_STORE
#define USE_GROUP_LOAD_STORE 1
#else
#if defined(__LIBSYCL_MAJOR_VERSION) && (__LIBSYCL_MAJOR_VERSION >= 8u)
#define USE_GROUP_LOAD_STORE 1
#else
#define USE_GROUP_LOAD_STORE 0
#endif
#endif
#endif

#if (USE_GROUP_LOAD_STORE)
namespace ls_ns = sycl::ext::oneapi::experimental;
#endif

template <std::uint8_t vec_sz,
          sycl::access::address_space Space,
          sycl::access::decorated DecorateAddress,
          typename ElementType>
auto sub_group_load(const sycl::sub_group &sg,
                    sycl::multi_ptr<ElementType, Space, DecorateAddress> m_ptr)
{
#if (USE_GROUP_LOAD_STORE)
    using ValueT = typename std::remove_cv_t<ElementType>;
    sycl::vec<ValueT, vec_sz> x{};
    static constexpr auto striped =
        ls_ns::properties{ls_ns::data_placement_striped};
    ls_ns::group_load(sg, m_ptr, x, striped);
    return x;
#else
    return sg.load<vec_sz>(m_ptr);
#endif
}

template <sycl::access::address_space Space,
          sycl::access::decorated DecorateAddress,
          typename ElementType>
auto sub_group_load(const sycl::sub_group &sg,
                    sycl::multi_ptr<ElementType, Space, DecorateAddress> m_ptr)
{
#if (USE_GROUP_LOAD_STORE)
    using ValueT = typename std::remove_cv_t<ElementType>;
    ValueT x{};
    static constexpr auto striped =
        ls_ns::properties{ls_ns::data_placement_striped};
    ls_ns::group_load(sg, m_ptr, x, striped);
    return x;
#else
    return sg.load(m_ptr);
#endif
}

template <std::uint8_t vec_sz,
          sycl::access::address_space Space,
          sycl::access::decorated DecorateAddress,
          typename VecT,
          typename ElementType>
std::enable_if_t<
    std::is_same_v<std::remove_cv_t<ElementType>, std::remove_cv_t<VecT>>,
    void>
    sub_group_store(const sycl::sub_group &sg,
                    const sycl::vec<VecT, vec_sz> &val,
                    sycl::multi_ptr<ElementType, Space, DecorateAddress> m_ptr)
{
#if (USE_GROUP_LOAD_STORE)
    static_assert(std::is_same_v<VecT, ElementType>);
    static constexpr auto striped =
        ls_ns::properties{ls_ns::data_placement_striped};
    ls_ns::group_store(sg, val, m_ptr, striped);
    return;
#else
    sg.store<vec_sz>(m_ptr, val);
    return;
#endif
}

template <sycl::access::address_space Space,
          sycl::access::decorated DecorateAddress,
          typename VecT,
          typename ElementType>
std::enable_if_t<
    std::is_same_v<std::remove_cv_t<ElementType>, std::remove_cv_t<VecT>>,
    void>
    sub_group_store(const sycl::sub_group &sg,
                    const VecT &val,
                    sycl::multi_ptr<ElementType, Space, DecorateAddress> m_ptr)
{
#if (USE_GROUP_LOAD_STORE)
    static constexpr auto striped =
        ls_ns::properties{ls_ns::data_placement_striped};
    ls_ns::group_store(sg, val, m_ptr, striped);
    return;
#else
    sg.store(m_ptr, val);
    return;
#endif
}
} // namespace dpctl::tensor::sycl_utils
