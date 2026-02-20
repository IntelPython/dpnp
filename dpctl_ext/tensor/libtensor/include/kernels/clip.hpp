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
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for dpctl.tensor.clip.
//===---------------------------------------------------------------------===//

#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include "dpctl_tensor_types.hpp"
#include "kernels/alignment.hpp"
#include "utils/math_utils.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::clip
{

using dpctl::tensor::ssize_t;
using namespace dpctl::tensor::offset_utils;

using dpctl::tensor::kernels::alignment_utils::
    disabled_sg_loadstore_wrapper_krn;
using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

using dpctl::tensor::sycl_utils::sub_group_load;
using dpctl::tensor::sycl_utils::sub_group_store;

template <typename T>
T clip(const T &x, const T &min, const T &max)
{
    using dpctl::tensor::type_utils::is_complex;
    if constexpr (is_complex<T>::value) {
        using dpctl::tensor::math_utils::max_complex;
        using dpctl::tensor::math_utils::min_complex;
        return min_complex(max_complex(x, min), max);
    }
    else if constexpr (std::is_floating_point_v<T> ||
                       std::is_same_v<T, sycl::half>) {
        auto tmp = (std::isnan(x) || x > min) ? x : min;
        return (std::isnan(tmp) || tmp < max) ? tmp : max;
    }
    else if constexpr (std::is_same_v<T, bool>) {
        return (x || min) && max;
    }
    else {
        auto tmp = (x > min) ? x : min;
        return (tmp < max) ? tmp : max;
    }
}

template <typename T,
          std::uint8_t vec_sz = 4,
          std::uint8_t n_vecs = 2,
          bool enable_sg_loadstore = true>
class ClipContigFunctor
{
private:
    std::size_t nelems = 0;
    const T *x_p = nullptr;
    const T *min_p = nullptr;
    const T *max_p = nullptr;
    T *dst_p = nullptr;

public:
    ClipContigFunctor(std::size_t nelems_,
                      const T *x_p_,
                      const T *min_p_,
                      const T *max_p_,
                      T *dst_p_)
        : nelems(nelems_), x_p(x_p_), min_p(min_p_), max_p(max_p_),
          dst_p(dst_p_)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        static constexpr std::uint8_t nelems_per_wi = n_vecs * vec_sz;

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<T>::value || !enable_sg_loadstore) {
            const std::uint16_t sgSize =
                ndit.get_sub_group().get_local_range()[0];
            const std::size_t gid = ndit.get_global_linear_id();
            const std::uint16_t nelems_per_sg = sgSize * nelems_per_wi;

            const std::size_t start =
                (gid / sgSize) * (nelems_per_sg - sgSize) + gid;
            const std::size_t end = std::min(nelems, start + nelems_per_sg);

            for (std::size_t offset = start; offset < end; offset += sgSize) {
                dst_p[offset] = clip(x_p[offset], min_p[offset], max_p[offset]);
            }
        }
        else {
            auto sg = ndit.get_sub_group();
            const std::uint16_t sgSize = sg.get_max_local_range()[0];

            const std::size_t base =
                nelems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                 sg.get_group_id()[0] * sgSize);

            if (base + nelems_per_wi * sgSize < nelems) {
                sycl::vec<T, vec_sz> dst_vec;
#pragma unroll
                for (std::uint8_t it = 0; it < n_vecs * vec_sz; it += vec_sz) {
                    const std::size_t idx = base + it * sgSize;
                    auto x_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&x_p[idx]);
                    auto min_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&min_p[idx]);
                    auto max_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&max_p[idx]);
                    auto dst_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&dst_p[idx]);

                    const sycl::vec<T, vec_sz> x_vec =
                        sub_group_load<vec_sz>(sg, x_multi_ptr);
                    const sycl::vec<T, vec_sz> min_vec =
                        sub_group_load<vec_sz>(sg, min_multi_ptr);
                    const sycl::vec<T, vec_sz> max_vec =
                        sub_group_load<vec_sz>(sg, max_multi_ptr);
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        dst_vec[vec_id] = clip(x_vec[vec_id], min_vec[vec_id],
                                               max_vec[vec_id]);
                    }
                    sub_group_store<vec_sz>(sg, dst_vec, dst_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems; k += sgSize) {
                    dst_p[k] = clip(x_p[k], min_p[k], max_p[k]);
                }
            }
        }
    }
};

template <typename T, int vec_sz, int n_vecs>
class clip_contig_kernel;

typedef sycl::event (*clip_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    const char *,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename T>
sycl::event clip_contig_impl(sycl::queue &q,
                             std::size_t nelems,
                             const char *x_cp,
                             const char *min_cp,
                             const char *max_cp,
                             char *dst_cp,
                             const std::vector<sycl::event> &depends)
{
    const T *x_tp = reinterpret_cast<const T *>(x_cp);
    const T *min_tp = reinterpret_cast<const T *>(min_cp);
    const T *max_tp = reinterpret_cast<const T *>(max_cp);
    T *dst_tp = reinterpret_cast<T *>(dst_cp);

    sycl::event clip_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        std::size_t lws = 64;
        static constexpr std::uint8_t vec_sz = 4;
        static constexpr std::uint8_t n_vecs = 2;
        const std::size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        if (is_aligned<required_alignment>(x_cp) &&
            is_aligned<required_alignment>(min_cp) &&
            is_aligned<required_alignment>(max_cp) &&
            is_aligned<required_alignment>(dst_cp))
        {
            static constexpr bool enable_sg_loadstore = true;
            using KernelName = clip_contig_kernel<T, vec_sz, n_vecs>;
            using Impl =
                ClipContigFunctor<T, vec_sz, n_vecs, enable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(nelems, x_tp, min_tp, max_tp, dst_tp));
        }
        else {
            static constexpr bool disable_sg_loadstore = false;
            using InnerKernelName = clip_contig_kernel<T, vec_sz, n_vecs>;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<InnerKernelName>;
            using Impl =
                ClipContigFunctor<T, vec_sz, n_vecs, disable_sg_loadstore>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                Impl(nelems, x_tp, min_tp, max_tp, dst_tp));
        }
    });

    return clip_ev;
}

template <typename T, typename IndexerT>
class ClipStridedFunctor
{
private:
    const T *x_p = nullptr;
    const T *min_p = nullptr;
    const T *max_p = nullptr;
    T *dst_p = nullptr;
    IndexerT indexer;

public:
    ClipStridedFunctor(const T *x_p_,
                       const T *min_p_,
                       const T *max_p_,
                       T *dst_p_,
                       const IndexerT &indexer_)
        : x_p(x_p_), min_p(min_p_), max_p(max_p_), dst_p(dst_p_),
          indexer(indexer_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        std::size_t gid = id[0];
        auto offsets = indexer(static_cast<ssize_t>(gid));
        dst_p[offsets.get_fourth_offset()] = clip(
            x_p[offsets.get_first_offset()], min_p[offsets.get_second_offset()],
            max_p[offsets.get_third_offset()]);
    }
};

template <typename T, typename IndexerT>
class clip_strided_kernel;

typedef sycl::event (*clip_strided_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    int,
    const char *,
    const char *,
    const char *,
    char *,
    const ssize_t *,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename T>
sycl::event clip_strided_impl(sycl::queue &q,
                              std::size_t nelems,
                              int nd,
                              const char *x_cp,
                              const char *min_cp,
                              const char *max_cp,
                              char *dst_cp,
                              const ssize_t *shape_strides,
                              ssize_t x_offset,
                              ssize_t min_offset,
                              ssize_t max_offset,
                              ssize_t dst_offset,
                              const std::vector<sycl::event> &depends)
{
    const T *x_tp = reinterpret_cast<const T *>(x_cp);
    const T *min_tp = reinterpret_cast<const T *>(min_cp);
    const T *max_tp = reinterpret_cast<const T *>(max_cp);
    T *dst_tp = reinterpret_cast<T *>(dst_cp);

    sycl::event clip_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const FourOffsets_StridedIndexer indexer{
            nd, x_offset, min_offset, max_offset, dst_offset, shape_strides};

        using KernelName = clip_strided_kernel<T, FourOffsets_StridedIndexer>;
        using Impl = ClipStridedFunctor<T, FourOffsets_StridedIndexer>;

        cgh.parallel_for<KernelName>(
            sycl::range<1>(nelems),
            Impl(x_tp, min_tp, max_tp, dst_tp, indexer));
    });

    return clip_ev;
}

template <typename fnT, typename T>
struct ClipStridedFactory
{
    fnT get()
    {
        fnT fn = clip_strided_impl<T>;
        return fn;
    }
};

template <typename fnT, typename T>
struct ClipContigFactory
{
    fnT get()
    {

        fnT fn = clip_contig_impl<T>;
        return fn;
    }
};

} // namespace dpctl::tensor::kernels::clip
