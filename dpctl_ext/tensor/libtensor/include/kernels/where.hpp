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
/// This file defines kernels for dpctl.tensor.where.
//===---------------------------------------------------------------------===//

#pragma once
#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl_tensor_types.hpp"
#include "kernels/alignment.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::search
{

using dpctl::tensor::ssize_t;
using namespace dpctl::tensor::offset_utils;

using dpctl::tensor::kernels::alignment_utils::
    disabled_sg_loadstore_wrapper_krn;
using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

using dpctl::tensor::sycl_utils::sub_group_load;
using dpctl::tensor::sycl_utils::sub_group_store;

template <typename T, typename condT, typename IndexerT>
class where_strided_kernel;
template <typename T, typename condT, std::uint8_t vec_sz, std::uint8_t n_vecs>
class where_contig_kernel;

template <typename T,
          typename condT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
class WhereContigFunctor
{
private:
    std::size_t nelems = 0;
    const condT *cond_p = nullptr;
    const T *x1_p = nullptr;
    const T *x2_p = nullptr;
    T *dst_p = nullptr;

public:
    WhereContigFunctor(std::size_t nelems_,
                       const condT *cond_p_,
                       const T *x1_p_,
                       const T *x2_p_,
                       T *dst_p_)
        : nelems(nelems_), cond_p(cond_p_), x1_p(x1_p_), x2_p(x2_p_),
          dst_p(dst_p_)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        static constexpr std::uint8_t nelems_per_wi = n_vecs * vec_sz;

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (!enable_sg_loadstore || is_complex<condT>::value ||
                      is_complex<T>::value)
        {
            const std::uint16_t sgSize =
                ndit.get_sub_group().get_local_range()[0];
            const std::size_t gid = ndit.get_global_linear_id();

            const std::uint16_t nelems_per_sg = sgSize * nelems_per_wi;
            const std::size_t start =
                (gid / sgSize) * (nelems_per_sg - sgSize) + gid;
            const std::size_t end = std::min(nelems, start + nelems_per_sg);
            for (std::size_t offset = start; offset < end; offset += sgSize) {
                using dpctl::tensor::type_utils::convert_impl;
                const bool check = convert_impl<bool, condT>(cond_p[offset]);
                dst_p[offset] = check ? x1_p[offset] : x2_p[offset];
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
                    auto x1_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&x1_p[idx]);
                    auto x2_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&x2_p[idx]);
                    auto cond_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&cond_p[idx]);
                    auto dst_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&dst_p[idx]);

                    const sycl::vec<T, vec_sz> x1_vec =
                        sub_group_load<vec_sz>(sg, x1_multi_ptr);
                    const sycl::vec<T, vec_sz> x2_vec =
                        sub_group_load<vec_sz>(sg, x2_multi_ptr);
                    const sycl::vec<condT, vec_sz> cond_vec =
                        sub_group_load<vec_sz>(sg, cond_multi_ptr);
#pragma unroll
                    for (std::uint8_t k = 0; k < vec_sz; ++k) {
                        dst_vec[k] = cond_vec[k] ? x1_vec[k] : x2_vec[k];
                    }
                    sub_group_store<vec_sz>(sg, dst_vec, dst_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems; k += sgSize) {
                    dst_p[k] = cond_p[k] ? x1_p[k] : x2_p[k];
                }
            }
        }
    }
};

typedef sycl::event (*where_contig_impl_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    const char *,
    const char *,
    char *,
    const std::vector<sycl::event> &);

template <typename T, typename condT>
sycl::event where_contig_impl(sycl::queue &q,
                              std::size_t nelems,
                              const char *cond_cp,
                              const char *x1_cp,
                              const char *x2_cp,
                              char *dst_cp,
                              const std::vector<sycl::event> &depends)
{
    const condT *cond_tp = reinterpret_cast<const condT *>(cond_cp);
    const T *x1_tp = reinterpret_cast<const T *>(x1_cp);
    const T *x2_tp = reinterpret_cast<const T *>(x2_cp);
    T *dst_tp = reinterpret_cast<T *>(dst_cp);

    sycl::event where_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        std::size_t lws = 64;
        static constexpr std::uint8_t vec_sz = 4u;
        static constexpr std::uint8_t n_vecs = 2u;
        const std::size_t n_groups =
            ((nelems + lws * n_vecs * vec_sz - 1) / (lws * n_vecs * vec_sz));
        const auto gws_range = sycl::range<1>(n_groups * lws);
        const auto lws_range = sycl::range<1>(lws);

        if (is_aligned<required_alignment>(cond_cp) &&
            is_aligned<required_alignment>(x1_cp) &&
            is_aligned<required_alignment>(x2_cp) &&
            is_aligned<required_alignment>(dst_cp))
        {
            static constexpr bool enable_sg_loadstore = true;
            using KernelName = where_contig_kernel<T, condT, vec_sz, n_vecs>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                WhereContigFunctor<T, condT, vec_sz, n_vecs,
                                   enable_sg_loadstore>(nelems, cond_tp, x1_tp,
                                                        x2_tp, dst_tp));
        }
        else {
            static constexpr bool disable_sg_loadstore = false;
            using InnerKernelName =
                where_contig_kernel<T, condT, vec_sz, n_vecs>;
            using KernelName =
                disabled_sg_loadstore_wrapper_krn<InnerKernelName>;

            cgh.parallel_for<KernelName>(
                sycl::nd_range<1>(gws_range, lws_range),
                WhereContigFunctor<T, condT, vec_sz, n_vecs,
                                   disable_sg_loadstore>(nelems, cond_tp, x1_tp,
                                                         x2_tp, dst_tp));
        }
    });

    return where_ev;
}

template <typename T, typename condT, typename IndexerT>
class WhereStridedFunctor
{
private:
    const T *x1_p = nullptr;
    const T *x2_p = nullptr;
    T *dst_p = nullptr;
    const condT *cond_p = nullptr;
    IndexerT indexer;

public:
    WhereStridedFunctor(const condT *cond_p_,
                        const T *x1_p_,
                        const T *x2_p_,
                        T *dst_p_,
                        const IndexerT &indexer_)
        : x1_p(x1_p_), x2_p(x2_p_), dst_p(dst_p_), cond_p(cond_p_),
          indexer(indexer_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        std::size_t gid = id[0];
        auto offsets = indexer(static_cast<ssize_t>(gid));

        using dpctl::tensor::type_utils::convert_impl;
        bool check =
            convert_impl<bool, condT>(cond_p[offsets.get_first_offset()]);

        dst_p[offsets.get_fourth_offset()] =
            check ? x1_p[offsets.get_second_offset()]
                  : x2_p[offsets.get_third_offset()];
    }
};

typedef sycl::event (*where_strided_impl_fn_ptr_t)(
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

template <typename T, typename condT>
sycl::event where_strided_impl(sycl::queue &q,
                               std::size_t nelems,
                               int nd,
                               const char *cond_cp,
                               const char *x1_cp,
                               const char *x2_cp,
                               char *dst_cp,
                               const ssize_t *shape_strides,
                               ssize_t x1_offset,
                               ssize_t x2_offset,
                               ssize_t cond_offset,
                               ssize_t dst_offset,
                               const std::vector<sycl::event> &depends)
{
    const condT *cond_tp = reinterpret_cast<const condT *>(cond_cp);
    const T *x1_tp = reinterpret_cast<const T *>(x1_cp);
    const T *x2_tp = reinterpret_cast<const T *>(x2_cp);
    T *dst_tp = reinterpret_cast<T *>(dst_cp);

    sycl::event where_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const FourOffsets_StridedIndexer indexer{
            nd, cond_offset, x1_offset, x2_offset, dst_offset, shape_strides};

        cgh.parallel_for<
            where_strided_kernel<T, condT, FourOffsets_StridedIndexer>>(
            sycl::range<1>(nelems),
            WhereStridedFunctor<T, condT, FourOffsets_StridedIndexer>(
                cond_tp, x1_tp, x2_tp, dst_tp, indexer));
    });

    return where_ev;
}

template <typename fnT, typename T, typename condT>
struct WhereStridedFactory
{
    fnT get()
    {
        fnT fn = where_strided_impl<T, condT>;
        return fn;
    }
};

template <typename fnT, typename T, typename condT>
struct WhereContigFactory
{
    fnT get()
    {
        fnT fn = where_contig_impl<T, condT>;
        return fn;
    }
};

} // namespace dpctl::tensor::kernels::search
