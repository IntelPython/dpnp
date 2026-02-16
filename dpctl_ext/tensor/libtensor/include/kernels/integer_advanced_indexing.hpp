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
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for advanced tensor index operations.
//===----------------------------------------------------------------------===//

#pragma once
#include <cstddef>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include "dpctl_tensor_types.hpp"
#include "utils/indexing_utils.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::indexing
{

using dpctl::tensor::ssize_t;

template <typename ProjectorT,
          typename OrthogIndexer,
          typename IndicesIndexer,
          typename AxesIndexer,
          typename T,
          typename indT>
class TakeFunctor
{
private:
    const char *src_ = nullptr;
    char *dst_ = nullptr;
    char **ind_ = nullptr;
    int k_ = 0;
    std::size_t ind_nelems_ = 0;
    const ssize_t *axes_shape_and_strides_ = nullptr;
    OrthogIndexer orthog_strider;
    IndicesIndexer ind_strider;
    AxesIndexer axes_strider;

public:
    TakeFunctor(const char *src_cp,
                char *dst_cp,
                char **ind_cp,
                int k,
                std::size_t ind_nelems,
                const ssize_t *axes_shape_and_strides,
                const OrthogIndexer &orthog_strider_,
                const IndicesIndexer &ind_strider_,
                const AxesIndexer &axes_strider_)
        : src_(src_cp), dst_(dst_cp), ind_(ind_cp), k_(k),
          ind_nelems_(ind_nelems),
          axes_shape_and_strides_(axes_shape_and_strides),
          orthog_strider(orthog_strider_), ind_strider(ind_strider_),
          axes_strider(axes_strider_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        const T *src = reinterpret_cast<const T *>(src_);
        T *dst = reinterpret_cast<T *>(dst_);

        ssize_t i_orthog = id / ind_nelems_;
        ssize_t i_along = id - (i_orthog * ind_nelems_);

        auto orthog_offsets = orthog_strider(i_orthog);

        ssize_t src_offset = orthog_offsets.get_first_offset();
        ssize_t dst_offset = orthog_offsets.get_second_offset();

        static constexpr ProjectorT proj{};
        for (int axis_idx = 0; axis_idx < k_; ++axis_idx) {
            indT *ind_data = reinterpret_cast<indT *>(ind_[axis_idx]);

            ssize_t ind_offset = ind_strider(i_along, axis_idx);
            // proj produces an index in the range of the given axis
            ssize_t projected_idx =
                proj(axes_shape_and_strides_[axis_idx], ind_data[ind_offset]);
            src_offset +=
                projected_idx * axes_shape_and_strides_[k_ + axis_idx];
        }

        dst_offset += axes_strider(i_along);

        dst[dst_offset] = src[src_offset];
    }
};

template <typename ProjectorT,
          typename OrthogIndexer,
          typename IndicesIndexer,
          typename AxesIndexer,
          typename T,
          typename indT>
class take_kernel;

typedef sycl::event (*take_fn_ptr_t)(sycl::queue &,
                                     std::size_t,
                                     std::size_t,
                                     int,
                                     int,
                                     int,
                                     const ssize_t *,
                                     const ssize_t *,
                                     const ssize_t *,
                                     const char *,
                                     char *,
                                     char **,
                                     ssize_t,
                                     ssize_t,
                                     const ssize_t *,
                                     const std::vector<sycl::event> &);

template <typename ProjectorT, typename Ty, typename indT>
sycl::event take_impl(sycl::queue &q,
                      std::size_t orthog_nelems,
                      std::size_t ind_nelems,
                      int nd,
                      int ind_nd,
                      int k,
                      const ssize_t *orthog_shape_and_strides,
                      const ssize_t *axes_shape_and_strides,
                      const ssize_t *ind_shape_and_strides,
                      const char *src_p,
                      char *dst_p,
                      char **ind_p,
                      ssize_t src_offset,
                      ssize_t dst_offset,
                      const ssize_t *ind_offsets,
                      const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    sycl::event take_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using OrthogIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
        const OrthogIndexerT orthog_indexer{nd, src_offset, dst_offset,
                                            orthog_shape_and_strides};

        using NthStrideIndexerT = dpctl::tensor::offset_utils::NthStrideOffset;
        const NthStrideIndexerT indices_indexer{ind_nd, ind_offsets,
                                                ind_shape_and_strides};

        using AxesIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        const AxesIndexerT axes_indexer{ind_nd, 0,
                                        axes_shape_and_strides + (2 * k)};

        using KernelName =
            take_kernel<ProjectorT, OrthogIndexerT, NthStrideIndexerT,
                        AxesIndexerT, Ty, indT>;

        const std::size_t gws = orthog_nelems * ind_nelems;

        cgh.parallel_for<KernelName>(
            sycl::range<1>(gws),
            TakeFunctor<ProjectorT, OrthogIndexerT, NthStrideIndexerT,
                        AxesIndexerT, Ty, indT>(
                src_p, dst_p, ind_p, k, ind_nelems, axes_shape_and_strides,
                orthog_indexer, indices_indexer, axes_indexer));
    });

    return take_ev;
}

template <typename ProjectorT,
          typename OrthogIndexer,
          typename IndicesIndexer,
          typename AxesIndexer,
          typename T,
          typename indT>
class PutFunctor
{
private:
    char *dst_ = nullptr;
    const char *val_ = nullptr;
    char **ind_ = nullptr;
    int k_ = 0;
    std::size_t ind_nelems_ = 0;
    const ssize_t *axes_shape_and_strides_ = nullptr;
    OrthogIndexer orthog_strider;
    IndicesIndexer ind_strider;
    AxesIndexer axes_strider;

public:
    PutFunctor(char *dst_cp,
               const char *val_cp,
               char **ind_cp,
               int k,
               std::size_t ind_nelems,
               const ssize_t *axes_shape_and_strides,
               const OrthogIndexer &orthog_strider_,
               const IndicesIndexer &ind_strider_,
               const AxesIndexer &axes_strider_)
        : dst_(dst_cp), val_(val_cp), ind_(ind_cp), k_(k),
          ind_nelems_(ind_nelems),
          axes_shape_and_strides_(axes_shape_and_strides),
          orthog_strider(orthog_strider_), ind_strider(ind_strider_),
          axes_strider(axes_strider_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        T *dst = reinterpret_cast<T *>(dst_);
        const T *val = reinterpret_cast<const T *>(val_);

        ssize_t i_orthog = id / ind_nelems_;
        ssize_t i_along = id - (i_orthog * ind_nelems_);

        auto orthog_offsets = orthog_strider(i_orthog);

        ssize_t dst_offset = orthog_offsets.get_first_offset();
        ssize_t val_offset = orthog_offsets.get_second_offset();

        static constexpr ProjectorT proj{};
        for (int axis_idx = 0; axis_idx < k_; ++axis_idx) {
            indT *ind_data = reinterpret_cast<indT *>(ind_[axis_idx]);

            ssize_t ind_offset = ind_strider(i_along, axis_idx);

            // proj produces an index in the range of the given axis
            ssize_t projected_idx =
                proj(axes_shape_and_strides_[axis_idx], ind_data[ind_offset]);
            dst_offset +=
                projected_idx * axes_shape_and_strides_[k_ + axis_idx];
        }

        val_offset += axes_strider(i_along);

        dst[dst_offset] = val[val_offset];
    }
};

template <typename ProjectorT,
          typename OrthogIndexer,
          typename IndicesIndexer,
          typename AxesIndexer,
          typename T,
          typename indT>
class put_kernel;

typedef sycl::event (*put_fn_ptr_t)(sycl::queue &,
                                    std::size_t,
                                    std::size_t,
                                    int,
                                    int,
                                    int,
                                    const ssize_t *,
                                    const ssize_t *,
                                    const ssize_t *,
                                    char *,
                                    const char *,
                                    char **,
                                    ssize_t,
                                    ssize_t,
                                    const ssize_t *,
                                    const std::vector<sycl::event> &);

template <typename ProjectorT, typename Ty, typename indT>
sycl::event put_impl(sycl::queue &q,
                     std::size_t orthog_nelems,
                     std::size_t ind_nelems,
                     int nd,
                     int ind_nd,
                     int k,
                     const ssize_t *orthog_shape_and_strides,
                     const ssize_t *axes_shape_and_strides,
                     const ssize_t *ind_shape_and_strides,
                     char *dst_p,
                     const char *val_p,
                     char **ind_p,
                     ssize_t dst_offset,
                     ssize_t val_offset,
                     const ssize_t *ind_offsets,
                     const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(q);

    sycl::event put_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using OrthogIndexerT =
            dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
        const OrthogIndexerT orthog_indexer{nd, dst_offset, val_offset,
                                            orthog_shape_and_strides};

        using NthStrideIndexerT = dpctl::tensor::offset_utils::NthStrideOffset;
        const NthStrideIndexerT indices_indexer{ind_nd, ind_offsets,
                                                ind_shape_and_strides};

        using AxesIndexerT = dpctl::tensor::offset_utils::StridedIndexer;
        const AxesIndexerT axes_indexer{ind_nd, 0,
                                        axes_shape_and_strides + (2 * k)};

        using KernelName =
            put_kernel<ProjectorT, OrthogIndexerT, NthStrideIndexerT,
                       AxesIndexerT, Ty, indT>;

        const std::size_t gws = orthog_nelems * ind_nelems;

        cgh.parallel_for<KernelName>(
            sycl::range<1>(gws),
            PutFunctor<ProjectorT, OrthogIndexerT, NthStrideIndexerT,
                       AxesIndexerT, Ty, indT>(
                dst_p, val_p, ind_p, k, ind_nelems, axes_shape_and_strides,
                orthog_indexer, indices_indexer, axes_indexer));
    });

    return put_ev;
}

template <typename fnT, typename T, typename indT>
struct TakeWrapFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<indT>::value &&
                      !std::is_same<indT, bool>::value) {
            using dpctl::tensor::indexing_utils::WrapIndex;
            fnT fn = take_impl<WrapIndex<indT>, T, indT>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

template <typename fnT, typename T, typename indT>
struct TakeClipFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<indT>::value &&
                      !std::is_same<indT, bool>::value) {
            using dpctl::tensor::indexing_utils::ClipIndex;
            fnT fn = take_impl<ClipIndex<indT>, T, indT>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

template <typename fnT, typename T, typename indT>
struct PutWrapFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<indT>::value &&
                      !std::is_same<indT, bool>::value) {
            using dpctl::tensor::indexing_utils::WrapIndex;
            fnT fn = put_impl<WrapIndex<indT>, T, indT>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

template <typename fnT, typename T, typename indT>
struct PutClipFactory
{
    fnT get()
    {
        if constexpr (std::is_integral<indT>::value &&
                      !std::is_same<indT, bool>::value) {
            using dpctl::tensor::indexing_utils::ClipIndex;
            fnT fn = put_impl<ClipIndex<indT>, T, indT>;
            return fn;
        }
        else {
            fnT fn = nullptr;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::kernels::indexing
