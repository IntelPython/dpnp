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
/// This file defines kernels for tensor repeating operations.
//===----------------------------------------------------------------------===//

#pragma once
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::repeat
{

using dpctl::tensor::ssize_t;
using namespace dpctl::tensor::offset_utils;

template <typename OrthogIndexer,
          typename SrcAxisIndexer,
          typename DstAxisIndexer,
          typename RepIndexer,
          typename T,
          typename repT>
class repeat_by_sequence_kernel;

template <typename OrthogIndexer,
          typename SrcAxisIndexer,
          typename DstAxisIndexer,
          typename RepIndexer,
          typename T,
          typename repT>
class RepeatSequenceFunctor
{
private:
    const T *src = nullptr;
    T *dst = nullptr;
    const repT *reps = nullptr;
    const repT *cumsum = nullptr;
    std::size_t src_axis_nelems = 1;
    OrthogIndexer orthog_strider;
    SrcAxisIndexer src_axis_strider;
    DstAxisIndexer dst_axis_strider;
    RepIndexer reps_strider;

public:
    RepeatSequenceFunctor(const T *src_,
                          T *dst_,
                          const repT *reps_,
                          const repT *cumsum_,
                          std::size_t src_axis_nelems_,
                          const OrthogIndexer &orthog_strider_,
                          const SrcAxisIndexer &src_axis_strider_,
                          const DstAxisIndexer &dst_axis_strider_,
                          const RepIndexer &reps_strider_)
        : src(src_), dst(dst_), reps(reps_), cumsum(cumsum_),
          src_axis_nelems(src_axis_nelems_), orthog_strider(orthog_strider_),
          src_axis_strider(src_axis_strider_),
          dst_axis_strider(dst_axis_strider_), reps_strider(reps_strider_)
    {
    }

    void operator()(sycl::id<1> idx) const
    {
        std::size_t id = idx[0];
        auto i_orthog = id / src_axis_nelems;
        auto i_along = id - (i_orthog * src_axis_nelems);

        auto orthog_offsets = orthog_strider(i_orthog);
        auto src_offset = orthog_offsets.get_first_offset();
        auto dst_offset = orthog_offsets.get_second_offset();

        auto val = src[src_offset + src_axis_strider(i_along)];
        auto last = cumsum[i_along];
        auto first = last - reps[reps_strider(i_along)];
        for (auto i = first; i < last; ++i) {
            dst[dst_offset + dst_axis_strider(i)] = val;
        }
    }
};

typedef sycl::event (*repeat_by_sequence_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    char *,
    const char *,
    const char *,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename T, typename repT>
sycl::event
    repeat_by_sequence_impl(sycl::queue &q,
                            std::size_t orthog_nelems,
                            std::size_t src_axis_nelems,
                            const char *src_cp,
                            char *dst_cp,
                            const char *reps_cp,
                            const char *cumsum_cp,
                            int orthog_nd,
                            const ssize_t *orthog_src_dst_shape_and_strides,
                            ssize_t src_offset,
                            ssize_t dst_offset,
                            ssize_t src_axis_shape,
                            ssize_t src_axis_stride,
                            ssize_t dst_axis_shape,
                            ssize_t dst_axis_stride,
                            ssize_t reps_shape,
                            ssize_t reps_stride,
                            const std::vector<sycl::event> &depends)
{
    sycl::event repeat_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const T *src_tp = reinterpret_cast<const T *>(src_cp);
        const repT *reps_tp = reinterpret_cast<const repT *>(reps_cp);
        const repT *cumsum_tp = reinterpret_cast<const repT *>(cumsum_cp);
        T *dst_tp = reinterpret_cast<T *>(dst_cp);

        // orthog ndim indexer
        const TwoOffsets_StridedIndexer orthog_indexer{
            orthog_nd, src_offset, dst_offset,
            orthog_src_dst_shape_and_strides};
        // indexers along repeated axis
        const Strided1DIndexer src_axis_indexer{/* size */ src_axis_shape,
                                                /* step */ src_axis_stride};
        const Strided1DIndexer dst_axis_indexer{/* size */ dst_axis_shape,
                                                /* step */ dst_axis_stride};
        // indexer along reps array
        const Strided1DIndexer reps_indexer{/* size */ reps_shape,
                                            /* step */ reps_stride};

        const std::size_t gws = orthog_nelems * src_axis_nelems;

        cgh.parallel_for<repeat_by_sequence_kernel<
            TwoOffsets_StridedIndexer, Strided1DIndexer, Strided1DIndexer,
            Strided1DIndexer, T, repT>>(
            sycl::range<1>(gws),
            RepeatSequenceFunctor<TwoOffsets_StridedIndexer, Strided1DIndexer,
                                  Strided1DIndexer, Strided1DIndexer, T, repT>(
                src_tp, dst_tp, reps_tp, cumsum_tp, src_axis_nelems,
                orthog_indexer, src_axis_indexer, dst_axis_indexer,
                reps_indexer));
    });

    return repeat_ev;
}

template <typename fnT, typename T>
struct RepeatSequenceFactory
{
    fnT get()
    {
        fnT fn = repeat_by_sequence_impl<T, std::int64_t>;
        return fn;
    }
};

typedef sycl::event (*repeat_by_sequence_1d_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    char *,
    const char *,
    const char *,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename T, typename repT>
sycl::event repeat_by_sequence_1d_impl(sycl::queue &q,
                                       std::size_t src_nelems,
                                       const char *src_cp,
                                       char *dst_cp,
                                       const char *reps_cp,
                                       const char *cumsum_cp,
                                       int src_nd,
                                       const ssize_t *src_shape_strides,
                                       ssize_t dst_shape,
                                       ssize_t dst_stride,
                                       ssize_t reps_shape,
                                       ssize_t reps_stride,
                                       const std::vector<sycl::event> &depends)
{
    sycl::event repeat_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const T *src_tp = reinterpret_cast<const T *>(src_cp);
        const repT *reps_tp = reinterpret_cast<const repT *>(reps_cp);
        const repT *cumsum_tp = reinterpret_cast<const repT *>(cumsum_cp);
        T *dst_tp = reinterpret_cast<T *>(dst_cp);

        // orthog ndim indexer
        static constexpr TwoZeroOffsets_Indexer orthog_indexer{};
        // indexers along repeated axis
        const StridedIndexer src_indexer{src_nd, 0, src_shape_strides};
        const Strided1DIndexer dst_indexer{/* size */ dst_shape,
                                           /* step */ dst_stride};
        // indexer along reps array
        const Strided1DIndexer reps_indexer{/* size */ reps_shape,
                                            /* step */ reps_stride};

        const std::size_t gws = src_nelems;

        cgh.parallel_for<repeat_by_sequence_kernel<
            TwoZeroOffsets_Indexer, StridedIndexer, Strided1DIndexer,
            Strided1DIndexer, T, repT>>(
            sycl::range<1>(gws),
            RepeatSequenceFunctor<TwoZeroOffsets_Indexer, StridedIndexer,
                                  Strided1DIndexer, Strided1DIndexer, T, repT>(
                src_tp, dst_tp, reps_tp, cumsum_tp, src_nelems, orthog_indexer,
                src_indexer, dst_indexer, reps_indexer));
    });

    return repeat_ev;
}

template <typename fnT, typename T>
struct RepeatSequence1DFactory
{
    fnT get()
    {
        fnT fn = repeat_by_sequence_1d_impl<T, std::int64_t>;
        return fn;
    }
};

template <typename OrthogIndexer,
          typename SrcAxisIndexer,
          typename DstAxisIndexer,
          typename T>
class repeat_by_scalar_kernel;

template <typename OrthogIndexer,
          typename SrcAxisIndexer,
          typename DstAxisIndexer,
          typename T>
class RepeatScalarFunctor
{
private:
    const T *src = nullptr;
    T *dst = nullptr;
    ssize_t reps = 1;
    std::size_t dst_axis_nelems = 0;
    OrthogIndexer orthog_strider;
    SrcAxisIndexer src_axis_strider;
    DstAxisIndexer dst_axis_strider;

public:
    RepeatScalarFunctor(const T *src_,
                        T *dst_,
                        const ssize_t reps_,
                        std::size_t dst_axis_nelems_,
                        const OrthogIndexer &orthog_strider_,
                        const SrcAxisIndexer &src_axis_strider_,
                        const DstAxisIndexer &dst_axis_strider_)
        : src(src_), dst(dst_), reps(reps_), dst_axis_nelems(dst_axis_nelems_),
          orthog_strider(orthog_strider_), src_axis_strider(src_axis_strider_),
          dst_axis_strider(dst_axis_strider_)
    {
    }

    void operator()(sycl::id<1> idx) const
    {
        std::size_t id = idx[0];
        auto i_orthog = id / dst_axis_nelems;
        auto i_along = id - (i_orthog * dst_axis_nelems);

        auto orthog_offsets = orthog_strider(i_orthog);
        auto src_offset = orthog_offsets.get_first_offset();
        auto dst_offset = orthog_offsets.get_second_offset();

        auto dst_axis_offset = dst_axis_strider(i_along);
        auto src_axis_offset = src_axis_strider(i_along / reps);
        dst[dst_offset + dst_axis_offset] = src[src_offset + src_axis_offset];
    }
};

typedef sycl::event (*repeat_by_scalar_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    std::size_t,
    const char *,
    char *,
    const ssize_t,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename T>
sycl::event repeat_by_scalar_impl(sycl::queue &q,
                                  std::size_t orthog_nelems,
                                  std::size_t dst_axis_nelems,
                                  const char *src_cp,
                                  char *dst_cp,
                                  const ssize_t reps,
                                  int orthog_nd,
                                  const ssize_t *orthog_shape_and_strides,
                                  ssize_t src_offset,
                                  ssize_t dst_offset,
                                  ssize_t src_axis_shape,
                                  ssize_t src_axis_stride,
                                  ssize_t dst_axis_shape,
                                  ssize_t dst_axis_stride,
                                  const std::vector<sycl::event> &depends)
{
    sycl::event repeat_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const T *src_tp = reinterpret_cast<const T *>(src_cp);
        T *dst_tp = reinterpret_cast<T *>(dst_cp);

        // orthog ndim indexer
        const TwoOffsets_StridedIndexer orthog_indexer{
            orthog_nd, src_offset, dst_offset, orthog_shape_and_strides};
        // indexers along repeated axis
        const Strided1DIndexer src_axis_indexer{/* size */ src_axis_shape,
                                                /* step */ src_axis_stride};
        const Strided1DIndexer dst_axis_indexer{/* size */ dst_axis_shape,
                                                /* step */ dst_axis_stride};

        const std::size_t gws = orthog_nelems * dst_axis_nelems;

        cgh.parallel_for<repeat_by_scalar_kernel<
            TwoOffsets_StridedIndexer, Strided1DIndexer, Strided1DIndexer, T>>(
            sycl::range<1>(gws),
            RepeatScalarFunctor<TwoOffsets_StridedIndexer, Strided1DIndexer,
                                Strided1DIndexer, T>(
                src_tp, dst_tp, reps, dst_axis_nelems, orthog_indexer,
                src_axis_indexer, dst_axis_indexer));
    });

    return repeat_ev;
}

template <typename fnT, typename T>
struct RepeatScalarFactory
{
    fnT get()
    {
        fnT fn = repeat_by_scalar_impl<T>;
        return fn;
    }
};

typedef sycl::event (*repeat_by_scalar_1d_fn_ptr_t)(
    sycl::queue &,
    std::size_t,
    const char *,
    char *,
    const ssize_t,
    int,
    const ssize_t *,
    ssize_t,
    ssize_t,
    const std::vector<sycl::event> &);

template <typename T>
sycl::event repeat_by_scalar_1d_impl(sycl::queue &q,
                                     std::size_t dst_nelems,
                                     const char *src_cp,
                                     char *dst_cp,
                                     const ssize_t reps,
                                     int src_nd,
                                     const ssize_t *src_shape_strides,
                                     ssize_t dst_shape,
                                     ssize_t dst_stride,
                                     const std::vector<sycl::event> &depends)
{
    sycl::event repeat_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        const T *src_tp = reinterpret_cast<const T *>(src_cp);
        T *dst_tp = reinterpret_cast<T *>(dst_cp);

        // orthog ndim indexer
        static constexpr TwoZeroOffsets_Indexer orthog_indexer{};
        // indexers along repeated axis
        const StridedIndexer src_indexer(src_nd, 0, src_shape_strides);
        const Strided1DIndexer dst_indexer{/* size */ dst_shape,
                                           /* step */ dst_stride};

        const std::size_t gws = dst_nelems;

        cgh.parallel_for<repeat_by_scalar_kernel<
            TwoZeroOffsets_Indexer, StridedIndexer, Strided1DIndexer, T>>(
            sycl::range<1>(gws),
            RepeatScalarFunctor<TwoZeroOffsets_Indexer, StridedIndexer,
                                Strided1DIndexer, T>(src_tp, dst_tp, reps,
                                                     dst_nelems, orthog_indexer,
                                                     src_indexer, dst_indexer));
    });

    return repeat_ev;
}

template <typename fnT, typename T>
struct RepeatScalar1DFactory
{
    fnT get()
    {
        fnT fn = repeat_by_scalar_1d_impl<T>;
        return fn;
    }
};

} // namespace dpctl::tensor::kernels::repeat
