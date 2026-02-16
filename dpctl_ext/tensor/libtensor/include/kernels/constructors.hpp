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
/// This file defines kernels for tensor constructors.
//===----------------------------------------------------------------------===//

#pragma once
#include <array>
#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/strided_iters.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::constructors
{

using dpctl::tensor::ssize_t;

/*!
  @defgroup CtorKernels
 */

template <typename Ty>
class full_strided_kernel;
// template <typename Ty> class eye_kernel;

using namespace dpctl::tensor::offset_utils;

/* ================ Full ================== */

/*!
 * @brief Function to submit kernel to fill given contiguous memory allocation
 * with specified value.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Length of the sequence
 * @param fill_v  Value to fill the array with
 * @param dst_p Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename dstTy>
sycl::event full_contig_impl(sycl::queue &q,
                             std::size_t nelems,
                             dstTy fill_v,
                             char *dst_p,
                             const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<dstTy>(q);
    sycl::event fill_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        dstTy *p = reinterpret_cast<dstTy *>(dst_p);
        cgh.fill<dstTy>(p, fill_v, nelems);
    });

    return fill_ev;
}

template <typename Ty, typename IndexerT>
class FullStridedFunctor
{
private:
    Ty *p = nullptr;
    Ty fill_v;
    IndexerT indexer;

public:
    FullStridedFunctor(Ty *p_, const Ty &fill_v_, const IndexerT &indexer_)
        : p(p_), fill_v(fill_v_), indexer(indexer_)
    {
    }

    void operator()(sycl::id<1> id) const
    {
        auto offset = indexer(id.get(0));
        p[offset] = fill_v;
    }
};

/*!
 * @brief Function to submit kernel to fill given contiguous memory allocation
 * with specified value.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nd  Array dimensionality
 * @param nelems  Length of the sequence
 * @param shape_strides  Kernel accessible USM pointer to packed shape and
 * strides of array.
 * @param fill_v  Value to fill the array with
 * @param dst_p  Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename dstTy>
sycl::event full_strided_impl(sycl::queue &q,
                              int nd,
                              std::size_t nelems,
                              const ssize_t *shape_strides,
                              dstTy fill_v,
                              char *dst_p,
                              const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<dstTy>(q);

    dstTy *dst_tp = reinterpret_cast<dstTy *>(dst_p);

    using dpctl::tensor::offset_utils::StridedIndexer;
    const StridedIndexer strided_indexer(nd, 0, shape_strides);

    sycl::event fill_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using KernelName = full_strided_kernel<dstTy>;
        using Impl = FullStridedFunctor<dstTy, StridedIndexer>;

        cgh.parallel_for<KernelName>(sycl::range<1>{nelems},
                                     Impl(dst_tp, fill_v, strided_indexer));
    });

    return fill_ev;
}

/* =========================== Tril and triu ============================== */

// define function type
typedef sycl::event (*tri_fn_ptr_t)(sycl::queue &,
                                    ssize_t,   // inner_range  //ssize_t
                                    ssize_t,   // outer_range
                                    char *,    // src_data_ptr
                                    char *,    // dst_data_ptr
                                    ssize_t,   // nd
                                    ssize_t *, // shape_and_strides
                                    ssize_t,   // k
                                    const std::vector<sycl::event> &,
                                    const std::vector<sycl::event> &);

/*!
 * @brief Function to copy triangular matrices from source stack to destination
 * stack.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param inner_range  Number of elements in each matrix.
 * @param outer_range  Number of matrices to copy.
 * @param src_p  Kernel accessible USM pointer for the source array.
 * @param dst_p  Kernel accessible USM pointer for the destination array.
 * @param nd  The array dimensionality of source and destination arrays.
 * @param shape_and_strides  Kernel accessible USM pointer to packed shape and
 * strides of arrays.
 * @param k Position of the diagonal above/below which to copy filling the rest
 * with zero elements.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 * @param additional_depends  List of additional events to wait for before
 * starting computations, if any.
 *
 * @return  Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty, bool>
class tri_kernel;
template <typename Ty, bool upper>
sycl::event tri_impl(sycl::queue &exec_q,
                     ssize_t inner_range,
                     ssize_t outer_range,
                     char *src_p,
                     char *dst_p,
                     ssize_t nd,
                     ssize_t *shape_and_strides,
                     ssize_t k,
                     const std::vector<sycl::event> &depends,
                     const std::vector<sycl::event> &additional_depends)
{
    static constexpr int d2 = 2;
    ssize_t src_s = nd;
    ssize_t dst_s = 2 * nd;
    ssize_t nd_1 = nd - 1;
    ssize_t nd_2 = nd - 2;
    Ty *src = reinterpret_cast<Ty *>(src_p);
    Ty *dst = reinterpret_cast<Ty *>(dst_p);

    dpctl::tensor::type_utils::validate_type_for_device<Ty>(exec_q);

    sycl::event tri_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.depends_on(additional_depends);

        cgh.parallel_for<tri_kernel<Ty, upper>>(
            sycl::range<1>(inner_range * outer_range), [=](sycl::id<1> idx) {
                ssize_t outer_gid = idx[0] / inner_range;
                ssize_t inner_gid = idx[0] - inner_range * outer_gid;

                ssize_t src_inner_offset = 0, dst_inner_offset = 0;
                bool to_copy{false};

                {
                    using dpctl::tensor::strides::CIndexer_array;
                    CIndexer_array<d2, ssize_t> indexer_i(
                        {shape_and_strides[nd_2], shape_and_strides[nd_1]});
                    indexer_i.set(inner_gid);
                    const std::array<ssize_t, d2> &inner = indexer_i.get();
                    src_inner_offset =
                        inner[0] * shape_and_strides[src_s + nd_2] +
                        inner[1] * shape_and_strides[src_s + nd_1];
                    dst_inner_offset =
                        inner[0] * shape_and_strides[dst_s + nd_2] +
                        inner[1] * shape_and_strides[dst_s + nd_1];

                    if constexpr (upper)
                        to_copy = (inner[0] + k >= inner[1]);
                    else
                        to_copy = (inner[0] + k <= inner[1]);
                }

                ssize_t src_offset = 0;
                ssize_t dst_offset = 0;
                {
                    using dpctl::tensor::strides::CIndexer_vector;
                    CIndexer_vector<ssize_t> outer(nd - d2);
                    outer.get_displacement(
                        outer_gid, shape_and_strides, shape_and_strides + src_s,
                        shape_and_strides + dst_s, src_offset, dst_offset);
                }

                src_offset += src_inner_offset;
                dst_offset += dst_inner_offset;

                dst[dst_offset] = (to_copy) ? src[src_offset] : Ty(0);
            });
    });
    return tri_ev;
}

/*!
 * @brief  Factory to get function pointer of type `fnT` for data type `Ty`.
 * @ingroup CtorKernels
 */
template <typename fnT, typename Ty>
struct TrilGenericFactory
{
    fnT get()
    {
        fnT f = tri_impl<Ty, /*tril*/ true>;
        return f;
    }
};

/*!
 * @brief  Factory to get function pointer of type `fnT` for data type `Ty`.
 * @ingroup CtorKernels
 */
template <typename fnT, typename Ty>
struct TriuGenericFactory
{
    fnT get()
    {
        fnT f = tri_impl<Ty, /*triu*/ false>;
        return f;
    }
};

} // namespace dpctl::tensor::kernels::constructors
