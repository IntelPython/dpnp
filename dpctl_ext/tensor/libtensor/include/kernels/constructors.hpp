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
#include <complex>
#include <cstddef>

#include <sycl/sycl.hpp>

#include "dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/strided_iters.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace constructors
{

using dpctl::tensor::ssize_t;

/*!
  @defgroup CtorKernels
 */

template <typename Ty>
class full_strided_kernel;

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

} // namespace constructors
} // namespace kernels
} // namespace tensor
} // namespace dpctl
