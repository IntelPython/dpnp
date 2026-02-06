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
class linear_sequence_step_kernel;
template <typename Ty, typename wTy>
class linear_sequence_affine_kernel;
template <typename Ty>
class full_strided_kernel;
// template <typename Ty> class eye_kernel;

using namespace dpctl::tensor::offset_utils;

template <typename Ty>
class LinearSequenceStepFunctor
{
private:
    Ty *p = nullptr;
    Ty start_v;
    Ty step_v;

public:
    LinearSequenceStepFunctor(char *dst_p, Ty v0, Ty dv)
        : p(reinterpret_cast<Ty *>(dst_p)), start_v(v0), step_v(dv)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<Ty>::value) {
            p[i] = Ty{start_v.real() + i * step_v.real(),
                      start_v.imag() + i * step_v.imag()};
        }
        else {
            p[i] = start_v + i * step_v;
        }
    }
};

/*!
 * @brief Function to submit kernel to populate given contiguous memory
 * allocation with linear sequence specified by typed starting value and
 * increment.
 *
 * @param q  Sycl queue to which the kernel is submitted
 * @param nelems Length of the sequence
 * @param start_v Typed starting value of the sequence
 * @param step_v  Typed increment of the sequence
 * @param array_data Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty>
sycl::event lin_space_step_impl(sycl::queue &exec_q,
                                std::size_t nelems,
                                Ty start_v,
                                Ty step_v,
                                char *array_data,
                                const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(exec_q);
    sycl::event lin_space_step_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        cgh.parallel_for<linear_sequence_step_kernel<Ty>>(
            sycl::range<1>{nelems},
            LinearSequenceStepFunctor<Ty>(array_data, start_v, step_v));
    });

    return lin_space_step_event;
}

// Constructor to populate tensor with linear sequence defined by
// start and and data

template <typename Ty, typename wTy>
class LinearSequenceAffineFunctor
{
private:
    Ty *p = nullptr;
    Ty start_v;
    Ty end_v;
    std::size_t n;

public:
    LinearSequenceAffineFunctor(char *dst_p, Ty v0, Ty v1, std::size_t den)
        : p(reinterpret_cast<Ty *>(dst_p)), start_v(v0), end_v(v1),
          n((den == 0) ? 1 : den)
    {
    }

    void operator()(sycl::id<1> wiid) const
    {
        auto i = wiid.get(0);
        wTy wc = wTy(i) / n;
        wTy w = wTy(n - i) / n;
        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<Ty>::value) {
            using reT = typename Ty::value_type;
            auto _w = static_cast<reT>(w);
            auto _wc = static_cast<reT>(wc);
            auto re_comb = sycl::fma(start_v.real(), _w, reT(0));
            re_comb =
                sycl::fma(end_v.real(), _wc,
                          re_comb); // start_v.real() * _w + end_v.real() * _wc;
            auto im_comb =
                sycl::fma(start_v.imag(), _w,
                          reT(0)); // start_v.imag() * _w + end_v.imag() * _wc;
            im_comb = sycl::fma(end_v.imag(), _wc, im_comb);
            Ty affine_comb = Ty{re_comb, im_comb};
            p[i] = affine_comb;
        }
        else if constexpr (std::is_floating_point<Ty>::value) {
            Ty _w = static_cast<Ty>(w);
            Ty _wc = static_cast<Ty>(wc);
            auto affine_comb =
                sycl::fma(start_v, _w, Ty(0)); // start_v * w + end_v * wc;
            affine_comb = sycl::fma(end_v, _wc, affine_comb);
            p[i] = affine_comb;
        }
        else {
            using dpctl::tensor::type_utils::convert_impl;
            auto affine_comb = start_v * w + end_v * wc;
            p[i] = convert_impl<Ty, decltype(affine_comb)>(affine_comb);
        }
    }
};

/*!
 * @brief Function to submit kernel to populate given contiguous memory
 * allocation with linear sequence specified by typed starting and end values.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Length of the sequence.
 * @param start_v Stating value of the sequence.
 * @param end_v   End-value of the sequence.
 * @param include_endpoint  Whether the end-value is included in the sequence.
 * @param array_data Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty>
sycl::event lin_space_affine_impl(sycl::queue &exec_q,
                                  std::size_t nelems,
                                  Ty start_v,
                                  Ty end_v,
                                  bool include_endpoint,
                                  char *array_data,
                                  const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<Ty>(exec_q);

    const bool device_supports_doubles =
        exec_q.get_device().has(sycl::aspect::fp64);
    const std::size_t den = (include_endpoint) ? nelems - 1 : nelems;

    sycl::event lin_space_affine_event = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);
        if (device_supports_doubles) {
            using KernelName = linear_sequence_affine_kernel<Ty, double>;
            using Impl = LinearSequenceAffineFunctor<Ty, double>;

            cgh.parallel_for<KernelName>(sycl::range<1>{nelems},
                                         Impl(array_data, start_v, end_v, den));
        }
        else {
            using KernelName = linear_sequence_affine_kernel<Ty, float>;
            using Impl = LinearSequenceAffineFunctor<Ty, float>;

            cgh.parallel_for<KernelName>(sycl::range<1>{nelems},
                                         Impl(array_data, start_v, end_v, den));
        }
    });

    return lin_space_affine_event;
}

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
