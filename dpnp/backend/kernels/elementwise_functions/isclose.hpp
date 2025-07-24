//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include <sycl/sycl.hpp>
// dpctl tensor headers
#include "kernels/alignment.hpp"
#include "kernels/dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::kernels::isclose
{

template <typename T>
inline bool isclose(const T a,
                    const T b,
                    const T rtol,
                    const T atol,
                    const bool equal_nan)
{
    if (sycl::isnan(a) || sycl::isnan(b)) {
        // static cast<T>?
        return equal_nan && sycl::isnan(a) && sycl::isnan(b);
    }
    if (sycl::isinf(a) || sycl::isinf(b)) {
        return a == b;
    }
    return sycl::fabs(a - b) <= atol + rtol * sycl::fabs(b);
}

template <typename T,
          typename scT,
          typename resTy,
          typename ThreeOffsets_IndexerT>
struct IsCloseStridedScalarFunctor
{
private:
    const T *a_ = nullptr;
    const T *b_ = nullptr;
    resTy *out_ = nullptr;
    const ThreeOffsets_IndexerT three_offsets_indexer_;
    const scT rtol_;
    const scT atol_;
    const bool equal_nan_;

public:
    IsCloseStridedScalarFunctor(const T *a,
                                const T *b,
                                resTy *out,
                                const ThreeOffsets_IndexerT &inps_res_indexer,
                                const scT rtol,
                                const scT atol,
                                const bool equal_nan)
        : a_(a), b_(b), out_(out), three_offsets_indexer_(inps_res_indexer),
          rtol_(rtol), atol_(atol), equal_nan_(equal_nan)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &three_offsets_ = three_offsets_indexer_(wid.get(0));
        const dpctl::tensor::ssize_t &inp1_offset =
            three_offsets_.get_first_offset();
        const dpctl::tensor::ssize_t &inp2_offset =
            three_offsets_.get_second_offset();
        const dpctl::tensor::ssize_t &out_offset =
            three_offsets_.get_third_offset();

        using dpctl::tensor::type_utils::is_complex_v;
        if constexpr (is_complex_v<T>) {
            T z_a = a_[inp1_offset];
            T z_b = b_[inp2_offset];
            bool x = isclose(z_a.real(), z_b.real(), rtol_, atol_, equal_nan_);
            bool y = isclose(z_a.imag(), z_b.imag(), rtol_, atol_, equal_nan_);
            out_[out_offset] = x && y;
        }
        else {
            out_[out_offset] = isclose(a_[inp1_offset], b_[inp2_offset], rtol_,
                                       atol_, equal_nan_);
        }
    }
};

template <typename T,
          typename scT,
          typename resTy,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
struct IsCloseContigScalarFunctor
{
private:
    const T *a_ = nullptr;
    const T *b_ = nullptr;
    resTy *out_ = nullptr;
    std::size_t nelems_;
    const scT rtol_;
    const scT atol_;
    const bool equal_nan_;

public:
    IsCloseContigScalarFunctor(const T *a,
                               const T *b,
                               resTy *out,
                               const std::size_t n_elems,
                               const scT rtol,
                               const scT atol,
                               const bool equal_nan)
        : a_(a), b_(b), out_(out), nelems_(n_elems), rtol_(rtol), atol_(atol),
          equal_nan_(equal_nan)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        constexpr std::uint8_t elems_per_wi = n_vecs * vec_sz;
        /* Each work-item processes vec_sz elements, contiguous in memory */
        /* NOTE: work-group size must be divisible by sub-group size */

        using dpctl::tensor::type_utils::is_complex_v;
        if constexpr (enable_sg_loadstore && !is_complex_v<T>) {
            auto sg = ndit.get_sub_group();
            const std::uint16_t sgSize = sg.get_max_local_range()[0];
            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);

            if (base + elems_per_wi * sgSize < nelems_) {
                using dpctl::tensor::sycl_utils::sub_group_load;
                using dpctl::tensor::sycl_utils::sub_group_store;
#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;
                    auto a_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&a_[offset]);
                    auto b_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&b_[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out_[offset]);

                    const sycl::vec<T, vec_sz> a_vec =
                        sub_group_load<vec_sz>(sg, a_multi_ptr);
                    const sycl::vec<T, vec_sz> b_vec =
                        sub_group_load<vec_sz>(sg, b_multi_ptr);

                    sycl::vec<resTy, vec_sz> res_vec;
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        res_vec[vec_id] = isclose(a_vec[vec_id], b_vec[vec_id],
                                                  rtol_, atol_, equal_nan_);
                    }
                    sub_group_store<vec_sz>(sg, res_vec, out_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out_[k] = isclose(a_[k], b_[k], rtol_, atol_, equal_nan_);
                }
            }
        }
        else {
            const std::uint16_t sgSize =
                ndit.get_sub_group().get_local_range()[0];
            const std::size_t gid = ndit.get_global_linear_id();
            const std::uint16_t elems_per_sg = sgSize * elems_per_wi;

            const std::size_t start =
                (gid / sgSize) * (elems_per_sg - sgSize) + gid;
            const std::size_t end = std::min(nelems_, start + elems_per_sg);
            for (std::size_t offset = start; offset < end; offset += sgSize) {
                if constexpr (is_complex_v<T>) {
                    T z_a = a_[offset];
                    T z_b = b_[offset];
                    bool x = isclose(z_a.real(), z_b.real(), rtol_, atol_,
                                     equal_nan_);
                    bool y = isclose(z_a.imag(), z_b.imag(), rtol_, atol_,
                                     equal_nan_);
                    out_[offset] = x && y;
                }
                else {
                    out_[offset] = isclose(a_[offset], b_[offset], rtol_, atol_,
                                           equal_nan_);
                }
            }
        }
    }
};

template <typename T, typename scT>
sycl::event
    isclose_strided_scalar_impl(sycl::queue &exec_q,
                                const int nd,
                                std::size_t nelems,
                                const dpctl::tensor::ssize_t *shape_strides,
                                const scT rtol,
                                const scT atol,
                                const bool equal_nan,
                                const char *a_cp,
                                const dpctl::tensor::ssize_t a_offset,
                                const char *b_cp,
                                const dpctl::tensor::ssize_t b_offset,
                                char *out_cp,
                                const dpctl::tensor::ssize_t out_offset,
                                const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<T>(exec_q);

    const T *a_tp = reinterpret_cast<const T *>(a_cp);
    const T *b_tp = reinterpret_cast<const T *>(b_cp);

    using resTy = bool;
    resTy *out_tp = reinterpret_cast<resTy *>(out_cp);

    using IndexerT =
        typename dpctl::tensor::offset_utils::ThreeOffsets_StridedIndexer;
    const IndexerT indexer{nd, a_offset, b_offset, out_offset, shape_strides};

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using IsCloseFunc =
            IsCloseStridedScalarFunctor<T, scT, resTy, IndexerT>;
        cgh.parallel_for<IsCloseFunc>(
            {nelems},
            IsCloseFunc(a_tp, b_tp, out_tp, indexer, atol, rtol, equal_nan));
    });
    return comp_ev;
}

template <typename T,
          typename scT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u>
sycl::event
    isclose_contig_scalar_impl(sycl::queue &exec_q,
                               std::size_t nelems,
                               const scT rtol,
                               const scT atol,
                               const bool equal_nan,
                               const char *a_cp,
                               ssize_t a_offset,
                               const char *b_cp,
                               ssize_t b_offset,
                               char *out_cp,
                               ssize_t out_offset,
                               const std::vector<sycl::event> &depends = {})
{
    constexpr std::uint8_t elems_per_wi = n_vecs * vec_sz;
    const std::size_t n_work_items_needed = nelems / elems_per_wi;
    const std::size_t empirical_threshold = std::size_t(1) << 21;
    const std::size_t lws = (n_work_items_needed <= empirical_threshold)
                                ? std::size_t(128)
                                : std::size_t(256);

    const std::size_t n_groups =
        ((nelems + lws * elems_per_wi - 1) / (lws * elems_per_wi));
    const auto gws_range = sycl::range<1>(n_groups * lws);
    const auto lws_range = sycl::range<1>(lws);

    // ? + offset
    const T *a_tp = reinterpret_cast<const T *>(a_cp) + a_offset;
    const T *b_tp = reinterpret_cast<const T *>(b_cp) + b_offset;

    using resTy = bool;
    resTy *out_tp = reinterpret_cast<resTy *>(out_cp) + out_offset;

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using dpctl::tensor::kernels::alignment_utils::is_aligned;
        using dpctl::tensor::kernels::alignment_utils::required_alignment;
        if (is_aligned<required_alignment>(a_tp) &&
            is_aligned<required_alignment>(b_tp) &&
            is_aligned<required_alignment>(out_tp))
        {
            constexpr bool enable_sg_loadstore = true;
            using IsCloseFunc =
                IsCloseContigScalarFunctor<T, scT, resTy, vec_sz, n_vecs,
                                           enable_sg_loadstore>;

            cgh.parallel_for<IsCloseFunc>(
                sycl::nd_range<1>(gws_range, lws_range),
                IsCloseFunc(a_tp, b_tp, out_tp, nelems, rtol, atol, equal_nan));
        }
        else {
            constexpr bool disable_sg_loadstore = false;
            using IsCloseFunc =
                IsCloseContigScalarFunctor<T, scT, resTy, vec_sz, n_vecs,
                                           disable_sg_loadstore>;

            cgh.parallel_for<IsCloseFunc>(
                sycl::nd_range<1>(gws_range, lws_range),
                IsCloseFunc(a_tp, b_tp, out_tp, nelems, rtol, atol, equal_nan));
        }
    });

    return comp_ev;
}

} // namespace dpnp::kernels::isclose
