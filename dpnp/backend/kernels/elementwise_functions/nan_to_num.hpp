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

namespace dpnp::kernels::nan_to_num
{

template <typename T>
inline T to_num(const T v, const T nan, const T posinf, const T neginf)
{
    return (sycl::isnan(v))   ? nan
           : (sycl::isinf(v)) ? (v > 0) ? posinf : neginf
                              : v;
}

template <typename T, typename scT, typename InOutIndexerT>
struct NanToNumFunctor
{
private:
    const T *inp_ = nullptr;
    T *out_ = nullptr;
    const InOutIndexerT inp_out_indexer_;
    const scT nan_;
    const scT posinf_;
    const scT neginf_;

public:
    NanToNumFunctor(const T *inp,
                    T *out,
                    const InOutIndexerT &inp_out_indexer,
                    const scT nan,
                    const scT posinf,
                    const scT neginf)
        : inp_(inp), out_(out), inp_out_indexer_(inp_out_indexer), nan_(nan),
          posinf_(posinf), neginf_(neginf)
    {
    }

    void operator()(sycl::id<1> wid) const
    {
        const auto &offsets_ = inp_out_indexer_(wid.get(0));
        const dpctl::tensor::ssize_t &inp_offset = offsets_.get_first_offset();
        const dpctl::tensor::ssize_t &out_offset = offsets_.get_second_offset();

        using dpctl::tensor::type_utils::is_complex_v;
        if constexpr (is_complex_v<T>) {
            using realT = typename T::value_type;
            static_assert(std::is_same_v<realT, scT>);
            T z = inp_[inp_offset];
            realT x = to_num(z.real(), nan_, posinf_, neginf_);
            realT y = to_num(z.imag(), nan_, posinf_, neginf_);
            out_[out_offset] = T{x, y};
        }
        else {
            out_[out_offset] = to_num(inp_[inp_offset], nan_, posinf_, neginf_);
        }
    }
};

template <typename T,
          typename scT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
struct NanToNumContigFunctor
{
private:
    const T *in_ = nullptr;
    T *out_ = nullptr;
    std::size_t nelems_;
    const scT nan_;
    const scT posinf_;
    const scT neginf_;

public:
    NanToNumContigFunctor(const T *in,
                          T *out,
                          const std::size_t n_elems,
                          const scT nan,
                          const scT posinf,
                          const scT neginf)
        : in_(in), out_(out), nelems_(n_elems), nan_(nan), posinf_(posinf),
          neginf_(neginf)
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
                    auto in_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&in_[offset]);
                    auto out_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&out_[offset]);

                    sycl::vec<T, vec_sz> arg_vec =
                        sub_group_load<vec_sz>(sg, in_multi_ptr);
#pragma unroll
                    for (std::uint32_t k = 0; k < vec_sz; ++k) {
                        arg_vec[k] = to_num(arg_vec[k], nan_, posinf_, neginf_);
                    }
                    sub_group_store<vec_sz>(sg, arg_vec, out_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    out_[k] = to_num(in_[k], nan_, posinf_, neginf_);
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
                    using realT = typename T::value_type;
                    static_assert(std::is_same_v<realT, scT>);

                    T z = in_[offset];
                    realT x = to_num(z.real(), nan_, posinf_, neginf_);
                    realT y = to_num(z.imag(), nan_, posinf_, neginf_);
                    out_[offset] = T{x, y};
                }
                else {
                    out_[offset] = to_num(in_[offset], nan_, posinf_, neginf_);
                }
            }
        }
    }
};

template <typename T, typename scT>
sycl::event nan_to_num_strided_impl(sycl::queue &q,
                                    const size_t nelems,
                                    const int nd,
                                    const dpctl::tensor::ssize_t *shape_strides,
                                    const scT nan,
                                    const scT posinf,
                                    const scT neginf,
                                    const char *in_cp,
                                    const dpctl::tensor::ssize_t in_offset,
                                    char *out_cp,
                                    const dpctl::tensor::ssize_t out_offset,
                                    const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<T>(q);

    const T *in_tp = reinterpret_cast<const T *>(in_cp);
    T *out_tp = reinterpret_cast<T *>(out_cp);

    using InOutIndexerT =
        typename dpctl::tensor::offset_utils::TwoOffsets_StridedIndexer;
    const InOutIndexerT indexer{nd, in_offset, out_offset, shape_strides};

    sycl::event comp_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using NanToNumFunc = NanToNumFunctor<T, scT, InOutIndexerT>;
        cgh.parallel_for<NanToNumFunc>(
            {nelems},
            NanToNumFunc(in_tp, out_tp, indexer, nan, posinf, neginf));
    });
    return comp_ev;
}

template <typename T,
          typename scT,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u>
sycl::event nan_to_num_contig_impl(sycl::queue &exec_q,
                                   std::size_t nelems,
                                   const scT nan,
                                   const scT posinf,
                                   const scT neginf,
                                   const char *in_cp,
                                   char *out_cp,
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

    const T *in_tp = reinterpret_cast<const T *>(in_cp);
    T *out_tp = reinterpret_cast<T *>(out_cp);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using dpctl::tensor::kernels::alignment_utils::is_aligned;
        using dpctl::tensor::kernels::alignment_utils::required_alignment;
        if (is_aligned<required_alignment>(in_tp) &&
            is_aligned<required_alignment>(out_tp))
        {
            constexpr bool enable_sg_loadstore = true;
            using NanToNumFunc = NanToNumContigFunctor<T, scT, vec_sz, n_vecs,
                                                       enable_sg_loadstore>;

            cgh.parallel_for<NanToNumFunc>(
                sycl::nd_range<1>(gws_range, lws_range),
                NanToNumFunc(in_tp, out_tp, nelems, nan, posinf, neginf));
        }
        else {
            constexpr bool disable_sg_loadstore = false;
            using NanToNumFunc = NanToNumContigFunctor<T, scT, vec_sz, n_vecs,
                                                       disable_sg_loadstore>;

            cgh.parallel_for<NanToNumFunc>(
                sycl::nd_range<1>(gws_range, lws_range),
                NanToNumFunc(in_tp, out_tp, nelems, nan, posinf, neginf));
        }
    });

    return comp_ev;
}

} // namespace dpnp::kernels::nan_to_num
