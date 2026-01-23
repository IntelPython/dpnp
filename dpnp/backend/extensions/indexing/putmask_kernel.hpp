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

//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#pragma once

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>
// dpctl tensor headers
#include "kernels/alignment.hpp"
#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/sycl_complex.hpp"
#include "utils/offset_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::extensions::indexing::kernels
{
template <typename T,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
struct PutMaskContigFunctor
{
private:
    T *dst_ = nullptr;
    const std::uint8_t *mask_u8_ = nullptr;
    const T *values_ = nullptr;
    std::size_t nelems_ = 0;
    std::size_t val_size_ = 0;

public:
    PutMaskContigFunctor(T *dst,
                         const bool *mask,
                         const T *values,
                         std::size_t nelems,
                         std::size_t val_size)
        : dst_(dst), mask_u8_(reinterpret_cast<const std::uint8_t *>(mask)),
          values_(values), nelems_(nelems), val_size_(val_size)
    {
    }

    void operator()(sycl::nd_item<1> ndit) const
    {
        if (val_size_ == 0 || nelems_ == 0) {
            return;
        }

        constexpr std::uint8_t elems_per_wi = n_vecs * vec_sz;
        /* Each work-item processes vec_sz elements, contiguous in memory */
        /* NOTE: work-group size must be divisible by sub-group size */

        using dpctl::tensor::type_utils::is_complex_v;
        if constexpr (enable_sg_loadstore && !is_complex_v<T>) {
            auto sg = ndit.get_sub_group();
            const std::uint32_t sgSize = sg.get_max_local_range()[0];
            const std::size_t lane_id = sg.get_local_id()[0];

            const std::size_t base =
                elems_per_wi * (ndit.get_group(0) * ndit.get_local_range(0) +
                                sg.get_group_id()[0] * sgSize);

            const bool values_no_repeat = (val_size_ >= nelems_);

            if (base + elems_per_wi * sgSize <= nelems_) {
                using dpctl::tensor::sycl_utils::sub_group_load;
                using dpctl::tensor::sycl_utils::sub_group_store;

#pragma unroll
                for (std::uint8_t it = 0; it < elems_per_wi; it += vec_sz) {
                    const std::size_t offset = base + it * sgSize;

                    auto dst_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&dst_[offset]);
                    auto mask_multi_ptr = sycl::address_space_cast<
                        sycl::access::address_space::global_space,
                        sycl::access::decorated::yes>(&mask_u8_[offset]);

                    const sycl::vec<T, vec_sz> dst_vec =
                        sub_group_load<vec_sz>(sg, dst_multi_ptr);
                    const sycl::vec<std::uint8_t, vec_sz> mask_vec =
                        sub_group_load<vec_sz>(sg, mask_multi_ptr);

                    sycl::vec<T, vec_sz> val_vec;

                    if (values_no_repeat) {
                        auto values_multi_ptr = sycl::address_space_cast<
                            sycl::access::address_space::global_space,
                            sycl::access::decorated::yes>(&values_[offset]);

                        val_vec = sub_group_load<vec_sz>(sg, values_multi_ptr);
                    }
                    else {
                        const std::size_t idx = offset + lane_id;
#pragma unroll
                        for (std::uint8_t k = 0; k < vec_sz; ++k) {
                            const std::size_t g =
                                idx + static_cast<std::size_t>(k) * sgSize;
                            val_vec[k] = values_[g % val_size_];
                        }
                    }

                    sycl::vec<T, vec_sz> out_vec;
#pragma unroll
                    for (std::uint8_t vec_id = 0; vec_id < vec_sz; ++vec_id) {
                        out_vec[vec_id] =
                            (mask_vec[vec_id] != static_cast<std::uint8_t>(0))
                                ? val_vec[vec_id]
                                : dst_vec[vec_id];
                    }

                    sub_group_store<vec_sz>(sg, out_vec, dst_multi_ptr);
                }
            }
            else {
                const std::size_t lane_id = sg.get_local_id()[0];
                for (std::size_t k = base + lane_id; k < nelems_; k += sgSize) {
                    if (mask_u8_[k]) {
                        const std::size_t v =
                            values_no_repeat ? k : (k % val_size_);
                        dst_[k] = values_[v];
                    }
                }
            }
        }
        else {
            const std::size_t gid = ndit.get_global_linear_id();
            const std::size_t gws = ndit.get_global_range(0);

            const bool values_no_repeat = (val_size_ >= nelems_);
            for (std::size_t offset = gid; offset < nelems_; offset += gws) {
                if (mask_u8_[offset]) {
                    const std::size_t v =
                        values_no_repeat ? offset : (offset % val_size_);
                    dst_[offset] = values_[v];
                }
            }
        }
    }
};

template <typename T, std::uint8_t vec_sz = 4u, std::uint8_t n_vecs = 2u>
sycl::event putmask_contig_impl(sycl::queue &exec_q,
                                std::size_t nelems,
                                char *dst_cp,
                                const char *mask_cp,
                                const char *values_cp,
                                std::size_t values_size,
                                const std::vector<sycl::event> &depends = {})
{
    T *dst_tp = reinterpret_cast<T *>(dst_cp);
    const bool *mask_tp = reinterpret_cast<const bool *>(mask_cp);
    const T *values_tp = reinterpret_cast<const T *>(values_cp);

    constexpr std::uint8_t elems_per_wi = n_vecs * vec_sz;
    // const std::size_t n_work_items_needed = (nelems + elems_per_wi - 1) /
    // elems_per_wi;
    const std::size_t n_work_items_needed = nelems / elems_per_wi;
    const std::size_t empirical_threshold = std::size_t(1) << 21;
    const std::size_t lws = (n_work_items_needed <= empirical_threshold)
                                ? std::size_t(128)
                                : std::size_t(256);

    const std::size_t n_groups =
        ((nelems + lws * elems_per_wi - 1) / (lws * elems_per_wi));
    const auto gws_range = sycl::range<1>(n_groups * lws);
    const auto lws_range = sycl::range<1>(lws);

    using dpctl::tensor::kernels::alignment_utils::is_aligned;
    using dpctl::tensor::kernels::alignment_utils::required_alignment;

    const bool aligned = is_aligned<required_alignment>(dst_tp) &&
                         is_aligned<required_alignment>(mask_tp) &&
                         is_aligned<required_alignment>(values_tp);

    sycl::event comp_ev = exec_q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        if (aligned) {
            constexpr bool enable_sg = true;
            using PutMaskFunc =
                PutMaskContigFunctor<T, vec_sz, n_vecs, enable_sg>;

            cgh.parallel_for<PutMaskFunc>(
                sycl::nd_range<1>(gws_range, lws_range),
                PutMaskFunc(dst_tp, mask_tp, values_tp, nelems, values_size));
        }
        else {
            constexpr bool enable_sg = false;
            using PutMaskFunc =
                PutMaskContigFunctor<T, vec_sz, n_vecs, enable_sg>;

            cgh.parallel_for<PutMaskFunc>(
                sycl::nd_range<1>(gws_range, lws_range),
                PutMaskFunc(dst_tp, mask_tp, values_tp, nelems, values_size));
        }
    });

    return comp_ev;
}

} // namespace dpnp::extensions::indexing::kernels
