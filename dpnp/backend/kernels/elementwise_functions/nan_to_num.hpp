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
#include "kernels/dpctl_tensor_types.hpp"
#include "utils/offset_utils.hpp"
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

        using dpctl::tensor::type_utils::is_complex;
        if constexpr (is_complex<T>::value) {
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

private:
    const T *inp_ = nullptr;
    T *out_ = nullptr;
    const InOutIndexerT inp_out_indexer_;
    const scT nan_;
    const scT posinf_;
    const scT neginf_;
};

template <typename T>
class NanToNumKernel;

template <typename T, typename scT>
sycl::event nan_to_num_impl(sycl::queue &q,
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

        using KernelName = NanToNumKernel<T>;
        cgh.parallel_for<KernelName>(
            {nelems}, NanToNumFunctor<T, scT, InOutIndexerT>(
                          in_tp, out_tp, indexer, nan, posinf, neginf));
    });
    return comp_ev;
}

template <typename T>
class NanToNumContigKernel;

template <typename T, typename scT>
sycl::event nan_to_num_contig_impl(sycl::queue &q,
                                   const size_t nelems,
                                   const scT nan,
                                   const scT posinf,
                                   const scT neginf,
                                   const char *in_cp,
                                   char *out_cp,
                                   const std::vector<sycl::event> &depends)
{
    dpctl::tensor::type_utils::validate_type_for_device<T>(q);

    const T *in_tp = reinterpret_cast<const T *>(in_cp);
    T *out_tp = reinterpret_cast<T *>(out_cp);

    using dpctl::tensor::offset_utils::NoOpIndexer;
    using InOutIndexerT =
        dpctl::tensor::offset_utils::TwoOffsets_CombinedIndexer<NoOpIndexer,
                                                                NoOpIndexer>;
    constexpr NoOpIndexer in_indexer{};
    constexpr NoOpIndexer out_indexer{};
    constexpr InOutIndexerT indexer{in_indexer, out_indexer};

    sycl::event comp_ev = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(depends);

        using KernelName = NanToNumContigKernel<T>;
        cgh.parallel_for<KernelName>(
            {nelems}, NanToNumFunctor<T, scT, InOutIndexerT>(
                          in_tp, out_tp, indexer, nan, posinf, neginf));
    });
    return comp_ev;
}

} // namespace dpnp::kernels::nan_to_num
