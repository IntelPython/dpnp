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

#include <pybind11/pybind11.h>

#include <sycl/sycl.hpp>
#include <oneapi/mkl/rng/device.hpp>

// dpctl tensor headers
// #include "utils/offset_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace rng
{
namespace device
{
namespace details
{
namespace py = pybind11;

namespace mkl_rng_dev = oneapi::mkl::rng::device;

/*! @brief Functor for unary function evaluation on contiguous array */
template <typename EngineBuilderT,
          typename DataT,
          typename GaussianDistrT,
          unsigned int items_per_wi = 4,
          bool enable_sg_load = true>
struct RngContigFunctor
{
private:
    // const std::uint32_t seed_;
    EngineBuilderT engine_;
    GaussianDistrT distr_;
    DataT * const res_ = nullptr;
    const size_t nelems_;

public:

    RngContigFunctor(EngineBuilderT& engine, GaussianDistrT& distr, DataT *res, const size_t n_elems)
        : engine_(engine), distr_(distr), res_(res), nelems_(n_elems)
    {
    }

    void operator()(sycl::nd_item<1> nd_it) const
    {
        auto global_id = nd_it.get_global_id();
        
        auto sg = nd_it.get_sub_group();
        const std::uint8_t sg_size = sg.get_local_range()[0];
        const std::uint8_t max_sg_size = sg.get_max_local_range()[0];

        using EngineT = typename EngineBuilderT::EngineType;
        EngineT engine = engine_(nelems_ * global_id); // offset is questionable...

        using DistrT = typename GaussianDistrT::distr_type;
        DistrT distr = distr_();

        constexpr std::size_t vec_sz = EngineT::vec_size;

        if constexpr (enable_sg_load) {
            const size_t base = items_per_wi * vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) + sg.get_group_id()[0] * max_sg_size);

            if ((sg_size == max_sg_size) && (base + items_per_wi * vec_sz * sg_size < nelems_)) {
#pragma unroll
                for (std::uint16_t it = 0; it < items_per_wi * vec_sz; it += vec_sz) {
                    size_t offset = base + static_cast<size_t>(it) * static_cast<size_t>(sg_size);
                    auto out_multi_ptr = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(&res_[offset]);

                    sycl::vec<DataT, vec_sz> rng_val_vec = mkl_rng_dev::generate<DistrT, EngineT>(distr, engine);
                    sg.store<vec_sz>(out_multi_ptr, rng_val_vec);
                }
            }
            else {
                for (size_t offset = base + sg.get_local_id()[0]; offset < nelems_; offset += sg_size) {
                    res_[offset] = mkl_rng_dev::generate_single<DistrT, EngineT>(distr, engine);
                }
            }
        }
        else {
            size_t base = nd_it.get_global_linear_id();

            base = (base / sg_size) * sg_size * items_per_wi * vec_sz + (base % sg_size);
            for (size_t offset = base; offset < std::min(nelems_, base + sg_size * (items_per_wi * vec_sz)); offset += sg_size)
            {
                res_[offset] = mkl_rng_dev::generate_single<DistrT, EngineT>(distr, engine);
            }
        }
    }
};
} // namespace details
} // namespace device
} // namespace rng
} // namespace ext
} // namespace backend
} // namespace dpnp