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

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include <dpctl4pybind11.hpp>

namespace dpnp::extensions::fft
{
namespace mkl_dft = oneapi::mkl::dft;
namespace py = pybind11;

template <mkl_dft::precision prec>
class ComplexDescriptorWrapper
{
public:
    using descr_type = mkl_dft::descriptor<prec, mkl_dft::domain::COMPLEX>;

    ComplexDescriptorWrapper(std::int64_t n) : descr_(n), queue_ptr_{} {}
    ComplexDescriptorWrapper(std::vector<std::int64_t> dimensions)
        : descr_(dimensions), queue_ptr_{}
    {
    }
    ~ComplexDescriptorWrapper() {}

    void commit(sycl::queue &q)
    {
        mkl_dft::precision fft_prec = get_precision();
        if (fft_prec == mkl_dft::precision::DOUBLE &&
            !q.get_device().has(sycl::aspect::fp64))
        {
            throw py::value_error("Descriptor is double precision but the "
                                  "device does not support double precision.");
        }

        descr_.commit(q);
        queue_ptr_ = std::make_unique<sycl::queue>(q);
    }

    descr_type &get_descriptor() { return descr_; }

    const sycl::queue &get_queue() const
    {
        if (queue_ptr_) {
            return *queue_ptr_;
        }
        else {
            throw std::runtime_error(
                "Attempt to get queue when it is not yet set");
        }
    }

    // config_param::DIMENSION
    template <typename valT = std::int64_t>
    valT get_dim()
    {
        valT dim = -1;
        descr_.get_value(mkl_dft::config_param::DIMENSION, &dim);

        return dim;
    }

    // config_param::NUMBER_OF_TRANSFORMS
    template <typename valT = std::int64_t>
    valT get_number_of_transforms()
    {
        valT transforms_count{};

        descr_.get_value(mkl_dft::config_param::NUMBER_OF_TRANSFORMS,
                         &transforms_count);
        return transforms_count;
    }

    template <typename valT = std::int64_t>
    void set_number_of_transforms(const valT num)
    {
        descr_.set_value(mkl_dft::config_param::NUMBER_OF_TRANSFORMS, num);
    }

    // config_param::FWD_STRIDES
    template <typename valT = std::vector<std::int64_t>>
    valT get_fwd_strides()
    {
        const typename valT::value_type dim = get_dim();

        valT fwd_strides(dim + 1);
        // TODO: Replace INPUT_STRIDES with FWD_STRIDES in MKL=2024.2
        descr_.get_value(mkl_dft::config_param::INPUT_STRIDES,
                         fwd_strides.data());
        return fwd_strides;
    }

    template <typename valT = std::vector<std::int64_t>>
    void set_fwd_strides(const valT &strides)
    {
        const typename valT::value_type dim = get_dim();

        if (static_cast<size_t>(dim + 1) != strides.size()) {
            throw py::value_error(
                "Strides length does not match descriptor's dimension");
        }
        // TODO: Replace INPUT_STRIDES with FWD_STRIDES in MKL=2024.2
        descr_.set_value(mkl_dft::config_param::INPUT_STRIDES, strides.data());
    }

    // config_param::BWD_STRIDES
    template <typename valT = std::vector<std::int64_t>>
    valT get_bwd_strides()
    {
        const typename valT::value_type dim = get_dim();

        valT bwd_strides(dim + 1);
        // TODO: Replace OUTPUT_STRIDES with BWD_STRIDES in MKL=2024.2
        descr_.get_value(mkl_dft::config_param::OUTPUT_STRIDES,
                         bwd_strides.data());
        return bwd_strides;
    }

    template <typename valT = std::vector<std::int64_t>>
    void set_bwd_strides(const valT &strides)
    {
        const typename valT::value_type dim = get_dim();

        if (static_cast<size_t>(dim + 1) != strides.size()) {
            throw py::value_error(
                "Strides length does not match descriptor's dimension");
        }
        // TODO: Replace OUTPUT_STRIDES with BWD_STRIDES in MKL=2024.2
        descr_.set_value(mkl_dft::config_param::OUTPUT_STRIDES, strides.data());
    }

    // config_param::FWD_DISTANCE
    template <typename valT = std::int64_t>
    valT get_fwd_distance()
    {
        valT dist = 0;

        descr_.get_value(mkl_dft::config_param::FWD_DISTANCE, &dist);
        return dist;
    }

    template <typename valT = std::int64_t>
    void set_fwd_distance(const valT dist)
    {
        descr_.set_value(mkl_dft::config_param::FWD_DISTANCE, dist);
    }

    // config_param::BWD_DISTANCE
    template <typename valT = std::int64_t>
    valT get_bwd_distance()
    {
        valT dist = 0;

        descr_.get_value(mkl_dft::config_param::BWD_DISTANCE, &dist);
        return dist;
    }

    template <typename valT = std::int64_t>
    void set_bwd_distance(const valT dist)
    {
        descr_.set_value(mkl_dft::config_param::BWD_DISTANCE, dist);
    }

    // config_param::PLACEMENT
    bool get_in_place()
    {
        DFTI_CONFIG_VALUE placement;

        descr_.get_value(mkl_dft::config_param::PLACEMENT, &placement);
        return (placement == DFTI_CONFIG_VALUE::DFTI_INPLACE);
    }

    void set_in_place(const bool in_place_request)
    {
        descr_.set_value(mkl_dft::config_param::PLACEMENT,
                         (in_place_request)
                             ? DFTI_CONFIG_VALUE::DFTI_INPLACE
                             : DFTI_CONFIG_VALUE::DFTI_NOT_INPLACE);
    }

    // config_param::PRECISION
    mkl_dft::precision get_precision()
    {
        mkl_dft::precision fft_prec;

        descr_.get_value(mkl_dft::config_param::PRECISION, &fft_prec);
        return fft_prec;
    }

    // config_param::COMMIT_STATUS
    bool is_committed()
    {
        DFTI_CONFIG_VALUE committed;

        descr_.get_value(mkl_dft::config_param::COMMIT_STATUS, &committed);
        return (committed == DFTI_CONFIG_VALUE::DFTI_COMMITTED);
    }

private:
    mkl_dft::descriptor<prec, mkl_dft::domain::COMPLEX> descr_;
    std::unique_ptr<sycl::queue> queue_ptr_;
};

template <mkl_dft::precision prec>
std::pair<sycl::event, sycl::event>
    compute_fft(ComplexDescriptorWrapper<prec> &descr,
                const dpctl::tensor::usm_ndarray &in,
                const dpctl::tensor::usm_ndarray &out,
                const bool is_forward,
                const std::vector<sycl::event> &depends);

} // namespace dpnp::extensions::fft
