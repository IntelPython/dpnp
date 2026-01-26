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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "dpnp4pybind11.hpp"

#include "histogram_common.hpp"

// utils extension header
#include "ext/validation_utils.hpp"

// dpctl tensor headers
#include "utils/type_dispatch.hpp"

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
using dpctl::tensor::usm_ndarray;
using dpctl_td_ns::typenum_t;

using ext::common::CeilDiv;

using ext::validation::array_names;
using ext::validation::array_ptr;

using ext::validation::check_max_dims;
using ext::validation::check_num_dims;
using ext::validation::check_size_at_least;
using ext::validation::common_checks;
using ext::validation::name_of;

namespace statistics::histogram
{
void validate(const usm_ndarray &sample,
              const std::optional<const dpctl::tensor::usm_ndarray> &bins,
              const std::optional<const dpctl::tensor::usm_ndarray> &weights,
              const usm_ndarray &histogram)
{
    auto exec_q = sample.get_queue();

    std::vector<array_ptr> arrays{&sample, &histogram};
    array_names names = {{arrays[0], "sample"}, {arrays[1], "histogram"}};

    array_ptr bins_ptr = nullptr;

    if (bins.has_value()) {
        bins_ptr = &bins.value();
        arrays.push_back(bins_ptr);
        names.insert({bins_ptr, "bins"});
    }

    array_ptr weights_ptr = nullptr;

    if (weights.has_value()) {
        weights_ptr = &weights.value();
        arrays.push_back(weights_ptr);
        names.insert({weights_ptr, "weights"});
    }

    common_checks({&sample, bins.has_value() ? &bins.value() : nullptr,
                   weights.has_value() ? &weights.value() : nullptr},
                  {&histogram}, names);

    check_size_at_least(bins_ptr, 2, names);
    check_size_at_least(&histogram, 1, names);

    if (weights_ptr) {
        check_num_dims(weights_ptr, 1, names);

        auto sample_size = sample.get_shape(0);
        auto weights_size = weights_ptr->get_size();
        if (sample_size != weights_ptr->get_size()) {
            throw py::value_error(name_of(&sample, names) + " size (" +
                                  std::to_string(sample_size) + ") and " +
                                  name_of(weights_ptr, names) + " size (" +
                                  std::to_string(weights_size) + ")" +
                                  " must match");
        }
    }

    check_max_dims(&sample, 2, names);

    if (sample.get_ndim() == 1) {
        check_num_dims(bins_ptr, 1, names);

        if (bins_ptr && histogram.get_size() != bins_ptr->get_size() - 1) {
            auto hist_size = histogram.get_size();
            auto bins_size = bins_ptr->get_size();
            throw py::value_error(
                name_of(&histogram, names) + " parameter and " +
                name_of(bins_ptr, names) + " parameters shape mismatch. " +
                name_of(&histogram, names) + " size is " +
                std::to_string(hist_size) + name_of(bins_ptr, names) +
                " must have size " + std::to_string(hist_size + 1) +
                " but have " + std::to_string(bins_size));
        }
    }
    else if (sample.get_ndim() == 2) {
        auto sample_count = sample.get_shape(0);
        auto expected_dims = sample.get_shape(1);

        if (histogram.get_ndim() != expected_dims) {
            throw py::value_error(
                name_of(&sample, names) + " parameter has shape (" +
                std::to_string(sample_count) + ", " +
                std::to_string(expected_dims) + ")" + ", so " +
                name_of(&histogram, names) + " parameter expected to be " +
                std::to_string(expected_dims) +
                "d. "
                "Actual " +
                std::to_string(histogram.get_ndim()) + "d");
        }

        if (bins_ptr != nullptr) {
            py::ssize_t expected_bins_size = 0;
            for (int i = 0; i < histogram.get_ndim(); ++i) {
                expected_bins_size += histogram.get_shape(i) + 1;
            }

            auto actual_bins_size = bins_ptr->get_size();
            if (actual_bins_size != expected_bins_size) {
                throw py::value_error(
                    name_of(&histogram, names) + " and " +
                    name_of(bins_ptr, names) + " shape mismatch. " +
                    name_of(bins_ptr, names) + " expected to have size = " +
                    std::to_string(expected_bins_size) + ". Actual " +
                    std::to_string(actual_bins_size));
            }
        }

        int64_t max_hist_size = std::numeric_limits<uint32_t>::max() - 1;
        if (histogram.get_size() > max_hist_size) {
            throw py::value_error(name_of(&histogram, names) +
                                  " parameter size expected to be less than " +
                                  std::to_string(max_hist_size) + ". Actual " +
                                  std::to_string(histogram.get_size()));
        }

        auto array_types = dpctl_td_ns::usm_ndarray_types();
        auto hist_type = static_cast<typenum_t>(
            array_types.typenum_to_lookup_id(histogram.get_typenum()));
        if (histogram.get_elemsize() == 8 && hist_type != typenum_t::CFLOAT) {
            auto device = exec_q.get_device();
            bool _64bit_atomics = device.has(sycl::aspect::atomic64);

            if (!_64bit_atomics) {
                auto device_name = device.get_info<sycl::info::device::name>();
                throw py::value_error(
                    name_of(&histogram, names) +
                    " parameter has 64-bit type, but 64-bit atomics " +
                    " are not supported for " + device_name);
            }
        }
    }
}

uint32_t get_local_hist_copies_count(uint32_t loc_mem_size_in_items,
                                     uint32_t local_size,
                                     uint32_t hist_size_in_items)
{
    constexpr uint32_t local_copies_limit = 16;
    constexpr uint32_t atomics_per_work_item = 4;

    const uint32_t preferred_local_copies =
        CeilDiv(atomics_per_work_item * local_size, hist_size_in_items);
    const uint32_t local_copies_fit_memory =
        loc_mem_size_in_items / hist_size_in_items;

    uint32_t local_hist_count =
        std::min(preferred_local_copies, local_copies_limit);
    local_hist_count = std::min(local_hist_count, local_copies_fit_memory);

    return local_hist_count;
}

} // namespace statistics::histogram
