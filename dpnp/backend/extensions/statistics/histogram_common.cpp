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

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "dpctl4pybind11.hpp"
#include "utils/type_dispatch.hpp"
#include <pybind11/pybind11.h>

#include "histogram_common.hpp"

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
using dpctl::tensor::usm_ndarray;
using dpctl_td_ns::typenum_t;

namespace statistics
{
namespace histogram
{

void validate(const usm_ndarray &sample,
              const usm_ndarray &bins,
              std::optional<const dpctl::tensor::usm_ndarray> &weights,
              const usm_ndarray &histogram)
{
    auto exec_q = sample.get_queue();
    using array_ptr = const usm_ndarray *;

    std::vector<array_ptr> arrays{&sample, &bins, &histogram};
    std::unordered_map<array_ptr, std::string> names = {
        {arrays[0], "sample"}, {arrays[1], "bins"}, {arrays[2], "histogram"}};

    array_ptr weights_ptr = nullptr;

    if (weights.has_value()) {
        weights_ptr = &weights.value();
        arrays.push_back(weights_ptr);
        names.insert({weights_ptr, "weights"});
    }

    auto get_name = [&](const array_ptr &arr) {
        auto name_it = names.find(arr);
        assert(name_it != names.end());

        return "'" + name_it->second + "'";
    };

    auto unequal_queue =
        std::find_if(arrays.cbegin(), arrays.cend(), [&](const array_ptr &arr) {
            return arr->get_queue() != exec_q;
        });

    if (unequal_queue != arrays.cend()) {
        throw py::value_error(
            get_name(*unequal_queue) +
            " parameter has incompatible queue with parameter " +
            get_name(&sample));
    }

    auto non_contig_array =
        std::find_if(arrays.cbegin(), arrays.cend(), [&](const array_ptr &arr) {
            return !arr->is_c_contiguous();
        });

    if (non_contig_array != arrays.cend()) {
        throw py::value_error(get_name(*non_contig_array) +
                              " parameter is not c-contiguos");
    }

    if (bins.get_size() < 2) {
        throw py::value_error(get_name(&bins) +
                              " parameter must have at least 2 elements");
    }

    if (histogram.get_size() < 1) {
        throw py::value_error(get_name(&histogram) +
                              " parameter must have at least 1 element");
    }

    if (histogram.get_ndim() != 1) {
        throw py::value_error(get_name(&histogram) +
                              " parameter must be 1d. Actual " +
                              std::to_string(histogram.get_ndim()) + "d");
    }

    if (weights_ptr) {
        if (weights_ptr->get_ndim() != 1) {
            throw py::value_error(
                get_name(weights_ptr) + " parameter must be 1d. Actual " +
                std::to_string(weights_ptr->get_ndim()) + "d");
        }

        auto sample_size = sample.get_size();
        auto weights_size = weights_ptr->get_size();
        if (sample.get_size() != weights_ptr->get_size()) {
            throw py::value_error(
                get_name(&sample) + " size (" + std::to_string(sample_size) +
                ") and " + get_name(weights_ptr) + " size (" +
                std::to_string(weights_size) + ")" + " must match");
        }
    }

    if (sample.get_ndim() > 2) {
        throw py::value_error(
            get_name(&sample) +
            " parameter must have no more than 2 dimensions. Actual " +
            std::to_string(sample.get_ndim()) + "d");
    }

    if (sample.get_ndim() == 1) {
        if (bins.get_ndim() != 1) {
            throw py::value_error(get_name(&sample) + " parameter is 1d, but " +
                                  get_name(&bins) + " is " +
                                  std::to_string(bins.get_ndim()) + "d");
        }
    }
    else if (sample.get_ndim() == 2) {
        auto sample_count = sample.get_shape(0);
        auto expected_dims = sample.get_shape(1);

        if (bins.get_ndim() != expected_dims) {
            throw py::value_error(get_name(&sample) + " parameter has shape {" +
                                  std::to_string(sample_count) + "x" +
                                  std::to_string(expected_dims) + "}" +
                                  ", so " + get_name(&bins) +
                                  " parameter expected to be " +
                                  std::to_string(expected_dims) +
                                  "d. "
                                  "Actual " +
                                  std::to_string(bins.get_ndim()) + "d");
        }
    }

    py::ssize_t expected_hist_size = 1;
    for (int i = 0; i < bins.get_ndim(); ++i) {
        expected_hist_size *= (bins.get_shape(i) - 1);
    }

    if (histogram.get_size() != expected_hist_size) {
        throw py::value_error(
            get_name(&histogram) + " and " + get_name(&bins) +
            " shape mismatch. " + get_name(&histogram) +
            " expected to have size = " + std::to_string(expected_hist_size) +
            ". Actual " + std::to_string(histogram.get_size()));
    }

    int64_t max_hist_size = std::numeric_limits<uint32_t>::max() - 1;
    if (histogram.get_size() > max_hist_size) {
        throw py::value_error(get_name(&histogram) +
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
                get_name(&histogram) +
                " parameter has 64-bit type, but 64-bit atomics " +
                " are not supported for " + device_name);
        }
    }
}

uint32_t get_local_hist_copies_count(uint32_t loc_mem_size_in_items,
                                     uint32_t local_size,
                                     uint32_t hist_size_in_items)
{
    uint32_t max_local_copies = loc_mem_size_in_items / hist_size_in_items;
    uint32_t local_hist_count = std::max(
        std::min(int(std::ceil((float(4 * local_size) / hist_size_in_items))),
                 16),
        1);
    local_hist_count = std::min(local_hist_count, max_local_copies);

    return local_hist_count;
}

} // namespace histogram
} // namespace statistics
