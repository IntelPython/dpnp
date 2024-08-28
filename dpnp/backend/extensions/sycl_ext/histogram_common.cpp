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
#include <string>
#include <unordered_map>
#include <vector>

#include "dpctl4pybind11.hpp"
#include <pybind11/pybind11.h>

#include "histogram_common.hpp"

using dpctl::tensor::usm_ndarray;

namespace histogram
{

void validate(const usm_ndarray &sample,
              const usm_ndarray &bins,
              std::optional<const dpctl::tensor::usm_ndarray> &weights,
              const usm_ndarray &histogram)
{
    auto exec_q = sample.get_queue();
    using array_ptr = const usm_ndarray *;
    using array_list = std::vector<array_ptr>;

    array_list arrays{&sample, &bins, &histogram};
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

    if (histogram.get_ndim() != 1) {
        throw py::value_error(get_name(&histogram) +
                              " parameter must be 1d. Actual " +
                              std::to_string(histogram.get_ndim()) + "d");
    }

    if (weights_ptr && weights_ptr->get_ndim() != 1) {
        throw py::value_error(get_name(weights_ptr) +
                              " parameter must be 1d. Actual " +
                              std::to_string(weights_ptr->get_ndim()) + "d");
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
}

} // namespace histogram
