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

#include "validation_utils.hpp"
#include "utils/memory_overlap.hpp"

using statistics::validation::array_names;
using statistics::validation::array_ptr;

namespace
{

sycl::queue get_queue(const std::vector<array_ptr> &inputs,
                      const std::vector<array_ptr> &outputs)
{
    auto it = std::find_if(inputs.cbegin(), inputs.cend(),
                           [](const array_ptr &arr) { return arr != nullptr; });

    if (it != inputs.cend()) {
        return (*it)->get_queue();
    }

    it = std::find_if(outputs.cbegin(), outputs.cend(),
                      [](const array_ptr &arr) { return arr != nullptr; });

    if (it != outputs.cend()) {
        return (*it)->get_queue();
    }

    throw py::value_error("No input or output arrays found");
}
} // namespace

namespace statistics
{
namespace validation
{
std::string name_of(const array_ptr &arr, const array_names &names)
{
    auto name_it = names.find(arr);
    assert(name_it != names.end());

    if (name_it != names.end())
        return "'" + name_it->second + "'";

    return "'unknown'";
}

void check_writable(const std::vector<array_ptr> &arrays,
                    const array_names &names)
{
    for (const auto &arr : arrays) {
        if (arr != nullptr && !arr->is_writable()) {
            throw py::value_error(name_of(arr, names) +
                                  " parameter is not writable");
        }
    }
}

void check_c_contig(const std::vector<array_ptr> &arrays,
                    const array_names &names)
{
    for (const auto &arr : arrays) {
        if (arr != nullptr && !arr->is_c_contiguous()) {
            throw py::value_error(name_of(arr, names) +
                                  " parameter is not c-contiguos");
        }
    }
}

void check_queue(const std::vector<array_ptr> &arrays,
                 const array_names &names,
                 const sycl::queue &exec_q)
{
    auto unequal_queue =
        std::find_if(arrays.cbegin(), arrays.cend(), [&](const array_ptr &arr) {
            return arr != nullptr && arr->get_queue() != exec_q;
        });

    if (unequal_queue != arrays.cend()) {
        throw py::value_error(
            name_of(*unequal_queue, names) +
            " parameter has incompatible queue with other parameters");
    }
}

void check_no_overlap(const array_ptr &input,
                      const array_ptr &output,
                      const array_names &names)
{
    if (input == nullptr || output == nullptr) {
        return;
    }

    const auto &overlap = dpctl::tensor::overlap::MemoryOverlap();

    if (overlap(*input, *output)) {
        throw py::value_error(name_of(input, names) +
                              " has overlapping memory segments with " +
                              name_of(output, names));
    }
}

void check_no_overlap(const std::vector<array_ptr> &inputs,
                      const std::vector<array_ptr> &outputs,
                      const array_names &names)
{
    for (const auto &input : inputs) {
        for (const auto &output : outputs) {
            check_no_overlap(input, output, names);
        }
    }
}

void check_num_dims(const array_ptr &arr,
                    const size_t ndim,
                    const array_names &names)
{
    if (arr != nullptr && arr->get_ndim() != ndim) {
        throw py::value_error("Array " + name_of(arr, names) + " must be " +
                              std::to_string(ndim) + "D, but got " +
                              std::to_string(arr->get_ndim()) + "D.");
    }
}

void check_max_dims(const array_ptr &arr,
                    const size_t max_ndim,
                    const array_names &names)
{
    if (arr != nullptr && arr->get_ndim() > max_ndim) {
        throw py::value_error(
            "Array " + name_of(arr, names) + " must have no more than " +
            std::to_string(max_ndim) + " dimensions, but got " +
            std::to_string(arr->get_ndim()) + " dimensions.");
    }
}

void check_size_at_least(const array_ptr &arr,
                         const size_t size,
                         const array_names &names)
{
    if (arr != nullptr && arr->get_size() < size) {
        throw py::value_error("Array " + name_of(arr, names) +
                              " must have at least " + std::to_string(size) +
                              " elements, but got " +
                              std::to_string(arr->get_size()) + " elements.");
    }
}

void common_checks(const std::vector<array_ptr> &inputs,
                   const std::vector<array_ptr> &outputs,
                   const array_names &names)
{
    check_writable(outputs, names);

    check_c_contig(inputs, names);
    check_c_contig(outputs, names);

    auto exec_q = get_queue(inputs, outputs);

    check_queue(inputs, names, exec_q);
    check_queue(outputs, names, exec_q);

    check_no_overlap(inputs, outputs, names);
}

} // namespace validation
} // namespace statistics
