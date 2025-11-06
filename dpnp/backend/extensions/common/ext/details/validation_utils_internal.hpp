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

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ext/common.hpp"

#include "ext/validation_utils.hpp"
#include "utils/memory_overlap.hpp"

namespace td_ns = dpctl::tensor::type_dispatch;
namespace common = ext::common;

namespace ext::validation
{
inline sycl::queue get_queue(const std::vector<array_ptr> &inputs,
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

inline std::string name_of(const array_ptr &arr, const array_names &names)
{
    auto name_it = names.find(arr);
    assert(name_it != names.end());

    if (name_it != names.end())
        return "'" + name_it->second + "'";

    return "'unknown'";
}

inline void check_writable(const std::vector<array_ptr> &arrays,
                           const array_names &names)
{
    for (const auto &arr : arrays) {
        if (arr != nullptr && !arr->is_writable()) {
            throw py::value_error(name_of(arr, names) +
                                  " parameter is not writable");
        }
    }
}

inline void check_c_contig(const std::vector<array_ptr> &arrays,
                           const array_names &names)
{
    for (const auto &arr : arrays) {
        if (arr != nullptr && !arr->is_c_contiguous()) {
            throw py::value_error(name_of(arr, names) +
                                  " parameter is not c-contiguos");
        }
    }
}

inline void check_queue(const std::vector<array_ptr> &arrays,
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

inline void check_no_overlap(const array_ptr &input,
                             const array_ptr &output,
                             const array_names &names)
{
    if (input == nullptr || output == nullptr) {
        return;
    }

    const auto &overlap = dpctl::tensor::overlap::MemoryOverlap();
    const auto &same_logical_tensors =
        dpctl::tensor::overlap::SameLogicalTensors();

    if (overlap(*input, *output) && !same_logical_tensors(*input, *output)) {
        throw py::value_error(name_of(input, names) +
                              " has overlapping memory segments with " +
                              name_of(output, names));
    }
}

inline void check_no_overlap(const std::vector<array_ptr> &inputs,
                             const std::vector<array_ptr> &outputs,
                             const array_names &names)
{
    for (const auto &input : inputs) {
        for (const auto &output : outputs) {
            check_no_overlap(input, output, names);
        }
    }
}

inline void check_num_dims(const array_ptr &arr,
                           const size_t ndim,
                           const array_names &names)
{
    size_t arr_n_dim = arr != nullptr ? arr->get_ndim() : 0;
    if (arr != nullptr && arr_n_dim != ndim) {
        throw py::value_error("Array " + name_of(arr, names) + " must be " +
                              std::to_string(ndim) + "D, but got " +
                              std::to_string(arr_n_dim) + "D.");
    }
}

inline void check_num_dims(const std::vector<array_ptr> &arrays,
                           const size_t ndim,
                           const array_names &names)
{
    for (const auto &arr : arrays) {
        check_num_dims(arr, ndim, names);
    }
}

inline void check_max_dims(const array_ptr &arr,
                           const size_t max_ndim,
                           const array_names &names)
{
    size_t arr_n_dim = arr != nullptr ? arr->get_ndim() : 0;
    if (arr != nullptr && arr_n_dim > max_ndim) {
        throw py::value_error(
            "Array " + name_of(arr, names) + " must have no more than " +
            std::to_string(max_ndim) + " dimensions, but got " +
            std::to_string(arr_n_dim) + " dimensions.");
    }
}

inline void check_size_at_least(const array_ptr &arr,
                                const size_t size,
                                const array_names &names)
{
    size_t arr_size = arr != nullptr ? arr->get_size() : 0;
    if (arr != nullptr && arr_size < size) {
        throw py::value_error("Array " + name_of(arr, names) +
                              " must have at least " + std::to_string(size) +
                              " elements, but got " + std::to_string(arr_size) +
                              " elements.");
    }
}

inline void check_has_dtype(const array_ptr &arr,
                            const typenum_t dtype,
                            const array_names &names)
{
    if (arr == nullptr) {
        return;
    }

    auto array_types = td_ns::usm_ndarray_types();
    int array_type_id = array_types.typenum_to_lookup_id(arr->get_typenum());
    int expected_type_id = static_cast<int>(dtype);

    if (array_type_id != expected_type_id) {
        py::dtype actual_dtype = common::dtype_from_typenum(array_type_id);
        py::dtype dtype_py = common::dtype_from_typenum(expected_type_id);

        std::string msg = "Array " + name_of(arr, names) + " must have dtype " +
                          std::string(py::str(dtype_py)) + ", but got " +
                          std::string(py::str(actual_dtype));

        throw py::value_error(msg);
    }
}

inline void check_same_dtype(const array_ptr &arr1,
                             const array_ptr &arr2,
                             const array_names &names)
{
    if (arr1 == nullptr || arr2 == nullptr) {
        return;
    }

    auto array_types = td_ns::usm_ndarray_types();
    int first_type_id = array_types.typenum_to_lookup_id(arr1->get_typenum());
    int second_type_id = array_types.typenum_to_lookup_id(arr2->get_typenum());

    if (first_type_id != second_type_id) {
        py::dtype first_dtype = common::dtype_from_typenum(first_type_id);
        py::dtype second_dtype = common::dtype_from_typenum(second_type_id);

        std::string msg = "Arrays " + name_of(arr1, names) + " and " +
                          name_of(arr2, names) +
                          " must have the same dtype, but got " +
                          std::string(py::str(first_dtype)) + " and " +
                          std::string(py::str(second_dtype));

        throw py::value_error(msg);
    }
}

inline void check_same_dtype(const std::vector<array_ptr> &arrays,
                             const array_names &names)
{
    if (arrays.empty()) {
        return;
    }

    const auto *first = arrays[0];
    for (size_t i = 1; i < arrays.size(); ++i) {
        check_same_dtype(first, arrays[i], names);
    }
}

inline void check_same_size(const array_ptr &arr1,
                            const array_ptr &arr2,
                            const array_names &names)
{
    if (arr1 == nullptr || arr2 == nullptr) {
        return;
    }

    auto size1 = arr1->get_size();
    auto size2 = arr2->get_size();

    if (size1 != size2) {
        std::string msg =
            "Arrays " + name_of(arr1, names) + " and " + name_of(arr2, names) +
            " must have the same size, but got " + std::to_string(size1) +
            " and " + std::to_string(size2);

        throw py::value_error(msg);
    }
}

inline void check_same_size(const std::vector<array_ptr> &arrays,
                            const array_names &names)
{
    if (arrays.empty()) {
        return;
    }

    auto first = arrays[0];
    for (size_t i = 1; i < arrays.size(); ++i) {
        check_same_size(first, arrays[i], names);
    }
}

inline void common_checks(const std::vector<array_ptr> &inputs,
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

} // namespace ext::validation
