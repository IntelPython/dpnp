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
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpnp.tensor._tensor_accumulation_impl
//  extensions
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "kernels/accumulators.hpp"
#include "simplify_iteration_space.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

template <typename strided_fnT, typename contig_fnT>
std::pair<sycl::event, sycl::event>
    py_accumulate_over_axis(const dpctl::tensor::usm_ndarray &src,
                            const int trailing_dims_to_accumulate,
                            const dpctl::tensor::usm_ndarray &dst,
                            sycl::queue &exec_q,
                            std::vector<sycl::event> const &depends,
                            const strided_fnT &strided_dispatch_table,
                            const contig_fnT &contig_dispatch_table)
{
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();
    if (src_nd != dst_nd) {
        throw py::value_error("The input and output arrays must have "
                              "the same array ranks");
    }
    int iter_nd = src_nd - trailing_dims_to_accumulate;
    if (trailing_dims_to_accumulate <= 0 || iter_nd < 0) {
        throw py::value_error(
            "trailing_dims_to_accumulate must be positive, but no "
            "greater than rank of the input array");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    std::size_t iter_nelems(1);
    for (int i = 0; same_shapes && (i < iter_nd); ++i) {
        auto src_shape_i = src_shape_ptr[i];
        same_shapes = same_shapes && (src_shape_i == dst_shape_ptr[i]);
        iter_nelems *= static_cast<std::size_t>(src_shape_i);
    }

    std::size_t acc_nelems(1);
    for (int i = iter_nd; same_shapes && (i < src_nd); ++i) {
        auto dst_shape_i = dst_shape_ptr[i];
        same_shapes = same_shapes && (src_shape_ptr[i] == dst_shape_i);
        acc_nelems *= static_cast<std::size_t>(dst_shape_i);
    }

    if (!same_shapes) {
        throw py::value_error(
            "Destination shape does not match the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    if ((iter_nelems == 0) || (acc_nelems == 0)) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, acc_nelems * iter_nelems);

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    std::vector<sycl::event> host_task_events;

    if ((is_src_c_contig && is_dst_c_contig) && iter_nd == 0) {
        auto fn = contig_dispatch_table[src_typeid][dst_typeid];
        if (fn == nullptr) {
            throw std::runtime_error("Datatypes are not supported");
        }

        sycl::event acc_ev = fn(exec_q, acc_nelems, src_data, dst_data,
                                host_task_events, depends);

        return std::make_pair(
            dpctl::utils::keep_args_alive(exec_q, {src, dst}, {acc_ev}),
            acc_ev);
    }

    auto src_shape_vec = src.get_shape_vector();
    auto src_strides_vec = src.get_strides_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    int acc_nd = trailing_dims_to_accumulate;

    using shT = std::vector<py::ssize_t>;
    shT acc_shape(std::begin(src_shape_vec) + iter_nd, std::end(src_shape_vec));

    shT acc_src_strides(std::begin(src_strides_vec) + iter_nd,
                        std::end(src_strides_vec));

    shT acc_dst_strides(std::begin(dst_strides_vec) + iter_nd,
                        std::end(dst_strides_vec));

    shT iter_shape(std::begin(src_shape_vec),
                   std::begin(src_shape_vec) + iter_nd);

    shT iter_src_strides(std::begin(src_strides_vec),
                         std::begin(src_strides_vec) + iter_nd);

    shT iter_dst_strides(std::begin(dst_strides_vec),
                         std::begin(dst_strides_vec) + iter_nd);

    shT simplified_iter_shape;
    shT simplified_iter_src_strides;
    shT simplified_iter_dst_strides;
    py::ssize_t iter_src_offset(0);
    py::ssize_t iter_dst_offset(0);

    if (iter_nd == 0) {
        iter_nd = 1;
        simplified_iter_shape.push_back(1);
        simplified_iter_src_strides.push_back(0);
        simplified_iter_dst_strides.push_back(0);
    }
    else {
        simplify_iteration_space(
            iter_nd, src_shape_ptr, iter_src_strides, iter_dst_strides,
            // output
            simplified_iter_shape, simplified_iter_src_strides,
            simplified_iter_dst_strides, iter_src_offset, iter_dst_offset);
    }

    // Strided implementation
    auto strided_fn = strided_dispatch_table[src_typeid][dst_typeid];
    if (strided_fn == nullptr) {
        throw std::runtime_error("Datatypes are not supported");
    }

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, simplified_iter_shape,
        simplified_iter_src_strides, simplified_iter_dst_strides, acc_shape,
        acc_src_strides, acc_dst_strides);
    auto packed_shapes_and_strides_owner =
        std::move(std::get<0>(ptr_size_event_tuple));
    const auto &copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple);
    const py::ssize_t *packed_shapes_and_strides =
        packed_shapes_and_strides_owner.get();

    const py::ssize_t *iter_shape_and_strides = packed_shapes_and_strides;
    const py::ssize_t *acc_shapes_and_strides =
        packed_shapes_and_strides + 3 * simplified_iter_shape.size();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), copy_shapes_strides_ev);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());

    sycl::event acc_ev = strided_fn(
        exec_q, iter_nelems, acc_nelems, src_data, iter_nd,
        iter_shape_and_strides, iter_src_offset, iter_dst_offset, acc_nd,
        acc_shapes_and_strides, dst_data, host_task_events, all_deps);

    sycl::event temp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {acc_ev}, packed_shapes_and_strides_owner);
    host_task_events.push_back(temp_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events),
        acc_ev);
}

template <typename strided_fnT, typename contig_fnT>
std::pair<sycl::event, sycl::event> py_accumulate_final_axis_include_initial(
    const dpctl::tensor::usm_ndarray &src,
    const dpctl::tensor::usm_ndarray &dst,
    sycl::queue &exec_q,
    std::vector<sycl::event> const &depends,
    const strided_fnT &strided_dispatch_table,
    const contig_fnT &contig_dispatch_table)
{
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    if (src_nd != dst_nd) {
        throw py::value_error("The input and output arrays must have "
                              "the same array ranks");
    }

    static constexpr int acc_nd = 1;

    int iter_nd = src_nd - acc_nd;
    if (iter_nd < 0) {
        throw py::value_error("accumulation axis must not be greater than rank "
                              "of the input array");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    std::size_t iter_nelems(1);
    for (int i = 0; same_shapes && (i < iter_nd); ++i) {
        auto src_shape_i = src_shape_ptr[i];
        same_shapes = same_shapes && (src_shape_i == dst_shape_ptr[i]);
        iter_nelems *= static_cast<std::size_t>(src_shape_i);
    }

    std::size_t acc_nelems(1);
    for (int i = iter_nd; same_shapes && (i < src_nd); ++i) {
        auto dst_shape_i = dst_shape_ptr[i];
        same_shapes = same_shapes && (src_shape_ptr[i] + 1 == dst_shape_i);
        acc_nelems *= static_cast<std::size_t>(dst_shape_i);
    }

    if (!same_shapes) {
        throw py::value_error(
            "Destination shape does not match the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    if ((iter_nelems == 0) || (acc_nelems == 0)) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, acc_nelems * iter_nelems);

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    std::vector<sycl::event> host_task_events;

    if ((is_src_c_contig && is_dst_c_contig) && iter_nd == 0) {
        auto fn = contig_dispatch_table[src_typeid][dst_typeid];
        if (fn == nullptr) {
            throw std::runtime_error("Datatypes are not supported");
        }

        sycl::event acc_ev = fn(exec_q, acc_nelems, src_data, dst_data,
                                host_task_events, depends);

        return std::make_pair(
            dpctl::utils::keep_args_alive(exec_q, {src, dst}, {acc_ev}),
            acc_ev);
    }

    auto src_shape_vec = src.get_shape_vector();
    auto src_strides_vec = src.get_strides_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT acc_shape(std::begin(src_shape_vec) + iter_nd, std::end(src_shape_vec));

    shT acc_src_strides(std::begin(src_strides_vec) + iter_nd,
                        std::end(src_strides_vec));

    shT acc_dst_strides(std::begin(dst_strides_vec) + iter_nd,
                        std::end(dst_strides_vec));

    shT iter_shape(std::begin(src_shape_vec),
                   std::begin(src_shape_vec) + iter_nd);

    shT iter_src_strides(std::begin(src_strides_vec),
                         std::begin(src_strides_vec) + iter_nd);

    shT iter_dst_strides(std::begin(dst_strides_vec),
                         std::begin(dst_strides_vec) + iter_nd);

    shT simplified_iter_shape;
    shT simplified_iter_src_strides;
    shT simplified_iter_dst_strides;
    py::ssize_t iter_src_offset(0);
    py::ssize_t iter_dst_offset(0);

    if (iter_nd == 0) {
        iter_nd = 1;
        simplified_iter_shape.push_back(1);
        simplified_iter_src_strides.push_back(0);
        simplified_iter_dst_strides.push_back(0);
    }
    else {
        simplify_iteration_space(
            iter_nd, src_shape_ptr, iter_src_strides, iter_dst_strides,
            // output
            simplified_iter_shape, simplified_iter_src_strides,
            simplified_iter_dst_strides, iter_src_offset, iter_dst_offset);
    }

    // Strided implementation
    auto strided_fn = strided_dispatch_table[src_typeid][dst_typeid];
    if (strided_fn == nullptr) {
        throw std::runtime_error("Datatypes are not supported");
    }

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, simplified_iter_shape,
        simplified_iter_src_strides, simplified_iter_dst_strides, acc_shape,
        acc_src_strides, acc_dst_strides);
    auto packed_shapes_and_strides_owner =
        std::move(std::get<0>(ptr_size_event_tuple));
    const auto &copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple);
    const py::ssize_t *packed_shapes_and_strides =
        packed_shapes_and_strides_owner.get();

    const py::ssize_t *iter_shape_and_strides = packed_shapes_and_strides;
    const py::ssize_t *acc_shapes_and_strides =
        packed_shapes_and_strides + 3 * simplified_iter_shape.size();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), copy_shapes_strides_ev);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());

    sycl::event acc_ev = strided_fn(
        exec_q, iter_nelems, acc_nelems, src_data, iter_nd,
        iter_shape_and_strides, iter_src_offset, iter_dst_offset, acc_nd,
        acc_shapes_and_strides, dst_data, host_task_events, all_deps);

    sycl::event temp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {acc_ev}, packed_shapes_and_strides_owner);
    host_task_events.push_back(temp_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events),
        acc_ev);
}

/*! @brief Template implementing Python API for querying accumulation
 * type support */
template <typename fnT>
bool py_accumulate_dtype_supported(const py::dtype &input_dtype,
                                   const py::dtype &output_dtype,
                                   const fnT &dispatch_table)
{
    int arg_tn =
        input_dtype.num(); // NumPy type numbers are the same as in dpctl
    int out_tn =
        output_dtype.num(); // NumPy type numbers are the same as in dpctl
    int arg_typeid = -1;
    int out_typeid = -1;

    auto array_types = td_ns::usm_ndarray_types();

    try {
        arg_typeid = array_types.typenum_to_lookup_id(arg_tn);
        out_typeid = array_types.typenum_to_lookup_id(out_tn);
    } catch (const std::exception &e) {
        throw py::value_error(e.what());
    }

    if (arg_typeid < 0 || arg_typeid >= td_ns::num_types || out_typeid < 0 ||
        out_typeid >= td_ns::num_types)
    {
        throw std::runtime_error("Reduction type support check: lookup failed");
    }

    // remove_all_extents gets underlying type of table
    using fn_ptrT = typename std::remove_all_extents<fnT>::type;
    fn_ptrT fn = nullptr;

    fn = dispatch_table[arg_typeid][out_typeid];

    return (fn != nullptr);
}

} // namespace dpctl::tensor::py_internal
