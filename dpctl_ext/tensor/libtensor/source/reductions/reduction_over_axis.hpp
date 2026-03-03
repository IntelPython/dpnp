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
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_reductions_impl
/// extension, specifically functions for reductions.
//===---------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/reductions.hpp"
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

/* ====================== dtype supported ======================== */

/*! @brief Template implementing Python API for querying type support by
 * reduction which may support atomics */
template <typename fnT, typename CheckAtomicSupportFnT>
bool py_reduction_dtype_supported(
    const py::dtype &input_dtype,
    const py::dtype &output_dtype,
    const std::string &dst_usm_type,
    sycl::queue &q,
    const fnT &atomic_dispatch_table,
    const fnT &temps_dispatch_table,
    const CheckAtomicSupportFnT &check_atomic_support)
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

    sycl::usm::alloc kind = sycl::usm::alloc::unknown;

    if (dst_usm_type == "device") {
        kind = sycl::usm::alloc::device;
    }
    else if (dst_usm_type == "shared") {
        kind = sycl::usm::alloc::shared;
    }
    else if (dst_usm_type == "host") {
        kind = sycl::usm::alloc::host;
    }
    else {
        throw py::value_error("Unrecognized `dst_usm_type` argument.");
    }

    bool supports_atomics = check_atomic_support[out_typeid](q, kind);

    if (supports_atomics) {
        fn = atomic_dispatch_table[arg_typeid][out_typeid];
    }

    if (fn == nullptr) {
        // use slower reduction implementation using temporaries
        fn = temps_dispatch_table[arg_typeid][out_typeid];
    }

    return (fn != nullptr);
}

/*! @brief Template implementing Python API for querying type support by tree
 * reduction */
template <typename fnT>
bool py_tree_reduction_dtype_supported(const py::dtype &input_dtype,
                                       const py::dtype &output_dtype,
                                       const fnT &temps_dispatch_table)
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

    auto fn = temps_dispatch_table[arg_typeid][out_typeid];

    return (fn != nullptr);
}

/* ==================== Generic reductions ====================== */

/*! @brief Template implementing Python API for reduction over axis which may
 * support atomics */
template <typename strided_fnT, typename contig_fnT, typename SupportAtomicFnT>
std::pair<sycl::event, sycl::event> py_reduction_over_axis(
    const dpctl::tensor::usm_ndarray &src,
    int trailing_dims_to_reduce, // comp over this many trailing indexes
    const dpctl::tensor::usm_ndarray &dst,
    sycl::queue &exec_q,
    const std::vector<sycl::event> &depends,
    const strided_fnT &atomic_dispatch_table,
    const contig_fnT &axis0_atomic_dispatch_table,
    const contig_fnT &axis1_atomic_dispatch_table,
    const strided_fnT &temps_dispatch_table,
    const contig_fnT &axis0_temps_dispatch_table,
    const contig_fnT &axis1_temps_dispatch_table,
    const SupportAtomicFnT &check_atomic_support)
{
    int src_nd = src.get_ndim();
    int iteration_nd = src_nd - trailing_dims_to_reduce;
    if (trailing_dims_to_reduce <= 0 || iteration_nd < 0) {
        throw py::value_error("Trailing_dim_to_reduce must be positive, but no "
                              "greater than rank of the array being reduced");
    }

    int dst_nd = dst.get_ndim();
    if (dst_nd != iteration_nd) {
        throw py::value_error("Destination array rank does not match input "
                              "array rank and number of reduced dimensions");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    for (int i = 0; same_shapes && (i < dst_nd); ++i) {
        same_shapes = same_shapes && (src_shape_ptr[i] == dst_shape_ptr[i]);
    }

    if (!same_shapes) {
        throw py::value_error("Destination shape does not match unreduced "
                              "dimensions of the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    std::size_t dst_nelems = dst.get_size();

    if (dst_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    std::size_t reduction_nelems(1);
    for (int i = dst_nd; i < src_nd; ++i) {
        reduction_nelems *= static_cast<std::size_t>(src_shape_ptr[i]);
    }

    // check that dst and src do not overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, dst_nelems);

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    namespace td_ns = dpctl::tensor::type_dispatch;
    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    void *data_ptr = dst.get_data();
    const auto &ctx = exec_q.get_context();
    auto usm_type = sycl::get_pointer_type(data_ptr, ctx);

    bool supports_atomics = check_atomic_support[dst_typeid](exec_q, usm_type);

    // handle special case when both reduction and iteration are 1D contiguous
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    if ((is_src_c_contig && is_dst_c_contig) ||
        (is_src_f_contig && dst_nelems == 1))
    {
        // remove_all_extents gets underlying type of table
        using contig_fn_ptr_T =
            typename std::remove_all_extents<contig_fnT>::type;
        contig_fn_ptr_T fn;
        if (supports_atomics) {
            fn = axis1_atomic_dispatch_table[src_typeid][dst_typeid];
        }
        else {
            fn = axis1_temps_dispatch_table[src_typeid][dst_typeid];
        }
        if (fn != nullptr) {
            std::size_t iter_nelems = dst_nelems;

            static constexpr py::ssize_t zero_offset = 0;

            sycl::event reduction_over_axis_contig_ev =
                fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                   dst.get_data(),
                   zero_offset, // iteration_src_offset
                   zero_offset, // iteration_dst_offset
                   zero_offset, // reduction_src_offset
                   depends);

            sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                exec_q, {src, dst}, {reduction_over_axis_contig_ev});

            return std::make_pair(keep_args_event,
                                  reduction_over_axis_contig_ev);
        }
    }
    else if (is_src_f_contig &&
             ((is_dst_c_contig && dst_nd == 1) || dst.is_f_contiguous()))
    {
        // remove_all_extents gets underlying type of table
        using contig_fn_ptr_T =
            typename std::remove_all_extents<contig_fnT>::type;
        contig_fn_ptr_T fn;
        if (supports_atomics) {
            fn = axis0_atomic_dispatch_table[src_typeid][dst_typeid];
        }
        else {
            fn = axis0_temps_dispatch_table[src_typeid][dst_typeid];
        }
        if (fn != nullptr) {
            std::size_t iter_nelems = dst_nelems;

            static constexpr py::ssize_t zero_offset = 0;

            sycl::event reduction_over_axis_contig_ev =
                fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                   dst.get_data(),
                   zero_offset, // iteration_src_offset
                   zero_offset, // iteration_dst_offset
                   zero_offset, // reduction_src_offset
                   depends);

            sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                exec_q, {src, dst}, {reduction_over_axis_contig_ev});

            return std::make_pair(keep_args_event,
                                  reduction_over_axis_contig_ev);
        }
    }

    using dpctl::tensor::py_internal::simplify_iteration_space;
    using dpctl::tensor::py_internal::simplify_iteration_space_1;

    // TODO: not used anywhere
    auto const &src_shape_vecs = src.get_shape_vector();
    auto const &src_strides_vecs = src.get_strides_vector();
    auto const &dst_strides_vecs = dst.get_strides_vector();

    int reduction_nd = trailing_dims_to_reduce;
    const py::ssize_t *reduction_shape_ptr = src_shape_ptr + dst_nd;
    using shT = std::vector<py::ssize_t>;
    shT reduction_src_strides(std::begin(src_strides_vecs) + dst_nd,
                              std::end(src_strides_vecs));

    shT simplified_reduction_shape;
    shT simplified_reduction_src_strides;
    py::ssize_t reduction_src_offset(0);

    simplify_iteration_space_1(
        reduction_nd, reduction_shape_ptr, reduction_src_strides,
        // output
        simplified_reduction_shape, simplified_reduction_src_strides,
        reduction_src_offset);

    const py::ssize_t *iteration_shape_ptr = src_shape_ptr;

    shT iteration_src_strides(std::begin(src_strides_vecs),
                              std::begin(src_strides_vecs) + iteration_nd);
    shT const &iteration_dst_strides = dst_strides_vecs;

    shT simplified_iteration_shape;
    shT simplified_iteration_src_strides;
    shT simplified_iteration_dst_strides;
    py::ssize_t iteration_src_offset(0);
    py::ssize_t iteration_dst_offset(0);

    if (iteration_nd == 0) {
        if (dst_nelems != 1) {
            throw std::runtime_error("iteration_nd == 0, but dst_nelems != 1");
        }
        iteration_nd = 1;
        simplified_iteration_shape.push_back(1);
        simplified_iteration_src_strides.push_back(0);
        simplified_iteration_dst_strides.push_back(0);
    }
    else {
        simplify_iteration_space(iteration_nd, iteration_shape_ptr,
                                 iteration_src_strides, iteration_dst_strides,
                                 // output
                                 simplified_iteration_shape,
                                 simplified_iteration_src_strides,
                                 simplified_iteration_dst_strides,
                                 iteration_src_offset, iteration_dst_offset);
    }

    if ((reduction_nd == 1) && (iteration_nd == 1)) {
        bool mat_reduce_over_axis1 = false;
        bool mat_reduce_over_axis0 = false;
        bool array_reduce_all_elems = false;
        std::size_t iter_nelems = dst_nelems;

        if (simplified_reduction_src_strides[0] == 1) {
            array_reduce_all_elems = (simplified_iteration_shape[0] == 1);
            mat_reduce_over_axis1 =
                (simplified_iteration_dst_strides[0] == 1) &&
                (static_cast<std::size_t>(
                     simplified_iteration_src_strides[0]) == reduction_nelems);
        }
        else if (static_cast<std::size_t>(
                     simplified_reduction_src_strides[0]) == iter_nelems)
        {
            mat_reduce_over_axis0 =
                (simplified_iteration_dst_strides[0] == 1) &&
                (simplified_iteration_src_strides[0] == 1);
        }

        if (mat_reduce_over_axis1 || array_reduce_all_elems) {
            using contig_fn_ptr_T =
                typename std::remove_all_extents<contig_fnT>::type;
            contig_fn_ptr_T fn;
            if (supports_atomics) {
                fn = axis1_atomic_dispatch_table[src_typeid][dst_typeid];
            }
            else {
                fn = axis1_temps_dispatch_table[src_typeid][dst_typeid];
            }
            if (fn != nullptr) {
                sycl::event reduction_over_axis1_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(), iteration_src_offset,
                       iteration_dst_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis1_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis1_contig_ev);
            }
        }
        else if (mat_reduce_over_axis0) {
            using contig_fn_ptr_T =
                typename std::remove_all_extents<contig_fnT>::type;
            contig_fn_ptr_T fn;
            if (supports_atomics) {
                fn = axis0_atomic_dispatch_table[src_typeid][dst_typeid];
            }
            else {
                fn = axis0_temps_dispatch_table[src_typeid][dst_typeid];
            }
            if (fn != nullptr) {
                sycl::event reduction_over_axis0_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(), iteration_src_offset,
                       iteration_dst_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis0_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis0_contig_ev);
            }
        }
    }

    // remove_all_extents gets underlying type of table
    using strided_fn_ptr_T =
        typename std::remove_all_extents<strided_fnT>::type;
    strided_fn_ptr_T fn = nullptr;

    if (supports_atomics) {
        fn = atomic_dispatch_table[src_typeid][dst_typeid];
    }

    if (fn == nullptr) {
        // use slower reduction implementation using temporaries
        fn = temps_dispatch_table[src_typeid][dst_typeid];
        if (fn == nullptr) {
            throw std::runtime_error("Datatypes are not supported");
        }
    }

    std::vector<sycl::event> host_task_events{};
    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto arrays_metainfo_packing_triple_ =
        device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events,
            // iteration metadata
            simplified_iteration_shape, simplified_iteration_src_strides,
            simplified_iteration_dst_strides,
            // reduction metadata
            simplified_reduction_shape, simplified_reduction_src_strides);
    auto tmp_alloc_owner =
        std::move(std::get<0>(arrays_metainfo_packing_triple_));
    const auto &copy_metadata_ev = std::get<2>(arrays_metainfo_packing_triple_);
    const py::ssize_t *temp_allocation_ptr = tmp_alloc_owner.get();

    const py::ssize_t *iter_shape_and_strides = temp_allocation_ptr;
    const py::ssize_t *reduction_shape_stride =
        temp_allocation_ptr + 3 * simplified_iteration_shape.size();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.resize(depends.size());
    std::copy(depends.begin(), depends.end(), all_deps.begin());
    all_deps.push_back(copy_metadata_ev);

    auto reduction_ev =
        fn(exec_q, dst_nelems, reduction_nelems, src.get_data(), dst.get_data(),
           iteration_nd, iter_shape_and_strides, iteration_src_offset,
           iteration_dst_offset,
           reduction_nd, // number dimensions being reduced
           reduction_shape_stride, reduction_src_offset, all_deps);

    sycl::event temp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {reduction_ev}, tmp_alloc_owner);
    host_task_events.push_back(temp_cleanup_ev);

    sycl::event keep_args_event =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events);

    return std::make_pair(keep_args_event, reduction_ev);
}

/* ================= No atomic reductions ====================== */

/*! @brief Template implementing Python API for reduction over axis without
 * atomics */
template <typename strided_fnT, typename contig_fnT>
std::pair<sycl::event, sycl::event> py_tree_reduction_over_axis(
    const dpctl::tensor::usm_ndarray &src,
    int trailing_dims_to_reduce, // comp over this many trailing indexes
    const dpctl::tensor::usm_ndarray &dst,
    sycl::queue &exec_q,
    const std::vector<sycl::event> &depends,
    const strided_fnT &temps_dispatch_table,
    const contig_fnT &axis0_temps_dispatch_table,
    const contig_fnT &axis1_temps_dispatch_table)
{
    int src_nd = src.get_ndim();
    int iteration_nd = src_nd - trailing_dims_to_reduce;
    if (trailing_dims_to_reduce <= 0 || iteration_nd < 0) {
        throw py::value_error("Trailing_dim_to_reduce must be positive, but no "
                              "greater than rank of the array being reduced");
    }

    int dst_nd = dst.get_ndim();
    if (dst_nd != iteration_nd) {
        throw py::value_error("Destination array rank does not match input "
                              "array rank and number of reduced dimensions");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    for (int i = 0; same_shapes && (i < dst_nd); ++i) {
        same_shapes = same_shapes && (src_shape_ptr[i] == dst_shape_ptr[i]);
    }

    if (!same_shapes) {
        throw py::value_error("Destination shape does not match unreduced "
                              "dimensions of the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    std::size_t dst_nelems = dst.get_size();

    if (dst_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    std::size_t reduction_nelems(1);
    for (int i = dst_nd; i < src_nd; ++i) {
        reduction_nelems *= static_cast<std::size_t>(src_shape_ptr[i]);
    }

    // check that dst and src do not overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, dst_nelems);

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    namespace td_ns = dpctl::tensor::type_dispatch;
    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    // handle special case when both reduction and iteration are 1D contiguous
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    if ((is_src_c_contig && is_dst_c_contig) ||
        (is_src_f_contig && dst_nelems == 1))
    {
        auto fn = axis1_temps_dispatch_table[src_typeid][dst_typeid];
        if (fn != nullptr) {
            std::size_t iter_nelems = dst_nelems;

            static constexpr py::ssize_t zero_offset = 0;

            sycl::event reduction_over_axis_contig_ev =
                fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                   dst.get_data(),
                   zero_offset, // iteration_src_offset
                   zero_offset, // iteration_dst_offset
                   zero_offset, // reduction_src_offset
                   depends);

            sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                exec_q, {src, dst}, {reduction_over_axis_contig_ev});

            return std::make_pair(keep_args_event,
                                  reduction_over_axis_contig_ev);
        }
    }
    else if (is_src_f_contig &&
             ((is_dst_c_contig && dst_nd == 1) || dst.is_f_contiguous()))
    {
        auto fn = axis0_temps_dispatch_table[src_typeid][dst_typeid];
        if (fn != nullptr) {
            std::size_t iter_nelems = dst_nelems;

            static constexpr py::ssize_t zero_offset = 0;

            sycl::event reduction_over_axis_contig_ev =
                fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                   dst.get_data(),
                   zero_offset, // iteration_src_offset
                   zero_offset, // iteration_dst_offset
                   zero_offset, // reduction_src_offset
                   depends);

            sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                exec_q, {src, dst}, {reduction_over_axis_contig_ev});

            return std::make_pair(keep_args_event,
                                  reduction_over_axis_contig_ev);
        }
    }

    using dpctl::tensor::py_internal::simplify_iteration_space;
    using dpctl::tensor::py_internal::simplify_iteration_space_1;

    auto const &src_shape_vecs = src.get_shape_vector();
    auto const &src_strides_vecs = src.get_strides_vector();
    auto const &dst_strides_vecs = dst.get_strides_vector();

    int reduction_nd = trailing_dims_to_reduce;
    const py::ssize_t *reduction_shape_ptr = src_shape_ptr + dst_nd;
    using shT = std::vector<py::ssize_t>;
    shT reduction_src_strides(std::begin(src_strides_vecs) + dst_nd,
                              std::end(src_strides_vecs));

    shT simplified_reduction_shape;
    shT simplified_reduction_src_strides;
    py::ssize_t reduction_src_offset(0);

    simplify_iteration_space_1(
        reduction_nd, reduction_shape_ptr, reduction_src_strides,
        // output
        simplified_reduction_shape, simplified_reduction_src_strides,
        reduction_src_offset);

    const py::ssize_t *iteration_shape_ptr = src_shape_ptr;

    shT iteration_src_strides(std::begin(src_strides_vecs),
                              std::begin(src_strides_vecs) + iteration_nd);
    shT const &iteration_dst_strides = dst_strides_vecs;

    shT simplified_iteration_shape;
    shT simplified_iteration_src_strides;
    shT simplified_iteration_dst_strides;
    py::ssize_t iteration_src_offset(0);
    py::ssize_t iteration_dst_offset(0);

    if (iteration_nd == 0) {
        if (dst_nelems != 1) {
            throw std::runtime_error("iteration_nd == 0, but dst_nelems != 1");
        }
        iteration_nd = 1;
        simplified_iteration_shape.push_back(1);
        simplified_iteration_src_strides.push_back(0);
        simplified_iteration_dst_strides.push_back(0);
    }
    else {
        simplify_iteration_space(iteration_nd, iteration_shape_ptr,
                                 iteration_src_strides, iteration_dst_strides,
                                 // output
                                 simplified_iteration_shape,
                                 simplified_iteration_src_strides,
                                 simplified_iteration_dst_strides,
                                 iteration_src_offset, iteration_dst_offset);
    }

    if ((reduction_nd == 1) && (iteration_nd == 1)) {
        bool mat_reduce_over_axis1 = false;
        bool mat_reduce_over_axis0 = false;
        bool array_reduce_all_elems = false;
        std::size_t iter_nelems = dst_nelems;

        if (simplified_reduction_src_strides[0] == 1) {
            array_reduce_all_elems = (simplified_iteration_shape[0] == 1);
            mat_reduce_over_axis1 =
                (simplified_iteration_dst_strides[0] == 1) &&
                (static_cast<std::size_t>(
                     simplified_iteration_src_strides[0]) == reduction_nelems);
        }
        else if (static_cast<std::size_t>(
                     simplified_reduction_src_strides[0]) == iter_nelems)
        {
            mat_reduce_over_axis0 =
                (simplified_iteration_dst_strides[0] == 1) &&
                (simplified_iteration_src_strides[0] == 1);
        }

        if (mat_reduce_over_axis1 || array_reduce_all_elems) {
            auto fn = axis1_temps_dispatch_table[src_typeid][dst_typeid];
            if (fn != nullptr) {
                sycl::event reduction_over_axis1_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(), iteration_src_offset,
                       iteration_dst_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis1_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis1_contig_ev);
            }
        }
        else if (mat_reduce_over_axis0) {
            auto fn = axis0_temps_dispatch_table[src_typeid][dst_typeid];
            if (fn != nullptr) {
                sycl::event reduction_over_axis0_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(), iteration_src_offset,
                       iteration_dst_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis0_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis0_contig_ev);
            }
        }
    }

    auto fn = temps_dispatch_table[src_typeid][dst_typeid];
    if (fn == nullptr) {
        throw std::runtime_error("Datatypes are not supported");
    }

    std::vector<sycl::event> host_task_events{};
    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto arrays_metainfo_packing_triple_ =
        device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events,
            // iteration metadata
            simplified_iteration_shape, simplified_iteration_src_strides,
            simplified_iteration_dst_strides,
            // reduction metadata
            simplified_reduction_shape, simplified_reduction_src_strides);
    auto tmp_owner = std::move(std::get<0>(arrays_metainfo_packing_triple_));
    const auto &copy_metadata_ev = std::get<2>(arrays_metainfo_packing_triple_);
    const py::ssize_t *temp_allocation_ptr = tmp_owner.get();

    const py::ssize_t *iter_shape_and_strides = temp_allocation_ptr;
    const py::ssize_t *reduction_shape_stride =
        temp_allocation_ptr + 3 * simplified_iteration_shape.size();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.resize(depends.size());
    std::copy(depends.begin(), depends.end(), all_deps.begin());
    all_deps.push_back(copy_metadata_ev);

    auto reduction_ev =
        fn(exec_q, dst_nelems, reduction_nelems, src.get_data(), dst.get_data(),
           iteration_nd, iter_shape_and_strides, iteration_src_offset,
           iteration_dst_offset,
           reduction_nd, // number dimensions being reduced
           reduction_shape_stride, reduction_src_offset, all_deps);

    sycl::event temp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {reduction_ev}, tmp_owner);
    host_task_events.push_back(temp_cleanup_ev);

    sycl::event keep_args_event =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events);

    return std::make_pair(keep_args_event, reduction_ev);
}

/*! @brief Template implementing Python API for searching over an axis */
template <typename strided_fnT, typename contig_fnT>
std::pair<sycl::event, sycl::event> py_search_over_axis(
    const dpctl::tensor::usm_ndarray &src,
    int trailing_dims_to_reduce, // comp over this many trailing indexes
    const dpctl::tensor::usm_ndarray &dst,
    sycl::queue &exec_q,
    const std::vector<sycl::event> &depends,
    const strided_fnT &strided_dispatch_table,
    const contig_fnT &axis0_contig_dispatch_table,
    const contig_fnT &axis1_contig_dispatch_table)
{
    int src_nd = src.get_ndim();
    int iteration_nd = src_nd - trailing_dims_to_reduce;
    if (trailing_dims_to_reduce <= 0 || iteration_nd < 0) {
        throw py::value_error("Trailing_dim_to_reduce must be positive, but no "
                              "greater than rank of the array being reduced");
    }

    int dst_nd = dst.get_ndim();
    if (dst_nd != iteration_nd) {
        throw py::value_error("Destination array rank does not match input "
                              "array rank and number of reduced dimensions");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    for (int i = 0; same_shapes && (i < dst_nd); ++i) {
        same_shapes = same_shapes && (src_shape_ptr[i] == dst_shape_ptr[i]);
    }

    if (!same_shapes) {
        throw py::value_error("Destination shape does not match unreduced "
                              "dimensions of the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    std::size_t dst_nelems = dst.get_size();

    if (dst_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    std::size_t reduction_nelems(1);
    for (int i = dst_nd; i < src_nd; ++i) {
        reduction_nelems *= static_cast<std::size_t>(src_shape_ptr[i]);
    }

    // check that dst and src do not overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, dst_nelems);

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    namespace td_ns = dpctl::tensor::type_dispatch;
    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    // handle special case when both reduction and iteration are 1D contiguous
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    if (is_src_c_contig && is_dst_c_contig) {
        auto fn = axis1_contig_dispatch_table[src_typeid][dst_typeid];
        if (fn != nullptr) {
            std::size_t iter_nelems = dst_nelems;

            static constexpr py::ssize_t zero_offset = 0;

            sycl::event reduction_over_axis_contig_ev =
                fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                   dst.get_data(),
                   zero_offset, // iteration_src_offset
                   zero_offset, // iteration_dst_offset
                   zero_offset, // reduction_src_offset
                   depends);

            sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                exec_q, {src, dst}, {reduction_over_axis_contig_ev});

            return std::make_pair(keep_args_event,
                                  reduction_over_axis_contig_ev);
        }
    }
    else if (is_src_f_contig && dst_nd == 1) {
        auto fn = axis0_contig_dispatch_table[src_typeid][dst_typeid];
        if (fn != nullptr) {
            std::size_t iter_nelems = dst_nelems;

            static constexpr py::ssize_t zero_offset = 0;

            sycl::event reduction_over_axis_contig_ev =
                fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                   dst.get_data(),
                   zero_offset, // iteration_src_offset
                   zero_offset, // iteration_dst_offset
                   zero_offset, // reduction_src_offset
                   depends);

            sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                exec_q, {src, dst}, {reduction_over_axis_contig_ev});

            return std::make_pair(keep_args_event,
                                  reduction_over_axis_contig_ev);
        }
    }

    using dpctl::tensor::py_internal::simplify_iteration_space;

    auto const &src_shape_vecs = src.get_shape_vector();
    auto const &src_strides_vecs = src.get_strides_vector();
    auto const &dst_strides_vecs = dst.get_strides_vector();

    int reduction_nd = trailing_dims_to_reduce;
    const py::ssize_t *reduction_shape_ptr = src_shape_ptr + dst_nd;
    using shT = std::vector<py::ssize_t>;
    shT reduction_src_strides(std::begin(src_strides_vecs) + dst_nd,
                              std::end(src_strides_vecs));

    shT compact_reduction_shape;
    shT compact_reduction_src_strides;
    py::ssize_t reduction_src_offset(0);

    // TODO: not used anywhere
    compact_iteration_space(
        reduction_nd, reduction_shape_ptr, reduction_src_strides,
        // output
        compact_reduction_shape, compact_reduction_src_strides);

    const py::ssize_t *iteration_shape_ptr = src_shape_ptr;

    shT iteration_src_strides(std::begin(src_strides_vecs),
                              std::begin(src_strides_vecs) + iteration_nd);
    shT const &iteration_dst_strides = dst_strides_vecs;

    shT simplified_iteration_shape;
    shT simplified_iteration_src_strides;
    shT simplified_iteration_dst_strides;
    py::ssize_t iteration_src_offset(0);
    py::ssize_t iteration_dst_offset(0);

    if (iteration_nd == 0) {
        if (dst_nelems != 1) {
            throw std::runtime_error("iteration_nd == 0, but dst_nelems != 1");
        }
        iteration_nd = 1;
        simplified_iteration_shape.push_back(1);
        simplified_iteration_src_strides.push_back(0);
        simplified_iteration_dst_strides.push_back(0);
    }
    else {
        simplify_iteration_space(iteration_nd, iteration_shape_ptr,
                                 iteration_src_strides, iteration_dst_strides,
                                 // output
                                 simplified_iteration_shape,
                                 simplified_iteration_src_strides,
                                 simplified_iteration_dst_strides,
                                 iteration_src_offset, iteration_dst_offset);
    }

    if ((reduction_nd == 1) && (iteration_nd == 1)) {
        bool mat_reduce_over_axis1 = false;
        bool mat_reduce_over_axis0 = false;
        std::size_t iter_nelems = dst_nelems;

        if (compact_reduction_src_strides[0] == 1) {
            mat_reduce_over_axis1 =
                (simplified_iteration_dst_strides[0] == 1) &&
                (static_cast<std::size_t>(
                     simplified_iteration_src_strides[0]) == reduction_nelems);
        }
        else if (static_cast<std::size_t>(compact_reduction_src_strides[0]) ==
                 iter_nelems)
        {
            mat_reduce_over_axis0 =
                (simplified_iteration_dst_strides[0] == 1) &&
                (simplified_iteration_src_strides[0] == 1);
        }

        if (mat_reduce_over_axis1) {
            auto fn = axis1_contig_dispatch_table[src_typeid][dst_typeid];
            if (fn != nullptr) {
                sycl::event reduction_over_axis1_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(), iteration_src_offset,
                       iteration_dst_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis1_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis1_contig_ev);
            }
        }
        else if (mat_reduce_over_axis0) {
            auto fn = axis0_contig_dispatch_table[src_typeid][dst_typeid];
            if (fn != nullptr) {
                sycl::event reduction_over_axis0_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst.get_data(), iteration_src_offset,
                       iteration_dst_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpctl::utils::keep_args_alive(
                    exec_q, {src, dst}, {reduction_over_axis0_contig_ev});

                return std::make_pair(keep_args_event,
                                      reduction_over_axis0_contig_ev);
            }
        }
    }

    auto fn = strided_dispatch_table[src_typeid][dst_typeid];
    if (fn == nullptr) {
        throw std::runtime_error("Datatypes are not supported");
    }

    std::vector<sycl::event> host_task_events{};

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    auto arrays_metainfo_packing_triple_ =
        device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events,
            // iteration metadata
            simplified_iteration_shape, simplified_iteration_src_strides,
            simplified_iteration_dst_strides,
            // reduction metadata
            compact_reduction_shape, compact_reduction_src_strides);
    auto tmp_owner = std::move(std::get<0>(arrays_metainfo_packing_triple_));
    const auto &copy_metadata_ev = std::get<2>(arrays_metainfo_packing_triple_);
    const py::ssize_t *temp_allocation_ptr = tmp_owner.get();

    const py::ssize_t *iter_shape_and_strides = temp_allocation_ptr;
    const py::ssize_t *reduction_shape_stride =
        temp_allocation_ptr + 3 * simplified_iteration_shape.size();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.resize(depends.size());
    std::copy(depends.begin(), depends.end(), all_deps.begin());
    all_deps.push_back(copy_metadata_ev);

    auto comp_ev = fn(exec_q, dst_nelems, reduction_nelems, src.get_data(),
                      dst.get_data(), iteration_nd, iter_shape_and_strides,
                      iteration_src_offset, iteration_dst_offset,
                      reduction_nd, // number dimensions being reduced
                      reduction_shape_stride, reduction_src_offset, all_deps);

    sycl::event temp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {comp_ev}, tmp_owner);
    host_task_events.push_back(temp_cleanup_ev);

    sycl::event keep_args_event =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events);

    return std::make_pair(keep_args_event, comp_ev);
}

/* ================= Atomic only reductions ====================== */

/*! @brief Template implementing Python API for boolean reductions over an axis
 */
template <typename contig_dispatchT,
          typename strided_dispatchT,
          typename atomic_support_fnT>
std::pair<sycl::event, sycl::event>
    py_boolean_reduction(const dpctl::tensor::usm_ndarray &src,
                         int trailing_dims_to_reduce,
                         const dpctl::tensor::usm_ndarray &dst,
                         sycl::queue &exec_q,
                         const std::vector<sycl::event> &depends,
                         const contig_dispatchT &axis1_contig_dispatch_vector,
                         const contig_dispatchT &axis0_contig_dispatch_vector,
                         const strided_dispatchT &strided_dispatch_vector,
                         const atomic_support_fnT check_atomic_support)
{
    int src_nd = src.get_ndim();
    int iter_nd = src_nd - trailing_dims_to_reduce;
    if (trailing_dims_to_reduce <= 0 || iter_nd < 0) {
        throw py::value_error("Trailing_dim_to_reduce must be positive, but no "
                              "greater than rank of the array being reduced");
    }

    int dst_nd = dst.get_ndim();
    if (dst_nd != iter_nd) {
        throw py::value_error("Destination array rank does not match input "
                              "array rank and number of reduced dimensions");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    for (int i = 0; same_shapes && (i < dst_nd); ++i) {
        same_shapes = same_shapes && (src_shape_ptr[i] == dst_shape_ptr[i]);
    }

    if (!same_shapes) {
        throw py::value_error("Destination shape does not match unreduced "
                              "dimensions of the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    std::size_t dst_nelems = dst.get_size();

    std::size_t red_nelems(1);
    for (int i = dst_nd; i < src_nd; ++i) {
        red_nelems *= static_cast<std::size_t>(src_shape_ptr[i]);
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(dst, src)) {
        throw py::value_error("Arrays are expected to have no memory overlap");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, dst_nelems);

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    static constexpr int int32_typeid =
        static_cast<int>(td_ns::typenum_t::INT32);
    if (dst_typeid != int32_typeid) {
        throw py::value_error(
            "Unexpected data type of destination array, expecting 'int32'");
    }

    void *data_ptr = dst.get_data();
    const auto &ctx = exec_q.get_context();
    auto usm_type = sycl::get_pointer_type(data_ptr, ctx);

    bool supports_atomics = check_atomic_support(exec_q, usm_type);
    if (!supports_atomics) {
        throw py::value_error(
            "This reduction is not supported for this device and usm_type.");
    }

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    // TODO: should be dst_nelems == 0?
    if ((is_src_c_contig && is_dst_c_contig) ||
        (is_src_f_contig && dst_nelems == 0))
    {
        auto fn = axis1_contig_dispatch_vector[src_typeid];
        static constexpr py::ssize_t zero_offset = 0;

        sycl::event red_ev =
            fn(exec_q, dst_nelems, red_nelems, src_data, dst_data, zero_offset,
               zero_offset, zero_offset, depends);

        sycl::event keep_args_event =
            dpctl::utils::keep_args_alive(exec_q, {src, dst}, {red_ev});

        return std::make_pair(keep_args_event, red_ev);
    }
    else if (is_src_f_contig &&
             ((is_dst_c_contig && dst_nd == 1) || dst.is_f_contiguous()))
    {
        auto fn = axis0_contig_dispatch_vector[src_typeid];
        static constexpr py::ssize_t zero_offset = 0;

        sycl::event red_ev =
            fn(exec_q, dst_nelems, red_nelems, src_data, dst_data, zero_offset,
               zero_offset, zero_offset, depends);

        sycl::event keep_args_event =
            dpctl::utils::keep_args_alive(exec_q, {src, dst}, {red_ev});

        return std::make_pair(keep_args_event, red_ev);
    }

    auto src_shape_vecs = src.get_shape_vector();
    auto src_strides_vecs = src.get_strides_vector();
    auto dst_strides_vecs = dst.get_strides_vector();

    int simplified_red_nd = trailing_dims_to_reduce;

    using shT = std::vector<py::ssize_t>;
    shT red_src_strides(std::begin(src_strides_vecs) + dst_nd,
                        std::end(src_strides_vecs));

    shT simplified_red_shape;
    shT simplified_red_src_strides;
    py::ssize_t red_src_offset(0);

    using dpctl::tensor::py_internal::simplify_iteration_space_1;
    simplify_iteration_space_1(
        simplified_red_nd, src_shape_ptr + dst_nd, red_src_strides,
        // output
        simplified_red_shape, simplified_red_src_strides, red_src_offset);

    shT iter_src_strides(std::begin(src_strides_vecs),
                         std::begin(src_strides_vecs) + iter_nd);
    shT const &iter_dst_strides = dst_strides_vecs;

    shT simplified_iter_shape;
    shT simplified_iter_src_strides;
    shT simplified_iter_dst_strides;
    py::ssize_t iter_src_offset(0);
    py::ssize_t iter_dst_offset(0);

    if (iter_nd == 0) {
        if (dst_nelems != 1) {
            throw std::runtime_error("iteration_nd == 0, but dst_nelems != 1");
        }
        iter_nd = 1;
        simplified_iter_shape.push_back(1);
        simplified_iter_src_strides.push_back(0);
        simplified_iter_dst_strides.push_back(0);
    }
    else {
        using dpctl::tensor::py_internal::simplify_iteration_space;
        simplify_iteration_space(
            iter_nd, src_shape_ptr, iter_src_strides, iter_dst_strides,
            // output
            simplified_iter_shape, simplified_iter_src_strides,
            simplified_iter_dst_strides, iter_src_offset, iter_dst_offset);
    }

    if (simplified_red_nd == 1 && iter_nd == 1) {
        bool mat_reduce_over_axis1 = false;
        bool mat_reduce_over_axis0 = false;
        bool array_reduce_all_elems = false;
        std::size_t iter_nelems = dst_nelems;

        if (simplified_red_src_strides[0] == 1) {
            array_reduce_all_elems = (simplified_iter_shape[0] == 1);
            mat_reduce_over_axis1 =
                (simplified_iter_dst_strides[0] == 1) &&
                (static_cast<std::size_t>(simplified_iter_src_strides[0]) ==
                 red_nelems);
        }
        else if (static_cast<std::size_t>(simplified_red_src_strides[0]) ==
                 iter_nelems) {
            mat_reduce_over_axis0 = (simplified_iter_dst_strides[0] == 1) &&
                                    (simplified_iter_src_strides[0] == 1);
        }
        if (mat_reduce_over_axis1 || array_reduce_all_elems) {
            auto fn = axis1_contig_dispatch_vector[src_typeid];

            sycl::event red_ev =
                fn(exec_q, iter_nelems, red_nelems, src_data, dst_data,
                   iter_src_offset, iter_dst_offset, red_src_offset, depends);

            sycl::event keep_args_event =
                dpctl::utils::keep_args_alive(exec_q, {src, dst}, {red_ev});

            return std::make_pair(keep_args_event, red_ev);
        }
        else if (mat_reduce_over_axis0) {
            auto fn = axis0_contig_dispatch_vector[src_typeid];

            sycl::event red_ev =
                fn(exec_q, iter_nelems, red_nelems, src_data, dst_data,
                   iter_src_offset, iter_dst_offset, red_src_offset, depends);

            sycl::event keep_args_event =
                dpctl::utils::keep_args_alive(exec_q, {src, dst}, {red_ev});

            return std::make_pair(keep_args_event, red_ev);
        }
    }

    auto fn = strided_dispatch_vector[src_typeid];

    std::vector<sycl::event> host_task_events{};
    auto iter_red_metadata_packing_triple_ =
        dpctl::tensor::offset_utils::device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, simplified_iter_shape,
            simplified_iter_src_strides, simplified_iter_dst_strides,
            simplified_red_shape, simplified_red_src_strides);
    auto packed_shapes_strides_owner =
        std::move(std::get<0>(iter_red_metadata_packing_triple_));
    const auto &copy_metadata_ev =
        std::get<2>(iter_red_metadata_packing_triple_);
    const py::ssize_t *packed_shapes_and_strides =
        packed_shapes_strides_owner.get();

    const py::ssize_t *iter_shape_and_strides = packed_shapes_and_strides;
    const py::ssize_t *red_shape_stride =
        packed_shapes_and_strides + 3 * simplified_iter_shape.size();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.resize(depends.size());
    std::copy(depends.begin(), depends.end(), all_deps.begin());
    all_deps.push_back(copy_metadata_ev);

    auto red_ev =
        fn(exec_q, dst_nelems, red_nelems, src_data, dst_data, iter_nd,
           iter_shape_and_strides, iter_src_offset, iter_dst_offset,
           simplified_red_nd, red_shape_stride, red_src_offset, all_deps);

    sycl::event temp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        exec_q, {red_ev}, packed_shapes_strides_owner);
    host_task_events.push_back(temp_cleanup_ev);

    sycl::event keep_args_event =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events);

    return std::make_pair(keep_args_event, red_ev);
}

extern void init_reduction_functions(py::module_ m);

} // namespace dpctl::tensor::py_internal
