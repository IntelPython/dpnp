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
/// This file defines functions of dpctl.tensor._tensor_elementwise_impl
/// extension, specifically functions for elementwise operations.
//===---------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "elementwise_functions_type_utils.hpp"
#include "kernels/alignment.hpp"
#include "kernels/dpctl_tensor_types.hpp"
#include "simplify_iteration_space.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

static_assert(std::is_same_v<py::ssize_t, dpctl::tensor::ssize_t>);

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

/*! @brief Template implementing Python API for unary elementwise functions */
template <typename output_typesT,
          typename contig_dispatchT,
          typename strided_dispatchT>
std::pair<sycl::event, sycl::event>
    py_unary_ufunc(const dpctl::tensor::usm_ndarray &src,
                   const dpctl::tensor::usm_ndarray &dst,
                   sycl::queue &q,
                   const std::vector<sycl::event> &depends,
                   //
                   const output_typesT &output_type_vec,
                   const contig_dispatchT &contig_dispatch_vector,
                   const strided_dispatchT &strided_dispatch_vector)
{
    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    int func_output_typeid = output_type_vec[src_typeid];

    // check that types are supported
    if (dst_typeid != func_output_typeid) {
        throw py::value_error(
            "Destination array has unexpected elemental data type.");
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    // check that dimensions are the same
    int src_nd = src.get_ndim();
    if (src_nd != dst.get_ndim()) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // check that shapes are the same
    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    std::size_t src_nelems(1);

    for (int i = 0; i < src_nd; ++i) {
        src_nelems *= static_cast<std::size_t>(src_shape[i]);
        shapes_equal = shapes_equal && (src_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    // if nelems is zero, return
    if (src_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, src_nelems);

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    auto const &same_logical_tensors =
        dpctl::tensor::overlap::SameLogicalTensors();
    if (overlap(src, dst) && !same_logical_tensors(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    // handle contiguous inputs
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    bool both_c_contig = (is_src_c_contig && is_dst_c_contig);
    bool both_f_contig = (is_src_f_contig && is_dst_f_contig);

    if (both_c_contig || both_f_contig) {
        auto contig_fn = contig_dispatch_vector[src_typeid];

        if (contig_fn == nullptr) {
            throw std::runtime_error(
                "Contiguous implementation is missing for src_typeid=" +
                std::to_string(src_typeid));
        }

        auto comp_ev = contig_fn(q, src_nelems, src_data, dst_data, depends);
        sycl::event ht_ev =
            dpctl::utils::keep_args_alive(q, {src, dst}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    // simplify iteration space
    //     if 1d with strides 1 - input is contig
    //     dispatch to strided

    auto const &src_strides = src.get_strides_vector();
    auto const &dst_strides = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_nd;
    const py::ssize_t *shape = src_shape;

    dpctl::tensor::py_internal::simplify_iteration_space(
        nd, shape, src_strides, dst_strides,
        // output
        simplified_shape, simplified_src_strides, simplified_dst_strides,
        src_offset, dst_offset);

    if (nd == 1 && simplified_src_strides[0] == 1 &&
        simplified_dst_strides[0] == 1) {
        // Special case of contiguous data
        auto contig_fn = contig_dispatch_vector[src_typeid];

        if (contig_fn == nullptr) {
            throw std::runtime_error(
                "Contiguous implementation is missing for src_typeid=" +
                std::to_string(src_typeid));
        }

        int src_elem_size = src.get_elemsize();
        int dst_elem_size = dst.get_elemsize();
        auto comp_ev =
            contig_fn(q, src_nelems, src_data + src_elem_size * src_offset,
                      dst_data + dst_elem_size * dst_offset, depends);

        sycl::event ht_ev =
            dpctl::utils::keep_args_alive(q, {src, dst}, {comp_ev});

        return std::make_pair(ht_ev, comp_ev);
    }

    // Strided implementation
    auto strided_fn = strided_dispatch_vector[src_typeid];

    if (strided_fn == nullptr) {
        throw std::runtime_error(
            "Strided implementation is missing for src_typeid=" +
            std::to_string(src_typeid));
    }

    using dpctl::tensor::offset_utils::device_allocate_and_pack;

    std::vector<sycl::event> host_tasks{};
    host_tasks.reserve(2);

    auto ptr_size_event_triple_ = device_allocate_and_pack<py::ssize_t>(
        q, host_tasks, simplified_shape, simplified_src_strides,
        simplified_dst_strides);
    auto shape_strides_owner = std::move(std::get<0>(ptr_size_event_triple_));
    const auto &copy_shape_ev = std::get<2>(ptr_size_event_triple_);
    const py::ssize_t *shape_strides = shape_strides_owner.get();

    sycl::event strided_fn_ev =
        strided_fn(q, src_nelems, nd, shape_strides, src_data, src_offset,
                   dst_data, dst_offset, depends, {copy_shape_ev});

    // async free of shape_strides temporary
    sycl::event tmp_cleanup_ev = dpctl::tensor::alloc_utils::async_smart_free(
        q, {strided_fn_ev}, shape_strides_owner);

    host_tasks.push_back(tmp_cleanup_ev);

    return std::make_pair(
        dpctl::utils::keep_args_alive(q, {src, dst}, host_tasks),
        strided_fn_ev);
}

/*! @brief Template implementing Python API for querying of type support by
 *         unary elementwise functions */
template <typename output_typesT>
py::object py_unary_ufunc_result_type(const py::dtype &input_dtype,
                                      const output_typesT &output_types)
{
    int tn = input_dtype.num(); // NumPy type numbers are the same as in dpctl
    int src_typeid = -1;

    auto array_types = td_ns::usm_ndarray_types();

    try {
        src_typeid = array_types.typenum_to_lookup_id(tn);
    } catch (const std::exception &e) {
        throw py::value_error(e.what());
    }

    using dpctl::tensor::py_internal::type_utils::_result_typeid;
    int dst_typeid = _result_typeid(src_typeid, output_types);

    if (dst_typeid < 0) {
        auto res = py::none();
        return py::cast<py::object>(res);
    }
    else {
        using dpctl::tensor::py_internal::type_utils::_dtype_from_typenum;

        auto dst_typenum_t = static_cast<td_ns::typenum_t>(dst_typeid);
        auto dt = _dtype_from_typenum(dst_typenum_t);

        return py::cast<py::object>(dt);
    }
}

} // namespace dpctl::tensor::py_internal
