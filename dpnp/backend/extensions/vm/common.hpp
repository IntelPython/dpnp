//*****************************************************************************
// Copyright (c) 2023, Intel Corporation
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

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <dpctl4pybind11.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "dpnp_utils.hpp"

static_assert(INTEL_MKL_VERSION >= __INTEL_MKL_2023_2_0_VERSION_REQUIRED,
              "OneMKL does not meet minimum version requirement");

// OneMKL namespace with VM functions
namespace mkl_vm = oneapi::mkl::vm;

// dpctl namespace for type utils
namespace type_utils = dpctl::tensor::type_utils;

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace vm
{
typedef sycl::event (*unary_impl_fn_ptr_t)(sycl::queue,
                                           const std::int64_t,
                                           const char *,
                                           char *,
                                           const std::vector<sycl::event> &);

typedef sycl::event (*binary_impl_fn_ptr_t)(sycl::queue,
                                            const std::int64_t,
                                            const char *,
                                            const char *,
                                            char *,
                                            const std::vector<sycl::event> &);

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace py = pybind11;

template <typename dispatchT>
std::pair<sycl::event, sycl::event>
    unary_ufunc(sycl::queue exec_q,
                dpctl::tensor::usm_ndarray src,
                dpctl::tensor::usm_ndarray dst, // dst = op(src), elementwise
                const std::vector<sycl::event> &depends,
                const dispatchT &dispatch_vector)
{
    // check type_nums
    int src_typenum = src.get_typenum();
    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues.");
    }

    // check that dimensions are the same
    int dst_nd = dst.get_ndim();
    if (dst_nd != src.get_ndim()) {
        throw py::value_error(
            "Input and output arrays have have different dimensions.");
    }

    // check that shapes are the same
    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    size_t src_nelems(1);

    for (int i = 0; i < dst_nd; ++i) {
        src_nelems *= static_cast<size_t>(src_shape[i]);
        shapes_equal = shapes_equal && (src_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Input and output arrays have different shapes.");
    }

    // if nelems is zero, return
    if (src_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    // ensure that output is ample enough to accommodate all elements
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accommodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accommodate all the elements "
                "of source array.");
        }
    }

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory.");
    }

    const char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    // handle contiguous inputs
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    bool all_c_contig = (is_src_c_contig && is_dst_c_contig);
    if (!all_c_contig) {
        throw py::value_error("Input and outpur arrays must be C-contiguous.");
    }

    auto dispatch_fn = dispatch_vector[src_typeid];
    if (dispatch_fn == nullptr) {
        throw py::value_error("No implementation is defined for ufunc.");
    }
    sycl::event comp_ev =
        dispatch_fn(exec_q, src_nelems, src_data, dst_data, depends);

    sycl::event ht_ev =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, {comp_ev});
    return std::make_pair(ht_ev, comp_ev);
}

template <typename dispatchT>
std::pair<sycl::event, sycl::event> binary_ufunc(
    sycl::queue exec_q,
    dpctl::tensor::usm_ndarray src1,
    dpctl::tensor::usm_ndarray src2,
    dpctl::tensor::usm_ndarray dst, // dst = op(src1, src2), elementwise
    const std::vector<sycl::event> &depends,
    const dispatchT &dispatch_vector)
{
    // check type_nums
    int src1_typenum = src1.get_typenum();
    int src2_typenum = src2.get_typenum();

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int src1_typeid = array_types.typenum_to_lookup_id(src1_typenum);
    int src2_typeid = array_types.typenum_to_lookup_id(src2_typenum);

    if (src1_typeid != src2_typeid) {
        throw py::value_error("Input arrays have different types.");
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {src1, src2, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues.");
    }

    // check shapes, broadcasting is assumed done by caller
    // check that dimensions are the same
    int dst_nd = dst.get_ndim();
    if (dst_nd != src1.get_ndim() || dst_nd != src2.get_ndim()) {
        throw py::value_error("Array dimensions are not the same.");
    }

    // check that shapes are the same
    const py::ssize_t *src1_shape = src1.get_shape_raw();
    const py::ssize_t *src2_shape = src2.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    size_t src_nelems(1);

    for (int i = 0; i < dst_nd; ++i) {
        src_nelems *= static_cast<size_t>(src1_shape[i]);
        shapes_equal = shapes_equal && (src1_shape[i] == dst_shape[i] &&
                                        src2_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        throw py::value_error("Array shapes are not the same.");
    }

    // if nelems is zero, return
    if (src_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    // ensure that output is ample enough to accommodate all elements
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accommodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accommodate all the "
                "elements of source array.");
        }
    }

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src1, dst) || overlap(src2, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory.");
    }

    const char *src1_data = src1.get_data();
    const char *src2_data = src2.get_data();
    char *dst_data = dst.get_data();

    // handle contiguous inputs
    bool is_src1_c_contig = src1.is_c_contiguous();
    bool is_src2_c_contig = src2.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    bool all_c_contig =
        (is_src1_c_contig && is_src2_c_contig && is_dst_c_contig);
    if (!all_c_contig) {
        throw py::value_error("Input and outpur arrays must be C-contiguous.");
    }

    auto dispatch_fn = dispatch_vector[src1_typeid];
    if (dispatch_fn == nullptr) {
        throw py::value_error("No implementation is defined for ufunc.");
    }
    sycl::event comp_ev = dispatch_fn(exec_q, src_nelems, src1_data, src2_data,
                                      dst_data, depends);

    sycl::event ht_ev =
        dpctl::utils::keep_args_alive(exec_q, {src1, src2, dst}, {comp_ev});
    return std::make_pair(ht_ev, comp_ev);
}

template <typename dispatchT>
bool need_to_call_unary_ufunc(sycl::queue exec_q,
                              dpctl::tensor::usm_ndarray src,
                              dpctl::tensor::usm_ndarray dst,
                              const dispatchT &dispatch_vector)
{
    // check type_nums
    int src_typenum = src.get_typenum();
    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);

    // OneMKL VM functions perform a copy on host if no double type support
    if (!exec_q.get_device().has(sycl::aspect::fp64)) {
        return false;
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        return false;
    }

    // dimensions must be the same
    int dst_nd = dst.get_ndim();
    if (dst_nd != src.get_ndim()) {
        return false;
    }
    else if (dst_nd == 0) {
        // don't call OneMKL for 0d arrays
        return false;
    }

    // shapes must be the same
    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    size_t src_nelems(1);

    for (int i = 0; i < dst_nd; ++i) {
        src_nelems *= static_cast<size_t>(src_shape[i]);
        shapes_equal = shapes_equal && (src_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        return false;
    }

    // if nelems is zero, return false
    if (src_nelems == 0) {
        return false;
    }

    // ensure that output is ample enough to accommodate all elements
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accommodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            return false;
        }
    }

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        return false;
    }

    // support only contiguous inputs
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    bool all_c_contig = (is_src_c_contig && is_dst_c_contig);
    if (!all_c_contig) {
        return false;
    }

    // MKL function is not defined for the type
    if (dispatch_vector[src_typeid] == nullptr) {
        return false;
    }
    return true;
}

template <typename dispatchT>
bool need_to_call_binary_ufunc(sycl::queue exec_q,
                               dpctl::tensor::usm_ndarray src1,
                               dpctl::tensor::usm_ndarray src2,
                               dpctl::tensor::usm_ndarray dst,
                               const dispatchT &dispatch_vector)
{
    // check type_nums
    int src1_typenum = src1.get_typenum();
    int src2_typenum = src2.get_typenum();

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int src1_typeid = array_types.typenum_to_lookup_id(src1_typenum);
    int src2_typeid = array_types.typenum_to_lookup_id(src2_typenum);

    // types must be the same
    if (src1_typeid != src2_typeid) {
        return false;
    }

    // OneMKL VM functions perform a copy on host if no double type support
    if (!exec_q.get_device().has(sycl::aspect::fp64)) {
        return false;
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {src1, src2, dst})) {
        return false;
    }

    // dimensions must be the same
    int dst_nd = dst.get_ndim();
    if (dst_nd != src1.get_ndim() || dst_nd != src2.get_ndim()) {
        return false;
    }
    else if (dst_nd == 0) {
        // don't call OneMKL for 0d arrays
        return false;
    }

    // shapes must be the same
    const py::ssize_t *src1_shape = src1.get_shape_raw();
    const py::ssize_t *src2_shape = src2.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool shapes_equal(true);
    size_t src_nelems(1);

    for (int i = 0; i < dst_nd; ++i) {
        src_nelems *= static_cast<size_t>(src1_shape[i]);
        shapes_equal = shapes_equal && (src1_shape[i] == dst_shape[i] &&
                                        src2_shape[i] == dst_shape[i]);
    }
    if (!shapes_equal) {
        return false;
    }

    // if nelems is zero, return false
    if (src_nelems == 0) {
        return false;
    }

    // ensure that output is ample enough to accommodate all elements
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accommodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            return false;
        }
    }

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src1, dst) || overlap(src2, dst)) {
        return false;
    }

    // support only contiguous inputs
    bool is_src1_c_contig = src1.is_c_contiguous();
    bool is_src2_c_contig = src2.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    bool all_c_contig =
        (is_src1_c_contig && is_src2_c_contig && is_dst_c_contig);
    if (!all_c_contig) {
        return false;
    }

    // MKL function is not defined for the type
    if (dispatch_vector[src1_typeid] == nullptr) {
        return false;
    }
    return true;
}

template <typename dispatchT,
          template <typename fnT, typename T>
          typename factoryT>
void init_ufunc_dispatch_vector(dispatchT dispatch_vector[])
{
    dpctl_td_ns::DispatchVectorBuilder<dispatchT, factoryT,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(dispatch_vector);
}
} // namespace vm
} // namespace ext
} // namespace backend
} // namespace dpnp
