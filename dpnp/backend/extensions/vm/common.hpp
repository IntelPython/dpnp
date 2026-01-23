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

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <oneapi/mkl.hpp>
#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/type_dispatch.hpp"

/**
 * Version of Intel MKL at which transition to OneMKL release 2023.2.0 occurs.
 *
 * @note with OneMKL=2023.1.0 the call of oneapi::mkl::vm::div() was dead
 * locked inside ~usm_wrapper_to_host()->{...; q_->wait_and_throw(); ...}
 */
#ifndef __INTEL_MKL_2023_2_0_VERSION_REQUIRED
#define __INTEL_MKL_2023_2_0_VERSION_REQUIRED 20230002L
#endif

static_assert(INTEL_MKL_VERSION >= __INTEL_MKL_2023_2_0_VERSION_REQUIRED,
              "OneMKL does not meet minimum version requirement");

namespace ext_ns = ext::common;
namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpnp::extensions::vm::py_internal
{
template <typename output_typesT, typename contig_dispatchT>
bool need_to_call_unary_ufunc(sycl::queue &exec_q,
                              const dpctl::tensor::usm_ndarray &src,
                              const dpctl::tensor::usm_ndarray &dst,
                              const output_typesT &output_type_vec,
                              const contig_dispatchT &contig_dispatch_vector)
{
    // check type_nums
    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    // check that types are supported
    int func_output_typeid = output_type_vec[src_typeid];
    if (dst_typeid != func_output_typeid) {
        return false;
    }

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
    if (contig_dispatch_vector[src_typeid] == nullptr) {
        return false;
    }
    return true;
}

template <typename output_typesT, typename contig_dispatchT>
bool need_to_call_unary_two_outputs_ufunc(
    sycl::queue &exec_q,
    const dpctl::tensor::usm_ndarray &src,
    const dpctl::tensor::usm_ndarray &dst1,
    const dpctl::tensor::usm_ndarray &dst2,
    const output_typesT &output_type_vec,
    const contig_dispatchT &contig_dispatch_vector)
{
    // check type_nums
    int src_typenum = src.get_typenum();
    int dst1_typenum = dst1.get_typenum();
    int dst2_typenum = dst2.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst1_typeid = array_types.typenum_to_lookup_id(dst1_typenum);
    int dst2_typeid = array_types.typenum_to_lookup_id(dst2_typenum);

    std::pair<int, int> func_output_typeids = output_type_vec[src_typeid];

    // check that types are supported
    if (dst1_typeid != func_output_typeids.first ||
        dst2_typeid != func_output_typeids.second)
    {
        return false;
    }

    // OneMKL VM functions perform a copy on host if no double type support
    if (!exec_q.get_device().has(sycl::aspect::fp64)) {
        return false;
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst1, dst2})) {
        return false;
    }

    // dimensions must be the same
    int src_nd = src.get_ndim();
    int dst1_nd = dst1.get_ndim();
    int dst2_nd = dst2.get_ndim();
    if (src_nd != dst1_nd || src_nd != dst2_nd) {
        return false;
    }
    else if (dst1_nd == 0 || dst2_nd == 0) {
        // don't call OneMKL for 0d arrays
        return false;
    }

    // shapes must be the same
    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst1_shape = dst1.get_shape_raw();
    const py::ssize_t *dst2_shape = dst2.get_shape_raw();
    bool shapes_equal(true);
    size_t src_nelems(1);

    for (int i = 0; i < src_nd; ++i) {
        src_nelems *= static_cast<std::size_t>(src_shape[i]);
        shapes_equal = shapes_equal && (src_shape[i] == dst1_shape[i]) &&
                       (src_shape[i] == dst2_shape[i]);
    }
    if (!shapes_equal) {
        return false;
    }

    // if nelems is zero, return false
    if (src_nelems == 0) {
        return false;
    }

    // ensure that outputs are ample enough to accommodate all elements
    auto dst1_offsets = dst1.get_minmax_offsets();
    auto dst2_offsets = dst2.get_minmax_offsets();
    // destinations must be ample enough to accommodate all elements
    {
        size_t range1 =
            static_cast<size_t>(dst1_offsets.second - dst1_offsets.first);
        size_t range2 =
            static_cast<size_t>(dst2_offsets.second - dst2_offsets.first);
        if ((range1 + 1 < src_nelems) || (range2 + 1 < src_nelems)) {
            return false;
        }
    }

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst1) || overlap(src, dst2) || overlap(dst1, dst2)) {
        return false;
    }

    // support only contiguous inputs
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst1_c_contig = dst1.is_c_contiguous();
    bool is_dst2_c_contig = dst2.is_c_contiguous();

    bool all_c_contig =
        (is_src_c_contig && is_dst1_c_contig && is_dst2_c_contig);
    if (!all_c_contig) {
        return false;
    }

    // MKL function is not defined for the type
    if (contig_dispatch_vector[src_typeid] == nullptr) {
        return false;
    }
    return true;
}

template <typename output_typesT, typename contig_dispatchT>
bool need_to_call_binary_ufunc(sycl::queue &exec_q,
                               const dpctl::tensor::usm_ndarray &src1,
                               const dpctl::tensor::usm_ndarray &src2,
                               const dpctl::tensor::usm_ndarray &dst,
                               const output_typesT &output_type_table,
                               const contig_dispatchT &contig_dispatch_table)
{
    // check type_nums
    int src1_typenum = src1.get_typenum();
    int src2_typenum = src2.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int src1_typeid = array_types.typenum_to_lookup_id(src1_typenum);
    int src2_typeid = array_types.typenum_to_lookup_id(src2_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    // check that types are supported
    int output_typeid = output_type_table[src1_typeid][src2_typeid];
    if (output_typeid != dst_typeid) {
        return false;
    }

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
    if (contig_dispatch_table[src1_typeid] == nullptr) {
        return false;
    }
    return true;
}

/**
 * @brief A macro used to define factories and a populating unary functions
 * to dispatch to a callback with proper OneMKL function within VM extension
 * scope.
 */
#define MACRO_POPULATE_DISPATCH_VECTORS(__name__)                              \
    template <typename fnT, typename T>                                        \
    struct ContigFactory                                                       \
    {                                                                          \
        fnT get()                                                              \
        {                                                                      \
            if constexpr (std::is_same_v<typename OutputType<T>::value_type,   \
                                         void>) {                              \
                return nullptr;                                                \
            }                                                                  \
            else {                                                             \
                return __name__##_contig_impl<T>;                              \
            }                                                                  \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <typename fnT, typename T>                                        \
    struct TypeMapFactory                                                      \
    {                                                                          \
        std::enable_if_t<std::is_same<fnT, int>::value, int> get()             \
        {                                                                      \
            using rT = typename OutputType<T>::value_type;                     \
            return td_ns::GetTypeid<rT>{}.get();                               \
        }                                                                      \
    };                                                                         \
                                                                               \
    static void populate_dispatch_vectors(void)                                \
    {                                                                          \
        ext_ns::init_dispatch_vector<int, TypeMapFactory>(                     \
            output_typeid_vector);                                             \
        ext_ns::init_dispatch_vector<unary_contig_impl_fn_ptr_t,               \
                                     ContigFactory>(contig_dispatch_vector);   \
    };

/**
 * @brief A macro used to define factories and a populating unary functions
 * with two output arrays to dispatch to a callback with proper OneMKL function
 * within VM extension scope.
 */
#define MACRO_POPULATE_DISPATCH_2OUTS_VECTORS(__name__)                        \
    template <typename fnT, typename T>                                        \
    struct ContigFactory                                                       \
    {                                                                          \
        fnT get()                                                              \
        {                                                                      \
            if constexpr (std::is_same_v<typename OutputType<T>::value_type1,  \
                                         void> ||                              \
                          std::is_same_v<typename OutputType<T>::value_type2,  \
                                         void>)                                \
            {                                                                  \
                fnT fn = nullptr;                                              \
                return fn;                                                     \
            }                                                                  \
            else {                                                             \
                fnT fn = __name__##_contig_impl<T>;                            \
                return fn;                                                     \
            }                                                                  \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <typename fnT, typename T>                                        \
    struct TypeMapFactory                                                      \
    {                                                                          \
        std::enable_if_t<std::is_same<fnT, std::pair<int, int>>::value,        \
                         std::pair<int, int>>                                  \
            get()                                                              \
        {                                                                      \
            using rT1 = typename OutputType<T>::value_type1;                   \
            using rT2 = typename OutputType<T>::value_type2;                   \
            return std::make_pair(td_ns::GetTypeid<rT1>{}.get(),               \
                                  td_ns::GetTypeid<rT2>{}.get());              \
        }                                                                      \
    };                                                                         \
                                                                               \
    static void populate_dispatch_vectors(void)                                \
    {                                                                          \
        ext_ns::init_dispatch_vector<std::pair<int, int>, TypeMapFactory>(     \
            output_typeid_vector);                                             \
        ext_ns::init_dispatch_vector<unary_two_outputs_contig_impl_fn_ptr_t,   \
                                     ContigFactory>(contig_dispatch_vector);   \
    };

/**
 * @brief A macro used to define factories and a populating binary functions
 * to dispatch to a callback with proper OneMKL function within VM extension
 * scope.
 */
#define MACRO_POPULATE_DISPATCH_TABLES(__name__)                               \
    template <typename fnT, typename T1, typename T2>                          \
    struct ContigFactory                                                       \
    {                                                                          \
        fnT get()                                                              \
        {                                                                      \
            if constexpr (std::is_same_v<                                      \
                              typename OutputType<T1, T2>::value_type, void>)  \
            {                                                                  \
                return nullptr;                                                \
            }                                                                  \
            else {                                                             \
                return __name__##_contig_impl<T1, T2>;                         \
            }                                                                  \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <typename fnT, typename T1, typename T2>                          \
    struct TypeMapFactory                                                      \
    {                                                                          \
        std::enable_if_t<std::is_same<fnT, int>::value, int> get()             \
        {                                                                      \
            using rT = typename OutputType<T1, T2>::value_type;                \
            return td_ns::GetTypeid<rT>{}.get();                               \
        }                                                                      \
    };                                                                         \
                                                                               \
    static void populate_dispatch_tables(void)                                 \
    {                                                                          \
        ext_ns::init_dispatch_table<int, TypeMapFactory>(                      \
            output_typeid_vector);                                             \
        ext_ns::init_dispatch_table<binary_contig_impl_fn_ptr_t,               \
                                    ContigFactory>(contig_dispatch_vector);    \
    };
} // namespace dpnp::extensions::vm::py_internal
