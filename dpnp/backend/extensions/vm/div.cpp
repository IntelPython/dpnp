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

#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/type_utils.hpp"

#include "div.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace vm
{
namespace mkl_vm     = oneapi::mkl::vm;
namespace py         = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*div_impl_fn_ptr_t)(sycl::queue,
                                         const std::int64_t,
                                         const char *,
                                         const char *,
                                         char *,
                                         const std::vector<sycl::event> &);

static div_impl_fn_ptr_t div_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event div_impl(sycl::queue exec_q,
                            const std::int64_t n,
                            const char *in_a,
                            const char *in_b,
                            char *out_y,
                            const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    const T *a = reinterpret_cast<const T *>(in_a);
    const T *b = reinterpret_cast<const T *>(in_b);
    T *y       = reinterpret_cast<T *>(out_y);

    return mkl_vm::div(exec_q,
                       n, // number of elements to be calculated
                       a, // pointer `a` containing 1st input vector of size n
                       b, // pointer `b` containing 2nd input vector of size n
                       y, // pointer `y` to the output vector of size n
                       depends);
}

std::pair<sycl::event, sycl::event>
    div(sycl::queue exec_q,
        dpctl::tensor::usm_ndarray src1,
        dpctl::tensor::usm_ndarray src2,
        dpctl::tensor::usm_ndarray dst, // dst = op(src1, src2), elementwise
        const std::vector<sycl::event> &depends)
{
    // check type_nums
    int src1_typenum = src1.get_typenum();
    int src2_typenum = src2.get_typenum();
    int dst_typenum  = dst.get_typenum();

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int src1_typeid  = array_types.typenum_to_lookup_id(src1_typenum);
    int src2_typeid  = array_types.typenum_to_lookup_id(src2_typenum);
    int dst_typeid   = array_types.typenum_to_lookup_id(dst_typenum);

    if (src1_typeid != src2_typeid || src2_typeid != dst_typeid) {
        throw py::value_error(
            "Either any of input arrays or output array have different types");
    }

    // check that queues are compatible
    if (!dpctl::utils::queues_are_compatible(exec_q, {src1, src2, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
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
    const py::ssize_t *dst_shape  = dst.get_shape_raw();
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

    // ensure that output is ample enough to accomodate all elements
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
    {
        size_t range =
            static_cast<size_t>(dst_offsets.second - dst_offsets.first);
        if (range + 1 < src_nelems) {
            throw py::value_error(
                "Destination array can not accomodate all the "
                "elements of source array.");
        }
    }

    // check memory overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src1, dst) || overlap(src2, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    const char *src1_data = src1.get_data();
    const char *src2_data = src2.get_data();
    char *dst_data        = dst.get_data();

    // handle contiguous inputs
    bool is_src1_c_contig = src1.is_c_contiguous();
    bool is_src2_c_contig = src2.is_c_contiguous();
    bool is_dst_c_contig  = dst.is_c_contiguous();

    bool all_c_contig =
        (is_src1_c_contig && is_src2_c_contig && is_dst_c_contig);
    if (!all_c_contig) {
        throw py::value_error("Input and outpur arrays must be C-contiguous");
    }

    auto div_fn = div_dispatch_vector[dst_typeid];
    if (div_fn == nullptr) {
        throw py::value_error("No div implementation defined");
    }
    sycl::event sum_ev =
        div_fn(exec_q, src_nelems, src1_data, src2_data, dst_data, depends);

    sycl::event ht_ev =
        dpctl::utils::keep_args_alive(exec_q, {src1, src2, dst}, {sum_ev});
    return std::make_pair(ht_ev, sum_ev);
    return std::make_pair(sycl::event(), sycl::event());
}

bool can_call_div(sycl::queue exec_q,
                  dpctl::tensor::usm_ndarray src1,
                  dpctl::tensor::usm_ndarray src2,
                  dpctl::tensor::usm_ndarray dst)
{
#if INTEL_MKL_VERSION >= 20230002
    // check type_nums
    int src1_typenum = src1.get_typenum();
    int src2_typenum = src2.get_typenum();
    int dst_typenum  = dst.get_typenum();

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int src1_typeid  = array_types.typenum_to_lookup_id(src1_typenum);
    int src2_typeid  = array_types.typenum_to_lookup_id(src2_typenum);
    int dst_typeid   = array_types.typenum_to_lookup_id(dst_typenum);

    // types must be the same
    if (src1_typeid != src2_typeid || src2_typeid != dst_typeid) {
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
    const py::ssize_t *dst_shape  = dst.get_shape_raw();
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

    // ensure that output is ample enough to accomodate all elements
    auto dst_offsets = dst.get_minmax_offsets();
    // destination must be ample enough to accomodate all elements
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

    // suppport only contiguous inputs
    bool is_src1_c_contig = src1.is_c_contiguous();
    bool is_src2_c_contig = src2.is_c_contiguous();
    bool is_dst_c_contig  = dst.is_c_contiguous();

    bool all_c_contig =
        (is_src1_c_contig && is_src2_c_contig && is_dst_c_contig);
    if (!all_c_contig) {
        return false;
    }

    // MKL function is not defined for the type
    if (div_dispatch_vector[src1_typeid] == nullptr) {
        return false;
    }
    return true;
#else
    // In OneMKL 2023.1.0 the call of oneapi::mkl::vm::div() is going to dead
    // lock inside ~usm_wrapper_to_host()->{...; q_->wait_and_throw(); ...}

    (void)exec_q;
    (void)src1;
    (void)src2;
    (void)dst;
    return false;
#endif // INTEL_MKL_VERSION >= 20230002
}

template <typename fnT, typename T>
struct DivContigFactory
{
    fnT get()
    {
        if constexpr (types::DivTypePairSupportFactory<T>::is_defined) {
            return div_impl<T>;
        }
        else {
            return nullptr;
        }
    }
};

void init_div_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<div_impl_fn_ptr_t, DivContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(div_dispatch_vector);
}
} // namespace vm
} // namespace ext
} // namespace backend
} // namespace dpnp
