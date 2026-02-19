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
//===--------------------------------------------------------------------===//
///
/// \file
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/repeat.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

#include "simplify_iteration_space.hpp"

namespace dpctl::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::repeat::repeat_by_sequence_fn_ptr_t;
static repeat_by_sequence_fn_ptr_t
    repeat_by_sequence_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::repeat::repeat_by_sequence_1d_fn_ptr_t;
static repeat_by_sequence_1d_fn_ptr_t
    repeat_by_sequence_1d_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::repeat::repeat_by_scalar_fn_ptr_t;
static repeat_by_scalar_fn_ptr_t
    repeat_by_scalar_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::repeat::repeat_by_scalar_1d_fn_ptr_t;
static repeat_by_scalar_1d_fn_ptr_t
    repeat_by_scalar_1d_dispatch_vector[td_ns::num_types];

void init_repeat_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::repeat::RepeatSequenceFactory;
    td_ns::DispatchVectorBuilder<repeat_by_sequence_fn_ptr_t,
                                 RepeatSequenceFactory, td_ns::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(repeat_by_sequence_dispatch_vector);

    using dpctl::tensor::kernels::repeat::RepeatSequence1DFactory;
    td_ns::DispatchVectorBuilder<repeat_by_sequence_1d_fn_ptr_t,
                                 RepeatSequence1DFactory, td_ns::num_types>
        dvb2;
    dvb2.populate_dispatch_vector(repeat_by_sequence_1d_dispatch_vector);

    using dpctl::tensor::kernels::repeat::RepeatScalarFactory;
    td_ns::DispatchVectorBuilder<repeat_by_scalar_fn_ptr_t, RepeatScalarFactory,
                                 td_ns::num_types>
        dvb3;
    dvb3.populate_dispatch_vector(repeat_by_scalar_dispatch_vector);

    using dpctl::tensor::kernels::repeat::RepeatScalar1DFactory;
    td_ns::DispatchVectorBuilder<repeat_by_scalar_1d_fn_ptr_t,
                                 RepeatScalar1DFactory, td_ns::num_types>
        dvb4;
    dvb4.populate_dispatch_vector(repeat_by_scalar_1d_dispatch_vector);
}

std::pair<sycl::event, sycl::event>
    py_repeat_by_sequence(const dpctl::tensor::usm_ndarray &src,
                          const dpctl::tensor::usm_ndarray &dst,
                          const dpctl::tensor::usm_ndarray &reps,
                          const dpctl::tensor::usm_ndarray &cumsum,
                          int axis,
                          sycl::queue &exec_q,
                          const std::vector<sycl::event> &depends)
{
    int src_nd = src.get_ndim();
    if (axis < 0 || (axis + 1 > src_nd && src_nd > 0) ||
        (axis > 0 && src_nd == 0)) {
        throw py::value_error("Specified axis is invalid.");
    }

    int dst_nd = dst.get_ndim();
    if ((src_nd != dst_nd && src_nd > 0) || (src_nd == 0 && dst_nd > 1)) {
        throw py::value_error("Number of dimensions of source and destination "
                              "arrays is not consistent");
    }

    int reps_nd = reps.get_ndim();
    if (reps_nd != 1) {
        throw py::value_error("`reps` array must be 1-dimensional");
    }

    if (cumsum.get_ndim() != 1) {
        throw py::value_error("`cumsum` array must be 1-dimensional.");
    }

    if (!cumsum.is_c_contiguous()) {
        throw py::value_error("Expecting `cumsum` array to be C-contiguous.");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, reps, cumsum, dst}))
    {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    std::size_t reps_sz = reps.get_size();
    std::size_t cumsum_sz = cumsum.get_size();

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool same_orthog_dims(true);
    std::size_t orthog_nelems(1); // number of orthogonal iterations
    for (auto i = 0; i < axis; ++i) {
        auto src_sh_i = src_shape[i];
        orthog_nelems *= src_sh_i;
        same_orthog_dims = same_orthog_dims && (src_sh_i == dst_shape[i]);
    }
    for (auto i = axis + 1; i < src_nd; ++i) {
        auto src_sh_i = src_shape[i];
        orthog_nelems *= src_sh_i;
        same_orthog_dims = same_orthog_dims && (src_sh_i == dst_shape[i]);
    }

    std::size_t src_axis_nelems(1);
    if (src_nd > 0) {
        src_axis_nelems = src_shape[axis];
    }
    std::size_t dst_axis_nelems(dst_shape[axis]);

    // shape at repeated axis must be equal to the sum of reps
    if (!same_orthog_dims || src_axis_nelems != reps_sz ||
        src_axis_nelems != cumsum_sz)
    {
        throw py::value_error("Inconsistent array dimensions");
    }

    if (orthog_nelems == 0 || src_axis_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, orthog_nelems * dst_axis_nelems);

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // check that dst does not intersect with src or reps
    if (overlap(dst, src) || overlap(dst, reps) || overlap(dst, cumsum)) {
        throw py::value_error("Destination array overlaps with inputs");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();
    int reps_typenum = reps.get_typenum();
    int cumsum_typenum = cumsum.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);
    int reps_typeid = array_types.typenum_to_lookup_id(reps_typenum);
    int cumsum_typeid = array_types.typenum_to_lookup_id(cumsum_typenum);

    if (src_typeid != dst_typeid) {
        throw py::value_error(
            "Destination array must have the same elemental data type");
    }

    static constexpr int int64_typeid =
        static_cast<int>(td_ns::typenum_t::INT64);
    if (cumsum_typeid != int64_typeid) {
        throw py::value_error(
            "Unexpected data type of `cumsum` array, expecting "
            "'int64'");
    }

    if (reps_typeid != cumsum_typeid) {
        throw py::value_error("`reps` array must have the same elemental "
                              "data type as cumsum");
    }

    const char *src_data_p = src.get_data();
    const char *reps_data_p = reps.get_data();
    const char *cumsum_data_p = cumsum.get_data();
    char *dst_data_p = dst.get_data();

    auto src_shape_vec = src.get_shape_vector();
    auto src_strides_vec = src.get_strides_vector();

    auto dst_shape_vec = dst.get_shape_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    auto reps_shape_vec = reps.get_shape_vector();
    auto reps_strides_vec = reps.get_strides_vector();

    sycl::event repeat_ev;
    std::vector<sycl::event> host_task_events{};
    if (axis == 0 && src_nd < 2) {
        // empty orthogonal directions

        auto fn = repeat_by_sequence_1d_dispatch_vector[src_typeid];

        assert(dst_shape_vec.size() == 1);
        assert(dst_strides_vec.size() == 1);

        if (src_nd == 0) {
            src_shape_vec = {0};
            src_strides_vec = {0};
        }

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, src_shape_vec, src_strides_vec);
        auto packed_src_shape_strides_owner =
            std::move(std::get<0>(ptr_size_event_tuple1));
        sycl::event copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple1);
        const py::ssize_t *packed_src_shape_strides =
            packed_src_shape_strides_owner.get();

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_shapes_strides_ev);

        assert(all_deps.size() == depends.size() + 1);

        repeat_ev =
            fn(exec_q, src_axis_nelems, src_data_p, dst_data_p, reps_data_p,
               cumsum_data_p, src_nd, packed_src_shape_strides,
               dst_shape_vec[0], dst_strides_vec[0], reps_shape_vec[0],
               reps_strides_vec[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {repeat_ev}, packed_src_shape_strides_owner);
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }
    else {
        // non-empty orthogonal directions

        auto fn = repeat_by_sequence_dispatch_vector[src_typeid];

        int orthog_nd = src_nd - 1;

        using shT = std::vector<py::ssize_t>;
        shT orthog_src_shape;
        shT orthog_src_strides;
        shT axis_src_shape;
        shT axis_src_stride;
        dpctl::tensor::py_internal::split_iteration_space(
            src_shape_vec, src_strides_vec, axis, axis + 1, orthog_src_shape,
            axis_src_shape, orthog_src_strides, axis_src_stride);

        shT orthog_dst_shape;
        shT orthog_dst_strides;
        shT axis_dst_shape;
        shT axis_dst_stride;
        dpctl::tensor::py_internal::split_iteration_space(
            dst_shape_vec, dst_strides_vec, axis, axis + 1, orthog_dst_shape,
            axis_dst_shape, orthog_dst_strides, axis_dst_stride);

        assert(orthog_src_shape.size() == static_cast<std::size_t>(orthog_nd));
        assert(orthog_dst_shape.size() == static_cast<std::size_t>(orthog_nd));
        assert(std::equal(orthog_src_shape.begin(), orthog_src_shape.end(),
                          orthog_dst_shape.begin()));

        shT simplified_orthog_shape;
        shT simplified_orthog_src_strides;
        shT simplified_orthog_dst_strides;

        const py::ssize_t *_shape = orthog_src_shape.data();

        py::ssize_t orthog_src_offset(0);
        py::ssize_t orthog_dst_offset(0);
        dpctl::tensor::py_internal::simplify_iteration_space(
            orthog_nd, _shape, orthog_src_strides, orthog_dst_strides,
            // output
            simplified_orthog_shape, simplified_orthog_src_strides,
            simplified_orthog_dst_strides, orthog_src_offset,
            orthog_dst_offset);

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, simplified_orthog_shape,
            simplified_orthog_src_strides, simplified_orthog_dst_strides);
        auto packed_shapes_strides_owner =
            std::move(std::get<0>(ptr_size_event_tuple1));
        sycl::event copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple1);
        const py::ssize_t *packed_shapes_strides =
            packed_shapes_strides_owner.get();

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_shapes_strides_ev);

        assert(all_deps.size() == depends.size() + 1);

        repeat_ev = fn(exec_q, orthog_nelems, src_axis_nelems, src_data_p,
                       dst_data_p, reps_data_p, cumsum_data_p,
                       // data to build orthog indexer
                       orthog_nd, packed_shapes_strides, orthog_src_offset,
                       orthog_dst_offset,
                       // data to build indexers along repeated axis in src
                       axis_src_shape[0], axis_src_stride[0],
                       // data to build indexer along repeated axis in dst
                       axis_dst_shape[0], axis_dst_stride[0],
                       // data to build indexer for reps array
                       reps_shape_vec[0], reps_strides_vec[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {repeat_ev}, packed_shapes_strides_owner);
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }

    sycl::event py_obj_management_host_task_ev = dpctl::utils::keep_args_alive(
        exec_q, {src, reps, cumsum, dst}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, repeat_ev);
}

std::pair<sycl::event, sycl::event>
    py_repeat_by_sequence(const dpctl::tensor::usm_ndarray &src,
                          const dpctl::tensor::usm_ndarray &dst,
                          const dpctl::tensor::usm_ndarray &reps,
                          const dpctl::tensor::usm_ndarray &cumsum,
                          sycl::queue &exec_q,
                          const std::vector<sycl::event> &depends)
{

    int dst_nd = dst.get_ndim();
    if (dst_nd != 1) {
        throw py::value_error(
            "`dst` array must be 1-dimensional when repeating a full array");
    }

    int reps_nd = reps.get_ndim();
    if (reps_nd != 1) {
        throw py::value_error("`reps` array must be 1-dimensional");
    }

    if (cumsum.get_ndim() != 1) {
        throw py::value_error("`cumsum` array must be 1-dimensional.");
    }

    if (!cumsum.is_c_contiguous()) {
        throw py::value_error("Expecting `cumsum` array to be C-contiguous.");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, reps, cumsum, dst}))
    {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    std::size_t src_sz = src.get_size();
    std::size_t reps_sz = reps.get_size();
    std::size_t cumsum_sz = cumsum.get_size();

    // shape at repeated axis must be equal to the sum of reps
    if (src_sz != reps_sz || src_sz != cumsum_sz) {
        throw py::value_error("Inconsistent array dimensions");
    }

    if (src_sz == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst,
                                                               dst.get_size());

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // check that dst does not intersect with src, cumsum, or reps
    if (overlap(dst, src) || overlap(dst, reps) || overlap(dst, cumsum)) {
        throw py::value_error("Destination array overlaps with inputs");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();
    int reps_typenum = reps.get_typenum();
    int cumsum_typenum = cumsum.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);
    int reps_typeid = array_types.typenum_to_lookup_id(reps_typenum);
    int cumsum_typeid = array_types.typenum_to_lookup_id(cumsum_typenum);

    if (src_typeid != dst_typeid) {
        throw py::value_error(
            "Destination array must have the same elemental data type");
    }

    static constexpr int int64_typeid =
        static_cast<int>(td_ns::typenum_t::INT64);
    if (cumsum_typeid != int64_typeid) {
        throw py::value_error(
            "Unexpected data type of `cumsum` array, expecting "
            "'int64'");
    }

    if (reps_typeid != cumsum_typeid) {
        throw py::value_error("`reps` array must have the same elemental "
                              "data type as cumsum");
    }

    const char *src_data_p = src.get_data();
    const char *reps_data_p = reps.get_data();
    const char *cumsum_data_p = cumsum.get_data();
    char *dst_data_p = dst.get_data();

    int src_nd = src.get_ndim();
    auto src_shape_vec = src.get_shape_vector();
    auto src_strides_vec = src.get_strides_vector();
    if (src_nd == 0) {
        src_shape_vec = {0};
        src_strides_vec = {0};
    }

    auto dst_shape_vec = dst.get_shape_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    auto reps_shape_vec = reps.get_shape_vector();
    auto reps_strides_vec = reps.get_strides_vector();

    std::vector<sycl::event> host_task_events{};

    auto fn = repeat_by_sequence_1d_dispatch_vector[src_typeid];

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, src_shape_vec, src_strides_vec);
    auto packed_src_shapes_strides_owner =
        std::move(std::get<0>(ptr_size_event_tuple1));
    sycl::event copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple1);
    const py::ssize_t *packed_src_shapes_strides =
        packed_src_shapes_strides_owner.get();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_shapes_strides_ev);

    assert(all_deps.size() == depends.size() + 1);

    sycl::event repeat_ev = fn(
        exec_q, src_sz, src_data_p, dst_data_p, reps_data_p, cumsum_data_p,
        src_nd, packed_src_shapes_strides, dst_shape_vec[0], dst_strides_vec[0],
        reps_shape_vec[0], reps_strides_vec[0], all_deps);

    sycl::event cleanup_tmp_allocations_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {repeat_ev}, packed_src_shapes_strides_owner);
    host_task_events.push_back(cleanup_tmp_allocations_ev);

    sycl::event py_obj_management_host_task_ev = dpctl::utils::keep_args_alive(
        exec_q, {src, reps, cumsum, dst}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, repeat_ev);
}

std::pair<sycl::event, sycl::event>
    py_repeat_by_scalar(const dpctl::tensor::usm_ndarray &src,
                        const dpctl::tensor::usm_ndarray &dst,
                        const py::ssize_t reps,
                        int axis,
                        sycl::queue &exec_q,
                        const std::vector<sycl::event> &depends)
{
    int src_nd = src.get_ndim();
    if (axis < 0 || (axis + 1 > src_nd && src_nd > 0) ||
        (axis > 0 && src_nd == 0)) {
        throw py::value_error("Specified axis is invalid.");
    }

    int dst_nd = dst.get_ndim();
    if ((src_nd != dst_nd && src_nd > 0) || (src_nd == 0 && dst_nd > 1)) {
        throw py::value_error("Number of dimensions of source and destination "
                              "arrays is not consistent");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool same_orthog_dims(true);
    std::size_t orthog_nelems(1); // number of orthogonal iterations
    for (auto i = 0; i < axis; ++i) {
        auto src_sh_i = src_shape[i];
        orthog_nelems *= src_sh_i;
        same_orthog_dims = same_orthog_dims && (src_sh_i == dst_shape[i]);
    }
    for (auto i = axis + 1; i < src_nd; ++i) {
        auto src_sh_i = src_shape[i];
        orthog_nelems *= src_sh_i;
        same_orthog_dims = same_orthog_dims && (src_sh_i == dst_shape[i]);
    }

    std::size_t src_axis_nelems(1);
    if (src_nd > 0) {
        src_axis_nelems = src_shape[axis];
    }
    std::size_t dst_axis_nelems(dst_shape[axis]);

    // shape at repeated axis must be equal to the shape of src at the axis *
    // reps
    if (!same_orthog_dims || (src_axis_nelems * reps) != dst_axis_nelems) {
        throw py::value_error("Inconsistent array dimensions");
    }

    if (orthog_nelems == 0 || src_axis_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, orthog_nelems * (src_axis_nelems * reps));

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // check that dst does not intersect with src
    if (overlap(dst, src)) {
        throw py::value_error("Destination array overlaps with inputs");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_typeid != dst_typeid) {
        throw py::value_error(
            "Destination array must have the same elemental data type");
    }

    const char *src_data_p = src.get_data();
    char *dst_data_p = dst.get_data();

    auto src_shape_vec = src.get_shape_vector();
    auto src_strides_vec = src.get_strides_vector();

    auto dst_shape_vec = dst.get_shape_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    sycl::event repeat_ev;
    std::vector<sycl::event> host_task_events{};
    if (axis == 0 && src_nd < 2) {
        // empty orthogonal directions

        auto fn = repeat_by_scalar_1d_dispatch_vector[src_typeid];

        assert(dst_shape_vec.size() == 1);
        assert(dst_strides_vec.size() == 1);

        if (src_nd == 0) {
            src_shape_vec = {0};
            src_strides_vec = {0};
        }

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, src_shape_vec, src_strides_vec);
        auto packed_src_shape_strides_owner =
            std::move(std::get<0>(ptr_size_event_tuple1));
        sycl::event copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple1);
        const py::ssize_t *packed_src_shape_strides =
            packed_src_shape_strides_owner.get();

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_shapes_strides_ev);

        assert(all_deps.size() == depends.size() + 1);

        repeat_ev = fn(exec_q, dst_axis_nelems, src_data_p, dst_data_p, reps,
                       src_nd, packed_src_shape_strides, dst_shape_vec[0],
                       dst_strides_vec[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {repeat_ev}, packed_src_shape_strides_owner);

        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }
    else {
        // non-empty orthogonal directions

        auto fn = repeat_by_scalar_dispatch_vector[src_typeid];

        int orthog_nd = src_nd - 1;

        using shT = std::vector<py::ssize_t>;
        shT orthog_src_shape;
        shT orthog_src_strides;
        shT axis_src_shape;
        shT axis_src_stride;
        dpctl::tensor::py_internal::split_iteration_space(
            src_shape_vec, src_strides_vec, axis, axis + 1, orthog_src_shape,
            axis_src_shape, orthog_src_strides, axis_src_stride);

        shT orthog_dst_shape;
        shT orthog_dst_strides;
        shT axis_dst_shape;
        shT axis_dst_stride;
        dpctl::tensor::py_internal::split_iteration_space(
            dst_shape_vec, dst_strides_vec, axis, axis + 1, orthog_dst_shape,
            axis_dst_shape, orthog_dst_strides, axis_dst_stride);

        assert(orthog_src_shape.size() == static_cast<std::size_t>(orthog_nd));
        assert(orthog_dst_shape.size() == static_cast<std::size_t>(orthog_nd));
        assert(std::equal(orthog_src_shape.begin(), orthog_src_shape.end(),
                          orthog_dst_shape.begin()));

        shT simplified_orthog_shape;
        shT simplified_orthog_src_strides;
        shT simplified_orthog_dst_strides;

        const py::ssize_t *_shape = orthog_src_shape.data();

        py::ssize_t orthog_src_offset(0);
        py::ssize_t orthog_dst_offset(0);

        dpctl::tensor::py_internal::simplify_iteration_space(
            orthog_nd, _shape, orthog_src_strides, orthog_dst_strides,
            // output
            simplified_orthog_shape, simplified_orthog_src_strides,
            simplified_orthog_dst_strides, orthog_src_offset,
            orthog_dst_offset);

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, simplified_orthog_shape,
            simplified_orthog_src_strides, simplified_orthog_dst_strides);
        auto packed_shapes_strides_owner =
            std::move(std::get<0>(ptr_size_event_tuple1));
        sycl::event copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple1);
        const py::ssize_t *packed_shapes_strides =
            packed_shapes_strides_owner.get();

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_shapes_strides_ev);

        assert(all_deps.size() == depends.size() + 1);

        repeat_ev = fn(exec_q, orthog_nelems, dst_axis_nelems, src_data_p,
                       dst_data_p, reps,
                       // data to build orthog indexer
                       orthog_nd, packed_shapes_strides, orthog_src_offset,
                       orthog_dst_offset,
                       // data to build indexer along repeated axis in src
                       axis_src_shape[0], axis_src_stride[0],
                       // data to build indexer along repeated axis in dst
                       axis_dst_shape[0], axis_dst_stride[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {repeat_ev}, packed_shapes_strides_owner);
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }

    sycl::event py_obj_management_host_task_ev =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, repeat_ev);
}

std::pair<sycl::event, sycl::event>
    py_repeat_by_scalar(const dpctl::tensor::usm_ndarray &src,
                        const dpctl::tensor::usm_ndarray &dst,
                        const py::ssize_t reps,
                        sycl::queue &exec_q,
                        const std::vector<sycl::event> &depends)
{
    int dst_nd = dst.get_ndim();
    if (dst_nd != 1) {
        throw py::value_error(
            "`dst` array must be 1-dimensional when repeating a full array");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    std::size_t src_sz = src.get_size();
    std::size_t dst_sz = dst.get_size();

    // shape at repeated axis must be equal to the shape of src at the axis *
    // reps
    if ((src_sz * reps) != dst_sz) {
        throw py::value_error("Inconsistent array dimensions");
    }

    if (src_sz == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst,
                                                               src_sz * reps);

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // check that dst does not intersect with src
    if (overlap(dst, src)) {
        throw py::value_error("Destination array overlaps with inputs");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_typeid != dst_typeid) {
        throw py::value_error(
            "Destination array must have the same elemental data type");
    }

    const char *src_data_p = src.get_data();
    char *dst_data_p = dst.get_data();

    int src_nd = src.get_ndim();
    auto src_shape_vec = src.get_shape_vector();
    auto src_strides_vec = src.get_strides_vector();

    if (src_nd == 0) {
        src_shape_vec = {0};
        src_strides_vec = {0};
    }

    auto dst_shape_vec = dst.get_shape_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    std::vector<sycl::event> host_task_events{};

    auto fn = repeat_by_scalar_1d_dispatch_vector[src_typeid];

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, src_shape_vec, src_strides_vec);
    auto packed_src_shape_strides_owner =
        std::move(std::get<0>(ptr_size_event_tuple1));
    sycl::event copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple1);
    const py::ssize_t *packed_src_shape_strides =
        packed_src_shape_strides_owner.get();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_shapes_strides_ev);

    assert(all_deps.size() == depends.size() + 1);

    sycl::event repeat_ev = fn(exec_q, dst_sz, src_data_p, dst_data_p, reps,
                               src_nd, packed_src_shape_strides,
                               dst_shape_vec[0], dst_strides_vec[0], all_deps);

    sycl::event cleanup_tmp_allocations_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {repeat_ev}, packed_src_shape_strides_owner);
    host_task_events.push_back(cleanup_tmp_allocations_ev);

    sycl::event py_obj_management_host_task_ev =
        dpctl::utils::keep_args_alive(exec_q, {src, dst}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, repeat_ev);
}

} // namespace dpctl::tensor::py_internal
