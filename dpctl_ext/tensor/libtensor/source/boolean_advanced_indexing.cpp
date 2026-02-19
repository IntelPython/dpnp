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
/// This file defines implementation functions of dpctl.tensor.place and
/// dpctl.tensor.extract, dpctl.tensor.nonzero
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simplify_iteration_space.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

#include "boolean_advanced_indexing.hpp"
#include "kernels/boolean_advanced_indexing.hpp"

namespace dpctl::tensor::py_internal
{

// Masked extraction

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::indexing::
    masked_extract_all_slices_strided_impl_fn_ptr_t;

static masked_extract_all_slices_strided_impl_fn_ptr_t
    masked_extract_all_slices_strided_i32_impl_dispatch_vector
        [td_ns::num_types];
static masked_extract_all_slices_strided_impl_fn_ptr_t
    masked_extract_all_slices_strided_i64_impl_dispatch_vector
        [td_ns::num_types];

using dpctl::tensor::kernels::indexing::
    masked_extract_all_slices_contig_impl_fn_ptr_t;

static masked_extract_all_slices_contig_impl_fn_ptr_t
    masked_extract_all_slices_contig_i32_impl_dispatch_vector[td_ns::num_types];
static masked_extract_all_slices_contig_impl_fn_ptr_t
    masked_extract_all_slices_contig_i64_impl_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::indexing::
    masked_extract_some_slices_strided_impl_fn_ptr_t;

static masked_extract_some_slices_strided_impl_fn_ptr_t
    masked_extract_some_slices_strided_i32_impl_dispatch_vector
        [td_ns::num_types];
static masked_extract_some_slices_strided_impl_fn_ptr_t
    masked_extract_some_slices_strided_i64_impl_dispatch_vector
        [td_ns::num_types];

void populate_masked_extract_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::indexing::
        MaskExtractAllSlicesStridedFactoryForInt32;
    td_ns::DispatchVectorBuilder<
        masked_extract_all_slices_strided_impl_fn_ptr_t,
        MaskExtractAllSlicesStridedFactoryForInt32, td_ns::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(
        masked_extract_all_slices_strided_i32_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::
        MaskExtractAllSlicesStridedFactoryForInt64;
    td_ns::DispatchVectorBuilder<
        masked_extract_all_slices_strided_impl_fn_ptr_t,
        MaskExtractAllSlicesStridedFactoryForInt64, td_ns::num_types>
        dvb2;
    dvb2.populate_dispatch_vector(
        masked_extract_all_slices_strided_i64_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::
        MaskExtractSomeSlicesStridedFactoryForInt32;
    td_ns::DispatchVectorBuilder<
        masked_extract_some_slices_strided_impl_fn_ptr_t,
        MaskExtractSomeSlicesStridedFactoryForInt32, td_ns::num_types>
        dvb3;
    dvb3.populate_dispatch_vector(
        masked_extract_some_slices_strided_i32_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::
        MaskExtractSomeSlicesStridedFactoryForInt64;
    td_ns::DispatchVectorBuilder<
        masked_extract_some_slices_strided_impl_fn_ptr_t,
        MaskExtractSomeSlicesStridedFactoryForInt64, td_ns::num_types>
        dvb4;
    dvb4.populate_dispatch_vector(
        masked_extract_some_slices_strided_i64_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::
        MaskExtractAllSlicesContigFactoryForInt32;
    td_ns::DispatchVectorBuilder<masked_extract_all_slices_contig_impl_fn_ptr_t,
                                 MaskExtractAllSlicesContigFactoryForInt32,
                                 td_ns::num_types>
        dvb5;
    dvb5.populate_dispatch_vector(
        masked_extract_all_slices_contig_i32_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::
        MaskExtractAllSlicesContigFactoryForInt64;
    td_ns::DispatchVectorBuilder<masked_extract_all_slices_contig_impl_fn_ptr_t,
                                 MaskExtractAllSlicesContigFactoryForInt64,
                                 td_ns::num_types>
        dvb6;
    dvb6.populate_dispatch_vector(
        masked_extract_all_slices_contig_i64_impl_dispatch_vector);
}

std::pair<sycl::event, sycl::event>
    py_extract(const dpctl::tensor::usm_ndarray &src,
               const dpctl::tensor::usm_ndarray &cumsum,
               int axis_start, // axis_start <= mask_i < axis_end
               int axis_end,
               const dpctl::tensor::usm_ndarray &dst,
               sycl::queue &exec_q,
               const std::vector<sycl::event> &depends)
{
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    int src_nd = src.get_ndim();
    if ((axis_start < 0 || axis_end > src_nd || axis_start >= axis_end)) {
        throw py::value_error("Specified axes_start and axes_end are invalid.");
    }
    int mask_span_sz = axis_end - axis_start;

    int dst_nd = dst.get_ndim();
    if (src_nd != dst_nd + (mask_span_sz - 1)) {
        throw py::value_error("Number of dimensions of source and destination "
                              "arrays is not consistent");
    }

    if (!cumsum.is_c_contiguous() || cumsum.get_ndim() != 1) {
        throw py::value_error("cumsum array must be a C-contiguous vector");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, cumsum, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    py::ssize_t cumsum_sz = cumsum.get_size();

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();
    bool same_ortho_dims(true);
    std::size_t ortho_nelems(1); // number of orthogonal iterations

    for (auto i = 0; i < axis_start; ++i) {
        auto src_sh_i = src_shape[i];
        ortho_nelems *= src_sh_i;
        same_ortho_dims = same_ortho_dims && (src_sh_i == dst_shape[i]);
    }
    for (auto i = axis_end; i < src_nd; ++i) {
        auto src_sh_i = src_shape[i];
        ortho_nelems *= src_sh_i;
        same_ortho_dims =
            same_ortho_dims && (src_sh_i == dst_shape[i - (mask_span_sz - 1)]);
    }

    std::size_t masked_src_nelems(1);
    std::size_t masked_dst_nelems(dst_shape[axis_start]);
    for (auto i = axis_start; i < axis_end; ++i) {
        masked_src_nelems *= src_shape[i];
    }

    // masked_dst_nelems is number of set elements in the mask, or last element
    // in cumsum
    if (!same_ortho_dims ||
        (masked_src_nelems != static_cast<std::size_t>(cumsum_sz)))
    {
        throw py::value_error("Inconsistent array dimensions");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, ortho_nelems * masked_dst_nelems);

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // check that dst does not intersect with src, not with cumsum.
    if (overlap(dst, cumsum) || overlap(dst, src)) {
        throw py::value_error("Destination array overlaps with inputs");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();
    int cumsum_typenum = cumsum.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);
    int cumsum_typeid = array_types.typenum_to_lookup_id(cumsum_typenum);

    static constexpr int int32_typeid =
        static_cast<int>(td_ns::typenum_t::INT32);
    static constexpr int int64_typeid =
        static_cast<int>(td_ns::typenum_t::INT64);
    if (cumsum_typeid != int32_typeid && cumsum_typeid != int64_typeid) {
        throw py::value_error("Unexpected data type of cumsum array, expecting "
                              "'int32' or 'int64'");
    }

    const bool use_i32 = (cumsum_typeid == int32_typeid);

    if (src_typeid != dst_typeid) {
        throw py::value_error(
            "Destination array must have the same elemental data types");
    }

    char *src_data_p = src.get_data();
    char *dst_data_p = dst.get_data();
    char *cumsum_data_p = cumsum.get_data();

    auto src_shape_vec = src.get_shape_vector();
    auto src_strides_vec = src.get_strides_vector();

    auto dst_shape_vec = dst.get_shape_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    sycl::event extract_ev;
    std::vector<sycl::event> host_task_events{};
    if (axis_start == 0 && axis_end == src_nd) {
        assert(dst_shape_vec.size() == 1);
        assert(dst_strides_vec.size() == 1);

        if (src.is_c_contiguous()) {
            auto fn =
                (use_i32)
                    ? masked_extract_all_slices_contig_i32_impl_dispatch_vector
                          [src_typeid]
                    : masked_extract_all_slices_contig_i64_impl_dispatch_vector
                          [src_typeid];

            extract_ev =
                fn(exec_q, cumsum_sz, src_data_p, cumsum_data_p, dst_data_p,
                   dst_shape_vec[0], dst_strides_vec[0], depends);

            //
            host_task_events.push_back(extract_ev);
        }
        else {
            // empty orthogonal directions
            auto fn =
                (use_i32)
                    ? masked_extract_all_slices_strided_i32_impl_dispatch_vector
                          [src_typeid]
                    : masked_extract_all_slices_strided_i64_impl_dispatch_vector
                          [src_typeid];

            using dpctl::tensor::offset_utils::device_allocate_and_pack;
            auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
                exec_q, host_task_events, src_shape_vec, src_strides_vec);
            auto packed_src_shape_strides_owner =
                std::move(std::get<0>(ptr_size_event_tuple1));
            sycl::event copy_src_shape_strides_ev =
                std::get<2>(ptr_size_event_tuple1);
            const py::ssize_t *packed_src_shape_strides =
                packed_src_shape_strides_owner.get();

            std::vector<sycl::event> all_deps;
            all_deps.reserve(depends.size() + 1);
            all_deps.insert(all_deps.end(), depends.begin(), depends.end());
            all_deps.push_back(copy_src_shape_strides_ev);

            assert(all_deps.size() == depends.size() + 1);

            extract_ev = fn(exec_q, cumsum_sz, src_data_p, cumsum_data_p,
                            dst_data_p, src_nd, packed_src_shape_strides,
                            dst_shape_vec[0], dst_strides_vec[0], all_deps);

            sycl::event cleanup_tmp_allocations_ev =
                dpctl::tensor::alloc_utils::async_smart_free(
                    exec_q, {extract_ev}, packed_src_shape_strides_owner);
            host_task_events.push_back(cleanup_tmp_allocations_ev);
        }
    }
    else {
        // non-empty orthogonal directions
        auto fn =
            (use_i32)
                ? masked_extract_some_slices_strided_i32_impl_dispatch_vector
                      [src_typeid]
                : masked_extract_some_slices_strided_i64_impl_dispatch_vector
                      [src_typeid];

        int masked_src_nd = mask_span_sz;
        int ortho_nd = src_nd - masked_src_nd;

        using shT = std::vector<py::ssize_t>;

        shT ortho_src_shape;
        shT masked_src_shape;
        shT ortho_src_strides;
        shT masked_src_strides;
        dpctl::tensor::py_internal::split_iteration_space(
            src_shape_vec, src_strides_vec, axis_start, axis_end,
            ortho_src_shape,
            masked_src_shape, // 4 vectors modified
            ortho_src_strides, masked_src_strides);

        shT ortho_dst_shape;
        shT masked_dst_shape;
        shT ortho_dst_strides;
        shT masked_dst_strides;
        dpctl::tensor::py_internal::split_iteration_space(
            dst_shape_vec, dst_strides_vec, axis_start, axis_start + 1,
            ortho_dst_shape,
            masked_dst_shape, // 4 vectors modified
            ortho_dst_strides, masked_dst_strides);

        assert(ortho_src_shape.size() == static_cast<std::size_t>(ortho_nd));
        assert(ortho_dst_shape.size() == static_cast<std::size_t>(ortho_nd));
        assert(std::equal(ortho_src_shape.begin(), ortho_src_shape.end(),
                          ortho_dst_shape.begin()));

        std::vector<py::ssize_t> simplified_ortho_shape;
        std::vector<py::ssize_t> simplified_ortho_src_strides;
        std::vector<py::ssize_t> simplified_ortho_dst_strides;

        const py::ssize_t *_shape = ortho_src_shape.data();

        py::ssize_t ortho_src_offset(0);
        py::ssize_t ortho_dst_offset(0);

        dpctl::tensor::py_internal::simplify_iteration_space(
            ortho_nd, _shape, ortho_src_strides, ortho_dst_strides,
            // output
            simplified_ortho_shape, simplified_ortho_src_strides,
            simplified_ortho_dst_strides, ortho_src_offset, ortho_dst_offset);

        assert(masked_dst_shape.size() == 1);
        assert(masked_dst_strides.size() == 1);

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, simplified_ortho_shape,
            simplified_ortho_src_strides, simplified_ortho_dst_strides,
            masked_src_shape, masked_src_strides);
        auto packed_shapes_strides_owner =
            std::move(std::get<0>(ptr_size_event_tuple1));
        sycl::event copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple1);
        const py::ssize_t *packed_shapes_strides =
            packed_shapes_strides_owner.get();

        const py::ssize_t *packed_ortho_src_dst_shape_strides =
            packed_shapes_strides;
        const py::ssize_t *packed_masked_src_shape_strides =
            packed_shapes_strides + (3 * ortho_nd);

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_shapes_strides_ev);

        assert(all_deps.size() == depends.size() + 1);

        // OrthogIndexerT orthog_src_dst_indexer_, MaskedIndexerT
        // masked_src_indexer_, MaskedIndexerT masked_dst_indexer_
        extract_ev = fn(exec_q, ortho_nelems, masked_src_nelems, src_data_p,
                        cumsum_data_p, dst_data_p,
                        // data to build orthog_src_dst_indexer
                        ortho_nd, packed_ortho_src_dst_shape_strides,
                        ortho_src_offset, ortho_dst_offset,
                        // data to build masked_src_indexer
                        masked_src_nd, packed_masked_src_shape_strides,
                        // data to build masked_dst_indexer,
                        masked_dst_shape[0], masked_dst_strides[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {extract_ev}, packed_shapes_strides_owner);
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }

    sycl::event py_obj_management_host_task_ev = dpctl::utils::keep_args_alive(
        exec_q, {src, cumsum, dst}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, extract_ev);
}

// Masked placement

using dpctl::tensor::kernels::indexing::
    masked_place_all_slices_strided_impl_fn_ptr_t;

static masked_place_all_slices_strided_impl_fn_ptr_t
    masked_place_all_slices_strided_i32_impl_dispatch_vector[td_ns::num_types];
static masked_place_all_slices_strided_impl_fn_ptr_t
    masked_place_all_slices_strided_i64_impl_dispatch_vector[td_ns::num_types];

using dpctl::tensor::kernels::indexing::
    masked_place_some_slices_strided_impl_fn_ptr_t;

static masked_place_some_slices_strided_impl_fn_ptr_t
    masked_place_some_slices_strided_i32_impl_dispatch_vector[td_ns::num_types];
static masked_place_some_slices_strided_impl_fn_ptr_t
    masked_place_some_slices_strided_i64_impl_dispatch_vector[td_ns::num_types];

void populate_masked_place_dispatch_vectors(void)
{
    using dpctl::tensor::kernels::indexing::
        MaskPlaceAllSlicesStridedFactoryForInt32;
    td_ns::DispatchVectorBuilder<masked_place_all_slices_strided_impl_fn_ptr_t,
                                 MaskPlaceAllSlicesStridedFactoryForInt32,
                                 td_ns::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(
        masked_place_all_slices_strided_i32_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::
        MaskPlaceAllSlicesStridedFactoryForInt64;
    td_ns::DispatchVectorBuilder<masked_place_all_slices_strided_impl_fn_ptr_t,
                                 MaskPlaceAllSlicesStridedFactoryForInt64,
                                 td_ns::num_types>
        dvb2;
    dvb2.populate_dispatch_vector(
        masked_place_all_slices_strided_i64_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::
        MaskPlaceSomeSlicesStridedFactoryForInt32;
    td_ns::DispatchVectorBuilder<masked_place_some_slices_strided_impl_fn_ptr_t,
                                 MaskPlaceSomeSlicesStridedFactoryForInt32,
                                 td_ns::num_types>
        dvb3;
    dvb3.populate_dispatch_vector(
        masked_place_some_slices_strided_i32_impl_dispatch_vector);

    using dpctl::tensor::kernels::indexing::
        MaskPlaceSomeSlicesStridedFactoryForInt64;
    td_ns::DispatchVectorBuilder<masked_place_some_slices_strided_impl_fn_ptr_t,
                                 MaskPlaceSomeSlicesStridedFactoryForInt64,
                                 td_ns::num_types>
        dvb4;
    dvb4.populate_dispatch_vector(
        masked_place_some_slices_strided_i64_impl_dispatch_vector);
}

/*
 * @brief Copy dst[i, ortho_id] = rhs[cumsum[i] - 1, ortho_id]  if cumsum[i] ==
 * ((i > 0) ? cumsum[i-1] + 1 : 1)
 */
std::pair<sycl::event, sycl::event>
    py_place(const dpctl::tensor::usm_ndarray &dst,
             const dpctl::tensor::usm_ndarray &cumsum,
             int axis_start, // axis_start <= mask_i < axis_end
             int axis_end,
             const dpctl::tensor::usm_ndarray &rhs,
             sycl::queue &exec_q,
             const std::vector<sycl::event> &depends)
{
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    int dst_nd = dst.get_ndim();
    if ((axis_start < 0 || axis_end > dst_nd || axis_start >= axis_end)) {
        throw py::value_error("Specified axes_start and axes_end are invalid.");
    }
    int mask_span_sz = axis_end - axis_start;

    int rhs_nd = rhs.get_ndim();
    if (dst_nd != rhs_nd + (mask_span_sz - 1)) {
        throw py::value_error("Number of dimensions of source and destination "
                              "arrays is not consistent");
    }

    if (!cumsum.is_c_contiguous() || cumsum.get_ndim() != 1) {
        throw py::value_error("cumsum array must be a C-contiguous vector");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst, cumsum, rhs})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    py::ssize_t cumsum_sz = cumsum.get_size();

    const py::ssize_t *dst_shape = dst.get_shape_raw();
    const py::ssize_t *rhs_shape = rhs.get_shape_raw();
    bool same_ortho_dims(true);
    std::size_t ortho_nelems(1); // number of orthogonal iterations

    for (auto i = 0; i < axis_start; ++i) {
        auto dst_sh_i = dst_shape[i];
        ortho_nelems *= dst_sh_i;
        same_ortho_dims = same_ortho_dims && (dst_sh_i == rhs_shape[i]);
    }
    for (auto i = axis_end; i < dst_nd; ++i) {
        auto dst_sh_i = dst_shape[i];
        ortho_nelems *= dst_sh_i;
        same_ortho_dims =
            same_ortho_dims && (dst_sh_i == rhs_shape[i - (mask_span_sz - 1)]);
    }

    std::size_t masked_dst_nelems(1);
    for (auto i = axis_start; i < axis_end; ++i) {
        masked_dst_nelems *= dst_shape[i];
    }

    if (!same_ortho_dims ||
        (masked_dst_nelems != static_cast<std::size_t>(cumsum_sz)))
    {
        throw py::value_error("Inconsistent array dimensions");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, ortho_nelems * masked_dst_nelems);

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // check that dst does not intersect with src, not with cumsum.
    if (overlap(dst, rhs) || overlap(dst, cumsum)) {
        throw py::value_error("Destination array overlaps with inputs");
    }

    int dst_typenum = dst.get_typenum();
    int rhs_typenum = rhs.get_typenum();
    int cumsum_typenum = cumsum.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);
    int rhs_typeid = array_types.typenum_to_lookup_id(rhs_typenum);
    int cumsum_typeid = array_types.typenum_to_lookup_id(cumsum_typenum);

    static constexpr int int32_typeid =
        static_cast<int>(td_ns::typenum_t::INT32);
    static constexpr int int64_typeid =
        static_cast<int>(td_ns::typenum_t::INT64);
    if (cumsum_typeid != int32_typeid && cumsum_typeid != int64_typeid) {
        throw py::value_error("Unexpected data type of cumsum array, expecting "
                              "'int32' or 'int64'");
    }

    const bool use_i32 = (cumsum_typeid == int32_typeid);

    if (dst_typeid != rhs_typeid) {
        throw py::value_error(
            "Destination array must have the same elemental data types");
    }

    char *dst_data_p = dst.get_data();
    char *rhs_data_p = rhs.get_data();
    char *cumsum_data_p = cumsum.get_data();

    auto dst_shape_vec = dst.get_shape_vector();
    auto dst_strides_vec = dst.get_strides_vector();

    auto rhs_shape_vec = rhs.get_shape_vector();
    auto rhs_strides_vec = rhs.get_strides_vector();

    sycl::event place_ev;
    std::vector<sycl::event> host_task_events{};
    if (axis_start == 0 && axis_end == dst_nd) {
        // empty orthogonal directions
        auto fn = (use_i32)
                      ? masked_place_all_slices_strided_i32_impl_dispatch_vector
                            [dst_typeid]
                      : masked_place_all_slices_strided_i64_impl_dispatch_vector
                            [dst_typeid];

        assert(rhs_shape_vec.size() == 1);
        assert(rhs_strides_vec.size() == 1);

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, dst_shape_vec, dst_strides_vec);
        auto packed_dst_shape_strides_owner =
            std::move(std::get<0>(ptr_size_event_tuple1));
        sycl::event copy_dst_shape_strides_ev =
            std::get<2>(ptr_size_event_tuple1);
        const py::ssize_t *packed_dst_shape_strides =
            packed_dst_shape_strides_owner.get();

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_dst_shape_strides_ev);

        assert(all_deps.size() == depends.size() + 1);

        place_ev = fn(exec_q, cumsum_sz, dst_data_p, cumsum_data_p, rhs_data_p,
                      dst_nd, packed_dst_shape_strides, rhs_shape_vec[0],
                      rhs_strides_vec[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {place_ev}, packed_dst_shape_strides_owner);
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }
    else {
        // non-empty orthogonal directions
        auto fn =
            (use_i32)
                ? masked_place_some_slices_strided_i32_impl_dispatch_vector
                      [dst_typeid]
                : masked_place_some_slices_strided_i64_impl_dispatch_vector
                      [dst_typeid];

        int masked_dst_nd = mask_span_sz;
        int ortho_nd = dst_nd - masked_dst_nd;

        using shT = std::vector<py::ssize_t>;

        shT ortho_dst_shape;
        shT masked_dst_shape;
        shT ortho_dst_strides;
        shT masked_dst_strides;
        dpctl::tensor::py_internal::split_iteration_space(
            dst_shape_vec, dst_strides_vec, axis_start, axis_end,
            ortho_dst_shape,
            masked_dst_shape, // 4 vectors modified
            ortho_dst_strides, masked_dst_strides);

        shT ortho_rhs_shape;
        shT masked_rhs_shape;
        shT ortho_rhs_strides;
        shT masked_rhs_strides;
        dpctl::tensor::py_internal::split_iteration_space(
            rhs_shape_vec, rhs_strides_vec, axis_start, axis_start + 1,
            ortho_rhs_shape,
            masked_rhs_shape, // 4 vectors modified
            ortho_rhs_strides, masked_rhs_strides);

        assert(ortho_dst_shape.size() == static_cast<std::size_t>(ortho_nd));
        assert(ortho_rhs_shape.size() == static_cast<std::size_t>(ortho_nd));
        assert(std::equal(ortho_dst_shape.begin(), ortho_dst_shape.end(),
                          ortho_rhs_shape.begin()));

        std::vector<py::ssize_t> simplified_ortho_shape;
        std::vector<py::ssize_t> simplified_ortho_dst_strides;
        std::vector<py::ssize_t> simplified_ortho_rhs_strides;

        const py::ssize_t *_shape = ortho_dst_shape.data();

        py::ssize_t ortho_dst_offset(0);
        py::ssize_t ortho_rhs_offset(0);

        dpctl::tensor::py_internal::simplify_iteration_space(
            ortho_nd, _shape, ortho_dst_strides, ortho_rhs_strides,
            simplified_ortho_shape, simplified_ortho_dst_strides,
            simplified_ortho_rhs_strides, ortho_dst_offset, ortho_rhs_offset);

        assert(masked_rhs_shape.size() == 1);
        assert(masked_rhs_strides.size() == 1);

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, simplified_ortho_shape,
            simplified_ortho_dst_strides, simplified_ortho_rhs_strides,
            masked_dst_shape, masked_dst_strides);
        auto packed_shapes_strides_owner =
            std::move(std::get<0>(ptr_size_event_tuple1));
        sycl::event copy_shapes_strides_ev = std::get<2>(ptr_size_event_tuple1);
        const py::ssize_t *packed_shapes_strides =
            packed_shapes_strides_owner.get();

        const py::ssize_t *packed_ortho_dst_rhs_shape_strides =
            packed_shapes_strides;
        const py::ssize_t *packed_masked_dst_shape_strides =
            packed_shapes_strides + (3 * ortho_nd);

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.insert(all_deps.end(), depends.begin(), depends.end());
        all_deps.push_back(copy_shapes_strides_ev);

        assert(all_deps.size() == depends.size() + 1);

        place_ev = fn(exec_q, ortho_nelems, masked_dst_nelems, dst_data_p,
                      cumsum_data_p, rhs_data_p,
                      // data to build orthog_dst_rhs_indexer
                      ortho_nd, packed_ortho_dst_rhs_shape_strides,
                      ortho_dst_offset, ortho_rhs_offset,
                      // data to build masked_dst_indexer
                      masked_dst_nd, packed_masked_dst_shape_strides,
                      // data to build masked_dst_indexer,
                      masked_rhs_shape[0], masked_rhs_strides[0], all_deps);

        sycl::event cleanup_tmp_allocations_ev =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {place_ev}, packed_shapes_strides_owner);
        host_task_events.push_back(cleanup_tmp_allocations_ev);
    }

    sycl::event py_obj_management_host_task_ev = dpctl::utils::keep_args_alive(
        exec_q, {dst, cumsum, rhs}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, place_ev);
}

// Non-zero

std::pair<sycl::event, sycl::event>
    py_nonzero(const dpctl::tensor::usm_ndarray
                   &cumsum, // int32/int64 input array, 1D, C-contiguous
               const dpctl::tensor::usm_ndarray
                   &indexes, // int32/int64 2D output array, C-contiguous
               const std::vector<py::ssize_t>
                   &mask_shape, // shape of array from which cumsum was computed
               sycl::queue &exec_q,
               const std::vector<sycl::event> &depends)
{
    if (!dpctl::utils::queues_are_compatible(exec_q, {cumsum, indexes})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(indexes);

    int cumsum_nd = cumsum.get_ndim();
    if (cumsum_nd != 1 || !cumsum.is_c_contiguous()) {
        throw py::value_error("Cumsum array must be a C-contiguous vector");
    }

    int indexes_nd = indexes.get_ndim();
    if (indexes_nd != 2 || !indexes.is_c_contiguous()) {
        throw py::value_error("Index array must be a C-contiguous matrix");
    }

    std::size_t _ndim = mask_shape.size();
    if (_ndim > std::numeric_limits<int>::max()) {
        throw py::value_error("Shape is too large");
    }
    int ndim = static_cast<int>(_ndim);

    const py::ssize_t *indexes_shape = indexes.get_shape_raw();

    if (ndim != indexes_shape[0]) {
        throw py::value_error(
            "Length of shape must equal width of index matrix");
    }

    auto cumsum_sz = cumsum.get_size();
    py::ssize_t shape_nelems =
        std::accumulate(mask_shape.begin(), mask_shape.end(), py::ssize_t(1),
                        std::multiplies<py::ssize_t>());

    if (cumsum_sz != shape_nelems) {
        throw py::value_error("Shape and cumsum size are not consistent");
    }

    py::ssize_t nz_elems = indexes_shape[1];

    int indexes_typenum = indexes.get_typenum();
    auto const &array_types = td_ns::usm_ndarray_types();
    int indexes_typeid = array_types.typenum_to_lookup_id(indexes_typenum);

    int cumsum_typenum = cumsum.get_typenum();
    int cumsum_typeid = array_types.typenum_to_lookup_id(cumsum_typenum);

    constexpr int int32_typeid = static_cast<int>(td_ns::typenum_t::INT32);
    constexpr int int64_typeid = static_cast<int>(td_ns::typenum_t::INT64);

    // cumsum must be int32_t or int64_t only
    if ((cumsum_typeid != int32_typeid && cumsum_typeid != int64_typeid) ||
        (indexes_typeid != int32_typeid && indexes_typeid != int64_typeid))
    {
        throw py::value_error("Cumulative sum array and index array must have "
                              "int32 or int64 data-type");
    }

    if (cumsum_sz == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(cumsum, indexes)) {
        throw py::value_error("Arrays are expected to ave no memory overlap");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        indexes, nz_elems * _ndim);

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto mask_shape_copying_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, mask_shape);
    auto src_shape_device_owner =
        std::move(std::get<0>(mask_shape_copying_tuple));
    sycl::event copy_ev = std::get<2>(mask_shape_copying_tuple);
    const py::ssize_t *src_shape_device_ptr = src_shape_device_owner.get();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);

    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_ev);

    using dpctl::tensor::kernels::indexing::non_zero_indexes_fn_ptr_t;
    using dpctl::tensor::kernels::indexing::non_zero_indexes_impl;

    int fn_index = ((cumsum_typeid == int64_typeid) ? 1 : 0) +
                   ((indexes_typeid == int64_typeid) ? 2 : 0);
    std::array<non_zero_indexes_fn_ptr_t, 4> fn_impls = {
        non_zero_indexes_impl<std::int32_t, std::int32_t>,
        non_zero_indexes_impl<std::int64_t, std::int32_t>,
        non_zero_indexes_impl<std::int32_t, std::int64_t>,
        non_zero_indexes_impl<std::int64_t, std::int64_t>};
    auto fn = fn_impls[fn_index];

    sycl::event non_zero_indexes_ev =
        fn(exec_q, cumsum_sz, nz_elems, ndim, cumsum.get_data(),
           indexes.get_data(), src_shape_device_ptr, all_deps);

    sycl::event temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {non_zero_indexes_ev}, src_shape_device_owner);
    host_task_events.push_back(temporaries_cleanup_ev);

    sycl::event py_obj_management_host_task_ev = dpctl::utils::keep_args_alive(
        exec_q, {cumsum, indexes}, host_task_events);

    return std::make_pair(py_obj_management_host_task_ev, non_zero_indexes_ev);
}

} // namespace dpctl::tensor::py_internal
