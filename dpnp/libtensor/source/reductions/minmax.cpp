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
/// This file defines functions of dpnp.tensor._tensor_reductions_impl
/// extension.
//===---------------------------------------------------------------------===//

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels/reductions.hpp"
#include "simplify_iteration_space.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/sycl_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_dispatch_building.hpp"

#include "minmax.hpp"

namespace dpnp::tensor::py_internal
{

namespace py = pybind11;
namespace td_ns = dpnp::tensor::type_dispatch;
namespace su_ns = dpnp::tensor::sycl_utils;

namespace impl
{

using dpnp::tensor::kernels::minmax_strided_impl_fn_ptr;
static minmax_strided_impl_fn_ptr
    minmax_over_axis_strided_temps_dispatch_vector[td_ns::num_types];

using dpnp::tensor::kernels::minmax_contig_impl_fn_ptr;
static minmax_contig_impl_fn_ptr
    minmax_over_axis1_contig_temps_dispatch_vector[td_ns::num_types];
static minmax_contig_impl_fn_ptr
    minmax_over_axis0_contig_temps_dispatch_vector[td_ns::num_types];

/* @brief Operators used by min and max reductions with temporaries */
template <typename T>
using MinOpT =
    std::conditional_t<std::is_integral_v<T> && !std::is_same_v<T, bool>,
                       sycl::minimum<T>,
                       su_ns::Minimum<T>>;

template <typename T>
using MaxOpT =
    std::conditional_t<std::is_integral_v<T> && !std::is_same_v<T, bool>,
                       sycl::maximum<T>,
                       su_ns::Maximum<T>>;

template <typename fnT, typename srcTy>
struct MinMaxOverAxisTempsStridedFactory
{
    fnT get() const
    {
        return dpnp::tensor::kernels::minmax_over_group_temps_strided_impl<
            srcTy, MinOpT<srcTy>, MaxOpT<srcTy>>;
    }
};

template <typename fnT, typename srcTy>
struct MinMaxOverAxis1TempsContigFactory
{
    fnT get() const
    {
        return dpnp::tensor::kernels::minmax_axis1_over_group_temps_contig_impl<
            srcTy, MinOpT<srcTy>, MaxOpT<srcTy>>;
    }
};

template <typename fnT, typename srcTy>
struct MinMaxOverAxis0TempsContigFactory
{
    fnT get() const
    {
        return dpnp::tensor::kernels::minmax_axis0_over_group_temps_contig_impl<
            srcTy, MinOpT<srcTy>, MaxOpT<srcTy>>;
    }
};

void populate_minmax_over_axis_dispatch_vectors(void)
{
    using td_ns::DispatchVectorBuilder;

    DispatchVectorBuilder<minmax_strided_impl_fn_ptr,
                          MinMaxOverAxisTempsStridedFactory, td_ns::num_types>
        dvb1;
    dvb1.populate_dispatch_vector(
        minmax_over_axis_strided_temps_dispatch_vector);

    DispatchVectorBuilder<minmax_contig_impl_fn_ptr,
                          MinMaxOverAxis1TempsContigFactory, td_ns::num_types>
        dvb2;
    dvb2.populate_dispatch_vector(
        minmax_over_axis1_contig_temps_dispatch_vector);

    DispatchVectorBuilder<minmax_contig_impl_fn_ptr,
                          MinMaxOverAxis0TempsContigFactory, td_ns::num_types>
        dvb3;
    dvb3.populate_dispatch_vector(
        minmax_over_axis0_contig_temps_dispatch_vector);
}

} // namespace impl

/*! @brief Computes the minimum and the maximum over the trailing axes in a
 * single pass */
std::pair<sycl::event, sycl::event>
    py_minmax_over_axis(const dpnp::tensor::usm_ndarray &src,
                        int trailing_dims_to_reduce, // comp over this many
                                                     // trailing indexes
                        const dpnp::tensor::usm_ndarray &dst_min,
                        const dpnp::tensor::usm_ndarray &dst_max,
                        sycl::queue &exec_q,
                        const std::vector<sycl::event> &depends)
{
    int src_nd = src.get_ndim();
    int iteration_nd = src_nd - trailing_dims_to_reduce;
    if (trailing_dims_to_reduce <= 0 || iteration_nd < 0) {
        throw py::value_error("Trailing_dim_to_reduce must be positive, but no "
                              "greater than rank of the array being reduced");
    }

    int dst_nd = dst_min.get_ndim();
    if (dst_nd != iteration_nd || dst_max.get_ndim() != iteration_nd) {
        throw py::value_error("Destination array rank does not match input "
                              "array rank and number of reduced dimensions");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_min_shape_ptr = dst_min.get_shape_raw();
    const py::ssize_t *dst_max_shape_ptr = dst_max.get_shape_raw();

    bool same_shapes = true;
    for (int i = 0; same_shapes && (i < dst_nd); ++i) {
        same_shapes = same_shapes &&
                      (src_shape_ptr[i] == dst_min_shape_ptr[i]) &&
                      (src_shape_ptr[i] == dst_max_shape_ptr[i]);
    }

    if (!same_shapes) {
        throw py::value_error("Destination shape does not match unreduced "
                              "dimensions of the input shape");
    }

    if (!dpnp::utils::queues_are_compatible(exec_q, {src, dst_min, dst_max})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpnp::tensor::validation::CheckWritable::throw_if_not_writable(dst_min);
    dpnp::tensor::validation::CheckWritable::throw_if_not_writable(dst_max);

    std::size_t dst_nelems = dst_min.get_size();

    if (dst_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    std::size_t reduction_nelems(1);
    for (int i = dst_nd; i < src_nd; ++i) {
        reduction_nelems *= static_cast<std::size_t>(src_shape_ptr[i]);
    }

    if (reduction_nelems == 0) {
        throw py::value_error(
            "reduction cannot be performed over zero-size axes");
    }

    // check that the arrays do not overlap
    auto const &overlap = dpnp::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst_min) || overlap(src, dst_max) ||
        overlap(dst_min, dst_max)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    dpnp::tensor::validation::AmpleMemory::throw_if_not_ample(dst_min,
                                                              dst_nelems);
    dpnp::tensor::validation::AmpleMemory::throw_if_not_ample(dst_max,
                                                              dst_nelems);

    int src_typenum = src.get_typenum();
    if (src_typenum != dst_min.get_typenum() ||
        src_typenum != dst_max.get_typenum()) {
        throw py::value_error("Destination arrays must have the same data "
                              "type as the input array");
    }

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);

    using impl::minmax_over_axis0_contig_temps_dispatch_vector;
    using impl::minmax_over_axis1_contig_temps_dispatch_vector;
    using impl::minmax_over_axis_strided_temps_dispatch_vector;

    // handle special case when both reduction and iteration are 1D contiguous
    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();
    bool is_dst_c_contig =
        dst_min.is_c_contiguous() && dst_max.is_c_contiguous();
    bool is_dst_f_contig =
        dst_min.is_f_contiguous() && dst_max.is_f_contiguous();

    static constexpr py::ssize_t zero_offset = 0;

    if ((is_src_c_contig && is_dst_c_contig) ||
        (is_src_f_contig && dst_nelems == 1)) {
        auto fn = minmax_over_axis1_contig_temps_dispatch_vector[src_typeid];
        if (fn != nullptr) {
            std::size_t iter_nelems = dst_nelems;

            sycl::event minmax_over_axis_contig_ev =
                fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                   dst_min.get_data(), dst_max.get_data(),
                   zero_offset, // iteration_src_offset
                   zero_offset, // iteration_dst_min_offset
                   zero_offset, // iteration_dst_max_offset
                   zero_offset, // reduction_src_offset
                   depends);

            sycl::event keep_args_event = dpnp::utils::keep_args_alive(
                exec_q, {src, dst_min, dst_max}, {minmax_over_axis_contig_ev});

            return std::make_pair(keep_args_event, minmax_over_axis_contig_ev);
        }
    }
    else if (is_src_f_contig &&
             ((is_dst_c_contig && dst_nd == 1) || is_dst_f_contig)) {
        auto fn = minmax_over_axis0_contig_temps_dispatch_vector[src_typeid];
        if (fn != nullptr) {
            std::size_t iter_nelems = dst_nelems;

            sycl::event minmax_over_axis_contig_ev =
                fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                   dst_min.get_data(), dst_max.get_data(),
                   zero_offset, // iteration_src_offset
                   zero_offset, // iteration_dst_min_offset
                   zero_offset, // iteration_dst_max_offset
                   zero_offset, // reduction_src_offset
                   depends);

            sycl::event keep_args_event = dpnp::utils::keep_args_alive(
                exec_q, {src, dst_min, dst_max}, {minmax_over_axis_contig_ev});

            return std::make_pair(keep_args_event, minmax_over_axis_contig_ev);
        }
    }

    auto const &src_strides_vecs = src.get_strides_vector();
    auto const &dst_min_strides_vecs = dst_min.get_strides_vector();
    auto const &dst_max_strides_vecs = dst_max.get_strides_vector();

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

    shT simplified_iteration_shape;
    shT simplified_iteration_src_strides;
    shT simplified_iteration_dst_min_strides;
    shT simplified_iteration_dst_max_strides;
    py::ssize_t iteration_src_offset(0);
    py::ssize_t iteration_dst_min_offset(0);
    py::ssize_t iteration_dst_max_offset(0);

    if (iteration_nd == 0) {
        if (dst_nelems != 1) {
            throw std::runtime_error("iteration_nd == 0, but dst_nelems != 1");
        }
        iteration_nd = 1;
        simplified_iteration_shape.push_back(1);
        simplified_iteration_src_strides.push_back(0);
        simplified_iteration_dst_min_strides.push_back(0);
        simplified_iteration_dst_max_strides.push_back(0);
    }
    else {
        simplify_iteration_space_3(
            iteration_nd, iteration_shape_ptr, iteration_src_strides,
            dst_min_strides_vecs, dst_max_strides_vecs,
            // output
            simplified_iteration_shape, simplified_iteration_src_strides,
            simplified_iteration_dst_min_strides,
            simplified_iteration_dst_max_strides, iteration_src_offset,
            iteration_dst_min_offset, iteration_dst_max_offset);
    }

    if ((reduction_nd == 1) && (iteration_nd == 1)) {
        bool mat_reduce_over_axis1 = false;
        bool mat_reduce_over_axis0 = false;
        bool array_reduce_all_elems = false;
        std::size_t iter_nelems = dst_nelems;

        bool unit_dst_strides =
            (simplified_iteration_dst_min_strides[0] == 1) &&
            (simplified_iteration_dst_max_strides[0] == 1);

        if (simplified_reduction_src_strides[0] == 1) {
            array_reduce_all_elems = (simplified_iteration_shape[0] == 1);
            mat_reduce_over_axis1 =
                unit_dst_strides &&
                (static_cast<std::size_t>(
                     simplified_iteration_src_strides[0]) == reduction_nelems);
        }
        else if (static_cast<std::size_t>(
                     simplified_reduction_src_strides[0]) == iter_nelems) {
            mat_reduce_over_axis0 =
                unit_dst_strides && (simplified_iteration_src_strides[0] == 1);
        }

        if (mat_reduce_over_axis1 || array_reduce_all_elems) {
            auto fn =
                minmax_over_axis1_contig_temps_dispatch_vector[src_typeid];
            if (fn != nullptr) {
                sycl::event minmax_over_axis1_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst_min.get_data(), dst_max.get_data(),
                       iteration_src_offset, iteration_dst_min_offset,
                       iteration_dst_max_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpnp::utils::keep_args_alive(
                    exec_q, {src, dst_min, dst_max},
                    {minmax_over_axis1_contig_ev});

                return std::make_pair(keep_args_event,
                                      minmax_over_axis1_contig_ev);
            }
        }
        else if (mat_reduce_over_axis0) {
            auto fn =
                minmax_over_axis0_contig_temps_dispatch_vector[src_typeid];
            if (fn != nullptr) {
                sycl::event minmax_over_axis0_contig_ev =
                    fn(exec_q, iter_nelems, reduction_nelems, src.get_data(),
                       dst_min.get_data(), dst_max.get_data(),
                       iteration_src_offset, iteration_dst_min_offset,
                       iteration_dst_max_offset, reduction_src_offset, depends);

                sycl::event keep_args_event = dpnp::utils::keep_args_alive(
                    exec_q, {src, dst_min, dst_max},
                    {minmax_over_axis0_contig_ev});

                return std::make_pair(keep_args_event,
                                      minmax_over_axis0_contig_ev);
            }
        }
    }

    auto fn = minmax_over_axis_strided_temps_dispatch_vector[src_typeid];
    if (fn == nullptr) {
        throw std::runtime_error("Datatypes are not supported");
    }

    std::vector<sycl::event> host_task_events{};
    using dpnp::tensor::offset_utils::device_allocate_and_pack;
    auto arrays_metainfo_packing_triple_ =
        device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events,
            // iteration metadata
            simplified_iteration_shape, simplified_iteration_src_strides,
            simplified_iteration_dst_min_strides,
            simplified_iteration_dst_max_strides,
            // reduction metadata
            simplified_reduction_shape, simplified_reduction_src_strides);
    auto tmp_owner = std::move(std::get<0>(arrays_metainfo_packing_triple_));
    const auto &copy_metadata_ev = std::get<2>(arrays_metainfo_packing_triple_);
    const py::ssize_t *temp_allocation_ptr = tmp_owner.get();

    const py::ssize_t *iter_shape_and_strides = temp_allocation_ptr;
    const py::ssize_t *reduction_shape_stride =
        temp_allocation_ptr + 4 * simplified_iteration_shape.size();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.resize(depends.size());
    std::copy(depends.begin(), depends.end(), all_deps.begin());
    all_deps.push_back(copy_metadata_ev);

    auto minmax_ev = fn(exec_q, dst_nelems, reduction_nelems, src.get_data(),
                        dst_min.get_data(), dst_max.get_data(), iteration_nd,
                        iter_shape_and_strides, iteration_src_offset,
                        iteration_dst_min_offset, iteration_dst_max_offset,
                        reduction_nd, // number dimensions being reduced
                        reduction_shape_stride, reduction_src_offset, all_deps);

    sycl::event temp_cleanup_ev = dpnp::tensor::alloc_utils::async_smart_free(
        exec_q, {minmax_ev}, tmp_owner);
    host_task_events.push_back(temp_cleanup_ev);

    sycl::event keep_args_event = dpnp::utils::keep_args_alive(
        exec_q, {src, dst_min, dst_max}, host_task_events);

    return std::make_pair(keep_args_event, minmax_ev);
}

void init_minmax(py::module_ m)
{
    impl::populate_minmax_over_axis_dispatch_vectors();

    m.def("_minmax_over_axis", &py_minmax_over_axis,
          "Computes the minimum and the maximum of `src` over its trailing "
          "`trailing_dims_to_reduce` axes in a single pass, storing them to "
          "`dst_min` and `dst_max`.",
          py::arg("src"), py::arg("trailing_dims_to_reduce"),
          py::arg("dst_min"), py::arg("dst_max"), py::arg("sycl_queue"),
          py::arg("depends") = py::list());
}

} // namespace dpnp::tensor::py_internal
