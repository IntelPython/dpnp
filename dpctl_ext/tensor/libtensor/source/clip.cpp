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
/// This file defines Python API for implementation functions of
/// dpctl.tensor.clip
//===---------------------------------------------------------------------===//

#include <complex>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clip.hpp"
#include "kernels/clip.hpp"
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

using dpctl::tensor::kernels::clip::clip_contig_impl_fn_ptr_t;
using dpctl::tensor::kernels::clip::clip_strided_impl_fn_ptr_t;

static clip_contig_impl_fn_ptr_t clip_contig_dispatch_vector[td_ns::num_types];
static clip_strided_impl_fn_ptr_t
    clip_strided_dispatch_vector[td_ns::num_types];

void init_clip_dispatch_vectors(void)
{
    using namespace td_ns;
    using dpctl::tensor::kernels::clip::ClipContigFactory;
    DispatchVectorBuilder<clip_contig_impl_fn_ptr_t, ClipContigFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(clip_contig_dispatch_vector);

    using dpctl::tensor::kernels::clip::ClipStridedFactory;
    DispatchVectorBuilder<clip_strided_impl_fn_ptr_t, ClipStridedFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(clip_strided_dispatch_vector);
}

using dpctl::utils::keep_args_alive;

std::pair<sycl::event, sycl::event>
    py_clip(const dpctl::tensor::usm_ndarray &src,
            const dpctl::tensor::usm_ndarray &min,
            const dpctl::tensor::usm_ndarray &max,
            const dpctl::tensor::usm_ndarray &dst,
            sycl::queue &exec_q,
            const std::vector<sycl::event> &depends)
{

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, min, max, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    int nd = src.get_ndim();
    int min_nd = min.get_ndim();
    int max_nd = max.get_ndim();
    int dst_nd = dst.get_ndim();

    if (nd != min_nd || nd != max_nd) {
        throw py::value_error(
            "Input arrays are not of appropriate dimension for clip kernel.");
    }

    if (nd != dst_nd) {
        throw py::value_error(
            "Destination is not of appropriate dimension for clip kernel.");
    }

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *min_shape = min.get_shape_raw();
    const py::ssize_t *max_shape = max.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();

    bool shapes_equal(true);
    std::size_t nelems(1);
    for (int i = 0; i < nd; ++i) {
        const auto &sh_i = dst_shape[i];
        nelems *= static_cast<std::size_t>(sh_i);
        shapes_equal = shapes_equal && (min_shape[i] == sh_i) &&
                       (max_shape[i] == sh_i) && (src_shape[i] == sh_i);
    }

    if (!shapes_equal) {
        throw py::value_error("Arrays are not of matching shapes.");
    }

    if (nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    auto const &same_logical_tensors =
        dpctl::tensor::overlap::SameLogicalTensors();
    if ((overlap(dst, src) && !same_logical_tensors(dst, src)) ||
        (overlap(dst, min) && !same_logical_tensors(dst, min)) ||
        (overlap(dst, max) && !same_logical_tensors(dst, max)))
    {
        throw py::value_error("Destination array overlaps with input.");
    }

    int min_typenum = min.get_typenum();
    int max_typenum = max.get_typenum();
    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int min_typeid = array_types.typenum_to_lookup_id(min_typenum);
    int max_typeid = array_types.typenum_to_lookup_id(max_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_typeid != dst_typeid || src_typeid != min_typeid ||
        src_typeid != max_typeid)
    {
        throw py::value_error("Input, min, max, and destination arrays must "
                              "have the same data type");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, nelems);

    char *src_data = src.get_data();
    char *min_data = min.get_data();
    char *max_data = max.get_data();
    char *dst_data = dst.get_data();

    bool is_min_c_contig = min.is_c_contiguous();
    bool is_min_f_contig = min.is_f_contiguous();

    bool is_max_c_contig = max.is_c_contiguous();
    bool is_max_f_contig = max.is_f_contiguous();

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_src_f_contig = src.is_f_contiguous();

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();

    bool all_c_contig = (is_min_c_contig && is_max_c_contig &&
                         is_src_c_contig && is_dst_c_contig);
    bool all_f_contig = (is_min_f_contig && is_max_f_contig &&
                         is_src_f_contig && is_dst_f_contig);

    if (all_c_contig || all_f_contig) {
        auto fn = clip_contig_dispatch_vector[src_typeid];

        sycl::event clip_ev =
            fn(exec_q, nelems, src_data, min_data, max_data, dst_data, depends);
        sycl::event ht_ev =
            keep_args_alive(exec_q, {src, min, max, dst}, {clip_ev});

        return std::make_pair(ht_ev, clip_ev);
    }

    auto const &src_strides = src.get_strides_vector();
    auto const &min_strides = min.get_strides_vector();
    auto const &max_strides = max.get_strides_vector();
    auto const &dst_strides = dst.get_strides_vector();

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_min_strides;
    shT simplified_max_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t min_offset(0);
    py::ssize_t max_offset(0);
    py::ssize_t dst_offset(0);

    dpctl::tensor::py_internal::simplify_iteration_space_4(
        nd, src_shape, src_strides, min_strides, max_strides, dst_strides,
        // outputs
        simplified_shape, simplified_src_strides, simplified_min_strides,
        simplified_max_strides, simplified_dst_strides, src_offset, min_offset,
        max_offset, dst_offset);

    auto fn = clip_strided_dispatch_vector[src_typeid];

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events,
        // common shape and strides
        simplified_shape, simplified_src_strides, simplified_min_strides,
        simplified_max_strides, simplified_dst_strides);
    auto packed_shape_strides_owner =
        std::move(std::get<0>(ptr_size_event_tuple));
    sycl::event copy_shape_strides_ev = std::get<2>(ptr_size_event_tuple);
    const py::ssize_t *packed_shape_strides = packed_shape_strides_owner.get();

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + 1);
    all_deps.insert(all_deps.end(), depends.begin(), depends.end());
    all_deps.push_back(copy_shape_strides_ev);

    assert(all_deps.size() == depends.size() + 1);

    sycl::event clip_ev = fn(exec_q, nelems, nd, src_data, min_data, max_data,
                             dst_data, packed_shape_strides, src_offset,
                             min_offset, max_offset, dst_offset, all_deps);

    // free packed temporaries
    sycl::event temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {clip_ev}, packed_shape_strides_owner);
    host_task_events.push_back(temporaries_cleanup_ev);

    sycl::event arg_cleanup_ev =
        keep_args_alive(exec_q, {src, min, max, dst}, host_task_events);

    return std::make_pair(arg_cleanup_ev, clip_ev);
}

} // namespace dpctl::tensor::py_internal
