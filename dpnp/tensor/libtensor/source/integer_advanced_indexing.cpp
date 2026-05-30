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
/// This file defines implementation functions of dpnp.tensor.take and
/// dpnp.tensor.put
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>

#include "kernels/integer_advanced_indexing.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "integer_advanced_indexing.hpp"

namespace dpnp::tensor::py_internal
{

namespace td_ns = dpnp::tensor::type_dispatch;

using dpnp::tensor::kernels::indexing::put_fn_ptr_t;
using dpnp::tensor::kernels::indexing::take_fn_ptr_t;

static take_fn_ptr_t take_wrap_dispatch_table[td_ns::num_types]
                                             [td_ns::num_types];

static take_fn_ptr_t take_clip_dispatch_table[td_ns::num_types]
                                             [td_ns::num_types];

static put_fn_ptr_t put_wrap_dispatch_table[td_ns::num_types][td_ns::num_types];

static put_fn_ptr_t put_clip_dispatch_table[td_ns::num_types][td_ns::num_types];

namespace py = pybind11;

namespace detail
{

using host_ptr_allocator_t =
    dpnp::tensor::alloc_utils::usm_host_allocator<char *>;
using host_ptr_vec_t = std::vector<char *, host_ptr_allocator_t>;
using host_ptr_shp_t = std::shared_ptr<host_ptr_vec_t>;

using host_sz_allocator_t =
    dpnp::tensor::alloc_utils::usm_host_allocator<py::ssize_t>;
using host_sz_vec_t = std::vector<py::ssize_t, host_sz_allocator_t>;
using host_sz_shp_t = std::shared_ptr<host_sz_vec_t>;

template <typename T>
sycl::event copy_to_device(sycl::queue &exec_q,
                           const T *host_data,
                           T *device_data,
                           std::size_t count)
{
    return exec_q.copy<T>(host_data, device_data, count);
}

host_ptr_shp_t allocate_and_copy_ptrs(sycl::queue &exec_q,
                                      const std::vector<char *> &ptrs)
{
    host_ptr_allocator_t allocator(exec_q);
    auto host_shp = std::make_shared<host_ptr_vec_t>(ptrs.size(), allocator);
    std::copy(ptrs.begin(), ptrs.end(), host_shp->begin());
    return host_shp;
}

host_sz_shp_t allocate_and_copy_sizes(sycl::queue &exec_q,
                                      const std::vector<py::ssize_t> &sizes)
{
    host_sz_allocator_t allocator(exec_q);
    auto host_shp = std::make_shared<host_sz_vec_t>(sizes.size(), allocator);
    std::copy(sizes.begin(), sizes.end(), host_shp->begin());
    return host_shp;
}

host_sz_shp_t allocate_host_buffer(sycl::queue &exec_q, std::size_t size)
{
    host_sz_allocator_t allocator(exec_q);
    return std::make_shared<host_sz_vec_t>(size, allocator);
}

void copy_axis_shape_strides(int axis_start,
                             int inp_nd,
                             int k,
                             int ind_nd,
                             const py::ssize_t *inp_shape,
                             const std::vector<py::ssize_t> &inp_strides,
                             const py::ssize_t *arr_shape,
                             const std::vector<py::ssize_t> &arr_strides,
                             py::ssize_t *host_along_sh_st)
{
    if (inp_nd > 0) {
        std::copy(inp_shape + axis_start, inp_shape + axis_start + k,
                  host_along_sh_st);
        std::copy(inp_strides.begin() + axis_start,
                  inp_strides.begin() + axis_start + k, host_along_sh_st + k);
    }

    if (ind_nd > 0) {
        std::copy(arr_shape + axis_start, arr_shape + axis_start + ind_nd,
                  host_along_sh_st + 2 * k);
        std::copy(arr_strides.begin() + axis_start,
                  arr_strides.begin() + axis_start + ind_nd,
                  host_along_sh_st + 2 * k + ind_nd);
    }
}

void copy_orthog_shape_strides(int axis_start,
                               int inp_nd,
                               int k,
                               int ind_nd,
                               int orthog_sh_elems,
                               const py::ssize_t *inp_shape,
                               const std::vector<py::ssize_t> &inp_strides,
                               const std::vector<py::ssize_t> &arr_strides,
                               py::ssize_t *host_orthog_sh_st)
{
    int orthog_nd = inp_nd - k;
    if (orthog_nd == 0) {
        return;
    }

    if (axis_start > 0) {
        std::copy(inp_shape, inp_shape + axis_start, host_orthog_sh_st);
        std::copy(inp_strides.begin(), inp_strides.begin() + axis_start,
                  host_orthog_sh_st + orthog_sh_elems);
        std::copy(arr_strides.begin(), arr_strides.begin() + axis_start,
                  host_orthog_sh_st + 2 * orthog_sh_elems);
    }
    if (inp_nd > (axis_start + k)) {
        std::copy(inp_shape + axis_start + k, inp_shape + inp_nd,
                  host_orthog_sh_st + axis_start);
        std::copy(inp_strides.begin() + axis_start + k, inp_strides.end(),
                  host_orthog_sh_st + orthog_sh_elems + axis_start);
        std::copy(arr_strides.begin() + axis_start + ind_nd, arr_strides.end(),
                  host_orthog_sh_st + 2 * orthog_sh_elems + axis_start);
    }
}

void validate_index_array(const dpnp::tensor::usm_ndarray &ind_,
                          const sycl::queue &exec_q,
                          int ind_nd,
                          int ind_type_id,
                          const py::ssize_t *ind_shape,
                          const td_ns::usm_ndarray_types &array_types,
                          const dpnp::tensor::overlap::MemoryOverlap &overlap,
                          const dpnp::tensor::usm_ndarray &other_array)
{
    if (!dpnp::utils::queues_are_compatible(exec_q, {ind_})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    if (ind_.get_ndim() != ind_nd) {
        throw py::value_error("Index dimensions are not the same");
    }

    if (ind_type_id != array_types.typenum_to_lookup_id(ind_.get_typenum())) {
        throw py::type_error("Indices array data types are not all the same.");
    }

    const py::ssize_t *ind_shape_ = ind_.get_shape_raw();
    for (int dim = 0; dim < ind_nd; ++dim) {
        if (ind_shape[dim] != ind_shape_[dim]) {
            throw py::value_error("Indices shapes are not all equal.");
        }
    }

    if (overlap(ind_, other_array)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }
}

void process_index_arrays(const std::vector<dpnp::tensor::usm_ndarray> &ind,
                          sycl::queue &exec_q,
                          int k,
                          int ind_nd,
                          int ind_sh_elems,
                          const py::ssize_t *ind_shape,
                          int ind_type_id,
                          const td_ns::usm_ndarray_types &array_types,
                          const dpnp::tensor::overlap::MemoryOverlap &overlap,
                          const dpnp::tensor::usm_ndarray &other_array,
                          std::vector<char *> &ind_ptrs,
                          std::vector<py::ssize_t> &ind_offsets,
                          std::vector<py::ssize_t> &ind_sh_sts)
{
    for (int i = 0; i < k; ++i) {
        const dpnp::tensor::usm_ndarray &ind_ = ind[i];

        if (i > 0) {
            validate_index_array(ind_, exec_q, ind_nd, ind_type_id, ind_shape,
                                 array_types, overlap, other_array);
        }
        else {
            if (!dpnp::utils::queues_are_compatible(exec_q, {ind_})) {
                throw py::value_error(
                    "Execution queue is not compatible with allocation queues");
            }
            if (overlap(ind_, other_array)) {
                throw py::value_error(
                    "Arrays index overlapping segments of memory");
            }
        }

        char *ind_data = ind_.get_data();

        if (ind_nd > 0) {
            auto ind_strides = ind_.get_strides_vector();
            std::copy(ind_strides.begin(), ind_strides.end(),
                      ind_sh_sts.begin() + (i + 1) * ind_nd);
        }

        ind_ptrs.push_back(ind_data);
        ind_offsets.push_back(py::ssize_t(0));
    }
}

} // namespace detail

std::vector<sycl::event>
    _populate_kernel_params(sycl::queue &exec_q,
                            std::vector<sycl::event> &host_task_events,
                            char **device_ind_ptrs,
                            py::ssize_t *device_ind_sh_st,
                            py::ssize_t *device_ind_offsets,
                            py::ssize_t *device_orthog_sh_st,
                            py::ssize_t *device_along_sh_st,
                            const py::ssize_t *inp_shape,
                            const py::ssize_t *arr_shape,
                            std::vector<py::ssize_t> &inp_strides,
                            std::vector<py::ssize_t> &arr_strides,
                            std::vector<py::ssize_t> &ind_sh_sts,
                            std::vector<char *> &ind_ptrs,
                            std::vector<py::ssize_t> &ind_offsets,
                            int axis_start,
                            int k,
                            int ind_nd,
                            int inp_nd,
                            int orthog_sh_elems,
                            int ind_sh_elems)
{
    auto host_ind_ptrs_shp = detail::allocate_and_copy_ptrs(exec_q, ind_ptrs);
    auto host_ind_sh_st_shp =
        detail::allocate_and_copy_sizes(exec_q, ind_sh_sts);
    auto host_ind_offsets_shp =
        detail::allocate_and_copy_sizes(exec_q, ind_offsets);

    auto host_orthog_sh_st_shp =
        detail::allocate_host_buffer(exec_q, 3 * orthog_sh_elems);
    auto host_along_sh_st_shp =
        detail::allocate_host_buffer(exec_q, 2 * (k + ind_sh_elems));

    detail::copy_orthog_shape_strides(
        axis_start, inp_nd, k, ind_nd, orthog_sh_elems, inp_shape, inp_strides,
        arr_strides, host_orthog_sh_st_shp->data());

    detail::copy_axis_shape_strides(axis_start, inp_nd, k, ind_nd, inp_shape,
                                    inp_strides, arr_shape, arr_strides,
                                    host_along_sh_st_shp->data());

    const sycl::event device_ind_ptrs_copy_ev =
        detail::copy_to_device(exec_q, host_ind_ptrs_shp->data(),
                               device_ind_ptrs, host_ind_ptrs_shp->size());

    const sycl::event device_ind_sh_st_copy_ev =
        detail::copy_to_device(exec_q, host_ind_sh_st_shp->data(),
                               device_ind_sh_st, host_ind_sh_st_shp->size());

    const sycl::event device_ind_offsets_copy_ev = detail::copy_to_device(
        exec_q, host_ind_offsets_shp->data(), device_ind_offsets,
        host_ind_offsets_shp->size());

    const sycl::event device_orthog_sh_st_copy_ev = detail::copy_to_device(
        exec_q, host_orthog_sh_st_shp->data(), device_orthog_sh_st,
        host_orthog_sh_st_shp->size());

    const sycl::event device_along_sh_st_copy_ev = detail::copy_to_device(
        exec_q, host_along_sh_st_shp->data(), device_along_sh_st,
        host_along_sh_st_shp->size());

    const sycl::event shared_ptr_cleanup_ev =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on({device_along_sh_st_copy_ev,
                            device_orthog_sh_st_copy_ev,
                            device_ind_offsets_copy_ev,
                            device_ind_sh_st_copy_ev, device_ind_ptrs_copy_ev});
            cgh.host_task(
                [host_ind_offsets_shp = std::move(host_ind_offsets_shp),
                 host_ind_sh_st_shp = std::move(host_ind_sh_st_shp),
                 host_ind_ptrs_shp = std::move(host_ind_ptrs_shp),
                 host_orthog_sh_st_shp = std::move(host_orthog_sh_st_shp),
                 host_along_sh_st_shp = std::move(host_along_sh_st_shp)] {});
        });
    host_task_events.push_back(shared_ptr_cleanup_ev);

    return {device_ind_ptrs_copy_ev, device_ind_sh_st_copy_ev,
            device_ind_offsets_copy_ev, device_orthog_sh_st_copy_ev,
            device_along_sh_st_copy_ev};
}

/* Utility to parse python object py_ind into vector of `usm_ndarray`s */
std::vector<dpnp::tensor::usm_ndarray> parse_py_ind(const sycl::queue &q,
                                                    const py::object &py_ind)
{
    std::size_t ind_count = py::len(py_ind);
    std::vector<dpnp::tensor::usm_ndarray> res;
    res.reserve(ind_count);

    bool nd_is_known = false;
    int nd = -1;
    for (std::size_t i = 0; i < ind_count; ++i) {
        py::object el_i = py_ind[py::cast(i)];
        dpnp::tensor::usm_ndarray arr_i =
            py::cast<dpnp::tensor::usm_ndarray>(el_i);
        if (!dpnp::utils::queues_are_compatible(q, {arr_i})) {
            throw py::value_error("Index allocation queue is not compatible "
                                  "with execution queue");
        }
        if (nd_is_known) {
            if (nd != arr_i.get_ndim()) {
                throw py::value_error(
                    "Indices must have the same number of dimensions.");
            }
        }
        else {
            nd_is_known = true;
            nd = arr_i.get_ndim();
        }
        res.push_back(arr_i);
    }

    return res;
}

std::pair<sycl::event, sycl::event>
    py_take(const dpnp::tensor::usm_ndarray &src,
            const py::object &py_ind,
            const dpnp::tensor::usm_ndarray &dst,
            int axis_start,
            std::uint8_t mode,
            sycl::queue &exec_q,
            const std::vector<sycl::event> &depends)
{
    std::vector<dpnp::tensor::usm_ndarray> ind = parse_py_ind(exec_q, py_ind);

    int k = ind.size();

    if (k == 0) {
        throw py::value_error("List of indices is empty.");
    }

    if (axis_start < 0) {
        throw py::value_error("Axis cannot be negative.");
    }

    if (mode != 0 && mode != 1) {
        throw py::value_error("Mode must be 0 or 1.");
    }

    dpnp::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    const dpnp::tensor::usm_ndarray ind_rep = ind[0];

    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();
    int ind_nd = ind_rep.get_ndim();

    auto sh_elems = std::max<int>(src_nd, 1);

    if (axis_start + k > sh_elems) {
        throw py::value_error("Axes are out of range for array of dimension " +
                              std::to_string(src_nd));
    }
    if (src_nd == 0) {
        if (dst_nd != ind_nd) {
            throw py::value_error(
                "Destination is not of appropriate dimension for take kernel.");
        }
    }
    else {
        if (dst_nd != (src_nd - k + ind_nd)) {
            throw py::value_error(
                "Destination is not of appropriate dimension for take kernel.");
        }
    }

    const py::ssize_t *src_shape = src.get_shape_raw();
    const py::ssize_t *dst_shape = dst.get_shape_raw();

    bool orthog_shapes_equal(true);
    std::size_t orthog_nelems(1);
    for (int i = 0; i < (src_nd - k); ++i) {
        auto idx1 = (i < axis_start) ? i : i + k;
        auto idx2 = (i < axis_start) ? i : i + ind_nd;

        orthog_nelems *= static_cast<std::size_t>(src_shape[idx1]);
        orthog_shapes_equal =
            orthog_shapes_equal && (src_shape[idx1] == dst_shape[idx2]);
    }

    if (!orthog_shapes_equal) {
        throw py::value_error(
            "Axes of basic indices are not of matching shapes.");
    }

    if (orthog_nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    char *src_data = src.get_data();
    char *dst_data = dst.get_data();

    if (!dpnp::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpnp::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Array memory overlap.");
    }

    py::ssize_t src_offset = py::ssize_t(0);
    py::ssize_t dst_offset = py::ssize_t(0);

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_type_id != dst_type_id) {
        throw py::type_error("Array data types are not the same.");
    }

    const py::ssize_t *ind_shape = ind_rep.get_shape_raw();

    int ind_typenum = ind_rep.get_typenum();
    int ind_type_id = array_types.typenum_to_lookup_id(ind_typenum);

    std::size_t ind_nelems(1);
    for (int i = 0; i < ind_nd; ++i) {
        ind_nelems *= static_cast<std::size_t>(ind_shape[i]);

        if (!(ind_shape[i] == dst_shape[axis_start + i])) {
            throw py::value_error(
                "Indices shape does not match shape of axis in destination.");
        }
    }

    dpnp::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, orthog_nelems * ind_nelems);

    int ind_sh_elems = std::max<int>(ind_nd, 1);

    std::vector<char *> ind_ptrs;
    ind_ptrs.reserve(k);

    std::vector<py::ssize_t> ind_offsets;
    ind_offsets.reserve(k);

    std::vector<py::ssize_t> ind_sh_sts((k + 1) * ind_sh_elems, 0);
    if (ind_nd > 0) {
        std::copy(ind_shape, ind_shape + ind_nd, ind_sh_sts.begin());
    }

    detail::process_index_arrays(ind, exec_q, k, ind_nd, ind_sh_elems,
                                 ind_shape, ind_type_id, array_types, overlap,
                                 dst, ind_ptrs, ind_offsets, ind_sh_sts);

    if (ind_nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    auto packed_ind_ptrs_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<char *>(k, exec_q);
    char **packed_ind_ptrs = packed_ind_ptrs_owner.get();

    // rearrange to past where indices shapes are checked
    // packed_ind_shapes_strides = [ind_shape,
    //                              ind[0] strides,
    //                              ...,
    //                              ind[k] strides]
    auto packed_ind_shapes_strides_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(
            (k + 1) * ind_sh_elems, exec_q);
    py::ssize_t *packed_ind_shapes_strides =
        packed_ind_shapes_strides_owner.get();

    auto packed_ind_offsets_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(k, exec_q);
    py::ssize_t *packed_ind_offsets = packed_ind_offsets_owner.get();

    int orthog_sh_elems = std::max<int>(src_nd - k, 1);

    // packed_shapes_strides = [src_shape[:axis] + src_shape[axis+k:],
    //                          src_strides[:axis] + src_strides[axis+k:],
    //                          dst_strides[:axis] +
    //                          dst_strides[axis+ind.ndim:]]
    auto packed_shapes_strides_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(
            3 * orthog_sh_elems, exec_q);
    py::ssize_t *packed_shapes_strides = packed_shapes_strides_owner.get();

    // packed_axes_shapes_strides = [src_shape[axis:axis+k],
    //                               src_strides[axis:axis+k],
    //                               dst_shape[axis:axis+ind.ndim],
    //                               dst_strides[axis:axis+ind.ndim]]
    auto packed_axes_shapes_strides_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(
            2 * (k + ind_sh_elems), exec_q);
    py::ssize_t *packed_axes_shapes_strides =
        packed_axes_shapes_strides_owner.get();

    auto src_strides = src.get_strides_vector();
    auto dst_strides = dst.get_strides_vector();

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    std::vector<sycl::event> pack_deps = _populate_kernel_params(
        exec_q, host_task_events, packed_ind_ptrs, packed_ind_shapes_strides,
        packed_ind_offsets, packed_shapes_strides, packed_axes_shapes_strides,
        src_shape, dst_shape, src_strides, dst_strides, ind_sh_sts, ind_ptrs,
        ind_offsets, axis_start, k, ind_nd, src_nd, orthog_sh_elems,
        ind_sh_elems);

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + pack_deps.size());
    all_deps.insert(std::end(all_deps), std::begin(pack_deps),
                    std::end(pack_deps));
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    auto fn = mode ? take_clip_dispatch_table[src_type_id][ind_type_id]
                   : take_wrap_dispatch_table[src_type_id][ind_type_id];

    if (fn == nullptr) {
        sycl::event::wait(host_task_events);
        throw std::runtime_error("Indices must be integer type, got " +
                                 std::to_string(ind_type_id));
    }

    sycl::event take_generic_ev =
        fn(exec_q, orthog_nelems, ind_nelems, orthog_sh_elems, ind_sh_elems, k,
           packed_shapes_strides, packed_axes_shapes_strides,
           packed_ind_shapes_strides, src_data, dst_data, packed_ind_ptrs,
           src_offset, dst_offset, packed_ind_offsets, all_deps);

    // free packed temporaries
    sycl::event temporaries_cleanup_ev =
        dpnp::tensor::alloc_utils::async_smart_free(
            exec_q, {take_generic_ev}, packed_shapes_strides_owner,
            packed_axes_shapes_strides_owner, packed_ind_shapes_strides_owner,
            packed_ind_ptrs_owner, packed_ind_offsets_owner);
    host_task_events.push_back(temporaries_cleanup_ev);

    using dpnp::utils::keep_args_alive;
    sycl::event arg_cleanup_ev =
        keep_args_alive(exec_q, {src, py_ind, dst}, host_task_events);

    return std::make_pair(arg_cleanup_ev, take_generic_ev);
}

std::pair<sycl::event, sycl::event>
    py_put(const dpnp::tensor::usm_ndarray &dst,
           const py::object &py_ind,
           const dpnp::tensor::usm_ndarray &val,
           int axis_start,
           std::uint8_t mode,
           sycl::queue &exec_q,
           const std::vector<sycl::event> &depends)
{
    std::vector<dpnp::tensor::usm_ndarray> ind = parse_py_ind(exec_q, py_ind);
    int k = ind.size();

    if (k == 0) {
        // no indices to write to
        throw py::value_error("List of indices is empty.");
    }

    if (axis_start < 0) {
        throw py::value_error("Axis cannot be negative.");
    }

    if (mode != 0 && mode != 1) {
        throw py::value_error("Mode must be 0 or 1.");
    }

    dpnp::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    const dpnp::tensor::usm_ndarray ind_rep = ind[0];

    int dst_nd = dst.get_ndim();
    int val_nd = val.get_ndim();
    int ind_nd = ind_rep.get_ndim();

    auto sh_elems = std::max<int>(dst_nd, 1);

    if (axis_start + k > sh_elems) {
        throw py::value_error("Axes are out of range for array of dimension " +
                              std::to_string(dst_nd));
    }
    if (dst_nd == 0) {
        if (val_nd != ind_nd) {
            throw py::value_error("Destination is not of appropriate dimension "
                                  "for put function.");
        }
    }
    else {
        if (val_nd != (dst_nd - k + ind_nd)) {
            throw py::value_error("Destination is not of appropriate dimension "
                                  "for put function.");
        }
    }

    std::size_t dst_nelems = dst.get_size();

    const py::ssize_t *dst_shape = dst.get_shape_raw();
    const py::ssize_t *val_shape = val.get_shape_raw();

    bool orthog_shapes_equal(true);
    std::size_t orthog_nelems(1);
    for (int i = 0; i < (dst_nd - k); ++i) {
        auto idx1 = (i < axis_start) ? i : i + k;
        auto idx2 = (i < axis_start) ? i : i + ind_nd;

        orthog_nelems *= static_cast<std::size_t>(dst_shape[idx1]);
        orthog_shapes_equal =
            orthog_shapes_equal && (dst_shape[idx1] == val_shape[idx2]);
    }

    if (!orthog_shapes_equal) {
        throw py::value_error(
            "Axes of basic indices are not of matching shapes.");
    }

    if (orthog_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    char *dst_data = dst.get_data();
    char *val_data = val.get_data();

    if (!dpnp::utils::queues_are_compatible(exec_q, {dst, val})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    auto const &overlap = dpnp::tensor::overlap::MemoryOverlap();
    if (overlap(val, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    py::ssize_t dst_offset = py::ssize_t(0);
    py::ssize_t val_offset = py::ssize_t(0);

    dpnp::tensor::validation::AmpleMemory::throw_if_not_ample(dst, dst_nelems);

    int dst_typenum = dst.get_typenum();
    int val_typenum = val.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);
    int val_type_id = array_types.typenum_to_lookup_id(val_typenum);

    if (dst_type_id != val_type_id) {
        throw py::type_error("Array data types are not the same.");
    }

    const py::ssize_t *ind_shape = ind_rep.get_shape_raw();

    int ind_typenum = ind_rep.get_typenum();
    int ind_type_id = array_types.typenum_to_lookup_id(ind_typenum);

    std::size_t ind_nelems(1);
    for (int i = 0; i < ind_nd; ++i) {
        ind_nelems *= static_cast<std::size_t>(ind_shape[i]);

        if (!(ind_shape[i] == val_shape[axis_start + i])) {
            throw py::value_error(
                "Indices shapes does not match shape of axis in vals.");
        }
    }

    auto ind_sh_elems = std::max<int>(ind_nd, 1);

    std::vector<char *> ind_ptrs;
    ind_ptrs.reserve(k);
    std::vector<py::ssize_t> ind_offsets;
    ind_offsets.reserve(k);
    std::vector<py::ssize_t> ind_sh_sts((k + 1) * ind_sh_elems, py::ssize_t(0));
    if (ind_nd > 0) {
        std::copy(ind_shape, ind_shape + ind_nd, ind_sh_sts.begin());
    }

    detail::process_index_arrays(ind, exec_q, k, ind_nd, ind_sh_elems,
                                 ind_shape, ind_type_id, array_types, overlap,
                                 dst, ind_ptrs, ind_offsets, ind_sh_sts);

    if (ind_nelems == 0) {
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    auto packed_ind_ptrs_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<char *>(k, exec_q);
    char **packed_ind_ptrs = packed_ind_ptrs_owner.get();

    // packed_ind_shapes_strides = [ind_shape,
    //                              ind[0] strides,
    //                              ...,
    //                              ind[k] strides]
    auto packed_ind_shapes_strides_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(
            (k + 1) * ind_sh_elems, exec_q);
    py::ssize_t *packed_ind_shapes_strides =
        packed_ind_shapes_strides_owner.get();

    auto packed_ind_offsets_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(k, exec_q);
    py::ssize_t *packed_ind_offsets = packed_ind_offsets_owner.get();

    int orthog_sh_elems = std::max<int>(dst_nd - k, 1);

    // packed_shapes_strides = [dst_shape[:axis] + dst_shape[axis+k:],
    //                          dst_strides[:axis] + dst_strides[axis+k:],
    //                          val_strides[:axis] +
    //                          val_strides[axis+ind.ndim:]]
    auto packed_shapes_strides_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(
            3 * orthog_sh_elems, exec_q);
    py::ssize_t *packed_shapes_strides = packed_shapes_strides_owner.get();

    // packed_axes_shapes_strides = [dst_shape[axis:axis+k],
    //                               dst_strides[axis:axis+k],
    //                               val_shape[axis:axis+ind.ndim],
    //                               val_strides[axis:axis+ind.ndim]]
    auto packed_axes_shapes_strides_owner =
        dpnp::tensor::alloc_utils::smart_malloc_device<py::ssize_t>(
            2 * (k + ind_sh_elems), exec_q);
    py::ssize_t *packed_axes_shapes_strides =
        packed_axes_shapes_strides_owner.get();

    auto dst_strides = dst.get_strides_vector();
    auto val_strides = val.get_strides_vector();

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    std::vector<sycl::event> pack_deps = _populate_kernel_params(
        exec_q, host_task_events, packed_ind_ptrs, packed_ind_shapes_strides,
        packed_ind_offsets, packed_shapes_strides, packed_axes_shapes_strides,
        dst_shape, val_shape, dst_strides, val_strides, ind_sh_sts, ind_ptrs,
        ind_offsets, axis_start, k, ind_nd, dst_nd, orthog_sh_elems,
        ind_sh_elems);

    std::vector<sycl::event> all_deps;
    all_deps.reserve(depends.size() + pack_deps.size());
    all_deps.insert(std::end(all_deps), std::begin(pack_deps),
                    std::end(pack_deps));
    all_deps.insert(std::end(all_deps), std::begin(depends), std::end(depends));

    auto fn = mode ? put_clip_dispatch_table[dst_type_id][ind_type_id]
                   : put_wrap_dispatch_table[dst_type_id][ind_type_id];

    if (fn == nullptr) {
        sycl::event::wait(host_task_events);
        throw std::runtime_error("Indices must be integer type, got " +
                                 std::to_string(ind_type_id));
    }

    sycl::event put_generic_ev =
        fn(exec_q, orthog_nelems, ind_nelems, orthog_sh_elems, ind_sh_elems, k,
           packed_shapes_strides, packed_axes_shapes_strides,
           packed_ind_shapes_strides, dst_data, val_data, packed_ind_ptrs,
           dst_offset, val_offset, packed_ind_offsets, all_deps);

    // free packed temporaries
    sycl::event temporaries_cleanup_ev =
        dpnp::tensor::alloc_utils::async_smart_free(
            exec_q, {put_generic_ev}, packed_shapes_strides_owner,
            packed_axes_shapes_strides_owner, packed_ind_shapes_strides_owner,
            packed_ind_ptrs_owner, packed_ind_offsets_owner);
    host_task_events.push_back(temporaries_cleanup_ev);

    using dpnp::utils::keep_args_alive;
    sycl::event arg_cleanup_ev =
        keep_args_alive(exec_q, {dst, py_ind, val}, host_task_events);

    return std::make_pair(arg_cleanup_ev, put_generic_ev);
}

void init_advanced_indexing_dispatch_tables(void)
{
    using namespace td_ns;

    using dpnp::tensor::kernels::indexing::TakeClipFactory;
    DispatchTableBuilder<take_fn_ptr_t, TakeClipFactory, num_types>
        dtb_takeclip;
    dtb_takeclip.populate_dispatch_table(take_clip_dispatch_table);

    using dpnp::tensor::kernels::indexing::TakeWrapFactory;
    DispatchTableBuilder<take_fn_ptr_t, TakeWrapFactory, num_types>
        dtb_takewrap;
    dtb_takewrap.populate_dispatch_table(take_wrap_dispatch_table);

    using dpnp::tensor::kernels::indexing::PutClipFactory;
    DispatchTableBuilder<put_fn_ptr_t, PutClipFactory, num_types> dtb_putclip;
    dtb_putclip.populate_dispatch_table(put_clip_dispatch_table);

    using dpnp::tensor::kernels::indexing::PutWrapFactory;
    DispatchTableBuilder<put_fn_ptr_t, PutWrapFactory, num_types> dtb_putwrap;
    dtb_putwrap.populate_dispatch_table(put_wrap_dispatch_table);
}

} // namespace dpnp::tensor::py_internal
