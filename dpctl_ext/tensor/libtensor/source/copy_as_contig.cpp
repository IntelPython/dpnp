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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>

#include "kernels/copy_as_contiguous.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"
#include "utils/type_dispatch.hpp"

#include "copy_as_contig.hpp"
#include "simplify_iteration_space.hpp"

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::kernels::copy_as_contig::
    as_c_contiguous_1d_batch_of_square_matrices_impl_fn_ptr_t;
using dpctl::tensor::kernels::copy_as_contig::
    as_c_contiguous_array_impl_fn_ptr_t;
using dpctl::tensor::kernels::copy_as_contig::
    as_c_contiguous_nd_batch_of_square_matrices_impl_fn_ptr_t;
using dpctl::utils::keep_args_alive;

static as_c_contiguous_array_impl_fn_ptr_t
    as_c_contig_array_dispatch_vector[td_ns::num_types];

static as_c_contiguous_1d_batch_of_square_matrices_impl_fn_ptr_t
    as_c_contig_1d_batch_of_square_matrices_dispatch_vector[td_ns::num_types];

static as_c_contiguous_nd_batch_of_square_matrices_impl_fn_ptr_t
    as_c_contig_nd_batch_of_square_matrices_dispatch_vector[td_ns::num_types];

void init_copy_as_contig_dispatch_vectors(void)
{

    using dpctl::tensor::kernels::copy_as_contig::
        AsCContig1DBatchOfSquareMatricesFactory;
    using dpctl::tensor::kernels::copy_as_contig::AsCContigFactory;
    using dpctl::tensor::kernels::copy_as_contig::
        AsCContigNDBatchOfSquareMatricesFactory;
    using td_ns::DispatchVectorBuilder;

    // Generic to c-contig
    DispatchVectorBuilder<as_c_contiguous_array_impl_fn_ptr_t, AsCContigFactory,
                          td_ns::num_types>
        dtv_as_c_contig_array;

    dtv_as_c_contig_array.populate_dispatch_vector(
        as_c_contig_array_dispatch_vector);

    // 1D batch of square views into F-contig matrices to c-contig array
    DispatchVectorBuilder<
        as_c_contiguous_1d_batch_of_square_matrices_impl_fn_ptr_t,
        AsCContig1DBatchOfSquareMatricesFactory, td_ns::num_types>
        dtv_as_c_contig_1d_batch_of_square_matrices;

    dtv_as_c_contig_1d_batch_of_square_matrices.populate_dispatch_vector(
        as_c_contig_1d_batch_of_square_matrices_dispatch_vector);

    // ND batch of square views into F-contig matrices to c-contig array
    DispatchVectorBuilder<
        as_c_contiguous_nd_batch_of_square_matrices_impl_fn_ptr_t,
        AsCContigNDBatchOfSquareMatricesFactory, td_ns::num_types>
        dtv_as_c_contig_nd_batch_of_square_matrices;

    dtv_as_c_contig_nd_batch_of_square_matrices.populate_dispatch_vector(
        as_c_contig_nd_batch_of_square_matrices_dispatch_vector);
}

namespace
{

template <typename dimT>
std::size_t get_nelems(const std::vector<dimT> &shape)
{
    auto mult_fn = [](std::size_t prod, const dimT &term) -> std::size_t {
        return prod * static_cast<std::size_t>(term);
    };

    static constexpr std::size_t unit{1};

    const std::size_t nelems =
        std::accumulate(std::begin(shape), std::end(shape), unit, mult_fn);
    return nelems;
}

} // end of anonymous namespace

std::pair<sycl::event, sycl::event>
    py_as_c_contig_f2c(const dpctl::tensor::usm_ndarray &src,
                       const dpctl::tensor::usm_ndarray &dst,
                       sycl::queue &exec_q,
                       const std::vector<sycl::event> &depends);

std::pair<sycl::event, sycl::event>
    py_as_c_contig(const dpctl::tensor::usm_ndarray &src,
                   const dpctl::tensor::usm_ndarray &dst,
                   sycl::queue &exec_q,
                   const std::vector<sycl::event> &depends)
{
    /*  Same dimensions, same shape, same data-type
     *  dst is C-contiguous.
     */
    const int src_nd = src.get_ndim();
    const int dst_nd = dst.get_ndim();

    if (src_nd != dst_nd) {
        throw py::value_error("Number of dimensions must be the same");
    }

    const auto &src_shape_vec = src.get_shape_vector();
    const auto &dst_shape_vec = dst.get_shape_vector();

    if (src_shape_vec != dst_shape_vec) {
        throw py::value_error("Shapes must be equal");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    const int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    const int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_type_id != dst_type_id) {
        throw py::value_error(
            "Source and destination arrays must have the same data type");
    }

    // ensures also that destination is plenty ample to accommodate all
    // elements of src array
    if (!dst.is_c_contiguous()) {
        throw py::value_error("Destination array must be C-contiguous");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    const auto &src_strides_vec = src.get_strides_vector();

    if (src_nd >= 2) {
        auto n = dst_shape_vec.back();
        if (n == dst_shape_vec[src_nd - 2]) {
            static constexpr auto unit_stride = py::ssize_t(1);
            if (src_strides_vec[src_nd - 2] == unit_stride) {
                return py_as_c_contig_f2c(src, dst, exec_q, depends);
            }
        }
    }

    const std::size_t nelems = get_nelems(src_shape_vec);

    if (nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    // simplify iteration space
    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_nd;

    // nd, simplified_* and *_offset are modified by reference
    dpctl::tensor::py_internal::simplify_iteration_space(
        nd, src_shape_vec.data(), src_strides_vec, dst.get_strides_vector(),
        // output
        simplified_shape, simplified_src_strides, simplified_dst_strides,
        src_offset, dst_offset);

    if (!((0 == src_offset) && (0 == dst_offset))) {
        throw std::runtime_error(
            "Unexpected result of simplifying iteration space, 1");
    }

    std::vector<sycl::event> host_task_events{};
    auto ptr_size_event_tuple =
        dpctl::tensor::offset_utils::device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, simplified_shape, simplified_src_strides);
    auto shape_stride_owner = std::move(std::get<0>(ptr_size_event_tuple));
    const sycl::event &copy_shape_ev = std::get<2>(ptr_size_event_tuple);
    const py::ssize_t *shape_stride = shape_stride_owner.get();

    auto ascontig_fn = as_c_contig_array_dispatch_vector[src_type_id];

    std::vector<sycl::event> all_depends;
    all_depends.reserve(depends.size() + 1);
    all_depends.insert(std::end(all_depends), std::begin(depends),
                       std::end(depends));
    all_depends.push_back(copy_shape_ev);

    sycl::event ascontig_ev =
        ascontig_fn(exec_q, nelems, nd, shape_stride, src.get_data(),
                    dst.get_data(), all_depends);

    const auto &temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(exec_q, {ascontig_ev},
                                                     shape_stride_owner);
    host_task_events.push_back(temporaries_cleanup_ev);

    return std::make_pair(keep_args_alive(exec_q, {src, dst}, host_task_events),
                          ascontig_ev);
}

std::pair<sycl::event, sycl::event>
    py_as_f_contig_c2f(const dpctl::tensor::usm_ndarray &src,
                       const dpctl::tensor::usm_ndarray &dst,
                       sycl::queue &exec_q,
                       const std::vector<sycl::event> &depends);

std::pair<sycl::event, sycl::event>
    py_as_f_contig(const dpctl::tensor::usm_ndarray &src,
                   const dpctl::tensor::usm_ndarray &dst,
                   sycl::queue &exec_q,
                   const std::vector<sycl::event> &depends)
{
    /*  Same dimensions, same shape, same data-type
     *  dst is F-contiguous.
     */
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    if (src_nd != dst_nd) {
        throw py::value_error("Number of dimensions must be the same");
    }

    const auto &src_shape_vec = src.get_shape_vector();
    const auto &dst_shape_vec = dst.get_shape_vector();

    if (src_shape_vec != dst_shape_vec) {
        throw py::value_error("Shapes must be equal");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    const int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    const int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_type_id != dst_type_id) {
        throw py::value_error(
            "Source and destination arrays must have the same data type");
    }

    // ensures also that destination is plenty ample to accommodate all
    // elements of src array
    if (!dst.is_f_contiguous()) {
        throw py::value_error("Destination array must be F-contiguous");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    const auto &src_strides_vec = src.get_strides_vector();

    if (src_nd >= 2) {
        auto n = dst_shape_vec.front();
        if (n == dst_shape_vec[1]) {
            static constexpr auto unit_stride = py::ssize_t(1);
            if (src_strides_vec[1] == unit_stride) {
                return py_as_f_contig_c2f(src, dst, exec_q, depends);
            }
        }
    }

    const std::size_t nelems = get_nelems(src_shape_vec);

    if (nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    // simplify batch iteration space
    // NB: simplification reverses dst strides to C contig,
    // it also reverses simplified_shape and simplified_src_strides

    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = src_nd;

    // nd, simplified_* and *_offset are modified by reference
    dpctl::tensor::py_internal::simplify_iteration_space(
        nd, src_shape_vec.data(), src_strides_vec, dst.get_strides_vector(),
        // output
        simplified_shape, simplified_src_strides, simplified_dst_strides,
        src_offset, dst_offset);

    if (!((0 == src_offset) && (0 == dst_offset))) {
        throw std::runtime_error(
            "Unexpected result of simplifying iteration space, 1");
    }

    std::vector<sycl::event> host_task_events{};
    auto ptr_size_event_tuple =
        dpctl::tensor::offset_utils::device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, simplified_shape, simplified_src_strides);
    auto shape_stride_owner = std::move(std::get<0>(ptr_size_event_tuple));
    const sycl::event &copy_shape_ev = std::get<2>(ptr_size_event_tuple);
    const py::ssize_t *shape_stride = shape_stride_owner.get();

    auto ascontig_fn = as_c_contig_array_dispatch_vector[src_type_id];

    std::vector<sycl::event> all_depends;
    all_depends.reserve(depends.size() + 1);
    all_depends.insert(std::end(all_depends), std::begin(depends),
                       std::end(depends));
    all_depends.push_back(copy_shape_ev);

    sycl::event ascontig_ev =
        ascontig_fn(exec_q, nelems, nd, shape_stride, src.get_data(),
                    dst.get_data(), all_depends);

    const auto &temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(exec_q, {ascontig_ev},
                                                     shape_stride_owner);
    host_task_events.push_back(temporaries_cleanup_ev);

    return std::make_pair(keep_args_alive(exec_q, {src, dst}, host_task_events),
                          ascontig_ev);
}

std::pair<sycl::event, sycl::event>
    py_as_c_contig_f2c(const dpctl::tensor::usm_ndarray &src,
                       const dpctl::tensor::usm_ndarray &dst,
                       sycl::queue &exec_q,
                       const std::vector<sycl::event> &depends)
{
    /*  Same dimensions, same shape, same data-type
     *  dst is C-contiguous.
     */
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    if (src_nd != dst_nd) {
        throw py::value_error("Number of dimensions must be the same.");
    }
    if (src_nd < 2) {
        throw py::value_error("Arrays must have 2 or more axes");
    }

    const auto &src_shape_vec = src.get_shape_vector();
    const auto &dst_shape_vec = dst.get_shape_vector();

    std::size_t nelems{1};
    bool equal_shapes = true;

    for (int i = 0; equal_shapes && (i < src_nd); ++i) {
        auto sh_i = src_shape_vec[i];
        equal_shapes = equal_shapes && (sh_i == dst_shape_vec[i]);
        nelems *= static_cast<std::size_t>(sh_i);
    }

    if (!equal_shapes) {
        throw py::value_error("Shapes must be equal");
    }

    const auto n = src_shape_vec.back();
    if (src_shape_vec[src_nd - 2] != n) {
        throw py::value_error("Matrices must be square");
    }

    const auto &src_strides_vec = src.get_strides_vector();

    if (src_strides_vec[src_nd - 2] != py::ssize_t(1)) {
        throw py::value_error("Unexpected destination array layout");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    const int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    const int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_type_id != dst_type_id) {
        throw py::value_error(
            "Source and destination arrays must have the same data type");
    }

    // ensures also that destination is plenty ample to accommodate all
    // elements of src array
    if (!dst.is_c_contiguous()) {
        throw py::value_error("Destination array must be C-contiguous");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    if (nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    const auto &dst_strides_vec = dst.get_strides_vector();

    const std::size_t batch_nelems =
        (src_nd == 2) ? std::size_t(1) : (nelems / (n * n));
    const py::ssize_t dst_batch_step =
        (src_nd == 2) ? py::ssize_t(0) : dst_strides_vec[src_nd - 3];

    std::vector<py::ssize_t> src_batch_strides_vec;
    std::vector<py::ssize_t> dst_batch_strides_vec;
    std::vector<py::ssize_t> batch_shape_vec;

    if (src_nd == 2) {
        batch_shape_vec.push_back(py::ssize_t(1));
        src_batch_strides_vec.push_back(py::ssize_t(0));
        dst_batch_strides_vec.push_back(dst_batch_step);
    }
    else {
        batch_shape_vec.insert(std::end(batch_shape_vec),
                               std::begin(src_shape_vec),
                               std::end(src_shape_vec) - 2);
        src_batch_strides_vec.insert(std::end(src_batch_strides_vec),
                                     std::begin(src_strides_vec),
                                     std::end(src_strides_vec) - 2);
        dst_batch_strides_vec.insert(std::end(dst_batch_strides_vec),
                                     std::begin(dst_strides_vec),
                                     std::end(dst_strides_vec) - 2);
    }

    // simplify batch iteration space
    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = static_cast<int>(batch_shape_vec.size());

    // nd, simplified_* and *_offset are modified by reference
    dpctl::tensor::py_internal::simplify_iteration_space(
        nd, batch_shape_vec.data(), src_batch_strides_vec,
        dst_batch_strides_vec,
        // output
        simplified_shape, simplified_src_strides, simplified_dst_strides,
        src_offset, dst_offset);

    if (!((0 == src_offset) && (0 == dst_offset))) {
        throw std::runtime_error(
            "Unexpected result of simplifying iteration space, 1");
    }

    if (1 == nd) {
        const auto expected_dim = static_cast<py::ssize_t>(batch_nelems);
        if ((simplified_shape.front() != expected_dim) ||
            (simplified_dst_strides.front() != dst_batch_step))
        {
            throw std::runtime_error(
                "Unexpected result of simplifying iteration space, 2");
        }

        auto impl_fn = as_c_contig_1d_batch_of_square_matrices_dispatch_vector
            [src_type_id];
        const py::ssize_t src_batch_step = simplified_src_strides.front();

        sycl::event ascontig_ev =
            impl_fn(exec_q, batch_nelems, src_batch_step, dst_batch_step, n,
                    src.get_data(), src_strides_vec.back(), dst.get_data(),
                    dst_strides_vec[src_nd - 2], depends);

        return std::make_pair(
            keep_args_alive(exec_q, {src, dst}, {ascontig_ev}), ascontig_ev);
    }

    auto impl_fn =
        as_c_contig_nd_batch_of_square_matrices_dispatch_vector[src_type_id];

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, simplified_shape, simplified_src_strides);
    auto packed_shape_strides_owner =
        std::move(std::get<0>(ptr_size_event_tuple));
    const sycl::event &copy_shape_ev = std::get<2>(ptr_size_event_tuple);
    const py::ssize_t *packed_shape_strides = packed_shape_strides_owner.get();

    std::vector<sycl::event> all_depends;
    all_depends.reserve(depends.size() + 1);
    all_depends.insert(std::end(all_depends), std::begin(depends),
                       std::end(depends));
    all_depends.push_back(copy_shape_ev);

    sycl::event ascontig_ev =
        impl_fn(exec_q, batch_nelems, nd, packed_shape_strides, dst_batch_step,
                n, src.get_data(), src_strides_vec.back(), dst.get_data(),
                dst_strides_vec[src_nd - 2], all_depends);

    // async free of shape_strides temporary
    sycl::event temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {ascontig_ev}, packed_shape_strides_owner);
    host_task_events.push_back(temporaries_cleanup_ev);

    return std::make_pair(keep_args_alive(exec_q, {src, dst}, host_task_events),
                          ascontig_ev);
}

std::pair<sycl::event, sycl::event>
    py_as_f_contig_c2f(const dpctl::tensor::usm_ndarray &src,
                       const dpctl::tensor::usm_ndarray &dst,
                       sycl::queue &exec_q,
                       const std::vector<sycl::event> &depends)
{
    /*  Same dimensions, same shape, same data-type
     *  dst is F-contiguous.
     */
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();

    if (src_nd != dst_nd) {
        throw py::value_error("Number of dimensions must be the same.");
    }
    if (src_nd < 2) {
        throw py::value_error("Arrays must have 2 or more axes");
    }

    // ensures also that destination is plenty ample to accommodate all
    // elements of src array
    if (!dst.is_f_contiguous()) {
        throw py::value_error("Destination array must be C-contiguous");
    }

    const auto &src_shape_vec = src.get_shape_vector();
    const auto &dst_shape_vec = dst.get_shape_vector();

    std::size_t nelems{1};
    bool equal_shapes = true;

    for (int i = 0; equal_shapes && (i < src_nd); ++i) {
        auto sh_i = src_shape_vec[i];
        equal_shapes = equal_shapes && (sh_i == dst_shape_vec[i]);
        nelems *= static_cast<std::size_t>(sh_i);
    }

    if (!equal_shapes) {
        throw py::value_error("Shapes must be equal");
    }

    const auto n = dst_shape_vec.front();
    if (dst_shape_vec[1] != n) {
        throw py::value_error("Matrices must be square");
    }

    const auto &src_strides_vec = src.get_strides_vector();

    if (src_strides_vec[1] != py::ssize_t(1)) {
        throw py::value_error("Unexpected destination array layout");
    }

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto array_types = td_ns::usm_ndarray_types();
    const int src_type_id = array_types.typenum_to_lookup_id(src_typenum);
    const int dst_type_id = array_types.typenum_to_lookup_id(dst_typenum);

    if (src_type_id != dst_type_id) {
        throw py::value_error(
            "Source and destination arrays must have the same data type");
    }

    if (nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    const auto &dst_strides_vec = dst.get_strides_vector();

    const std::size_t batch_nelems =
        (src_nd == 2) ? std::size_t(1) : (nelems / (n * n));
    const py::ssize_t dst_batch_step =
        (src_nd == 2) ? py::ssize_t(0) : dst_strides_vec[2];

    std::vector<py::ssize_t> src_batch_strides_vec;
    std::vector<py::ssize_t> dst_batch_strides_vec;
    std::vector<py::ssize_t> batch_shape_vec;

    if (src_nd == 2) {
        batch_shape_vec.push_back(py::ssize_t(1));
        src_batch_strides_vec.push_back(py::ssize_t(0));
        dst_batch_strides_vec.push_back(dst_batch_step);
    }
    else {
        batch_shape_vec.insert(std::end(batch_shape_vec),
                               std::begin(src_shape_vec) + 2,
                               std::end(src_shape_vec));
        src_batch_strides_vec.insert(std::end(src_batch_strides_vec),
                                     std::begin(src_strides_vec) + 2,
                                     std::end(src_strides_vec));
        dst_batch_strides_vec.insert(std::end(dst_batch_strides_vec),
                                     std::begin(dst_strides_vec) + 2,
                                     std::end(dst_strides_vec));
    }

    // simplify batch iteration space
    // NB: simplification reverses dst strides to C contig,
    // it also reverses simplified_shape and simplified_src_strides
    using shT = std::vector<py::ssize_t>;
    shT simplified_shape;
    shT simplified_src_strides;
    shT simplified_dst_strides;
    py::ssize_t src_offset(0);
    py::ssize_t dst_offset(0);

    int nd = static_cast<int>(batch_shape_vec.size());

    // nd, simplified_* and *_offset are modified by reference
    dpctl::tensor::py_internal::simplify_iteration_space(
        nd, batch_shape_vec.data(), src_batch_strides_vec,
        dst_batch_strides_vec,
        // output
        simplified_shape, simplified_src_strides, simplified_dst_strides,
        src_offset, dst_offset);

    if (!((0 == src_offset) && (0 == dst_offset))) {
        throw std::runtime_error(
            "Unexpected result of simplifying iteration space, 1");
    }

    if (1 == nd) {
        const auto expected_dim = static_cast<py::ssize_t>(batch_nelems);
        if ((simplified_shape.front() != expected_dim) ||
            (simplified_dst_strides.front() != dst_batch_step))
        {
            throw std::runtime_error(
                "Unexpected result of simplifying iteration space, 2");
        }

        auto impl_fn = as_c_contig_1d_batch_of_square_matrices_dispatch_vector
            [src_type_id];
        const py::ssize_t src_batch_step = simplified_src_strides.front();

        sycl::event ascontig_ev =
            impl_fn(exec_q, batch_nelems, src_batch_step, dst_batch_step, n,
                    src.get_data(), src_strides_vec.front(), dst.get_data(),
                    dst_strides_vec[1], depends);

        return std::make_pair(
            keep_args_alive(exec_q, {src, dst}, {ascontig_ev}), ascontig_ev);
    }

    auto impl_fn =
        as_c_contig_nd_batch_of_square_matrices_dispatch_vector[src_type_id];

    std::vector<sycl::event> host_task_events;
    host_task_events.reserve(2);

    using dpctl::tensor::offset_utils::device_allocate_and_pack;
    auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
        exec_q, host_task_events, simplified_shape, simplified_src_strides);
    auto packed_shape_strides_owner =
        std::move(std::get<0>(ptr_size_event_tuple));
    const sycl::event &copy_shape_ev = std::get<2>(ptr_size_event_tuple);
    const py::ssize_t *packed_shape_strides = packed_shape_strides_owner.get();

    std::vector<sycl::event> all_depends;
    all_depends.reserve(depends.size() + 1);
    all_depends.insert(std::end(all_depends), std::begin(depends),
                       std::end(depends));
    all_depends.push_back(copy_shape_ev);

    sycl::event ascontig_ev =
        impl_fn(exec_q, batch_nelems, nd, packed_shape_strides, dst_batch_step,
                n, src.get_data(), src_strides_vec.front(), dst.get_data(),
                dst_strides_vec[1], all_depends);

    // async free of shape_strides
    sycl::event temporaries_cleanup_ev =
        dpctl::tensor::alloc_utils::async_smart_free(
            exec_q, {ascontig_ev}, packed_shape_strides_owner);
    host_task_events.push_back(temporaries_cleanup_ev);

    return std::make_pair(keep_args_alive(exec_q, {src, dst}, host_task_events),
                          ascontig_ev);
}

} // end of namespace py_internal
} // end of namespace tensor
} // end of namespace dpctl
