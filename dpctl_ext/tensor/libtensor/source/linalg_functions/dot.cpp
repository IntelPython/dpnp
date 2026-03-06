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
/// This file defines functions of dpctl.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "dpnp4pybind11.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dot.hpp"
#include "dot_atomic_support.hpp"
#include "dot_dispatch.hpp"
#include "elementwise_functions/elementwise_functions_type_utils.hpp"
#include "kernels/linalg_functions/dot_product.hpp"
#include "kernels/linalg_functions/gemm.hpp"
#include "reductions/reduction_atomic_support.hpp"
#include "simplify_iteration_space.hpp"
#include "utils/memory_overlap.hpp"
#include "utils/offset_utils.hpp"
#include "utils/output_validation.hpp"
#include "utils/sycl_alloc_utils.hpp"

namespace dpctl::tensor::py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

static int dot_output_id_table[td_ns::num_types][td_ns::num_types];

using dpctl::tensor::kernels::dot_product_impl_fn_ptr_t;
static dot_product_impl_fn_ptr_t dot_product_dispatch_table[td_ns::num_types]
                                                           [td_ns::num_types];

static dot_product_impl_fn_ptr_t
    dot_product_temps_dispatch_table[td_ns::num_types][td_ns::num_types];

using dpctl::tensor::kernels::dot_product_contig_impl_fn_ptr_t;
static dot_product_contig_impl_fn_ptr_t
    dot_product_contig_dispatch_table[td_ns::num_types][td_ns::num_types];

static dot_product_contig_impl_fn_ptr_t
    dot_product_contig_temps_dispatch_table[td_ns::num_types][td_ns::num_types];

using dpctl::tensor::kernels::gemm_impl_fn_ptr_t;
static gemm_impl_fn_ptr_t gemm_atomic_dispatch_table[td_ns::num_types]
                                                    [td_ns::num_types];

static gemm_impl_fn_ptr_t gemm_temps_dispatch_table[td_ns::num_types]
                                                   [td_ns::num_types];

using dpctl::tensor::kernels::gemm_contig_impl_fn_ptr_t;
static gemm_contig_impl_fn_ptr_t
    gemm_contig_atomic_dispatch_table[td_ns::num_types][td_ns::num_types];

static gemm_contig_impl_fn_ptr_t
    gemm_contig_temps_dispatch_table[td_ns::num_types][td_ns::num_types];

using dpctl::tensor::kernels::gemm_batch_impl_fn_ptr_t;
static gemm_batch_impl_fn_ptr_t
    gemm_batch_atomic_dispatch_table[td_ns::num_types][td_ns::num_types];

static gemm_batch_impl_fn_ptr_t
    gemm_batch_temps_dispatch_table[td_ns::num_types][td_ns::num_types];

using dpctl::tensor::kernels::gemm_batch_contig_impl_fn_ptr_t;
static gemm_batch_contig_impl_fn_ptr_t
    gemm_batch_contig_atomic_dispatch_table[td_ns::num_types][td_ns::num_types];

static gemm_batch_contig_impl_fn_ptr_t
    gemm_batch_contig_temps_dispatch_table[td_ns::num_types][td_ns::num_types];

void init_dot_dispatch_tables(void)
{
    td_ns::DispatchTableBuilder<int, DotTypeMapFactory, td_ns::num_types> dtb1;
    dtb1.populate_dispatch_table(dot_output_id_table);

    td_ns::DispatchTableBuilder<gemm_batch_impl_fn_ptr_t,
                                GemmBatchAtomicFactory, td_ns::num_types>
        dtb2;
    dtb2.populate_dispatch_table(gemm_batch_atomic_dispatch_table);

    td_ns::DispatchTableBuilder<gemm_batch_contig_impl_fn_ptr_t,
                                GemmBatchContigAtomicFactory, td_ns::num_types>
        dtb3;
    dtb3.populate_dispatch_table(gemm_batch_contig_atomic_dispatch_table);

    td_ns::DispatchTableBuilder<gemm_impl_fn_ptr_t, GemmAtomicFactory,
                                td_ns::num_types>
        dtb4;
    dtb4.populate_dispatch_table(gemm_atomic_dispatch_table);

    td_ns::DispatchTableBuilder<gemm_contig_impl_fn_ptr_t,
                                GemmContigAtomicFactory, td_ns::num_types>
        dtb5;
    dtb5.populate_dispatch_table(gemm_contig_atomic_dispatch_table);

    td_ns::DispatchTableBuilder<gemm_batch_impl_fn_ptr_t, GemmBatchTempsFactory,
                                td_ns::num_types>
        dtb6;
    dtb6.populate_dispatch_table(gemm_batch_temps_dispatch_table);

    td_ns::DispatchTableBuilder<gemm_batch_contig_impl_fn_ptr_t,
                                GemmBatchContigTempsFactory, td_ns::num_types>
        dtb7;
    dtb7.populate_dispatch_table(gemm_batch_contig_temps_dispatch_table);

    td_ns::DispatchTableBuilder<gemm_impl_fn_ptr_t, GemmTempsFactory,
                                td_ns::num_types>
        dtb8;
    dtb8.populate_dispatch_table(gemm_temps_dispatch_table);

    td_ns::DispatchTableBuilder<gemm_contig_impl_fn_ptr_t,
                                GemmContigTempsFactory, td_ns::num_types>
        dtb9;
    dtb9.populate_dispatch_table(gemm_contig_temps_dispatch_table);

    td_ns::DispatchTableBuilder<dot_product_impl_fn_ptr_t,
                                DotProductAtomicFactory, td_ns::num_types>
        dtb10;
    dtb10.populate_dispatch_table(dot_product_dispatch_table);

    td_ns::DispatchTableBuilder<dot_product_impl_fn_ptr_t,
                                DotProductNoAtomicFactory, td_ns::num_types>
        dtb11;
    dtb11.populate_dispatch_table(dot_product_temps_dispatch_table);

    td_ns::DispatchTableBuilder<dot_product_contig_impl_fn_ptr_t,
                                DotProductContigAtomicFactory, td_ns::num_types>
        dtb12;
    dtb12.populate_dispatch_table(dot_product_contig_dispatch_table);

    td_ns::DispatchTableBuilder<dot_product_contig_impl_fn_ptr_t,
                                DotProductContigNoAtomicFactory,
                                td_ns::num_types>
        dtb13;
    dtb13.populate_dispatch_table(dot_product_contig_temps_dispatch_table);
}

using atomic_support::atomic_support_fn_ptr_t;
static atomic_support_fn_ptr_t dot_atomic_support_vector[td_ns::num_types];

void init_dot_atomic_support_vector(void)
{

    using atomic_support::DotAtomicSupportFactory;
    td_ns::DispatchVectorBuilder<atomic_support_fn_ptr_t,
                                 DotAtomicSupportFactory, td_ns::num_types>
        dvb;
    dvb.populate_dispatch_vector(dot_atomic_support_vector);
}

std::pair<sycl::event, sycl::event>
    py_dot(const dpctl::tensor::usm_ndarray &x1,
           const dpctl::tensor::usm_ndarray &x2,
           int batch_dims,
           int x1_outer_dims,
           int x2_outer_dims,
           int inner_dims,
           const dpctl::tensor::usm_ndarray &dst,
           sycl::queue &exec_q,
           const std::vector<sycl::event> &depends)
{
    if (!dpctl::utils::queues_are_compatible(exec_q, {x1, x2, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    if (inner_dims == 0) {
        throw py::value_error("No inner dimension for dot");
    }

    int x1_nd = x1.get_ndim();
    int x2_nd = x2.get_ndim();
    if (x1_nd != (batch_dims + x1_outer_dims + inner_dims) ||
        x2_nd != (batch_dims + x2_outer_dims + inner_dims))
    {
        throw py::value_error("Input arrays do not have dimensions consistent "
                              "with input dimensions");
    }

    int dst_nd = dst.get_ndim();
    if (dst_nd != (batch_dims + x1_outer_dims + x2_outer_dims)) {
        throw py::value_error("Destination array rank does not match input "
                              "array rank and number of input dimensions");
    }

    const py::ssize_t *x1_shape_ptr = x1.get_shape_raw();
    const py::ssize_t *x2_shape_ptr = x2.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    std::size_t batches(1);
    for (int i = 0; same_shapes && (i < batch_dims); ++i) {
        same_shapes = same_shapes && (x1_shape_ptr[i] == dst_shape_ptr[i]) &&
                      (x2_shape_ptr[i] == dst_shape_ptr[i]);
        batches *= x1_shape_ptr[i];
    }
    std::size_t x1_outer_nelems(1);
    for (int i = batch_dims; same_shapes && (i < (batch_dims + x1_outer_dims));
         ++i) {
        same_shapes = same_shapes && (x1_shape_ptr[i] == dst_shape_ptr[i]);
        x1_outer_nelems *= x1_shape_ptr[i];
    }
    std::size_t inner_nelems(1);
    for (int i = batch_dims; i < (batch_dims + inner_dims); ++i) {
        auto x1_shape_idx = x1_outer_dims + i;
        same_shapes =
            same_shapes && (x1_shape_ptr[x1_shape_idx] == x2_shape_ptr[i]);
        inner_nelems *= x1_shape_ptr[x1_shape_idx];
    }
    std::size_t x2_outer_nelems(1);
    for (int i = 0; same_shapes && (i < x2_outer_dims); ++i) {
        auto x2_shape_idx = batch_dims + inner_dims + i;
        same_shapes =
            same_shapes && (x2_shape_ptr[x2_shape_idx] ==
                            dst_shape_ptr[batch_dims + x1_outer_dims + i]);
        x2_outer_nelems *= x2_shape_ptr[x2_shape_idx];
    }
    if (!same_shapes) {
        throw py::value_error("Input arrays to tensor dot product do not have "
                              "appropriate shapes");
    }

    std::size_t dst_nelems = batches * x1_outer_nelems * x2_outer_nelems;
    if (dst_nelems == 0) {
        return std::make_pair(sycl::event(), sycl::event());
    }

    if (static_cast<std::size_t>(dst.get_size()) != dst_nelems) {
        throw py::value_error("dst shape and size mismatch");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(dst, dst_nelems);

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    // check that dst does not intersect with x1 or x2
    if (overlap(dst, x1) || overlap(dst, x2)) {
        throw py::value_error("Result array overlaps with inputs");
    }

    int x1_typenum = x1.get_typenum();
    int x2_typenum = x2.get_typenum();
    int dst_typenum = dst.get_typenum();

    auto const &array_types = td_ns::usm_ndarray_types();
    int x1_typeid = array_types.typenum_to_lookup_id(x1_typenum);
    int x2_typeid = array_types.typenum_to_lookup_id(x2_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    int output_typeid = dot_output_id_table[x1_typeid][x2_typeid];

    if (output_typeid != dst_typeid) {
        throw py::value_error(
            "Result array has unexpected elemental data type.");
    }

    void *data_ptr = dst.get_data();
    const auto &ctx = exec_q.get_context();
    auto usm_type = sycl::get_pointer_type(data_ptr, ctx);
    bool supports_atomics =
        dot_atomic_support_vector[output_typeid](exec_q, usm_type);

    const char *x1_data = x1.get_data();
    const char *x2_data = x2.get_data();
    char *dst_data = dst.get_data();

    const auto &x1_shape_vec = x1.get_shape_vector();
    const auto &x1_strides_vec = x1.get_strides_vector();

    const auto &x2_shape_vec = x2.get_shape_vector();
    const auto &x2_strides_vec = x2.get_strides_vector();

    const auto &dst_shape_vec = dst.get_shape_vector();
    const auto &dst_strides_vec = dst.get_strides_vector();

    bool is_x1_c_contig = x1.is_c_contiguous();
    bool is_x1_f_contig = x1.is_f_contiguous();
    bool is_x2_c_contig = x2.is_c_contiguous();
    bool is_x2_f_contig = x2.is_f_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    bool call_vecdot = ((x1_outer_dims == 0 && x1_outer_nelems == 1) &&
                        (x2_outer_dims == 0 && x2_outer_nelems == 1));

    bool call_batched = (batch_dims != 0 || batches > 1);
    std::vector<sycl::event> host_task_events{};
    sycl::event dot_ev;
    if (call_vecdot) {
        if ((is_x1_c_contig && is_x2_c_contig && is_dst_c_contig) ||
            ((is_x1_f_contig && is_x2_f_contig) && !call_batched))
        {
            dot_product_contig_impl_fn_ptr_t fn = nullptr;
            if (supports_atomics) {
                fn = dot_product_contig_dispatch_table[x1_typeid][x2_typeid];
            }
            else {
                fn = dot_product_contig_temps_dispatch_table[x1_typeid]
                                                            [x2_typeid];
            }
            if (fn != nullptr) {
                static constexpr py::ssize_t zero_offset = 0;
                dot_ev = fn(exec_q, dst_nelems, inner_nelems, x1.get_data(),
                            x2.get_data(), dst.get_data(),
                            zero_offset, // lhs batch offset
                            zero_offset, // rhs batch offset
                            zero_offset, // res batch offset
                            zero_offset, // lhs reduction offset
                            zero_offset, // rhs reduction offset
                            depends);
                return std::make_pair(dpctl::utils::keep_args_alive(
                                          exec_q, {x1, x2, dst}, {dot_ev}),
                                      dot_ev);
            }
        }
        int inner_nd = inner_dims;
        const py::ssize_t *inner_shape_ptr = x1_shape_ptr + batch_dims;
        using shT = std::vector<py::ssize_t>;
        const shT inner_x1_strides(std::begin(x1_strides_vec) + batch_dims,
                                   std::end(x1_strides_vec));
        const shT inner_x2_strides(std::begin(x2_strides_vec) + batch_dims,
                                   std::end(x2_strides_vec));

        shT simplified_inner_shape;
        shT simplified_inner_x1_strides;
        shT simplified_inner_x2_strides;
        py::ssize_t inner_x1_offset(0);
        py::ssize_t inner_x2_offset(0);

        simplify_iteration_space(
            inner_nd, inner_shape_ptr, inner_x1_strides, inner_x2_strides,
            // output
            simplified_inner_shape, simplified_inner_x1_strides,
            simplified_inner_x2_strides, inner_x1_offset, inner_x2_offset);

        const py::ssize_t *batch_shape_ptr = x1_shape_ptr;

        const shT batch_x1_strides(std::begin(x1_strides_vec),
                                   std::begin(x1_strides_vec) + batch_dims);
        const shT batch_x2_strides(std::begin(x2_strides_vec),
                                   std::begin(x2_strides_vec) + batch_dims);
        shT const &batch_dst_strides = dst_strides_vec;

        shT simplified_batch_shape;
        shT simplified_batch_x1_strides;
        shT simplified_batch_x2_strides;
        shT simplified_batch_dst_strides;
        py::ssize_t batch_x1_offset(0);
        py::ssize_t batch_x2_offset(0);
        py::ssize_t batch_dst_offset(0);

        if (batch_dims == 0) {
            if (dst_nelems != 1) {
                throw std::runtime_error(
                    "batch_dims == 0, but dst_nelems != 1");
            }
            batch_dims = 1;
            simplified_batch_shape.push_back(1);
            simplified_batch_x1_strides.push_back(0);
            simplified_batch_x2_strides.push_back(0);
            simplified_batch_dst_strides.push_back(0);
        }
        else {
            simplify_iteration_space_3(
                batch_dims, batch_shape_ptr, batch_x1_strides, batch_x2_strides,
                batch_dst_strides,
                // output
                simplified_batch_shape, simplified_batch_x1_strides,
                simplified_batch_x2_strides, simplified_batch_dst_strides,
                batch_x1_offset, batch_x2_offset, batch_dst_offset);
        }

        if (inner_nd == 1 && batch_dims == 1) {
            bool dot_product_c_contig = false;
            bool reduce_all_elems = false;

            if (simplified_inner_x1_strides[0] == 1 &&
                simplified_inner_x2_strides[0] == 1) {
                reduce_all_elems = (simplified_batch_shape[0] == 1);
                dot_product_c_contig =
                    (simplified_batch_dst_strides[0] == 1) &&
                    (static_cast<std::size_t>(simplified_batch_x1_strides[0]) ==
                     inner_nelems) &&
                    (static_cast<std::size_t>(simplified_batch_x2_strides[0]) ==
                     inner_nelems);
            }

            if (dot_product_c_contig || reduce_all_elems) {
                dot_product_contig_impl_fn_ptr_t fn = nullptr;
                if (supports_atomics) {
                    fn =
                        dot_product_contig_dispatch_table[x1_typeid][x2_typeid];
                }
                else {
                    fn = dot_product_contig_temps_dispatch_table[x1_typeid]
                                                                [x2_typeid];
                }
                if (fn != nullptr) {
                    dot_ev = fn(exec_q, dst_nelems, inner_nelems, x1.get_data(),
                                x2.get_data(), dst.get_data(),
                                batch_x1_offset,  // lhs batch offset
                                batch_x2_offset,  // rhs batch offset
                                batch_dst_offset, // res batch offset
                                inner_x1_offset,  // lhs reduction offset
                                inner_x2_offset,  // rhs reduction offset
                                depends);
                    return std::make_pair(dpctl::utils::keep_args_alive(
                                              exec_q, {x1, x2, dst}, {dot_ev}),
                                          dot_ev);
                }
            }
        }

        dot_product_impl_fn_ptr_t fn = nullptr;
        if (supports_atomics) {
            fn = dot_product_dispatch_table[x1_typeid][x2_typeid];
        }
        if (fn == nullptr) {
            fn = dot_product_temps_dispatch_table[x1_typeid][x2_typeid];
            if (fn == nullptr) {
                throw std::runtime_error(
                    "Implementation is missing for x1_typeid=" +
                    std::to_string(x1_typeid) +
                    " and x2_typeid=" + std::to_string(x2_typeid));
            }
        }

        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        auto arrays_metainfo_packing_triple_ =
            device_allocate_and_pack<py::ssize_t>(
                exec_q, host_task_events,
                // iteration metadata
                simplified_batch_shape, simplified_batch_x1_strides,
                simplified_batch_x2_strides, simplified_batch_dst_strides,
                // reduction metadata
                simplified_inner_shape, simplified_inner_x1_strides,
                simplified_inner_x2_strides);
        auto tmp_alloc_owner =
            std::move(std::get<0>(arrays_metainfo_packing_triple_));
        const auto &copy_metadata_ev =
            std::get<2>(arrays_metainfo_packing_triple_);
        const py::ssize_t *temp_allocation_ptr = tmp_alloc_owner.get();

        const py::ssize_t *iter_shape_and_strides = temp_allocation_ptr;
        const py::ssize_t *inner_shape_stride =
            temp_allocation_ptr + 4 * simplified_batch_shape.size();

        std::vector<sycl::event> all_deps;
        all_deps.reserve(depends.size() + 1);
        all_deps.resize(depends.size());
        std::copy(depends.begin(), depends.end(), all_deps.begin());
        all_deps.push_back(copy_metadata_ev);

        dot_ev =
            fn(exec_q, dst_nelems, inner_nelems, x1.get_data(), x2.get_data(),
               dst.get_data(), batch_dims, iter_shape_and_strides,
               batch_x1_offset, batch_x2_offset, batch_dst_offset,
               inner_nd, // number dimensions being reduced
               inner_shape_stride, inner_x1_offset, inner_x2_offset, all_deps);

        sycl::event temp_cleanup_ev =
            dpctl::tensor::alloc_utils::async_smart_free(exec_q, {dot_ev},
                                                         tmp_alloc_owner);
        host_task_events.push_back(temp_cleanup_ev);
    }
    else { // if (!call_vecdot)
        if (!call_batched) {
            if ((is_x1_c_contig && is_x2_c_contig && is_dst_c_contig)) {
                gemm_contig_impl_fn_ptr_t fn = nullptr;
                if (supports_atomics) {
                    fn =
                        gemm_contig_atomic_dispatch_table[x1_typeid][x2_typeid];
                }
                else {
                    fn = gemm_contig_temps_dispatch_table[x1_typeid][x2_typeid];
                }
                if (fn != nullptr) {
                    dot_ev = fn(exec_q, x1_data, x2_data, dst_data,
                                x1_outer_nelems, // n
                                inner_nelems,    // k
                                x2_outer_nelems, // m
                                depends);
                    return std::make_pair(dpctl::utils::keep_args_alive(
                                              exec_q, {x1, x2, dst}, {dot_ev}),
                                          dot_ev);
                }
            }
            gemm_impl_fn_ptr_t fn = nullptr;
            if (supports_atomics) {
                fn = gemm_atomic_dispatch_table[x1_typeid][x2_typeid];
            }
            if (fn == nullptr) {
                fn = gemm_temps_dispatch_table[x1_typeid][x2_typeid];
                if (fn == nullptr) {
                    throw std::runtime_error(
                        "Implementation is missing for x1_typeid=" +
                        std::to_string(x1_typeid) +
                        " and x2_typeid=" + std::to_string(x2_typeid));
                }
            }
            using dpctl::tensor::offset_utils::device_allocate_and_pack;
            auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
                exec_q, host_task_events, x1_shape_vec, x1_strides_vec,
                x2_shape_vec, x2_strides_vec, dst_shape_vec, dst_strides_vec);
            auto packed_shapes_strides_owner =
                std::move(std::get<0>(ptr_size_event_tuple1));
            sycl::event copy_shapes_strides_ev =
                std::get<2>(ptr_size_event_tuple1);
            const py::ssize_t *packed_shapes_strides =
                packed_shapes_strides_owner.get();

            const py::ssize_t *x1_shape_strides = packed_shapes_strides;
            const py::ssize_t *x2_shape_strides =
                packed_shapes_strides + 2 * (x1_nd);
            const py::ssize_t *dst_shape_strides =
                packed_shapes_strides + 2 * (x1_nd + x2_nd);

            std::vector<sycl::event> all_deps;
            all_deps.reserve(depends.size() + 1);
            all_deps.insert(all_deps.end(), depends.begin(), depends.end());
            all_deps.push_back(copy_shapes_strides_ev);

            // change gemm calls to pass inner dims and outer dims separately
            dot_ev =
                fn(exec_q, x1_data, x2_data, dst_data, x1_outer_nelems,
                   inner_nelems, x2_outer_nelems, inner_dims, x1_outer_dims,
                   x1_shape_strides, x2_outer_dims, x2_shape_strides,
                   x1_outer_dims + x2_outer_dims, dst_shape_strides, all_deps);

            sycl::event cleanup_tmp_allocations_ev =
                dpctl::tensor::alloc_utils::async_smart_free(
                    exec_q, {dot_ev}, packed_shapes_strides_owner);
            host_task_events.push_back(cleanup_tmp_allocations_ev);
        }
        else { // if (call_batched)
            using shT = std::vector<py::ssize_t>;
            // temporary asserts for matmul
            assert(x1_outer_dims == 1);
            assert(x2_outer_dims == 1);
            assert(inner_dims == 1);

            if ((is_x1_c_contig && is_x2_c_contig && is_dst_c_contig)) {
                gemm_batch_contig_impl_fn_ptr_t fn = nullptr;
                if (supports_atomics) {
                    fn = gemm_batch_contig_atomic_dispatch_table[x1_typeid]
                                                                [x2_typeid];
                }
                else {
                    fn = gemm_batch_contig_temps_dispatch_table[x1_typeid]
                                                               [x2_typeid];
                }
                if (fn != nullptr) {
                    static constexpr py::ssize_t zero_offset = 0;
                    dot_ev = fn(exec_q, x1_data, x2_data, dst_data, batches,
                                x1_outer_nelems, // n
                                inner_nelems,    // k
                                x2_outer_nelems, // m
                                zero_offset, zero_offset, zero_offset, depends);
                    return std::make_pair(dpctl::utils::keep_args_alive(
                                              exec_q, {x1, x2, dst}, {dot_ev}),
                                          dot_ev);
                }
            }

            auto x1_outer_inner_dims = x1_nd - batch_dims;
            auto x2_outer_inner_dims = x2_nd - batch_dims;
            auto dst_outer_inner_dims = dst_nd - batch_dims;

            shT batch_x1_shape;
            shT outer_inner_x1_shape;
            shT batch_x1_strides;
            shT outer_inner_x1_strides;
            split_iteration_space(x1_shape_vec, x1_strides_vec, batch_dims,
                                  batch_dims + x1_outer_inner_dims,
                                  // 4 vectors modified
                                  batch_x1_shape, outer_inner_x1_shape,
                                  batch_x1_strides, outer_inner_x1_strides);

            shT batch_x2_shape;
            shT outer_inner_x2_shape;
            shT batch_x2_strides;
            shT outer_inner_x2_strides;
            split_iteration_space(x2_shape_vec, x2_strides_vec, batch_dims,
                                  batch_dims + x2_outer_inner_dims,
                                  // 4 vectors modified
                                  batch_x2_shape, outer_inner_x2_shape,
                                  batch_x2_strides, outer_inner_x2_strides);

            shT batch_dst_shape;
            shT outer_inner_dst_shape;
            shT batch_dst_strides;
            shT outer_inner_dst_strides;
            split_iteration_space(dst_shape_vec, dst_strides_vec, batch_dims,
                                  batch_dims + dst_outer_inner_dims,
                                  // 4 vectors modified
                                  batch_dst_shape, outer_inner_dst_shape,
                                  batch_dst_strides, outer_inner_dst_strides);

            using shT = std::vector<py::ssize_t>;
            shT simplified_batch_shape;
            shT simplified_batch_x1_strides;
            shT simplified_batch_x2_strides;
            shT simplified_batch_dst_strides;
            py::ssize_t x1_batch_offset(0);
            py::ssize_t x2_batch_offset(0);
            py::ssize_t dst_batch_offset(0);

            const py::ssize_t *shape = x1_shape_ptr;

            simplify_iteration_space_3(
                batch_dims, shape, batch_x1_strides, batch_x2_strides,
                batch_dst_strides,
                // outputs
                simplified_batch_shape, simplified_batch_x1_strides,
                simplified_batch_x2_strides, simplified_batch_dst_strides,
                x1_batch_offset, x2_batch_offset, dst_batch_offset);

            if (batch_dims == 1 && x1_outer_dims == 1 && x2_outer_dims == 1 &&
                inner_dims == 1)
            {
                bool gemm_batch_c_contig = false;

                if ((static_cast<std::size_t>(outer_inner_x1_strides[0]) ==
                         inner_nelems &&
                     outer_inner_x1_strides[1] == 1) &&
                    (static_cast<std::size_t>(outer_inner_x2_strides[0]) ==
                         inner_nelems &&
                     outer_inner_x2_strides[1] == 1) &&
                    (static_cast<std::size_t>(outer_inner_dst_strides[0]) ==
                         x2_outer_nelems &&
                     outer_inner_dst_strides[1] == 1))
                {
                    gemm_batch_c_contig =
                        (static_cast<std::size_t>(
                             simplified_batch_x1_strides[0]) ==
                         x1_outer_nelems * inner_nelems) &&
                        (static_cast<std::size_t>(
                             simplified_batch_x2_strides[0]) ==
                         x2_outer_nelems * inner_nelems) &&
                        (static_cast<std::size_t>(
                             simplified_batch_dst_strides[0]) ==
                         x1_outer_nelems * x2_outer_nelems);
                }

                if (gemm_batch_c_contig) {
                    gemm_batch_contig_impl_fn_ptr_t fn = nullptr;
                    if (supports_atomics) {
                        fn = gemm_batch_contig_atomic_dispatch_table[x1_typeid]
                                                                    [x2_typeid];
                    }
                    else {
                        fn = gemm_batch_contig_temps_dispatch_table[x1_typeid]
                                                                   [x2_typeid];
                    }
                    if (fn != nullptr) {
                        dot_ev = fn(exec_q, x1_data, x2_data, dst_data, batches,
                                    x1_outer_nelems, // n
                                    inner_nelems,    // k
                                    x2_outer_nelems, // m
                                    x1_batch_offset, x2_batch_offset,
                                    dst_batch_offset, depends);
                        return std::make_pair(
                            dpctl::utils::keep_args_alive(exec_q, {x1, x2, dst},
                                                          {dot_ev}),
                            dot_ev);
                    }
                }
            }

            gemm_batch_impl_fn_ptr_t fn = nullptr;
            if (supports_atomics) {
                fn = gemm_batch_atomic_dispatch_table[x1_typeid][x2_typeid];
            }
            if (fn == nullptr) {
                fn = gemm_batch_temps_dispatch_table[x1_typeid][x2_typeid];
                if (fn == nullptr) {
                    throw std::runtime_error(
                        "Implementation is missing for x1_typeid=" +
                        std::to_string(x1_typeid) +
                        " and x2_typeid=" + std::to_string(x2_typeid));
                }
            }
            using dpctl::tensor::offset_utils::device_allocate_and_pack;
            auto ptr_size_event_tuple1 = device_allocate_and_pack<py::ssize_t>(
                exec_q, host_task_events, simplified_batch_shape,
                simplified_batch_x1_strides, simplified_batch_x2_strides,
                simplified_batch_dst_strides, outer_inner_x1_shape,
                outer_inner_x1_strides, outer_inner_x2_shape,
                outer_inner_x2_strides, outer_inner_dst_shape,
                outer_inner_dst_strides,
                // full shape and strides of the result array
                // necessary for reduction and initialization
                simplified_batch_shape, outer_inner_dst_shape,
                simplified_batch_dst_strides, outer_inner_dst_strides);
            auto packed_shapes_strides_owner =
                std::move(std::get<0>(ptr_size_event_tuple1));
            sycl::event copy_shapes_strides_ev =
                std::get<2>(ptr_size_event_tuple1);
            const py::ssize_t *packed_shapes_strides =
                packed_shapes_strides_owner.get();

            const auto batch_shape_strides = packed_shapes_strides;
            const auto x1_outer_inner_shapes_strides =
                packed_shapes_strides + 4 * batch_dims;
            const auto x2_outer_inner_shapes_strides =
                packed_shapes_strides + 4 * batch_dims +
                2 * (x1_outer_inner_dims);
            const auto dst_outer_shapes_strides =
                packed_shapes_strides + 4 * batch_dims +
                2 * (x1_outer_inner_dims) + 2 * (x2_outer_inner_dims);
            const auto dst_full_shape_strides =
                packed_shapes_strides + 4 * batch_dims +
                2 * (x1_outer_inner_dims) + 2 * (x2_outer_inner_dims) +
                2 * (dst_outer_inner_dims);

            std::vector<sycl::event> all_deps;
            all_deps.reserve(depends.size() + 1);
            all_deps.insert(all_deps.end(), depends.begin(), depends.end());
            all_deps.push_back(copy_shapes_strides_ev);

            dot_ev = fn(
                exec_q, x1_data, x2_data, dst_data, batches, x1_outer_nelems,
                inner_nelems, x2_outer_nelems, batch_dims, batch_shape_strides,
                x1_batch_offset, x2_batch_offset, dst_batch_offset, inner_dims,
                x1_outer_dims, x1_outer_inner_shapes_strides, x2_outer_dims,
                x2_outer_inner_shapes_strides, x1_outer_dims + x2_outer_dims,
                dst_outer_shapes_strides, dst_full_shape_strides, all_deps);

            sycl::event cleanup_tmp_allocations_ev =
                dpctl::tensor::alloc_utils::async_smart_free(
                    exec_q, {dot_ev}, packed_shapes_strides_owner);
            host_task_events.push_back(cleanup_tmp_allocations_ev);
        }
    }
    return std::make_pair(
        dpctl::utils::keep_args_alive(exec_q, {x1, x2, dst}, host_task_events),
        dot_ev);
}

template <typename output_typesT>
py::object py_dot_result_type(const py::dtype &input1_dtype,
                              const py::dtype &input2_dtype,
                              const output_typesT &output_types_table)
{
    int tn1 = input1_dtype.num(); // NumPy type numbers are the same as in dpctl
    int tn2 = input2_dtype.num(); // NumPy type numbers are the same as in dpctl
    int src1_typeid = -1;
    int src2_typeid = -1;

    auto array_types = td_ns::usm_ndarray_types();

    try {
        src1_typeid = array_types.typenum_to_lookup_id(tn1);
        src2_typeid = array_types.typenum_to_lookup_id(tn2);
    } catch (const std::exception &e) {
        throw py::value_error(e.what());
    }

    if (src1_typeid < 0 || src1_typeid >= td_ns::num_types || src2_typeid < 0 ||
        src2_typeid >= td_ns::num_types)
    {
        throw std::runtime_error("binary output type lookup failed");
    }
    int dst_typeid = output_types_table[src1_typeid][src2_typeid];

    if (dst_typeid < 0) {
        auto res = py::none();
        return py::cast<py::object>(res);
    }
    else {
        auto dst_typenum_t = static_cast<td_ns::typenum_t>(dst_typeid);
        auto dt = type_utils::_dtype_from_typenum(dst_typenum_t);

        return py::cast<py::object>(dt);
    }
}

void init_dot(py::module_ m)
{
    init_dot_atomic_support_vector();
    init_dot_dispatch_tables();

    m.def("_dot", &py_dot, "", py::arg("x1"), py::arg("x2"),
          py::arg("batch_dims"), py::arg("x1_outer_dims"),
          py::arg("x2_outer_dims"), py::arg("inner_dims"), py::arg("dst"),
          py::arg("sycl_queue"), py::arg("depends") = py::list());

    auto dot_result_type_pyapi = [&](const py::dtype &dtype1,
                                     const py::dtype &dtype2) {
        return py_dot_result_type(dtype1, dtype2, dot_output_id_table);
    };
    m.def("_dot_result_type", dot_result_type_pyapi, "");
}

} // namespace dpctl::tensor::py_internal
