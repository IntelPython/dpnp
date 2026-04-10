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
/// This file defines functions of dpnp.tensor._tensor_sorting_impl
/// extension.
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>

#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl::tensor::py_internal
{

template <typename sorting_contig_impl_fnT>
std::pair<sycl::event, sycl::event>
    py_argsort(const dpctl::tensor::usm_ndarray &src,
               const int trailing_dims_to_sort,
               const dpctl::tensor::usm_ndarray &dst,
               sycl::queue &exec_q,
               const std::vector<sycl::event> &depends,
               const sorting_contig_impl_fnT &sort_contig_fns)
{
    int src_nd = src.get_ndim();
    int dst_nd = dst.get_ndim();
    if (src_nd != dst_nd) {
        throw py::value_error("The input and output arrays must have "
                              "the same array ranks");
    }
    int iteration_nd = src_nd - trailing_dims_to_sort;
    if (trailing_dims_to_sort <= 0 || iteration_nd < 0) {
        throw py::value_error("Trailing_dim_to_sort must be positive, but no "
                              "greater than rank of the array being sorted");
    }

    const py::ssize_t *src_shape_ptr = src.get_shape_raw();
    const py::ssize_t *dst_shape_ptr = dst.get_shape_raw();

    bool same_shapes = true;
    std::size_t iter_nelems(1);

    for (int i = 0; same_shapes && (i < iteration_nd); ++i) {
        auto src_shape_i = src_shape_ptr[i];
        same_shapes = same_shapes && (src_shape_i == dst_shape_ptr[i]);
        iter_nelems *= static_cast<std::size_t>(src_shape_i);
    }

    std::size_t sort_nelems(1);
    for (int i = iteration_nd; same_shapes && (i < src_nd); ++i) {
        auto src_shape_i = src_shape_ptr[i];
        same_shapes = same_shapes && (src_shape_i == dst_shape_ptr[i]);
        sort_nelems *= static_cast<std::size_t>(src_shape_i);
    }

    if (!same_shapes) {
        throw py::value_error(
            "Destination shape does not match the input shape");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {src, dst})) {
        throw py::value_error(
            "Execution queue is not compatible with allocation queues");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    if ((iter_nelems == 0) || (sort_nelems == 0)) {
        // Nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    // check that dst and src do not overlap
    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(src, dst)) {
        throw py::value_error("Arrays index overlapping segments of memory");
    }

    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(
        dst, sort_nelems * iter_nelems);

    int src_typenum = src.get_typenum();
    int dst_typenum = dst.get_typenum();

    const auto &array_types = td_ns::usm_ndarray_types();
    int src_typeid = array_types.typenum_to_lookup_id(src_typenum);
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    if ((dst_typeid != static_cast<int>(td_ns::typenum_t::INT64)) &&
        (dst_typeid != static_cast<int>(td_ns::typenum_t::INT32)))
    {
        throw py::value_error(
            "Output index array must have data type int32 or int64");
    }

    bool is_src_c_contig = src.is_c_contiguous();
    bool is_dst_c_contig = dst.is_c_contiguous();

    if (is_src_c_contig && is_dst_c_contig) {
        if (sort_nelems > 1) {
            static constexpr py::ssize_t zero_offset = py::ssize_t(0);

            auto fn = sort_contig_fns[src_typeid][dst_typeid];

            if (fn == nullptr) {
                throw py::value_error(
                    "Not implemented for dtypes of input arrays");
            }

            sycl::event comp_ev =
                fn(exec_q, iter_nelems, sort_nelems, src.get_data(),
                   dst.get_data(), zero_offset, zero_offset, zero_offset,
                   zero_offset, depends);

            sycl::event keep_args_alive_ev =
                dpctl::utils::keep_args_alive(exec_q, {src, dst}, {comp_ev});

            return std::make_pair(keep_args_alive_ev, comp_ev);
        }
        else {
            assert(dst.get_size() == iter_nelems);
            int dst_elemsize = dst.get_elemsize();
            static constexpr int memset_val(0);

            sycl::event fill_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                cgh.memset(reinterpret_cast<void *>(dst.get_data()), memset_val,
                           iter_nelems * dst_elemsize);
            });

            sycl::event keep_args_alive_ev =
                dpctl::utils::keep_args_alive(exec_q, {src, dst}, {fill_ev});

            return std::make_pair(keep_args_alive_ev, fill_ev);
        }
    }

    throw py::value_error(
        "Both source and destination arrays must be C-contiguous");
}

} // namespace dpctl::tensor::py_internal
