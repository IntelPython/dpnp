//*****************************************************************************
// Copyright (c) 2024, Intel Corporation
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

#include "nrm2.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp::extensions::blas
{
namespace py = pybind11;

size_t computeProduct(const std::vector<size_t> &shapes, size_t start)
{
    size_t product = 1;
    for (size_t i = start; i < shapes.size(); ++i) {
        product *= shapes[i];
    }
    return product;
}

// Convert Linear Index to N-Dimensional Index
size_t calc_offset(const std::vector<size_t> &shapes,
                   const std::vector<size_t> &strides,
                   size_t linearIndex)
{
    std::vector<size_t> indices(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
        indices[i] = linearIndex / computeProduct(shapes, i + 1);
        linearIndex %= computeProduct(shapes, i + 1);
    }

    int offset = 0;
    for (size_t i = 0; i < shapes.size(); ++i) {
        offset += indices[i] * strides[i];
    }

    return offset;
}

std::pair<sycl::event, std::vector<sycl::event>>
    nrm2_batch(sycl::queue &exec_q,
               dpctl::tensor::usm_ndarray arrayX,
               dpctl::tensor::usm_ndarray result,
               const std::vector<sycl::event> &depends)
{
    const int arrayX_nd = arrayX.get_ndim();
    // should be 2D array if it is reshaped in the python side

    const py::ssize_t *x_shape = arrayX.get_shape_raw();
    const std::int64_t last_shape = x_shape[arrayX_nd - 1];

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(arrayX, result)) {
        throw py::value_error(
            "The first input array and output array are overlapping "
            "segments of memory");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(
            exec_q, {arrayX.get_queue(), result.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    // check writable
    // check ample memory
    const py::ssize_t res_size = result.get_size();
    const std::vector<py::ssize_t> x_stride = arrayX.get_strides_vector();
    const std::int64_t last_stride = x_stride[arrayX_nd - 1];

    const int arrayX_typenum = arrayX.get_typenum();
    const int result_typenum = result.get_typenum();

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int arrayX_type_id = array_types.typenum_to_lookup_id(arrayX_typenum);
    const int result_type_id = array_types.typenum_to_lookup_id(result_typenum);

    nrm2_impl_fn_ptr_t nrm2_fn =
        nrm2_dispatch_table[arrayX_type_id][result_type_id];
    if (nrm2_fn == nullptr) {
        throw py::value_error(
            "Types of input array and result array are mismatched.");
    }

    char *x_typeless_ptr = arrayX.get_data();
    char *r_typeless_ptr = result.get_data();

    const int x_elemsize = arrayX.get_elemsize();
    const int res_elemsize = result.get_elemsize();
    if (last_stride < 0) {
        x_typeless_ptr -= (last_shape - 1) * std::abs(last_stride) * x_elemsize;
    }

    std::vector<sycl::event> host_task_events;
    sycl::event nrm2_ev;

    const bool is_arrayX_c_contig = arrayX.is_c_contiguous();
    if (is_arrayX_c_contig) {
        for (int index = 0; index < res_size; index++) {
            nrm2_ev = nrm2_fn(exec_q, last_shape, x_typeless_ptr, last_stride,
                              r_typeless_ptr, depends);
            x_typeless_ptr += last_shape * x_elemsize;
            r_typeless_ptr += res_elemsize;
            host_task_events.push_back(nrm2_ev);
        }
    }
    else {
        int offset;
        std::vector<size_t> shape_1(x_shape, x_shape + arrayX_nd - 1);
        std::vector<size_t> strides_1(x_stride.begin(), x_stride.end() - 1);
        for (int index = 0; index < res_size; index++) {
            offset = calc_offset(shape_1, strides_1, index);
            nrm2_ev = nrm2_fn(exec_q, last_shape,
                              x_typeless_ptr + offset * x_elemsize, last_stride,
                              r_typeless_ptr, depends);
            r_typeless_ptr += res_elemsize;

            host_task_events.push_back(nrm2_ev);
        }
    }

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {arrayX, result}, host_task_events);

    return std::make_pair(args_ev, host_task_events);
}

} // namespace dpnp::extensions::blas
