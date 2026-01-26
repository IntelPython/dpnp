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

#pragma once

#include <oneapi/mkl.hpp>

#include <pybind11/pybind11.h>

// dpctl tensor headers
#include "utils/memory_overlap.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "types_matrix.hpp"

namespace dpnp::extensions::blas::dot
{
typedef sycl::event (*dot_impl_fn_ptr_t)(sycl::queue &,
                                         const std::int64_t,
                                         const char *,
                                         const std::int64_t,
                                         const char *,
                                         const std::int64_t,
                                         char *,
                                         const std::vector<sycl::event> &);

namespace dpctl_td_ns = dpctl::tensor::type_dispatch;
namespace py = pybind11;

std::pair<sycl::event, sycl::event>
    dot_func(sycl::queue &exec_q,
             const dpctl::tensor::usm_ndarray &vectorX,
             const dpctl::tensor::usm_ndarray &vectorY,
             const dpctl::tensor::usm_ndarray &result,
             const std::vector<sycl::event> &depends,
             const dot_impl_fn_ptr_t *dot_dispatch_vector)
{
    const int vectorX_nd = vectorX.get_ndim();
    const int vectorY_nd = vectorY.get_ndim();
    const int result_nd = result.get_ndim();

    if ((vectorX_nd != 1)) {
        throw py::value_error(
            "The first input array has ndim=" + std::to_string(vectorX_nd) +
            ", but a 1-dimensional array is expected.");
    }

    if ((vectorY_nd != 1)) {
        throw py::value_error(
            "The second input array has ndim=" + std::to_string(vectorY_nd) +
            ", but a 1-dimensional array is expected.");
    }

    if ((result_nd != 0)) {
        throw py::value_error(
            "The output array has ndim=" + std::to_string(result_nd) +
            ", but a 0-dimensional array is expected.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(vectorX, result)) {
        throw py::value_error(
            "The first input array and output array are overlapping "
            "segments of memory");
    }
    if (overlap(vectorY, result)) {
        throw py::value_error(
            "The second input array and output array are overlapping "
            "segments of memory");
    }

    if (!dpctl::utils::queues_are_compatible(
            exec_q,
            {vectorX.get_queue(), vectorY.get_queue(), result.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    const int src_nelems = 1;
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(result);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(result,
                                                               src_nelems);

    const py::ssize_t x_size = vectorX.get_size();
    const py::ssize_t y_size = vectorY.get_size();
    const std::int64_t n = x_size;
    if (x_size != y_size) {
        throw py::value_error("The size of the first input array must be "
                              "equal to the size of the second input array.");
    }

    const int vectorX_typenum = vectorX.get_typenum();
    const int vectorY_typenum = vectorY.get_typenum();
    const int result_typenum = result.get_typenum();

    if (result_typenum != vectorX_typenum || result_typenum != vectorY_typenum)
    {
        throw py::value_error("Given arrays must be of the same type.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int type_id = array_types.typenum_to_lookup_id(vectorX_typenum);

    dot_impl_fn_ptr_t dot_fn = dot_dispatch_vector[type_id];
    if (dot_fn == nullptr) {
        throw py::value_error(
            "No dot implementation is available for the specified data type "
            "of the input and output arrays.");
    }

    char *x_typeless_ptr = vectorX.get_data();
    char *y_typeless_ptr = vectorY.get_data();
    char *r_typeless_ptr = result.get_data();

    const std::vector<py::ssize_t> x_stride = vectorX.get_strides_vector();
    const std::vector<py::ssize_t> y_stride = vectorY.get_strides_vector();
    const int x_elemsize = vectorX.get_elemsize();
    const int y_elemsize = vectorY.get_elemsize();

    const std::int64_t incx = x_stride[0];
    const std::int64_t incy = y_stride[0];
    // In OneMKL, the pointer should always point out to the first element of
    // the array and OneMKL handle the rest depending on the sign of stride.
    // In OneMKL, when the stride is positive, the data is read in order and
    // when it is negative, the data is read in reverse order while pointer
    // always point to the first element
    // When the stride is negative, the pointer of the array coming from dpnp
    // points to the last element. So, we need to adjust the pointer
    if (incx < 0) {
        x_typeless_ptr -= (n - 1) * std::abs(incx) * x_elemsize;
    }
    if (incy < 0) {
        y_typeless_ptr -= (n - 1) * std::abs(incy) * y_elemsize;
    }

    sycl::event dot_ev = dot_fn(exec_q, n, x_typeless_ptr, incx, y_typeless_ptr,
                                incy, r_typeless_ptr, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {vectorX, vectorY, result}, {dot_ev});

    return std::make_pair(args_ev, dot_ev);
}
} // namespace dpnp::extensions::blas::dot
