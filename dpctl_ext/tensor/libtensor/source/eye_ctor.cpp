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
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"
#include <pybind11/pybind11.h>

#include "eye_ctor.hpp"
#include "kernels/constructors.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl::tensor::py_internal
{

using dpctl::utils::keep_args_alive;

using dpctl::tensor::kernels::constructors::eye_fn_ptr_t;
static eye_fn_ptr_t eye_dispatch_vector[td_ns::num_types];

std::pair<sycl::event, sycl::event>
    usm_ndarray_eye(py::ssize_t k,
                    const dpctl::tensor::usm_ndarray &dst,
                    sycl::queue &exec_q,
                    const std::vector<sycl::event> &depends)
{
    // dst must be 2D

    if (dst.get_ndim() != 2) {
        throw py::value_error(
            "usm_ndarray_eye: Expecting 2D array to populate");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst})) {
        throw py::value_error("Execution queue is not compatible with the "
                              "allocation queue");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    auto array_types = td_ns::usm_ndarray_types();
    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    const py::ssize_t nelem = dst.get_size();
    const py::ssize_t rows = dst.get_shape(0);
    const py::ssize_t cols = dst.get_shape(1);
    if (rows == 0 || cols == 0) {
        // nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    bool is_dst_c_contig = dst.is_c_contiguous();
    bool is_dst_f_contig = dst.is_f_contiguous();
    if (!is_dst_c_contig && !is_dst_f_contig) {
        throw py::value_error("USM array is not contiguous");
    }

    py::ssize_t start;
    if (is_dst_c_contig) {
        start = (k < 0) ? -k * cols : k;
    }
    else {
        start = (k < 0) ? -k : k * rows;
    }

    const py::ssize_t *strides = dst.get_strides_raw();
    py::ssize_t step;
    if (strides == nullptr) {
        step = (is_dst_c_contig) ? cols + 1 : rows + 1;
    }
    else {
        step = strides[0] + strides[1];
    }

    const py::ssize_t length = std::min({rows, cols, rows + k, cols - k});
    const py::ssize_t end = start + step * (length - 1);

    char *dst_data = dst.get_data();
    sycl::event eye_event;

    auto fn = eye_dispatch_vector[dst_typeid];

    eye_event = fn(exec_q, static_cast<std::size_t>(nelem), start, end, step,
                   dst_data, depends);

    return std::make_pair(keep_args_alive(exec_q, {dst}, {eye_event}),
                          eye_event);
}

void init_eye_ctor_dispatch_vectors(void)
{
    using namespace td_ns;
    using dpctl::tensor::kernels::constructors::EyeFactory;

    DispatchVectorBuilder<eye_fn_ptr_t, EyeFactory, num_types> dvb;
    dvb.populate_dispatch_vector(eye_dispatch_vector);

    return;
}

} // namespace dpctl::tensor::py_internal
