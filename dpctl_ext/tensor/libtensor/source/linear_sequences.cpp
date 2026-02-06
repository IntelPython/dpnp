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

#include "dpnp4pybind11.hpp"
#include <complex>
#include <cstddef>
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "kernels/constructors.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "linear_sequences.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

// Constructor to populate tensor with linear sequence defined by
// start and step data

typedef sycl::event (*lin_space_step_fn_ptr_t)(
    sycl::queue &,
    std::size_t, // num_elements
    const py::object &start,
    const py::object &step,
    char *, // dst_data_ptr
    const std::vector<sycl::event> &);

/*!
 * @brief Function to submit kernel to populate given contiguous memory
 * allocation with linear sequence specified by starting value and increment
 * given as Python objects.
 *
 * @param q  Sycl queue to which the kernel is submitted
 * @param nelems Length of the sequence
 * @param start Starting value of the sequence as Python object. Must be
 * convertible to array element data type `Ty`.
 * @param step  Increment of the sequence as Python object. Must be convertible
 * to array element data type `Ty`.
 * @param array_data Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty>
sycl::event lin_space_step_impl(sycl::queue &exec_q,
                                std::size_t nelems,
                                const py::object &start,
                                const py::object &step,
                                char *array_data,
                                const std::vector<sycl::event> &depends)
{
    Ty start_v = py::cast<Ty>(start);
    Ty step_v = py::cast<Ty>(step);

    using dpctl::tensor::kernels::constructors::lin_space_step_impl;

    auto lin_space_step_event = lin_space_step_impl<Ty>(
        exec_q, nelems, start_v, step_v, array_data, depends);

    return lin_space_step_event;
}

typedef sycl::event (*lin_space_affine_fn_ptr_t)(
    sycl::queue &,
    std::size_t, // num_elements
    const py::object &start,
    const py::object &end,
    bool include_endpoint,
    char *, // dst_data_ptr
    const std::vector<sycl::event> &);

/*!
 * @brief Function to submit kernel to populate given contiguous memory
 * allocation with linear sequence specified  by starting and end values given
 * as Python objects.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Length of the sequence
 * @param start Stating value of the sequence as Python object. Must be
 * convertible to array data element type `Ty`.
 * @param end   End-value of the sequence as Python object. Must be convertible
 * to array data element type `Ty`.
 * @param include_endpoint  Whether the end-value is included in the sequence
 * @param array_data Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename Ty>
sycl::event lin_space_affine_impl(sycl::queue &exec_q,
                                  std::size_t nelems,
                                  const py::object &start,
                                  const py::object &end,
                                  bool include_endpoint,
                                  char *array_data,
                                  const std::vector<sycl::event> &depends)
{
    Ty start_v = py::cast<Ty>(start);
    Ty end_v = py::cast<Ty>(end);

    using dpctl::tensor::kernels::constructors::lin_space_affine_impl;

    auto lin_space_affine_event = lin_space_affine_impl<Ty>(
        exec_q, nelems, start_v, end_v, include_endpoint, array_data, depends);

    return lin_space_affine_event;
}

using dpctl::utils::keep_args_alive;

static lin_space_step_fn_ptr_t lin_space_step_dispatch_vector[td_ns::num_types];

static lin_space_affine_fn_ptr_t
    lin_space_affine_dispatch_vector[td_ns::num_types];

std::pair<sycl::event, sycl::event>
    usm_ndarray_linear_sequence_step(const py::object &start,
                                     const py::object &dt,
                                     const dpctl::tensor::usm_ndarray &dst,
                                     sycl::queue &exec_q,
                                     const std::vector<sycl::event> &depends)
{
    // dst must be 1D and C-contiguous
    // start, end should be coercible into data type of dst

    if (dst.get_ndim() != 1) {
        throw py::value_error(
            "usm_ndarray_linspace: Expecting 1D array to populate");
    }

    if (!dst.is_c_contiguous()) {
        throw py::value_error(
            "usm_ndarray_linspace: Non-contiguous arrays are not supported");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst})) {
        throw py::value_error(
            "Execution queue is not compatible with the allocation queue");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    auto array_types = td_ns::usm_ndarray_types();
    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    py::ssize_t len = dst.get_shape(0);
    if (len == 0) {
        // nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    char *dst_data = dst.get_data();
    sycl::event linspace_step_event;

    auto fn = lin_space_step_dispatch_vector[dst_typeid];

    linspace_step_event =
        fn(exec_q, static_cast<std::size_t>(len), start, dt, dst_data, depends);

    return std::make_pair(keep_args_alive(exec_q, {dst}, {linspace_step_event}),
                          linspace_step_event);
}

std::pair<sycl::event, sycl::event>
    usm_ndarray_linear_sequence_affine(const py::object &start,
                                       const py::object &end,
                                       const dpctl::tensor::usm_ndarray &dst,
                                       bool include_endpoint,
                                       sycl::queue &exec_q,
                                       const std::vector<sycl::event> &depends)
{
    // dst must be 1D and C-contiguous
    // start, end should be coercible into data type of dst

    if (dst.get_ndim() != 1) {
        throw py::value_error(
            "usm_ndarray_linspace: Expecting 1D array to populate");
    }

    if (!dst.is_c_contiguous()) {
        throw py::value_error(
            "usm_ndarray_linspace: Non-contiguous arrays are not supported");
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst})) {
        throw py::value_error(
            "Execution queue context is not the same as allocation context");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    auto array_types = td_ns::usm_ndarray_types();
    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    py::ssize_t len = dst.get_shape(0);
    if (len == 0) {
        // nothing to do
        return std::make_pair(sycl::event{}, sycl::event{});
    }

    char *dst_data = dst.get_data();
    sycl::event linspace_affine_event;

    auto fn = lin_space_affine_dispatch_vector[dst_typeid];

    linspace_affine_event = fn(exec_q, static_cast<std::size_t>(len), start,
                               end, include_endpoint, dst_data, depends);

    return std::make_pair(
        keep_args_alive(exec_q, {dst}, {linspace_affine_event}),
        linspace_affine_event);
}

/*!
 * @brief  Factor to get function pointer of type `fnT` for array with elements
 * of type `Ty`.
 * @defgroup CtorKernels
 */
template <typename fnT, typename Ty>
struct LinSpaceStepFactory
{
    fnT get()
    {
        fnT f = lin_space_step_impl<Ty>;
        return f;
    }
};

/*!
 * @brief Factory to get function pointer of type `fnT` for array data type
 * `Ty`.
 */
template <typename fnT, typename Ty>
struct LinSpaceAffineFactory
{
    fnT get()
    {
        fnT f = lin_space_affine_impl<Ty>;
        return f;
    }
};

void init_linear_sequences_dispatch_vectors(void)
{
    using namespace td_ns;

    DispatchVectorBuilder<lin_space_step_fn_ptr_t, LinSpaceStepFactory,
                          num_types>
        dvb1;
    dvb1.populate_dispatch_vector(lin_space_step_dispatch_vector);

    DispatchVectorBuilder<lin_space_affine_fn_ptr_t, LinSpaceAffineFactory,
                          num_types>
        dvb2;
    dvb2.populate_dispatch_vector(lin_space_affine_dispatch_vector);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
