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

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

#include "dpnp4pybind11.hpp"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include "kernels/constructors.hpp"
#include "utils/output_validation.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

#include "full_ctor.hpp"

namespace py = pybind11;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace dpctl
{
namespace tensor
{
namespace py_internal
{

using dpctl::utils::keep_args_alive;

typedef sycl::event (*full_contig_fn_ptr_t)(sycl::queue &,
                                            std::size_t,
                                            const py::object &,
                                            char *,
                                            const std::vector<sycl::event> &);

/*!
 * @brief Function to submit kernel to fill given contiguous memory allocation
 * with specified value.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nelems  Length of the sequence
 * @param py_value  Python object representing the value to fill the array with.
 * Must be convertible to `dstTy`.
 * @param dst_p  Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename dstTy>
sycl::event full_contig_impl(sycl::queue &exec_q,
                             std::size_t nelems,
                             const py::object &py_value,
                             char *dst_p,
                             const std::vector<sycl::event> &depends)
{
    dstTy fill_v = py::cast<dstTy>(py_value);

    sycl::event fill_ev;

    if constexpr (sizeof(dstTy) == sizeof(char)) {
        const auto memset_val = sycl::bit_cast<unsigned char>(fill_v);
        fill_ev = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            cgh.memset(reinterpret_cast<void *>(dst_p), memset_val,
                       nelems * sizeof(dstTy));
        });
    }
    else {
        bool is_zero = false;
        if constexpr (sizeof(dstTy) == 1) {
            is_zero = (std::uint8_t{0} == sycl::bit_cast<std::uint8_t>(fill_v));
        }
        else if constexpr (sizeof(dstTy) == 2) {
            is_zero =
                (std::uint16_t{0} == sycl::bit_cast<std::uint16_t>(fill_v));
        }
        else if constexpr (sizeof(dstTy) == 4) {
            is_zero =
                (std::uint32_t{0} == sycl::bit_cast<std::uint32_t>(fill_v));
        }
        else if constexpr (sizeof(dstTy) == 8) {
            is_zero =
                (std::uint64_t{0} == sycl::bit_cast<std::uint64_t>(fill_v));
        }
        else if constexpr (sizeof(dstTy) == 16) {
            struct UInt128
            {

                constexpr UInt128() : v1{}, v2{} {}
                UInt128(const UInt128 &) = default;

                operator bool() const
                {
                    return bool(!v1) && bool(!v2);
                }

                std::uint64_t v1;
                std::uint64_t v2;
            };
            is_zero = static_cast<bool>(sycl::bit_cast<UInt128>(fill_v));
        }

        if (is_zero) {
            static constexpr int memset_val = 0;
            fill_ev = exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends);

                cgh.memset(reinterpret_cast<void *>(dst_p), memset_val,
                           nelems * sizeof(dstTy));
            });
        }
        else {
            using dpctl::tensor::kernels::constructors::full_contig_impl;

            fill_ev =
                full_contig_impl<dstTy>(exec_q, nelems, fill_v, dst_p, depends);
        }
    }

    return fill_ev;
}

template <typename fnT, typename Ty>
struct FullContigFactory
{
    fnT get()
    {
        fnT f = full_contig_impl<Ty>;
        return f;
    }
};

typedef sycl::event (*full_strided_fn_ptr_t)(sycl::queue &,
                                             int,
                                             std::size_t,
                                             py::ssize_t *,
                                             const py::object &,
                                             char *,
                                             const std::vector<sycl::event> &);

/*!
 * @brief Function to submit kernel to fill given strided memory allocation
 * with specified value.
 *
 * @param exec_q  Sycl queue to which kernel is submitted for execution.
 * @param nd  Array dimensionality
 * @param nelems  Length of the sequence
 * @param shape_strides  Kernel accessible USM pointer to packed shape and
 * strides of array.
 * @param py_value  Python object representing the value to fill the array with.
 * Must be convertible to `dstTy`.
 * @param dst_p  Kernel accessible USM pointer to the start of array to be
 * populated.
 * @param depends  List of events to wait for before starting computations, if
 * any.
 *
 * @return Event to wait on to ensure that computation completes.
 * @defgroup CtorKernels
 */
template <typename dstTy>
sycl::event full_strided_impl(sycl::queue &exec_q,
                              int nd,
                              std::size_t nelems,
                              py::ssize_t *shape_strides,
                              const py::object &py_value,
                              char *dst_p,
                              const std::vector<sycl::event> &depends)
{
    dstTy fill_v = py::cast<dstTy>(py_value);

    using dpctl::tensor::kernels::constructors::full_strided_impl;
    sycl::event fill_ev = full_strided_impl<dstTy>(
        exec_q, nd, nelems, shape_strides, fill_v, dst_p, depends);

    return fill_ev;
}

template <typename fnT, typename Ty>
struct FullStridedFactory
{
    fnT get()
    {
        fnT f = full_strided_impl<Ty>;
        return f;
    }
};

static full_contig_fn_ptr_t full_contig_dispatch_vector[td_ns::num_types];
static full_strided_fn_ptr_t full_strided_dispatch_vector[td_ns::num_types];

std::pair<sycl::event, sycl::event>
    usm_ndarray_full(const py::object &py_value,
                     const dpctl::tensor::usm_ndarray &dst,
                     sycl::queue &exec_q,
                     const std::vector<sycl::event> &depends)
{
    // py_value should be coercible into data type of dst

    py::ssize_t dst_nelems = dst.get_size();

    if (dst_nelems == 0) {
        // nothing to do
        return std::make_pair(sycl::event(), sycl::event());
    }

    if (!dpctl::utils::queues_are_compatible(exec_q, {dst})) {
        throw py::value_error(
            "Execution queue is not compatible with the allocation queue");
    }

    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(dst);

    auto array_types = td_ns::usm_ndarray_types();
    int dst_typenum = dst.get_typenum();
    int dst_typeid = array_types.typenum_to_lookup_id(dst_typenum);

    char *dst_data = dst.get_data();

    if (dst_nelems == 1 || dst.is_c_contiguous() || dst.is_f_contiguous()) {
        auto fn = full_contig_dispatch_vector[dst_typeid];

        sycl::event full_contig_event =
            fn(exec_q, static_cast<std::size_t>(dst_nelems), py_value, dst_data,
               depends);

        return std::make_pair(
            keep_args_alive(exec_q, {dst}, {full_contig_event}),
            full_contig_event);
    }
    else {
        int nd = dst.get_ndim();
        auto const &dst_shape = dst.get_shape_vector();
        auto const &dst_strides = dst.get_strides_vector();

        auto fn = full_strided_dispatch_vector[dst_typeid];

        std::vector<sycl::event> host_task_events;
        host_task_events.reserve(2);
        using dpctl::tensor::offset_utils::device_allocate_and_pack;
        auto ptr_size_event_tuple = device_allocate_and_pack<py::ssize_t>(
            exec_q, host_task_events, dst_shape, dst_strides);
        auto shape_strides_owner = std::move(std::get<0>(ptr_size_event_tuple));
        const sycl::event &copy_shape_ev = std::get<2>(ptr_size_event_tuple);
        py::ssize_t *shape_strides = shape_strides_owner.get();

        const sycl::event &full_strided_ev =
            fn(exec_q, nd, dst_nelems, shape_strides, py_value, dst_data,
               {copy_shape_ev});

        // free shape_strides
        const auto &temporaries_cleanup_ev =
            dpctl::tensor::alloc_utils::async_smart_free(
                exec_q, {full_strided_ev}, shape_strides_owner);
        host_task_events.push_back(temporaries_cleanup_ev);

        return std::make_pair(keep_args_alive(exec_q, {dst}, host_task_events),
                              full_strided_ev);
    }
}

void init_full_ctor_dispatch_vectors(void)
{
    using namespace td_ns;

    DispatchVectorBuilder<full_contig_fn_ptr_t, FullContigFactory, num_types>
        dvb1;
    dvb1.populate_dispatch_vector(full_contig_dispatch_vector);

    DispatchVectorBuilder<full_strided_fn_ptr_t, FullStridedFactory, num_types>
        dvb2;
    dvb2.populate_dispatch_vector(full_strided_dispatch_vector);
}

} // namespace py_internal
} // namespace tensor
} // namespace dpctl
