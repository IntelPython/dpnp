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
#include "utils/output_validation.hpp"
#include "utils/type_utils.hpp"

#include "scal.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp::extensions::blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*scal_impl_fn_ptr_t)(sycl::queue &,
                                          const std::int64_t,
                                          const char *,
                                          char *,
                                          const std::int64_t,
                                          const std::vector<sycl::event> &);

static scal_impl_fn_ptr_t scal_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event scal_impl(sycl::queue &exec_q,
                             const std::int64_t n,
                             const char *scalarA,
                             char *vectorX,
                             const std::int64_t incx,
                             const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    T *x = reinterpret_cast<T *>(vectorX);
    const T *alpha = reinterpret_cast<const T *>(scalarA);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event scal_event;
    try {
        scal_event = mkl_blas::row_major::scal(
            exec_q,
            n,     // Size of the input vectors
            alpha, // The scalar alpha
            x,     // Pointer to vector x, input and output.
            incx,  // The stride of vector x.
            depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during scal() call:\nreason: "
            << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during scal() call:\n"
                  << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return scal_event;
}

std::pair<sycl::event, sycl::event>
    scal(sycl::queue &exec_q,
         const dpctl::tensor::usm_ndarray &scalarA,
         const dpctl::tensor::usm_ndarray &vectorX,
         const std::vector<sycl::event> &depends)
{
    const int scalarA_nd = scalarA.get_ndim();
    const int vectorX_nd = vectorX.get_ndim();

    if ((scalarA_nd != 0)) {
        throw py::value_error(
            "The first input array has ndim=" + std::to_string(scalarA_nd) +
            ", but a 0-dimensional array is expected.");
    }

    if ((vectorX_nd != 1)) {
        throw py::value_error(
            "The second input array has ndim=" + std::to_string(vectorX_nd) +
            ", but a 1-dimensional array is expected.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(scalarA, vectorX)) {
        throw py::value_error(
            "The input scalar and output array are overlapping "
            "segments of memory");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(
            exec_q, {vectorX.get_queue(), scalarA.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    const py::ssize_t x_size = vectorX.get_size();
    const std::int64_t n = x_size;
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(vectorX);

    const std::vector<py::ssize_t> x_stride = vectorX.get_strides_vector();
    const std::int64_t incx = x_stride[0];

    const int vectorX_typenum = vectorX.get_typenum();
    const int scalarA_typenum = scalarA.get_typenum();

    if (scalarA_typenum != vectorX_typenum) {
        throw py::value_error("Inputs must be of the same type.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int type_id = array_types.typenum_to_lookup_id(vectorX_typenum);

    scal_impl_fn_ptr_t scal_fn = scal_dispatch_vector[type_id];
    if (scal_fn == nullptr) {
        throw py::value_error(
            "Types of input and result vectors are mismatched.");
    }

    const char *a_typeless_ptr = scalarA.get_data();
    char *x_typeless_ptr = vectorX.get_data();

    const int x_elemsize = vectorX.get_elemsize();
    if (incx < 0) {
        x_typeless_ptr -= (n - 1) * std::abs(incx) * x_elemsize;
    }

    sycl::event scal_ev =
        scal_fn(exec_q, n, a_typeless_ptr, x_typeless_ptr, incx, depends);

    sycl::event args_ev =
        dpctl::utils::keep_args_alive(exec_q, {vectorX, scalarA}, {scal_ev});

    return std::make_pair(args_ev, scal_ev);
}

template <typename fnT, typename varT>
struct ScalContigFactory
{
    fnT get()
    {
        if constexpr (types::ScalTypePairSupportFactory<varT>::is_defined) {
            return scal_impl<varT>;
        }
        else {
            return nullptr;
        }
    }
};

void init_scal_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<scal_impl_fn_ptr_t, ScalContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(scal_dispatch_vector);
}
} // namespace dpnp::extensions::blas
