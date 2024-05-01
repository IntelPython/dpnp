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

#include "axpy.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp::extensions::blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*axpy_batch_impl_fn_ptr_t)(
    sycl::queue &,
    const std::int64_t,
    const char *,
    const char *,
    const std::int64_t,
    const std::int64_t,
    char *,
    const std::int64_t,
    const std::int64_t,
    const std::int64_t,
    const std::vector<sycl::event> &);

static axpy_batch_impl_fn_ptr_t
    axpy_batch_dispatch_vector[dpctl_td_ns::num_types];

template <typename T>
static sycl::event axpy_batch_impl(sycl::queue &exec_q,
                                   const std::int64_t n,
                                   const char *scalarA,
                                   const char *vectorX,
                                   const std::int64_t incx,
                                   const std::int64_t stridex,
                                   char *vectorY,
                                   const std::int64_t incy,
                                   const std::int64_t stridey,
                                   const std::int64_t batch_size,
                                   const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<T>(exec_q);

    const T *x = reinterpret_cast<const T *>(vectorX);
    const T *alpha = reinterpret_cast<const T *>(scalarA);
    T *y = reinterpret_cast<T *>(vectorY);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event axpy_batch_event;
    try {
        axpy_batch_event = mkl_blas::row_major::axpy_batch(
            exec_q,
            n,       // Size of the input vectors
            alpha,   // The scalar alpha
            x,       // Pointer to vector x.
            incx,    // The stride of vector x.
            stridex, // Stride between different consecutive X vectors.
            y,       // Pointer to output vector y, where the result is stored.
            incy,    // The stride of vector y.
            stridey, // Stride between two consecutive Y vectors.
            batch_size, // Specifies the number of matrix-vector operations
                        // to perform.
            depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg << "Unexpected MKL exception caught during axpy_batch() "
                     "call:\nreason: "
                  << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg
            << "Unexpected SYCL exception caught during axpy_batch() call:\n"
            << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return axpy_batch_event;
}

std::pair<sycl::event, sycl::event>
    axpy_batch(sycl::queue &exec_q,
               const dpctl::tensor::usm_ndarray &scalarA,
               const dpctl::tensor::usm_ndarray &vectorX,
               const dpctl::tensor::usm_ndarray &vectorY,
               const std::vector<sycl::event> &depends)
{
    const int scalarA_nd = scalarA.get_ndim();
    const int vectorX_nd = vectorX.get_ndim();
    const int vectorY_nd = vectorY.get_ndim();

    if ((scalarA_nd != 0)) {
        throw py::value_error(
            "The first input array has ndim=" + std::to_string(scalarA_nd) +
            ", but a 0-dimensional array is expected.");
    }

    if ((vectorX_nd != vectorY_nd)) {
        throw py::value_error(
            "The input and output arrays must have the same dimension.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(vectorX, vectorY)) {
        throw py::value_error("The input and output vectors are overlapping "
                              "segments of memory");
    }
    if (overlap(scalarA, vectorY)) {
        throw py::value_error(
            "The input scalar and output array are overlapping "
            "segments of memory");
    }

    if (!dpctl::utils::queues_are_compatible(
            exec_q,
            {vectorX.get_queue(), scalarA.get_queue(), vectorY.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    const py::ssize_t x_size = vectorX.get_size();
    const py::ssize_t y_size = vectorY.get_size();
    if (x_size != y_size) {
        throw py::value_error("The size of the input vector must be "
                              "equal to the size of the output vector.");
    }

    const py::ssize_t *x_shape = vectorX.get_shape_raw();
    const py::ssize_t *y_shape = vectorY.get_shape_raw();
    const std::int64_t batch_size = x_shape[0];
    const std::int64_t n = x_shape[1];

    if (batch_size != y_shape[0]) {
        throw py::value_error("The number of rows in X must be equal to "
                              "the number of rows in Y.");
    }
    if (n != y_shape[1]) {
        throw py::value_error("The number of columns in X must be equal to "
                              "the number of columns in Y.");
    }

    const size_t src_nelems = n * batch_size;
    dpctl::tensor::validation::CheckWritable::throw_if_not_writable(vectorY);
    dpctl::tensor::validation::AmpleMemory::throw_if_not_ample(vectorY,
                                                               src_nelems);

    const int vectorX_typenum = vectorX.get_typenum();
    const int scalarA_typenum = scalarA.get_typenum();
    const int vectorY_typenum = vectorY.get_typenum();

    if (scalarA_typenum != vectorY_typenum ||
        vectorX_typenum != vectorY_typenum)
    {
        throw py::value_error("Input and output must be of the same type.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int type_id = array_types.typenum_to_lookup_id(vectorX_typenum);

    axpy_batch_impl_fn_ptr_t axpy_batch_fn =
        axpy_batch_dispatch_vector[type_id];
    if (axpy_batch_fn == nullptr) {
        throw py::value_error(
            "Types of input and result vectors are mismatched.");
    }

    const char *a_typeless_ptr = scalarA.get_data();
    char *x_typeless_ptr = vectorX.get_data();
    char *y_typeless_ptr = vectorY.get_data();

    const std::vector<py::ssize_t> x_stride = vectorX.get_strides_vector();
    const std::vector<py::ssize_t> y_stride = vectorY.get_strides_vector();
    const std::int64_t stridex = x_stride[0];
    const std::int64_t stridey = y_stride[0];
    const std::int64_t incx = x_stride[1];
    const std::int64_t incy = y_stride[1];
    const int x_elemsize = vectorX.get_elemsize();
    const int y_elemsize = vectorY.get_elemsize();
    if (incx < 0) {
        x_typeless_ptr -= (n - 1) * std::abs(incx) * x_elemsize;
    }
    if (incy < 0) {
        y_typeless_ptr -= (n - 1) * std::abs(incy) * y_elemsize;
    }

    sycl::event axpy_batch_ev =
        axpy_batch_fn(exec_q, n, a_typeless_ptr, x_typeless_ptr, incx, stridex,
                      y_typeless_ptr, incy, stridey, batch_size, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {vectorX, scalarA, vectorY}, {axpy_batch_ev});

    return std::make_pair(args_ev, axpy_batch_ev);
}

template <typename fnT, typename varT>
struct AxpyBatchContigFactory
{
    fnT get()
    {
        if constexpr (types::AxpyBatchTypePairSupportFactory<varT>::is_defined)
        {
            return axpy_batch_impl<varT>;
        }
        else {
            return nullptr;
        }
    }
};

void init_axpy_batch_dispatch_vector(void)
{
    dpctl_td_ns::DispatchVectorBuilder<axpy_batch_impl_fn_ptr_t,
                                       AxpyBatchContigFactory,
                                       dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_vector(axpy_batch_dispatch_vector);
}
} // namespace dpnp::extensions::blas
