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
#include "utils/type_utils.hpp"

#include "dot.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp
{
namespace backend
{
namespace ext
{
namespace blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

typedef sycl::event (*dotu_impl_fn_ptr_t)(sycl::queue &,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
                                          char *,
                                          const std::int64_t,
                                          char *,
                                          const std::vector<sycl::event> &);

static dotu_impl_fn_ptr_t dotu_dispatch_table[dpctl_td_ns::num_types]
                                             [dpctl_td_ns::num_types];

template <typename Tab, typename Tc>
static sycl::event dotu_impl(sycl::queue &exec_q,
                             const std::int64_t n,
                             char *vectorX,
                             const std::int64_t incx,
                             char *vectorY,
                             const std::int64_t incy,
                             char *result,
                             const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<Tab>(exec_q);
    type_utils::validate_type_for_device<Tc>(exec_q);

    T *x = reinterpret_cast<T *>(vectorX);
    T *y = reinterpret_cast<T *>(vectorY);
    T *res = reinterpret_cast<T *>(result);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event dotu_event;
    try {
        dotu_event = mkl_blas::row_major::dotu(exec_q,
                                               n, // size of the input vectors
                                               x, // Pointer to vector x.
                                               incx, // Stride of vector x.
                                               y,    // Pointer to vector y.
                                               incy, // Stride of vector y.
                                               res,  // Pointer to result.
                                               depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during dotu() call:\nreason: "
            << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during dotu() call:\n"
                  << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return dotu_event;
}

std::pair<sycl::event, sycl::event>
    dotu(sycl::queue &exec_q,
         dpctl::tensor::usm_ndarray vectorX,
         dpctl::tensor::usm_ndarray vectorY,
         dpctl::tensor::usm_ndarray result,
         const std::vector<sycl::event> &depends)
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

    py::ssize_t x_size = vectorX.get_size();
    py::ssize_t y_size = vectorY.get_size();
    const std::int64_t n = x_size;
    if (x_size != y_size) {
        throw py::value_error("The size of the first input array must be "
                              "equal to the size of the second input array.");
    }

    int vectorX_typenum = vectorX.get_typenum();
    int vectorY_typenum = vectorY.get_typenum();
    int result_typenum = result.get_typenum();

    if (vectorX_typenum != vectorY_typenum) {
        throw py::value_error(
            "Input arrays must be of must be of the same type.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int vectorXB_type_id = array_types.typenum_to_lookup_id(vectorX_typenum);
    int result_type_id = array_types.typenum_to_lookup_id(result_typenum);

    dotu_impl_fn_ptr_t dotu_fn =
        dotu_dispatch_table[vectorXB_type_id][result_type_id];
    if (dotu_fn == nullptr) {
        throw py::value_error(
            "Types of input vectors and result array are mismatched.");
    }

    char *x_typeless_ptr = vectorX.get_data();
    char *y_typeless_ptr = vectorY.get_data();
    char *r_typeless_ptr = result.get_data();

    std::vector<py::ssize_t> x_stride = vectorX.get_strides_vector();
    std::vector<py::ssize_t> y_stride = vectorY.get_strides_vector();
    const int x_elemsize = vectorX.get_elemsize();
    const int y_elemsize = vectorY.get_elemsize();

    const std::int64_t incx = x_stride[0];
    const std::int64_t incy = y_stride[0];
    if (incx < 0) {
        x_typeless_ptr -= (n - 1) * std::abs(incx) * x_elemsize;
    }
    if (incy < 0) {
        y_typeless_ptr -= (n - 1) * std::abs(incy) * y_elemsize;
    }

    sycl::event dotu_ev =
        dotu_fn(exec_q, n, x_typeless_ptr, incx, y_typeless_ptr, incy,
                r_typeless_ptr, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {vectorX, vectorY, result}, {dotu_ev});

    return std::make_pair(args_ev, dotu_ev);
}

template <typename fnT, typename Tab, typename Tc>
struct DotuContigFactory
{
    fnT get()
    {
        if constexpr (types::DotuTypePairSupportFactory<Tab, Tc>::is_defined) {
            return dotu_impl<Tab, Tc>;
        }
        else {
            return nullptr;
        }
    }
};

void init_dotu_dispatch_table(void)
{
    dpctl_td_ns::DispatchTableBuilder<dotu_impl_fn_ptr_t, DotuContigFactory,
                                      dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_table(dotu_dispatch_table);
}
} // namespace blas
} // namespace ext
} // namespace backend
} // namespace dpnp
