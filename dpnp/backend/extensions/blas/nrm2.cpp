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

#include "nrm2.hpp"
#include "types_matrix.hpp"

#include "dpnp_utils.hpp"

namespace dpnp::extensions::blas
{
namespace mkl_blas = oneapi::mkl::blas;
namespace py = pybind11;
namespace type_utils = dpctl::tensor::type_utils;

template <typename Tab, typename Tc>
static sycl::event nrm2_impl(sycl::queue &exec_q,
                             const std::int64_t n,
                             const char *vectorX,
                             const std::int64_t stride_x,
                             char *result,
                             const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<Tab>(exec_q);
    type_utils::validate_type_for_device<Tc>(exec_q);

    const Tab *x = reinterpret_cast<const Tab *>(vectorX);
    Tc *res = reinterpret_cast<Tc *>(result);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event nrm2_event;
    try {
        nrm2_event = mkl_blas::row_major::nrm2(exec_q,
                                               n, // size of the input vectors
                                               x, // Pointer to vector a.
                                               stride_x, // Stride of vector a.
                                               res,      // Pointer to result.
                                               depends);
    } catch (oneapi::mkl::exception const &e) {
        error_msg
            << "Unexpected MKL exception caught during nrm2() call:\nreason: "
            << e.what();
        is_exception_caught = true;
    } catch (sycl::exception const &e) {
        error_msg << "Unexpected SYCL exception caught during nrm2() call:\n"
                  << e.what();
        is_exception_caught = true;
    }

    if (is_exception_caught) // an unexpected error occurs
    {
        throw std::runtime_error(error_msg.str());
    }

    return nrm2_event;
}

std::pair<sycl::event, sycl::event>
    nrm2(sycl::queue &exec_q,
         const dpctl::tensor::usm_ndarray &vectorX,
         const dpctl::tensor::usm_ndarray &result,
         const std::vector<sycl::event> &depends)
{
    const int vectorX_nd = vectorX.get_ndim();
    const int result_nd = result.get_ndim();

    if ((vectorX_nd != 1)) {
        throw py::value_error(
            "The input array has ndim=" + std::to_string(vectorX_nd) +
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

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(
            exec_q, {vectorX.get_queue(), result.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    // check writable
    // check ample memory
    const py::ssize_t x_size = vectorX.get_size();
    const std::vector<py::ssize_t> x_stride = vectorX.get_strides_vector();

    const std::int64_t n = x_size;
    const std::int64_t str_x = x_stride[0];

    const int vectorX_typenum = vectorX.get_typenum();
    const int result_typenum = result.get_typenum();

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    const int vectorX_type_id =
        array_types.typenum_to_lookup_id(vectorX_typenum);
    const int result_type_id = array_types.typenum_to_lookup_id(result_typenum);

    nrm2_impl_fn_ptr_t nrm2_fn =
        nrm2_dispatch_table[vectorX_type_id][result_type_id];
    if (nrm2_fn == nullptr) {
        throw py::value_error(
            "Types of input vector and result array are mismatched.");
    }

    char *x_typeless_ptr = vectorX.get_data();
    char *r_typeless_ptr = result.get_data();

    const int x_elemsize = vectorX.get_elemsize();
    if (str_x < 0) {
        x_typeless_ptr -= (n - 1) * std::abs(str_x) * x_elemsize;
    }

    sycl::event nrm2_ev =
        nrm2_fn(exec_q, n, x_typeless_ptr, str_x, r_typeless_ptr, depends);

    sycl::event args_ev =
        dpctl::utils::keep_args_alive(exec_q, {vectorX, result}, {nrm2_ev});

    return std::make_pair(args_ev, nrm2_ev);
}

template <typename fnT, typename Tab, typename Tc>
struct Nrm2ContigFactory
{
    fnT get()
    {
        if constexpr (types::Nrm2TypePairSupportFactory<Tab, Tc>::is_defined) {
            return nrm2_impl<Tab, Tc>;
        }
        else {
            return nullptr;
        }
    }
};

void init_nrm2_dispatch_table(void)
{
    dpctl_td_ns::DispatchTableBuilder<nrm2_impl_fn_ptr_t, Nrm2ContigFactory,
                                      dpctl_td_ns::num_types>
        contig;
    contig.populate_dispatch_table(nrm2_dispatch_table);
}
} // namespace dpnp::extensions::blas
