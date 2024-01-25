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

typedef sycl::event (*dotu_impl_fn_ptr_t)(sycl::queue,
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
static sycl::event dotu_impl(sycl::queue exec_q,
                             const std::int64_t n,
                             char *vectorA,
                             const std::int64_t stride_a,
                             char *vectorB,
                             const std::int64_t stride_b,
                             char *result,
                             const std::vector<sycl::event> &depends)
{
    type_utils::validate_type_for_device<Tab>(exec_q);
    type_utils::validate_type_for_device<Tc>(exec_q);

    Tab *a = reinterpret_cast<Tab *>(vectorA);
    Tab *b = reinterpret_cast<Tab *>(vectorB);
    Tc *res = reinterpret_cast<Tc *>(result);

    std::stringstream error_msg;
    bool is_exception_caught = false;

    sycl::event dotu_event;
    try {
        dotu_event = mkl_blas::row_major::dotu(exec_q,
                                               n, // size of the input vectors
                                               a, // Pointer to vector a.
                                               stride_a, // Stride of vector a.
                                               b,        // Pointer to vector b.
                                               stride_b, // Stride of vector b.
                                               res,      // Pointer to result.
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
    dotu(sycl::queue exec_q,
         dpctl::tensor::usm_ndarray vectorA,
         dpctl::tensor::usm_ndarray vectorB,
         dpctl::tensor::usm_ndarray result,
         const std::vector<sycl::event> &depends)
{
    const int vectorA_nd = vectorA.get_ndim();
    const int vectorB_nd = vectorB.get_ndim();
    const int result_nd = result.get_ndim();

    if ((vectorA_nd != 1)) {
        throw py::value_error(
            "The first input array has ndim=" + std::to_string(vectorA_nd) +
            ", but a 1-dimensional array is expected.");
    }

    if ((vectorB_nd != 1)) {
        throw py::value_error(
            "The second input array has ndim=" + std::to_string(vectorB_nd) +
            ", but a 1-dimensional array is expected.");
    }

    if ((result_nd != 0)) {
        throw py::value_error(
            "The output array has ndim=" + std::to_string(result_nd) +
            ", but a 0-dimensional array is expected.");
    }

    auto const &overlap = dpctl::tensor::overlap::MemoryOverlap();
    if (overlap(vectorA, result)) {
        throw py::value_error(
            "The first input array and output array are overlapping "
            "segments of memory");
    }
    if (overlap(vectorB, result)) {
        throw py::value_error(
            "The second input array and output array are overlapping "
            "segments of memory");
    }

    // check compatibility of execution queue and allocation queue
    if (!dpctl::utils::queues_are_compatible(
            exec_q,
            {vectorA.get_queue(), vectorB.get_queue(), result.get_queue()}))
    {
        throw py::value_error(
            "USM allocations are not compatible with the execution queue.");
    }

    py::ssize_t a_size = vectorA.get_size();
    py::ssize_t b_size = vectorB.get_size();
    if (a_size != b_size) {
        throw py::value_error("The size of the first input array must be "
                              "equal to the size of the second input array.");
    }

    std::vector<py::ssize_t> a_stride = vectorA.get_strides_vector();
    std::vector<py::ssize_t> b_stride = vectorB.get_strides_vector();

    const std::int64_t n = a_size;
    const std::int64_t str_a = a_stride[0];
    const std::int64_t str_b = b_stride[0];

    int vectorA_typenum = vectorA.get_typenum();
    int vectorB_typenum = vectorB.get_typenum();
    int result_typenum = result.get_typenum();

    if (vectorA_typenum != vectorB_typenum) {
        throw py::value_error(
            "Input arrays must be of must be of the same type.");
    }

    auto array_types = dpctl_td_ns::usm_ndarray_types();
    int vectorAB_type_id = array_types.typenum_to_lookup_id(vectorA_typenum);
    int result_type_id = array_types.typenum_to_lookup_id(result_typenum);

    dotu_impl_fn_ptr_t dotu_fn =
        dotu_dispatch_table[vectorAB_type_id][result_type_id];
    if (dotu_fn == nullptr) {
        throw py::value_error(
            "Types of input vectors and result array are mismatched.");
    }

    char *a_typeless_ptr = vectorA.get_data();
    char *b_typeless_ptr = vectorB.get_data();
    char *r_typeless_ptr = result.get_data();

    sycl::event dotu_ev =
        dotu_fn(exec_q, n, a_typeless_ptr, str_a, b_typeless_ptr, str_b,
                r_typeless_ptr, depends);

    sycl::event args_ev = dpctl::utils::keep_args_alive(
        exec_q, {vectorA, vectorB, result}, {dotu_ev});

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
