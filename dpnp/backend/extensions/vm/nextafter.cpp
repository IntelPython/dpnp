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

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

#include "common.hpp"
#include "nextafter.hpp"

// include a local copy of elementwise common header from dpctl tensor:
// dpctl/tensor/libtensor/source/elementwise_functions/elementwise_functions.hpp
// TODO: replace by including dpctl header once available
#include "../elementwise_functions/elementwise_functions.hpp"

// dpctl tensor headers
#include "kernels/elementwise_functions/common.hpp"
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::extensions::vm
{
namespace py = pybind11;
namespace py_int = dpnp::extensions::py_internal;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace impl
{
namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
namespace mkl_vm = oneapi::mkl::vm; // OneMKL namespace with VM functions
namespace tu_ns = dpctl::tensor::type_utils;

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::nextafter<T> function.
 *
 * @tparam T Type of input vectors `a` and `b` and of result vector `y`.
 */
template <typename T1, typename T2>
struct OutputType
{
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2>
static sycl::event
    nextafter_contig_impl(sycl::queue &exec_q,
                          std::size_t in_n,
                          const char *in_a,
                          py::ssize_t a_offset,
                          const char *in_b,
                          py::ssize_t b_offset,
                          char *out_y,
                          py::ssize_t out_offset,
                          const std::vector<sycl::event> &depends)
{
    tu_ns::validate_type_for_device<T1>(exec_q);
    tu_ns::validate_type_for_device<T2>(exec_q);

    if ((a_offset != 0) || (b_offset != 0) || (out_offset != 0)) {
        throw std::runtime_error("Arrays offsets have to be equals to 0");
    }

    std::int64_t n = static_cast<std::int64_t>(in_n);
    const T1 *a = reinterpret_cast<const T1 *>(in_a);
    const T2 *b = reinterpret_cast<const T2 *>(in_b);

    using resTy = typename OutputType<T1, T2>::value_type;
    resTy *y = reinterpret_cast<resTy *>(out_y);

    return mkl_vm::nextafter(
        exec_q,
        n, // number of elements to be calculated
        a, // pointer `a` containing 1st input vector of size n
        b, // pointer `b` containing 2nd input vector of size n
        y, // pointer `y` to the output vector of size n
        depends);
}

using ew_cmn_ns::binary_contig_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t;
using ew_cmn_ns::binary_strided_impl_fn_ptr_t;

static int output_typeid_vector[td_ns::num_types][td_ns::num_types];
static binary_contig_impl_fn_ptr_t contig_dispatch_vector[td_ns::num_types]
                                                         [td_ns::num_types];

MACRO_POPULATE_DISPATCH_TABLES(nextafter);
} // namespace impl

void init_nextafter(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    impl::populate_dispatch_tables();
    using impl::contig_dispatch_vector;
    using impl::output_typeid_vector;

    auto nextafter_pyapi = [&](sycl::queue &exec_q, const arrayT &src1,
                               const arrayT &src2, const arrayT &dst,
                               const event_vecT &depends = {}) {
        return py_int::py_binary_ufunc(
            src1, src2, dst, exec_q, depends, output_typeid_vector,
            contig_dispatch_vector,
            // no support of strided implementation in OneMKL
            td_ns::NullPtrTable<impl::binary_strided_impl_fn_ptr_t>{},
            // no support of C-contig row with broadcasting in OneMKL
            td_ns::NullPtrTable<
                impl::
                    binary_contig_matrix_contig_row_broadcast_impl_fn_ptr_t>{},
            td_ns::NullPtrTable<
                impl::
                    binary_contig_row_contig_matrix_broadcast_impl_fn_ptr_t>{});
    };
    m.def(
        "_nextafter", nextafter_pyapi,
        "Call `nextafter` function from OneMKL VM library to return `dst` of "
        "elements containing the next representable floating-point values "
        "following the values from the elements of `src1` in the direction of "
        "the corresponding elements of `src2`",
        py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"), py::arg("dst"),
        py::arg("depends") = py::list());

    auto nextafter_need_to_call_pyapi = [&](sycl::queue &exec_q,
                                            const arrayT &src1,
                                            const arrayT &src2,
                                            const arrayT &dst) {
        return py_internal::need_to_call_binary_ufunc(exec_q, src1, src2, dst,
                                                      output_typeid_vector,
                                                      contig_dispatch_vector);
    };
    m.def("_mkl_nextafter_to_call", nextafter_need_to_call_pyapi,
          "Check input arguments to answer if `nextafter` function from "
          "OneMKL VM library can be used",
          py::arg("sycl_queue"), py::arg("src1"), py::arg("src2"),
          py::arg("dst"));
}
} // namespace dpnp::extensions::vm
