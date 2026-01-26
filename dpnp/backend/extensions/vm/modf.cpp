//*****************************************************************************
// Copyright (c) 2025, Intel Corporation
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

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dpnp4pybind11.hpp"

#include "common.hpp"
#include "modf.hpp"

// include a local copy of elementwise common header from dpctl tensor:
// dpctl/tensor/libtensor/source/elementwise_functions/elementwise_functions.hpp
// TODO: replace by including dpctl header once available
#include "../elementwise_functions/elementwise_functions.hpp"

#include "../elementwise_functions/common.hpp"
#include "../elementwise_functions/type_dispatch_building.hpp"

// dpctl tensor headers
#include "utils/type_dispatch.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::extensions::vm
{
namespace py = pybind11;
namespace py_int = dpnp::extensions::py_internal;
namespace td_ns = dpctl::tensor::type_dispatch;

namespace impl
{
namespace ew_cmn_ns = dpnp::extensions::py_internal::elementwise_common;
namespace mkl_vm = oneapi::mkl::vm; // OneMKL namespace with VM functions
namespace td_int_ns = py_int::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::modf<T> function.
 *
 * @tparam T Type of input vector `a` and of result vectors `y` and `z`.
 */
template <typename T>
struct OutputType
{
    using table_type =
        std::disjunction<td_int_ns::TypeMapTwoResultsEntry<T, sycl::half>,
                         td_int_ns::TypeMapTwoResultsEntry<T, float>,
                         td_int_ns::TypeMapTwoResultsEntry<T, double>,
                         td_int_ns::DefaultTwoResultsEntry<void>>;
    using value_type1 = typename table_type::result_type1;
    using value_type2 = typename table_type::result_type2;
};

template <typename T>
static sycl::event modf_contig_impl(sycl::queue &exec_q,
                                    std::size_t in_n,
                                    const char *in_a,
                                    char *out_y,
                                    char *out_z,
                                    const std::vector<sycl::event> &depends)
{
    tu_ns::validate_type_for_device<T>(exec_q);

    std::int64_t n = static_cast<std::int64_t>(in_n);
    const T *a = reinterpret_cast<const T *>(in_a);

    using fractT = typename OutputType<T>::value_type1;
    using intT = typename OutputType<T>::value_type2;
    fractT *y = reinterpret_cast<fractT *>(out_y);
    intT *z = reinterpret_cast<intT *>(out_z);

    return mkl_vm::modf(exec_q,
                        n, // number of elements to be calculated
                        a, // pointer `a` containing input vector of size n
                        z, // pointer `z` to the output truncated integer values
                        y, // pointer `y` to the output remaining fraction parts
                        depends);
}

using ew_cmn_ns::unary_two_outputs_contig_impl_fn_ptr_t;
using ew_cmn_ns::unary_two_outputs_strided_impl_fn_ptr_t;

static std::pair<int, int> output_typeid_vector[td_ns::num_types];
static unary_two_outputs_contig_impl_fn_ptr_t
    contig_dispatch_vector[td_ns::num_types];

MACRO_POPULATE_DISPATCH_2OUTS_VECTORS(modf);
} // namespace impl

void init_modf(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    impl::populate_dispatch_vectors();
    using impl::contig_dispatch_vector;
    using impl::output_typeid_vector;

    auto modf_pyapi = [&](sycl::queue &exec_q, const arrayT &src,
                          const arrayT &dst1, const arrayT &dst2,
                          const event_vecT &depends = {}) {
        return py_int::py_unary_two_outputs_ufunc(
            src, dst1, dst2, exec_q, depends, output_typeid_vector,
            contig_dispatch_vector,
            // no support of strided implementation in OneMKL
            td_ns::NullPtrVector<
                impl::unary_two_outputs_strided_impl_fn_ptr_t>{});
    };
    m.def("_modf", modf_pyapi,
          "Call `modf` function from OneMKL VM library to compute "
          "a truncated integer value and the remaining fraction part "
          "for each vector elements",
          py::arg("sycl_queue"), py::arg("src"), py::arg("dst1"),
          py::arg("dst2"), py::arg("depends") = py::list());

    auto modf_need_to_call_pyapi = [&](sycl::queue &exec_q, const arrayT &src,
                                       const arrayT &dst1, const arrayT &dst2) {
        return py_internal::need_to_call_unary_two_outputs_ufunc(
            exec_q, src, dst1, dst2, output_typeid_vector,
            contig_dispatch_vector);
    };
    m.def("_mkl_modf_to_call", modf_need_to_call_pyapi,
          "Check input arguments to answer if `modf` function from "
          "OneMKL VM library can be used",
          py::arg("sycl_queue"), py::arg("src"), py::arg("dst1"),
          py::arg("dst2"));
}
} // namespace dpnp::extensions::vm
