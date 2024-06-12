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

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"

#include "ceil.hpp"
#include "common.hpp"

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
namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
namespace py = pybind11;
namespace py_int = dpnp::extensions::py_internal;
namespace td_ns = dpctl::tensor::type_dispatch;
namespace tu_ns = dpctl::tensor::type_utils;

namespace impl
{
// OneMKL namespace with VM functions
namespace mkl_vm = oneapi::mkl::vm;

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support in oneapi::mkl::vm::ceil<T> function.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct OutputType
{
    using value_type =
        typename std::disjunction<td_ns::TypeMapResultEntry<T, double>,
                                  td_ns::TypeMapResultEntry<T, float>,
                                  td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T>
static sycl::event ceil_contig_impl(sycl::queue &exec_q,
                                    std::size_t in_n,
                                    const char *in_a,
                                    char *out_y,
                                    const std::vector<sycl::event> &depends)
{
    tu_ns::validate_type_for_device<T>(exec_q);

    std::int64_t n = static_cast<std::int64_t>(in_n);
    const T *a = reinterpret_cast<const T *>(in_a);

    using resTy = typename OutputType<T>::value_type;
    resTy *y = reinterpret_cast<resTy *>(out_y);

    return mkl_vm::ceil(exec_q,
                        n, // number of elements to be calculated
                        a, // pointer `a` containing input vector of size n
                        y, // pointer `y` to the output vector of size n
                        depends);
}

using ew_cmn_ns::unary_contig_impl_fn_ptr_t;
using ew_cmn_ns::unary_strided_impl_fn_ptr_t;

static int output_typeid_vector[td_ns::num_types];
static unary_contig_impl_fn_ptr_t contig_dispatch_vector[td_ns::num_types];

MACRO_POPULATE_DISPATCH_VECTORS(ceil);
} // namespace impl

void init_ceil(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;

    impl::populate_dispatch_vectors();
    using impl::contig_dispatch_vector;
    using impl::output_typeid_vector;

    auto ceil_pyapi = [&](sycl::queue &exec_q, const arrayT &src,
                          const arrayT &dst, const event_vecT &depends = {}) {
        return py_int::py_unary_ufunc(
            src, dst, exec_q, depends, output_typeid_vector,
            contig_dispatch_vector,
            // no support of strided implementation in OneMKL
            td_ns::NullPtrVector<impl::unary_strided_impl_fn_ptr_t>{});
    };
    m.def("_ceil", ceil_pyapi,
          "Call `ceil` function from OneMKL VM library to compute "
          "the ceiling of vector elements",
          py::arg("sycl_queue"), py::arg("src"), py::arg("dst"),
          py::arg("depends") = py::list());

    auto ceil_need_to_call_pyapi = [&](sycl::queue &exec_q, const arrayT &src,
                                       const arrayT &dst) {
        return py_internal::need_to_call_unary_ufunc(
            exec_q, src, dst, output_typeid_vector, contig_dispatch_vector);
    };
    m.def("_mkl_ceil_to_call", ceil_need_to_call_pyapi,
          "Check input arguments to answer if `ceil` function from "
          "OneMKL VM library can be used",
          py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));
}
} // namespace dpnp::extensions::vm
