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
#include "erf_funcs.hpp"

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

using ext::common::init_dispatch_vector;

namespace impl
{
namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
namespace mkl_vm = oneapi::mkl::vm; // OneMKL namespace with VM functions
namespace tu_ns = dpctl::tensor::type_utils;

/**
 * @brief A factory to define pairs of supported types for which
 * MKL VM library provides support for a family of erf-like functions.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct OutputType
{
    using value_type =
        typename std::disjunction<td_ns::TypeMapResultEntry<T, float>,
                                  td_ns::TypeMapResultEntry<T, double>,
                                  td_ns::DefaultResultEntry<void>>::result_type;
};

static int output_typeid_vector[td_ns::num_types];

template <typename fnT, typename T>
struct TypeMapFactory
{
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename OutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

using ew_cmn_ns::unary_contig_impl_fn_ptr_t;

#define MACRO_DEFINE_IMPL(__name__, __f_name__)                                \
    template <typename T>                                                      \
    static sycl::event __name__##_contig_impl(                                 \
        sycl::queue &exec_q, std::size_t in_n, const char *in_a, char *out_y,  \
        const std::vector<sycl::event> &depends)                               \
    {                                                                          \
        tu_ns::validate_type_for_device<T>(exec_q);                            \
                                                                               \
        std::int64_t n = static_cast<std::int64_t>(in_n);                      \
        const T *a = reinterpret_cast<const T *>(in_a);                        \
                                                                               \
        using resTy = typename OutputType<T>::value_type;                      \
        resTy *y = reinterpret_cast<resTy *>(out_y);                           \
                                                                               \
        return mkl_vm::__name__(                                               \
            exec_q, n, /* number of elements to be calculated*/                \
            a,         /* pointer `a` containing input vector of size n*/      \
            y,         /* pointer `y` to the output vector of size n*/         \
            depends);                                                          \
    }                                                                          \
                                                                               \
    static unary_contig_impl_fn_ptr_t                                          \
        __name__##_contig_dispatch_vector[td_ns::num_types];                   \
                                                                               \
    template <typename fnT, typename T>                                        \
    struct __f_name__##ContigFactory                                           \
    {                                                                          \
        fnT get()                                                              \
        {                                                                      \
            if constexpr (std::is_same_v<typename OutputType<T>::value_type,   \
                                         void>) {                              \
                return nullptr;                                                \
            }                                                                  \
            else {                                                             \
                return __name__##_contig_impl<T>;                              \
            }                                                                  \
        }                                                                      \
    };

MACRO_DEFINE_IMPL(erf, Erf);
MACRO_DEFINE_IMPL(erfc, Erfc);
MACRO_DEFINE_IMPL(erfcx, Erfcx);
MACRO_DEFINE_IMPL(erfinv, Erfinv);
MACRO_DEFINE_IMPL(erfcinv, Erfcinv);

template <template <typename fnT, typename T> typename factoryT>
static void populate(py::module_ m,
                     const char *name,
                     const char *docstring,
                     unary_contig_impl_fn_ptr_t *contig_dispatch_vector)
{
    init_dispatch_vector<unary_contig_impl_fn_ptr_t, factoryT>(
        contig_dispatch_vector);

    using arrayT = dpctl::tensor::usm_ndarray;
    auto pyapi = [&, contig_dispatch_vector](
                     sycl::queue &exec_q, const arrayT &src, const arrayT &dst,
                     const std::vector<sycl::event> &depends = {}) {
        return py_int::py_unary_ufunc(
            src, dst, exec_q, depends, output_typeid_vector,
            contig_dispatch_vector,
            // no support of strided implementation in OneMKL
            td_ns::NullPtrVector<ew_cmn_ns::unary_strided_impl_fn_ptr_t>{});
    };
    m.def(name, pyapi, docstring, py::arg("sycl_queue"), py::arg("src"),
          py::arg("dst"), py::arg("depends") = py::list());
}
} // namespace impl

void init_erf_funcs(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using impl::output_typeid_vector;

    init_dispatch_vector<int, impl::TypeMapFactory>(output_typeid_vector);

    auto erf_need_to_call_pyapi = [&](sycl::queue &exec_q, const arrayT &src,
                                      const arrayT &dst) {
        return py_internal::need_to_call_unary_ufunc(
            exec_q, src, dst, output_typeid_vector,
            impl::erf_contig_dispatch_vector);
    };
    m.def("_mkl_erf_to_call", erf_need_to_call_pyapi,
          "Check input arguments to answer if any erf-like function from "
          "OneMKL VM library can be used",
          py::arg("sycl_queue"), py::arg("src"), py::arg("dst"));

    impl::populate<impl::ErfContigFactory>(
        m, "_erf",
        "Call `erf` function from OneMKL VM library to compute the error "
        "function value of vector elements",
        impl::erf_contig_dispatch_vector);

    impl::populate<impl::ErfcContigFactory>(
        m, "_erfc",
        "Call `erfc` function from OneMKL VM library to compute the "
        "complementary error function value of vector elements",
        impl::erfc_contig_dispatch_vector);

    impl::populate<impl::ErfcxContigFactory>(
        m, "_erfcx",
        "Call `erfcx` function from OneMKL VM library to compute the scaled "
        "complementary error function value of vector elements",
        impl::erfcx_contig_dispatch_vector);

    impl::populate<impl::ErfinvContigFactory>(
        m, "_erfinv",
        "Call `erfinv` function from OneMKL VM library to compute the inverse "
        "of the error function value of vector elements",
        impl::erfinv_contig_dispatch_vector);

    impl::populate<impl::ErfcinvContigFactory>(
        m, "_erfcinv",
        "Call `erfcinv` function from OneMKL VM library to compute the inverse "
        "of the complementary error function value of vector elements",
        impl::erfcinv_contig_dispatch_vector);
}
} // namespace dpnp::extensions::vm
