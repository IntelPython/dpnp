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

#include <type_traits>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sycl/sycl.hpp>

#include "dpnp4pybind11.hpp"

#include "erf_funcs.hpp"
#include "kernels/elementwise_functions/erf.hpp"

// utils extension header
#include "ext/common.hpp"

// include a local copy of elementwise common header from dpctl tensor:
// dpctl/tensor/libtensor/source/elementwise_functions/elementwise_functions.hpp
// TODO: replace by including dpctl header once available
#include "../../elementwise_functions/elementwise_functions.hpp"

// dpctl tensor headers
#include "kernels/elementwise_functions/common.hpp"
#include "utils/type_dispatch.hpp"

namespace dpnp::extensions::ufunc
{
namespace py = pybind11;
namespace py_int = dpnp::extensions::py_internal;

using ext::common::init_dispatch_vector;

namespace impl
{
namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
namespace td_ns = dpctl::tensor::type_dispatch;

/**
 * @brief A factory to define pairs of supported types for which
 * an erf-like function in SYCL namespace is available.
 *
 * @tparam T Type of input vector `a` and of result vector `y`.
 */
template <typename T>
struct OutputType
{
    /**
     * scipy>=1.16 assumes a pair 'e->d', but dpnp 'e->f' without an extra
     * kernel 'e->d' (when fp64 supported) to reduce memory footprint
     */
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, sycl::half, float>,
        td_ns::TypeMapResultEntry<T, float>,
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
using ew_cmn_ns::unary_strided_impl_fn_ptr_t;

#define MACRO_DEFINE_IMPL(__name__, __f_name__)                                \
                                                                               \
    using dpnp::kernels::erfs::__f_name__##Functor;                            \
                                                                               \
    template <typename argT, typename resT = argT, unsigned int vec_sz = 4,    \
              unsigned int n_vecs = 2, bool enable_sg_loadstore = true>        \
    using __f_name__##ContigFunctor =                                          \
        ew_cmn_ns::UnaryContigFunctor<argT, resT,                              \
                                      __f_name__##Functor<argT, resT>, vec_sz, \
                                      n_vecs, enable_sg_loadstore>;            \
                                                                               \
    template <typename argTy, typename resTy, typename IndexerT>               \
    using __f_name__##StridedFunctor =                                         \
        ew_cmn_ns::UnaryStridedFunctor<argTy, resTy, IndexerT,                 \
                                       __f_name__##Functor<argTy, resTy>>;     \
                                                                               \
    static unary_contig_impl_fn_ptr_t                                          \
        __name__##_contig_dispatch_vector[td_ns::num_types];                   \
    static unary_strided_impl_fn_ptr_t                                         \
        __name__##_strided_dispatch_vector[td_ns::num_types];                  \
                                                                               \
    template <typename T1, typename T2, unsigned int vec_sz,                   \
              unsigned int n_vecs>                                             \
    class __name__##_contig_kernel;                                            \
                                                                               \
    template <typename argTy>                                                  \
    sycl::event __name__##_contig_impl(                                        \
        sycl::queue &exec_q, size_t nelems, const char *arg_p, char *res_p,    \
        const std::vector<sycl::event> &depends = {})                          \
    {                                                                          \
        return ew_cmn_ns::unary_contig_impl<argTy, OutputType,                 \
                                            __f_name__##ContigFunctor,         \
                                            __name__##_contig_kernel>(         \
            exec_q, nelems, arg_p, res_p, depends);                            \
    }                                                                          \
                                                                               \
    template <typename fnT, typename T>                                        \
    struct __f_name__##ContigFactory                                           \
    {                                                                          \
        fnT get()                                                              \
        {                                                                      \
            if constexpr (std::is_same_v<typename OutputType<T>::value_type,   \
                                         void>) {                              \
                fnT fn = nullptr;                                              \
                return fn;                                                     \
            }                                                                  \
            else {                                                             \
                fnT fn = __name__##_contig_impl<T>;                            \
                return fn;                                                     \
            }                                                                  \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <typename T1, typename T2, typename T3>                           \
    class __name__##_strided_kernel;                                           \
                                                                               \
    template <typename argTy>                                                  \
    sycl::event __name__##_strided_impl(                                       \
        sycl::queue &exec_q, size_t nelems, int nd,                            \
        const py::ssize_t *shape_and_strides, const char *arg_p,               \
        py::ssize_t arg_offset, char *res_p, py::ssize_t res_offset,           \
        const std::vector<sycl::event> &depends,                               \
        const std::vector<sycl::event> &additional_depends)                    \
    {                                                                          \
        return ew_cmn_ns::unary_strided_impl<argTy, OutputType,                \
                                             __f_name__##StridedFunctor,       \
                                             __name__##_strided_kernel>(       \
            exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,   \
            res_offset, depends, additional_depends);                          \
    }                                                                          \
                                                                               \
    template <typename fnT, typename T>                                        \
    struct __f_name__##StridedFactory                                          \
    {                                                                          \
        fnT get()                                                              \
        {                                                                      \
            if constexpr (std::is_same_v<typename OutputType<T>::value_type,   \
                                         void>) {                              \
                fnT fn = nullptr;                                              \
                return fn;                                                     \
            }                                                                  \
            else {                                                             \
                fnT fn = __name__##_strided_impl<T>;                           \
                return fn;                                                     \
            }                                                                  \
        }                                                                      \
    };

template <template <typename fnT, typename T> typename contigFactoryT,
          template <typename fnT, typename T>
          typename stridedFactoryT>
static void populate(py::module_ m,
                     const char *name,
                     const char *docstring,
                     unary_contig_impl_fn_ptr_t *contig_dispatch_vector,
                     unary_strided_impl_fn_ptr_t *strided_dispatch_vector)
{
    init_dispatch_vector<unary_contig_impl_fn_ptr_t, contigFactoryT>(
        contig_dispatch_vector);

    init_dispatch_vector<unary_strided_impl_fn_ptr_t, stridedFactoryT>(
        strided_dispatch_vector);

    using arrayT = dpctl::tensor::usm_ndarray;
    auto pyapi = [&, contig_dispatch_vector, strided_dispatch_vector](
                     const arrayT &src, const arrayT &dst, sycl::queue &exec_q,
                     const std::vector<sycl::event> &depends = {}) {
        return py_int::py_unary_ufunc(
            src, dst, exec_q, depends, output_typeid_vector,
            contig_dispatch_vector, strided_dispatch_vector);
    };
    m.def(name, pyapi, docstring, py::arg("sycl_queue"), py::arg("src"),
          py::arg("dst"), py::arg("depends") = py::list());
}

MACRO_DEFINE_IMPL(erf, Erf);
MACRO_DEFINE_IMPL(erfc, Erfc);
MACRO_DEFINE_IMPL(erfcx, Erfcx);
MACRO_DEFINE_IMPL(erfinv, Erfinv);
MACRO_DEFINE_IMPL(erfcinv, Erfcinv);
} // namespace impl

void init_erf_funcs(py::module_ m)
{
    using impl::output_typeid_vector;
    init_dispatch_vector<int, impl::TypeMapFactory>(output_typeid_vector);

    auto erf_result_type_pyapi = [&](const py::dtype &dtype) {
        return py_int::py_unary_ufunc_result_type(dtype, output_typeid_vector);
    };
    m.def("_erf_result_type", erf_result_type_pyapi);

    impl::populate<impl::ErfContigFactory, impl::ErfStridedFactory>(
        m, "_erf", "", impl::erf_contig_dispatch_vector,
        impl::erf_strided_dispatch_vector);

    impl::populate<impl::ErfcContigFactory, impl::ErfcStridedFactory>(
        m, "_erfc", "", impl::erfc_contig_dispatch_vector,
        impl::erfc_strided_dispatch_vector);

    impl::populate<impl::ErfcxContigFactory, impl::ErfcxStridedFactory>(
        m, "_erfcx", "", impl::erfcx_contig_dispatch_vector,
        impl::erfcx_strided_dispatch_vector);

    impl::populate<impl::ErfinvContigFactory, impl::ErfinvStridedFactory>(
        m, "_erfinv", "", impl::erfinv_contig_dispatch_vector,
        impl::erfinv_strided_dispatch_vector);

    impl::populate<impl::ErfcinvContigFactory, impl::ErfcinvStridedFactory>(
        m, "_erfcinv", "", impl::erfcinv_contig_dispatch_vector,
        impl::erfcinv_strided_dispatch_vector);
}
} // namespace dpnp::extensions::ufunc
