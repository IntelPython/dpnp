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

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "dpctl4pybind11.hpp"

#include "frexp.hpp"
#include "kernels/elementwise_functions/frexp.hpp"
#include "populate.hpp"

// include a local copy of elementwise common header from dpctl tensor:
// dpctl/tensor/libtensor/source/elementwise_functions/elementwise_functions.hpp
// TODO: replace by including dpctl header once available
#include "../../elementwise_functions/elementwise_functions.hpp"

#include "../../elementwise_functions/common.hpp"
#include "../../elementwise_functions/type_dispatch_building.hpp"

// utils extension header
#include "ext/common.hpp"

// dpctl tensor headers
#include "kernels/elementwise_functions/common.hpp"
#include "utils/type_dispatch.hpp"

namespace dpnp::extensions::ufunc
{
namespace py = pybind11;
namespace py_int = dpnp::extensions::py_internal;

namespace impl
{
namespace ew_cmn_ns = dpnp::extensions::py_internal::elementwise_common;
namespace td_int_ns = py_int::type_dispatch;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpnp::kernels::frexp::FrexpFunctor;
using ext::common::init_dispatch_vector;

template <typename T>
struct FrexpOutputType
{
    using table_type = std::disjunction< // disjunction is C++17
                                         // feature, supported by DPC++
        td_int_ns::
            TypeMapTwoResultsEntry<T, sycl::half, sycl::half, std::int32_t>,
        td_int_ns::TypeMapTwoResultsEntry<T, float, float, std::int32_t>,
        td_int_ns::TypeMapTwoResultsEntry<T, double, double, std::int32_t>,
        td_int_ns::DefaultTwoResultsEntry<void>>;
    using value_type1 = typename table_type::result_type1;
    using value_type2 = typename table_type::result_type2;
};

// contiguous implementation

template <typename argTy,
          typename resTy1 = argTy,
          typename resTy2 = argTy,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using FrexpContigFunctor =
    ew_cmn_ns::UnaryTwoOutputsContigFunctor<argTy,
                                            resTy1,
                                            resTy2,
                                            FrexpFunctor<argTy, resTy1, resTy2>,
                                            vec_sz,
                                            n_vecs,
                                            enable_sg_loadstore>;

// strided implementation

template <typename argTy, typename resTy1, typename resTy2, typename IndexerT>
using FrexpStridedFunctor = ew_cmn_ns::UnaryTwoOutputsStridedFunctor<
    argTy,
    resTy1,
    resTy2,
    IndexerT,
    FrexpFunctor<argTy, resTy1, resTy2>>;

template <typename T1,
          typename T2,
          typename T3,
          unsigned int vec_sz,
          unsigned int n_vecs>
class frexp_contig_kernel;

template <typename argTy>
sycl::event frexp_contig_impl(sycl::queue &exec_q,
                              size_t nelems,
                              const char *arg_p,
                              char *res1_p,
                              char *res2_p,
                              const std::vector<sycl::event> &depends = {})
{
    return ew_cmn_ns::unary_two_outputs_contig_impl<
        argTy, FrexpOutputType, FrexpContigFunctor, frexp_contig_kernel>(
        exec_q, nelems, arg_p, res1_p, res2_p, depends);
}

template <typename fnT, typename T>
struct FrexpContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename FrexpOutputType<T>::value_type1,
                                     void> ||
                      std::is_same_v<typename FrexpOutputType<T>::value_type2,
                                     void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = frexp_contig_impl<T>;
            return fn;
        }
    }
};

template <typename T1, typename T2, typename T3, typename T4>
class frexp_strided_kernel;

template <typename argTy>
sycl::event
    frexp_strided_impl(sycl::queue &exec_q,
                       size_t nelems,
                       int nd,
                       const ssize_t *shape_and_strides,
                       const char *arg_p,
                       ssize_t arg_offset,
                       char *res1_p,
                       ssize_t res1_offset,
                       char *res2_p,
                       ssize_t res2_offset,
                       const std::vector<sycl::event> &depends,
                       const std::vector<sycl::event> &additional_depends)
{
    return ew_cmn_ns::unary_two_outputs_strided_impl<
        argTy, FrexpOutputType, FrexpStridedFunctor, frexp_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res1_p,
        res1_offset, res2_p, res2_offset, depends, additional_depends);
}

template <typename fnT, typename T>
struct FrexpStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename FrexpOutputType<T>::value_type1,
                                     void> ||
                      std::is_same_v<typename FrexpOutputType<T>::value_type2,
                                     void>)
        {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = frexp_strided_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T>
struct FrexpTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::frexp(T x) */
    std::enable_if_t<std::is_same<fnT, std::pair<int, int>>::value,
                     std::pair<int, int>>
        get()
    {
        using rT1 = typename FrexpOutputType<T>::value_type1;
        using rT2 = typename FrexpOutputType<T>::value_type2;
        return std::make_pair(td_ns::GetTypeid<rT1>{}.get(),
                              td_ns::GetTypeid<rT2>{}.get());
    }
};

using ew_cmn_ns::unary_two_outputs_contig_impl_fn_ptr_t;
using ew_cmn_ns::unary_two_outputs_strided_impl_fn_ptr_t;

static unary_two_outputs_contig_impl_fn_ptr_t
    frexp_contig_dispatch_vector[td_ns::num_types];
static std::pair<int, int> frexp_output_typeid_vector[td_ns::num_types];
static unary_two_outputs_strided_impl_fn_ptr_t
    frexp_strided_dispatch_vector[td_ns::num_types];

void populate_frexp_dispatch_vectors(void)
{
    init_dispatch_vector<unary_two_outputs_contig_impl_fn_ptr_t,
                         FrexpContigFactory>(frexp_contig_dispatch_vector);
    init_dispatch_vector<unary_two_outputs_strided_impl_fn_ptr_t,
                         FrexpStridedFactory>(frexp_strided_dispatch_vector);
    init_dispatch_vector<std::pair<int, int>, FrexpTypeMapFactory>(
        frexp_output_typeid_vector);
};

// MACRO_POPULATE_DISPATCH_TABLES(ldexp);
} // namespace impl

void init_frexp(py::module_ m)
{
    using arrayT = dpctl::tensor::usm_ndarray;
    using event_vecT = std::vector<sycl::event>;
    {
        impl::populate_frexp_dispatch_vectors();
        using impl::frexp_contig_dispatch_vector;
        using impl::frexp_output_typeid_vector;
        using impl::frexp_strided_dispatch_vector;

        auto frexp_pyapi = [&](const arrayT &src, const arrayT &dst1,
                               const arrayT &dst2, sycl::queue &exec_q,
                               const event_vecT &depends = {}) {
            return py_int::py_unary_two_outputs_ufunc(
                src, dst1, dst2, exec_q, depends, frexp_output_typeid_vector,
                frexp_contig_dispatch_vector, frexp_strided_dispatch_vector);
        };
        m.def("_frexp", frexp_pyapi, "", py::arg("src"), py::arg("dst1"),
              py::arg("dst2"), py::arg("sycl_queue"),
              py::arg("depends") = py::list());

        auto frexp_result_type_pyapi = [&](const py::dtype &dtype) {
            return py_int::py_unary_two_outputs_ufunc_result_type(
                dtype, frexp_output_typeid_vector);
        };
        m.def("_frexp_result_type", frexp_result_type_pyapi);
    }
}
} // namespace dpnp::extensions::ufunc
