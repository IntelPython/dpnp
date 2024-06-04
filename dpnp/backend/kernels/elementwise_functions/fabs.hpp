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

#pragma once

#include <sycl/sycl.hpp>

// dpctl tensor headers
#include "kernels/elementwise_functions/common.hpp"
#include "utils/type_utils.hpp"

namespace dpnp::backend::kernels::fabs
{

namespace ew_cmn_ns = dpctl::tensor::kernels::elementwise_common;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT> struct FabsFunctor
{

    using is_constant = typename std::false_type;
    // constexpr resT constant_value = resT{};
    using supports_vec = typename std::false_type;
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &x) const
    {

        // if constexpr (std::is_same_v<argT, bool> || (std::is_integral<argT>::value && std::is_unsigned<argT>::value))
        // {
        //     return resT(x);
        // }
        // else if constexpr (std::is_same_v<argT, sycl::half> || std::is_floating_point_v<argT>)
        // {
        //     return sycl::fabs(x);
        // }
        // else // std::is_integral<argT>::value
        // {
        //     return resT(sycl::abs(x));
        // }
        return sycl::fabs(x);
    }
};

template <typename argT,
          typename resT = argT,
          unsigned int vec_sz = 4,
          unsigned int n_vecs = 2,
          bool enable_sg_loadstore = true>
using FabsContigFunctor = ew_cmn_ns::UnaryContigFunctor<argT,
                                           resT,
                                           FabsFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename T> struct FabsOutputType
{
    using value_type = typename std::disjunction< // disjunction is C++17
                                                  // feature, supported by DPC++
        // td_ns::TypeMapResultEntry<T, bool>,
        // td_ns::TypeMapResultEntry<T, std::uint8_t>,
        // td_ns::TypeMapResultEntry<T, std::uint16_t>,
        // td_ns::TypeMapResultEntry<T, std::uint32_t>,
        // td_ns::TypeMapResultEntry<T, std::uint64_t>,
        // td_ns::TypeMapResultEntry<T, std::int8_t>,
        // td_ns::TypeMapResultEntry<T, std::int16_t>,
        // td_ns::TypeMapResultEntry<T, std::int32_t>,
        // td_ns::TypeMapResultEntry<T, std::int64_t>,
        td_ns::TypeMapResultEntry<T, sycl::half>,
        td_ns::TypeMapResultEntry<T, float>,
        td_ns::TypeMapResultEntry<T, double>,
        // td_ns::TypeMapResultEntry<T, std::complex<float>, float>,
        // td_ns::TypeMapResultEntry<T, std::complex<double>, double>,
        td_ns::DefaultResultEntry<void>>::result_type;
};

template <typename T1, typename T2, unsigned int vec_sz, unsigned int n_vecs>
class fabs_contig_kernel;

template <typename argTy>
sycl::event fabs_contig_impl(sycl::queue &exec_q,
                             size_t nelems,
                             const char *arg_p,
                             char *res_p,
                             const std::vector<sycl::event> &depends = {})
{
    return ew_cmn_ns::unary_contig_impl<
        argTy, FabsOutputType, FabsContigFunctor, fabs_contig_kernel>(
        exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct FabsContigFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename FabsOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = fabs_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct FabsTypeMapFactory
{
    /*! @brief get typeid for output type of fabs(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename FabsOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename argTy, typename resTy, typename IndexerT>
using FabsStridedFunctor = ew_cmn_ns::
    UnaryStridedFunctor<argTy, resTy, IndexerT, FabsFunctor<argTy, resTy>>;

template <typename T1, typename T2, typename T3> class fabs_strided_kernel;

template <typename argTy>
sycl::event fabs_strided_impl(sycl::queue &exec_q,
                              size_t nelems,
                              int nd,
                              const ssize_t *shape_and_strides,
                              const char *arg_p,
                              ssize_t arg_offset,
                              char *res_p,
                              ssize_t res_offset,
                              const std::vector<sycl::event> &depends,
                              const std::vector<sycl::event> &additional_depends)
{
    return ew_cmn_ns::unary_strided_impl<
        argTy, FabsOutputType, FabsStridedFunctor, fabs_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct FabsStridedFactory
{
    fnT get()
    {
        if constexpr (std::is_same_v<typename FabsOutputType<T>::value_type,
                                     void>) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = fabs_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace dpnp::backend::kernels::fabs
