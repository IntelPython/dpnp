//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
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
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for elementwise evaluation of COS(x) function.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "sycl_complex.hpp"
#include "vec_size_util.hpp"

#include "kernels/dpctl_tensor_types.hpp"
#include "kernels/elementwise_functions/common.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl::tensor::kernels::cos
{

using dpctl::tensor::ssize_t;
namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;

template <typename argT, typename resT>
struct CosFunctor
{

    // is function constant for given argT
    using is_constant = typename std::false_type;
    // constant value, if constant
    // constexpr resT constant_value = resT{};
    // is function defined for sycl::vec
    using supports_vec = typename std::false_type;
    // do both argTy and resTy support sugroup store/load operation
    using supports_sg_loadstore = typename std::negation<
        std::disjunction<is_complex<resT>, is_complex<argT>>>;

    resT operator()(const argT &in) const
    {
        if constexpr (is_complex<argT>::value) {
            using realT = typename argT::value_type;

            static constexpr realT q_nan =
                std::numeric_limits<realT>::quiet_NaN();

            realT const &in_re = std::real(in);
            realT const &in_im = std::imag(in);

            const bool in_re_finite = std::isfinite(in_re);
            const bool in_im_finite = std::isfinite(in_im);

            /*
             * Handle the nearly-non-exceptional cases where
             * real and imaginary parts of input are finite.
             */
            if (in_re_finite && in_im_finite) {
                return exprm_ns::cos(exprm_ns::complex<realT>(in)); // cos(in);
            }

            /*
             * since cos(in) = cosh(I * in), for special cases,
             * we return cosh(I * in).
             */
            const realT x = -in_im;
            const realT y = in_re;

            const bool xfinite = in_im_finite;
            const bool yfinite = in_re_finite;
            /*
             * cosh(+-0 +- I Inf) = dNaN + I sign(d(+-0, dNaN))0.
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as dNaN.
             *
             * cosh(+-0 +- I NaN) = d(NaN) + I sign(d(+-0, NaN))0.
             * The sign of 0 in the result is unspecified.  Choice = normally
             * the same as d(NaN).
             */
            if (x == realT(0) && !yfinite) {
                const realT y_m_y = (y - y);
                const realT res_im = sycl::copysign(realT(0), x * y_m_y);
                return resT{y_m_y, res_im};
            }

            /*
             * cosh(+-Inf +- I 0) = +Inf + I (+-)(+-)0.
             *
             * cosh(NaN +- I 0)   = d(NaN) + I sign(d(NaN, +-0))0.
             * The sign of 0 in the result is unspecified.
             */
            if (y == realT(0) && !xfinite) {
                const realT res_im = sycl::copysign(realT(0), x) * y;
                return resT{x * x, res_im};
            }

            /*
             * cosh(x +- I Inf) = dNaN + I dNaN.
             *
             * cosh(x + I NaN) = d(NaN) + I d(NaN).
             */
            if (xfinite && !yfinite) {
                const realT y_m_y = (y - y);
                return resT{y_m_y, x * y_m_y};
            }

            /*
             * cosh(+-Inf + I NaN)  = +Inf + I d(NaN).
             *
             * cosh(+-Inf +- I Inf) = +Inf + I dNaN.
             * The sign of Inf in the result is unspecified.  Choice = always +.
             *
             * cosh(+-Inf + I y)   = +Inf cos(y) +- I Inf sin(y)
             */
            if (std::isinf(x)) {
                if (!yfinite) {
                    return resT{x * x, sycl::copysign(q_nan, x)};
                }
                return resT{(x * x) * sycl::cos(y), x * sycl::sin(y)};
            }

            /*
             * cosh(NaN + I NaN)  = d(NaN) + I d(NaN).
             *
             * cosh(NaN +- I Inf) = d(NaN) + I d(NaN).
             *
             * cosh(NaN + I y)    = d(NaN) + I d(NaN).
             */
            return resT{(x * x) * q_nan, (x + x) * q_nan};
        }
        else {
            static_assert(std::is_floating_point_v<argT> ||
                          std::is_same_v<argT, sycl::half>);
            return sycl::cos(in);
        }
    }
};

template <typename argTy,
          typename resTy = argTy,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using CosContigFunctor =
    elementwise_common::UnaryContigFunctor<argTy,
                                           resTy,
                                           CosFunctor<argTy, resTy>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using CosStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, CosFunctor<argTy, resTy>>;

template <typename T>
struct CosOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<T, sycl::half, sycl::half>,
        td_ns::TypeMapResultEntry<T, float, float>,
        td_ns::TypeMapResultEntry<T, double, double>,
        td_ns::TypeMapResultEntry<T, std::complex<float>, std::complex<float>>,
        td_ns::
            TypeMapResultEntry<T, std::complex<double>, std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace hyperparam_detail
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy>
struct CosContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of namespace hyperparam_detail

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class cos_contig_kernel;

template <typename argTy>
sycl::event cos_contig_impl(sycl::queue &exec_q,
                            std::size_t nelems,
                            const char *arg_p,
                            char *res_p,
                            const std::vector<sycl::event> &depends = {})
{
    using CosHS = hyperparam_detail::CosContigHyperparameterSet<argTy>;
    static constexpr std::uint8_t vec_sz = CosHS::vec_sz;
    static constexpr std::uint8_t n_vecs = CosHS::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, CosOutputType, CosContigFunctor, cos_contig_kernel, vec_sz,
        n_vecs>(exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T>
struct CosContigFactory
{
    fnT get()
    {
        if constexpr (!CosOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cos_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T>
struct CosTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::cos(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename CosOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3>
class cos_strided_kernel;

template <typename argTy>
sycl::event cos_strided_impl(sycl::queue &exec_q,
                             std::size_t nelems,
                             int nd,
                             const ssize_t *shape_and_strides,
                             const char *arg_p,
                             ssize_t arg_offset,
                             char *res_p,
                             ssize_t res_offset,
                             const std::vector<sycl::event> &depends,
                             const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::unary_strided_impl<
        argTy, CosOutputType, CosStridedFunctor, cos_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T>
struct CosStridedFactory
{
    fnT get()
    {
        if constexpr (!CosOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = cos_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::kernels::cos
