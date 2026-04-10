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
/// This file defines functions of dpnp.tensor._tensor_impl extensions
//===--------------------------------------------------------------------===//

#pragma once

#include <complex>
#include <cstdint>
#include <type_traits>

#include "kernels/linalg_functions/dot_product.hpp"
#include "kernels/linalg_functions/gemm.hpp"
#include "utils/type_dispatch_building.hpp"

namespace dpctl::tensor::py_internal
{

namespace td_ns = dpctl::tensor::type_dispatch;

template <typename T1, typename T2>
struct DotAtomicOutputType
{
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint64_t,
                                        T2,
                                        std::uint64_t,
                                        std::uint64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int64_t,
                                        T2,
                                        std::int64_t,
                                        std::int64_t>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, double>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

// add separate type support lists for atomic vs. temps
// gemm, gevm, and dot product share output type struct
template <typename T1, typename T2>
struct DotNoAtomicOutputType
{
    using value_type = typename std::disjunction<
        td_ns::BinaryTypeMapResultEntry<T1, bool, T2, bool, bool>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint8_t,
                                        T2,
                                        std::uint8_t,
                                        std::uint8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int8_t,
                                        T2,
                                        std::int8_t,
                                        std::int8_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint16_t,
                                        T2,
                                        std::uint16_t,
                                        std::uint16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int16_t,
                                        T2,
                                        std::int16_t,
                                        std::int16_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint32_t,
                                        T2,
                                        std::uint32_t,
                                        std::uint64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int32_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int32_t,
                                        T2,
                                        std::int32_t,
                                        std::int64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::uint64_t,
                                        T2,
                                        std::uint64_t,
                                        std::uint64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::int64_t,
                                        T2,
                                        std::int64_t,
                                        std::int64_t>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        sycl::half,
                                        T2,
                                        sycl::half,
                                        sycl::half>,
        td_ns::BinaryTypeMapResultEntry<T1, float, T2, float, float>,
        td_ns::BinaryTypeMapResultEntry<T1, double, T2, double, double>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        std::complex<float>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<float>,
                                        T2,
                                        std::complex<float>,
                                        std::complex<double>>,
        td_ns::BinaryTypeMapResultEntry<T1,
                                        std::complex<double>,
                                        T2,
                                        std::complex<double>,
                                        std::complex<double>>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

template <typename fnT, typename T1, typename T2>
struct DotTypeMapFactory
{
    /*! @brief get typeid for output type of kernels called by py_dot */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT1 = typename DotNoAtomicOutputType<T1, T2>::value_type;
        using rT2 = typename DotAtomicOutputType<T1, T2>::value_type;
        static_assert(std::is_same_v<rT1, rT2> || std::is_same_v<rT2, void>);
        return td_ns::GetTypeid<rT1>{}.get();
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmBatchAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_batch_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_batch_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmBatchContigAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_batch_contig_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_batch_contig_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmContigAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_contig_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_contig_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmTempsFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmContigTempsFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_contig_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_contig_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmBatchTempsFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_batch_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_batch_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct GemmBatchContigTempsFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::gemm_batch_contig_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = gemm_batch_contig_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct DotProductAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::dot_product_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = dot_product_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct DotProductNoAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::dot_product_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = dot_product_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct DotProductContigAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::dot_product_contig_impl;
            using T3 = typename DotAtomicOutputType<T1, T2>::value_type;
            fnT fn = dot_product_contig_impl<T1, T2, T3>;
            return fn;
        }
    }
};

template <typename fnT, typename T1, typename T2>
struct DotProductContigNoAtomicFactory
{
    fnT get()
    {
        if constexpr (!DotNoAtomicOutputType<T1, T2>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            using dpctl::tensor::kernels::dot_product_contig_tree_impl;
            using T3 = typename DotNoAtomicOutputType<T1, T2>::value_type;
            fnT fn = dot_product_contig_tree_impl<T1, T2, T3>;
            return fn;
        }
    }
};

} // namespace dpctl::tensor::py_internal
