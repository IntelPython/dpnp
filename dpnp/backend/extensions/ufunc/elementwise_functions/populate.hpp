//*****************************************************************************
// Copyright (c) 2024-2025, Intel Corporation
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

// utils extension header
#include "ext/common.hpp"

namespace ext_ns = ext::common;

/**
 * @brief A macro used to define factories and a populating unary universal
 * functions.
 */
#define MACRO_POPULATE_DISPATCH_VECTORS(__name__)                              \
    template <typename T1, typename T2, unsigned int vec_sz,                   \
              unsigned int n_vecs>                                             \
    class __name__##_contig_kernel;                                            \
                                                                               \
    template <typename argTy>                                                  \
    sycl::event __name__##_contig_impl(                                        \
        sycl::queue &exec_q, size_t nelems, const char *arg_p, char *res_p,    \
        const std::vector<sycl::event> &depends = {})                          \
    {                                                                          \
        return ew_cmn_ns::unary_contig_impl<argTy, OutputType, ContigFunctor,  \
                                            __name__##_contig_kernel>(         \
            exec_q, nelems, arg_p, res_p, depends);                            \
    }                                                                          \
                                                                               \
    template <typename fnT, typename T>                                        \
    struct ContigFactory                                                       \
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
    template <typename fnT, typename T>                                        \
    struct TypeMapFactory                                                      \
    {                                                                          \
        std::enable_if_t<std::is_same<fnT, int>::value, int> get()             \
        {                                                                      \
            using rT = typename OutputType<T>::value_type;                     \
            return td_ns::GetTypeid<rT>{}.get();                               \
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
        return ew_cmn_ns::unary_strided_impl<                                  \
            argTy, OutputType, StridedFunctor, __name__##_strided_kernel>(     \
            exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,   \
            res_offset, depends, additional_depends);                          \
    }                                                                          \
                                                                               \
    template <typename fnT, typename T>                                        \
    struct StridedFactory                                                      \
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
    };                                                                         \
                                                                               \
    void populate_##__name__##_dispatch_vectors(void)                          \
    {                                                                          \
        ext_ns::init_dispatch_vector<unary_contig_impl_fn_ptr_t,               \
                                     ContigFactory>(                           \
            __name__##_contig_dispatch_vector);                                \
        ext_ns::init_dispatch_vector<unary_strided_impl_fn_ptr_t,              \
                                     StridedFactory>(                          \
            __name__##_strided_dispatch_vector);                               \
        ext_ns::init_dispatch_vector<int, TypeMapFactory>(                     \
            __name__##_output_typeid_vector);                                  \
    };

/**
 * @brief A macro used to define factories and a populating binary universal
 * functions.
 */
#define MACRO_POPULATE_DISPATCH_TABLES(__name__)                               \
    template <typename argT1, typename argT2, typename resT,                   \
              unsigned int vec_sz, unsigned int n_vecs>                        \
    class __name__##_contig_kernel;                                            \
                                                                               \
    template <typename argTy1, typename argTy2>                                \
    sycl::event __name__##_contig_impl(                                        \
        sycl::queue &exec_q, size_t nelems, const char *arg1_p,                \
        py::ssize_t arg1_offset, const char *arg2_p, py::ssize_t arg2_offset,  \
        char *res_p, py::ssize_t res_offset,                                   \
        const std::vector<sycl::event> &depends = {})                          \
    {                                                                          \
        return ew_cmn_ns::binary_contig_impl<argTy1, argTy2, OutputType,       \
                                             ContigFunctor,                    \
                                             __name__##_contig_kernel>(        \
            exec_q, nelems, arg1_p, arg1_offset, arg2_p, arg2_offset, res_p,   \
            res_offset, depends);                                              \
    }                                                                          \
                                                                               \
    template <typename fnT, typename T1, typename T2>                          \
    struct ContigFactory                                                       \
    {                                                                          \
        fnT get()                                                              \
        {                                                                      \
            if constexpr (std::is_same_v<                                      \
                              typename OutputType<T1, T2>::value_type, void>)  \
            {                                                                  \
                                                                               \
                fnT fn = nullptr;                                              \
                return fn;                                                     \
            }                                                                  \
            else {                                                             \
                fnT fn = __name__##_contig_impl<T1, T2>;                       \
                return fn;                                                     \
            }                                                                  \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <typename fnT, typename T1, typename T2>                          \
    struct TypeMapFactory                                                      \
    {                                                                          \
        std::enable_if_t<std::is_same<fnT, int>::value, int> get()             \
        {                                                                      \
            using rT = typename OutputType<T1, T2>::value_type;                \
            return td_ns::GetTypeid<rT>{}.get();                               \
        }                                                                      \
    };                                                                         \
                                                                               \
    template <typename T1, typename T2, typename resT, typename IndexerT>      \
    class __name__##_strided_kernel;                                           \
                                                                               \
    template <typename argTy1, typename argTy2>                                \
    sycl::event __name__##_strided_impl(                                       \
        sycl::queue &exec_q, size_t nelems, int nd,                            \
        const py::ssize_t *shape_and_strides, const char *arg1_p,              \
        py::ssize_t arg1_offset, const char *arg2_p, py::ssize_t arg2_offset,  \
        char *res_p, py::ssize_t res_offset,                                   \
        const std::vector<sycl::event> &depends,                               \
        const std::vector<sycl::event> &additional_depends)                    \
    {                                                                          \
        return ew_cmn_ns::binary_strided_impl<argTy1, argTy2, OutputType,      \
                                              StridedFunctor,                  \
                                              __name__##_strided_kernel>(      \
            exec_q, nelems, nd, shape_and_strides, arg1_p, arg1_offset,        \
            arg2_p, arg2_offset, res_p, res_offset, depends,                   \
            additional_depends);                                               \
    }                                                                          \
                                                                               \
    template <typename fnT, typename T1, typename T2>                          \
    struct StridedFactory                                                      \
    {                                                                          \
        fnT get()                                                              \
        {                                                                      \
            if constexpr (std::is_same_v<                                      \
                              typename OutputType<T1, T2>::value_type, void>)  \
            {                                                                  \
                fnT fn = nullptr;                                              \
                return fn;                                                     \
            }                                                                  \
            else {                                                             \
                fnT fn = __name__##_strided_impl<T1, T2>;                      \
                return fn;                                                     \
            }                                                                  \
        }                                                                      \
    };                                                                         \
                                                                               \
    void populate_##__name__##_dispatch_tables(void)                           \
    {                                                                          \
        ext_ns::init_dispatch_table<binary_contig_impl_fn_ptr_t,               \
                                    ContigFactory>(                            \
            __name__##_contig_dispatch_table);                                 \
        ext_ns::init_dispatch_table<binary_strided_impl_fn_ptr_t,              \
                                    StridedFactory>(                           \
            __name__##_strided_dispatch_table);                                \
        ext_ns::init_dispatch_table<int, TypeMapFactory>(                      \
            __name__##_output_typeid_table);                                   \
    };
