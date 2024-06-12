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

/**
 * @brief A macro used to define factories and a populating universal functions.
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
        const ssize_t *shape_and_strides, const char *arg_p,                   \
        ssize_t arg_offset, char *res_p, ssize_t res_offset,                   \
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
        td_ns::DispatchVectorBuilder<unary_contig_impl_fn_ptr_t,               \
                                     ContigFactory, td_ns::num_types>          \
            dvb1;                                                              \
        dvb1.populate_dispatch_vector(__name__##_contig_dispatch_vector);      \
                                                                               \
        td_ns::DispatchVectorBuilder<unary_strided_impl_fn_ptr_t,              \
                                     StridedFactory, td_ns::num_types>         \
            dvb2;                                                              \
        dvb2.populate_dispatch_vector(__name__##_strided_dispatch_vector);     \
                                                                               \
        td_ns::DispatchVectorBuilder<int, TypeMapFactory, td_ns::num_types>    \
            dvb3;                                                              \
        dvb3.populate_dispatch_vector(__name__##_output_typeid_vector);        \
    };
