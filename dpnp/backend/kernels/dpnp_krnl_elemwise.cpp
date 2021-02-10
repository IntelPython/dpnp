//*****************************************************************************
// Copyright (c) 2016-2020, Intel Corporation
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

#include <cmath>
#include <iostream>

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "queue_sycl.hpp"

#define MACRO_1ARG_2TYPES_OP(__name__, __operation1__, __operation2__)                                                 \
    template <typename _KernelNameSpecialization>                                                                      \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _DataType_input, typename _DataType_output>                                                     \
    void __name__(void* array1_in, void* result1, size_t size)                                                         \
    {                                                                                                                  \
        cl::sycl::event event;                                                                                         \
                                                                                                                       \
        _DataType_input* array1 = reinterpret_cast<_DataType_input*>(array1_in);                                       \
        _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);                                       \
                                                                                                                       \
        cl::sycl::range<1> gws(size);                                                                                  \
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                               \
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/                                             \
            {                                                                                                          \
                _DataType_output input_elem = array1[i];                                                               \
                result[i] = __operation1__;                                                                            \
            }                                                                                                          \
        };                                                                                                             \
                                                                                                                       \
        auto kernel_func = [&](cl::sycl::handler& cgh) {                                                               \
            cgh.parallel_for<class __name__##_kernel<_DataType_input>>(gws, kernel_parallel_for_func);                 \
        };                                                                                                             \
                                                                                                                       \
        if constexpr (std::is_same<_DataType_input, double>::value || std::is_same<_DataType_input, float>::value)     \
        {                                                                                                              \
            event = __operation2__;                                                                                    \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
        }                                                                                                              \
                                                                                                                       \
        event.wait();                                                                                                  \
    }

#include <dpnp_gen_1arg_2type_tbl.hpp>

static void func_map_init_elemwise_1arg_2type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_acos_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_acos_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_acos_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_acos_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_acosh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_acosh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_acosh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_acosh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_asin_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_asin_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_asin_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_asin_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_asinh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_asinh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_asinh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_asinh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_atan_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_atan_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_atan_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_atan_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_atanh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_atanh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_atanh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_atanh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cbrt_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cbrt_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cbrt_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cbrt_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_ceil_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_ceil_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_ceil_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_ceil_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cos_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cos_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cos_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cos_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COSH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cosh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cosh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cosh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cosh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_degrees_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_degrees_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_degrees_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_degrees_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_exp2_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_exp2_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_exp2_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_exp2_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_exp_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_exp_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_exp_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_exp_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_expm1_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_expm1_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_expm1_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_expm1_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FABS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_fabs_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_fabs_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_fabs_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_fabs_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_floor_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_floor_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_floor_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_floor_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log10_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log10_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log10_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log10_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log1p_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log1p_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log1p_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log1p_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log2_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log2_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log2_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log2_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_radians_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_radians_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_radians_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_radians_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SIN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_sin_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_sin_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sin_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sin_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SINH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_sinh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_sinh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sinh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sinh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_sqrt_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_sqrt_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sqrt_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sqrt_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TAN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_tan_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_tan_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tan_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tan_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TANH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_tanh_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_tanh_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tanh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tanh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_trunc_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_trunc_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_trunc_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_trunc_c<double, double>};

    return;
}

#define MACRO_1ARG_1TYPE_OP(__name__, __operation1__, __operation2__)                                                  \
    template <typename _KernelNameSpecialization>                                                                      \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _DataType>                                                                                      \
    void __name__(void* array1_in, void* result1, size_t size)                                                         \
    {                                                                                                                  \
        cl::sycl::event event;                                                                                         \
                                                                                                                       \
        if (!size)                                                                                                     \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        _DataType* array1 = reinterpret_cast<_DataType*>(array1_in);                                                   \
        _DataType* result = reinterpret_cast<_DataType*>(result1);                                                     \
                                                                                                                       \
        cl::sycl::range<1> gws(size);                                                                                  \
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                               \
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/                                             \
            {                                                                                                          \
                _DataType input_elem = array1[i];                                                                      \
                result[i] = __operation1__;                                                                            \
            }                                                                                                          \
        };                                                                                                             \
                                                                                                                       \
        auto kernel_func = [&](cl::sycl::handler& cgh) {                                                               \
            cgh.parallel_for<class __name__##_kernel<_DataType>>(gws, kernel_parallel_for_func);                       \
        };                                                                                                             \
                                                                                                                       \
        if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)                 \
        {                                                                                                              \
            event = __operation2__;                                                                                    \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
        }                                                                                                              \
                                                                                                                       \
        event.wait();                                                                                                  \
    }

#include <dpnp_gen_1arg_1type_tbl.hpp>

static void func_map_init_elemwise_1arg_1type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copy_c<int>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copy_c<long>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copy_c<float>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copy_c<double>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_C128][eft_C128] = {eft_C128,
                                                                  (void*)dpnp_conjugate_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_COPY][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_copy_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copy_c<int>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copy_c<long>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copy_c<float>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copy_c<double>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_C128][eft_C128] = {eft_C128, (void*)dpnp_copy_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_ERF][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_erf_c<int>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_erf_c<long>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_erf_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_erf_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_recip_c<int>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_recip_c<long>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_recip_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_recip_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_sign_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_sign_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sign_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sign_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_square_c<int>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_square_c<long>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_square_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_square_c<double>};

    return;
}

#define MACRO_2ARG_3TYPES_OP(__name__, __operation1__, __operation2__)                                                 \
    template <typename _KernelNameSpecialization1,                                                                     \
              typename _KernelNameSpecialization2,                                                                     \
              typename _KernelNameSpecialization3>                                                                     \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>                         \
    void __name__(void* array1_in, void* array2_in, void* result1, size_t size)                                        \
    {                                                                                                                  \
        cl::sycl::event event;                                                                                         \
        _DataType_input1* array1 = reinterpret_cast<_DataType_input1*>(array1_in);                                     \
        _DataType_input2* array2 = reinterpret_cast<_DataType_input2*>(array2_in);                                     \
        _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);                                       \
                                                                                                                       \
        if constexpr ((std::is_same<_DataType_input1, double>::value ||                                                \
                       std::is_same<_DataType_input1, float>::value) &&                                                \
                      std::is_same<_DataType_input2, _DataType_input1>::value)                                         \
        {                                                                                                              \
            event = __operation2__(DPNP_QUEUE, size, array1, array2, result);                                          \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            cl::sycl::range<1> gws(size);                                                                              \
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                           \
                size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/                                         \
                {                                                                                                      \
                    _DataType_output input_elem1 = array1[i];                                                          \
                    _DataType_output input_elem2 = array2[i];                                                          \
                    result[i] = __operation1__;                                                                        \
                }                                                                                                      \
            };                                                                                                         \
                                                                                                                       \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class __name__##_kernel<_DataType_input1, _DataType_input2, _DataType_output>>(       \
                    gws, kernel_parallel_for_func);                                                                    \
            };                                                                                                         \
                                                                                                                       \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
        }                                                                                                              \
                                                                                                                       \
        event.wait();                                                                                                  \
    }

#include <dpnp_gen_2arg_3type_tbl.hpp>

static void func_map_init_elemwise_2arg_3type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_add_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_add_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_add_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_add_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_add_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_add_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_add_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_add_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_add_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_add_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_add_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_add_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_add_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_add_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_add_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_add_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_c<int, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_arctan2_c<int, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_arctan2_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_arctan2_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_c<long, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_arctan2_c<long, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_arctan2_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_arctan2_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_arctan2_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_arctan2_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_arctan2_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_arctan2_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_arctan2_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_arctan2_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_copysign_c<int, int, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_copysign_c<int, long, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_copysign_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_copysign_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_copysign_c<long, int, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_copysign_c<long, long, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_copysign_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_copysign_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_copysign_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_copysign_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copysign_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_copysign_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_copysign_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_copysign_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_copysign_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copysign_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_divide_c<int, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_divide_c<int, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_divide_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_divide_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_divide_c<long, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_divide_c<long, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_divide_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_divide_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_divide_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_divide_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_divide_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_divide_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_divide_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_divide_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_divide_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_divide_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_fmod_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_fmod_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_fmod_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_fmod_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_fmod_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_fmod_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_fmod_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_fmod_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_fmod_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_fmod_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_fmod_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_fmod_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_hypot_c<int, int, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_c<int, long, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_hypot_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_hypot_c<long, int, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_c<long, long, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_hypot_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_hypot_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_hypot_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_hypot_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_hypot_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_maximum_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_maximum_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_maximum_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_maximum_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_maximum_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_maximum_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_maximum_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_maximum_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_maximum_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_maximum_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_maximum_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_maximum_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_maximum_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_maximum_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_maximum_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_maximum_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_minimum_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_minimum_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_minimum_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_minimum_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_minimum_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_minimum_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_minimum_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_minimum_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_minimum_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_minimum_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_minimum_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_minimum_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_minimum_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_minimum_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_minimum_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_minimum_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_multiply_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_multiply_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_multiply_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_multiply_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_multiply_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_multiply_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_multiply_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_multiply_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_multiply_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_multiply_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_multiply_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_multiply_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_power_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_power_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_power_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_power_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_power_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_power_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_power_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_power_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_power_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_power_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_power_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_power_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_power_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_power_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_power_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_power_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_subtract_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_subtract_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_subtract_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_subtract_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_subtract_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_subtract_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_subtract_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_subtract_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_subtract_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_subtract_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_subtract_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_subtract_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_subtract_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_subtract_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_subtract_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_subtract_c<double, double, double>};

    return;
}

void func_map_init_elemwise(func_map_t& fmap)
{
    func_map_init_elemwise_1arg_1type(fmap);
    func_map_init_elemwise_1arg_2type(fmap);
    func_map_init_elemwise_2arg_3type(fmap);

    return;
}
