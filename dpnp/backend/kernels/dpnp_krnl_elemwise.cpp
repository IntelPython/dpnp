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
#include "dpnp_iterator.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

#define MACRO_1ARG_2TYPES_OP(__name__, __operation1__, __operation2__)                                                 \
    template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>                                \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>                                \
    class __name__##_strides_kernel;                                                                                   \
                                                                                                                       \
    template <typename _DataType_input, typename _DataType_output>                                                     \
    void __name__(void* result_out,                                                                                    \
                  const size_t result_size,                                                                            \
                  const size_t result_ndim,                                                                            \
                  const shape_elem_type* result_shape,                                                                 \
                  const shape_elem_type* result_strides,                                                               \
                  const void* input1_in,                                                                               \
                  const size_t input1_size,                                                                            \
                  const size_t input1_ndim,                                                                            \
                  const shape_elem_type* input1_shape,                                                                 \
                  const shape_elem_type* input1_strides,                                                               \
                  const size_t* where)                                                                                 \
    {                                                                                                                  \
        /* avoid warning unused variable*/                                                                             \
        (void)result_shape;                                                                                            \
        (void)where;                                                                                                   \
                                                                                                                       \
        if (!input1_size)                                                                                              \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType_input> input1_ptr(input1_in, input1_size);                                         \
        DPNPC_ptr_adapter<shape_elem_type> input1_shape_ptr(input1_shape, input1_ndim, true);                          \
        DPNPC_ptr_adapter<shape_elem_type> input1_strides_ptr(input1_strides, input1_ndim, true);                      \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType_output> result_ptr(result_out, result_size, false, true);                          \
        DPNPC_ptr_adapter<shape_elem_type> result_strides_ptr(result_strides, result_ndim);                            \
                                                                                                                       \
        _DataType_input* input1_data = input1_ptr.get_ptr();                                                           \
        shape_elem_type* input1_shape_data = input1_shape_ptr.get_ptr();                                               \
        shape_elem_type* input1_strides_data = input1_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        _DataType_output* result = result_ptr.get_ptr();                                                               \
        shape_elem_type* result_strides_data = result_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        const size_t input1_shape_size_in_bytes = input1_ndim * sizeof(shape_elem_type);                               \
        shape_elem_type* input1_shape_offsets =                                                                        \
            reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(input1_shape_size_in_bytes));                       \
        get_shape_offsets_inkernel(input1_shape_data, input1_ndim, input1_shape_offsets);                              \
        bool use_strides = !array_equal(input1_strides_data, input1_ndim, input1_shape_offsets, input1_ndim);          \
        dpnp_memory_free_c(input1_shape_offsets);                                                                      \
                                                                                                                       \
        cl::sycl::event event;                                                                                         \
        cl::sycl::range<1> gws(result_size);                                                                           \
                                                                                                                       \
        if (use_strides)                                                                                               \
        {                                                                                                              \
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                           \
                size_t output_id = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                          \
                {                                                                                                      \
                    size_t input_id = 0;                                                                               \
                    for (size_t i = 0; i < input1_ndim; ++i)                                                           \
                    {                                                                                                  \
                        const size_t output_xyz_id = get_xyz_id_by_id_inkernel(output_id,                              \
                                                                               result_strides_data,                    \
                                                                               result_ndim,                            \
                                                                               i);                                     \
                        input_id += output_xyz_id * input1_strides_data[i];                                            \
                    }                                                                                                  \
                                                                                                                       \
                    const _DataType_output input_elem = input1_data[input_id];                                         \
                    result[output_id] = __operation1__;                                                                \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class __name__##_strides_kernel<_DataType_input,                                      \
                                                                 _DataType_output>>(gws, kernel_parallel_for_func);    \
            };                                                                                                         \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
            event.wait();                                                                                              \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                           \
                size_t output_id = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                          \
                {                                                                                                      \
                    const _DataType_output input_elem = input1_data[output_id];                                        \
                    result[output_id] = __operation1__;                                                                \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class __name__##_kernel<_DataType_input,                                              \
                                                         _DataType_output>>(gws, kernel_parallel_for_func);            \
            };                                                                                                         \
                                                                                                                       \
            if constexpr ((std::is_same<_DataType_input, double>::value ||                                             \
                           std::is_same<_DataType_input, float>::value) &&                                             \
                          std::is_same<_DataType_input, _DataType_output>::value)                                      \
            {                                                                                                          \
                event = __operation2__;                                                                                \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                event = DPNP_QUEUE.submit(kernel_func);                                                                \
            }                                                                                                          \
            event.wait();                                                                                              \
        }                                                                                                              \
    }

#include <dpnp_gen_1arg_2type_tbl.hpp>

static void func_map_init_elemwise_1arg_2type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_acos_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_acos_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_acos_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_acos_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_acosh_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_acosh_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_acosh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_acosh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_asin_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_asin_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_asin_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_asin_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_asinh_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_asinh_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_asinh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_asinh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_atan_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_atan_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_atan_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_atan_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_atanh_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_atanh_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_atanh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_atanh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cbrt_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cbrt_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cbrt_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cbrt_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_ceil_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_ceil_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_ceil_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_ceil_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_copyto_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_INT] = {eft_INT, (void*)dpnp_copyto_c<bool, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_LNG] = {eft_LNG, (void*)dpnp_copyto_c<bool, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_FLT] = {eft_FLT, (void*)dpnp_copyto_c<bool, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_DBL] = {eft_DBL, (void*)dpnp_copyto_c<bool, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_BLN] = {eft_BLN, (void*)dpnp_copyto_c<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copyto_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_copyto_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_FLT] = {eft_FLT, (void*)dpnp_copyto_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_copyto_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_BLN] = {eft_BLN, (void*)dpnp_copyto_c<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_copyto_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copyto_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_FLT] = {eft_FLT, (void*)dpnp_copyto_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_copyto_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_BLN] = {eft_BLN, (void*)dpnp_copyto_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_copyto_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_copyto_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copyto_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_copyto_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_BLN] = {eft_BLN, (void*)dpnp_copyto_c<double, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_copyto_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_copyto_c<double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_FLT] = {eft_FLT, (void*)dpnp_copyto_c<double, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copyto_c<double, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_copyto_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_COS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cos_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cos_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cos_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cos_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COSH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_cosh_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_cosh_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cosh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cosh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_degrees_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_degrees_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_degrees_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_degrees_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EDIFF1D][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_ediff1d_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_EDIFF1D][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_ediff1d_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_EDIFF1D][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_ediff1d_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EDIFF1D][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_ediff1d_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_exp2_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_exp2_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_exp2_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_exp2_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_exp_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_exp_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_exp_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_exp_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_expm1_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_expm1_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_expm1_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_expm1_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FABS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_fabs_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_fabs_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_fabs_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_fabs_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_floor_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_floor_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_floor_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_floor_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log10_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log10_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log10_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log10_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log1p_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log1p_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log1p_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log1p_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log2_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log2_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log2_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log2_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_log_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_log_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_log_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_log_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_radians_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_radians_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_radians_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_radians_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SIN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_sin_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_sin_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sin_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sin_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SINH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_sinh_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_sinh_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sinh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sinh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_sqrt_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_sqrt_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sqrt_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sqrt_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TAN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_tan_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_tan_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tan_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tan_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TANH][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_tanh_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_tanh_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_tanh_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_tanh_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_trunc_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_trunc_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_trunc_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_trunc_c<double, double>};

    return;
}

#define MACRO_1ARG_1TYPE_OP(__name__, __operation1__, __operation2__)                                                  \
    template <typename _KernelNameSpecialization>                                                                      \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _KernelNameSpecialization>                                                                      \
    class __name__##_strides_kernel;                                                                                   \
                                                                                                                       \
    template <typename _DataType>                                                                                      \
    void __name__(void* result_out,                                                                                    \
                  const size_t result_size,                                                                            \
                  const size_t result_ndim,                                                                            \
                  const shape_elem_type* result_shape,                                                                 \
                  const shape_elem_type* result_strides,                                                               \
                  const void* input1_in,                                                                               \
                  const size_t input1_size,                                                                            \
                  const size_t input1_ndim,                                                                            \
                  const shape_elem_type* input1_shape,                                                                 \
                  const shape_elem_type* input1_strides,                                                               \
                  const size_t* where)                                                                                 \
    {                                                                                                                  \
        /* avoid warning unused variable*/                                                                             \
        (void)result_shape;                                                                                            \
        (void)where;                                                                                                   \
                                                                                                                       \
        if (!input1_size)                                                                                              \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType> input1_ptr(input1_in, input1_size);                                               \
        DPNPC_ptr_adapter<shape_elem_type> input1_shape_ptr(input1_shape, input1_ndim, true);                          \
        DPNPC_ptr_adapter<shape_elem_type> input1_strides_ptr(input1_strides, input1_ndim, true);                      \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType> result_ptr(result_out, result_size, false, true);                                 \
        DPNPC_ptr_adapter<shape_elem_type> result_strides_ptr(result_strides, result_ndim);                            \
                                                                                                                       \
        _DataType* input1_data = input1_ptr.get_ptr();                                                                 \
        shape_elem_type* input1_shape_data = input1_shape_ptr.get_ptr();                                               \
        shape_elem_type* input1_strides_data = input1_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        _DataType* result = result_ptr.get_ptr();                                                                      \
        shape_elem_type* result_strides_data = result_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        const size_t input1_shape_size_in_bytes = input1_ndim * sizeof(shape_elem_type);                               \
        shape_elem_type* input1_shape_offsets =                                                                        \
        reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(input1_shape_size_in_bytes));                           \
        get_shape_offsets_inkernel(input1_shape_data, input1_ndim, input1_shape_offsets);                              \
        bool use_strides = !array_equal(input1_strides_data, input1_ndim, input1_shape_offsets, input1_ndim);          \
        dpnp_memory_free_c(input1_shape_offsets);                                                                      \
                                                                                                                       \
        cl::sycl::event event;                                                                                         \
        cl::sycl::range<1> gws(result_size);                                                                           \
                                                                                                                       \
        if (use_strides)                                                                                               \
        {                                                                                                              \
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                           \
                size_t output_id = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                          \
                {                                                                                                      \
                    size_t input_id = 0;                                                                               \
                    for (size_t i = 0; i < input1_ndim; ++i)                                                           \
                    {                                                                                                  \
                        const size_t output_xyz_id = get_xyz_id_by_id_inkernel(output_id,                              \
                                                                               result_strides_data,                    \
                                                                               result_ndim,                            \
                                                                               i);                                     \
                        input_id += output_xyz_id * input1_strides_data[i];                                            \
                    }                                                                                                  \
                                                                                                                       \
                    const _DataType input_elem = input1_data[input_id];                                                \
                    result[output_id] = __operation1__;                                                                \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class __name__##_strides_kernel<_DataType>>(gws, kernel_parallel_for_func);           \
            };                                                                                                         \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                           \
                size_t i = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                                  \
                {                                                                                                      \
                    const _DataType input_elem = input1_data[i];                                                       \
                    result[i] = __operation1__;                                                                        \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class __name__##_kernel<_DataType>>(gws, kernel_parallel_for_func);                   \
            };                                                                                                         \
                                                                                                                       \
            if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)             \
            {                                                                                                          \
                event = __operation2__;                                                                                \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                event = DPNP_QUEUE.submit(kernel_func);                                                                \
            }                                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        event.wait();                                                                                                  \
    }

#include <dpnp_gen_1arg_1type_tbl.hpp>

static void func_map_init_elemwise_1arg_1type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copy_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copy_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copy_c<float>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copy_c<double>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_C128][eft_C128] = {eft_C128,
                                                                  (void*)dpnp_conjugate_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_COPY][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_copy_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copy_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copy_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copy_c<float>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copy_c<double>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_C128][eft_C128] = {eft_C128, (void*)dpnp_copy_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_ERF][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_erf_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_erf_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_erf_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_erf_c<double>};

    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_copy_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copy_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copy_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copy_c<float>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copy_c<double>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_C128][eft_C128] = {eft_C128, (void*)dpnp_copy_c<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_negative_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_negative_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_negative_c<float>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_negative_c<double>};

    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_recip_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_recip_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_recip_c<float>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_recip_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_sign_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_sign_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sign_c<float>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sign_c<double>};

    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_square_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_square_c<int64_t>};
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
    template <typename _KernelNameSpecialization1,                                                                     \
              typename _KernelNameSpecialization2,                                                                     \
              typename _KernelNameSpecialization3>                                                                     \
    class __name__##_broadcast_kernel;                                                                                 \
                                                                                                                       \
    template <typename _KernelNameSpecialization1,                                                                     \
              typename _KernelNameSpecialization2,                                                                     \
              typename _KernelNameSpecialization3>                                                                     \
    class __name__##_strides_kernel;                                                                                   \
                                                                                                                       \
    template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>                         \
    void __name__(void* result_out,                                                                                    \
                  const size_t result_size,                                                                            \
                  const size_t result_ndim,                                                                            \
                  const shape_elem_type* result_shape,                                                                 \
                  const shape_elem_type* result_strides,                                                               \
                  const void* input1_in,                                                                               \
                  const size_t input1_size,                                                                            \
                  const size_t input1_ndim,                                                                            \
                  const shape_elem_type* input1_shape,                                                                 \
                  const shape_elem_type* input1_strides,                                                               \
                  const void* input2_in,                                                                               \
                  const size_t input2_size,                                                                            \
                  const size_t input2_ndim,                                                                            \
                  const shape_elem_type* input2_shape,                                                                 \
                  const shape_elem_type* input2_strides,                                                               \
                  const size_t* where)                                                                                 \
    {                                                                                                                  \
        /* avoid warning unused variable*/                                                                             \
        (void)where;                                                                                                   \
                                                                                                                       \
        if (!input1_size || !input2_size)                                                                              \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType_input1> input1_ptr(input1_in, input1_size);                                        \
        DPNPC_ptr_adapter<shape_elem_type> input1_shape_ptr(input1_shape, input1_ndim, true);                          \
        DPNPC_ptr_adapter<shape_elem_type> input1_strides_ptr(input1_strides, input1_ndim, true);                      \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType_input2> input2_ptr(input2_in, input2_size);                                        \
        DPNPC_ptr_adapter<shape_elem_type> input2_shape_ptr(input2_shape, input2_ndim, true);                          \
        DPNPC_ptr_adapter<shape_elem_type> input2_strides_ptr(input2_strides, input2_ndim, true);                      \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType_output> result_ptr(result_out, result_size, false, true);                          \
        DPNPC_ptr_adapter<shape_elem_type> result_shape_ptr(result_shape, result_ndim);                                \
        DPNPC_ptr_adapter<shape_elem_type> result_strides_ptr(result_strides, result_ndim);                            \
                                                                                                                       \
        _DataType_input1* input1_data = input1_ptr.get_ptr();                                                          \
        shape_elem_type* input1_shape_data = input1_shape_ptr.get_ptr();                                               \
        shape_elem_type* input1_strides_data = input1_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        _DataType_input2* input2_data = input2_ptr.get_ptr();                                                          \
        shape_elem_type* input2_shape_data = input2_shape_ptr.get_ptr();                                               \
        shape_elem_type* input2_strides_data = input2_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        _DataType_output* result = result_ptr.get_ptr();                                                               \
        shape_elem_type* result_shape_data = result_shape_ptr.get_ptr();                                               \
        shape_elem_type* result_strides_data = result_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        bool use_broadcasting = !array_equal(input1_shape_data, input1_ndim, input2_shape_data, input2_ndim);          \
                                                                                                                       \
        const size_t input1_shape_size_in_bytes = input1_ndim * sizeof(shape_elem_type);                               \
        shape_elem_type* input1_shape_offsets =                                                                        \
            reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(input1_shape_size_in_bytes));                       \
        get_shape_offsets_inkernel(input1_shape_data, input1_ndim, input1_shape_offsets);                              \
        bool use_strides = !array_equal(input1_strides_data, input1_ndim, input1_shape_offsets, input1_ndim);          \
        dpnp_memory_free_c(input1_shape_offsets);                                                                      \
                                                                                                                       \
        const size_t input2_shape_size_in_bytes = input2_ndim * sizeof(shape_elem_type);                               \
        shape_elem_type* input2_shape_offsets =                                                                        \
        reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(input2_shape_size_in_bytes));                           \
        get_shape_offsets_inkernel(input2_shape_data, input2_ndim, input2_shape_offsets);                              \
        use_strides = use_strides || !array_equal(input2_strides_data, input2_ndim, input2_shape_offsets, input2_ndim);\
        dpnp_memory_free_c(input2_shape_offsets);                                                                      \
                                                                                                                       \
        cl::sycl::event event;                                                                                         \
        cl::sycl::range<1> gws(result_size);                                                                           \
                                                                                                                       \
        if (use_broadcasting)                                                                                          \
        {                                                                                                              \
            DPNPC_id<_DataType_input1>* input1_it;                                                                     \
            const size_t input1_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input1>);                                 \
            input1_it = reinterpret_cast<DPNPC_id<_DataType_input1>*>(dpnp_memory_alloc_c(input1_it_size_in_bytes));   \
            new (input1_it) DPNPC_id<_DataType_input1>(input1_data,                                                    \
                                                       input1_shape_data,                                              \
                                                       input1_strides_data,                                            \
                                                       input1_ndim);                                                   \
                                                                                                                       \
            input1_it->broadcast_to_shape(result_shape_data, result_ndim);                                             \
                                                                                                                       \
            DPNPC_id<_DataType_input2>* input2_it;                                                                     \
            const size_t input2_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input2>);                                 \
            input2_it = reinterpret_cast<DPNPC_id<_DataType_input2>*>(dpnp_memory_alloc_c(input2_it_size_in_bytes));   \
            new (input2_it) DPNPC_id<_DataType_input2>(input2_data,                                                    \
                                                       input2_shape_data,                                              \
                                                       input2_strides_data,                                            \
                                                       input2_ndim);                                                   \
                                                                                                                       \
            input2_it->broadcast_to_shape(result_shape_data, result_ndim);                                             \
                                                                                                                       \
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                           \
                const size_t i = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                            \
                {                                                                                                      \
                    const _DataType_output input1_elem = (*input1_it)[i];                                              \
                    const _DataType_output input2_elem = (*input2_it)[i];                                              \
                    result[i] = __operation1__;                                                                        \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class __name__##_broadcast_kernel<_DataType_output,                                   \
                                                                   _DataType_input1,                                   \
                                                                   _DataType_input2>>(gws, kernel_parallel_for_func);  \
            };                                                                                                         \
                                                                                                                       \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
            event.wait();                                                                                              \
                                                                                                                       \
            input1_it->~DPNPC_id();                                                                                    \
            input2_it->~DPNPC_id();                                                                                    \
        }                                                                                                              \
        else if (use_strides)                                                                                          \
        {                                                                                                              \
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                           \
                const size_t output_id = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                    \
                {                                                                                                      \
                    size_t input1_id = 0;                                                                              \
                    size_t input2_id = 0;                                                                              \
                    for (size_t i = 0; i < result_ndim; ++i)                                                           \
                    {                                                                                                  \
                        const size_t output_xyz_id = get_xyz_id_by_id_inkernel(output_id,                              \
                                                                               result_strides_data,                    \
                                                                               result_ndim,                            \
                                                                               i);                                     \
                        input1_id += output_xyz_id * input1_strides_data[i];                                           \
                        input2_id += output_xyz_id * input2_strides_data[i];                                           \
                    }                                                                                                  \
                                                                                                                       \
                    const _DataType_output input1_elem = input1_data[input1_id];                                       \
                    const _DataType_output input2_elem = input2_data[input2_id];                                       \
                    result[output_id] = __operation1__;                                                                \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class __name__##_strides_kernel<_DataType_output,                                     \
                                                                 _DataType_input1,                                     \
                                                                 _DataType_input2>>(gws, kernel_parallel_for_func);    \
            };                                                                                                         \
                                                                                                                       \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
            event.wait();                                                                                              \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            if constexpr ((std::is_same<_DataType_input1, double>::value ||                                            \
                           std::is_same<_DataType_input1, float>::value) &&                                            \
                          std::is_same<_DataType_input2, _DataType_input1>::value)                                     \
            {                                                                                                          \
                event = __operation2__(DPNP_QUEUE, result_size, input1_data, input2_data, result);                     \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                       \
                    const size_t i = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                        \
                    {                                                                                                  \
                        const _DataType_output input1_elem = input1_data[i];                                           \
                        const _DataType_output input2_elem = input2_data[i];                                           \
                        result[i] = __operation1__;                                                                    \
                    }                                                                                                  \
                };                                                                                                     \
                auto kernel_func = [&](cl::sycl::handler& cgh) {                                                       \
                    cgh.parallel_for<class __name__##_kernel<_DataType_output,                                         \
                                                             _DataType_input1,                                         \
                                                             _DataType_input2>>(gws, kernel_parallel_for_func);        \
                };                                                                                                     \
                event = DPNP_QUEUE.submit(kernel_func);                                                                \
            }                                                                                                          \
            event.wait();                                                                                              \
        }                                                                                                              \
    }

#include <dpnp_gen_2arg_3type_tbl.hpp>

static void func_map_init_elemwise_2arg_3type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_add_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_add_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_add_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_add_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_add_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_add_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_add_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_add_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_add_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_add_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_add_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_add_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_add_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_add_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_add_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_add_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_c<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_arctan2_c<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_arctan2_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_arctan2_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_c<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_arctan2_c<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_arctan2_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_arctan2_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_arctan2_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_arctan2_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_arctan2_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_arctan2_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_arctan2_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_arctan2_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_arctan2_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_copysign_c<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_copysign_c<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_copysign_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_copysign_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_copysign_c<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_copysign_c<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_copysign_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_copysign_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_copysign_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_copysign_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copysign_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_copysign_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_copysign_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_copysign_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_copysign_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copysign_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_divide_c<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_divide_c<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_divide_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_divide_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_divide_c<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_divide_c<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_divide_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_divide_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_divide_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_divide_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_divide_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_divide_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_divide_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_divide_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_divide_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_divide_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_fmod_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_fmod_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_fmod_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_fmod_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_fmod_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_fmod_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_fmod_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_fmod_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_fmod_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_fmod_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_fmod_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_fmod_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_fmod_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_hypot_c<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_c<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_hypot_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_hypot_c<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_c<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_hypot_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_hypot_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_hypot_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_hypot_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_hypot_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_hypot_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_hypot_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_maximum_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_maximum_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_maximum_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_maximum_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_maximum_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_maximum_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_maximum_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_maximum_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_maximum_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_maximum_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_maximum_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_maximum_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_maximum_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_maximum_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_maximum_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_maximum_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_minimum_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_minimum_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_minimum_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_minimum_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_minimum_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_minimum_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_minimum_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_minimum_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_minimum_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_minimum_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_minimum_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_minimum_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_minimum_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_minimum_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_minimum_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_minimum_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_multiply_c<bool, bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_INT] = {eft_INT, (void*)dpnp_multiply_c<int32_t, bool, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_LNG] = {eft_LNG, (void*)dpnp_multiply_c<int64_t, bool, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_FLT] = {eft_FLT, (void*)dpnp_multiply_c<float, bool, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, bool, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_BLN] = {eft_INT, (void*)dpnp_multiply_c<int32_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_multiply_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_multiply_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_multiply_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_BLN] = {eft_LNG, (void*)dpnp_multiply_c<int64_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_multiply_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_multiply_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_multiply_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_BLN] = {eft_FLT, (void*)dpnp_multiply_c<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_multiply_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_multiply_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_multiply_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_BLN] = {eft_DBL, (void*)dpnp_multiply_c<double, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_multiply_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_multiply_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_multiply_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_multiply_c<std::complex<double>, std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_power_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_power_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_power_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_power_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_power_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_power_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_power_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_power_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_power_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_power_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_power_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_power_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_power_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_power_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_power_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_power_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_subtract_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_subtract_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_subtract_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_subtract_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_subtract_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_subtract_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_subtract_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_subtract_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_subtract_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_subtract_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_subtract_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_subtract_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_subtract_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_subtract_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_subtract_c<double, double, float>};
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
