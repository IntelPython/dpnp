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

#include <iostream>

#include "dpnp_fptr.hpp"
#include "dpnp_utils.hpp"
#include "dpnp_iface.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class dpnp_invert_c_kernel;

template <typename _DataType>
void dpnp_invert_c(void* array1_in, void* result1, size_t size)
{
    cl::sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size);
    _DataType* array1 = input1_ptr.get_ptr();
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
        {
            _DataType input_elem1 = array1[i];
            result[i] = ~input_elem1;
        }
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_invert_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

static void func_map_init_bitwise_1arg_1type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_INVERT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_invert_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_INVERT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_invert_c<int64_t>};

    return;
}

#define MACRO_2ARG_1TYPE_OP(__name__, __operation__)                                                                   \
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
                  const size_t* result_shape,                                                                          \
                  const size_t* result_strides,                                                                        \
                  const void* input1_in,                                                                               \
                  const size_t input1_size,                                                                            \
                  const size_t input1_ndim,                                                                            \
                  const size_t* input1_shape,                                                                          \
                  const size_t* input1_strides,                                                                        \
                  const void* input2_in,                                                                               \
                  const size_t input2_size,                                                                            \
                  const size_t input2_ndim,                                                                            \
                  const size_t* input2_shape,                                                                          \
                  const size_t* input2_strides,                                                                        \
                  const size_t* where)                                                                                 \
    {                                                                                                                  \
        /* avoid warning unused variable*/                                                                             \
        (void)result_shape;                                                                                            \
        (void)where;                                                                                                   \
                                                                                                                       \
        if (!input1_size || !input2_size)                                                                              \
        {                                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType> input1_ptr(input1_in, input1_size);                                               \
        DPNPC_ptr_adapter<size_t> input1_shape_ptr(input1_shape, input1_ndim, true);                                   \
        DPNPC_ptr_adapter<size_t> input1_strides_ptr(input1_strides, input1_ndim, true);                               \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType> input2_ptr(input2_in, input2_size);                                               \
        DPNPC_ptr_adapter<size_t> input2_shape_ptr(input2_shape, input2_ndim, true);                                   \
        DPNPC_ptr_adapter<size_t> input2_strides_ptr(input2_strides, input2_ndim, true);                               \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType> result_ptr(result_out, result_size, false, true);                                 \
        DPNPC_ptr_adapter<size_t> result_strides_ptr(result_strides, result_ndim);                                     \
                                                                                                                       \
        _DataType* input1_data = input1_ptr.get_ptr();                                                                 \
        size_t* input1_shape_data = input1_shape_ptr.get_ptr();                                                        \
        size_t* input1_strides_data = input1_strides_ptr.get_ptr();                                                    \
                                                                                                                       \
        _DataType* input2_data = input2_ptr.get_ptr();                                                                 \
        size_t* input2_shape_data = input2_shape_ptr.get_ptr();                                                        \
        size_t* input2_strides_data = input2_strides_ptr.get_ptr();                                                    \
                                                                                                                       \
        _DataType* result = result_ptr.get_ptr();                                                                      \
        size_t* result_strides_data = result_strides_ptr.get_ptr();                                                    \
                                                                                                                       \
        const size_t input1_shape_size_in_bytes = input1_ndim * sizeof(size_t);                                        \
        size_t* input1_shape_offsets = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(input1_shape_size_in_bytes));     \
        get_shape_offsets_inkernel(input1_shape_data, input1_ndim, input1_shape_offsets);                              \
        bool use_strides = !array_equal(input1_strides_data, input1_ndim, input1_shape_offsets, input1_ndim);          \
        dpnp_memory_free_c(input1_shape_offsets);                                                                      \
                                                                                                                       \
        const size_t input2_shape_size_in_bytes = input2_ndim * sizeof(size_t);                                        \
        size_t* input2_shape_offsets = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(input2_shape_size_in_bytes));     \
        get_shape_offsets_inkernel(input2_shape_data, input2_ndim, input2_shape_offsets);                              \
        use_strides = use_strides || !array_equal(input2_strides_data, input2_ndim, input2_shape_offsets, input2_ndim);\
        dpnp_memory_free_c(input2_shape_offsets);                                                                      \
                                                                                                                       \
        cl::sycl::event event;                                                                                         \
        cl::sycl::range<1> gws(result_size);                                                                           \
                                                                                                                       \
        if (use_strides)                                                                                               \
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
                    const _DataType input1_elem = (input1_size == 1) ? input1_data[0] : input1_data[input1_id];        \
                    const _DataType input2_elem = (input2_size == 1) ? input2_data[0] : input2_data[input2_id];        \
                    result[output_id] = __operation__;                                                                 \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class __name__##_strides_kernel<_DataType>>(gws, kernel_parallel_for_func);           \
            };                                                                                                         \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
            event.wait();                                                                                              \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                           \
                size_t i = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                                  \
                const _DataType input1_elem = (input1_size == 1) ? input1_data[0] : input1_data[i];                    \
                const _DataType input2_elem = (input2_size == 1) ? input2_data[0] : input2_data[i];                    \
                result[i] = __operation__;                                                                             \
            };                                                                                                         \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class __name__##_kernel<_DataType>>(gws, kernel_parallel_for_func);                   \
            };                                                                                                         \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
            event.wait();                                                                                              \
        }                                                                                                              \
    }

#include <dpnp_gen_2arg_1type_tbl.hpp>

static void func_map_init_bitwise_2arg_1type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_and_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_and_c<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_or_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_or_c<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_xor_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_xor_c<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_left_shift_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_left_shift_c<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_right_shift_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_right_shift_c<int64_t>};

    return;
}

void func_map_init_bitwise(func_map_t& fmap)
{
    func_map_init_bitwise_1arg_1type(fmap);
    func_map_init_bitwise_2arg_1type(fmap);

    return;
}
