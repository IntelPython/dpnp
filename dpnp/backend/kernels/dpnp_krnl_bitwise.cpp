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
#include "dpnp_iface.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class dpnp_invert_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_invert_c(DPCTLSyclQueueRef q_ref,
                                void* array1_in,
                                void* result1,
                                size_t size,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    sycl::event event;

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, size);
    _DataType* array1 = input1_ptr.get_ptr();
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
        {
            _DataType input_elem1 = array1[i];
            result[i] = ~input_elem1;
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_invert_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_invert_c(void* array1_in, void* result1, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_invert_c<_DataType>(q_ref,
                                                           array1_in,
                                                           result1,
                                                           size,
                                                           dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType>
void (*dpnp_invert_default_c)(void*, void*, size_t) = dpnp_invert_c<_DataType>;

template <typename _DataType>
DPCTLSyclEventRef (*dpnp_invert_ext_c)(DPCTLSyclQueueRef,
                                       void*,
                                       void*,
                                       size_t,
                                       const DPCTLEventVectorRef) = dpnp_invert_c<_DataType>;

static void func_map_init_bitwise_1arg_1type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_INVERT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_invert_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_INVERT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_invert_default_c<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_INVERT_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_invert_ext_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_INVERT_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_invert_ext_c<int64_t>};

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
    DPCTLSyclEventRef __name__(DPCTLSyclQueueRef q_ref,                                                                \
                               void* result_out,                                                                       \
                               const size_t result_size,                                                               \
                               const size_t result_ndim,                                                               \
                               const shape_elem_type* result_shape,                                                    \
                               const shape_elem_type* result_strides,                                                  \
                               const void* input1_in,                                                                  \
                               const size_t input1_size,                                                               \
                               const size_t input1_ndim,                                                               \
                               const shape_elem_type* input1_shape,                                                    \
                               const shape_elem_type* input1_strides,                                                  \
                               const void* input2_in,                                                                  \
                               const size_t input2_size,                                                               \
                               const size_t input2_ndim,                                                               \
                               const shape_elem_type* input2_shape,                                                    \
                               const shape_elem_type* input2_strides,                                                  \
                               const size_t* where,                                                                    \
                               const DPCTLEventVectorRef dep_event_vec_ref)                                            \
    {                                                                                                                  \
        /* avoid warning unused variable*/                                                                             \
        (void)result_shape;                                                                                            \
        (void)where;                                                                                                   \
        (void)dep_event_vec_ref;                                                                                       \
                                                                                                                       \
        DPCTLSyclEventRef event_ref = nullptr;                                                                         \
                                                                                                                       \
        if (!input1_size || !input2_size)                                                                              \
        {                                                                                                              \
            return event_ref;                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));                                                      \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, input1_in, input1_size);                                        \
        DPNPC_ptr_adapter<shape_elem_type> input1_shape_ptr(q_ref, input1_shape, input1_ndim, true);                   \
        DPNPC_ptr_adapter<shape_elem_type> input1_strides_ptr(q_ref, input1_strides, input1_ndim, true);               \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType> input2_ptr(q_ref, input2_in, input2_size);                                        \
        DPNPC_ptr_adapter<shape_elem_type> input2_shape_ptr(q_ref, input2_shape, input2_ndim, true);                   \
        DPNPC_ptr_adapter<shape_elem_type> input2_strides_ptr(q_ref, input2_strides, input2_ndim, true);               \
                                                                                                                       \
        DPNPC_ptr_adapter<_DataType> result_ptr(q_ref, result_out, result_size, false, true);                          \
        DPNPC_ptr_adapter<shape_elem_type> result_strides_ptr(q_ref, result_strides, result_ndim);                     \
                                                                                                                       \
        _DataType* input1_data = input1_ptr.get_ptr();                                                                 \
        shape_elem_type* input1_shape_data = input1_shape_ptr.get_ptr();                                               \
        shape_elem_type* input1_strides_data = input1_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        _DataType* input2_data = input2_ptr.get_ptr();                                                                 \
        shape_elem_type* input2_shape_data = input2_shape_ptr.get_ptr();                                               \
        shape_elem_type* input2_strides_data = input2_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        _DataType* result = result_ptr.get_ptr();                                                                      \
        shape_elem_type* result_strides_data = result_strides_ptr.get_ptr();                                           \
                                                                                                                       \
        const size_t input1_shape_size_in_bytes = input1_ndim * sizeof(shape_elem_type);                               \
        shape_elem_type* input1_shape_offsets =                                                                        \
            reinterpret_cast<shape_elem_type*>(sycl::malloc_shared(input1_shape_size_in_bytes, q));                    \
        get_shape_offsets_inkernel(input1_shape_data, input1_ndim, input1_shape_offsets);                              \
        bool use_strides = !array_equal(input1_strides_data, input1_ndim, input1_shape_offsets, input1_ndim);          \
        sycl::free(input1_shape_offsets, q);                                                                           \
                                                                                                                       \
        const size_t input2_shape_size_in_bytes = input2_ndim * sizeof(shape_elem_type);                               \
        shape_elem_type* input2_shape_offsets =                                                                        \
            reinterpret_cast<shape_elem_type*>(sycl::malloc_shared(input2_shape_size_in_bytes, q));                    \
        get_shape_offsets_inkernel(input2_shape_data, input2_ndim, input2_shape_offsets);                              \
        use_strides =                                                                                                  \
            use_strides || !array_equal(input2_strides_data, input2_ndim, input2_shape_offsets, input2_ndim);          \
        sycl::free(input2_shape_offsets, q);                                                                           \
                                                                                                                       \
        sycl::event event;                                                                                             \
        sycl::range<1> gws(result_size);                                                                               \
                                                                                                                       \
        if (use_strides)                                                                                               \
        {                                                                                                              \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {                                               \
                const size_t output_id = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                    \
                {                                                                                                      \
                    size_t input1_id = 0;                                                                              \
                    size_t input2_id = 0;                                                                              \
                    for (size_t i = 0; i < result_ndim; ++i)                                                           \
                    {                                                                                                  \
                        const size_t output_xyz_id =                                                                   \
                            get_xyz_id_by_id_inkernel(output_id, result_strides_data, result_ndim, i);                 \
                        input1_id += output_xyz_id * input1_strides_data[i];                                           \
                        input2_id += output_xyz_id * input2_strides_data[i];                                           \
                    }                                                                                                  \
                                                                                                                       \
                    const _DataType input1_elem = (input1_size == 1) ? input1_data[0] : input1_data[input1_id];        \
                    const _DataType input2_elem = (input2_size == 1) ? input2_data[0] : input2_data[input2_id];        \
                    result[output_id] = __operation__;                                                                 \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](sycl::handler& cgh) {                                                               \
                cgh.parallel_for<class __name__##_strides_kernel<_DataType>>(gws, kernel_parallel_for_func);           \
            };                                                                                                         \
            event = q.submit(kernel_func);                                                                             \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {                                               \
                size_t i = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/                                  \
                const _DataType input1_elem = (input1_size == 1) ? input1_data[0] : input1_data[i];                    \
                const _DataType input2_elem = (input2_size == 1) ? input2_data[0] : input2_data[i];                    \
                result[i] = __operation__;                                                                             \
            };                                                                                                         \
            auto kernel_func = [&](sycl::handler& cgh) {                                                               \
                cgh.parallel_for<class __name__##_kernel<_DataType>>(gws, kernel_parallel_for_func);                   \
            };                                                                                                         \
            event = q.submit(kernel_func);                                                                             \
        }                                                                                                              \
        input1_ptr.depends_on(event);                                                                                  \
        input1_shape_ptr.depends_on(event);                                                                            \
        input1_strides_ptr.depends_on(event);                                                                          \
        input2_ptr.depends_on(event);                                                                                  \
        input2_shape_ptr.depends_on(event);                                                                            \
        input2_strides_ptr.depends_on(event);                                                                          \
        result_ptr.depends_on(event);                                                                                  \
        result_strides_ptr.depends_on(event);                                                                          \
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);                                                       \
                                                                                                                       \
        return DPCTLEvent_Copy(event_ref);                                                                             \
    }                                                                                                                  \
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
                  const void* input2_in,                                                                               \
                  const size_t input2_size,                                                                            \
                  const size_t input2_ndim,                                                                            \
                  const shape_elem_type* input2_shape,                                                                 \
                  const shape_elem_type* input2_strides,                                                               \
                  const size_t* where)                                                                                 \
    {                                                                                                                  \
        DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);                                    \
        DPCTLEventVectorRef dep_event_vec_ref = nullptr;                                                               \
        DPCTLSyclEventRef event_ref = __name__<_DataType>(q_ref,                                                       \
                                                          result_out,                                                  \
                                                          result_size,                                                 \
                                                          result_ndim,                                                 \
                                                          result_shape,                                                \
                                                          result_strides,                                              \
                                                          input1_in,                                                   \
                                                          input1_size,                                                 \
                                                          input1_ndim,                                                 \
                                                          input1_shape,                                                \
                                                          input1_strides,                                              \
                                                          input2_in,                                                   \
                                                          input2_size,                                                 \
                                                          input2_ndim,                                                 \
                                                          input2_shape,                                                \
                                                          input2_strides,                                              \
                                                          where,                                                       \
                                                          dep_event_vec_ref);                                          \
        DPCTLEvent_WaitAndThrow(event_ref);                                                                            \
    }                                                                                                                  \
                                                                                                                       \
    template <typename _DataType>                                                                                      \
    void (*__name__##_default)(void*,                                                                                  \
                               const size_t,                                                                           \
                               const size_t,                                                                           \
                               const shape_elem_type*,                                                                 \
                               const shape_elem_type*,                                                                 \
                               const void*,                                                                            \
                               const size_t,                                                                           \
                               const size_t,                                                                           \
                               const shape_elem_type*,                                                                 \
                               const shape_elem_type*,                                                                 \
                               const void*,                                                                            \
                               const size_t,                                                                           \
                               const size_t,                                                                           \
                               const shape_elem_type*,                                                                 \
                               const shape_elem_type*,                                                                 \
                               const size_t*) = __name__<_DataType>;                                                   \
                                                                                                                       \
    template <typename _DataType>                                                                                      \
    DPCTLSyclEventRef (*__name__##_ext)(DPCTLSyclQueueRef,                                                             \
                                        void*,                                                                         \
                                        const size_t,                                                                  \
                                        const size_t,                                                                  \
                                        const shape_elem_type*,                                                        \
                                        const shape_elem_type*,                                                        \
                                        const void*,                                                                   \
                                        const size_t,                                                                  \
                                        const size_t,                                                                  \
                                        const shape_elem_type*,                                                        \
                                        const shape_elem_type*,                                                        \
                                        const void*,                                                                   \
                                        const size_t,                                                                  \
                                        const size_t,                                                                  \
                                        const shape_elem_type*,                                                        \
                                        const shape_elem_type*,                                                        \
                                        const size_t*,                                                                 \
                                        const DPCTLEventVectorRef) = __name__<_DataType>;

#include <dpnp_gen_2arg_1type_tbl.hpp>

static void func_map_init_bitwise_2arg_1type(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_and_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_and_c_default<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_and_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_and_c_ext<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_or_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_or_c_default<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_or_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_or_c_ext<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_xor_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_xor_c_default<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_bitwise_xor_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_bitwise_xor_c_ext<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_left_shift_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_left_shift_c_default<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_left_shift_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_left_shift_c_ext<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_right_shift_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_right_shift_c_default<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_right_shift_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_right_shift_c_ext<int64_t>};

    return;
}

void func_map_init_bitwise(func_map_t& fmap)
{
    func_map_init_bitwise_1arg_1type(fmap);
    func_map_init_bitwise_2arg_1type(fmap);

    return;
}
