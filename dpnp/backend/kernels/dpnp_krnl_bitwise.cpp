//*****************************************************************************
// Copyright (c) 2016-2023, Intel Corporation
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
#include "dpnp_iterator.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

// dpctl tensor headers
#include "kernels/alignment.hpp"

using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

template <typename _KernelNameSpecialization>
class dpnp_invert_c_kernel;

template <typename _DataType>
DPCTLSyclEventRef dpnp_invert_c(DPCTLSyclQueueRef q_ref,
                                void *array1_in,
                                void *result1,
                                size_t size,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));
    sycl::event event;

    _DataType *input_data = static_cast<_DataType *>(array1_in);
    _DataType *result = static_cast<_DataType *>(result1);

    constexpr size_t lws = 64;
    constexpr unsigned int vec_sz = 8;

    auto gws_range =
        sycl::range<1>(((size + lws * vec_sz - 1) / (lws * vec_sz)) * lws);
    auto lws_range = sycl::range<1>(lws);

    auto kernel_parallel_for_func = [=](sycl::nd_item<1> nd_it) {
        auto sg = nd_it.get_sub_group();
        const auto max_sg_size = sg.get_max_local_range()[0];
        const size_t start =
            vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) +
                      sg.get_group_id()[0] * max_sg_size);

        if (is_aligned<required_alignment>(input_data) &&
            is_aligned<required_alignment>(result) &&
            (start + static_cast<size_t>(vec_sz) * max_sg_size < size))
        {
            auto input_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&input_data[start]);
            auto result_multi_ptr = sycl::address_space_cast<
                sycl::access::address_space::global_space,
                sycl::access::decorated::yes>(&result[start]);

            sycl::vec<_DataType, vec_sz> x = sg.load<vec_sz>(input_multi_ptr);
            sycl::vec<_DataType, vec_sz> res_vec;

            if constexpr (std::is_same_v<_DataType, bool>) {
#pragma unroll
                for (size_t k = 0; k < vec_sz; ++k) {
                    res_vec[k] = !(x[k]);
                }
            }
            else {
                res_vec = ~x;
            }

            sg.store<vec_sz>(result_multi_ptr, res_vec);
        }
        else {
            for (size_t k = start + sg.get_local_id()[0]; k < size;
                 k += max_sg_size) {
                if constexpr (std::is_same_v<_DataType, bool>) {
                    result[k] = !(input_data[k]);
                }
                else {
                    result[k] = ~(input_data[k]);
                }
            }
        }
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.parallel_for<class dpnp_invert_c_kernel<_DataType>>(
            sycl::nd_range<1>(gws_range, lws_range), kernel_parallel_for_func);
    };
    event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType>
void dpnp_invert_c(void *array1_in, void *result1, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_invert_c<_DataType>(
        q_ref, array1_in, result1, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType>
void (*dpnp_invert_default_c)(void *,
                              void *,
                              size_t) = dpnp_invert_c<_DataType>;

static void func_map_init_bitwise_1arg_1type(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_INVERT][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_invert_default_c<bool>};
    fmap[DPNPFuncName::DPNP_FN_INVERT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_invert_default_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_INVERT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_invert_default_c<int64_t>};

    return;
}

#define MACRO_2ARG_1TYPE_OP(__name__, __operation__)                           \
    template <typename _KernelNameSpecialization>                              \
    class __name__##_kernel;                                                   \
                                                                               \
    template <typename _KernelNameSpecialization>                              \
    class __name__##_strides_kernel;                                           \
                                                                               \
    template <typename _KernelNameSpecialization>                              \
    class __name__##_broadcast_kernel;                                         \
                                                                               \
    template <typename _DataType>                                              \
    DPCTLSyclEventRef __name__(                                                \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const void *input2_in,          \
        const size_t input2_size, const size_t input2_ndim,                    \
        const shape_elem_type *input2_shape,                                   \
        const shape_elem_type *input2_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref)                           \
    {                                                                          \
        /* avoid warning unused variable*/                                     \
        (void)result_shape;                                                    \
        (void)where;                                                           \
        (void)dep_event_vec_ref;                                               \
                                                                               \
        DPCTLSyclEventRef event_ref = nullptr;                                 \
                                                                               \
        if (!input1_size || !input2_size) {                                    \
            return event_ref;                                                  \
        }                                                                      \
                                                                               \
        sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));             \
                                                                               \
        _DataType *input1_data =                                               \
            static_cast<_DataType *>(const_cast<void *>(input1_in));           \
        _DataType *input2_data =                                               \
            static_cast<_DataType *>(const_cast<void *>(input2_in));           \
        _DataType *result = static_cast<_DataType *>(result_out);              \
                                                                               \
        bool use_broadcasting = !array_equal(input1_shape, input1_ndim,        \
                                             input2_shape, input2_ndim);       \
                                                                               \
        shape_elem_type *input1_shape_offsets =                                \
            new shape_elem_type[input1_ndim];                                  \
                                                                               \
        get_shape_offsets_inkernel(input1_shape, input1_ndim,                  \
                                   input1_shape_offsets);                      \
        bool use_strides = !array_equal(input1_strides, input1_ndim,           \
                                        input1_shape_offsets, input1_ndim);    \
        delete[] input1_shape_offsets;                                         \
                                                                               \
        shape_elem_type *input2_shape_offsets =                                \
            new shape_elem_type[input2_ndim];                                  \
                                                                               \
        get_shape_offsets_inkernel(input2_shape, input2_ndim,                  \
                                   input2_shape_offsets);                      \
        use_strides =                                                          \
            use_strides || !array_equal(input2_strides, input2_ndim,           \
                                        input2_shape_offsets, input2_ndim);    \
        delete[] input2_shape_offsets;                                         \
                                                                               \
        sycl::event event;                                                     \
        sycl::range<1> gws(result_size);                                       \
                                                                               \
        if (use_broadcasting) {                                                \
            DPNPC_id<_DataType> *input1_it;                                    \
            const size_t input1_it_size_in_bytes =                             \
                sizeof(DPNPC_id<_DataType>);                                   \
            input1_it = reinterpret_cast<DPNPC_id<_DataType> *>(               \
                dpnp_memory_alloc_c(q_ref, input1_it_size_in_bytes));          \
            new (input1_it)                                                    \
                DPNPC_id<_DataType>(q_ref, input1_data, input1_shape,          \
                                    input1_strides, input1_ndim);              \
                                                                               \
            input1_it->broadcast_to_shape(result_shape, result_ndim);          \
                                                                               \
            DPNPC_id<_DataType> *input2_it;                                    \
            const size_t input2_it_size_in_bytes =                             \
                sizeof(DPNPC_id<_DataType>);                                   \
            input2_it = reinterpret_cast<DPNPC_id<_DataType> *>(               \
                dpnp_memory_alloc_c(q_ref, input2_it_size_in_bytes));          \
            new (input2_it)                                                    \
                DPNPC_id<_DataType>(q_ref, input2_data, input2_shape,          \
                                    input2_strides, input2_ndim);              \
                                                                               \
            input2_it->broadcast_to_shape(result_shape, result_ndim);          \
                                                                               \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {       \
                const size_t i = global_id[0]; /* for (size_t i = 0; i <       \
                                                  result_size; ++i) */         \
                {                                                              \
                    const _DataType input1_elem = (*input1_it)[i];             \
                    const _DataType input2_elem = (*input2_it)[i];             \
                    result[i] = __operation__;                                 \
                }                                                              \
            };                                                                 \
            auto kernel_func = [&](sycl::handler &cgh) {                       \
                cgh.parallel_for<                                              \
                    class __name__##_broadcast_kernel<_DataType>>(             \
                    gws, kernel_parallel_for_func);                            \
            };                                                                 \
                                                                               \
            q.submit(kernel_func).wait();                                      \
                                                                               \
            input1_it->~DPNPC_id();                                            \
            input2_it->~DPNPC_id();                                            \
                                                                               \
            return event_ref;                                                  \
        }                                                                      \
        else if (use_strides) {                                                \
            if ((result_ndim != input1_ndim) || (result_ndim != input2_ndim))  \
            {                                                                  \
                throw std::runtime_error(                                      \
                    "Result ndim=" + std::to_string(result_ndim) +             \
                    " mismatches with either input1 ndim=" +                   \
                    std::to_string(input1_ndim) +                              \
                    " or input2 ndim=" + std::to_string(input2_ndim));         \
            }                                                                  \
                                                                               \
            /* memory transfer optimization, use USM-host for temporary speeds \
             * up transfer to device */                                        \
            using usm_host_allocatorT =                                        \
                sycl::usm_allocator<shape_elem_type, sycl::usm::alloc::host>;  \
                                                                               \
            size_t strides_size = 3 * result_ndim;                             \
            shape_elem_type *dev_strides_data =                                \
                sycl::malloc_device<shape_elem_type>(strides_size, q);         \
                                                                               \
            /* create host temporary for packed strides managed by shared      \
             * pointer */                                                      \
            auto strides_host_packed =                                         \
                std::vector<shape_elem_type, usm_host_allocatorT>(             \
                    strides_size, usm_host_allocatorT(q));                     \
                                                                               \
            /* packed vector is concatenation of result_strides,               \
             * input1_strides and input2_strides */                            \
            std::copy(result_strides, result_strides + result_ndim,            \
                      strides_host_packed.begin());                            \
            std::copy(input1_strides, input1_strides + result_ndim,            \
                      strides_host_packed.begin() + result_ndim);              \
            std::copy(input2_strides, input2_strides + result_ndim,            \
                      strides_host_packed.begin() + 2 * result_ndim);          \
                                                                               \
            auto copy_strides_ev = q.copy<shape_elem_type>(                    \
                strides_host_packed.data(), dev_strides_data,                  \
                strides_host_packed.size());                                   \
                                                                               \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {       \
                const size_t output_id =                                       \
                    global_id[0]; /* for (size_t i = 0; i < result_size; ++i)  \
                                   */                                          \
                {                                                              \
                    const shape_elem_type *result_strides_data =               \
                        &dev_strides_data[0];                                  \
                    const shape_elem_type *input1_strides_data =               \
                        &dev_strides_data[result_ndim];                        \
                    const shape_elem_type *input2_strides_data =               \
                        &dev_strides_data[2 * result_ndim];                    \
                                                                               \
                    size_t input1_id = 0;                                      \
                    size_t input2_id = 0;                                      \
                                                                               \
                    for (size_t i = 0; i < result_ndim; ++i) {                 \
                        const size_t output_xyz_id =                           \
                            get_xyz_id_by_id_inkernel(output_id,               \
                                                      result_strides_data,     \
                                                      result_ndim, i);         \
                        input1_id += output_xyz_id * input1_strides_data[i];   \
                        input2_id += output_xyz_id * input2_strides_data[i];   \
                    }                                                          \
                                                                               \
                    const _DataType input1_elem =                              \
                        (input1_size == 1) ? input1_data[0]                    \
                                           : input1_data[input1_id];           \
                    const _DataType input2_elem =                              \
                        (input2_size == 1) ? input2_data[0]                    \
                                           : input2_data[input2_id];           \
                    result[output_id] = __operation__;                         \
                }                                                              \
            };                                                                 \
            auto kernel_func = [&](sycl::handler &cgh) {                       \
                cgh.depends_on(copy_strides_ev);                               \
                cgh.parallel_for<class __name__##_strides_kernel<_DataType>>(  \
                    gws, kernel_parallel_for_func);                            \
            };                                                                 \
                                                                               \
            q.submit(kernel_func).wait();                                      \
                                                                               \
            sycl::free(dev_strides_data, q);                                   \
            return event_ref;                                                  \
        }                                                                      \
        else {                                                                 \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {       \
                size_t i = global_id[0]; /* for (size_t i = 0; i <             \
                                            result_size; ++i) */               \
                const _DataType input1_elem =                                  \
                    (input1_size == 1) ? input1_data[0] : input1_data[i];      \
                const _DataType input2_elem =                                  \
                    (input2_size == 1) ? input2_data[0] : input2_data[i];      \
                result[i] = __operation__;                                     \
            };                                                                 \
            auto kernel_func = [&](sycl::handler &cgh) {                       \
                cgh.parallel_for<class __name__##_kernel<_DataType>>(          \
                    gws, kernel_parallel_for_func);                            \
            };                                                                 \
            event = q.submit(kernel_func);                                     \
        }                                                                      \
                                                                               \
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);               \
        return DPCTLEvent_Copy(event_ref);                                     \
    }                                                                          \
                                                                               \
    template <typename _DataType>                                              \
    void __name__(                                                             \
        void *result_out, const size_t result_size, const size_t result_ndim,  \
        const shape_elem_type *result_shape,                                   \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const void *input2_in,          \
        const size_t input2_size, const size_t input2_ndim,                    \
        const shape_elem_type *input2_shape,                                   \
        const shape_elem_type *input2_strides, const size_t *where)            \
    {                                                                          \
        DPCTLSyclQueueRef q_ref =                                              \
            reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);                  \
        DPCTLEventVectorRef dep_event_vec_ref = nullptr;                       \
        DPCTLSyclEventRef event_ref = __name__<_DataType>(                     \
            q_ref, result_out, result_size, result_ndim, result_shape,         \
            result_strides, input1_in, input1_size, input1_ndim, input1_shape, \
            input1_strides, input2_in, input2_size, input2_ndim, input2_shape, \
            input2_strides, where, dep_event_vec_ref);                         \
        DPCTLEvent_WaitAndThrow(event_ref);                                    \
        DPCTLEvent_Delete(event_ref);                                          \
    }                                                                          \
                                                                               \
    template <typename _DataType>                                              \
    void (*__name__##_default)(                                                \
        void *, const size_t, const size_t, const shape_elem_type *,           \
        const shape_elem_type *, const void *, const size_t, const size_t,     \
        const shape_elem_type *, const shape_elem_type *, const void *,        \
        const size_t, const size_t, const shape_elem_type *,                   \
        const shape_elem_type *, const size_t *) = __name__<_DataType>;

#include <dpnp_gen_2arg_1type_tbl.hpp>

static void func_map_init_bitwise_2arg_1type(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_bitwise_and_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_AND][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_bitwise_and_c_default<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_bitwise_or_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_OR][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_bitwise_or_c_default<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_bitwise_xor_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_BITWISE_XOR][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_bitwise_xor_c_default<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_left_shift_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_LEFT_SHIFT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_left_shift_c_default<int64_t>};

    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_right_shift_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_RIGHT_SHIFT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_right_shift_c_default<int64_t>};

    return;
}

void func_map_init_bitwise(func_map_t &fmap)
{
    func_map_init_bitwise_1arg_1type(fmap);
    func_map_init_bitwise_2arg_1type(fmap);

    return;
}
