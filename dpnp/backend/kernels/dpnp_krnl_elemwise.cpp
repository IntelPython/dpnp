//*****************************************************************************
// Copyright (c) 2016-2025, Intel Corporation
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
#include <stdexcept>

#include <dpnp_iface.hpp>

#include "dpnp_fptr.hpp"
#include "dpnp_iterator.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

// dpctl tensor headers
#include "kernels/alignment.hpp"

using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

namespace syclex = sycl::ext::oneapi::experimental;
using syclex::group_load;
using syclex::group_store;

constexpr auto striped = syclex::properties{syclex::data_placement_striped};

template <typename T>
constexpr T dispatch_erf_op(T elem)
{
    if constexpr (is_any_v<T, std::int32_t, std::int64_t>) {
        // TODO: need to convert to double when possible
        return sycl::erf((float)elem);
    }
    else {
        return sycl::erf(elem);
    }
}

#define MACRO_1ARG_1TYPE_OP(__name__, __operation1__, __operation2__)          \
    template <typename _KernelNameSpecialization>                              \
    class __name__##_kernel;                                                   \
                                                                               \
    template <typename _KernelNameSpecialization>                              \
    class __name__##_strides_kernel;                                           \
                                                                               \
    template <typename _DataType>                                              \
    DPCTLSyclEventRef __name__(                                                \
        DPCTLSyclQueueRef q_ref, void *result_out, const size_t result_size,   \
        const size_t result_ndim, const shape_elem_type *result_shape,         \
        const shape_elem_type *result_strides, const void *input1_in,          \
        const size_t input1_size, const size_t input1_ndim,                    \
        const shape_elem_type *input1_shape,                                   \
        const shape_elem_type *input1_strides, const size_t *where,            \
        const DPCTLEventVectorRef dep_event_vec_ref)                           \
    {                                                                          \
        /* avoid warning unused variable*/                                     \
        (void)result_shape;                                                    \
        (void)where;                                                           \
        (void)dep_event_vec_ref;                                               \
                                                                               \
        DPCTLSyclEventRef event_ref = nullptr;                                 \
                                                                               \
        if (!input1_size) {                                                    \
            return event_ref;                                                  \
        }                                                                      \
                                                                               \
        sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));             \
                                                                               \
        _DataType *input1_data =                                               \
            static_cast<_DataType *>(const_cast<void *>(input1_in));           \
        _DataType *result = static_cast<_DataType *>(result_out);              \
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
        sycl::event event;                                                     \
        sycl::range<1> gws(result_size);                                       \
                                                                               \
        if (use_strides) {                                                     \
            if (result_ndim != input1_ndim) {                                  \
                throw std::runtime_error(                                      \
                    "Result ndim=" + std::to_string(result_ndim) +             \
                    " mismatches with input1 ndim=" +                          \
                    std::to_string(input1_ndim));                              \
            }                                                                  \
                                                                               \
            /* memory transfer optimization, use USM-host for temporary speeds \
             * up transfer to device */                                        \
            using usm_host_allocatorT =                                        \
                sycl::usm_allocator<shape_elem_type, sycl::usm::alloc::host>;  \
                                                                               \
            size_t strides_size = 2 * result_ndim;                             \
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
                                                                               \
            auto copy_strides_ev = q.copy<shape_elem_type>(                    \
                strides_host_packed.data(), dev_strides_data,                  \
                strides_host_packed.size());                                   \
                                                                               \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {       \
                size_t output_id = global_id[0]; /* for (size_t i = 0; i <     \
                                                    result_size; ++i) */       \
                {                                                              \
                    const shape_elem_type *result_strides_data =               \
                        &dev_strides_data[0];                                  \
                    const shape_elem_type *input1_strides_data =               \
                        &dev_strides_data[result_ndim];                        \
                                                                               \
                    size_t input_id = 0;                                       \
                    for (size_t i = 0; i < input1_ndim; ++i) {                 \
                        const size_t output_xyz_id =                           \
                            get_xyz_id_by_id_inkernel(output_id,               \
                                                      result_strides_data,     \
                                                      result_ndim, i);         \
                        input_id += output_xyz_id * input1_strides_data[i];    \
                    }                                                          \
                                                                               \
                    const _DataType input_elem = input1_data[input_id];        \
                    result[output_id] = __operation1__;                        \
                }                                                              \
            };                                                                 \
            auto kernel_func = [&](sycl::handler &cgh) {                       \
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
                {                                                              \
                    const _DataType input_elem = input1_data[i];               \
                    result[i] = __operation1__;                                \
                }                                                              \
            };                                                                 \
            auto kernel_func = [&](sycl::handler &cgh) {                       \
                cgh.parallel_for<class __name__##_kernel<_DataType>>(          \
                    gws, kernel_parallel_for_func);                            \
            };                                                                 \
                                                                               \
            if constexpr (is_any_v<_DataType, float, double>) {                \
                if (q.get_device().has(sycl::aspect::fp64)) {                  \
                    event = __operation2__;                                    \
                                                                               \
                    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);   \
                    return DPCTLEvent_Copy(event_ref);                         \
                }                                                              \
            }                                                                  \
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
        const shape_elem_type *input1_strides, const size_t *where)            \
    {                                                                          \
        DPCTLSyclQueueRef q_ref =                                              \
            reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);                  \
        DPCTLEventVectorRef dep_event_vec_ref = nullptr;                       \
        DPCTLSyclEventRef event_ref = __name__<_DataType>(                     \
            q_ref, result_out, result_size, result_ndim, result_shape,         \
            result_strides, input1_in, input1_size, input1_ndim, input1_shape, \
            input1_strides, where, dep_event_vec_ref);                         \
        DPCTLEvent_WaitAndThrow(event_ref);                                    \
        DPCTLEvent_Delete(event_ref);                                          \
    }                                                                          \
                                                                               \
    template <typename _DataType>                                              \
    void (*__name__##_default)(                                                \
        void *, const size_t, const size_t, const shape_elem_type *,           \
        const shape_elem_type *, const void *, const size_t, const size_t,     \
        const shape_elem_type *, const shape_elem_type *, const size_t *) =    \
        __name__<_DataType>;                                                   \
                                                                               \
    template <typename _DataType>                                              \
    DPCTLSyclEventRef (*__name__##_ext)(                                       \
        DPCTLSyclQueueRef, void *, const size_t, const size_t,                 \
        const shape_elem_type *, const shape_elem_type *, const void *,        \
        const size_t, const size_t, const shape_elem_type *,                   \
        const shape_elem_type *, const size_t *, const DPCTLEventVectorRef) =  \
        __name__<_DataType>;

#include <dpnp_gen_1arg_1type_tbl.hpp>

static void func_map_init_elemwise_1arg_1type(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_erf_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_erf_c_default<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_erf_c_default<float>};
    fmap[DPNPFuncName::DPNP_FN_ERF][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_erf_c_default<double>};

    fmap[DPNPFuncName::DPNP_FN_ERF_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_erf_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ERF_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_erf_c_ext<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ERF_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_erf_c_ext<float>};
    fmap[DPNPFuncName::DPNP_FN_ERF_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_erf_c_ext<double>};

    return;
}

#define MACRO_2ARG_3TYPES_OP(__name__, __operation__, __vec_operation__,       \
                             __vec_types__, __mkl_operation__, __mkl_types__)  \
    template <typename _KernelNameSpecialization1,                             \
              typename _KernelNameSpecialization2,                             \
              typename _KernelNameSpecialization3>                             \
    class __name__##_kernel;                                                   \
                                                                               \
    template <typename _KernelNameSpecialization1,                             \
              typename _KernelNameSpecialization2,                             \
              typename _KernelNameSpecialization3>                             \
    class __name__##_sg_kernel;                                                \
                                                                               \
    template <typename _KernelNameSpecialization1,                             \
              typename _KernelNameSpecialization2,                             \
              typename _KernelNameSpecialization3>                             \
    class __name__##_broadcast_kernel;                                         \
                                                                               \
    template <typename _KernelNameSpecialization1,                             \
              typename _KernelNameSpecialization2,                             \
              typename _KernelNameSpecialization3>                             \
    class __name__##_strides_kernel;                                           \
                                                                               \
    template <typename _DataType_output, typename _DataType_input1,            \
              typename _DataType_input2>                                       \
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
        _DataType_input1 *input1_data =                                        \
            static_cast<_DataType_input1 *>(const_cast<void *>(input1_in));    \
        _DataType_input2 *input2_data =                                        \
            static_cast<_DataType_input2 *>(const_cast<void *>(input2_in));    \
        _DataType_output *result =                                             \
            static_cast<_DataType_output *>(result_out);                       \
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
            DPNPC_id<_DataType_input1> *input1_it;                             \
            const size_t input1_it_size_in_bytes =                             \
                sizeof(DPNPC_id<_DataType_input1>);                            \
            input1_it = reinterpret_cast<DPNPC_id<_DataType_input1> *>(        \
                dpnp_memory_alloc_c(q_ref, input1_it_size_in_bytes));          \
            new (input1_it)                                                    \
                DPNPC_id<_DataType_input1>(q_ref, input1_data, input1_shape,   \
                                           input1_strides, input1_ndim);       \
                                                                               \
            input1_it->broadcast_to_shape(result_shape, result_ndim);          \
                                                                               \
            DPNPC_id<_DataType_input2> *input2_it;                             \
            const size_t input2_it_size_in_bytes =                             \
                sizeof(DPNPC_id<_DataType_input2>);                            \
            input2_it = reinterpret_cast<DPNPC_id<_DataType_input2> *>(        \
                dpnp_memory_alloc_c(q_ref, input2_it_size_in_bytes));          \
            new (input2_it)                                                    \
                DPNPC_id<_DataType_input2>(q_ref, input2_data, input2_shape,   \
                                           input2_strides, input2_ndim);       \
                                                                               \
            input2_it->broadcast_to_shape(result_shape, result_ndim);          \
                                                                               \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {       \
                const size_t i = global_id[0]; /* for (size_t i = 0; i <       \
                                                  result_size; ++i) */         \
                {                                                              \
                    const _DataType_output input1_elem = (*input1_it)[i];      \
                    const _DataType_output input2_elem = (*input2_it)[i];      \
                    result[i] = __operation__;                                 \
                }                                                              \
            };                                                                 \
            auto kernel_func = [&](sycl::handler &cgh) {                       \
                cgh.parallel_for<class __name__##_broadcast_kernel<            \
                    _DataType_output, _DataType_input1, _DataType_input2>>(    \
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
                    const _DataType_output input1_elem =                       \
                        input1_data[input1_id];                                \
                    const _DataType_output input2_elem =                       \
                        input2_data[input2_id];                                \
                    result[output_id] = __operation__;                         \
                }                                                              \
            };                                                                 \
            auto kernel_func = [&](sycl::handler &cgh) {                       \
                cgh.depends_on(copy_strides_ev);                               \
                cgh.parallel_for<class __name__##_strides_kernel<              \
                    _DataType_output, _DataType_input1, _DataType_input2>>(    \
                    gws, kernel_parallel_for_func);                            \
            };                                                                 \
                                                                               \
            q.submit(kernel_func).wait();                                      \
                                                                               \
            sycl::free(dev_strides_data, q);                                   \
            return event_ref;                                                  \
        }                                                                      \
        else {                                                                 \
            if constexpr (both_types_are_same<_DataType_input1,                \
                                              _DataType_input2,                \
                                              __mkl_types__>)                  \
            {                                                                  \
                if (q.get_device().has(sycl::aspect::fp64)) {                  \
                    event = __mkl_operation__(q, result_size, input1_data,     \
                                              input2_data, result);            \
                                                                               \
                    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);   \
                    return DPCTLEvent_Copy(event_ref);                         \
                }                                                              \
            }                                                                  \
                                                                               \
            if constexpr (none_of_both_types<                                  \
                              _DataType_input1, _DataType_input2,              \
                              std::complex<float>, std::complex<double>>)      \
            {                                                                  \
                constexpr size_t lws = 64;                                     \
                constexpr unsigned int vec_sz = 8;                             \
                                                                               \
                auto gws_range = sycl::range<1>(                               \
                    ((result_size + lws * vec_sz - 1) / (lws * vec_sz)) *      \
                    lws);                                                      \
                auto lws_range = sycl::range<1>(lws);                          \
                                                                               \
                auto kernel_parallel_for_func = [=](sycl::nd_item<1> nd_it) {  \
                    auto sg = nd_it.get_sub_group();                           \
                    const auto max_sg_size = sg.get_max_local_range()[0];      \
                    const size_t start =                                       \
                        vec_sz *                                               \
                        (nd_it.get_group(0) * nd_it.get_local_range(0) +       \
                         sg.get_group_id()[0] * max_sg_size);                  \
                                                                               \
                    if (is_aligned<required_alignment>(input1_data) &&         \
                        is_aligned<required_alignment>(input2_data) &&         \
                        is_aligned<required_alignment>(result) &&              \
                        (start + static_cast<size_t>(vec_sz) * max_sg_size <   \
                         result_size))                                         \
                    {                                                          \
                        auto input1_multi_ptr = sycl::address_space_cast<      \
                            sycl::access::address_space::global_space,         \
                            sycl::access::decorated::yes>(                     \
                            &input1_data[start]);                              \
                        auto input2_multi_ptr = sycl::address_space_cast<      \
                            sycl::access::address_space::global_space,         \
                            sycl::access::decorated::yes>(                     \
                            &input2_data[start]);                              \
                        auto result_multi_ptr = sycl::address_space_cast<      \
                            sycl::access::address_space::global_space,         \
                            sycl::access::decorated::yes>(&result[start]);     \
                                                                               \
                        sycl::vec<_DataType_output, vec_sz> res_vec;           \
                                                                               \
                        if constexpr (both_types_are_any_of<_DataType_input1,  \
                                                            _DataType_input2,  \
                                                            __vec_types__>)    \
                        {                                                      \
                            if constexpr (both_types_are_same<                 \
                                              _DataType_input1,                \
                                              _DataType_input2,                \
                                              _DataType_output>)               \
                            {                                                  \
                                sycl::vec<_DataType_input1, vec_sz> x1{};      \
                                sycl::vec<_DataType_input2, vec_sz> x2{};      \
                                                                               \
                                group_load(sg, input1_multi_ptr, x1, striped); \
                                group_load(sg, input2_multi_ptr, x2, striped); \
                                                                               \
                                res_vec = __vec_operation__;                   \
                            }                                                  \
                            else /* input types don't match result type, so    \
                                    explicit casting is required */            \
                            {                                                  \
                                sycl::vec<_DataType_input1, vec_sz> tmp_x1{};  \
                                sycl::vec<_DataType_input2, vec_sz> tmp_x2{};  \
                                                                               \
                                group_load(sg, input1_multi_ptr, tmp_x1,       \
                                           striped);                           \
                                group_load(sg, input2_multi_ptr, tmp_x2,       \
                                           striped);                           \
                                                                               \
                                sycl::vec<_DataType_output, vec_sz> x1 =       \
                                    dpnp_vec_cast<_DataType_output,            \
                                                  _DataType_input1, vec_sz>(   \
                                        tmp_x1);                               \
                                sycl::vec<_DataType_output, vec_sz> x2 =       \
                                    dpnp_vec_cast<_DataType_output,            \
                                                  _DataType_input2, vec_sz>(   \
                                        tmp_x2);                               \
                                                                               \
                                res_vec = __vec_operation__;                   \
                            }                                                  \
                        }                                                      \
                        else {                                                 \
                            sycl::vec<_DataType_input1, vec_sz> x1{};          \
                            sycl::vec<_DataType_input2, vec_sz> x2{};          \
                                                                               \
                            group_load(sg, input1_multi_ptr, x1, striped);     \
                            group_load(sg, input2_multi_ptr, x2, striped);     \
                                                                               \
                            for (size_t k = 0; k < vec_sz; ++k) {              \
                                const _DataType_output input1_elem = x1[k];    \
                                const _DataType_output input2_elem = x2[k];    \
                                res_vec[k] = __operation__;                    \
                            }                                                  \
                        }                                                      \
                        group_store(sg, res_vec, result_multi_ptr, striped);   \
                    }                                                          \
                    else {                                                     \
                        for (size_t k = start + sg.get_local_id()[0];          \
                             k < result_size; k += max_sg_size) {              \
                            const _DataType_output input1_elem =               \
                                input1_data[k];                                \
                            const _DataType_output input2_elem =               \
                                input2_data[k];                                \
                            result[k] = __operation__;                         \
                        }                                                      \
                    }                                                          \
                };                                                             \
                                                                               \
                auto kernel_func = [&](sycl::handler &cgh) {                   \
                    cgh.parallel_for<class __name__##_sg_kernel<               \
                        _DataType_output, _DataType_input1,                    \
                        _DataType_input2>>(                                    \
                        sycl::nd_range<1>(gws_range, lws_range),               \
                        kernel_parallel_for_func);                             \
                };                                                             \
                event = q.submit(kernel_func);                                 \
            }                                                                  \
            else /* either input1 or input2 has complex type */ {              \
                auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {   \
                    const size_t i = global_id[0]; /* for (size_t i = 0; i <   \
                                                      result_size; ++i) */     \
                                                                               \
                    const _DataType_output input1_elem = input1_data[i];       \
                    const _DataType_output input2_elem = input2_data[i];       \
                    result[i] = __operation__;                                 \
                };                                                             \
                auto kernel_func = [&](sycl::handler &cgh) {                   \
                    cgh.parallel_for<class __name__##_kernel<                  \
                        _DataType_output, _DataType_input1,                    \
                        _DataType_input2>>(gws, kernel_parallel_for_func);     \
                };                                                             \
                event = q.submit(kernel_func);                                 \
            }                                                                  \
        }                                                                      \
                                                                               \
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);               \
        return DPCTLEvent_Copy(event_ref);                                     \
    }                                                                          \
                                                                               \
    template <typename _DataType_output, typename _DataType_input1,            \
              typename _DataType_input2>                                       \
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
        DPCTLSyclEventRef event_ref =                                          \
            __name__<_DataType_output, _DataType_input1, _DataType_input2>(    \
                q_ref, result_out, result_size, result_ndim, result_shape,     \
                result_strides, input1_in, input1_size, input1_ndim,           \
                input1_shape, input1_strides, input2_in, input2_size,          \
                input2_ndim, input2_shape, input2_strides, where,              \
                dep_event_vec_ref);                                            \
        DPCTLEvent_WaitAndThrow(event_ref);                                    \
        DPCTLEvent_Delete(event_ref);                                          \
    }                                                                          \
                                                                               \
    template <typename _DataType_output, typename _DataType_input1,            \
              typename _DataType_input2>                                       \
    void (*__name__##_default)(                                                \
        void *, const size_t, const size_t, const shape_elem_type *,           \
        const shape_elem_type *, const void *, const size_t, const size_t,     \
        const shape_elem_type *, const shape_elem_type *, const void *,        \
        const size_t, const size_t, const shape_elem_type *,                   \
        const shape_elem_type *, const size_t *) =                             \
        __name__<_DataType_output, _DataType_input1, _DataType_input2>;        \
                                                                               \
    template <typename _DataType_output, typename _DataType_input1,            \
              typename _DataType_input2>                                       \
    DPCTLSyclEventRef (*__name__##_ext)(                                       \
        DPCTLSyclQueueRef, void *, const size_t, const size_t,                 \
        const shape_elem_type *, const shape_elem_type *, const void *,        \
        const size_t, const size_t, const shape_elem_type *,                   \
        const shape_elem_type *, const void *, const size_t, const size_t,     \
        const shape_elem_type *, const shape_elem_type *, const size_t *,      \
        const DPCTLEventVectorRef) =                                           \
        __name__<_DataType_output, _DataType_input1, _DataType_input2>;

#include <dpnp_gen_2arg_3type_tbl.hpp>

static void func_map_init_elemwise_2arg_3type(func_map_t &fmap)
{
    // Used in dpnp_dot_c
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_multiply_c_default<bool, bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_INT] = {
        eft_INT, (void *)dpnp_multiply_c_default<int32_t, bool, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_LNG] = {
        eft_LNG, (void *)dpnp_multiply_c_default<int64_t, bool, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_FLT] = {
        eft_FLT, (void *)dpnp_multiply_c_default<float, bool, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_DBL] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, bool, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_BLN] = {
        eft_INT, (void *)dpnp_multiply_c_default<int32_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_multiply_c_default<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_multiply_c_default<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_BLN] = {
        eft_LNG, (void *)dpnp_multiply_c_default<int64_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_multiply_c_default<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_multiply_c_default<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_BLN] = {
        eft_FLT, (void *)dpnp_multiply_c_default<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_multiply_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_BLN] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_multiply_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C64][eft_BLN] = {
        eft_C64, (void *)dpnp_multiply_c_default<std::complex<float>,
                                                 std::complex<float>, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C64][eft_INT] = {
        eft_C64, (void *)dpnp_multiply_c_default<std::complex<float>,
                                                 std::complex<float>, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C64][eft_LNG] = {
        eft_C64, (void *)dpnp_multiply_c_default<std::complex<float>,
                                                 std::complex<float>, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C64][eft_FLT] = {
        eft_C64, (void *)dpnp_multiply_c_default<std::complex<float>,
                                                 std::complex<float>, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C64][eft_DBL] = {
        eft_C128, (void *)dpnp_multiply_c_default<std::complex<double>,
                                                  std::complex<float>, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C64][eft_C64] = {
        eft_C64,
        (void *)dpnp_multiply_c_default<
            std::complex<float>, std::complex<float>, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C64][eft_C128] = {
        eft_C128,
        (void *)dpnp_multiply_c_default<
            std::complex<double>, std::complex<float>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_BLN] = {
        eft_C128, (void *)dpnp_multiply_c_default<std::complex<double>,
                                                  std::complex<double>, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_INT] = {
        eft_C128,
        (void *)dpnp_multiply_c_default<std::complex<double>,
                                        std::complex<double>, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_LNG] = {
        eft_C128,
        (void *)dpnp_multiply_c_default<std::complex<double>,
                                        std::complex<double>, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_FLT] = {
        eft_C128, (void *)dpnp_multiply_c_default<std::complex<double>,
                                                  std::complex<double>, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_DBL] = {
        eft_C128,
        (void *)dpnp_multiply_c_default<std::complex<double>,
                                        std::complex<double>, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_C64] = {
        eft_C128,
        (void *)dpnp_multiply_c_default<
            std::complex<double>, std::complex<double>, std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_C128] = {
        eft_C128,
        (void *)dpnp_multiply_c_default<
            std::complex<double>, std::complex<double>, std::complex<double>>};

    return;
}

void func_map_init_elemwise(func_map_t &fmap)
{
    func_map_init_elemwise_1arg_1type(fmap);
    func_map_init_elemwise_2arg_3type(fmap);

    return;
}
