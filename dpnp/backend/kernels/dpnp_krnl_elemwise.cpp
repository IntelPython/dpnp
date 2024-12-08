//*****************************************************************************
// Copyright (c) 2016-2024, Intel Corporation
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

using sycl::ext::oneapi::experimental::group_load;
using sycl::ext::oneapi::experimental::group_store;

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

void func_map_init_elemwise(func_map_t &fmap)
{
    func_map_init_elemwise_1arg_1type(fmap);

    return;
}
