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

#include <cmath>
#include <iostream>

#include <dpnp_iface.hpp>

#include "dpnp_fptr.hpp"
#include "dpnp_iterator.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

#define MACRO_1ARG_2TYPES_OP(__name__, __operation1__, __operation2__)         \
    template <typename _KernelNameSpecialization1,                             \
              typename _KernelNameSpecialization2>                             \
    class __name__##_kernel;                                                   \
                                                                               \
    template <typename _KernelNameSpecialization1,                             \
              typename _KernelNameSpecialization2>                             \
    class __name__##_strides_kernel;                                           \
                                                                               \
    template <typename _DataType_input, typename _DataType_output>             \
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
        _DataType_input *input1_data =                                         \
            static_cast<_DataType_input *>(const_cast<void *>(input1_in));     \
        _DataType_output *result =                                             \
            static_cast<_DataType_output *>(result_out);                       \
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
             * up tranfer to device */                                         \
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
                    const _DataType_output input_elem = input1_data[input_id]; \
                    result[output_id] = __operation1__;                        \
                }                                                              \
            };                                                                 \
            auto kernel_func = [&](sycl::handler &cgh) {                       \
                cgh.parallel_for<class __name__##_strides_kernel<              \
                    _DataType_input, _DataType_output>>(                       \
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
                size_t output_id = global_id[0]; /* for (size_t i = 0; i <     \
                                                    result_size; ++i) */       \
                {                                                              \
                    const _DataType_output input_elem =                        \
                        input1_data[output_id];                                \
                    result[output_id] = __operation1__;                        \
                }                                                              \
            };                                                                 \
            auto kernel_func = [&](sycl::handler &cgh) {                       \
                cgh.parallel_for<class __name__##_kernel<_DataType_input,      \
                                                         _DataType_output>>(   \
                    gws, kernel_parallel_for_func);                            \
            };                                                                 \
                                                                               \
            if constexpr (both_types_are_same<_DataType_input,                 \
                                              _DataType_output, float,         \
                                              double>)                         \
            {                                                                  \
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
    template <typename _DataType_input, typename _DataType_output>             \
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
        DPCTLSyclEventRef event_ref =                                          \
            __name__<_DataType_input, _DataType_output>(                       \
                q_ref, result_out, result_size, result_ndim, result_shape,     \
                result_strides, input1_in, input1_size, input1_ndim,           \
                input1_shape, input1_strides, where, dep_event_vec_ref);       \
        DPCTLEvent_WaitAndThrow(event_ref);                                    \
        DPCTLEvent_Delete(event_ref);                                          \
    }                                                                          \
                                                                               \
    template <typename _DataType_input, typename _DataType_output>             \
    void (*__name__##_default)(                                                \
        void *, const size_t, const size_t, const shape_elem_type *,           \
        const shape_elem_type *, const void *, const size_t, const size_t,     \
        const shape_elem_type *, const shape_elem_type *, const size_t *) =    \
        __name__<_DataType_input, _DataType_output>;                           \
                                                                               \
    template <typename _DataType_input, typename _DataType_output>             \
    DPCTLSyclEventRef (*__name__##_ext)(                                       \
        DPCTLSyclQueueRef, void *, const size_t, const size_t,                 \
        const shape_elem_type *, const shape_elem_type *, const void *,        \
        const size_t, const size_t, const shape_elem_type *,                   \
        const shape_elem_type *, const size_t *, const DPCTLEventVectorRef) =  \
        __name__<_DataType_input, _DataType_output>;

#include <dpnp_gen_1arg_2type_tbl.hpp>

static void func_map_init_elemwise_1arg_2type(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_acos_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_acos_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_acos_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_acos_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCCOS_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_acos_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_acos_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_acos_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOS_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_acos_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_acosh_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_acosh_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_acosh_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_acosh_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCCOSH_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_acosh_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_acosh_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_acosh_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCCOSH_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_acosh_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_asin_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_asin_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_asin_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_asin_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSIN_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_asin_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_asin_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_asin_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSIN_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_asin_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_asinh_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_asinh_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_asinh_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_asinh_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCSINH_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_asinh_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_asinh_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_asinh_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCSINH_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_asinh_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_atan_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_atan_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_atan_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_atan_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_atan_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_atan_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_atan_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_atan_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_atanh_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_atanh_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_atanh_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_atanh_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTANH_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_atanh_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_atanh_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_atanh_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTANH_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_atanh_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_cbrt_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_cbrt_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_cbrt_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CBRT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_cbrt_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CBRT_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_cbrt_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_cbrt_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CBRT_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_cbrt_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CBRT_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_cbrt_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_ceil_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_ceil_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_ceil_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CEIL][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_ceil_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CEIL_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_ceil_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_ceil_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CEIL_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_ceil_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CEIL_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_ceil_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_default<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_default<bool, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_default<bool, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_default<bool, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_default<bool, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_default<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_default<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_default<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_default<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_default<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_default<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_default<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_default<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_default<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_default<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_default<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_default<float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_default<double, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_default<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_default<double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_default<double, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_default<double, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_C128][eft_C128] = {
        eft_C128,
        (void *)
            dpnp_copyto_c_default<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_ext<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_BLN][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_ext<bool, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_BLN][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_ext<bool, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_BLN][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_ext<bool, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_BLN][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_ext<bool, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_INT][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_ext<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_ext<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_ext<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_ext<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_LNG][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_ext<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_LNG][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_ext<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_ext<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_ext<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_FLT][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_ext<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_FLT][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_ext<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_FLT][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_ext<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_ext<float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_DBL][eft_BLN] = {
        eft_BLN, (void *)dpnp_copyto_c_ext<double, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_DBL][eft_INT] = {
        eft_INT, (void *)dpnp_copyto_c_ext<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_DBL][eft_LNG] = {
        eft_LNG, (void *)dpnp_copyto_c_ext<double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_DBL][eft_FLT] = {
        eft_FLT, (void *)dpnp_copyto_c_ext<double, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copyto_c_ext<double, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO_EXT][eft_C128][eft_C128] = {
        eft_C128,
        (void *)dpnp_copyto_c_ext<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_COS][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_cos_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_cos_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_cos_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COS][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_cos_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COSH][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_cosh_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_cosh_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_cosh_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COSH][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_cosh_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_COSH_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_cosh_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_cosh_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COSH_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_cosh_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COSH_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_cosh_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_degrees_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_degrees_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_degrees_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_degrees_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_DEGREES_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_degrees_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_degrees_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_degrees_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_DEGREES_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_degrees_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_exp2_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_exp2_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_exp2_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP2][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_exp2_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP2_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_exp2_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_exp2_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP2_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_exp2_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP2_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_exp2_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_exp_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_exp_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_exp_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_exp_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXP_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_exp_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_exp_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXP_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_exp_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXP_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_exp_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_expm1_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_expm1_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_expm1_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_expm1_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_EXPM1_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_expm1_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_expm1_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_expm1_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_EXPM1_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_expm1_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FABS][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_fabs_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_fabs_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_fabs_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FABS][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_fabs_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FABS_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_fabs_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_fabs_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FABS_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_fabs_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FABS_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_fabs_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_floor_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_floor_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_floor_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_floor_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FLOOR_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_floor_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_floor_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_floor_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_floor_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_log10_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_log10_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_log10_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG10][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_log10_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG10_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_log10_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_log10_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG10_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_log10_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG10_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_log10_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_log1p_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_log1p_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_log1p_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_log1p_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG1P_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_log1p_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_log1p_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_log1p_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG1P_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_log1p_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_log2_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_log2_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_log2_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG2][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_log2_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG2_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_log2_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_log2_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG2_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_log2_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG2_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_log2_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_LOG][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_log_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_log_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_log_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_LOG][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_log_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_radians_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_radians_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_radians_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_radians_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_RADIANS_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_radians_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_radians_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_radians_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_RADIANS_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_radians_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SIN][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_sin_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_sin_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sin_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SIN][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_sin_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SINH][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_sinh_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_sinh_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sinh_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SINH][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_sinh_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SINH_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_sinh_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_sinh_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SINH_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sinh_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SINH_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_sinh_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_sqrt_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_sqrt_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sqrt_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SQRT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_sqrt_c_default<double, double>};

    // Used in dpnp_std_c
    fmap[DPNPFuncName::DPNP_FN_SQRT_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sqrt_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SQRT_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_sqrt_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TAN][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_tan_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_tan_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_tan_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TAN][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_tan_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TAN_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_tan_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_tan_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TAN_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_tan_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TAN_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_tan_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TANH][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_tanh_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_tanh_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_tanh_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TANH][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_tanh_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TANH_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_tanh_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_tanh_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TANH_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_tanh_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TANH_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_tanh_c_ext<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_trunc_c_default<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_trunc_c_default<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_trunc_c_default<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_trunc_c_default<double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRUNC_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_trunc_c_ext<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_trunc_c_ext<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_trunc_c_ext<float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRUNC_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_trunc_c_ext<double, double>};

    return;
}

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

template <typename T>
constexpr T dispatch_sign_op(T elem)
{
    if constexpr (is_any_v<T, std::int32_t, std::int64_t>) {
        if (elem > 0)
            return T(1);
        if (elem < 0)
            return T(-1);
        return elem; // elem is 0
    }
    else {
        return sycl::sign(elem);
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
             * up tranfer to device */                                         \
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
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_copy_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_copy_c_default<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copy_c_default<float>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copy_c_default<double>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_conjugate_c_default<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_copy_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_copy_c_ext<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copy_c_ext<float>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copy_c_ext<double>};
    fmap[DPNPFuncName::DPNP_FN_CONJIGUATE_EXT][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_conjugate_c_ext<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_COPY][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_copy_c_default<bool>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_copy_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_copy_c_default<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copy_c_default<float>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copy_c_default<double>};
    fmap[DPNPFuncName::DPNP_FN_COPY][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_copy_c_default<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_COPY_EXT][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_copy_c_ext<bool>};
    fmap[DPNPFuncName::DPNP_FN_COPY_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_copy_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPY_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_copy_c_ext<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPY_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copy_c_ext<float>};
    fmap[DPNPFuncName::DPNP_FN_COPY_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copy_c_ext<double>};
    fmap[DPNPFuncName::DPNP_FN_COPY_EXT][eft_C64][eft_C64] = {
        eft_C64, (void *)dpnp_copy_c_ext<std::complex<float>>};
    fmap[DPNPFuncName::DPNP_FN_COPY_EXT][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_copy_c_ext<std::complex<double>>};

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

    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_copy_c_default<bool>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_copy_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_copy_c_default<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copy_c_default<float>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copy_c_default<double>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_copy_c_default<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_FLATTEN_EXT][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_copy_c_ext<bool>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_copy_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_copy_c_ext<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copy_c_ext<float>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copy_c_ext<double>};
    fmap[DPNPFuncName::DPNP_FN_FLATTEN_EXT][eft_C128][eft_C128] = {
        eft_C128, (void *)dpnp_copy_c_ext<std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_negative_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_negative_c_default<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_negative_c_default<float>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_negative_c_default<double>};

    fmap[DPNPFuncName::DPNP_FN_NEGATIVE_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_negative_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_negative_c_ext<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_negative_c_ext<float>};
    fmap[DPNPFuncName::DPNP_FN_NEGATIVE_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_negative_c_ext<double>};

    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_recip_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_recip_c_default<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_recip_c_default<float>};
    fmap[DPNPFuncName::DPNP_FN_RECIP][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_recip_c_default<double>};

    fmap[DPNPFuncName::DPNP_FN_RECIP_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_recip_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_RECIP_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_recip_c_ext<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_RECIP_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_recip_c_ext<float>};
    fmap[DPNPFuncName::DPNP_FN_RECIP_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_recip_c_ext<double>};

    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_sign_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_sign_c_default<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sign_c_default<float>};
    fmap[DPNPFuncName::DPNP_FN_SIGN][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_sign_c_default<double>};

    fmap[DPNPFuncName::DPNP_FN_SIGN_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_sign_c_ext<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SIGN_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_sign_c_ext<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SIGN_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sign_c_ext<float>};
    fmap[DPNPFuncName::DPNP_FN_SIGN_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_sign_c_ext<double>};

    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_square_c_default<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_square_c_default<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_square_c_default<float>};
    fmap[DPNPFuncName::DPNP_FN_SQUARE][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_square_c_default<double>};

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
             * up tranfer to device */                                         \
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
                constexpr sycl::access::address_space global_space =           \
                    sycl::access::address_space::global_space;                 \
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
                    if (start + static_cast<size_t>(vec_sz) * max_sg_size <    \
                        result_size) {                                         \
                        using input1_ptrT =                                    \
                            sycl::multi_ptr<_DataType_input1, global_space>;   \
                        using input2_ptrT =                                    \
                            sycl::multi_ptr<_DataType_input2, global_space>;   \
                        using result_ptrT =                                    \
                            sycl::multi_ptr<_DataType_output, global_space>;   \
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
                                sycl::vec<_DataType_input1, vec_sz> x1 =       \
                                    sg.load<vec_sz>(                           \
                                        input1_ptrT(&input1_data[start]));     \
                                sycl::vec<_DataType_input2, vec_sz> x2 =       \
                                    sg.load<vec_sz>(                           \
                                        input2_ptrT(&input2_data[start]));     \
                                                                               \
                                res_vec = __vec_operation__;                   \
                            }                                                  \
                            else /* input types don't match result type, so    \
                                    explicit casting is required */            \
                            {                                                  \
                                sycl::vec<_DataType_output, vec_sz> x1 =       \
                                    dpnp_vec_cast<_DataType_output,            \
                                                  _DataType_input1, vec_sz>(   \
                                        sg.load<vec_sz>(input1_ptrT(           \
                                            &input1_data[start])));            \
                                sycl::vec<_DataType_output, vec_sz> x2 =       \
                                    dpnp_vec_cast<_DataType_output,            \
                                                  _DataType_input2, vec_sz>(   \
                                        sg.load<vec_sz>(input2_ptrT(           \
                                            &input2_data[start])));            \
                                                                               \
                                res_vec = __vec_operation__;                   \
                            }                                                  \
                        }                                                      \
                        else {                                                 \
                            sycl::vec<_DataType_input1, vec_sz> x1 =           \
                                sg.load<vec_sz>(                               \
                                    input1_ptrT(&input1_data[start]));         \
                            sycl::vec<_DataType_input2, vec_sz> x2 =           \
                                sg.load<vec_sz>(                               \
                                    input2_ptrT(&input2_data[start]));         \
                                                                               \
                            for (size_t k = 0; k < vec_sz; ++k) {              \
                                const _DataType_output input1_elem = x1[k];    \
                                const _DataType_output input2_elem = x2[k];    \
                                res_vec[k] = __operation__;                    \
                            }                                                  \
                        }                                                      \
                        sg.store<vec_sz>(result_ptrT(&result[start]),          \
                                         res_vec);                             \
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

template <DPNPFuncType FT1,
          DPNPFuncType FT2,
          typename has_fp64 = std::true_type>
static constexpr DPNPFuncType get_divide_res_type()
{
    constexpr auto widest_type = populate_func_types<FT1, FT2>();
    constexpr auto shortes_type = (widest_type == FT1) ? FT2 : FT1;

    if constexpr (widest_type == DPNPFuncType::DPNP_FT_CMPLX128 ||
                  widest_type == DPNPFuncType::DPNP_FT_DOUBLE)
    {
        return widest_type;
    }
    else if constexpr (widest_type == DPNPFuncType::DPNP_FT_CMPLX64) {
        if constexpr (shortes_type == DPNPFuncType::DPNP_FT_DOUBLE) {
            return DPNPFuncType::DPNP_FT_CMPLX128;
        }
        else if constexpr (has_fp64::value &&
                           (shortes_type == DPNPFuncType::DPNP_FT_INT ||
                            shortes_type == DPNPFuncType::DPNP_FT_LONG))
        {
            return DPNPFuncType::DPNP_FT_CMPLX128;
        }
    }
    else if constexpr (widest_type == DPNPFuncType::DPNP_FT_FLOAT) {
        if constexpr (has_fp64::value &&
                      (shortes_type == DPNPFuncType::DPNP_FT_INT ||
                       shortes_type == DPNPFuncType::DPNP_FT_LONG))
        {
            return DPNPFuncType::DPNP_FT_DOUBLE;
        }
    }
    else if constexpr (has_fp64::value) {
        return DPNPFuncType::DPNP_FT_DOUBLE;
    }
    else {
        return DPNPFuncType::DPNP_FT_FLOAT;
    }
    return widest_type;
}

template <DPNPFuncType FT1, DPNPFuncType... FTs>
static void func_map_elemwise_2arg_3type_core(func_map_t &fmap)
{
    ((fmap[DPNPFuncName::DPNP_FN_ADD_EXT][FT1][FTs] =
          {populate_func_types<FT1, FTs>(),
           (void *)dpnp_add_c_ext<
               func_type_map_t::find_type<populate_func_types<FT1, FTs>()>,
               func_type_map_t::find_type<FT1>,
               func_type_map_t::find_type<FTs>>}),
     ...);
    ((fmap[DPNPFuncName::DPNP_FN_DIVIDE_EXT][FT1][FTs] =
          {get_divide_res_type<FT1, FTs>(),
           (void *)dpnp_divide_c_ext<
               func_type_map_t::find_type<get_divide_res_type<FT1, FTs>()>,
               func_type_map_t::find_type<FT1>,
               func_type_map_t::find_type<FTs>>,
           get_divide_res_type<FT1, FTs, std::false_type>(),
           (void *)
               dpnp_divide_c_ext<func_type_map_t::find_type<get_divide_res_type<
                                     FT1, FTs, std::false_type>()>,
                                 func_type_map_t::find_type<FT1>,
                                 func_type_map_t::find_type<FTs>>}),
     ...);
    ((fmap[DPNPFuncName::DPNP_FN_MULTIPLY_EXT][FT1][FTs] =
          {populate_func_types<FT1, FTs>(),
           (void *)dpnp_multiply_c_ext<
               func_type_map_t::find_type<populate_func_types<FT1, FTs>()>,
               func_type_map_t::find_type<FT1>,
               func_type_map_t::find_type<FTs>>}),
     ...);
    ((fmap[DPNPFuncName::DPNP_FN_POWER_EXT][FT1][FTs] =
          {populate_func_types<FT1, FTs>(),
           (void *)dpnp_power_c_ext<
               func_type_map_t::find_type<populate_func_types<FT1, FTs>()>,
               func_type_map_t::find_type<FT1>,
               func_type_map_t::find_type<FTs>>}),
     ...);
    ((fmap[DPNPFuncName::DPNP_FN_SUBTRACT_EXT][FT1][FTs] =
          {populate_func_types<FT1, FTs>(),
           (void *)dpnp_subtract_c_ext<
               func_type_map_t::find_type<populate_func_types<FT1, FTs>()>,
               func_type_map_t::find_type<FT1>,
               func_type_map_t::find_type<FTs>>}),
     ...);
}

template <DPNPFuncType... FTs>
static void func_map_elemwise_2arg_3type_helper(func_map_t &fmap)
{
    ((func_map_elemwise_2arg_3type_core<FTs, FTs...>(fmap)), ...);
}

static void func_map_init_elemwise_2arg_3type(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_add_c_default<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_add_c_default<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_add_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_add_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_add_c_default<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_add_c_default<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_add_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_add_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_add_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_add_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_add_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_add_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_add_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_add_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_add_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_ADD][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_add_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_LNG] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_INT] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_arctan2_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_arctan2_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_INT][eft_INT] = {
        get_default_floating_type(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<get_default_floating_type()>, int32_t,
            int32_t>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<
                get_default_floating_type<std::false_type>()>,
            int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_INT][eft_LNG] = {
        get_default_floating_type(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<get_default_floating_type()>, int32_t,
            int64_t>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<
                get_default_floating_type<std::false_type>()>,
            int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_INT][eft_FLT] = {
        get_default_floating_type(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<get_default_floating_type()>, int32_t,
            float>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<
                get_default_floating_type<std::false_type>()>,
            int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_arctan2_c_ext<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_LNG][eft_INT] = {
        get_default_floating_type(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<get_default_floating_type()>, int64_t,
            int32_t>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<
                get_default_floating_type<std::false_type>()>,
            int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_LNG][eft_LNG] = {
        get_default_floating_type(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<get_default_floating_type()>, int64_t,
            int64_t>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<
                get_default_floating_type<std::false_type>()>,
            int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_LNG][eft_FLT] = {
        get_default_floating_type(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<get_default_floating_type()>, int64_t,
            float>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<
                get_default_floating_type<std::false_type>()>,
            int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_arctan2_c_ext<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_FLT][eft_INT] = {
        get_default_floating_type(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<get_default_floating_type()>, float,
            int32_t>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<
                get_default_floating_type<std::false_type>()>,
            float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_FLT][eft_LNG] = {
        get_default_floating_type(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<get_default_floating_type()>, float,
            int64_t>,
        get_default_floating_type<std::false_type>(),
        (void *)dpnp_arctan2_c_ext<
            func_type_map_t::find_type<
                get_default_floating_type<std::false_type>()>,
            float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_arctan2_c_ext<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_arctan2_c_ext<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_arctan2_c_ext<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_arctan2_c_ext<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_arctan2_c_ext<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_ARCTAN2_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_arctan2_c_ext<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_LNG] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_INT] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copysign_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copysign_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_INT][eft_LNG] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_LNG][eft_INT] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_copysign_c_ext<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYSIGN_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_copysign_c_ext<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_LNG] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_INT] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_divide_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_DIVIDE][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_divide_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_fmod_c_default<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_fmod_c_default<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_fmod_c_default<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_fmod_c_default<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_fmod_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_fmod_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_fmod_c_ext<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_fmod_c_ext<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_fmod_c_ext<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_fmod_c_ext<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_fmod_c_ext<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_FMOD_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_fmod_c_ext<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_LNG] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_INT] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_hypot_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_hypot_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_INT][eft_INT] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_INT][eft_LNG] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_LNG][eft_INT] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_LNG][eft_LNG] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_hypot_c_ext<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_HYPOT_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_hypot_c_ext<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_maximum_c_default<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_maximum_c_default<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_maximum_c_default<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_maximum_c_default<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_maximum_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_maximum_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_maximum_c_ext<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_maximum_c_ext<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_maximum_c_ext<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_maximum_c_ext<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_maximum_c_ext<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MAXIMUM_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_maximum_c_ext<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_minimum_c_default<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_minimum_c_default<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_minimum_c_default<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_minimum_c_default<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_minimum_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_minimum_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_minimum_c_ext<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_minimum_c_ext<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_minimum_c_ext<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_minimum_c_ext<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_minimum_c_ext<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MINIMUM_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_minimum_c_ext<double, double, double>};

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

    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_power_c_default<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_power_c_default<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_power_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_power_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_power_c_default<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_power_c_default<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_power_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_power_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_power_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_power_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_power_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_power_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_power_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_power_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_power_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_POWER][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_power_c_default<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_subtract_c_default<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_subtract_c_default<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_FLT] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_INT] = {
        eft_LNG, (void *)dpnp_subtract_c_default<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_subtract_c_default<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_FLT] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_INT] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_LNG] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_subtract_c_default<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_INT] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_LNG] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_FLT] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_SUBTRACT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_subtract_c_default<double, double, double>};

    func_map_elemwise_2arg_3type_helper<eft_BLN, eft_INT, eft_LNG, eft_FLT,
                                        eft_DBL, eft_C64, eft_C128>(fmap);

    return;
}

void func_map_init_elemwise(func_map_t &fmap)
{
    func_map_init_elemwise_1arg_1type(fmap);
    func_map_init_elemwise_1arg_2type(fmap);
    func_map_init_elemwise_2arg_3type(fmap);

    return;
}
