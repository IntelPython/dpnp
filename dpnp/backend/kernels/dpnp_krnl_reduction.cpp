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

#include "dpnp_fptr.hpp"
#include "dpnp_iterator.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"
#include <dpnp_iface.hpp>

namespace mkl_stats = oneapi::mkl::stats;

template <typename _DataType>
_DataType *get_array_ptr(const void *__array)
{
    void *const_ptr = const_cast<void *>(__array);
    _DataType *ptr = reinterpret_cast<_DataType *>(const_ptr);

    return ptr;
}

template <typename _DataType>
_DataType get_initial_value(const void *__initial, _DataType default_val)
{
    const _DataType *initial_ptr =
        reinterpret_cast<const _DataType *>(__initial);
    const _DataType init_val =
        (initial_ptr == nullptr) ? default_val : *initial_ptr;

    return init_val;
}

template <typename _KernelNameSpecialization1,
          typename _KernelNameSpecialization2>
class dpnp_sum_c_kernel;

template <typename _DataType_output, typename _DataType_input>
DPCTLSyclEventRef
    dpnp_sum_c(DPCTLSyclQueueRef q_ref,
               void *result_out,
               const void *input_in,
               const shape_elem_type *input_shape,
               const size_t input_shape_ndim,
               const shape_elem_type *axes,
               const size_t axes_ndim,
               const void *initial, // type must be _DataType_output
               const long *where,
               const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)where;
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if ((input_in == nullptr) || (result_out == nullptr)) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    const _DataType_output init =
        get_initial_value<_DataType_output>(initial, 0);

    const size_t input_size =
        std::accumulate(input_shape, input_shape + input_shape_ndim, 1,
                        std::multiplies<shape_elem_type>());

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(q_ref, input_in, input_size,
                                                  true);
    _DataType_input *input = input1_ptr.get_ptr();
    _DataType_output *result = get_array_ptr<_DataType_output>(result_out);

    if (!input_shape && !input_shape_ndim) { // it is a scalar
        // result[0] = input[0];
        _DataType_input input_elem = 0;
        _DataType_output result_elem = 0;
        q.memcpy(&input_elem, input, sizeof(_DataType_input)).wait();
        result_elem = input_elem;
        q.memcpy(result, &result_elem, sizeof(_DataType_output)).wait();

        return event_ref;
    }

    if constexpr ((std::is_same<_DataType_input, double>::value ||
                   std::is_same<_DataType_input, float>::value) &&
                  std::is_same<_DataType_input, _DataType_output>::value)
    {
        // Support is limited by
        // - 1D array (no axes)
        // - same types for input and output
        // - float64 and float32 types only
        if (axes_ndim < 1) {
            auto dataset =
                mkl_stats::make_dataset<mkl_stats::layout::row_major>(
                    1, input_size, input);
            sycl::event event = mkl_stats::raw_sum(q, dataset, result);
            event.wait();

            return event_ref;
        }
    }

    DPNPC_id<_DataType_input> input_it(q_ref, input, input_shape,
                                       input_shape_ndim);
    input_it.set_axes(axes, axes_ndim);

    const size_t output_size = input_it.get_output_size();
    auto policy = oneapi::dpl::execution::make_device_policy<
        dpnp_sum_c_kernel<_DataType_output, _DataType_input>>(q);
    for (size_t output_id = 0; output_id < output_size; ++output_id) {
        // type of "init" determine internal algorithm accumulator type
        _DataType_output accumulator = std::reduce(
            policy, input_it.begin(output_id), input_it.end(output_id), init,
            std::plus<_DataType_output>());
        policy.queue().wait(); // TODO move out of the loop

        q.memcpy(result + output_id, &accumulator, sizeof(_DataType_output))
            .wait(); // result[output_id] = accumulator;
    }

    return event_ref;
}

template <typename _DataType_output, typename _DataType_input>
void dpnp_sum_c(void *result_out,
                const void *input_in,
                const shape_elem_type *input_shape,
                const size_t input_shape_ndim,
                const shape_elem_type *axes,
                const size_t axes_ndim,
                const void *initial, // type must be _DataType_output
                const long *where)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_sum_c<_DataType_output, _DataType_input>(
        q_ref, result_out, input_in, input_shape, input_shape_ndim, axes,
        axes_ndim, initial, where, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType_output, typename _DataType_input>
void (*dpnp_sum_default_c)(void *,
                           const void *,
                           const shape_elem_type *,
                           const size_t,
                           const shape_elem_type *,
                           const size_t,
                           const void *,
                           const long *) =
    dpnp_sum_c<_DataType_output, _DataType_input>;

template <typename _DataType_output, typename _DataType_input>
DPCTLSyclEventRef (*dpnp_sum_ext_c)(DPCTLSyclQueueRef,
                                    void *,
                                    const void *,
                                    const shape_elem_type *,
                                    const size_t,
                                    const shape_elem_type *,
                                    const size_t,
                                    const void *,
                                    const long *,
                                    const DPCTLEventVectorRef) =
    dpnp_sum_c<_DataType_output, _DataType_input>;

template <typename _KernelNameSpecialization1,
          typename _KernelNameSpecialization2>
class dpnp_prod_c_kernel;

template <typename _DataType_output, typename _DataType_input>
DPCTLSyclEventRef
    dpnp_prod_c(DPCTLSyclQueueRef q_ref,
                void *result_out,
                const void *input_in,
                const shape_elem_type *input_shape,
                const size_t input_shape_ndim,
                const shape_elem_type *axes,
                const size_t axes_ndim,
                const void *initial, // type must be _DataType_output
                const long *where,
                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)where;
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if ((input_in == nullptr) || (result_out == nullptr)) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    const _DataType_output init =
        get_initial_value<_DataType_output>(initial, 1);

    const size_t input_size =
        std::accumulate(input_shape, input_shape + input_shape_ndim, 1,
                        std::multiplies<shape_elem_type>());

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(q_ref, input_in, input_size,
                                                  true);
    _DataType_input *input = input1_ptr.get_ptr();
    _DataType_output *result = get_array_ptr<_DataType_output>(result_out);

    if (!input_shape && !input_shape_ndim) { // it is a scalar
        // result[0] = input[0];
        _DataType_input input_elem = 0;
        _DataType_output result_elem = 0;
        q.memcpy(&input_elem, input, sizeof(_DataType_input)).wait();
        result_elem = input_elem;
        q.memcpy(result, &result_elem, sizeof(_DataType_output)).wait();

        return event_ref;
    }

    DPNPC_id<_DataType_input> input_it(q_ref, input, input_shape,
                                       input_shape_ndim);
    input_it.set_axes(axes, axes_ndim);

    const size_t output_size = input_it.get_output_size();
    auto policy = oneapi::dpl::execution::make_device_policy<
        dpnp_prod_c_kernel<_DataType_output, _DataType_input>>(q);
    for (size_t output_id = 0; output_id < output_size; ++output_id) {
        // type of "init" determine internal algorithm accumulator type
        _DataType_output accumulator = std::reduce(
            policy, input_it.begin(output_id), input_it.end(output_id), init,
            std::multiplies<_DataType_output>());
        policy.queue().wait(); // TODO move out of the loop

        q.memcpy(result + output_id, &accumulator, sizeof(_DataType_output))
            .wait(); // result[output_id] = accumulator;
    }

    return event_ref;
}

template <typename _DataType_output, typename _DataType_input>
void dpnp_prod_c(void *result_out,
                 const void *input_in,
                 const shape_elem_type *input_shape,
                 const size_t input_shape_ndim,
                 const shape_elem_type *axes,
                 const size_t axes_ndim,
                 const void *initial, // type must be _DataType_output
                 const long *where)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_prod_c<_DataType_output, _DataType_input>(
            q_ref, result_out, input_in, input_shape, input_shape_ndim, axes,
            axes_ndim, initial, where, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType_output, typename _DataType_input>
void (*dpnp_prod_default_c)(void *,
                            const void *,
                            const shape_elem_type *,
                            const size_t,
                            const shape_elem_type *,
                            const size_t,
                            const void *,
                            const long *) =
    dpnp_prod_c<_DataType_output, _DataType_input>;

void func_map_init_reduction(func_map_t &fmap)
{
    // WARNING. The meaning of the fmap is changed. Second argument represents
    // RESULT_TYPE for this function handle "out" and "type" parameters require
    // user selection of return type
    // TODO. required refactoring of fmap to some kernelSelector
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_INT] = {
        eft_LNG, (void *)dpnp_prod_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_prod_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_prod_default_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_prod_default_c<double, int32_t>};

    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_INT] = {
        eft_INT, (void *)dpnp_prod_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_prod_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_prod_default_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_prod_default_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_INT] = {
        eft_INT, (void *)dpnp_prod_default_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_LNG] = {
        eft_LNG, (void *)dpnp_prod_default_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_prod_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_prod_default_c<double, float>};

    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_INT] = {
        eft_INT, (void *)dpnp_prod_default_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_LNG] = {
        eft_LNG, (void *)dpnp_prod_default_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_FLT] = {
        eft_FLT, (void *)dpnp_prod_default_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_prod_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_INT] = {
        eft_LNG, (void *)dpnp_sum_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_sum_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sum_default_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_sum_default_c<double, int32_t>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_INT] = {
        eft_INT, (void *)dpnp_sum_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_sum_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_sum_default_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_sum_default_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_INT] = {
        eft_INT, (void *)dpnp_sum_default_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_LNG] = {
        eft_LNG, (void *)dpnp_sum_default_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sum_default_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_sum_default_c<double, float>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_INT] = {
        eft_INT, (void *)dpnp_sum_default_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_LNG] = {
        eft_LNG, (void *)dpnp_sum_default_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_FLT] = {
        eft_FLT, (void *)dpnp_sum_default_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_sum_default_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_INT][eft_INT] = {
        eft_LNG, (void *)dpnp_sum_ext_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_INT][eft_LNG] = {
        eft_LNG, (void *)dpnp_sum_ext_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_INT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sum_ext_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_INT][eft_DBL] = {
        eft_DBL, (void *)dpnp_sum_ext_c<double, int32_t>};

    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_LNG][eft_INT] = {
        eft_INT, (void *)dpnp_sum_ext_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_sum_ext_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_LNG][eft_FLT] = {
        eft_FLT, (void *)dpnp_sum_ext_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_LNG][eft_DBL] = {
        eft_DBL, (void *)dpnp_sum_ext_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_FLT][eft_INT] = {
        eft_INT, (void *)dpnp_sum_ext_c<int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_FLT][eft_LNG] = {
        eft_LNG, (void *)dpnp_sum_ext_c<int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_sum_ext_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_FLT][eft_DBL] = {
        eft_DBL, (void *)dpnp_sum_ext_c<double, float>};

    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_DBL][eft_INT] = {
        eft_INT, (void *)dpnp_sum_ext_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_DBL][eft_LNG] = {
        eft_LNG, (void *)dpnp_sum_ext_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_DBL][eft_FLT] = {
        eft_FLT, (void *)dpnp_sum_ext_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUM_EXT][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_sum_ext_c<double, double>};

    return;
}
