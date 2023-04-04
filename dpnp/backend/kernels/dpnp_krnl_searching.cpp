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

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "dpnp_iterator.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _DataType, typename _idx_DataType>
class dpnp_argmax_c_kernel;

template <typename _DataType, typename _idx_DataType>
DPCTLSyclEventRef dpnp_argmax_c(DPCTLSyclQueueRef q_ref,
                                void* array1_in,
                                void* result1,
                                size_t size,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, size);
    _DataType* array_1 = input1_ptr.get_ptr();
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    auto policy =
        oneapi::dpl::execution::make_device_policy<class dpnp_argmax_c_kernel<_DataType, _idx_DataType>>(q);

    _DataType* res = std::max_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    _idx_DataType result_val = std::distance(array_1, res);
    q.memcpy(result, &result_val, sizeof(_idx_DataType)).wait(); // result[0] = std::distance(array_1, res);

    return event_ref;
}

template <typename _DataType, typename _idx_DataType>
void dpnp_argmax_c(void* array1_in, void* result1, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_argmax_c<_DataType, _idx_DataType>(q_ref,
                                                                          array1_in,
                                                                          result1,
                                                                          size,
                                                                          dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType, typename _idx_DataType>
void (*dpnp_argmax_default_c)(void*, void*, size_t) = dpnp_argmax_c<_DataType, _idx_DataType>;

template <typename _DataType, typename _idx_DataType>
DPCTLSyclEventRef (*dpnp_argmax_ext_c)(DPCTLSyclQueueRef,
                                       void*,
                                       void*,
                                       size_t,
                                       const DPCTLEventVectorRef) = dpnp_argmax_c<_DataType, _idx_DataType>;

template <typename _DataType, typename _idx_DataType>
class dpnp_argmin_c_kernel;

template <typename _DataType, typename _idx_DataType>
DPCTLSyclEventRef dpnp_argmin_c(DPCTLSyclQueueRef q_ref,
                                void* array1_in,
                                void* result1,
                                size_t size,
                                const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;
    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    DPNPC_ptr_adapter<_DataType> input1_ptr(q_ref, array1_in, size);
    _DataType* array_1 = input1_ptr.get_ptr();
    _idx_DataType* result = reinterpret_cast<_idx_DataType*>(result1);

    auto policy =
        oneapi::dpl::execution::make_device_policy<class dpnp_argmin_c_kernel<_DataType, _idx_DataType>>(q);

    _DataType* res = std::min_element(policy, array_1, array_1 + size);
    policy.queue().wait();

    _idx_DataType result_val = std::distance(array_1, res);
    q.memcpy(result, &result_val, sizeof(_idx_DataType)).wait(); // result[0] = std::distance(array_1, res);

    return event_ref;
}

template <typename _DataType, typename _idx_DataType>
void dpnp_argmin_c(void* array1_in, void* result1, size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_argmin_c<_DataType, _idx_DataType>(q_ref,
                                                                          array1_in,
                                                                          result1,
                                                                          size,
                                                                          dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType, typename _idx_DataType>
void (*dpnp_argmin_default_c)(void*, void*, size_t) = dpnp_argmin_c<_DataType, _idx_DataType>;

template <typename _DataType, typename _idx_DataType>
DPCTLSyclEventRef (*dpnp_argmin_ext_c)(DPCTLSyclQueueRef,
                                       void*,
                                       void*,
                                       size_t,
                                       const DPCTLEventVectorRef) = dpnp_argmin_c<_DataType, _idx_DataType>;


template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
class dpnp_where_c_broadcast_kernel;

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
class dpnp_where_c_strides_kernel;

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
class dpnp_where_c_kernel;

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
DPCTLSyclEventRef dpnp_where_c(DPCTLSyclQueueRef q_ref,
                               void* result_out,
                               const size_t result_size,
                               const size_t result_ndim,
                               const shape_elem_type* result_shape,
                               const shape_elem_type* result_strides,
                               const void* condition_in,
                               const size_t condition_size,
                               const size_t condition_ndim,
                               const shape_elem_type* condition_shape,
                               const shape_elem_type* condition_strides,
                               const void* input1_in,
                               const size_t input1_size,
                               const size_t input1_ndim,
                               const shape_elem_type* input1_shape,
                               const shape_elem_type* input1_strides,
                               const void* input2_in,
                               const size_t input2_size,
                               const size_t input2_ndim,
                               const shape_elem_type* input2_shape,
                               const shape_elem_type* input2_strides,
                               const DPCTLEventVectorRef dep_event_vec_ref)
{
    /* avoid warning unused variable*/
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!condition_size || !input1_size || !input2_size)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    bool* condition_data = static_cast<bool*>(const_cast<void*>(condition_in));
    _DataType_input1* input1_data = static_cast<_DataType_input1*>(const_cast<void*>(input1_in));
    _DataType_input2* input2_data = static_cast<_DataType_input2*>(const_cast<void*>(input2_in));
    _DataType_output* result = static_cast<_DataType_output*>(result_out);

    bool use_broadcasting = !array_equal(input1_shape, input1_ndim, input2_shape, input2_ndim);
    use_broadcasting = use_broadcasting || !array_equal(condition_shape, condition_ndim, input1_shape, input1_ndim);
    use_broadcasting = use_broadcasting || !array_equal(condition_shape, condition_ndim, input2_shape, input2_ndim);

    shape_elem_type* condition_shape_offsets = new shape_elem_type[condition_ndim];

    get_shape_offsets_inkernel(condition_shape, condition_ndim, condition_shape_offsets);
    bool use_strides = !array_equal(condition_strides, condition_ndim, condition_shape_offsets, condition_ndim);
    delete[] condition_shape_offsets;

    shape_elem_type* input1_shape_offsets = new shape_elem_type[input1_ndim];

    get_shape_offsets_inkernel(input1_shape, input1_ndim, input1_shape_offsets);
    use_strides = use_strides || !array_equal(input1_strides, input1_ndim, input1_shape_offsets, input1_ndim);
    delete[] input1_shape_offsets;

    shape_elem_type* input2_shape_offsets = new shape_elem_type[input2_ndim];

    get_shape_offsets_inkernel(input2_shape, input2_ndim, input2_shape_offsets);
    use_strides = use_strides || !array_equal(input2_strides, input2_ndim, input2_shape_offsets, input2_ndim);
    delete[] input2_shape_offsets;

    sycl::event event;
    sycl::range<1> gws(result_size);

    if (use_broadcasting)
    {
        DPNPC_id<bool>* condition_it;
        const size_t condition_it_it_size_in_bytes = sizeof(DPNPC_id<bool>);
        condition_it = reinterpret_cast<DPNPC_id<bool>*>(dpnp_memory_alloc_c(q_ref, condition_it_it_size_in_bytes));
        new (condition_it) DPNPC_id<bool>(q_ref, condition_data, condition_shape, condition_strides, condition_ndim);

        condition_it->broadcast_to_shape(result_shape, result_ndim);

        DPNPC_id<_DataType_input1>* input1_it;
        const size_t input1_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input1>);
        input1_it = reinterpret_cast<DPNPC_id<_DataType_input1>*>(dpnp_memory_alloc_c(q_ref, input1_it_size_in_bytes));
        new (input1_it) DPNPC_id<_DataType_input1>(q_ref, input1_data, input1_shape, input1_strides, input1_ndim);

        input1_it->broadcast_to_shape(result_shape, result_ndim);

        DPNPC_id<_DataType_input2>* input2_it;
        const size_t input2_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input2>);
        input2_it = reinterpret_cast<DPNPC_id<_DataType_input2>*>(dpnp_memory_alloc_c(q_ref, input2_it_size_in_bytes));
        new (input2_it) DPNPC_id<_DataType_input2>(q_ref, input2_data, input2_shape, input2_strides, input2_ndim);

        input2_it->broadcast_to_shape(result_shape, result_ndim);

        auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
            const size_t i = global_id[0]; /* for (size_t i = 0; i < result_size; ++i) */
            {
                const bool condition = (*condition_it)[i];
                const _DataType_output input1_elem = (*input1_it)[i];
                const _DataType_output input2_elem = (*input2_it)[i];
                result[i] = (condition) ? input1_elem : input2_elem;
            }
        };
        auto kernel_func = [&](sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_where_c_broadcast_kernel<_DataType_output, _DataType_input1, _DataType_input2>>(
                gws, kernel_parallel_for_func);
        };

        q.submit(kernel_func).wait();

        condition_it->~DPNPC_id();
        input1_it->~DPNPC_id();
        input2_it->~DPNPC_id();

        return event_ref;
    }
    else if (use_strides)
    {
        if ((result_ndim != condition_ndim) || (result_ndim != input1_ndim) || (result_ndim != input2_ndim))
        {
            throw std::runtime_error("Result ndim=" + std::to_string(result_ndim) +
                                     " mismatches with either condition ndim=" + std::to_string(condition_ndim) +
                                     " or input1 ndim=" + std::to_string(input1_ndim) +
                                     " or input2 ndim=" + std::to_string(input2_ndim));
        }

        /* memory transfer optimization, use USM-host for temporary speeds up tranfer to device */
        using usm_host_allocatorT = sycl::usm_allocator<shape_elem_type, sycl::usm::alloc::host>;

        size_t strides_size = 4 * result_ndim;
        shape_elem_type* dev_strides_data = sycl::malloc_device<shape_elem_type>(strides_size, q);

        /* create host temporary for packed strides managed by shared pointer */
        auto strides_host_packed =
            std::vector<shape_elem_type, usm_host_allocatorT>(strides_size, usm_host_allocatorT(q));

        /* packed vector is concatenation of result_strides, condition_strides, input1_strides and input2_strides */
        std::copy(result_strides, result_strides + result_ndim, strides_host_packed.begin());
        std::copy(condition_strides, condition_strides + result_ndim, strides_host_packed.begin() + result_ndim);
        std::copy(input1_strides, input1_strides + result_ndim, strides_host_packed.begin() + 2 * result_ndim);
        std::copy(input2_strides, input2_strides + result_ndim, strides_host_packed.begin() + 3 * result_ndim);

        auto copy_strides_ev =
            q.copy<shape_elem_type>(strides_host_packed.data(), dev_strides_data, strides_host_packed.size());

        auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
            const size_t output_id = global_id[0]; /* for (size_t i = 0; i < result_size; ++i) */
            {
                const shape_elem_type* result_strides_data = &dev_strides_data[0];
                const shape_elem_type* condition_strides_data = &dev_strides_data[result_ndim];
                const shape_elem_type* input1_strides_data = &dev_strides_data[2 * result_ndim];
                const shape_elem_type* input2_strides_data = &dev_strides_data[3 * result_ndim];

                size_t condition_id = 0;
                size_t input1_id = 0;
                size_t input2_id = 0;

                for (size_t i = 0; i < result_ndim; ++i)
                {
                    const size_t output_xyz_id =
                        get_xyz_id_by_id_inkernel(output_id, result_strides_data, result_ndim, i);
                    condition_id += output_xyz_id * condition_strides_data[i];
                    input1_id    += output_xyz_id * input1_strides_data[i];
                    input2_id    += output_xyz_id * input2_strides_data[i];
                }

                const bool condition = condition_data[condition_id];
                const _DataType_output input1_elem = input1_data[input1_id];
                const _DataType_output input2_elem = input2_data[input2_id];
                result[output_id] = (condition) ? input1_elem : input2_elem;
            }
        };
        auto kernel_func = [&](sycl::handler& cgh) {
            cgh.depends_on(copy_strides_ev);
            cgh.parallel_for<class dpnp_where_c_strides_kernel<_DataType_output, _DataType_input1, _DataType_input2>>(
                gws, kernel_parallel_for_func);
        };

        q.submit(kernel_func).wait();

        sycl::free(dev_strides_data, q);
        return event_ref;
    }
    else
    {
        auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
            const size_t i = global_id[0]; /* for (size_t i = 0; i < result_size; ++i) */

            const bool condition = condition_data[i];
            const _DataType_output input1_elem = input1_data[i];
            const _DataType_output input2_elem = input2_data[i];
            result[i] = (condition) ? input1_elem : input2_elem;
        };
        auto kernel_func = [&](sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_where_c_kernel<_DataType_output, _DataType_input1, _DataType_input2>>(
                gws, kernel_parallel_for_func);
        };
        event = q.submit(kernel_func);
    }

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    return DPCTLEvent_Copy(event_ref);

    return event_ref;
}

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
DPCTLSyclEventRef (*dpnp_where_ext_c)(DPCTLSyclQueueRef,
                                      void*,
                                      const size_t,
                                      const size_t,
                                      const shape_elem_type*,
                                      const shape_elem_type*,
                                      const void*,
                                      const size_t,
                                      const size_t,
                                      const shape_elem_type*,
                                      const shape_elem_type*,
                                      const void*,
                                      const size_t,
                                      const size_t,
                                      const shape_elem_type*,
                                      const shape_elem_type*,
                                      const void*,
                                      const size_t,
                                      const size_t,
                                      const shape_elem_type*,
                                      const shape_elem_type*,
                                      const DPCTLEventVectorRef) = dpnp_where_c<_DataType_output, _DataType_input1, _DataType_input2>;

template <DPNPFuncType FT1, DPNPFuncType... FTs>
static void func_map_searching_2arg_3type_core(func_map_t& fmap)
{
    ((fmap[DPNPFuncName::DPNP_FN_WHERE_EXT][FT1][FTs] =
          {populate_func_types<FT1, FTs>(),
           (void*)dpnp_where_ext_c<func_type_map_t::find_type<populate_func_types<FT1, FTs>()>,
                                   func_type_map_t::find_type<FT1>,
                                   func_type_map_t::find_type<FTs>>}),
     ...);
}

template <DPNPFuncType... FTs>
static void func_map_searching_2arg_3type_helper(func_map_t& fmap)
{
    ((func_map_searching_2arg_3type_core<FTs, FTs...>(fmap)), ...);
}

void func_map_init_searching(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmax_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmax_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmax_default_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_default_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmax_default_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_default_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmax_ext_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_ext_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmax_ext_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_ext_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmax_ext_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_ext_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmax_ext_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMAX_EXT][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmax_ext_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmin_default_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_default_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmin_default_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_default_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmin_default_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_default_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmin_default_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_default_c<double, int64_t>};

    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_argmin_ext_c<int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_ext_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_argmin_ext_c<int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_ext_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_argmin_ext_c<float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_ext_c<float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_argmin_ext_c<double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ARGMIN_EXT][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_argmin_ext_c<double, int64_t>};

    func_map_searching_2arg_3type_helper<eft_BLN, eft_INT, eft_LNG, eft_FLT, eft_DBL, eft_C64, eft_C128>(fmap);

    return;
}
