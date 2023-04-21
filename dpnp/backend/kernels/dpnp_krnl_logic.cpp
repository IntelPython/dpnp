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
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _DataType, typename _ResultType>
class dpnp_all_c_kernel;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef dpnp_all_c(DPCTLSyclQueueRef q_ref,
                             const void* array1_in,
                             void* result1,
                             const size_t size,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    static_assert(std::is_same_v<_ResultType, bool>, "Boolean result type is required");

    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!array1_in || !result1)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    const _DataType* array_in = static_cast<const _DataType*>(array1_in);
    bool* result = static_cast<bool*>(result1);

    auto fill_event = q.fill(result, true, 1);

    if (!size)
    {
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&fill_event);
        return DPCTLEvent_Copy(event_ref);
    }

    constexpr size_t lws = 64;
    constexpr size_t vec_sz = 8;

    auto gws_range = sycl::range<1>(((size + lws * vec_sz - 1) / (lws * vec_sz)) * lws);
    auto lws_range = sycl::range<1>(lws);
    sycl::nd_range<1> gws(gws_range, lws_range);

    auto kernel_parallel_for_func = [=](sycl::nd_item<1> nd_it) {
        auto gr = nd_it.get_group();
        const auto max_gr_size = gr.get_max_local_range()[0];
        const size_t start =
            vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) + gr.get_group_id()[0] * max_gr_size);
        const size_t end = sycl::min(start + vec_sz * max_gr_size, size);

        // each work-item reduces over "vec_sz" elements in the input array
        bool local_reduction = sycl::joint_none_of(
            gr, &array_in[start], &array_in[end], [&](_DataType elem) { return elem == static_cast<_DataType>(0); });

        if (gr.leader() && (local_reduction == false))
        {
            result[0] = false;
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for<class dpnp_all_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    auto event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType, typename _ResultType>
void dpnp_all_c(const void* array1_in, void* result1, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_all_c<_DataType, _ResultType>(q_ref,
                                                                     array1_in,
                                                                     result1,
                                                                     size,
                                                                     dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType, typename _ResultType>
void (*dpnp_all_default_c)(const void*, void*, const size_t) = dpnp_all_c<_DataType, _ResultType>;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef (*dpnp_all_ext_c)(DPCTLSyclQueueRef,
                                    const void*,
                                    void*,
                                    const size_t,
                                    const DPCTLEventVectorRef) = dpnp_all_c<_DataType, _ResultType>;

template <typename _DataType1, typename _DataType2, typename _ResultType>
class dpnp_allclose_c_kernel;

template <typename _DataType1, typename _DataType2, typename _ResultType>
DPCTLSyclEventRef dpnp_allclose_c(DPCTLSyclQueueRef q_ref,
                                  const void* array1_in,
                                  const void* array2_in,
                                  void* result1,
                                  const size_t size,
                                  double rtol_val,
                                  double atol_val,
                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!array1_in || !result1)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));
    sycl::event event;

    DPNPC_ptr_adapter<_DataType1> input1_ptr(q_ref, array1_in, size);
    DPNPC_ptr_adapter<_DataType2> input2_ptr(q_ref, array2_in, size);
    DPNPC_ptr_adapter<_ResultType> result1_ptr(q_ref, result1, 1, true, true);
    const _DataType1* array1 = input1_ptr.get_ptr();
    const _DataType2* array2 = input2_ptr.get_ptr();
    _ResultType* result = result1_ptr.get_ptr();

    result[0] = true;

    if (!size)
    {
        return event_ref;
    }

    sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        size_t i = global_id[0];

        if (std::abs(array1[i] - array2[i]) > (atol_val + rtol_val * std::abs(array2[i])))
        {
            result[0] = false;
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_allclose_c_kernel<_DataType1, _DataType2, _ResultType>>(gws,
                                                                                            kernel_parallel_for_func);
    };

    event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);

    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType1, typename _DataType2, typename _ResultType>
void dpnp_allclose_c(
    const void* array1_in, const void* array2_in, void* result1, const size_t size, double rtol_val, double atol_val)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_allclose_c<_DataType1, _DataType2, _ResultType>(q_ref,
                                                                                       array1_in,
                                                                                       array2_in,
                                                                                       result1,
                                                                                       size,
                                                                                       rtol_val,
                                                                                       atol_val,
                                                                                       dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType1, typename _DataType2, typename _ResultType>
void (*dpnp_allclose_default_c)(const void*,
                                const void*,
                                void*,
                                const size_t,
                                double,
                                double) = dpnp_allclose_c<_DataType1, _DataType2, _ResultType>;

template <typename _DataType1, typename _DataType2, typename _ResultType>
DPCTLSyclEventRef (*dpnp_allclose_ext_c)(
    DPCTLSyclQueueRef,
    const void*,
    const void*,
    void*,
    const size_t,
    double,
    double,
    const DPCTLEventVectorRef) = dpnp_allclose_c<_DataType1, _DataType2, _ResultType>;

template <typename _DataType, typename _ResultType>
class dpnp_any_c_kernel;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef dpnp_any_c(DPCTLSyclQueueRef q_ref,
                             const void* array1_in,
                             void* result1,
                             const size_t size,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    static_assert(std::is_same_v<_ResultType, bool>, "Boolean result type is required");

    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!array1_in || !result1)
    {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue*>(q_ref));

    const _DataType* array_in = static_cast<const _DataType*>(array1_in);
    bool* result = static_cast<bool*>(result1);

    auto fill_event = q.fill(result, false, 1);

    if (!size)
    {
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&fill_event);
        return DPCTLEvent_Copy(event_ref);
    }

    constexpr size_t lws = 64;
    constexpr size_t vec_sz = 8;

    auto gws_range = sycl::range<1>(((size + lws * vec_sz - 1) / (lws * vec_sz)) * lws);
    auto lws_range = sycl::range<1>(lws);
    sycl::nd_range<1> gws(gws_range, lws_range);

    auto kernel_parallel_for_func = [=](sycl::nd_item<1> nd_it) {
        auto gr = nd_it.get_group();
        const auto max_gr_size = gr.get_max_local_range()[0];
        const size_t start =
            vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) + gr.get_group_id()[0] * max_gr_size);
        const size_t end = sycl::min(start + vec_sz * max_gr_size, size);

        // each work-item reduces over "vec_sz" elements in the input array
        bool local_reduction = sycl::joint_any_of(
            gr, &array_in[start], &array_in[end], [&](_DataType elem) { return elem != static_cast<_DataType>(0); });

        if (gr.leader() && (local_reduction == true))
        {
            result[0] = true;
        }
    };

    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for<class dpnp_any_c_kernel<_DataType, _ResultType>>(gws, kernel_parallel_for_func);
    };

    auto event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType, typename _ResultType>
void dpnp_any_c(const void* array1_in, void* result1, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_any_c<_DataType, _ResultType>(q_ref,
                                                                     array1_in,
                                                                     result1,
                                                                     size,
                                                                     dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType, typename _ResultType>
void (*dpnp_any_default_c)(const void*, void*, const size_t) = dpnp_any_c<_DataType, _ResultType>;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef (*dpnp_any_ext_c)(DPCTLSyclQueueRef,
                                    const void*,
                                    void*,
                                    const size_t,
                                    const DPCTLEventVectorRef) = dpnp_any_c<_DataType, _ResultType>;


#define MACRO_1ARG_1TYPE_LOGIC_OP(__name__, __operation__)                                                             \
    template <typename _KernelNameSpecialization>                                                                      \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _KernelNameSpecialization>                                                                      \
    class __name__##_broadcast_kernel;                                                                                 \
                                                                                                                       \
    template <typename _KernelNameSpecialization>                                                                      \
    class __name__##_strides_kernel;                                                                                   \
                                                                                                                       \
    template <typename _DataType_input1>                                                                               \
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
                               const size_t* where,                                                                    \
                               const DPCTLEventVectorRef dep_event_vec_ref)                                            \
    {                                                                                                                  \
        /* avoid warning unused variable*/                                                                             \
        (result_shape);                                                                                                \
        (void)where;                                                                                                   \
        (void)dep_event_vec_ref;                                                                                       \
                                                                                                                       \
        DPCTLSyclEventRef event_ref = nullptr;                                                                         \
                                                                                                                       \
        if (!input1_size)                                                                                              \
        {                                                                                                              \
            return event_ref;                                                                                          \
        }                                                                                                              \
                                                                                                                       \
        sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));                                                     \
                                                                                                                       \
        _DataType_input1* input1_data = static_cast<_DataType_input1 *>(const_cast<void *>(input1_in));                \
        bool* result = static_cast<bool *>(result_out);                                                                \
                                                                                                                       \
        shape_elem_type* input1_shape_offsets = new shape_elem_type[input1_ndim];                                      \
                                                                                                                       \
        get_shape_offsets_inkernel(input1_shape, input1_ndim, input1_shape_offsets);                                   \
        bool use_strides = !array_equal(input1_strides, input1_ndim, input1_shape_offsets, input1_ndim);               \
        delete[] input1_shape_offsets;                                                                                 \
                                                                                                                       \
        if (use_strides)                                                                                               \
        {                                                                                                              \
            if (result_ndim != input1_ndim)                                                                            \
            {                                                                                                          \
                throw std::runtime_error("Result ndim=" + std::to_string(result_ndim) +                                \
                                         " mismatches with input1 ndim=" + std::to_string(input1_ndim));               \
            }                                                                                                          \
                                                                                                                       \
            /* memory transfer optimization, use USM-host for temporary speeds up tranfer to device */                 \
            using usm_host_allocatorT = sycl::usm_allocator<shape_elem_type, sycl::usm::alloc::host>;                  \
                                                                                                                       \
            size_t strides_size = 2 * result_ndim;                                                                     \
            shape_elem_type *dev_strides_data = sycl::malloc_device<shape_elem_type>(strides_size, q);                 \
                                                                                                                       \
            /* create host temporary for packed strides managed by shared pointer */                                   \
            auto strides_host_packed = std::vector<shape_elem_type, usm_host_allocatorT>(strides_size,                 \
                                                                                         usm_host_allocatorT(q));      \
                                                                                                                       \
            /* packed vector is concatenation of result_strides and input1_strides */                                  \
            std::copy(result_strides, result_strides + result_ndim, strides_host_packed.begin());                      \
            std::copy(input1_strides, input1_strides + result_ndim, strides_host_packed.begin() + result_ndim);        \
                                                                                                                       \
            auto copy_strides_ev = q.copy<shape_elem_type>(strides_host_packed.data(),                                 \
                                                           dev_strides_data,                                           \
                                                           strides_host_packed.size());                                \
                                                                                                                       \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {                                               \
                const size_t output_id = global_id[0]; /* for (size_t i = 0; i < result_size; ++i) */                  \
                {                                                                                                      \
                    const shape_elem_type *result_strides_data = &dev_strides_data[0];                                 \
                    const shape_elem_type *input1_strides_data = &dev_strides_data[result_ndim];                       \
                                                                                                                       \
                    size_t input1_id = 0;                                                                              \
                                                                                                                       \
                    for (size_t i = 0; i < result_ndim; ++i)                                                           \
                    {                                                                                                  \
                        const size_t output_xyz_id =                                                                   \
                            get_xyz_id_by_id_inkernel(output_id, result_strides_data, result_ndim, i);                 \
                        input1_id += output_xyz_id * input1_strides_data[i];                                           \
                    }                                                                                                  \
                                                                                                                       \
                    const _DataType_input1 input1_elem = input1_data[input1_id];                                       \
                    result[output_id] = __operation__;                                                                 \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](sycl::handler& cgh) {                                                               \
                cgh.depends_on(copy_strides_ev);                                                                       \
                cgh.parallel_for<class __name__##_strides_kernel<_DataType_input1>>(                                   \
                    sycl::range<1>(result_size), kernel_parallel_for_func);                                            \
            };                                                                                                         \
                                                                                                                       \
            q.submit(kernel_func).wait();                                                                              \
                                                                                                                       \
            sycl::free(dev_strides_data, q);                                                                           \
            return event_ref;                                                                                          \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            constexpr size_t lws = 64;                                                                                 \
            constexpr unsigned int vec_sz = 8;                                                                         \
            constexpr sycl::access::address_space global_space = sycl::access::address_space::global_space;            \
                                                                                                                       \
            auto gws_range = sycl::range<1>(((result_size + lws * vec_sz - 1) / (lws * vec_sz)) * lws);                \
            auto lws_range = sycl::range<1>(lws);                                                                      \
                                                                                                                       \
            auto kernel_parallel_for_func = [=](sycl::nd_item<1> nd_it) {                                              \
                auto sg = nd_it.get_sub_group();                                                                       \
                const auto max_sg_size = sg.get_max_local_range()[0];                                                  \
                const size_t start = vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) +                         \
                                               sg.get_group_id()[0] * max_sg_size);                                    \
                                                                                                                       \
                if (start + static_cast<size_t>(vec_sz) * max_sg_size < result_size) {                                 \
                    sycl::vec<_DataType_input1, vec_sz> x1 =                                                           \
                        sg.load<vec_sz>(sycl::multi_ptr<_DataType_input1, global_space>(&input1_data[start]));         \
                    sycl::vec<bool, vec_sz> res_vec;                                                                   \
                                                                                                                       \
                    for (size_t k = 0; k < vec_sz; ++k) {                                                              \
                        const _DataType_input1 input1_elem = x1[k];                                                    \
                        res_vec[k] = __operation__;                                                                    \
                    }                                                                                                  \
                    sg.store<vec_sz>(sycl::multi_ptr<bool, global_space>(&result[start]), res_vec);                    \
                                                                                                                       \
                }                                                                                                      \
                else {                                                                                                 \
                    for (size_t k = start; k < result_size; ++k) {                                                     \
                        const _DataType_input1 input1_elem = input1_data[k];                                           \
                        result[k] = __operation__;                                                                     \
                    }                                                                                                  \
                }                                                                                                      \
            };                                                                                                         \
                                                                                                                       \
            auto kernel_func = [&](sycl::handler& cgh) {                                                               \
                cgh.parallel_for<class __name__##_kernel<_DataType_input1>>(                                           \
                    sycl::nd_range<1>(gws_range, lws_range), kernel_parallel_for_func);                                \
            };                                                                                                         \
            sycl::event event = q.submit(kernel_func);                                                                 \
                                                                                                                       \
            event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);                                                   \
            return DPCTLEvent_Copy(event_ref);                                                                         \
        }                                                                                                              \
        return event_ref;                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    template <typename _DataType_input1>                                                                               \
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
                                        const size_t*,                                                                 \
                                        const DPCTLEventVectorRef) = __name__<_DataType_input1>;

#include <dpnp_gen_1arg_1type_tbl.hpp>

template <DPNPFuncType ... FTs>
static void func_map_logic_1arg_1type_helper(func_map_t& fmap)
{
    ((fmap[DPNPFuncName::DPNP_FN_LOGICAL_NOT_EXT][FTs][FTs] =
        {eft_BLN, (void*)dpnp_logical_not_c_ext<func_type_map_t::find_type<FTs>>}), ...);
}


#define MACRO_2ARG_2TYPES_LOGIC_OP(__name__, __operation__)                                                            \
    template <typename _KernelNameSpecialization1,                                                                     \
              typename _KernelNameSpecialization2>                                                                     \
    class __name__##_kernel;                                                                                           \
                                                                                                                       \
    template <typename _KernelNameSpecialization1,                                                                     \
              typename _KernelNameSpecialization2>                                                                     \
    class __name__##_broadcast_kernel;                                                                                 \
                                                                                                                       \
    template <typename _KernelNameSpecialization1,                                                                     \
              typename _KernelNameSpecialization2>                                                                     \
    class __name__##_strides_kernel;                                                                                   \
                                                                                                                       \
    template <typename _DataType_input1, typename _DataType_input2>                                                    \
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
        sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));                                                     \
                                                                                                                       \
        _DataType_input1* input1_data = static_cast<_DataType_input1 *>(const_cast<void *>(input1_in));                \
        _DataType_input2* input2_data = static_cast<_DataType_input2 *>(const_cast<void *>(input2_in));                \
        bool* result = static_cast<bool *>(result_out);                                                                \
                                                                                                                       \
        bool use_broadcasting = !array_equal(input1_shape, input1_ndim, input2_shape, input2_ndim);                    \
                                                                                                                       \
        shape_elem_type* input1_shape_offsets = new shape_elem_type[input1_ndim];                                      \
                                                                                                                       \
        get_shape_offsets_inkernel(input1_shape, input1_ndim, input1_shape_offsets);                                   \
        bool use_strides = !array_equal(input1_strides, input1_ndim, input1_shape_offsets, input1_ndim);               \
        delete[] input1_shape_offsets;                                                                                 \
                                                                                                                       \
        shape_elem_type* input2_shape_offsets = new shape_elem_type[input2_ndim];                                      \
                                                                                                                       \
        get_shape_offsets_inkernel(input2_shape, input2_ndim, input2_shape_offsets);                                   \
        use_strides =                                                                                                  \
            use_strides || !array_equal(input2_strides, input2_ndim, input2_shape_offsets, input2_ndim);               \
        delete[] input2_shape_offsets;                                                                                 \
                                                                                                                       \
        sycl::event event;                                                                                             \
        sycl::range<1> gws(result_size); /* used only when use_broadcasting or use_strides is true */                  \
                                                                                                                       \
        if (use_broadcasting)                                                                                          \
        {                                                                                                              \
            DPNPC_id<_DataType_input1>* input1_it;                                                                     \
            const size_t input1_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input1>);                                 \
            input1_it = reinterpret_cast<DPNPC_id<_DataType_input1>*>(dpnp_memory_alloc_c(q_ref,                       \
                                                                                          input1_it_size_in_bytes));   \
            new (input1_it)                                                                                            \
                DPNPC_id<_DataType_input1>(q_ref, input1_data, input1_shape, input1_strides, input1_ndim);             \
                                                                                                                       \
            input1_it->broadcast_to_shape(result_shape, result_ndim);                                                  \
                                                                                                                       \
            DPNPC_id<_DataType_input2>* input2_it;                                                                     \
            const size_t input2_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input2>);                                 \
            input2_it = reinterpret_cast<DPNPC_id<_DataType_input2>*>(dpnp_memory_alloc_c(q_ref,                       \
                                                                                          input2_it_size_in_bytes));   \
            new (input2_it)                                                                                            \
                DPNPC_id<_DataType_input2>(q_ref, input2_data, input2_shape, input2_strides, input2_ndim);             \
                                                                                                                       \
            input2_it->broadcast_to_shape(result_shape, result_ndim);                                                  \
                                                                                                                       \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {                                               \
                const size_t i = global_id[0]; /* for (size_t i = 0; i < result_size; ++i) */                          \
                {                                                                                                      \
                    const _DataType_input1 input1_elem = (*input1_it)[i];                                              \
                    const _DataType_input2 input2_elem = (*input2_it)[i];                                              \
                    result[i] = __operation__;                                                                         \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](sycl::handler& cgh) {                                                               \
                cgh.parallel_for<                                                                                      \
                    class __name__##_broadcast_kernel<_DataType_input1, _DataType_input2>>(                            \
                    gws, kernel_parallel_for_func);                                                                    \
            };                                                                                                         \
                                                                                                                       \
            q.submit(kernel_func).wait();                                                                              \
                                                                                                                       \
            input1_it->~DPNPC_id();                                                                                    \
            input2_it->~DPNPC_id();                                                                                    \
                                                                                                                       \
            return event_ref;                                                                                          \
        }                                                                                                              \
        else if (use_strides)                                                                                          \
        {                                                                                                              \
            if ((result_ndim != input1_ndim) || (result_ndim != input2_ndim))                                          \
            {                                                                                                          \
                throw std::runtime_error("Result ndim=" + std::to_string(result_ndim) +                                \
                                         " mismatches with either input1 ndim=" + std::to_string(input1_ndim) +        \
                                         " or input2 ndim=" + std::to_string(input2_ndim));                            \
            }                                                                                                          \
                                                                                                                       \
            /* memory transfer optimization, use USM-host for temporary speeds up tranfer to device */                 \
            using usm_host_allocatorT = sycl::usm_allocator<shape_elem_type, sycl::usm::alloc::host>;                  \
                                                                                                                       \
            size_t strides_size = 3 * result_ndim;                                                                     \
            shape_elem_type *dev_strides_data = sycl::malloc_device<shape_elem_type>(strides_size, q);                 \
                                                                                                                       \
            /* create host temporary for packed strides managed by shared pointer */                                   \
            auto strides_host_packed = std::vector<shape_elem_type, usm_host_allocatorT>(strides_size,                 \
                                                                                         usm_host_allocatorT(q));      \
                                                                                                                       \
            /* packed vector is concatenation of result_strides, input1_strides and input2_strides */                  \
            std::copy(result_strides, result_strides + result_ndim, strides_host_packed.begin());                      \
            std::copy(input1_strides, input1_strides + result_ndim, strides_host_packed.begin() + result_ndim);        \
            std::copy(input2_strides, input2_strides + result_ndim, strides_host_packed.begin() + 2 * result_ndim);    \
                                                                                                                       \
            auto copy_strides_ev = q.copy<shape_elem_type>(strides_host_packed.data(),                                 \
                                                           dev_strides_data,                                           \
                                                           strides_host_packed.size());                                \
                                                                                                                       \
            auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {                                               \
                const size_t output_id = global_id[0]; /* for (size_t i = 0; i < result_size; ++i) */                  \
                {                                                                                                      \
                    const shape_elem_type *result_strides_data = &dev_strides_data[0];                                 \
                    const shape_elem_type *input1_strides_data = &dev_strides_data[result_ndim];                       \
                    const shape_elem_type *input2_strides_data = &dev_strides_data[2 * result_ndim];                   \
                                                                                                                       \
                    size_t input1_id = 0;                                                                              \
                    size_t input2_id = 0;                                                                              \
                                                                                                                       \
                    for (size_t i = 0; i < result_ndim; ++i)                                                           \
                    {                                                                                                  \
                        const size_t output_xyz_id =                                                                   \
                            get_xyz_id_by_id_inkernel(output_id, result_strides_data, result_ndim, i);                 \
                        input1_id += output_xyz_id * input1_strides_data[i];                                           \
                        input2_id += output_xyz_id * input2_strides_data[i];                                           \
                    }                                                                                                  \
                                                                                                                       \
                    const _DataType_input1 input1_elem = input1_data[input1_id];                                       \
                    const _DataType_input2 input2_elem = input2_data[input2_id];                                       \
                    result[output_id] = __operation__;                                                                 \
                }                                                                                                      \
            };                                                                                                         \
            auto kernel_func = [&](sycl::handler& cgh) {                                                               \
                cgh.depends_on(copy_strides_ev);                                                                       \
                cgh.parallel_for<                                                                                      \
                    class __name__##_strides_kernel<_DataType_input1, _DataType_input2>>(                              \
                    gws, kernel_parallel_for_func);                                                                    \
            };                                                                                                         \
                                                                                                                       \
            q.submit(kernel_func).wait();                                                                              \
                                                                                                                       \
            sycl::free(dev_strides_data, q);                                                                           \
            return event_ref;                                                                                          \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            constexpr size_t lws = 64;                                                                                 \
            constexpr unsigned int vec_sz = 8;                                                                         \
            constexpr sycl::access::address_space global_space = sycl::access::address_space::global_space;            \
                                                                                                                       \
            auto gws_range = sycl::range<1>(((result_size + lws * vec_sz - 1) / (lws * vec_sz)) * lws);                \
            auto lws_range = sycl::range<1>(lws);                                                                      \
                                                                                                                       \
            auto kernel_parallel_for_func = [=](sycl::nd_item<1> nd_it) {                                              \
                auto sg = nd_it.get_sub_group();                                                                       \
                const auto max_sg_size = sg.get_max_local_range()[0];                                                  \
                const size_t start = vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) +                         \
                                               sg.get_group_id()[0] * max_sg_size);                                    \
                                                                                                                       \
                if (start + static_cast<size_t>(vec_sz) * max_sg_size < result_size) {                                 \
                    sycl::vec<_DataType_input1, vec_sz> x1 =                                                           \
                        sg.load<vec_sz>(sycl::multi_ptr<_DataType_input1, global_space>(&input1_data[start]));         \
                    sycl::vec<_DataType_input2, vec_sz> x2 =                                                           \
                        sg.load<vec_sz>(sycl::multi_ptr<_DataType_input2, global_space>(&input2_data[start]));         \
                    sycl::vec<bool, vec_sz> res_vec;                                                                   \
                                                                                                                       \
                    for (size_t k = 0; k < vec_sz; ++k) {                                                              \
                        const _DataType_input1 input1_elem = x1[k];                                                    \
                        const _DataType_input2 input2_elem = x2[k];                                                    \
                        res_vec[k] = __operation__;                                                                    \
                    }                                                                                                  \
                    sg.store<vec_sz>(sycl::multi_ptr<bool, global_space>(&result[start]), res_vec);                    \
                                                                                                                       \
                }                                                                                                      \
                else {                                                                                                 \
                    for (size_t k = start; k < result_size; ++k) {                                                     \
                        const _DataType_input1 input1_elem = input1_data[k];                                           \
                        const _DataType_input2 input2_elem = input2_data[k];                                           \
                        result[k] = __operation__;                                                                     \
                    }                                                                                                  \
                }                                                                                                      \
            };                                                                                                         \
                                                                                                                       \
            auto kernel_func = [&](sycl::handler& cgh) {                                                               \
                cgh.parallel_for<class __name__##_kernel<_DataType_input1, _DataType_input2>>(                         \
                    sycl::nd_range<1>(gws_range, lws_range), kernel_parallel_for_func);                                \
            };                                                                                                         \
            event = q.submit(kernel_func);                                                                             \
        }                                                                                                              \
                                                                                                                       \
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);                                                       \
        return DPCTLEvent_Copy(event_ref);                                                                             \
    }                                                                                                                  \
                                                                                                                       \
    template <typename _DataType_input1, typename _DataType_input2>                                                    \
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
                                        const DPCTLEventVectorRef) = __name__<_DataType_input1,                        \
                                                                              _DataType_input2>;

#include <dpnp_gen_2arg_2type_tbl.hpp>

template <DPNPFuncType FT1, DPNPFuncType ... FTs>
static void func_map_logic_2arg_2type_core(func_map_t& fmap)
{
    ((fmap[DPNPFuncName::DPNP_FN_EQUAL_EXT][FT1][FTs] =
        {eft_BLN, (void*)dpnp_equal_c_ext<func_type_map_t::find_type<FT1>, func_type_map_t::find_type<FTs>>}), ...);
    ((fmap[DPNPFuncName::DPNP_FN_GREATER_EXT][FT1][FTs] =
        {eft_BLN, (void*)dpnp_greater_c_ext<func_type_map_t::find_type<FT1>, func_type_map_t::find_type<FTs>>}), ...);
    ((fmap[DPNPFuncName::DPNP_FN_GREATER_EQUAL_EXT][FT1][FTs] =
        {eft_BLN, (void*)dpnp_greater_equal_c_ext<func_type_map_t::find_type<FT1>, func_type_map_t::find_type<FTs>>}), ...);
    ((fmap[DPNPFuncName::DPNP_FN_LESS_EXT][FT1][FTs] =
        {eft_BLN, (void*)dpnp_less_c_ext<func_type_map_t::find_type<FT1>, func_type_map_t::find_type<FTs>>}), ...);
    ((fmap[DPNPFuncName::DPNP_FN_LESS_EQUAL_EXT][FT1][FTs] =
        {eft_BLN, (void*)dpnp_less_equal_c_ext<func_type_map_t::find_type<FT1>, func_type_map_t::find_type<FTs>>}), ...);
    ((fmap[DPNPFuncName::DPNP_FN_LOGICAL_AND_EXT][FT1][FTs] =
        {eft_BLN, (void*)dpnp_logical_and_c_ext<func_type_map_t::find_type<FT1>, func_type_map_t::find_type<FTs>>}), ...);
    ((fmap[DPNPFuncName::DPNP_FN_LOGICAL_OR_EXT][FT1][FTs] =
        {eft_BLN, (void*)dpnp_logical_or_c_ext<func_type_map_t::find_type<FT1>, func_type_map_t::find_type<FTs>>}), ...);
    ((fmap[DPNPFuncName::DPNP_FN_LOGICAL_XOR_EXT][FT1][FTs] =
        {eft_BLN, (void*)dpnp_logical_xor_c_ext<func_type_map_t::find_type<FT1>, func_type_map_t::find_type<FTs>>}), ...);
    ((fmap[DPNPFuncName::DPNP_FN_NOT_EQUAL_EXT][FT1][FTs] =
        {eft_BLN, (void*)dpnp_not_equal_c_ext<func_type_map_t::find_type<FT1>, func_type_map_t::find_type<FTs>>}), ...);
}

template <DPNPFuncType ... FTs>
static void func_map_logic_2arg_2type_helper(func_map_t& fmap)
{
    ((func_map_logic_2arg_2type_core<FTs, FTs...>(fmap)), ...);
}

void func_map_init_logic(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_all_default_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_all_default_c<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_all_default_c<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_all_default_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_all_default_c<double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ALL_EXT][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_all_ext_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_all_ext_c<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_all_ext_c<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_all_ext_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_all_ext_c<double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL_EXT][eft_C64][eft_C64] = {eft_C64, (void*)dpnp_all_ext_c<std::complex<float>, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL_EXT][eft_C128][eft_C128] = {eft_C128, (void*)dpnp_all_ext_c<std::complex<double>, bool>};

    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_INT] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<int32_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_INT] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<int64_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_INT] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<float, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_INT] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<double, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_LNG] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<int32_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_LNG] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<int64_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_LNG] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<float, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_LNG] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<double, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_FLT] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<int32_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_FLT] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<int64_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_FLT] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_FLT] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<double, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_DBL] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<int32_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_DBL] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<int64_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_DBL] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<float, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_DBL] = {eft_BLN,
                                                              (void*)dpnp_allclose_default_c<double, double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_INT][eft_INT] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<int32_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_LNG][eft_INT] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<int64_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_FLT][eft_INT] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<float, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_DBL][eft_INT] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<double, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_INT][eft_LNG] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<int32_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_LNG][eft_LNG] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<int64_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_FLT][eft_LNG] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<float, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_DBL][eft_LNG] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<double, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_INT][eft_FLT] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<int32_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_LNG][eft_FLT] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<int64_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_FLT][eft_FLT] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_DBL][eft_FLT] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<double, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_INT][eft_DBL] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<int32_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_LNG][eft_DBL] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<int64_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_FLT][eft_DBL] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<float, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_DBL][eft_DBL] = {eft_BLN,
                                                                  (void*)dpnp_allclose_ext_c<double, double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ANY][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_any_default_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_any_default_c<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_any_default_c<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_any_default_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_any_default_c<double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ANY_EXT][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_any_ext_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY_EXT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_any_ext_c<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY_EXT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_any_ext_c<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY_EXT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_any_ext_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY_EXT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_any_ext_c<double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY_EXT][eft_C64][eft_C64] = {eft_C64, (void*)dpnp_any_ext_c<std::complex<float>, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY_EXT][eft_C128][eft_C128] = {eft_C128, (void*)dpnp_any_ext_c<std::complex<double>, bool>};

    func_map_logic_1arg_1type_helper<eft_BLN, eft_INT, eft_LNG, eft_FLT, eft_DBL>(fmap);
    func_map_logic_2arg_2type_helper<eft_BLN, eft_INT, eft_LNG, eft_FLT, eft_DBL>(fmap);

    return;
}
