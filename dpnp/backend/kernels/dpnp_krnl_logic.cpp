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

#include <iostream>

#include "dpnp_fptr.hpp"
#include "dpnp_iface.hpp"
#include "dpnp_iterator.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

// dpctl tensor headers
#include "kernels/alignment.hpp"

using dpctl::tensor::kernels::alignment_utils::is_aligned;
using dpctl::tensor::kernels::alignment_utils::required_alignment;

template <typename _DataType, typename _ResultType>
class dpnp_all_c_kernel;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef dpnp_all_c(DPCTLSyclQueueRef q_ref,
                             const void *array1_in,
                             void *result1,
                             const size_t size,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    static_assert(std::is_same_v<_ResultType, bool>,
                  "Boolean result type is required");

    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!array1_in || !result1) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    const _DataType *array_in = static_cast<const _DataType *>(array1_in);
    bool *result = static_cast<bool *>(result1);

    auto fill_event = q.fill(result, true, 1);

    if (!size) {
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&fill_event);
        return DPCTLEvent_Copy(event_ref);
    }

    constexpr size_t lws = 64;
    constexpr size_t vec_sz = 8;

    auto gws_range =
        sycl::range<1>(((size + lws * vec_sz - 1) / (lws * vec_sz)) * lws);
    auto lws_range = sycl::range<1>(lws);
    sycl::nd_range<1> gws(gws_range, lws_range);

    auto kernel_parallel_for_func = [=](sycl::nd_item<1> nd_it) {
        auto gr = nd_it.get_sub_group();
        const auto max_gr_size = gr.get_max_local_range()[0];
        const size_t start =
            vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) +
                      gr.get_group_id()[0] * max_gr_size);
        const size_t end = sycl::min(start + vec_sz * max_gr_size, size);

        // each work-item reduces over "vec_sz" elements in the input array
        bool local_reduction = sycl::joint_none_of(
            gr, &array_in[start], &array_in[end],
            [&](_DataType elem) { return elem == static_cast<_DataType>(0); });

        if (gr.leader() && (local_reduction == false)) {
            result[0] = false;
        }
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for<class dpnp_all_c_kernel<_DataType, _ResultType>>(
            gws, kernel_parallel_for_func);
    };

    auto event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType, typename _ResultType>
void dpnp_all_c(const void *array1_in, void *result1, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_all_c<_DataType, _ResultType>(
        q_ref, array1_in, result1, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType, typename _ResultType>
void (*dpnp_all_default_c)(const void *,
                           void *,
                           const size_t) = dpnp_all_c<_DataType, _ResultType>;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef (*dpnp_all_ext_c)(DPCTLSyclQueueRef,
                                    const void *,
                                    void *,
                                    const size_t,
                                    const DPCTLEventVectorRef) =
    dpnp_all_c<_DataType, _ResultType>;

template <typename _DataType1, typename _DataType2, typename _TolType>
class dpnp_allclose_kernel;

template <typename _DataType1, typename _DataType2, typename _TolType>
static sycl::event dpnp_allclose(sycl::queue &q,
                                 const _DataType1 *array1,
                                 const _DataType2 *array2,
                                 bool *result,
                                 const size_t size,
                                 const _TolType rtol_val,
                                 const _TolType atol_val)
{
    sycl::event fill_event = q.fill(result, true, 1);
    if (!size) {
        return fill_event;
    }

    constexpr size_t lws = 64;
    constexpr size_t vec_sz = 8;

    auto gws_range =
        sycl::range<1>(((size + lws * vec_sz - 1) / (lws * vec_sz)) * lws);
    auto lws_range = sycl::range<1>(lws);
    sycl::nd_range<1> gws(gws_range, lws_range);

    auto kernel_parallel_for_func = [=](sycl::nd_item<1> nd_it) {
        auto gr = nd_it.get_sub_group();
        const auto max_gr_size = gr.get_max_local_range()[0];
        const auto gr_size = gr.get_local_linear_range();
        const size_t start =
            vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) +
                      gr.get_group_linear_id() * max_gr_size);
        const size_t end = sycl::min(start + vec_sz * gr_size, size);

        // each work-item iterates over "vec_sz" elements in the input arrays
        bool partial = true;

        for (size_t i = start + gr.get_local_linear_id(); i < end; i += gr_size)
        {
            if constexpr (std::is_floating_point_v<_DataType1> &&
                          std::is_floating_point_v<_DataType2>)
            {
                if (std::isinf(array1[i]) || std::isinf(array2[i])) {
                    partial &= (array1[i] == array2[i]);
                    continue;
                }

                // workaround for std::inf which does not work on CPU
                // [CMPLRLLVM-51856]
                if (array1[i] == std::numeric_limits<_DataType1>::infinity()) {
                    partial &= (array1[i] == array2[i]);
                    continue;
                }
                else if (array1[i] ==
                         -std::numeric_limits<_DataType1>::infinity()) {
                    partial &= (array1[i] == array2[i]);
                    continue;
                }
                else if (array2[i] ==
                         std::numeric_limits<_DataType2>::infinity()) {
                    partial &= (array1[i] == array2[i]);
                    continue;
                }
                else if (array2[i] ==
                         -std::numeric_limits<_DataType2>::infinity()) {
                    partial &= (array1[i] == array2[i]);
                    continue;
                }
            }

            // casting integral to floating type to avoid bad behavior
            // on abs(MIN_INT), which leads to undefined result
            using _Arr2Type = std::conditional_t<std::is_integral_v<_DataType2>,
                                                 _TolType, _DataType2>;
            _Arr2Type arr2 = static_cast<_Arr2Type>(array2[i]);

            partial &= (std::abs(array1[i] - arr2) <=
                        (atol_val + rtol_val * std::abs(arr2)));
        }
        partial = sycl::all_of_group(gr, partial);

        if (gr.leader() && (partial == false)) {
            result[0] = false;
        }
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for<
            class dpnp_allclose_kernel<_DataType1, _DataType2, _TolType>>(
            gws, kernel_parallel_for_func);
    };

    return q.submit(kernel_func);
}

template <typename _DataType1, typename _DataType2, typename _ResultType>
DPCTLSyclEventRef dpnp_allclose_c(DPCTLSyclQueueRef q_ref,
                                  const void *array1_in,
                                  const void *array2_in,
                                  void *result1,
                                  const size_t size,
                                  double rtol_val,
                                  double atol_val,
                                  const DPCTLEventVectorRef dep_event_vec_ref)
{
    static_assert(std::is_same_v<_ResultType, bool>,
                  "Boolean result type is required");

    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!array1_in || !result1) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));
    sycl::event event;

    const _DataType1 *array1 = static_cast<const _DataType1 *>(array1_in);
    const _DataType2 *array2 = static_cast<const _DataType2 *>(array2_in);
    bool *result = static_cast<bool *>(result1);

    if (q.get_device().has(sycl::aspect::fp64)) {
        event =
            dpnp_allclose(q, array1, array2, result, size, rtol_val, atol_val);
    }
    else {
        float rtol = static_cast<float>(rtol_val);
        float atol = static_cast<float>(atol_val);
        event = dpnp_allclose(q, array1, array2, result, size, rtol, atol);
    }

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType1, typename _DataType2, typename _ResultType>
void dpnp_allclose_c(const void *array1_in,
                     const void *array2_in,
                     void *result1,
                     const size_t size,
                     double rtol_val,
                     double atol_val)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref =
        dpnp_allclose_c<_DataType1, _DataType2, _ResultType>(
            q_ref, array1_in, array2_in, result1, size, rtol_val, atol_val,
            dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
}

template <typename _DataType1, typename _DataType2, typename _ResultType>
void (*dpnp_allclose_default_c)(const void *,
                                const void *,
                                void *,
                                const size_t,
                                double,
                                double) =
    dpnp_allclose_c<_DataType1, _DataType2, _ResultType>;

template <typename _DataType1, typename _DataType2, typename _ResultType>
DPCTLSyclEventRef (*dpnp_allclose_ext_c)(DPCTLSyclQueueRef,
                                         const void *,
                                         const void *,
                                         void *,
                                         const size_t,
                                         double,
                                         double,
                                         const DPCTLEventVectorRef) =
    dpnp_allclose_c<_DataType1, _DataType2, _ResultType>;

template <typename _DataType, typename _ResultType>
class dpnp_any_c_kernel;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef dpnp_any_c(DPCTLSyclQueueRef q_ref,
                             const void *array1_in,
                             void *result1,
                             const size_t size,
                             const DPCTLEventVectorRef dep_event_vec_ref)
{
    static_assert(std::is_same_v<_ResultType, bool>,
                  "Boolean result type is required");

    // avoid warning unused variable
    (void)dep_event_vec_ref;

    DPCTLSyclEventRef event_ref = nullptr;

    if (!array1_in || !result1) {
        return event_ref;
    }

    sycl::queue q = *(reinterpret_cast<sycl::queue *>(q_ref));

    const _DataType *array_in = static_cast<const _DataType *>(array1_in);
    bool *result = static_cast<bool *>(result1);

    auto fill_event = q.fill(result, false, 1);

    if (!size) {
        event_ref = reinterpret_cast<DPCTLSyclEventRef>(&fill_event);
        return DPCTLEvent_Copy(event_ref);
    }

    constexpr size_t lws = 64;
    constexpr size_t vec_sz = 8;

    auto gws_range =
        sycl::range<1>(((size + lws * vec_sz - 1) / (lws * vec_sz)) * lws);
    auto lws_range = sycl::range<1>(lws);
    sycl::nd_range<1> gws(gws_range, lws_range);

    auto kernel_parallel_for_func = [=](sycl::nd_item<1> nd_it) {
        auto gr = nd_it.get_sub_group();
        const auto max_gr_size = gr.get_max_local_range()[0];
        const size_t start =
            vec_sz * (nd_it.get_group(0) * nd_it.get_local_range(0) +
                      gr.get_group_id()[0] * max_gr_size);
        const size_t end = sycl::min(start + vec_sz * max_gr_size, size);

        // each work-item reduces over "vec_sz" elements in the input array
        bool local_reduction = sycl::joint_any_of(
            gr, &array_in[start], &array_in[end],
            [&](_DataType elem) { return elem != static_cast<_DataType>(0); });

        if (gr.leader() && (local_reduction == true)) {
            result[0] = true;
        }
    };

    auto kernel_func = [&](sycl::handler &cgh) {
        cgh.depends_on(fill_event);
        cgh.parallel_for<class dpnp_any_c_kernel<_DataType, _ResultType>>(
            gws, kernel_parallel_for_func);
    };

    auto event = q.submit(kernel_func);

    event_ref = reinterpret_cast<DPCTLSyclEventRef>(&event);
    return DPCTLEvent_Copy(event_ref);
}

template <typename _DataType, typename _ResultType>
void dpnp_any_c(const void *array1_in, void *result1, const size_t size)
{
    DPCTLSyclQueueRef q_ref = reinterpret_cast<DPCTLSyclQueueRef>(&DPNP_QUEUE);
    DPCTLEventVectorRef dep_event_vec_ref = nullptr;
    DPCTLSyclEventRef event_ref = dpnp_any_c<_DataType, _ResultType>(
        q_ref, array1_in, result1, size, dep_event_vec_ref);
    DPCTLEvent_WaitAndThrow(event_ref);
    DPCTLEvent_Delete(event_ref);
}

template <typename _DataType, typename _ResultType>
void (*dpnp_any_default_c)(const void *,
                           void *,
                           const size_t) = dpnp_any_c<_DataType, _ResultType>;

template <typename _DataType, typename _ResultType>
DPCTLSyclEventRef (*dpnp_any_ext_c)(DPCTLSyclQueueRef,
                                    const void *,
                                    void *,
                                    const size_t,
                                    const DPCTLEventVectorRef) =
    dpnp_any_c<_DataType, _ResultType>;

void func_map_init_logic(func_map_t &fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_all_default_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_all_default_c<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_all_default_c<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_all_default_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALL][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_all_default_c<double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_INT] = {
        eft_BLN, (void *)dpnp_allclose_default_c<int32_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_INT] = {
        eft_BLN, (void *)dpnp_allclose_default_c<int64_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_INT] = {
        eft_BLN, (void *)dpnp_allclose_default_c<float, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_INT] = {
        eft_BLN, (void *)dpnp_allclose_default_c<double, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_LNG] = {
        eft_BLN, (void *)dpnp_allclose_default_c<int32_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_LNG] = {
        eft_BLN, (void *)dpnp_allclose_default_c<int64_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_LNG] = {
        eft_BLN, (void *)dpnp_allclose_default_c<float, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_LNG] = {
        eft_BLN, (void *)dpnp_allclose_default_c<double, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_FLT] = {
        eft_BLN, (void *)dpnp_allclose_default_c<int32_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_FLT] = {
        eft_BLN, (void *)dpnp_allclose_default_c<int64_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_FLT] = {
        eft_BLN, (void *)dpnp_allclose_default_c<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_FLT] = {
        eft_BLN, (void *)dpnp_allclose_default_c<double, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_INT][eft_DBL] = {
        eft_BLN, (void *)dpnp_allclose_default_c<int32_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_LNG][eft_DBL] = {
        eft_BLN, (void *)dpnp_allclose_default_c<int64_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_FLT][eft_DBL] = {
        eft_BLN, (void *)dpnp_allclose_default_c<float, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE][eft_DBL][eft_DBL] = {
        eft_BLN, (void *)dpnp_allclose_default_c<double, double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_INT][eft_INT] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<int32_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_LNG][eft_INT] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<int64_t, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_FLT][eft_INT] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<float, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_DBL][eft_INT] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<double, int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_INT][eft_LNG] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<int32_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_LNG][eft_LNG] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<int64_t, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_FLT][eft_LNG] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<float, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_DBL][eft_LNG] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<double, int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_INT][eft_FLT] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<int32_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_LNG][eft_FLT] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<int64_t, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_FLT][eft_FLT] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_DBL][eft_FLT] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<double, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_INT][eft_DBL] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<int32_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_LNG][eft_DBL] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<int64_t, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_FLT][eft_DBL] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<float, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_ALLCLOSE_EXT][eft_DBL][eft_DBL] = {
        eft_BLN, (void *)dpnp_allclose_ext_c<double, double, bool>};

    fmap[DPNPFuncName::DPNP_FN_ANY][eft_BLN][eft_BLN] = {
        eft_BLN, (void *)dpnp_any_default_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_INT][eft_INT] = {
        eft_INT, (void *)dpnp_any_default_c<int32_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_LNG][eft_LNG] = {
        eft_LNG, (void *)dpnp_any_default_c<int64_t, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_FLT][eft_FLT] = {
        eft_FLT, (void *)dpnp_any_default_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_ANY][eft_DBL][eft_DBL] = {
        eft_DBL, (void *)dpnp_any_default_c<double, bool>};

    return;
}
