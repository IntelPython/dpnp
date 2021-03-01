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

#include <cmath>
#include <iostream>

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "dpnp_iterator.hpp"
#include "dpnp_utils.hpp"
#include "queue_sycl.hpp"

namespace mkl_stats = oneapi::mkl::stats;

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>
class dpnp_sum_c_kernel;

template <typename _DataType_input, typename _DataType_output>
void dpnp_sum_c(const void* input_in,
                void* result_out,
                const size_t* input_shape,
                const size_t input_shape_ndim,
                const long* axes,
                const size_t axes_ndim,
                const void* initial, // type must be _DataType_output
                const long* where)
{
    (void)where; // avoid warning unused variable

    if ((input_in == nullptr) || (result_out == nullptr))
    {
        return;
    }

    const _DataType_output* initial_ptr = reinterpret_cast<const _DataType_output*>(initial);
    const _DataType_output init = (initial_ptr == nullptr) ? _DataType_output{0} : *initial_ptr;

    _DataType_input* input = reinterpret_cast<_DataType_input*>(const_cast<void*>(input_in));
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result_out);

    if (!input_shape && !input_shape_ndim)
    { // it is a scalar
        result[0] = input[0];

        return;
    }

    if constexpr ((std::is_same<_DataType_input, double>::value || std::is_same<_DataType_input, float>::value) &&
                  std::is_same<_DataType_input, _DataType_output>::value)
    {
        // Support is limited by
        // - 1D array (no axes)
        // - same types for input and output
        // - float64 and float32 types only
        if (axes_ndim < 1)
        {
            const size_t input_size =
                std::accumulate(input_shape, input_shape + input_shape_ndim, size_t(1), std::multiplies<size_t>());
            auto dataset = mkl_stats::make_dataset<mkl_stats::layout::row_major>(1, input_size, input);
            cl::sycl::event event = mkl_stats::raw_sum(DPNP_QUEUE, dataset, result);
            event.wait();

            return;
        }
    }

    DPNPC_id<_DataType_input> input_it(input, input_shape, input_shape_ndim);
    if ((axes != nullptr) && (axes_ndim > 0))
    {
        const std::vector<long> axes_vec(axes, axes + axes_ndim);
        input_it.set_axes(axes_vec);
    }

    const size_t output_size = input_it.get_output_size();
    auto policy =
        oneapi::dpl::execution::make_device_policy<dpnp_sum_c_kernel<_DataType_input, _DataType_output>>(DPNP_QUEUE);
    for (size_t output_id = 0; output_id < output_size; ++output_id)
    {
        // type of "init" determine internal algorithm accumulator type
        _DataType_output accumulator = std::reduce(
            policy, input_it.begin(output_id), input_it.end(output_id), init, std::plus<_DataType_output>());
        policy.queue().wait();

        result[output_id] = accumulator;
    }

    return;
}

template <typename _KernelNameSpecialization>
class dpnp_prod_c_kernel;

template <typename _DataType>
void dpnp_prod_c(void* array1_in, void* result1, size_t size)
{
    if (!size)
    {
        return;
    }

    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    auto policy = oneapi::dpl::execution::make_device_policy<dpnp_prod_c_kernel<_DataType>>(DPNP_QUEUE);

    result[0] = std::reduce(policy, array_1, array_1 + size, _DataType(1), std::multiplies<_DataType>());

    policy.queue().wait();

    return;
}

void func_map_init_reduction(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_prod_c<int>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_prod_c<long>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_prod_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_prod_c<double>};

    // WARNING. The meaning of the fmap is changed. Second argument represents RESULT_TYPE for this function
    // handle "out" and "type" parameters require user selection of return type
    // TODO. required refactoring of fmap to some kernelSelector
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_sum_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_sum_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_FLT] = {eft_FLT, (void*)dpnp_sum_c<int, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_sum_c<int, double>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_sum_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_sum_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_FLT] = {eft_FLT, (void*)dpnp_sum_c<long, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_sum_c<long, double>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_sum_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_sum_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sum_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_sum_c<float, double>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_sum_c<double, int>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_sum_c<double, long>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_FLT] = {eft_FLT, (void*)dpnp_sum_c<double, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sum_c<double, double>};

    return;
}
