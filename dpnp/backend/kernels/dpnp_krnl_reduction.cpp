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

template <typename _DataType>
_DataType* get_array_ptr(const void* __array)
{
    void* const_ptr = const_cast<void*>(__array);
    _DataType* ptr = reinterpret_cast<_DataType*>(const_ptr);

    return ptr;
}

template <typename _DataType>
_DataType get_initial_value(const void* __initial, _DataType default_val)
{
    const _DataType* initial_ptr = reinterpret_cast<const _DataType*>(__initial);
    const _DataType init_val = (initial_ptr == nullptr) ? default_val : *initial_ptr;

    return init_val;
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>
class dpnp_sum_c_kernel;

template <typename _DataType_output, typename _DataType_input>
void dpnp_sum_c(void* result_out,
                const void* input_in,
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

    const _DataType_output init = get_initial_value<_DataType_output>(initial, 0);

    _DataType_input* input = get_array_ptr<_DataType_input>(input_in);
    _DataType_output* result = get_array_ptr<_DataType_output>(result_out);

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
    input_it.set_axes(axes, axes_ndim);

    const size_t output_size = input_it.get_output_size();
    auto policy =
        oneapi::dpl::execution::make_device_policy<dpnp_sum_c_kernel<_DataType_output, _DataType_input>>(DPNP_QUEUE);
    for (size_t output_id = 0; output_id < output_size; ++output_id)
    {
        // type of "init" determine internal algorithm accumulator type
        _DataType_output accumulator = std::reduce(
            policy, input_it.begin(output_id), input_it.end(output_id), init, std::plus<_DataType_output>());
        policy.queue().wait(); // TODO move out of the loop

        result[output_id] = accumulator;
    }

    return;
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>
class dpnp_prod_c_kernel;

template <typename _DataType_output, typename _DataType_input>
void dpnp_prod_c(void* result_out,
                 const void* input_in,
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

    const _DataType_output init = get_initial_value<_DataType_output>(initial, 1);

    _DataType_input* input = get_array_ptr<_DataType_input>(input_in);
    _DataType_output* result = get_array_ptr<_DataType_output>(result_out);

    if (!input_shape && !input_shape_ndim)
    { // it is a scalar
        result[0] = input[0];

        return;
    }

    DPNPC_id<_DataType_input> input_it(input, input_shape, input_shape_ndim);
    input_it.set_axes(axes, axes_ndim);

    const size_t output_size = input_it.get_output_size();
    auto policy =
        oneapi::dpl::execution::make_device_policy<dpnp_prod_c_kernel<_DataType_output, _DataType_input>>(DPNP_QUEUE);
    for (size_t output_id = 0; output_id < output_size; ++output_id)
    {
        // type of "init" determine internal algorithm accumulator type
        _DataType_output accumulator = std::reduce(
            policy, input_it.begin(output_id), input_it.end(output_id), init, std::multiplies<_DataType_output>());
        policy.queue().wait(); // TODO move out of the loop

        result[output_id] = accumulator;
    }

    return;
}

void func_map_init_reduction(func_map_t& fmap)
{
    // WARNING. The meaning of the fmap is changed. Second argument represents RESULT_TYPE for this function
    // handle "out" and "type" parameters require user selection of return type
    // TODO. required refactoring of fmap to some kernelSelector
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_prod_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_prod_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_FLT] = {eft_FLT, (void*)dpnp_prod_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_prod_c<double, int>};

    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_prod_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_prod_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_FLT] = {eft_FLT, (void*)dpnp_prod_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_prod_c<double, long>};

    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_prod_c<int, float>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_prod_c<long, float>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_prod_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_prod_c<double, float>};

    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_prod_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_prod_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_FLT] = {eft_FLT, (void*)dpnp_prod_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_PROD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_prod_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_sum_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_sum_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_FLT] = {eft_FLT, (void*)dpnp_sum_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_sum_c<double, int>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_INT] = {eft_INT, (void*)dpnp_sum_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_sum_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_FLT] = {eft_FLT, (void*)dpnp_sum_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_sum_c<double, long>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_INT] = {eft_INT, (void*)dpnp_sum_c<int, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_LNG] = {eft_LNG, (void*)dpnp_sum_c<long, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_sum_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_sum_c<double, float>};

    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_INT] = {eft_INT, (void*)dpnp_sum_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_LNG] = {eft_LNG, (void*)dpnp_sum_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_FLT] = {eft_FLT, (void*)dpnp_sum_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_SUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_sum_c<double, double>};

    return;
}
