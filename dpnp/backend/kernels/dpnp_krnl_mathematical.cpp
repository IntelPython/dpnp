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
#include <vector>

#include <dpnp_iface.hpp>

#include "dpnp_fptr.hpp"
#include "dpnp_iterator.hpp"
#include "dpnp_utils.hpp"
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class dpnp_around_c_kernel;

template <typename _DataType>
void dpnp_around_c(const void* input_in, void* result_out, const size_t input_size, const int decimals)
{
    (void)decimals;

    if (!input_size)
    {
        return;
    }

    sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(input_in, input_size);
    _DataType* input = input1_ptr.get_ptr();
    _DataType* result = reinterpret_cast<_DataType*>(result_out);

    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        event = oneapi::mkl::vm::rint(DPNP_QUEUE, input_size, input, result);
    }
    else
    {
        sycl::range<1> gws(input_size);
        auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
            size_t i = global_id[0];
            {
                result[i] = std::rint(input[i]);
            }
        };

        auto kernel_func = [&](sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_around_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();
}

template <typename _KernelNameSpecialization>
class dpnp_elemwise_absolute_c_kernel;

template <typename _DataType>
void dpnp_elemwise_absolute_c(const void* input1_in, void* result1, size_t size)
{
    if (!size)
    {
        return;
    }

    sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(input1_in, size);
    _DataType* array1 = input1_ptr.get_ptr();
    DPNPC_ptr_adapter<_DataType> result1_ptr(result1, size, false, true);
    _DataType* result = result1_ptr.get_ptr();

    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        // https://docs.oneapi.com/versions/latest/onemkl/abs.html
        event = oneapi::mkl::vm::abs(DPNP_QUEUE, size, array1, result);
    }
    else
    {
        sycl::range<1> gws(size);
        auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
            const size_t idx = global_id[0];

            if (array1[idx] >= 0)
            {
                result[idx] = array1[idx];
            }
            else
            {
                result[idx] = -1 * array1[idx];
            }
        };

        auto kernel_func = [&](sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_elemwise_absolute_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();
}

// template void dpnp_elemwise_absolute_c<double>(void* array1_in, void* result1, size_t size);
// template void dpnp_elemwise_absolute_c<float>(void* array1_in, void* result1, size_t size);
// template void dpnp_elemwise_absolute_c<long>(void* array1_in, void* result1, size_t size);
// template void dpnp_elemwise_absolute_c<int>(void* array1_in, void* result1, size_t size);

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
void dpnp_cross_c(void* result_out,
                  const void* input1_in,
                  const size_t input1_size,
                  const shape_elem_type* input1_shape,
                  const size_t input1_shape_ndim,
                  const void* input2_in,
                  const size_t input2_size,
                  const shape_elem_type* input2_shape,
                  const size_t input2_shape_ndim,
                  const size_t* where)
{
    (void)input1_size; // avoid warning unused variable
    (void)input1_shape;
    (void)input1_shape_ndim;
    (void)input2_size;
    (void)input2_shape;
    (void)input2_shape_ndim;
    (void)where;

    DPNPC_ptr_adapter<_DataType_input1> input1_ptr(input1_in, input1_size, true);
    DPNPC_ptr_adapter<_DataType_input2> input2_ptr(input2_in, input2_size, true);
    DPNPC_ptr_adapter<_DataType_output> result_ptr(result_out, input1_size, true, true);
    const _DataType_input1* input1 = input1_ptr.get_ptr();
    const _DataType_input2* input2 = input2_ptr.get_ptr();
    _DataType_output* result = result_ptr.get_ptr();

    result[0] = input1[1] * input2[2] - input1[2] * input2[1];

    result[1] = input1[2] * input2[0] - input1[0] * input2[2];

    result[2] = input1[0] * input2[1] - input1[1] * input2[0];

    return;
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>
class dpnp_cumprod_c_kernel;

template <typename _DataType_input, typename _DataType_output>
void dpnp_cumprod_c(void* array1_in, void* result1, size_t size)
{
    if (!size)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(array1_in, size, true);
    DPNPC_ptr_adapter<_DataType_output> result_ptr(result1, size, true, true);
    _DataType_input* array1 = input1_ptr.get_ptr();
    _DataType_output* result = result_ptr.get_ptr();

    _DataType_output cur_res = 1;

    for (size_t i = 0; i < size; ++i)
    {
        cur_res *= array1[i];
        result[i] = cur_res;
    }

    return;
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>
class dpnp_cumsum_c_kernel;

template <typename _DataType_input, typename _DataType_output>
void dpnp_cumsum_c(void* array1_in, void* result1, size_t size)
{
    if (!size)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType_input> input1_ptr(array1_in, size, true);
    DPNPC_ptr_adapter<_DataType_output> result_ptr(result1, size, true, true);
    _DataType_input* array1 = input1_ptr.get_ptr();
    _DataType_output* result = result_ptr.get_ptr();

    _DataType_output cur_res = 0;

    for (size_t i = 0; i < size; ++i)
    {
        cur_res += array1[i];
        result[i] = cur_res;
    }

    return;
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2, typename _KernelNameSpecialization3>
class dpnp_floor_divide_c_kernel;

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
void dpnp_floor_divide_c(void* result_out,
                         const void* input1_in,
                         const size_t input1_size,
                         const shape_elem_type* input1_shape,
                         const size_t input1_shape_ndim,
                         const void* input2_in,
                         const size_t input2_size,
                         const shape_elem_type* input2_shape,
                         const size_t input2_shape_ndim,
                         const size_t* where)
{
    (void)where;

    if (!input1_size || !input2_size)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType_input1> input1_ptr(input1_in, input1_size);
    DPNPC_ptr_adapter<_DataType_input2> input2_ptr(input2_in, input2_size);
    _DataType_input1* input1_data = input1_ptr.get_ptr();
    _DataType_input2* input2_data = input2_ptr.get_ptr();
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result_out);

    std::vector<shape_elem_type> result_shape =
        get_result_shape(input1_shape, input1_shape_ndim, input2_shape, input2_shape_ndim);

    DPNPC_id<_DataType_input1>* input1_it;
    const size_t input1_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input1>);
    input1_it = reinterpret_cast<DPNPC_id<_DataType_input1>*>(dpnp_memory_alloc_c(input1_it_size_in_bytes));
    new (input1_it) DPNPC_id<_DataType_input1>(input1_data, input1_shape, input1_shape_ndim);

    input1_it->broadcast_to_shape(result_shape);

    DPNPC_id<_DataType_input2>* input2_it;
    const size_t input2_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input2>);
    input2_it = reinterpret_cast<DPNPC_id<_DataType_input2>*>(dpnp_memory_alloc_c(input2_it_size_in_bytes));
    new (input2_it) DPNPC_id<_DataType_input2>(input2_data, input2_shape, input2_shape_ndim);

    input2_it->broadcast_to_shape(result_shape);

    const size_t result_size = input1_it->get_output_size();

    sycl::range<1> gws(result_size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        const size_t i = global_id[0]; /* for (size_t i = 0; i < result_size; ++i) */
        const _DataType_output input1_elem = (*input1_it)[i];
        const _DataType_output input2_elem = (*input2_it)[i];

        double div = (double)input1_elem / (double)input2_elem;
        result[i] = static_cast<_DataType_output>(sycl::floor(div));
    };
    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_floor_divide_c_kernel<_DataType_output, _DataType_input1, _DataType_input2>>(
            gws, kernel_parallel_for_func);
    };

    sycl::event event;

    if (input1_size == input2_size)
    {
        if constexpr ((std::is_same<_DataType_input1, double>::value || std::is_same<_DataType_input1, float>::value) &&
                      std::is_same<_DataType_input2, _DataType_input1>::value)
        {
            event = oneapi::mkl::vm::div(DPNP_QUEUE, input1_size, input1_data, input2_data, result);
            event.wait();
            event = oneapi::mkl::vm::floor(DPNP_QUEUE, input1_size, result, result);
        }
        else
        {
            event = DPNP_QUEUE.submit(kernel_func);
        }
    }
    else
    {
        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();

    input1_it->~DPNPC_id();
    input2_it->~DPNPC_id();

    dpnp_memory_free_c(input1_it);
    dpnp_memory_free_c(input2_it);
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>
class dpnp_modf_c_kernel;

template <typename _DataType_input, typename _DataType_output>
void dpnp_modf_c(void* array1_in, void* result1_out, void* result2_out, size_t size)
{
    sycl::event event;
    DPNPC_ptr_adapter<_DataType_input> input1_ptr(array1_in, size);
    _DataType_input* array1 = input1_ptr.get_ptr();
    _DataType_output* result1 = reinterpret_cast<_DataType_output*>(result1_out);
    _DataType_output* result2 = reinterpret_cast<_DataType_output*>(result2_out);

    if constexpr (std::is_same<_DataType_input, double>::value || std::is_same<_DataType_input, float>::value)
    {
        event = oneapi::mkl::vm::modf(DPNP_QUEUE, size, array1, result2, result1);
    }
    else
    {
        sycl::range<1> gws(size);
        auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
            {
                _DataType_input input_elem1 = array1[i];
                result2[i] = sycl::modf(double(input_elem1), &result1[i]);
            }
        };

        auto kernel_func = [&](sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_modf_c_kernel<_DataType_input, _DataType_output>>(gws,
                                                                                          kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2, typename _KernelNameSpecialization3>
class dpnp_remainder_c_kernel;

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
void dpnp_remainder_c(void* result_out,
                      const void* input1_in,
                      const size_t input1_size,
                      const shape_elem_type* input1_shape,
                      const size_t input1_shape_ndim,
                      const void* input2_in,
                      const size_t input2_size,
                      const shape_elem_type* input2_shape,
                      const size_t input2_shape_ndim,
                      const size_t* where)
{
    (void)where;

    if (!input1_size || !input2_size)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType_input1> input1_ptr(input1_in, input1_size);
    DPNPC_ptr_adapter<_DataType_input2> input2_ptr(input2_in, input2_size);
    _DataType_input1* input1_data = input1_ptr.get_ptr();
    _DataType_input2* input2_data = input2_ptr.get_ptr();
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result_out);

    std::vector<shape_elem_type> result_shape =
        get_result_shape(input1_shape, input1_shape_ndim, input2_shape, input2_shape_ndim);

    DPNPC_id<_DataType_input1>* input1_it;
    const size_t input1_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input1>);
    input1_it = reinterpret_cast<DPNPC_id<_DataType_input1>*>(dpnp_memory_alloc_c(input1_it_size_in_bytes));
    new (input1_it) DPNPC_id<_DataType_input1>(input1_data, input1_shape, input1_shape_ndim);

    input1_it->broadcast_to_shape(result_shape);

    DPNPC_id<_DataType_input2>* input2_it;
    const size_t input2_it_size_in_bytes = sizeof(DPNPC_id<_DataType_input2>);
    input2_it = reinterpret_cast<DPNPC_id<_DataType_input2>*>(dpnp_memory_alloc_c(input2_it_size_in_bytes));
    new (input2_it) DPNPC_id<_DataType_input2>(input2_data, input2_shape, input2_shape_ndim);

    input2_it->broadcast_to_shape(result_shape);

    const size_t result_size = input1_it->get_output_size();

    sycl::range<1> gws(result_size);
    auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
        const size_t i = global_id[0];
        const _DataType_output input1_elem = (*input1_it)[i];
        const _DataType_output input2_elem = (*input2_it)[i];
        double fmod_res = sycl::fmod((double)input1_elem, (double)input2_elem);
        double add = fmod_res + input2_elem;
        result[i] = sycl::fmod(add, (double)input2_elem);
    };
    auto kernel_func = [&](sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_remainder_c_kernel<_DataType_output, _DataType_input1, _DataType_input2>>(
            gws, kernel_parallel_for_func);
    };

    sycl::event event;

    if (input1_size == input2_size)
    {
        if constexpr ((std::is_same<_DataType_input1, double>::value || std::is_same<_DataType_input1, float>::value) &&
                      std::is_same<_DataType_input2, _DataType_input1>::value)
        {
            event = oneapi::mkl::vm::fmod(DPNP_QUEUE, input1_size, input1_data, input2_data, result);
            event.wait();
            event = oneapi::mkl::vm::add(DPNP_QUEUE, input1_size, result, input2_data, result);
            event.wait();
            event = oneapi::mkl::vm::fmod(DPNP_QUEUE, input1_size, result, input2_data, result);
        }
        else
        {
            event = DPNP_QUEUE.submit(kernel_func);
        }
    }
    else
    {
        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();

    input1_it->~DPNPC_id();
    input2_it->~DPNPC_id();
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2, typename _KernelNameSpecialization3>
class dpnp_trapz_c_kernel;

template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
void dpnp_trapz_c(
    const void* array1_in, const void* array2_in, void* result1, double dx, size_t array1_size, size_t array2_size)
{
    if ((array1_in == nullptr) || (array2_in == nullptr && array2_size > 1))
    {
        return;
    }

    sycl::event event;
    DPNPC_ptr_adapter<_DataType_input1> input1_ptr(array1_in, array1_size);
    DPNPC_ptr_adapter<_DataType_input2> input2_ptr(array2_in, array2_size);
    _DataType_input1* array1 = input1_ptr.get_ptr();
    _DataType_input2* array2 = input2_ptr.get_ptr();
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

    if (array1_size < 2)
    {
        const _DataType_output init_val = 0;
        dpnp_memory_memcpy_c(result, &init_val, sizeof(_DataType_output)); // result[0] = 0;
        return;
    }

    if (array1_size == array2_size)
    {
        size_t cur_res_size = array1_size - 2;

        _DataType_output* cur_res =
            reinterpret_cast<_DataType_output*>(dpnp_memory_alloc_c((cur_res_size) * sizeof(_DataType_output)));

        sycl::range<1> gws(cur_res_size);
        auto kernel_parallel_for_func = [=](sycl::id<1> global_id) {
            size_t i = global_id[0];
            {
                cur_res[i] = array1[i + 1] * (array2[i + 2] - array2[i]);
            }
        };

        auto kernel_func = [&](sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_trapz_c_kernel<_DataType_input1, _DataType_input2, _DataType_output>>(
                gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);

        event.wait();

        shape_elem_type _shape = cur_res_size;
        dpnp_sum_c<_DataType_output, _DataType_output>(result, cur_res, &_shape, 1, NULL, 0, NULL, NULL);

        dpnp_memory_free_c(cur_res);

        result[0] += array1[0] * (array2[1] - array2[0]) +
                     array1[array1_size - 1] * (array2[array2_size - 1] - array2[array2_size - 2]);

        result[0] *= 0.5;
    }
    else
    {
        shape_elem_type _shape = array1_size;
        dpnp_sum_c<_DataType_output, _DataType_input1>(result, array1, &_shape, 1, NULL, 0, NULL, NULL);

        result[0] -= (array1[0] + array1[array1_size - 1]) * 0.5;
        result[0] *= dx;
    }
}

void func_map_init_mathematical(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_elemwise_absolute_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_elemwise_absolute_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_elemwise_absolute_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_elemwise_absolute_c<double>};

    fmap[DPNPFuncName::DPNP_FN_AROUND][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_around_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_AROUND][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_around_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_AROUND][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_around_c<float>};
    fmap[DPNPFuncName::DPNP_FN_AROUND][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_around_c<double>};

    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_cross_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_cross_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_cross_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_cross_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_cross_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_cross_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_cross_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_cross_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_cross_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_cross_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cross_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_cross_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_cross_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_cross_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_cross_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cross_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_CUMPROD][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_cumprod_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CUMPROD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_cumprod_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CUMPROD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cumprod_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CUMPROD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cumprod_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CUMSUM][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_cumsum_c<int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CUMSUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_cumsum_c<int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_CUMSUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cumsum_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CUMSUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cumsum_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_INT][eft_INT] = {
        eft_INT, (void*)dpnp_floor_divide_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_INT][eft_LNG] = {
        eft_LNG, (void*)dpnp_floor_divide_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_INT][eft_FLT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_INT][eft_DBL] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_LNG][eft_INT] = {
        eft_LNG, (void*)dpnp_floor_divide_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_LNG][eft_LNG] = {
        eft_LNG, (void*)dpnp_floor_divide_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_LNG][eft_FLT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_LNG][eft_DBL] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_FLT][eft_INT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_FLT][eft_LNG] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_FLT][eft_FLT] = {eft_FLT,
                                                                  (void*)dpnp_floor_divide_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_FLT][eft_DBL] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_DBL][eft_INT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_DBL][eft_LNG] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_DBL][eft_FLT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_DBL][eft_DBL] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MODF][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_modf_c<int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MODF][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_modf_c<int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_MODF][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_modf_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_MODF][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_modf_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_INT][eft_INT] = {eft_INT,
                                                               (void*)dpnp_remainder_c<int32_t, int32_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_INT][eft_LNG] = {eft_LNG,
                                                               (void*)dpnp_remainder_c<int64_t, int32_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_INT][eft_FLT] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, int32_t, float>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_INT][eft_DBL] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_LNG][eft_INT] = {eft_LNG,
                                                               (void*)dpnp_remainder_c<int64_t, int64_t, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_LNG][eft_LNG] = {eft_LNG,
                                                               (void*)dpnp_remainder_c<int64_t, int64_t, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_LNG][eft_FLT] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, int64_t, float>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_LNG][eft_DBL] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_FLT][eft_INT] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, float, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_FLT][eft_LNG] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, float, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_remainder_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_remainder_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_DBL][eft_INT] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, double, int32_t>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_DBL][eft_LNG] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, double, int64_t>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_remainder_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_DBL][eft_DBL] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_trapz_c<int32_t, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_trapz_c<int32_t, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_trapz_c<int32_t, float, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_trapz_c<int32_t, double, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_trapz_c<int64_t, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_trapz_c<int64_t, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_trapz_c<int64_t, float, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_trapz_c<int64_t, double, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_trapz_c<float, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_trapz_c<float, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_trapz_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_trapz_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_trapz_c<double, int32_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_trapz_c<double, int64_t, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_trapz_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_trapz_c<double, double, double>};

    return;
}
