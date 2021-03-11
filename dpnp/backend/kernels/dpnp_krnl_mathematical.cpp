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
#include "dpnp_utils.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class dpnp_elemwise_absolute_c_kernel;

template <typename _DataType>
void dpnp_elemwise_absolute_c(void* array1_in, void* result1, size_t size)
{
    if (!size)
    {
        return;
    }

    cl::sycl::event event;
    _DataType* array1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    if constexpr (std::is_same<_DataType, double>::value || std::is_same<_DataType, float>::value)
    {
        // https://docs.oneapi.com/versions/latest/onemkl/abs.html
        event = oneapi::mkl::vm::abs(DPNP_QUEUE, size, array1, result);
    }
    else
    {
        cl::sycl::range<1> gws(size);
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
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

        auto kernel_func = [&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_elemwise_absolute_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();
}

template void dpnp_elemwise_absolute_c<double>(void* array1_in, void* result1, size_t size);
template void dpnp_elemwise_absolute_c<float>(void* array1_in, void* result1, size_t size);
template void dpnp_elemwise_absolute_c<long>(void* array1_in, void* result1, size_t size);
template void dpnp_elemwise_absolute_c<int>(void* array1_in, void* result1, size_t size);

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2, typename _KernelNameSpecialization3>
class dpnp_cross_c_kernel;

template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
void dpnp_cross_c(void* array1_in, void* array2_in, void* result1, size_t size)
{
    (void)size; // avoid warning unused variable
    _DataType_input1* array1 = reinterpret_cast<_DataType_input1*>(array1_in);
    _DataType_input2* array2 = reinterpret_cast<_DataType_input2*>(array2_in);
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

    result[0] = array1[1] * array2[2] - array1[2] * array2[1];

    result[1] = array1[2] * array2[0] - array1[0] * array2[2];

    result[2] = array1[0] * array2[1] - array1[1] * array2[0];

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

    _DataType_input* array1 = reinterpret_cast<_DataType_input*>(array1_in);
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

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

    _DataType_input* array1 = reinterpret_cast<_DataType_input*>(array1_in);
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

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

template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
void dpnp_floor_divide_c(void* array1_in, void* array2_in, void* result1, size_t size)
{
    cl::sycl::event event;
    _DataType_input1* array1 = reinterpret_cast<_DataType_input1*>(array1_in);
    _DataType_input2* array2 = reinterpret_cast<_DataType_input2*>(array2_in);
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

    if constexpr ((std::is_same<_DataType_input1, double>::value || std::is_same<_DataType_input1, float>::value) &&
                  std::is_same<_DataType_input2, _DataType_input1>::value)
    {
        event = oneapi::mkl::vm::div(DPNP_QUEUE, size, array1, array2, result);
        event.wait();
        event = oneapi::mkl::vm::floor(DPNP_QUEUE, size, result, result);
    }
    else
    {
        cl::sycl::range<1> gws(size);
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
            {
                _DataType_input1 input_elem1 = array1[i];
                _DataType_input2 input_elem2 = array2[i];
                double div = (double)input_elem1 / (double)input_elem2;
                result[i] = static_cast<_DataType_output>(cl::sycl::floor(div));
            }
        };

        auto kernel_func = [&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_floor_divide_c_kernel<_DataType_input1, _DataType_input2, _DataType_output>>(
                gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2>
class dpnp_modf_c_kernel;

template <typename _DataType_input, typename _DataType_output>
void dpnp_modf_c(void* array1_in, void* result1_out, void* result2_out, size_t size)
{
    cl::sycl::event event;
    _DataType_input* array1 = reinterpret_cast<_DataType_input*>(array1_in);
    _DataType_output* result1 = reinterpret_cast<_DataType_output*>(result1_out);
    _DataType_output* result2 = reinterpret_cast<_DataType_output*>(result2_out);

    if constexpr (std::is_same<_DataType_input, double>::value || std::is_same<_DataType_input, float>::value)
    {
        event = oneapi::mkl::vm::modf(DPNP_QUEUE, size, array1, result2, result1);
    }
    else
    {
        cl::sycl::range<1> gws(size);
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
            {
                _DataType_input input_elem1 = array1[i];
                result2[i] = cl::sycl::modf(double(input_elem1), &result1[i]);
            }
        };

        auto kernel_func = [&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_modf_c_kernel<_DataType_input, _DataType_output>>(gws,
                                                                                          kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();
}

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
class dpnp_multiply_c_kernel;

template <typename _DataType_output, typename _DataType_input1, typename _DataType_input2>
void dpnp_multiply_c(void* result_out,
                     const void* input1_in,
                     const size_t input1_size,
                     const size_t* input1_shape,
                     const size_t input1_shape_ndim,
                     const void* input2_in,
                     const size_t input2_size,
                     const size_t* input2_shape,
                     const size_t input2_shape_ndim,
                     const size_t* where)
{
    // avoid warning unused variable
    (void)input1_shape;
    (void)input1_shape_ndim;
    (void)input2_shape;
    (void)input2_shape_ndim;
    (void)where;

    if (!input1_size || !input2_size)
    {
        return;
    }

    const size_t result_size = (input2_size > input1_size) ? input2_size : input1_size;

    const _DataType_input1* input1_data = reinterpret_cast<const _DataType_input1*>(input1_in);
    const _DataType_input2* input2_data = reinterpret_cast<const _DataType_input2*>(input2_in);
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result_out);

    cl::sycl::range<1> gws(result_size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        size_t i = global_id[0]; /*for (size_t i = 0; i < result_size; ++i)*/
        {
            const _DataType_input1 input1_elem = (input1_size == 1) ? input1_data[0] : input1_data[i];
            const _DataType_input2 input2_elem = (input2_size == 1) ? input2_data[0] : input2_data[i];
            result[i] = input1_elem * input2_elem;
        }
    };
    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_multiply_c_kernel<_DataType_output, _DataType_input1,
                                                      _DataType_input2>>(gws, kernel_parallel_for_func);
    };

    cl::sycl::event event;

    if (input1_size == input2_size)
    {
        if constexpr ((std::is_same<_DataType_input1, double>::value ||
                       std::is_same<_DataType_input1, float>::value) &&
                      std::is_same<_DataType_input2, _DataType_input1>::value)
        {
            _DataType_input1* input1 = const_cast<_DataType_input1*>(input1_data);
            _DataType_input2* input2 = const_cast<_DataType_input2*>(input2_data);
            // https://docs.oneapi.com/versions/latest/onemkl/mul.html
            event = oneapi::mkl::vm::mul(DPNP_QUEUE, result_size, input1, input2, result);
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
}

template <typename _KernelNameSpecialization1, typename _KernelNameSpecialization2, typename _KernelNameSpecialization3>
class dpnp_remainder_c_kernel;

template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>
void dpnp_remainder_c(void* array1_in, void* array2_in, void* result1, size_t size)
{
    cl::sycl::event event;
    _DataType_input1* array1 = reinterpret_cast<_DataType_input1*>(array1_in);
    _DataType_input2* array2 = reinterpret_cast<_DataType_input2*>(array2_in);
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

    if constexpr ((std::is_same<_DataType_input1, double>::value || std::is_same<_DataType_input1, float>::value) &&
                  std::is_same<_DataType_input2, _DataType_input1>::value)
    {
        event = oneapi::mkl::vm::fmod(DPNP_QUEUE, size, array1, array2, result);
        event.wait();
        event = oneapi::mkl::vm::add(DPNP_QUEUE, size, result, array2, result);
        event.wait();
        event = oneapi::mkl::vm::fmod(DPNP_QUEUE, size, result, array2, result);
    }
    else
    {
        cl::sycl::range<1> gws(size);
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/
            {
                _DataType_input1 input_elem1 = array1[i];
                _DataType_input2 input_elem2 = array2[i];
                double fmod = cl::sycl::fmod((double)input_elem1, (double)input_elem2);
                double add = fmod + input_elem2;
                result[i] = cl::sycl::fmod(add, (double)input_elem2);
            }
        };

        auto kernel_func = [&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_remainder_c_kernel<_DataType_input1, _DataType_input2, _DataType_output>>(
                gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();
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

    cl::sycl::event event;
    _DataType_input1* array1 = reinterpret_cast<_DataType_input1*>(const_cast<void*>(array1_in));
    _DataType_input2* array2 = reinterpret_cast<_DataType_input2*>(const_cast<void*>(array2_in));
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

    if (array1_size < 2)
    {
        result[0] = 0;
        return;
    }

    if (array1_size == array2_size)
    {
        size_t cur_res_size = array1_size - 2;

        _DataType_output* cur_res =
            reinterpret_cast<_DataType_output*>(dpnp_memory_alloc_c((cur_res_size) * sizeof(_DataType_output)));

        cl::sycl::range<1> gws(cur_res_size);
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
            size_t i = global_id[0];
            {
                cur_res[i] = array1[i + 1] * (array2[i + 2] - array2[i]);
            }
        };

        auto kernel_func = [&](cl::sycl::handler& cgh) {
            cgh.parallel_for<class dpnp_trapz_c_kernel<_DataType_input1, _DataType_input2, _DataType_output>>(
                gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);

        event.wait();

        dpnp_sum_c<_DataType_output, _DataType_output>(result, cur_res, &cur_res_size, 1, NULL, 0, NULL, NULL);

        dpnp_memory_free_c(cur_res);

        result[0] += array1[0] * (array2[1] - array2[0]) +
                     array1[array1_size - 1] * (array2[array2_size - 1] - array2[array2_size - 2]);

        result[0] *= 0.5;
    }
    else
    {
        dpnp_sum_c<_DataType_output, _DataType_input1>(result, array1, &array1_size, 1, NULL, 0, NULL, NULL);

        result[0] -= (array1[0] + array1[array1_size - 1]) * 0.5;
        result[0] *= dx;
    }
}

void func_map_init_mathematical(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_elemwise_absolute_c<int>};
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_elemwise_absolute_c<long>};
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_elemwise_absolute_c<float>};
    fmap[DPNPFuncName::DPNP_FN_ABSOLUTE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_elemwise_absolute_c<double>};

    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_cross_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_cross_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_cross_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_cross_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_cross_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_cross_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_cross_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_cross_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_cross_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_cross_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cross_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_cross_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_cross_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_cross_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_cross_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_CROSS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cross_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_CUMPROD][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_cumprod_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_CUMPROD][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_cumprod_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_CUMPROD][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cumprod_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CUMPROD][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cumprod_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_CUMSUM][eft_INT][eft_INT] = {eft_LNG, (void*)dpnp_cumsum_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_CUMSUM][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_cumsum_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_CUMSUM][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_cumsum_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_CUMSUM][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_cumsum_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_floor_divide_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_floor_divide_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_INT][eft_FLT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_INT][eft_DBL] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_floor_divide_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_LNG][eft_LNG] = {eft_LNG,
                                                                  (void*)dpnp_floor_divide_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_LNG][eft_FLT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_LNG][eft_DBL] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_FLT][eft_INT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_FLT][eft_LNG] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_FLT][eft_FLT] = {eft_FLT,
                                                                  (void*)dpnp_floor_divide_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_FLT][eft_DBL] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_DBL][eft_INT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_DBL][eft_LNG] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_DBL][eft_FLT] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_FLOOR_DIVIDE][eft_DBL][eft_DBL] = {eft_DBL,
                                                                  (void*)dpnp_floor_divide_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_MODF][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_modf_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_MODF][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_modf_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_MODF][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_modf_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_MODF][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_modf_c<double, double>};

    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_multiply_c<bool, bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_INT] = {eft_INT, (void*)dpnp_multiply_c<int, bool, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_LNG] = {eft_LNG, (void*)dpnp_multiply_c<long, bool, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_FLT] = {eft_FLT, (void*)dpnp_multiply_c<float, bool, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_BLN][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, bool, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_BLN] = {eft_INT, (void*)dpnp_multiply_c<int, int, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_multiply_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_multiply_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_multiply_c<double, int, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_BLN] = {eft_LNG, (void*)dpnp_multiply_c<long, long, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_multiply_c<long, long, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_multiply_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_multiply_c<double, long, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_BLN] = {eft_FLT, (void*)dpnp_multiply_c<float, float, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_multiply_c<double, float, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_multiply_c<double, float, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_multiply_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_BLN] = {eft_DBL, (void*)dpnp_multiply_c<double, double, bool>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_multiply_c<double, double, int>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_multiply_c<double, double, long>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_multiply_c<double, double, float>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_multiply_c<double, double, double>};
    fmap[DPNPFuncName::DPNP_FN_MULTIPLY][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_multiply_c<std::complex<double>, std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_remainder_c<int, int, int>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_INT][eft_LNG] = {eft_LNG, (void*)dpnp_remainder_c<int, long, long>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_remainder_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_remainder_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_remainder_c<long, int, long>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_remainder_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_remainder_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_remainder_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_remainder_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_remainder_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_remainder_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_remainder_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_remainder_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_remainder_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_remainder_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_REMAINDER][eft_DBL][eft_DBL] = {eft_DBL,
                                                               (void*)dpnp_remainder_c<double, double, double>};

    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_INT][eft_INT] = {eft_DBL, (void*)dpnp_trapz_c<int, int, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_INT][eft_LNG] = {eft_DBL, (void*)dpnp_trapz_c<int, long, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_INT][eft_FLT] = {eft_DBL, (void*)dpnp_trapz_c<int, float, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_INT][eft_DBL] = {eft_DBL, (void*)dpnp_trapz_c<int, double, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_LNG][eft_INT] = {eft_DBL, (void*)dpnp_trapz_c<long, int, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_LNG][eft_LNG] = {eft_DBL, (void*)dpnp_trapz_c<long, long, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_LNG][eft_FLT] = {eft_DBL, (void*)dpnp_trapz_c<long, float, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_LNG][eft_DBL] = {eft_DBL, (void*)dpnp_trapz_c<long, double, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_FLT][eft_INT] = {eft_DBL, (void*)dpnp_trapz_c<float, int, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_FLT][eft_LNG] = {eft_DBL, (void*)dpnp_trapz_c<float, long, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_trapz_c<float, float, float>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_FLT][eft_DBL] = {eft_DBL, (void*)dpnp_trapz_c<float, double, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_trapz_c<double, int, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_trapz_c<double, long, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_trapz_c<double, float, double>};
    fmap[DPNPFuncName::DPNP_FN_TRAPZ][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_trapz_c<double, double, double>};

    return;
}
