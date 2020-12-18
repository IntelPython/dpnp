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

#include <backend_iface.hpp>
#include "backend_fptr.hpp"
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

template <typename _KernelNameSpecialization>
class custom_elemwise_absolute_c_kernel;

template <typename _DataType>
void custom_elemwise_absolute_c(void* array1_in, void* result1, size_t size)
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
            cgh.parallel_for<class custom_elemwise_absolute_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
        };

        event = DPNP_QUEUE.submit(kernel_func);
    }

    event.wait();
}

template void custom_elemwise_absolute_c<double>(void* array1_in, void* result1, size_t size);
template void custom_elemwise_absolute_c<float>(void* array1_in, void* result1, size_t size);
template void custom_elemwise_absolute_c<long>(void* array1_in, void* result1, size_t size);
template void custom_elemwise_absolute_c<int>(void* array1_in, void* result1, size_t size);

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

void func_map_init_mathematical(func_map_t& fmap)
{
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

    return;
}
