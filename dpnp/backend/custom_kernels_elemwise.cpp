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

#include <backend_iface.hpp>
#include "backend_utils.hpp"
#include "queue_sycl.hpp"

#define MACRO_CUSTOM_1ARG_2TYPES_OP(__name__, __operation__)                                                           \
    template <typename _KernelNameSpecialization>                                                                      \
    class custom_elemwise_##__name__##_c_kernel;                                                                       \
                                                                                                                       \
    template <typename _DataType_input, typename _DataType_output>                                                     \
    void custom_elemwise_##__name__##_c(void* array1_in, void* result1, size_t size)                                   \
    {                                                                                                                  \
        cl::sycl::event event;                                                                                         \
        _DataType_input* array1 = reinterpret_cast<_DataType_input*>(array1_in);                                       \
        _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);                                       \
                                                                                                                       \
        cl::sycl::range<1> gws(size);                                                                                  \
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                               \
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/                                             \
            {                                                                                                          \
                _DataType_output input_elem = array1[i];                                                               \
                result[i] = __operation__;                                                                             \
            }                                                                                                          \
        };                                                                                                             \
                                                                                                                       \
        auto kernel_func = [&](cl::sycl::handler& cgh) {                                                               \
            cgh.parallel_for<class custom_elemwise_##__name__##_c_kernel<_DataType_input>>(gws,                        \
                                                                                           kernel_parallel_for_func);  \
        };                                                                                                             \
                                                                                                                       \
        event = DPNP_QUEUE.submit(kernel_func);                                                                        \
                                                                                                                       \
        event.wait();                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    template void custom_elemwise_##__name__##_c<double, double>(void* array1_in, void* result1, size_t size);         \
    template void custom_elemwise_##__name__##_c<float, float>(void* array1_in, void* result1, size_t size);           \
    template void custom_elemwise_##__name__##_c<long, double>(void* array1_in, void* result1, size_t size);           \
    template void custom_elemwise_##__name__##_c<int, double>(void* array1_in, void* result1, size_t size);

#include <custom_1arg_2type_tbl.hpp>

/* ========================================================================== */
#define MACRO_CUSTOM_1ARG_1TYPE_OP(__name__, __operation__)                                                            \
    template <typename _KernelNameSpecialization>                                                                      \
    class custom_elemwise_##__name__##_c_kernel;                                                                       \
                                                                                                                       \
    template <typename _DataType>                                                                                      \
    void custom_elemwise_##__name__##_c(void* array1_in, void* result1, size_t size)                                   \
    {                                                                                                                  \
        cl::sycl::event event;                                                                                         \
        _DataType* array1 = reinterpret_cast<_DataType*>(array1_in);                                                   \
        _DataType* result = reinterpret_cast<_DataType*>(result1);                                                     \
                                                                                                                       \
        cl::sycl::range<1> gws(size);                                                                                  \
        auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                               \
            size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/                                             \
            {                                                                                                          \
                _DataType input_elem = array1[i];                                                                      \
                result[i] = __operation__;                                                                             \
            }                                                                                                          \
        };                                                                                                             \
                                                                                                                       \
        auto kernel_func = [&](cl::sycl::handler& cgh) {                                                               \
            cgh.parallel_for<class custom_elemwise_##__name__##_c_kernel<_DataType>>(gws, kernel_parallel_for_func);   \
        };                                                                                                             \
                                                                                                                       \
        event = DPNP_QUEUE.submit(kernel_func);                                                                        \
                                                                                                                       \
        event.wait();                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    template void custom_elemwise_##__name__##_c<double>(void* array1_in, void* result1, size_t size);                 \
    template void custom_elemwise_##__name__##_c<float>(void* array1_in, void* result1, size_t size);                  \
    template void custom_elemwise_##__name__##_c<long>(void* array1_in, void* result1, size_t size);                   \
    template void custom_elemwise_##__name__##_c<int>(void* array1_in, void* result1, size_t size);

#include <custom_1arg_1type_tbl.hpp>

/* ========================================================================== */
#define MACRO_CUSTOM_2ARG_3TYPES_OP(__name__, __operation__, __mkl_operation__)                                        \
    template <typename _KernelNameSpecialization1,                                                                     \
              typename _KernelNameSpecialization2,                                                                     \
              typename _KernelNameSpecialization3>                                                                     \
    class custom_elemwise_##__name__##_c_kernel;                                                                       \
                                                                                                                       \
    template <typename _DataType_input1, typename _DataType_input2, typename _DataType_output>                         \
    void custom_elemwise_##__name__##_c(void* array1_in, void* array2_in, void* result1, size_t size)                  \
    {                                                                                                                  \
        cl::sycl::event event;                                                                                         \
        _DataType_input1* array1 = reinterpret_cast<_DataType_input1*>(array1_in);                                     \
        _DataType_input2* array2 = reinterpret_cast<_DataType_input2*>(array2_in);                                     \
        _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);                                       \
                                                                                                                       \
        if constexpr ((std::is_same<_DataType_input1, double>::value || std::is_same<_DataType_input1, float>::value)  \
                    && std::is_same<_DataType_input2, _DataType_input1>::value)                                        \
        {                                                                                                              \
            event = __mkl_operation__(DPNP_QUEUE, size, array1, array2, result);                                       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            cl::sycl::range<1> gws(size);                                                                              \
            auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {                                           \
                size_t i = global_id[0]; /*for (size_t i = 0; i < size; ++i)*/                                         \
                {                                                                                                      \
                    _DataType_output input_elem1 = array1[i];                                                          \
                    _DataType_output input_elem2 = array2[i];                                                          \
                    result[i] = __operation__;                                                                         \
                }                                                                                                      \
            };                                                                                                         \
                                                                                                                       \
            auto kernel_func = [&](cl::sycl::handler& cgh) {                                                           \
                cgh.parallel_for<class custom_elemwise_##__name__##_c_kernel<_DataType_input1, _DataType_input2,       \
                                                                             _DataType_output>>(                       \
                                                                                 gws, kernel_parallel_for_func);       \
            };                                                                                                         \
                                                                                                                       \
            event = DPNP_QUEUE.submit(kernel_func);                                                                    \
        }                                                                                                              \
                                                                                                                       \
        event.wait();                                                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    /* double - XXX */                                                                                                 \
    template void custom_elemwise_##__name__##_c<double, double, double>(                                              \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<double, float, double>(                                               \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<double, long, double>(                                                \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<double, int, double>(                                                 \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    /* float - XXX */                                                                                                  \
    template void custom_elemwise_##__name__##_c<float, float, float>(                                                 \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<float, double, double>(                                               \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<float, long, double>(                                                 \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<float, int, double>(                                                  \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    /* long - XXX */                                                                                                   \
    template void custom_elemwise_##__name__##_c<long, long, long>(                                                    \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<long, int, long>(                                                     \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<long, long, double>(                                                  \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<long, int, double>(                                                   \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<long, float, double>(                                                 \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<long, double, double>(                                                \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    /* int - XXX */                                                                                                    \
    template void custom_elemwise_##__name__##_c<int, int, int>(                                                       \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<int, long, long>(                                                     \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<int, int, double>(                                                    \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<int, long, double>(                                                   \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<int, float, double>(                                                  \
        void* array1_in, void* array2_in, void* result1, size_t size);                                                 \
    template void custom_elemwise_##__name__##_c<int, double, double>(                                                 \
        void* array1_in, void* array2_in, void* result1, size_t size);

#include <custom_2arg_3type_tbl.hpp>

/* ========================================================================== */
#if 0 // Switch between SYCL and OpenCL kernels

#include <map>
#include <typeindex>

static std::map<std::type_index, std::string> types_map = {
    {typeid(double), "double"}, {typeid(float), "float"}, {typeid(long), "long"}, {typeid(int), "int"}};

static const char* opencl_elemwise_naive =
    "__kernel void elemwise_naive(__global __KERNEL_TYPE_IN__* array_1,            \n"
    "                             __global __KERNEL_TYPE_OUT__* result,            \n"
    "                             unsigned long size)                              \n"
    "{                                                                             \n"
    "    size_t i = get_global_id(0); //for (size_t i = 0; i < size; ++i)          \n"
    "    {                                                                         \n"
    "        __KERNEL_TYPE_OUT__ aux_val = array_1[i];                             \n"
    "        result[i] = cos(aux_val);                                             \n"
    "    }                                                                         \n"
    "}                                                                             \n";

template <typename _DataType_input, typename _DataType_output>
void custom_elemwise_cos_c(void* array1_in, void* result1, size_t size)
{
    _DataType_input* array_1 = reinterpret_cast<_DataType_input*>(array1_in);
    _DataType_output* result = reinterpret_cast<_DataType_output*>(result1);

    std::string compile_time_options("-cl-std=CL1.2");
    compile_time_options += " -D__KERNEL_TYPE_IN__=" + types_map.at(typeid(_DataType_input)) +
                            " -D__KERNEL_TYPE_OUT__=" + types_map.at(typeid(_DataType_output));

    cl::sycl::program program_src(DPNP_QUEUE.get_context());
    program_src.build_with_source(opencl_elemwise_naive, compile_time_options);

    cl::sycl::range<1> kernel_work_ids(size); // dimension "i"
    DPNP_QUEUE.submit([&](cl::sycl::handler& cgh) {
        cgh.set_args(array_1, result, size);
        cgh.parallel_for(kernel_work_ids, program_src.get_kernel("elemwise_naive"));
    });

    DPNP_QUEUE.wait();
}

template void custom_elemwise_cos_c<double, double>(void* array1_in, void* result1, size_t size);
template void custom_elemwise_cos_c<float, float>(void* array1_in, void* result1, size_t size);
template void custom_elemwise_cos_c<long, double>(void* array1_in, void* result1, size_t size);
template void custom_elemwise_cos_c<int, double>(void* array1_in, void* result1, size_t size);

#endif // Switch between SYCL and OpenCL kernels
