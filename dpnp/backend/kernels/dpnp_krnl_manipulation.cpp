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
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _DataType_dst, typename _DataType_src>
void dpnp_copyto_c(void* destination, void* source, const size_t size)
{
    __dpnp_copyto_c<_DataType_src, _DataType_dst>(source, destination, size);
}

template <typename _DataType>
class dpnp_repeat_c_kernel;

template <typename _DataType>
void dpnp_repeat_c(const void* array1_in, void* result1, const size_t repeats, const size_t size)
{
    if (!array1_in || !result1)
    {
        return;
    }

    if (!size || !repeats)
    {
        return;
    }

    cl::sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size);
    const _DataType* array_in = input1_ptr.get_ptr();
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    cl::sycl::range<2> gws(size, repeats);
    auto kernel_parallel_for_func = [=](cl::sycl::id<2> global_id) {
        size_t idx1 = global_id[0];
        size_t idx2 = global_id[1];
        result[(idx1 * repeats) + idx2] = array_in[idx1];
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_repeat_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();
}

template <typename _KernelNameSpecialization>
class dpnp_elemwise_transpose_c_kernel;

template <typename _DataType>
void dpnp_elemwise_transpose_c(void* array1_in,
                               const size_t* input_shape,
                               const size_t* result_shape,
                               const size_t* permute_axes,
                               size_t ndim,
                               void* result1,
                               size_t size)
{
    if (!size)
    {
        return;
    }

    cl::sycl::event event;
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, size);
    _DataType* array1 = input1_ptr.get_ptr();
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t* input_offset_shape = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(ndim * sizeof(long)));
    get_shape_offsets_inkernel(input_shape, ndim, input_offset_shape);

    size_t* temp_result_offset_shape = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(ndim * sizeof(long)));
    get_shape_offsets_inkernel(result_shape, ndim, temp_result_offset_shape);

    size_t* result_offset_shape = reinterpret_cast<size_t*>(dpnp_memory_alloc_c(ndim * sizeof(long)));
    for (size_t axis = 0; axis < ndim; ++axis)
    {
        result_offset_shape[permute_axes[axis]] = temp_result_offset_shape[axis];
    }

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        const size_t idx = global_id[0];

        size_t output_index = 0;
        size_t reminder = idx;
        for (size_t axis = 0; axis < ndim; ++axis)
        {
            /* reconstruct [x][y][z] from given linear idx */
            size_t xyz_id = reminder / input_offset_shape[axis];
            reminder = reminder % input_offset_shape[axis];

            /* calculate destination index based on reconstructed [x][y][z] */
            output_index += (xyz_id * result_offset_shape[axis]);
        }

        result[output_index] = array1[idx];
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_elemwise_transpose_c_kernel<_DataType>>(gws, kernel_parallel_for_func);
    };

    event = DPNP_QUEUE.submit(kernel_func);

    event.wait();

    dpnp_memory_free_c(input_offset_shape);
    dpnp_memory_free_c(temp_result_offset_shape);
    dpnp_memory_free_c(result_offset_shape);
}

void func_map_init_manipulation(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_copyto_c<bool, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_INT] = {eft_BLN, (void*)dpnp_copyto_c<bool, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_LNG] = {eft_BLN, (void*)dpnp_copyto_c<bool, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_FLT] = {eft_BLN, (void*)dpnp_copyto_c<bool, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_BLN][eft_DBL] = {eft_BLN, (void*)dpnp_copyto_c<bool, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_BLN] = {eft_INT, (void*)dpnp_copyto_c<int, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_copyto_c<int, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_LNG] = {eft_INT, (void*)dpnp_copyto_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_FLT] = {eft_INT, (void*)dpnp_copyto_c<int, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_INT][eft_DBL] = {eft_INT, (void*)dpnp_copyto_c<int, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_BLN] = {eft_LNG, (void*)dpnp_copyto_c<long, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_INT] = {eft_LNG, (void*)dpnp_copyto_c<long, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_copyto_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_FLT] = {eft_LNG, (void*)dpnp_copyto_c<long, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_LNG][eft_DBL] = {eft_LNG, (void*)dpnp_copyto_c<long, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_BLN] = {eft_FLT, (void*)dpnp_copyto_c<float, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_INT] = {eft_FLT, (void*)dpnp_copyto_c<float, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_LNG] = {eft_FLT, (void*)dpnp_copyto_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_copyto_c<float, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_FLT][eft_DBL] = {eft_FLT, (void*)dpnp_copyto_c<float, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_BLN] = {eft_DBL, (void*)dpnp_copyto_c<double, bool>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_INT] = {eft_DBL, (void*)dpnp_copyto_c<double, int>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_LNG] = {eft_DBL, (void*)dpnp_copyto_c<double, long>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_FLT] = {eft_DBL, (void*)dpnp_copyto_c<double, float>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_copyto_c<double, double>};
    fmap[DPNPFuncName::DPNP_FN_COPYTO][eft_C128][eft_C128] = {
        eft_C128, (void*)dpnp_copyto_c<std::complex<double>, std::complex<double>>};

    fmap[DPNPFuncName::DPNP_FN_REPEAT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_repeat_c<int>};
    fmap[DPNPFuncName::DPNP_FN_REPEAT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_repeat_c<long>};
    fmap[DPNPFuncName::DPNP_FN_REPEAT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_repeat_c<float>};
    fmap[DPNPFuncName::DPNP_FN_REPEAT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_repeat_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TRANSPOSE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_elemwise_transpose_c<int>};
    fmap[DPNPFuncName::DPNP_FN_TRANSPOSE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_elemwise_transpose_c<long>};
    fmap[DPNPFuncName::DPNP_FN_TRANSPOSE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_elemwise_transpose_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRANSPOSE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_elemwise_transpose_c<double>};

    return;
}
