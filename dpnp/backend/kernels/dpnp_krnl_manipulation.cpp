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
                               const shape_elem_type* input_shape,
                               const shape_elem_type* result_shape,
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

    shape_elem_type* input_offset_shape = reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));
    get_shape_offsets_inkernel(input_shape, ndim, input_offset_shape);

    shape_elem_type* temp_result_offset_shape = reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));
    get_shape_offsets_inkernel(result_shape, ndim, temp_result_offset_shape);

    shape_elem_type* result_offset_shape = reinterpret_cast<shape_elem_type*>(dpnp_memory_alloc_c(ndim * sizeof(shape_elem_type)));
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
    fmap[DPNPFuncName::DPNP_FN_REPEAT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_repeat_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_REPEAT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_repeat_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_REPEAT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_repeat_c<float>};
    fmap[DPNPFuncName::DPNP_FN_REPEAT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_repeat_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TRANSPOSE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_elemwise_transpose_c<int32_t>};
    fmap[DPNPFuncName::DPNP_FN_TRANSPOSE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_elemwise_transpose_c<int64_t>};
    fmap[DPNPFuncName::DPNP_FN_TRANSPOSE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_elemwise_transpose_c<float>};
    fmap[DPNPFuncName::DPNP_FN_TRANSPOSE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_elemwise_transpose_c<double>};

    return;
}
