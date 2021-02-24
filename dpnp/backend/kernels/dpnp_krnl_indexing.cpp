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

#include <iostream>
#include <list>
#include <vector>

#include <dpnp_iface.hpp>
#include "dpnp_fptr.hpp"
#include "queue_sycl.hpp"

template <typename _DataType>
class dpnp_diagonal_c_kernel;

template <typename _DataType>
void dpnp_diagonal_c(
    void* array1_in, void* result1, const size_t offset, size_t* shape, size_t* res_shape, const size_t res_ndim)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);

    size_t res_size = 1;
    for (size_t i = 0; i < res_ndim; ++i)
    {
        res_size *= res_shape[i];
    }

    if (res_ndim <= 1)
    {
        for (size_t i = 0; i < res_shape[res_ndim - 1]; ++i)
        {
            result[i] = array_1[i * shape[res_ndim] + i + offset];
        }
    }
    else
    {
        std::map<size_t, std::vector<size_t>> xyz;
        for (size_t i = 0; i < res_shape[0]; i++)
        {
            xyz[i] = {i};
        }

        size_t index = 1;
        while (index < res_ndim - 1)
        {
            size_t shape_element = res_shape[index];
            std::map<size_t, std::vector<size_t>> new_shape_array;
            size_t ind = 0;
            for (size_t i = 0; i < shape_element; i++)
            {
                for (size_t j = 0; j < xyz.size(); j++)
                {
                    std::vector<size_t> new_shape;
                    std::vector<size_t> list_ind = xyz[j];
                    for (size_t k = 0; k < list_ind.size(); k++)
                    {
                        new_shape.push_back(list_ind.at(k));
                    }
                    new_shape.push_back(i);
                    new_shape_array[ind] = new_shape;
                    ind += 1;
                }
            }
            size_t len_new_shape_array = new_shape_array.size() * (index + 1);

            for (size_t k = 0; k < len_new_shape_array; k++)
            {
                xyz[k] = new_shape_array[k];
            }
            index += 1;
        }

        for (size_t i = 0; i < res_shape[res_ndim - 1]; i++)
        {
            for (size_t j = 0; j < xyz.size(); j++)
            {
                std::vector<size_t> ind_list = xyz[j];
                if (ind_list.size() == 0)
                {
                    continue;
                }
                else
                {
                    size_t ind_input_size = ind_list.size() + 2;
                    size_t ind_input_[ind_input_size];
                    ind_input_[0] = i;
                    ind_input_[1] = i + offset;
                    size_t ind_output_size = ind_list.size() + 1;
                    size_t ind_output_[ind_output_size];
                    for (size_t k = 0; k < ind_list.size(); k++)
                    {
                        ind_input_[k + 2] = ind_list.at(k);
                        ind_output_[k] = ind_list.at(k);
                    }
                    ind_output_[ind_list.size()] = i;

                    size_t ind_output = 0;
                    size_t n = 1;
                    for (size_t k = 0; k < ind_output_size; k++)
                    {
                        size_t ind = ind_output_size - 1 - k;
                        ind_output += n * ind_output_[ind];
                        n *= res_shape[ind];
                    }

                    size_t ind_input = 0;
                    size_t m = 1;
                    for (size_t k = 0; k < ind_input_size; k++)
                    {
                        size_t ind = ind_input_size - 1 - k;
                        ind_input += m * ind_input_[ind];
                        m *= shape[ind];
                    }

                    result[ind_output] = array_1[ind_input];
                }
            }
        }
    }

    return;
}

template <typename _DataType>
void dpnp_fill_diagonal_c(void* array1_in, void* val_in, size_t* shape, const size_t ndim)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* val_arr = reinterpret_cast<_DataType*>(val_in);

    size_t min_shape = shape[0];
    for (size_t i = 0; i < ndim; ++i)
    {
        if (shape[i] < min_shape)
        {
            min_shape = shape[i];
        }
    }

    _DataType val = val_arr[0];

    for (size_t i = 0; i < min_shape; ++i)
    {
        size_t ind = 0;
        size_t n = 1;
        for (size_t k = 0; k < ndim; k++)
        {
            size_t ind_ = ndim - 1 - k;
            ind += n * i;
            n *= shape[ind_];
        }
        array_1[ind] = val;
    }
    return;
}

template <typename _DataType>
void dpnp_place_c(void* arr_in, long* mask_in, void* vals_in, const size_t arr_size, const size_t vals_size)
{
    if (!arr_size)
    {
        return;
    }
    _DataType* arr = reinterpret_cast<_DataType*>(arr_in);

    if (!vals_size)
    {
        return;
    }
    _DataType* vals = reinterpret_cast<_DataType*>(vals_in);

    size_t counter = 0;
    for (size_t i = 0; i < arr_size; ++i)
    {
        if (mask_in[i])
        {
            arr[i] = vals[counter % vals_size];
            counter += 1;
        }
    }
    return;
}

template <typename _DataType, typename _IndecesType, typename _ValueType>
void dpnp_put_c(
    void* array1_in, void* ind_in, void* v_in, const size_t size, const size_t size_ind, const size_t size_v)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    size_t* ind = reinterpret_cast<size_t*>(ind_in);
    _DataType* v = reinterpret_cast<_DataType*>(v_in);

    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size_ind; ++j)
        {
            if (i == ind[j])
            {
                array_1[i] = v[j % size_v];
                break;
            }
        }
    }
    return;
}

template <typename _DataType>
void dpnp_put_along_axis_c(void* arr_in, long* indices_in, void* values_in, size_t axis, const size_t* shape, size_t ndim, size_t size_indices, size_t values_size)
{
    _DataType* arr = reinterpret_cast<_DataType*>(arr_in);
    size_t* indices = reinterpret_cast<size_t*>(indices_in);
    _DataType* values = reinterpret_cast<_DataType*>(values_in);

    size_t res_ndim = ndim - 1;
    size_t res_shape[res_ndim];

    size_t size_arr = 1;
    for (size_t i = 0; i < ndim; ++i)
    {
        size_arr *= shape[i];
    }
    
    if (axis != res_ndim)
    {
        int ind = 0;
        for (size_t i = 0; i < ndim; i++)
        {
            if (axis != i)
            {
                res_shape[ind] = shape[i];
                ind++;
            }
        }

        size_t prod = 1;
        for (size_t i = 0; i < res_ndim; ++i)
        {
            if (res_shape[i] != 0)
            {
                prod *= res_shape[i];
            }
        }

        size_t ind_array[prod];
        bool bool_ind_array[prod];
        for (size_t i = 0; i < prod; ++i)
        {
            bool_ind_array[i] = true;
        }
        size_t arr_shape_offsets[ndim];
        size_t acc = 1;
        for (size_t i = ndim - 1; i > 0; --i)
        {
            arr_shape_offsets[i] = acc;
            acc *= shape[i];
        }
        arr_shape_offsets[0] = acc;

        size_t output_shape_offsets[res_ndim];
        acc = 1;
        if (res_ndim > 0)
        {
            for (size_t i = res_ndim - 1; i > 0; --i)
            {
                output_shape_offsets[i] = acc;
                acc *= res_shape[i];
            }
        }
        output_shape_offsets[0] = acc;

        size_t size_result = 1;
        for (size_t i = 0; i < res_ndim; ++i)
        {
            size_result *= res_shape[i];
        }

        //init result array
        for (size_t result_idx = 0; result_idx < size_result; ++result_idx)
        {
            size_t xyz[res_ndim];
            size_t remainder = result_idx;
            for (size_t i = 0; i < res_ndim; ++i)
            {
                xyz[i] = remainder / output_shape_offsets[i];
                remainder = remainder - xyz[i] * output_shape_offsets[i];
            }

            size_t source_axis[ndim];
            size_t result_axis_idx = 0;
            for (size_t idx = 0; idx < ndim; ++idx)
            {
                bool found = false;
                if (axis == idx)
                {
                    found = true;
                }
                if (found)
                {
                    source_axis[idx] = 0;
                }
                else
                {
                    source_axis[idx] = xyz[result_axis_idx];
                    result_axis_idx++;
                }
            }

            size_t source_idx = 0;
            for (size_t i = 0; i < ndim; ++i)
            {
                source_idx += arr_shape_offsets[i] * source_axis[i];
            }
        }

        for (size_t source_idx = 0; source_idx < size_arr; ++source_idx)
        {
            // reconstruct x,y,z from linear source_idx
            size_t xyz[ndim];
            size_t remainder = source_idx;
            for (size_t i = 0; i < ndim; ++i)
            {
                xyz[i] = remainder / arr_shape_offsets[i];
                remainder = remainder - xyz[i] * arr_shape_offsets[i];
            }

            // extract result axis
            size_t result_axis[res_ndim];
            size_t result_idx = 0;
            for (size_t idx = 0; idx < ndim; ++idx)
            {
                // try to find current idx in axis array
                bool found = false;
                if (axis == idx)
                {
                    found = true;
                }
                if (!found)
                {
                    result_axis[result_idx] = xyz[idx];
                    result_idx++;
                }
            }

            // Construct result offset
            size_t result_offset = 0;
            for (size_t i = 0; i < res_ndim; ++i)
            {
                result_offset += output_shape_offsets[i] * result_axis[i];
            }

            if (bool_ind_array[result_offset])
            {
                ind_array[result_offset] = 0;
                bool_ind_array[result_offset] = false;
            }
            else
            {
                ind_array[result_offset] += 1;
            }

            if ((ind_array[result_offset] % size_indices) == indices[result_offset % size_indices])
            {
                arr[source_idx] = values[source_idx % values_size];
            }
        }
    }
    else
    {
        for (size_t i = 0; i < size_arr; ++i)
        {
            size_t ind = size_indices * (i / size_indices) + indices[i % size_indices];
            arr[ind] = values[i % values_size];
        }
    }
    return;
}

template <typename _DataType, typename _IndecesType>
class dpnp_take_c_kernel;

template <typename _DataType, typename _IndecesType>
void dpnp_take_c(void* array1_in, void* indices1, void* result1, size_t size)
{
    _DataType* array_1 = reinterpret_cast<_DataType*>(array1_in);
    _DataType* result = reinterpret_cast<_DataType*>(result1);
    _IndecesType* indices = reinterpret_cast<_IndecesType*>(indices1);

    cl::sycl::range<1> gws(size);
    auto kernel_parallel_for_func = [=](cl::sycl::id<1> global_id) {
        const size_t idx = global_id[0];
        result[idx] = array_1[indices[idx]];
    };

    auto kernel_func = [&](cl::sycl::handler& cgh) {
        cgh.parallel_for<class dpnp_take_c_kernel<_DataType, _IndecesType>>(gws, kernel_parallel_for_func);
    };

    cl::sycl::event event = DPNP_QUEUE.submit(kernel_func);

    event.wait();

    return;
}

void func_map_init_indexing_func(func_map_t& fmap)
{
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_diagonal_c<int>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_diagonal_c<long>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_diagonal_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_diagonal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_fill_diagonal_c<int>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_fill_diagonal_c<long>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_fill_diagonal_c<float>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_fill_diagonal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_PLACE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_place_c<int>};
    fmap[DPNPFuncName::DPNP_FN_PLACE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_place_c<long>};
    fmap[DPNPFuncName::DPNP_FN_PLACE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_place_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PLACE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_place_c<double>};

    fmap[DPNPFuncName::DPNP_FN_PUT][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_put_c<int, long, int>};
    fmap[DPNPFuncName::DPNP_FN_PUT][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_put_c<long, long, long>};
    fmap[DPNPFuncName::DPNP_FN_PUT][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_put_c<float, long, float>};
    fmap[DPNPFuncName::DPNP_FN_PUT][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_put_c<double, long, double>};

    fmap[DPNPFuncName::DPNP_FN_PUT_ALONG_AXIS][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_put_along_axis_c<int>};
    fmap[DPNPFuncName::DPNP_FN_PUT_ALONG_AXIS][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_put_along_axis_c<long>};
    fmap[DPNPFuncName::DPNP_FN_PUT_ALONG_AXIS][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_put_along_axis_c<float>};
    fmap[DPNPFuncName::DPNP_FN_PUT_ALONG_AXIS][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_put_along_axis_c<double>};

    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_BLN][eft_BLN] = {eft_BLN, (void*)dpnp_take_c<bool, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_take_c<int, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_take_c<long, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_take_c<float, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_take_c<double, long>};
    fmap[DPNPFuncName::DPNP_FN_TAKE][eft_C128][eft_C128] = {eft_C128, (void*)dpnp_take_c<std::complex<double>, long>};

    return;
}
