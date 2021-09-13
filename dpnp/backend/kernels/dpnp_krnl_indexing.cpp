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
#include "dpnpc_memory_adapter.hpp"
#include "queue_sycl.hpp"

template <typename _DataType>
class dpnp_diag_indices_c_kernel;

template <typename _DataType>
void dpnp_diag_indices_c(void* result1, size_t size)
{
    dpnp_arange_c<_DataType>(0, 1, result1, size);
}

template <typename _DataType>
class dpnp_diagonal_c_kernel;

template <typename _DataType>
void dpnp_diagonal_c(
    void* array1_in, const size_t input1_size, void* result1, const size_t offset, size_t* shape, size_t* res_shape, const size_t res_ndim)
{
    const size_t res_size = std::accumulate(res_shape, res_shape + res_ndim, 1, std::multiplies<size_t>());
    if (!(res_size && input1_size))
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, input1_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(result1, res_size, true, true);
    _DataType* array_1 = input1_ptr.get_ptr();
    _DataType* result = result_ptr.get_ptr();

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
    const size_t result_size = std::accumulate(shape, shape + ndim, 1, std::multiplies<size_t>());
    if (!(result_size && array1_in))
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> result_ptr(array1_in, result_size, true, true);
    _DataType* array_1 = result_ptr.get_ptr();
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
void dpnp_nonzero_c(const void* in_array1, void* result1, const size_t result_size, const size_t* shape, const size_t ndim, const size_t j)
{
    if ((in_array1 == nullptr) || (result1 == nullptr))
    {
        return;
    }

    if (ndim == 0)
    {
        return;
    }

    const size_t input1_size = std::accumulate(shape, shape + ndim, 1, std::multiplies<size_t>());

    DPNPC_ptr_adapter<_DataType> input1_ptr(in_array1, input1_size, true);
    DPNPC_ptr_adapter<long> result_ptr(result1, result_size, true, true);
    const _DataType* arr = input1_ptr.get_ptr();
    long* result = result_ptr.get_ptr();


    size_t idx = 0;
    for (size_t i = 0; i < input1_size; ++i)
    {
        if (arr[i] != 0)
        {
            size_t ids[ndim];
            size_t ind1 = input1_size;
            size_t ind2 = i;
            for (size_t k = 0; k < ndim; ++k)
            {
                ind1 = ind1 / shape[k];
                ids[k] = ind2 / ind1;
                ind2 = ind2 % ind1;
            }

            result[idx] = ids[j];
            idx += 1;
        }
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

    if (!vals_size)
    {
        return;
    }

    DPNPC_ptr_adapter<_DataType> input1_ptr(vals_in, vals_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(arr_in, arr_size, true, true);
    _DataType* vals = input1_ptr.get_ptr();
    _DataType* arr = result_ptr.get_ptr();

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
    if ((array1_in == nullptr) || (ind_in == nullptr) || (v_in == nullptr))
    {
        return;
    }

    if (size_v == 0)
    {
        return;
    }

    DPNPC_ptr_adapter<size_t> input1_ptr(ind_in, size_ind, true);
    DPNPC_ptr_adapter<_DataType> input2_ptr(v_in, size_v, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(array1_in, size, true, true);
    size_t* ind = input1_ptr.get_ptr();
    _DataType* v = input2_ptr.get_ptr();
    _DataType* array_1 = result_ptr.get_ptr();

    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size_ind; ++j)
        {
            if (i == ind[j] || (i == (size + ind[j])))
            {
                array_1[i] = v[j % size_v];
            }
        }
    }

    return;
}

template <typename _DataType>
void dpnp_put_along_axis_c(void* arr_in,
                           long* indices_in,
                           void* values_in,
                           size_t axis,
                           const size_t* shape,
                           size_t ndim,
                           size_t size_indices,
                           size_t values_size)
{
    size_t res_ndim = ndim - 1;
    size_t res_shape[res_ndim];
    const size_t size_arr = std::accumulate(shape, shape + ndim, 1, std::multiplies<size_t>());

    DPNPC_ptr_adapter<size_t> input1_ptr(indices_in, size_indices, true);
    DPNPC_ptr_adapter<_DataType> input2_ptr(values_in, values_size, true);
    DPNPC_ptr_adapter<_DataType> result_ptr(arr_in, size_arr, true, true);
    size_t* indices = input1_ptr.get_ptr();
    _DataType* values = input2_ptr.get_ptr();
    _DataType* arr = result_ptr.get_ptr();

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
void dpnp_take_c(void* array1_in, const size_t array1_size, void* indices1, void* result1, size_t size)
{
    DPNPC_ptr_adapter<_DataType> input1_ptr(array1_in, array1_size, true);
    DPNPC_ptr_adapter<_IndecesType> input2_ptr(indices1, size);
    _DataType* array_1 = input1_ptr.get_ptr();
    _IndecesType* indices = input2_ptr.get_ptr();
    _DataType* result = reinterpret_cast<_DataType*>(result1);

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
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_diag_indices_c<int>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_diag_indices_c<long>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_diag_indices_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAG_INDICES][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_diag_indices_c<double>};

    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_diagonal_c<int>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_diagonal_c<long>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_diagonal_c<float>};
    fmap[DPNPFuncName::DPNP_FN_DIAGONAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_diagonal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_fill_diagonal_c<int>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_fill_diagonal_c<long>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_fill_diagonal_c<float>};
    fmap[DPNPFuncName::DPNP_FN_FILL_DIAGONAL][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_fill_diagonal_c<double>};

    fmap[DPNPFuncName::DPNP_FN_NONZERO][eft_INT][eft_INT] = {eft_INT, (void*)dpnp_nonzero_c<int>};
    fmap[DPNPFuncName::DPNP_FN_NONZERO][eft_LNG][eft_LNG] = {eft_LNG, (void*)dpnp_nonzero_c<long>};
    fmap[DPNPFuncName::DPNP_FN_NONZERO][eft_FLT][eft_FLT] = {eft_FLT, (void*)dpnp_nonzero_c<float>};
    fmap[DPNPFuncName::DPNP_FN_NONZERO][eft_DBL][eft_DBL] = {eft_DBL, (void*)dpnp_nonzero_c<double>};

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
