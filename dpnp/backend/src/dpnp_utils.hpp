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

#pragma once
#ifndef BACKEND_UTILS_H // Cython compatibility
#define BACKEND_UTILS_H

#include <algorithm>
#include <iostream>
#include <iterator>

#include <dpnp_iface_fptr.hpp>

/**
 * @defgroup BACKEND_UTILS Backend C++ library utilities
 * @{
 * This section describes utilities used in Backend API.
 * @}
 */

/**
 * @ingroup BACKEND_UTILS
 * @brief Shape offset calculation used in kernels
 *
 * Calculates offsets of the array with given shape
 * for example:
 *   input_array_shape[3, 4, 5]
 *   offsets should be [20, 5, 1]
 *
 * @param [in]  shape       array with input shape.
 * @param [in]  shape_size  array size for @ref shape parameter.
 * @param [out] offsets     Result array with @ref shape_size size.
 */
template <typename _DataType>
void get_shape_offsets_inkernel(const _DataType* shape, size_t shape_size, _DataType* offsets)
{
    size_t dim_prod_input = 1;
    for (size_t i = 0; i < shape_size; ++i)
    {
        long i_reverse = shape_size - 1 - i;
        offsets[i_reverse] = dim_prod_input;
        dim_prod_input *= shape[i_reverse];
    }

    return;
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Calculate ids for all given axes from linear index
 *
 * Calculates ids of the array with given shape. This is reverse operation of @ref get_id_by_xyz_inkernel
 * for example:
 *   input_array_shape_offsets[20, 5, 1]
 *   global_id == 5
 *   xyz array ids should be [0, 1, 0]
 *
 * @param [in]  global_id     linear index id of the element in multy-D array.
 * @param [in]  offsets       array with input offsets.
 * @param [in]  offsets_size  array size for @ref offsets parameter.
 * @param [out] xyz           Result array with @ref offsets_size size.
 */
template <typename _DataType>
void get_xyz_by_id_inkernel(size_t global_id, const _DataType* offsets, size_t offsets_size, _DataType* xyz)
{
    long reminder = global_id;
    for (size_t axis = 0; axis < offsets_size; ++axis)
    {
        /* reconstruct [x][y][z] from given linear idx */
        const _DataType axis_val = offsets[axis];
        _DataType xyz_id = reminder / axis_val;
        reminder = reminder % axis_val;
        xyz[axis] = xyz_id;
    }

    return;
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Calculate linear index from ids array
 *
 * Calculates linear index from ids array by given offsets. This is reverse operation of @ref get_xyz_by_id_inkernel
 * for example:
 *   xyz array ids should be [0, 1, 0]
 *   input_array_shape_offsets[20, 5, 1]
 *   global_id == 5
 *
 * @param [in] xyz       array with array indexes.
 * @param [in] xyz_size  array size for @ref xyz parameter.
 * @param [in] offsets   array with input offsets.
 * @return               linear index id of the element in multy-D array.
 */
template <typename _DataType>
size_t get_id_by_xyz_inkernel(const _DataType* xyz, size_t xyz_size, const _DataType* offsets)
{
    size_t global_id = 0;

    /* calculate linear index based on reconstructed [x][y][z] */
    for (size_t it = 0; it < xyz_size; ++it)
    {
        global_id += (xyz[it] * offsets[it]);
    }

    return global_id;
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Check input shape is broadcastable to output one.
 *
 * @param [in] input_shape        Input shape.
 * @param [in] output_shape       Output shape.
 *
 * @return                        Input shape is broadcastable to output one or not.
 */
static inline bool
    broadcastable(const std::vector<size_t>& input_shape, const std::vector<size_t>& output_shape)
{
    if (input_shape.size() > output_shape.size())
    {
        return false;
    }

    std::vector<size_t>::const_reverse_iterator irit = input_shape.rbegin();
    std::vector<size_t>::const_reverse_iterator orit = output_shape.rbegin();
    for (; irit != input_shape.rend(); ++irit, ++orit)
    {
        if (*irit != 1 && *irit != *orit)
        {
            return false;
        }
    }

    return true;
}

static inline bool
    broadcastable(const size_t* input_shape, const size_t input_shape_size, const std::vector<size_t>& output_shape)
{
    const std::vector<size_t> input_shape_vec(input_shape, input_shape + input_shape_size);
    return broadcastable(input_shape_vec, output_shape);
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Get common shape based on input shapes.
 * 
 * Example:
 *   Input1 shape A[8, 1, 6, 1]
 *   Input2 shape B[7, 1, 5]
 *   Output shape will be C[8, 7, 6, 5]
 *
 * @param [in] input1_shape        Input1 shape.
 * @param [in] input1_shape_size   Input1 shape size.
 * @param [in] input2_shape        Input2 shape.
 * @param [in] input2_shape_size   Input2 shape size.
 *
 * @exception std::domain_error    Input shapes are not broadcastable.
 * @return                         Common shape.
 */
static inline std::vector<size_t>
    get_common_shape(const size_t* input1_shape, const size_t input1_shape_size,
                     const size_t* input2_shape, const size_t input2_shape_size)
{
    const size_t result_shape_size = (input2_shape_size > input1_shape_size) ? input2_shape_size : input1_shape_size;
    std::vector<size_t> result_shape;
    result_shape.reserve(result_shape_size);

    int in1_idx = input1_shape_size - 1;
    int in2_idx = input2_shape_size - 1;
    for (; in1_idx >= 0 || in2_idx >= 0; --in1_idx, --in2_idx)
    {
        size_t input1_val = (in1_idx >= 0) ? input1_shape[in1_idx] : 1;
        size_t input2_val = (in2_idx >= 0) ? input2_shape[in2_idx] : 1;

        if (input1_val == input2_val || input1_val == 1)
        {
            result_shape.insert(result_shape.begin(), input2_val);
        }
        else if (input2_val == 1)
        {
            result_shape.insert(result_shape.begin(), input1_val);
        }
        else
        {
            throw std::domain_error("DPNP Error: get_common_shape() failed with input shapes check");
        }
    }

    return result_shape;
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Normalizes an axes into a non-negative integer axes.
 *
 * Return vector of normalized axes with a non-negative integer axes.
 *
 * By default, this forbids axes from being specified multiple times.
 *
 * @param [in] __axes             Array with positive or negative indexes.
 * @param [in] __shape_size       The number of dimensions of the array that @ref __axes should be normalized against.
 * @param [in] __allow_duplicate  Disallow an axis from being specified twice. Default: false
 *
 * @exception std::range_error    Particular axis is out of range or other error.
 * @return                        The normalized axes indexes, such that `0 <= result < __shape_size`
 */
static inline std::vector<size_t>
    get_validated_axes(const std::vector<long>& __axes, const size_t __shape_size, const bool __allow_duplicate = false)
{
    std::vector<size_t> result;

    if (__axes.empty())
    {
        goto out;
    }

    if (__axes.size() > __shape_size)
    {
        goto err;
    }

    result.reserve(__axes.size());
    for (std::vector<long>::const_iterator it = __axes.cbegin(); it != __axes.cend(); ++it)
    {
        const long _axis = *it;
        const long input_shape_size_signed = static_cast<long>(__shape_size);
        if (_axis >= input_shape_size_signed)
        { // positive axis range check
            goto err;
        }

        if (_axis < -input_shape_size_signed)
        { // negative axis range check
            goto err;
        }

        const size_t positive_axis = _axis < 0 ? (_axis + input_shape_size_signed) : _axis;

        if (!__allow_duplicate)
        {
            if (std::find(result.begin(), result.end(), positive_axis) != result.end())
            { // find axis duplication
                goto err;
            }
        }

        result.push_back(positive_axis);
    }

out:
    return result;

err:
    // TODO exception if wrong axis? need common function for throwing exceptions
    throw std::range_error("DPNP Error: validate_axes() failed with axis check");
}

/**
 * @ingroup BACKEND_UTILS
 * @brief print std::vector to std::ostream.
 *
 * To print std::vector with POD types to std::out.
 *
 * @param [in]  vec  std::vector with types with ability to be printed.
 */
template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec)
{
    std::string delimeter;
    out << "{";
    // std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(out, ", "));
    // out << "\b\b}"; // last two 'backspaces' needs to eliminate last delimiter. ex: {2, 3, 4, }
    for (auto& elem : vec)
    {
        out << delimeter << elem;
        if (delimeter.empty())
        {
            delimeter.assign(", ");
        }
    }
    out << "}";

    return out;
}

/**
 * @ingroup BACKEND_UTILS
 * @brief print @ref DPNPFuncType to std::ostream.
 *
 * To print DPNPFuncType type to std::out.
 * TODO implement string representation of the enum
 *
 * @param [in]  elem  DPNPFuncType value to be printed.
 */
template <typename T>
std::ostream& operator<<(std::ostream& out, DPNPFuncType elem)
{
    out << static_cast<size_t>(elem);

    return out;
}

#endif // BACKEND_UTILS_H
