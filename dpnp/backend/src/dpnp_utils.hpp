//*****************************************************************************
// Copyright (c) 2016-2024, Intel Corporation
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
#include <cassert>
#include <complex>
#include <iostream>
#include <iterator>

#include <sycl/sycl.hpp>

#include <dpnp_iface_fptr.hpp>

#define LIBSYCL_VERSION_GREATER(major, minor, patch)                           \
    (__LIBSYCL_MAJOR_VERSION > major) ||                                       \
        (__LIBSYCL_MAJOR_VERSION == major and                                  \
         __LIBSYCL_MINOR_VERSION > minor) ||                                   \
        (__LIBSYCL_MAJOR_VERSION == major and                                  \
         __LIBSYCL_MINOR_VERSION == minor and                                  \
         __LIBSYCL_PATCH_VERSION >= patch)

/**
 * Version of SYCL DPC++ 2023 compiler where a return type of sycl::abs() is
 * changed from unsigned integer to signed one of input vector.
 */
#ifndef __SYCL_COMPILER_VECTOR_ABS_CHANGED
#define __SYCL_COMPILER_VECTOR_ABS_CHANGED 20230503L
#endif

/**
 * Version of SYCL DPC++ 2023 compiler at which transition to SYCL 2020 occurs.
 * Intel(R) oneAPI DPC++ 2022.2.1 compiler has version 20221020L on Linux and
 * 20221101L on Windows.
 */
#ifndef __SYCL_COMPILER_VERSION_REQUIRED
#define __SYCL_COMPILER_VERSION_REQUIRED 20221102L
#endif

/**
 * Version of Intel MKL at which transition to OneMKL release 2023.0.0 occurs.
 */
#ifndef __INTEL_MKL_2023_0_0_VERSION_REQUIRED
#define __INTEL_MKL_2023_0_0_VERSION_REQUIRED 20230000
#endif

/**
 * Version of Intel MKL at which transition to OneMKL release 2023.2.0 occurs.
 *
 * @note with OneMKL=2023.1.0 the call of oneapi::mkl::vm::div() was dead
 * locked inside ~usm_wrapper_to_host()->{...; q_->wait_and_throw(); ...}
 */
#ifndef __INTEL_MKL_2023_2_0_VERSION_REQUIRED
#define __INTEL_MKL_2023_2_0_VERSION_REQUIRED 20230002L
#endif

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
void get_shape_offsets_inkernel(const _DataType *shape,
                                size_t shape_size,
                                _DataType *offsets)
{
    size_t dim_prod_input = 1;
    for (size_t i = 0; i < shape_size; ++i) {
        long i_reverse = shape_size - 1 - i;
        offsets[i_reverse] = dim_prod_input;
        dim_prod_input *= shape[i_reverse];
    }

    return;
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Calculation of indices in array
 *
 * Calculates indices of element in array with given linear position
 * for example:
 *   idx = 5, shape = (2, 3), ndim = 2,
 *   indices xyz should be [1, 1]
 *
 * @param [in]  idx     linear index of the element in multy-D array.
 * @param [in]  ndim    number of dimensions.
 * @param [in]  shape   offsets of array.
 * @param [out] xyz     indices.
 */
template <typename _DataType>
void get_xyz_by_id(size_t idx,
                   size_t ndim,
                   const _DataType *offsets,
                   _DataType *xyz)
{
    size_t quotient;
    size_t remainder = idx;

    for (size_t i = 0; i < ndim; ++i) {
        quotient = remainder / offsets[i];
        remainder = remainder - quotient * offsets[i];
        xyz[i] = quotient;
    }
    return;
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Calculate xyz id for given axis from linear index
 *
 * Calculates xyz id of the array with given shape.
 * for example:
 *   input_array_shape_offsets[20, 5, 1]
 *   global_id == 5
 *   axis == 1
 *   xyz_id should be 1
 *
 * @param [in]  global_id     linear index of the element in multy-D array.
 * @param [in]  offsets       array with input offsets.
 * @param [in]  offsets_size  array size for @ref offsets parameter.
 * @param [in]  axis          axis.
 */
template <typename _DataType>
_DataType get_xyz_id_by_id_inkernel(size_t global_id,
                                    const _DataType *offsets,
                                    size_t offsets_size,
                                    size_t axis)
{
    /* avoid warning unused variable*/
    (void)offsets_size;

    assert(axis < offsets_size);

    _DataType xyz_id = 0;
    long reminder = global_id;
    for (size_t i = 0; i < axis + 1; ++i) {
        const _DataType axis_val = offsets[i];
        xyz_id = reminder / axis_val;
        reminder = reminder % axis_val;
    }

    return xyz_id;
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Check input shape is broadcastable to output one.
 *
 * @param [in] input_shape        Input shape.
 * @param [in] output_shape       Output shape.
 *
 * @return                        Input shape is broadcastable to output one or
 * not.
 */
static inline bool
    broadcastable(const std::vector<shape_elem_type> &input_shape,
                  const std::vector<shape_elem_type> &output_shape)
{
    if (input_shape.size() > output_shape.size()) {
        return false;
    }

    std::vector<shape_elem_type>::const_reverse_iterator irit =
        input_shape.rbegin();
    std::vector<shape_elem_type>::const_reverse_iterator orit =
        output_shape.rbegin();
    for (; irit != input_shape.rend(); ++irit, ++orit) {
        if (*irit != 1 && *irit != *orit) {
            return false;
        }
    }

    return true;
}

static inline bool
    broadcastable(const shape_elem_type *input_shape,
                  const size_t input_shape_size,
                  const std::vector<shape_elem_type> &output_shape)
{
    const std::vector<shape_elem_type> input_shape_vec(
        input_shape, input_shape + input_shape_size);
    return broadcastable(input_shape_vec, output_shape);
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Check arrays are equal.
 *
 * @param [in] input1        Input1.
 * @param [in] input1_size   Input1 size.
 * @param [in] input2        Input2.
 * @param [in] input2_size   Input2 size.
 *
 * @return                   Arrays are equal.
 */
template <typename _DataType>
static inline bool array_equal(const _DataType *input1,
                               const size_t input1_size,
                               const _DataType *input2,
                               const size_t input2_size)
{
    if (input1_size != input2_size)
        return false;

    const std::vector<_DataType> input1_vec(input1, input1 + input1_size);
    const std::vector<_DataType> input2_vec(input2, input2 + input2_size);

    return std::equal(std::begin(input1_vec), std::end(input1_vec),
                      std::begin(input2_vec));
}

/**
 * @ingroup BACKEND_UTILS
 * @brief Cast vector of DPCtl events to vector of SYCL enents.
 *
 * @param [in] events_ref      Reference to vector of DPCtl events.
 *
 * @return                     Vector of SYCL events.
 */
namespace
{
[[maybe_unused]] std::vector<sycl::event>
    cast_event_vector(const DPCTLEventVectorRef event_vec_ref)
{
    const size_t event_vec_size = DPCTLEventVector_Size(event_vec_ref);

    std::vector<sycl::event> event_vec;
    event_vec.reserve(event_vec_size);
    for (size_t i = 0; i < event_vec_size; ++i) {
        DPCTLSyclEventRef event_ref = DPCTLEventVector_GetAt(event_vec_ref, i);
        sycl::event event = *(reinterpret_cast<sycl::event *>(event_ref));
        event_vec.push_back(event);
    }
    return event_vec;
}
} // namespace

/**
 * @ingroup BACKEND_UTILS
 * @brief Normalizes an axes into a non-negative integer axes.
 *
 * Return vector of normalized axes with a non-negative integer axes.
 *
 * By default, this forbids axes from being specified multiple times.
 *
 * @param [in] __axes             Array with positive or negative indexes.
 * @param [in] __shape_size       The number of dimensions of the array that
 * @ref __axes should be normalized against.
 * @param [in] __allow_duplicate  Disallow an axis from being specified twice.
 * Default: false
 *
 * @exception std::range_error    Particular axis is out of range or other
 * error.
 * @return                        The normalized axes indexes, such that `0 <=
 * result < __shape_size`
 */
static inline std::vector<shape_elem_type>
    get_validated_axes(const std::vector<shape_elem_type> &__axes,
                       const size_t __shape_size,
                       const bool __allow_duplicate = false)
{
    std::vector<shape_elem_type> result;

    if (__axes.empty()) {
        goto out;
    }

    if (__axes.size() > __shape_size) {
        goto err;
    }

    result.reserve(__axes.size());
    for (std::vector<shape_elem_type>::const_iterator it = __axes.cbegin();
         it != __axes.cend(); ++it)
    {
        const shape_elem_type _axis = *it;
        const shape_elem_type input_shape_size_signed =
            static_cast<shape_elem_type>(__shape_size);
        if (_axis >= input_shape_size_signed) { // positive axis range check
            goto err;
        }

        if (_axis < -input_shape_size_signed) { // negative axis range check
            goto err;
        }

        const shape_elem_type positive_axis =
            _axis < 0 ? (_axis + input_shape_size_signed) : _axis;

        if (!__allow_duplicate) {
            if (std::find(result.begin(), result.end(), positive_axis) !=
                result.end()) { // find axis duplication
                goto err;
            }
        }

        result.push_back(positive_axis);
    }

out:
    return result;

err:
    // TODO exception if wrong axis? need common function for throwing
    // exceptions
    throw std::range_error(
        "DPNP Error: validate_axes() failed with axis check");
}

/**
 * @ingroup BACKEND_UTILS
 * @brief check support of type T by SYCL device.
 *
 * To check if sycl::device may use templated type T.
 *
 * @param [in]  q  sycl::device which is examined for type support.
 *
 * @exception std::runtime_error    type T is out of support by the queue.
 */
template <typename T>
static inline void validate_type_for_device(const sycl::device &d)
{
    if constexpr (std::is_same_v<T, double>) {
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'double'");
        }
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        if (!d.has(sycl::aspect::fp64)) {
            throw std::runtime_error(
                "Device " + d.get_info<sycl::info::device::name>() +
                " does not support type 'complex<double>'");
        }
    }
    else if constexpr (std::is_same_v<T, sycl::half>) {
        if (!d.has(sycl::aspect::fp16)) {
            throw std::runtime_error("Device " +
                                     d.get_info<sycl::info::device::name>() +
                                     " does not support type 'half'");
        }
    }
}

/**
 * @ingroup BACKEND_UTILS
 * @brief check support of type T by SYCL queue.
 *
 * To check if sycl::queue assigned to a device may use templated type T.
 *
 * @param [in]  q  sycl::queue which is examined for type support.
 *
 * @exception std::runtime_error    type T is out of support by the queue.
 */
template <typename T>
static inline void validate_type_for_device(const sycl::queue &q)
{
    validate_type_for_device<T>(q.get_device());
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
std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec)
{
    std::string delimiter;
    out << "{";
    // std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(out, ", "));
    // out << "\b\b}"; // last two 'backspaces' needs to eliminate last
    // delimiter. ex: {2, 3, 4, }
    for (auto &elem : vec) {
        out << delimiter << elem;
        if (delimiter.empty()) {
            delimiter.assign(", ");
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
std::ostream &operator<<(std::ostream &out, DPNPFuncType elem)
{
    out << static_cast<size_t>(elem);

    return out;
}

#endif // BACKEND_UTILS_H
