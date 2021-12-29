# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""Module Backend (Statistics part)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_average",
    "dpnp_correlate",
    "dpnp_cov",
    "dpnp_max",
    "dpnp_mean",
    "dpnp_median",
    "dpnp_min",
    "dpnp_nanvar",
    "dpnp_std",
    "dpnp_var",
]


# C function pointer to the C library template functions
ctypedef void(*fptr_custom_cov_1in_1out_t)(void *, void * , size_t, size_t)
ctypedef void(*fptr_custom_nanvar_t)(void *, void * , void * , size_t, size_t)
ctypedef void(*fptr_custom_std_var_1in_1out_t)(void *, void * , shape_elem_type * , size_t,
                                               shape_elem_type * , size_t, size_t)

# C function pointer to the C library template functions
ctypedef void(*custom_statistic_1in_1out_func_ptr_t)(void *, void * , shape_elem_type * , size_t,
                                                     shape_elem_type * , size_t)
ctypedef void(*custom_statistic_1in_1out_func_ptr_t_max)(void *, void * , const size_t, shape_elem_type * , size_t,
                                                         shape_elem_type * , size_t)


cdef utils.dpnp_descriptor call_fptr_custom_std_var_1in_1out(DPNPFuncName fptr_name, utils.dpnp_descriptor x1, ddof):
    cdef shape_type_c x1_shape = x1.shape

    """ Convert string type names (array.dtype) to C enum DPNPFuncType """
    cdef DPNPFuncType param_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    """ get the FPTR data structure """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(fptr_name, param_type, DPNP_FT_NONE)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_custom_std_var_1in_1out_t func = <fptr_custom_std_var_1in_1out_t > kernel_data.ptr

    # stub for interface support
    cdef shape_type_c axis
    cdef Py_ssize_t axis_size = 0

    """ Call FPTR function """
    func(x1.get_data(), result.get_data(), x1_shape.data(),
         x1.ndim, axis.data(), axis_size, ddof)

    return result


cpdef dpnp_average(utils.dpnp_descriptor x1):
    array_sum = dpnp_sum(x1).get_pyobj()

    """ Numpy interface inconsistency """
    return_type = numpy.float32 if (x1.dtype == numpy.float32) else numpy.float64

    return (return_type(array_sum / x1.size))


cpdef utils.dpnp_descriptor dpnp_correlate(utils.dpnp_descriptor x1, utils.dpnp_descriptor x2):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(x2.dtype)

    cdef shape_type_c x1_shape = x1.shape
    cdef shape_type_c x2_shape = x2.shape

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CORRELATE, param1_type, param2_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr

    func(result.get_data(), x1.get_data(), x1.size, x1_shape.data(), x1_shape.size(),
         x2.get_data(), x2.size, x2_shape.data(), x2_shape.size(), NULL)

    return result


# supports "double" input only
cpdef utils.dpnp_descriptor dpnp_cov(utils.dpnp_descriptor array1):
    cdef shape_type_c input_shape = array1.shape

    if array1.ndim == 1:
        input_shape.insert(input_shape.begin(), 1)

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_COV, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (input_shape[0], input_shape[0])
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_custom_cov_1in_1out_t func = <fptr_custom_cov_1in_1out_t > kernel_data.ptr
    # call FPTR function
    func(array1.get_data(), result.get_data(), input_shape[0], input_shape[1])

    return result


cdef utils.dpnp_descriptor _dpnp_max(utils.dpnp_descriptor input, _axis_, shape_type_c result_shape):
    cdef shape_type_c input_shape = input.shape
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MAX, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef custom_statistic_1in_1out_func_ptr_t_max func = <custom_statistic_1in_1out_func_ptr_t_max > kernel_data.ptr
    cdef shape_type_c axis
    cdef Py_ssize_t axis_size = 0
    cdef shape_type_c axis_ = axis

    if _axis_ is not None:
        axis = _axis_
        axis_.reserve(len(axis))
        for shape_it in axis:
            axis_.push_back(shape_it)
        axis_size = len(axis)

    func(input.get_data(),
         result.get_data(),
         result.size,
         input_shape.data(),
         input.ndim,
         axis_.data(),
         axis_size)

    return result


cpdef utils.dpnp_descriptor dpnp_max(utils.dpnp_descriptor input, axis):
    cdef shape_type_c shape_input = input.shape
    cdef shape_type_c output_shape

    if axis is None:
        axis_ = axis
        output_shape.push_back(1)
    else:
        if isinstance(axis, int):
            if axis < 0:
                axis_ = tuple([input.ndim - axis])
            else:
                axis_ = tuple([axis])
        else:
            _axis_ = []
            for i in range(len(axis)):
                if axis[i] < 0:
                    _axis_.append(input.ndim - axis[i])
                else:
                    _axis_.append(axis[i])
            axis_ = tuple(_axis_)

        output_shape.resize(len(shape_input) - len(axis_), 0)
        ind = 0
        for id, shape_axis in enumerate(shape_input):
            if id not in axis_:
                output_shape[ind] = shape_axis
                ind += 1

    return _dpnp_max(input, axis_, output_shape)


cpdef utils.dpnp_descriptor _dpnp_mean(utils.dpnp_descriptor input):
    cdef shape_type_c input_shape = input.shape
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MEAN, param1_type, param1_type)

    cdef utils.dpnp_descriptor result = utils.create_output_descriptor((1,), kernel_data.return_type, None)

    cdef custom_statistic_1in_1out_func_ptr_t func = <custom_statistic_1in_1out_func_ptr_t > kernel_data.ptr

    # stub for interface support
    cdef shape_type_c axis
    cdef Py_ssize_t axis_size = 0

    func(input.get_data(),
         result.get_data(),
         input_shape.data(),
         input.ndim,
         axis.data(),
         axis_size)

    return result


cpdef object dpnp_mean(utils.dpnp_descriptor input, axis):
    cdef shape_type_c output_shape

    if axis is None:
        return _dpnp_mean(input).get_pyobj()

    cdef long size_input = input.size
    cdef shape_type_c shape_input = input.shape

    if input.dtype == dpnp.float32:
        res_type = dpnp.float32
    else:
        res_type = dpnp.float64

    if size_input == 0:
        return dpnp.array([dpnp.nan], dtype=res_type)

    if isinstance(axis, int):
        axis_ = tuple([axis])
    else:
        axis_ = axis

    if axis_ is None:
        output_shape.push_back(1)
    else:
        output_shape = (0, ) * (len(shape_input) - len(axis_))
        ind = 0
        for id, shape_axis in enumerate(shape_input):
            if id not in axis_:
                output_shape[ind] = shape_axis
                ind += 1

    cdef long prod = 1
    for i in range(len(output_shape)):
        if output_shape[i] != 0:
            prod *= output_shape[i]

    result_array = [None] * prod
    input_shape_offsets = [None] * len(shape_input)
    acc = 1

    for i in range(len(shape_input)):
        ind = len(shape_input) - 1 - i
        input_shape_offsets[ind] = acc
        acc *= shape_input[ind]

    output_shape_offsets = [None] * len(shape_input)
    acc = 1

    if axis_ is not None:
        for i in range(len(output_shape)):
            ind = len(output_shape) - 1 - i
            output_shape_offsets[ind] = acc
            acc *= output_shape[ind]
            result_offsets = input_shape_offsets[:]  # need copy. not a reference
        for i in axis_:
            result_offsets[i] = 0

    for source_idx in range(size_input):

        # reconstruct x,y,z from linear source_idx
        xyz = []
        remainder = source_idx
        for i in input_shape_offsets:
            quotient, remainder = divmod(remainder, i)
            xyz.append(quotient)

        # extract result axis
        result_axis = []
        if axis_ is None:
            result_axis = xyz
        else:
            for idx, offset in enumerate(xyz):
                if idx not in axis_:
                    result_axis.append(offset)

        # Construct result offset
        result_offset = 0
        if axis_ is not None:
            for i, result_axis_val in enumerate(result_axis):
                result_offset += (output_shape_offsets[i] * result_axis_val)

        input_elem = input.get_pyobj().item(source_idx)
        if axis_ is None:
            if result_array[0] is None:
                result_array[0] = input_elem
            else:
                result_array[0] += input_elem
        else:
            if result_array[result_offset] is None:
                result_array[result_offset] = input_elem
            else:
                result_array[result_offset] += input_elem

    del_ = size_input
    if axis_ is not None:
        for i in range(len(shape_input)):
            if i not in axis_:
                del_ = del_ / shape_input[i]
    dpnp_array = dpnp.array(result_array, dtype=input.dtype)
    dpnp_result_array = dpnp.reshape(dpnp_array, output_shape)
    return dpnp_result_array / del_


cpdef utils.dpnp_descriptor dpnp_median(utils.dpnp_descriptor array1):
    cdef shape_type_c x1_shape = array1.shape
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(array1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MEDIAN, param1_type, param1_type)

    cdef utils.dpnp_descriptor result = utils.create_output_descriptor((1,), kernel_data.return_type, None)

    cdef custom_statistic_1in_1out_func_ptr_t func = <custom_statistic_1in_1out_func_ptr_t > kernel_data.ptr

    # stub for interface support
    cdef shape_type_c axis
    cdef Py_ssize_t axis_size = 0

    func(array1.get_data(),
         result.get_data(),
         x1_shape.data(),
         array1.ndim,
         axis.data(),
         axis_size)

    return result


cpdef utils.dpnp_descriptor _dpnp_min(utils.dpnp_descriptor input, _axis_, shape_type_c shape_output):
    cdef shape_type_c input_shape = input.shape
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MIN, param1_type, param1_type)

    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(shape_output, kernel_data.return_type, None)

    cdef custom_statistic_1in_1out_func_ptr_t_max func = <custom_statistic_1in_1out_func_ptr_t_max > kernel_data.ptr
    cdef shape_type_c axis
    cdef Py_ssize_t axis_size = 0
    cdef shape_type_c axis_ = axis

    if _axis_ is not None:
        axis = _axis_
        axis_.reserve(len(axis))
        for shape_it in axis:
            if shape_it < 0:
                raise ValueError("DPNP algo::_dpnp_min(): Negative values in 'shape' are not allowed")
            axis_.push_back(shape_it)
        axis_size = len(axis)

    func(input.get_data(),
         result.get_data(),
         result.size,
         input_shape.data(),
         input.ndim,
         axis_.data(),
         axis_size)

    return result


cpdef utils.dpnp_descriptor dpnp_min(utils.dpnp_descriptor input, axis):
    cdef shape_type_c shape_input = input.shape
    cdef shape_type_c shape_output

    if axis is None:
        axis_ = axis
        shape_output = (1,)
    else:
        if isinstance(axis, int):
            if axis < 0:
                axis_ = tuple([input.ndim - axis])
            else:
                axis_ = tuple([axis])
        else:
            _axis_ = []
            for i in range(len(axis)):
                if axis[i] < 0:
                    _axis_.append(input.ndim - axis[i])
                else:
                    _axis_.append(axis[i])
            axis_ = tuple(_axis_)

        for id, shape_axis in enumerate(shape_input):
            if id not in axis_:
                shape_output.push_back(shape_axis)

    return _dpnp_min(input, axis_, shape_output)


cpdef utils.dpnp_descriptor dpnp_nanvar(utils.dpnp_descriptor arr, ddof):
    # dpnp_isnan does not support USM array as input in comparison to dpnp.isnan
    cdef utils.dpnp_descriptor mask_arr = dpnp.get_dpnp_descriptor(dpnp.isnan(arr.get_pyobj()))
    n = dpnp.count_nonzero(mask_arr.get_pyobj())
    res_size = arr.size - n
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(arr.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_NANVAR, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(res_size)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_custom_nanvar_t func = <fptr_custom_nanvar_t > kernel_data.ptr

    func(arr.get_data(), mask_arr.get_data(), result.get_data(), result.size, arr.size)

    return call_fptr_custom_std_var_1in_1out(DPNP_FN_VAR, result, ddof)


cpdef utils.dpnp_descriptor dpnp_std(utils.dpnp_descriptor a, size_t ddof):
    return call_fptr_custom_std_var_1in_1out(DPNP_FN_STD, a, ddof)


cpdef utils.dpnp_descriptor dpnp_var(utils.dpnp_descriptor a, size_t ddof):
    return call_fptr_custom_std_var_1in_1out(DPNP_FN_VAR, a, ddof)
