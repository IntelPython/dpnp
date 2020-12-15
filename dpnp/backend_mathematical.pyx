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

"""Module Backend (Mathematical part)

This module contains interface functions between C backend layer
and the rest of the library

"""


from dpnp.dpnp_utils cimport *
import dpnp
import numpy
cimport numpy


__all__ += [
    "dpnp_absolute",
    "dpnp_add",
    'dpnp_arctan2',
    "dpnp_ceil",
    "dpnp_copysign",
    "dpnp_cumprod",
    "dpnp_cumsum",
    "dpnp_diff",
    "dpnp_divide",
    "dpnp_fabs",
    "dpnp_floor",
    "dpnp_floor_divide",
    "dpnp_fmod",
    'dpnp_hypot',
    "dpnp_maximum",
    "dpnp_minimum",
    "dpnp_modf",
    "dpnp_multiply",
    "dpnp_nancumprod",
    "dpnp_nancumsum",
    "dpnp_nanprod",
    "dpnp_nansum",
    "dpnp_negative",
    "dpnp_power",
    "dpnp_prod",
    "dpnp_sign",
    "dpnp_subtract",
    "dpnp_sum",
    "dpnp_trunc"
]


ctypedef void(*fptr_custom_elemwise_absolute_1in_1out_t)(void *, void * , size_t)
ctypedef void(*fptr_1in_2out_t)(void *, void * , void * , size_t)


cpdef dparray dpnp_absolute(dparray input):
    cdef dparray_shape_type input_shape = input.shape
    cdef size_t input_shape_size = input.ndim

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ABSOLUTE, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(input_shape, dtype=result_type)

    cdef fptr_custom_elemwise_absolute_1in_1out_t func = <fptr_custom_elemwise_absolute_1in_1out_t > kernel_data.ptr
    # call FPTR function
    func(input.get_data(), result.get_data(), input.size)

    return result


cpdef dparray dpnp_add(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_ADD, x1, x2, x1.shape)


cpdef dparray dpnp_arctan2(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_ARCTAN2, x1, x2, x1.shape)


cpdef dparray dpnp_ceil(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_CEIL, x1, x1.shape)


cpdef dparray dpnp_copysign(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_COPYSIGN, x1, x2, x1.shape)


cpdef dparray dpnp_cumprod(dparray x1, bint usenan=False):

    x1_dtype = x1.dtype

    types_map = {
        'int32': dpnp.int64,
        'int64': dpnp.int64,
        'float32': dpnp.float32,
        'float64': dpnp.float64
    }

    res_type = types_map[x1_dtype.name]

    cdef dparray result = dparray(x1.size, dtype=res_type)

    if usenan and dpnp.isnan(x1[0]):
        cur_res = 1
    else:
        cur_res = x1[0]
    
    result._setitem_scalar(0, cur_res)

    for i in range(1, result.size):

        if usenan and dpnp.isnan(x1[i]):
            cur_res *= 1
        else:
            cur_res *= x1[i]

        result._setitem_scalar(i, cur_res)

    return result


cpdef dparray dpnp_cumsum(dparray x1, bint usenan=False):

    x1_dtype = x1.dtype

    types_map = {
        'int32': dpnp.int64,
        'int64': dpnp.int64,
        'float32': dpnp.float32,
        'float64': dpnp.float64
    }

    res_type = dpnp.dtype(types_map[x1_dtype.name])

    cdef dparray result = dparray(x1.size, dtype=res_type)

    if usenan and dpnp.isnan(x1[0]):
        cur_res = 0
    else:
        cur_res = x1[0]

    result._setitem_scalar(0, cur_res)

    for i in range(1, result.size):

        if usenan and dpnp.isnan(x1[i]):
            cur_res *= 1
        else:
            cur_res *= x1[i]

        result._setitem_scalar(i, cur_res)

    return result


cpdef dparray dpnp_diff(dparray input):
    size_i = input.size
    shape_i = input.shape
    list_shape_i = list(shape_i)
    list_shape_i[-1] = list_shape_i[-1] - 1
    output_shape = tuple(list_shape_i)
    res = []
    size_idx = output_shape[-1]
    size_arr_ = size_i /shape_i[-1]
    for i in range(size_arr_):
        for j in range(size_idx):
            idx = i * size_idx + j
            input_elem = input.item(idx) - input.item(idx + 1)
            res.appen(input_elem)

    dpnp_array = dpnp.array(res)
    dpnp_result_array = dpnp_array.reshape(output_shape)
    return dpnp_result_array


cpdef dparray dpnp_divide(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_DIVIDE, x1, x2, x1.shape)


cpdef dparray dpnp_fabs(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_FABS, x1, x1.shape)


cpdef dparray dpnp_floor(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_FLOOR, x1, x1.shape)


cpdef dparray dpnp_floor_divide(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_FLOOR_DIVIDE, x1, x2, x1.shape)


cpdef dparray dpnp_fmod(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_FMOD, x1, x2, x1.shape)


cpdef dparray dpnp_hypot(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_HYPOT, x1, x2, x1.shape)


cpdef dparray dpnp_maximum(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_MAXIMUM, x1, x2, x1.shape)


cpdef dparray dpnp_minimum(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_MINIMUM, x1, x2, x1.shape)


cpdef tuple dpnp_modf(dparray x1):
    """ Convert string type names (dparray.dtype) to C enum DPNPFuncType """
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    """ get the FPTR data structure """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MODF, param1_type, DPNP_FT_NONE)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    """ Create result arrays with type given by FPTR data """
    cdef dparray result1 = dparray(x1.shape, dtype=result_type)
    cdef dparray result2 = dparray(x1.shape, dtype=result_type)

    cdef fptr_1in_2out_t func = <fptr_1in_2out_t > kernel_data.ptr
    """ Call FPTR function """
    func(x1.get_data(), result1.get_data(), result2.get_data(), x1.size)

    return result1, result2


cpdef dparray dpnp_multiply(dparray x1, x2):
    x2_is_scalar = dpnp.isscalar(x2)

    x1_dtype_ = x1.dtype
    x2_dtype_ = type(x2) if x2_is_scalar else x2.dtype

    types_map = {float: dpnp.float64, int: dpnp.int64}
    x1_dtype = types_map.get(x1_dtype_, x1_dtype_)
    x2_dtype = types_map.get(x2_dtype_, x2_dtype_)

    if x1_dtype == dpnp.float64:
        if x2_dtype == dpnp.float64:
            res_type = dpnp.float64
        elif x2_dtype == dpnp.float32:
            res_type = dpnp.float64
        elif x2_dtype == dpnp.int64:
            res_type = dpnp.float64
        elif x2_dtype == dpnp.int32:
            res_type = dpnp.float64
    elif x1_dtype == dpnp.float32:
        if x2_dtype == dpnp.float64:
            res_type = dpnp.float32
        elif x2_dtype == dpnp.float32:
            res_type = dpnp.float32
        elif x2_dtype == dpnp.int64:
            res_type = dpnp.float32
        elif x2_dtype == dpnp.int32:
            res_type = dpnp.float32
    elif x1_dtype == dpnp.int64:
        if x2_dtype == dpnp.float64:
            res_type = dpnp.float64
        elif x2_dtype == dpnp.float32:
            res_type = dpnp.float32
        elif x2_dtype == dpnp.int64:
            res_type = dpnp.int64
        elif x2_dtype == dpnp.int32:
            res_type = dpnp.int64
    elif x1_dtype == dpnp.int32:
        if x2_dtype == dpnp.float64:
            res_type = dpnp.float64
        elif x2_dtype == dpnp.float32:
            res_type = dpnp.float32
        elif x2_dtype == dpnp.int64:
            res_type = dpnp.int32
        elif x2_dtype == dpnp.int32:
            res_type = dpnp.int32

    cdef dparray result = dparray(x1.shape, dtype=res_type)

    if x2_is_scalar:
        for i in range(result.size):
            result[i] = x1[i] * x2
        return result
    else:
        return call_fptr_2in_1out(DPNP_FN_MULTIPLY, x1, x2, x1.shape)


cpdef dparray dpnp_nancumprod(dparray x1):

    for i in range(x1.size):
        if dpnp.isnan(x1[i]):
            x1._setitem_scalar(i, 1)

    return dpnp_cumprod(x1)


cpdef dparray dpnp_nancumsum(dparray x1):

    for i in range(x1.size):
        if dpnp.isnan(x1[i]):
            x1._setitem_scalar(i, 0)

    return dpnp_cumsum(x1)


cpdef dpnp_nanprod(dparray x1):
    cdef dparray result = dparray(x1.shape, dtype=x1.dtype)

    for i in range(result.size):
        input_elem = x1.item(i)

        if dpnp.isnan(input_elem):
            result._setitem_scalar(i, 1)
        else:
            result._setitem_scalar(i, input_elem)

    return dpnp_prod(result)


cpdef dpnp_nansum(dparray x1):
    cdef dparray result = dparray(x1.shape, dtype=x1.dtype)

    for i in range(result.size):
        input_elem = x1.item(i)

        if dpnp.isnan(input_elem):
            result._setitem_scalar(i, 0)
        else:
            result._setitem_scalar(i, input_elem)

    # due to bug in dpnp_sum need this workaround
    # return dpnp_sum(result)
    sum_result = dpnp_sum(result)
    return x1.dtype.type(sum_result[0])


cpdef dparray dpnp_negative(dparray array1):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = -(array1[i])

    return result


cpdef dparray dpnp_power(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_POWER, x1, x2, x1.shape)


cpdef dpnp_prod(dparray x1):
    """
    input:float64   : outout:float64   : name:prod
    input:float32   : outout:float32   : name:prod
    input:int64     : outout:int64     : name:prod
    input:int32     : outout:int64     : name:prod
    input:bool      : outout:int64     : name:prod
    input:complex64 : outout:complex64 : name:prod
    input:complex128: outout:complex128: name:prod
    """

    cdef dparray result = call_fptr_1in_1out(DPNP_FN_PROD, x1, (1,))

    """ Numpy interface inconsistency """
    return_type = numpy.dtype(numpy.int64) if (x1.dtype == numpy.int32) else x1.dtype

    return return_type.type(result[0])


cpdef dparray dpnp_remainder(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_REMAINDER, x1, x2, x1.shape)


cpdef dparray dpnp_sign(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_SIGN, x1, x1.shape)


cpdef dparray dpnp_subtract(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_SUBTRACT, x1, x2, x1.shape)


cpdef dparray dpnp_sum_no_axis(dparray x1):
    """
    input:float64   : outout:float64   : name:sum
    input:float32   : outout:float32   : name:sum
    input:int64     : outout:int64     : name:sum
    input:int32     : outout:int64     : name:sum
    input:bool      : outout:int64     : name:sum
    input:complex64 : outout:complex64 : name:sum
    input:complex128: outout:complex128: name:sum
    """

    cdef dparray result = call_fptr_1in_1out(DPNP_FN_SUM, x1, (1,))

    """ Numpy interface inconsistency """
    return_type = numpy.dtype(numpy.int64) if (x1.dtype == numpy.int32) else x1.dtype

    cdef dparray result_ = dparray((1,), dtype=return_type)
    result_[0] = return_type.type(result)

    return result_


cpdef dparray dpnp_sum(dparray input, axis=None):
    if axis is None:
        return dpnp_sum_no_axis(input)

    cdef long size_input = input.size
    cdef dparray_shape_type shape_input = input.shape

    return_type = numpy.int64 if input.dtype == numpy.int32 else input.dtype

    axis_ = _object_to_tuple(axis)

    output_shape = dparray(len(shape_input) - len(axis_), dtype=numpy.int64)
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

    if len(output_shape) > 0:
        for i in range(len(output_shape)):
            ind = len(output_shape) - 1 - i
            output_shape_offsets[ind] = acc
            acc *= output_shape[ind]

    for source_idx in range(size_input):

        # reconstruct x,y,z from linear source_idx
        xyz = []
        remainder = source_idx
        for i in input_shape_offsets:
            quotient, remainder = divmod(remainder, i)
            xyz.append(quotient)

        # extract result axis
        result_axis = []
        for idx, offset in enumerate(xyz):
            if idx not in axis_:
                result_axis.append(offset)

        # Construct result offset
        result_offset = 0
        for i, result_axis_val in enumerate(result_axis):
            result_offset += (output_shape_offsets[i] * result_axis_val)

        input_elem = input.item(source_idx)
        if result_array[result_offset] is None:
            result_array[result_offset] = input_elem
        else:
            result_array[result_offset] += input_elem

    dpnp_array = dpnp.array(result_array, dtype=input.dtype)
    dpnp_result_array = dpnp_array.reshape(output_shape)
    return dpnp_result_array


cpdef dparray dpnp_trunc(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_TRUNC, x1, x1.shape)
