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
    "dpnp_nanprod",
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


cpdef dparray dpnp_multiply(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_MULTIPLY, x1, x2, x1.shape)


cpdef dpnp_nanprod(dparray x1):
    cdef dparray result = dparray(x1.shape, dtype=x1.dtype)

    for i in range(result.size):
        input_elem = x1.item(i)

        if dpnp.isnan(input_elem):
            result._setitem_scalar(i, 1)
        else:
            result._setitem_scalar(i, input_elem)

    return dpnp_prod(result)


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
