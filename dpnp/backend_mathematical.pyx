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
    "dpnp_divide",
    "dpnp_fabs",
    "dpnp_floor",
    "dpnp_floor_divide",
    "dpnp_fmod",
    'dpnp_hypot',
    "dpnp_maximum",
    "dpnp_minimum",
    "dpnp_multiply",
    "dpnp_negative",
    "dpnp_power",
    "dpnp_prod",
    "dpnp_sign",
    "dpnp_subtract",
    "dpnp_sum",
    "dpnp_trunc"
]


cpdef dparray dpnp_absolute(dparray input):
    cdef dparray_shape_type shape_input = input.shape
    cdef long size_input = input.size
    output_shape = dparray(len(shape_input), dtype=numpy.int64)
    for id, shape_ in enumerate(shape_input):
        output_shape[id] = shape_
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

    for source_idx in range(size_input):

        # reconstruct x,y,z from linear source_idx
        xyz = []
        remainder = source_idx
        for i in input_shape_offsets:
            quotient, remainder = divmod(remainder, i)
            xyz.append(quotient)

        result_indexes = []
        for idx, offset in enumerate(xyz):
            result_indexes.append(offset)

        result_offset = 0
        for i, result_indexes_val in enumerate(result_indexes):
            result_offset += (output_shape_offsets[i] * result_indexes_val)

        input_elem = input.item(source_idx)
        result_array[result_offset] = input_elem if input_elem >= 0 else -1 * input_elem

    dpnp_array = dpnp.array(result_array, dtype=input.dtype)
    dpnp_result_array = dpnp_array.reshape(output_shape)
    return dpnp_result_array


cpdef dparray dpnp_add(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_ADD, x1, x2, x1.shape)


cpdef dparray dpnp_arctan2(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_ARCTAN2, x1, x2, x1.shape)


cpdef dparray dpnp_ceil(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_CEIL, x1, x1.shape)


cpdef dparray dpnp_divide(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_DIVIDE, x1, x2, x1.shape)


cpdef dparray dpnp_fabs(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_FABS, x1, x1.shape)


cpdef dparray dpnp_floor(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_FLOOR, x1, x1.shape)


cpdef dparray dpnp_floor_divide(dparray array1, dparray array2):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = (array1[i] // array2[i])

    return result


cpdef dparray dpnp_hypot(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_HYPOT, x1, x2, x1.shape)


cpdef dparray dpnp_maximum(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_MAXIMUM, x1, x2, x1.shape)


cpdef dparray dpnp_minimum(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_MINIMUM, x1, x2, x1.shape)


cpdef dparray dpnp_multiply(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_MULTIPLY, x1, x2, x1.shape)


cpdef dparray dpnp_negative(dparray array1):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = -(array1[i])

    return result


cpdef dparray dpnp_power(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_POWER, x1, x2, x1.shape)


cpdef dparray dpnp_fmod(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_FMOD, x1, x2, x1.shape)


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


cpdef dparray dpnp_sign(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_SIGN, x1, x1.shape)


cpdef dparray dpnp_subtract(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_SUBTRACT, x1, x2, x1.shape)


cpdef dpnp_sum(dparray x1):
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

    return return_type.type(result[0])


cpdef dparray dpnp_trunc(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_TRUNC, x1, x1.shape)
