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

"""Module Backend (Logic part)

This module contains interface functions between C backend layer
and the rest of the library

"""


from dpnp.dpnp_utils cimport checker_throw_type_error


__all__ += [
    "dpnp_equal",
    "dpnp_greater",
    "dpnp_greater_equal",
    "dpnp_isclose",
    "dpnp_less",
    "dpnp_less_equal",
    "dpnp_logical_and",
    "dpnp_logical_not",
    "dpnp_logical_or",
    "dpnp_logical_xor",
    "dpnp_not_equal"
]


cpdef dparray dpnp_equal(dparray array1, input2):
    cdef dparray result = dparray(array1.shape, dtype=numpy.bool)

    if isinstance(input2, int):
        for i in range(result.size):
            result[i] = numpy.bool(array1[i] == input2)
    else:
        for i in range(result.size):
            result[i] = numpy.bool(array1[i] == input2[i])

    return result


cpdef dparray dpnp_greater(dparray input1, dparray input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.bool(input1[i] > input2[i])

    return result


cpdef dparray dpnp_greater_equal(dparray input1, dparray input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.bool(input1[i] >= input2[i])

    return result


cpdef dparray dpnp_isclose(dparray input1, input2, double rtol=1e-05, double atol=1e-08, bool equal_nan=False):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    if isinstance(input2, int):
        for i in range(result.size):
            result[i] = numpy.isclose(input1[i], input2, rtol, atol, equal_nan)
    else:
        for i in range(result.size):
            result[i] = numpy.isclose(input1[i], input2[i], rtol, atol, equal_nan)

    return result


cpdef dparray dpnp_less(dparray input1, dparray input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.bool(input1[i] < input2[i])

    return result


cpdef dparray dpnp_less_equal(dparray input1, dparray input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.bool(input1[i] <= input2[i])

    return result


cpdef dparray dpnp_logical_and(dparray input1, dparray input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.logical_and(input1[i], input2[i])

    return result


cpdef dparray dpnp_logical_not(dparray input1):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.logical_not(input1[i])

    return result


cpdef dparray dpnp_logical_or(dparray input1, dparray input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.logical_or(input1[i], input2[i])

    return result


cpdef dparray dpnp_logical_xor(dparray input1, dparray input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.logical_xor(input1[i], input2[i])

    return result


cpdef dparray dpnp_not_equal(dparray input1, dparray input2):
    cdef dparray result = dparray(input1.shape, dtype=numpy.bool)

    for i in range(result.size):
        result[i] = numpy.bool(input1[i] != input2[i])

    return result
