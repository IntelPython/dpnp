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

"""Module Backend (Binary operations)

This module contains interface functions between C backend layer
and the rest of the library

"""


from dpnp.dpnp_utils cimport *


__all__ += [
    "dpnp_bitwise_and",
    "dpnp_bitwise_or",
    "dpnp_bitwise_xor",
    "dpnp_invert",
    "dpnp_left_shift",
    "dpnp_right_shift",
]


cpdef dparray dpnp_bitwise_and(dparray array1, dparray array2):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = array1[i] & array2[i]

    return result


cpdef dparray dpnp_bitwise_or(dparray array1, dparray array2):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = array1[i] | array2[i]

    return result


cpdef dparray dpnp_bitwise_xor(dparray array1, dparray array2):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = array1[i] ^ array2[i]

    return result


cpdef dparray dpnp_invert(dparray arr):
    cdef dparray result = dparray(arr.shape, dtype=arr.dtype)

    for i in range(result.size):
        result[i] = ~arr[i]

    return result


cpdef dparray dpnp_left_shift(dparray array1, dparray array2):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = array1[i] << array2[i]

    return result


cpdef dparray dpnp_right_shift(dparray array1, dparray array2):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = array1[i] >> array2[i]

    return result
