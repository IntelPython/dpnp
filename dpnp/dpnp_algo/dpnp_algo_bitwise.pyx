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


cpdef dparray dpnp_bitwise_and(object x1_obj, object x2_obj, object dtype=None, dparray out=None, object where=True):
    return call_fptr_2in_1out(DPNP_FN_BITWISE_AND, x1_obj, x2_obj, dtype=dtype, out=out, where=where, new_version=True)


cpdef dparray dpnp_bitwise_or(object x1_obj, object x2_obj, object dtype=None, dparray out=None, object where=True):
    return call_fptr_2in_1out(DPNP_FN_BITWISE_OR, x1_obj, x2_obj, dtype=dtype, out=out, where=where, new_version=True)


cpdef dparray dpnp_bitwise_xor(object x1_obj, object x2_obj, object dtype=None, dparray out=None, object where=True):
    return call_fptr_2in_1out(DPNP_FN_BITWISE_XOR, x1_obj, x2_obj, dtype=dtype, out=out, where=where, new_version=True)


cpdef dparray dpnp_invert(dparray arr):
    return call_fptr_1in_1out(DPNP_FN_INVERT, arr, arr.shape)


cpdef dparray dpnp_left_shift(object x1_obj, object x2_obj, object dtype=None, dparray out=None, object where=True):
    return call_fptr_2in_1out(DPNP_FN_LEFT_SHIFT, x1_obj, x2_obj, dtype=dtype, out=out, where=where, new_version=True)

cpdef dparray dpnp_right_shift(object x1_obj, object x2_obj, object dtype=None, dparray out=None, object where=True):
    return call_fptr_2in_1out(DPNP_FN_RIGHT_SHIFT, x1_obj, x2_obj, dtype=dtype, out=out, where=where, new_version=True)
