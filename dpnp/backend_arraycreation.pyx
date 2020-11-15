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

"""Module Backend (array creation part)

This module contains interface functions between C backend layer
and the rest of the library

"""


import dpnp
import numpy

from dpnp.dpnp_utils cimport *
from dpnp.backend cimport *


__all__ += [
    "dpnp_geomspace",
    "dpnp_linspace",
    "dpnp_logspace",
]


cpdef dparray dpnp_geomspace(start, stop, num, endpoint, dtype, axis):
    cdef dparray result = dparray(num, dtype=dtype)

    if endpoint:
        steps_count = num - 1
    else:
        steps_count = num

    # if there are steps, then fill values
    if steps_count > 0:
        step = dpnp.power(dpnp.float64(stop) / start, 1.0 / steps_count)
        mult = step
        for i in range(1, result.size):
            result[i] = start * mult
            mult = mult * step
    else:
        step = dpnp.nan

    # if result is not empty, then fiil first and last elements
    if num > 0:
        result[0] = start
        if endpoint and result.size > 1:
            result[result.size - 1] = stop

    return result


# TODO this function should work through dpnp_arange_c
cpdef tuple dpnp_linspace(start, stop, num, endpoint, retstep, dtype, axis):
    cdef dparray result = dparray(num, dtype=dtype)

    if endpoint:
        steps_count = num - 1
    else:
        steps_count = num

    # if there are steps, then fill values
    if steps_count > 0:
        step = (dpnp.float64(stop) - start) / steps_count
        for i in range(1, result.size):
            result[i] = start + step * i
    else:
        step = dpnp.nan

    # if result is not empty, then fiil first and last elements
    if num > 0:
        result[0] = start
        if endpoint and result.size > 1:
            result[result.size - 1] = stop

    return (result, step)


cpdef dparray dpnp_logspace(start, stop, num, endpoint, base, dtype, axis):
    temp = dpnp.linspace(start, stop, num=num, endpoint=endpoint)
    return dpnp.power(base, temp).astype(dtype)
