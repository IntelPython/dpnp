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
    "dpnp_diag",
    "dpnp_geomspace",
    "dpnp_linspace",
    "dpnp_logspace",
    "dpnp_tri",
    "dpnp_tril",
    "dpnp_triu",
]


cpdef dparray dpnp_diag(v, k):
    cdef dparray result

    # computation of initial position
    init0 = max(0, -k)
    init1 = max(0, k)

    if v.ndim == 1:
        size = v.shape[0] + abs(k)

        result = dpnp.zeros(shape=(size, size), dtype=v.dtype)

        for i in range(v.shape[0]):
            result[init0 + i, init1 + i] = v[i]
    elif v.ndim == 2:
        # computation of result size
        size0 = min(v.shape[0], v.shape[0] + k)
        size1 = min(v.shape[1], v.shape[1] - k)
        size = min(size0, size1)
        if size < 0:
            size = 0

        result = dparray((size, ), dtype=v.dtype)

        for i in range(size):
            result[i] = v[init0 + i, init1 + i]
    else:
        checker_throw_value_error("dpnp_diag", "v.ndim", v.ndim, "1 or 2")

    return result


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


cpdef dparray dpnp_tri(N, M, k, dtype):
    cdef dparray result

    if M is None:
        M = N

    result = dparray(shape=(N, M), dtype=dtype)

    for i in range(N):
        diag_idx = max(0, i + k + 1)
        diag_idx = min(diag_idx, M)
        for j in range(diag_idx):
            result[i, j] = 1
        for j in range(diag_idx, M):
            result[i, j] = 0

    return result


cpdef dparray dpnp_tril(m, k):
    cdef dparray result

    if m.ndim == 1:
        result = dparray(shape=(m.shape[0], m.shape[0]), dtype=m.dtype)

        for i in range(result.size):
            ids = get_axis_indeces(i, result.shape)

            diag_idx = max(-1, ids[result.ndim - 2] + k)
            diag_idx = min(diag_idx, result.shape[result.ndim - 1])

            if ids[result.ndim - 1] <= diag_idx:
                result[i] = m[ids[result.ndim - 1]]
            else:
                result[i] = 0
    else:
        result = dparray(shape=m.shape, dtype=m.dtype)

        for i in range(result.size):
            ids = get_axis_indeces(i, result.shape)

            diag_idx = max(-1, ids[result.ndim - 2] + k)
            diag_idx = min(diag_idx, result.shape[result.ndim - 1])

            if ids[result.ndim - 1] <= diag_idx:
                result[i] = m[i]
            else:
                result[i] = 0

    return result


cpdef dparray dpnp_triu(m, k):
    cdef dparray result
    if m.ndim == 1:

        result = dparray(shape=(m.shape[0], m.shape[0]), dtype=m.dtype)

        for i in range(result.size):
            ids = get_axis_indeces(i, result.shape)

            diag_idx = max(-1, ids[result.ndim - 2] + k)
            diag_idx = min(diag_idx, result.shape[result.ndim - 1])

            if ids[result.ndim - 1] >= diag_idx:
                result[i] = m[ids[result.ndim - 1]]
            else:
                result[i] = 0
    else:
        result = dparray(shape=m.shape, dtype=m.dtype)

        for i in range(result.size):
            ids = get_axis_indeces(i, result.shape)

            diag_idx = max(-1, ids[result.ndim - 2] + k)
            diag_idx = min(diag_idx, result.shape[result.ndim - 1])

            if ids[result.ndim - 1] >= diag_idx:
                result[i] = m[i]
            else:
                result[i] = 0

    return result
