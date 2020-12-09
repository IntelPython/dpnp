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

"""Module Backend (Indexing part)

This module contains interface functions between C backend layer
and the rest of the library

"""


import numpy
from dpnp.dpnp_utils cimport *
from dpnp.dpnp_iface_counting import count_nonzero


__all__ += [
    "dpnp_diag_indices",
    "dpnp_nonzero",
    "dpnp_tril_indices",
    "dpnp_tril_indices_from",
    "dpnp_triu_indices",
    "dpnp_triu_indices_from"
]


cpdef tuple dpnp_diag_indices(n, ndim):
    cdef dparray res_item = dpnp.arange(n, dtype=dpnp.int64)

    # yes, all are the same item
    result = []
    for i in range(ndim):
        result.append(res_item)

    return tuple(result)


cpdef tuple dpnp_nonzero(dparray in_array1):
    res_count = in_array1.ndim

    # have to go through array one extra time to count size of result arrays
    res_size = count_nonzero(in_array1)

    res_list = []
    for i in range(res_count):
        res_list.append(dparray((res_size, ), dtype=dpnp.int64))
    result = _object_to_tuple(res_list)

    idx = 0
    for i in range(in_array1.size):
        if in_array1[i] == 0:
            ids = get_axis_indeces(i, in_array1.shape)
            for j in range(res_count):
                result[j][idx] = ids[j]
            idx = idx + 1

    return result


cpdef tuple dpnp_tril_indices(n, k=0, m=None):
    array1 = []
    array2 = []
    if m is None:
        for i in range(n):
            for j in range(i + 1 + k):
                if j >= n:
                    continue
                else:
                    array1.append(i)
                    array2.append(j)
    else:
        for i in range(n):
            for j in range(i + 1 + k):
                if j < m :
                    array1.append(i)
                    array2.append(j)

    dparray1 = dpnp.array(array1, dtype=dpnp.int64)
    dparray2 = dpnp.array(array2, dtype=dpnp.int64)
    return (dparray1, dparray2)


cpdef tuple dpnp_tril_indices_from(arr, k=0):
    m = arr.shape[0]
    n = arr.shape[1]
    array1 = []
    array2 = []
    if m is None:
        for i in range(n):
            for j in range(i + 1 + k):
                if j >= n:
                    continue
                else:
                    array1.append(i)
                    array2.append(j)
    else:
        for i in range(n):
            for j in range(i + 1 + k):
                if j < m :
                    array1.append(i)
                    array2.append(j)

    dparray1 = dpnp.array(array1, dtype=dpnp.int64)
    dparray2 = dpnp.array(array2, dtype=dpnp.int64)
    return (dparray1, dparray2)


cpdef tuple dpnp_triu_indices(n, k=0, m=None):
    array1 = []
    array2 = []
    if m is None:
        for i in range(n):
            for j in range(i + k, n):
                array1.append(i)
                array2.append(j)
    else:
        for i in range(n):
            for j in range(i + k, m):
                array1.append(i)
                array2.append(j)

    dparray1 = dpnp.array(array1, dtype=dpnp.int64)
    dparray2 = dpnp.array(array2, dtype=dpnp.int64)
    return (dparray1, dparray2)


cpdef tuple dpnp_triu_indices_from(arr, k=0):
    m = arr.shape[0]
    n = arr.shape[1]
    array1 = []
    array2 = []
    if m is None:
        for i in range(n):
            for j in range(i + k, n):
                array1.append(i)
                array2.append(j)
    else:
        for i in range(n):
            for j in range(i + k, m):
                array1.append(i)
                array2.append(j)

    dparray1 = dpnp.array(array1, dtype=dpnp.int64)
    dparray2 = dpnp.array(array2, dtype=dpnp.int64)
    return (dparray1, dparray2)
