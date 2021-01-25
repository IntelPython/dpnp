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
    "dpnp_diagonal",
    "dpnp_fill_diagonal",
    "dpnp_indices",
    "dpnp_nonzero",
    "dpnp_place",
    "dpnp_put",
    "dpnp_putmask",
    "dpnp_select",
    "dpnp_take",
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


cpdef dparray dpnp_diagonal(dparray input, offset=0):
    n = min(input.shape[0], input.shape[1])
    res_shape = [None] * (input.ndim - 1)

    if input.ndim > 2:
        for i in range(input.ndim - 2):
            res_shape[i] = input.shape[i + 2]

    if (n + offset) > input.shape[1]:
        res_shape[-1] = input.shape[1] - offset
    elif (n + offset) > input.shape[0]:
        res_shape[-1] = input.shape[0]
    else:
        res_shape[-1] = n + offset

    res_size = 1
    for i in range(len(res_shape)):
        res_size *= res_shape[i]

    xyz = {}
    if input.ndim > 2:
        for i in range(res_shape[0]):
            xyz[i] = [i]

        index = 1

        while index < len(res_shape) - 1:
            shape_element = res_shape[index]
            new_shape_array = {}
            ind = 0
            for i in range(shape_element):
                for j in range(len(xyz)):
                    new_shape = []
                    list_ind = xyz[j]
                    for k in range(len(list_ind)):
                        new_shape.append(list_ind[k])
                    new_shape.append(i)
                    new_shape_array[ind] = new_shape
                    ind += 1
            for k in range(len(new_shape_array)):
                if k < len(xyz):
                    del xyz[k]
                list_ind = new_shape_array[k]
                xyz[k] = list_ind
            index += 1

    result = dparray(res_shape, dtype=input.dtype)
    for i in range(res_shape[-1]):
        if len(xyz) != 0:
            for j in range(len(xyz)):
                ind_input_ = [i, i + offset]
                ind_output_ = []
                ind_list = xyz[j]
                for k in range(len(ind_list)):
                    ind_input_.append(ind_list[k])
                    ind_output_.append(ind_list[k])
                ind_output_.append(ind_input_[0])
                ind_input = tuple(ind_input_)
                ind_output = tuple(ind_output_)
                result[ind_output] = input[ind_input]
        else:
            ind_input_ = [i, i + offset]
            ind_output_ = []
            ind_output_.append(ind_input_[0])
            ind_input = tuple(ind_input_)
            ind_output = tuple(ind_output_)
            result[ind_output] = input[ind_input]

    return result


cpdef dpnp_fill_diagonal(dparray input, val):
    for i in range(min(input.shape)):
        ind_list = [i] * input.ndim
        ind = tuple(ind_list)
        input[ind] = val


cpdef dparray dpnp_indices(dimensions):
    len_dimensions = len(dimensions)
    res_shape = []
    res_shape.append(len_dimensions)
    for i in range(len_dimensions):
        res_shape.append(dimensions[i])

    result = []
    if len_dimensions == 1:
        res = []
        for i in range(dimensions[0]):
            res.append(i)
        result.append(res)
    else:
        res1 = []
        for i in range(dimensions[0]):
            res = []
            for j in range(dimensions[1]):
                res.append(i)
            res1.append(res)
        result.append(res1)

        res2 = []
        for i in range(dimensions[0]):
            res = []
            for j in range(dimensions[1]):
                res.append(j)
            res2.append(res)
        result.append(res2)

    dpnp_result = dpnp.array(result)
    return dpnp_result


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
        if in_array1[i] != 0:
            ids = get_axis_indeces(i, in_array1.shape)
            for j in range(res_count):
                result[j][idx] = ids[j]
            idx = idx + 1

    return result


cpdef dpnp_place(dparray arr, dparray mask, vals):
    cpdef int counter = 0
    cpdef int vals_len = len(vals)
    for i in range(arr.size):
        if mask[i]:
            arr[i] = vals[counter % vals_len]
            counter += 1


cpdef dpnp_put(input, ind, v):
    ind_is_list = isinstance(ind, list)
    for i in range(input.size):
        if ind_is_list:
            for j in range(len(ind)):
                val = ind[j]
                if i == val:
                    input[i] = v[j]
                    in_ind = 1
                    break
        else:
            if i == ind:
                input[i] = v
                in_ind = 1


cpdef dpnp_putmask(dparray arr, dparray mask, dparray values):
    cpdef int values_size = values.size
    for i in range(arr.size):
        if mask[i]:
            arr[i] = values[i % values_size]


cpdef dparray dpnp_select(condlist, choicelist, default):
    size_ = condlist[0].size
    res_array = dparray(size_, dtype=choicelist[0].dtype)
    pass_val = {a: default for a in range(size_)}
    for i in range(len(condlist)):
        for j in range(size_):
            if (condlist[i])[j]:
                res_array[j] = (choicelist[i])[j]
                pass_val.pop(j)

    for ind, val in pass_val.items():
        res_array[ind] = val

    return res_array.reshape(condlist[0].shape)


cpdef dparray dpnp_take(dparray input, dparray indices):
    indices_size = indices.size
    res_array = dparray(indices_size, dtype=input.dtype)
    for i in range(indices_size):
        ind = indices[i]
        res_array[i] = input[ind]
    result = res_array.reshape(indices.shape)
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
                if j < m:
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
                if j < m:
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
