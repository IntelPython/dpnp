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

"""Module Utilities

This module contains differnt helpers and utilities

"""

import dpnp
import dpnp.config as config
import numpy
cimport cpython
cimport cython
cimport numpy


"""
Python import functions
"""
__all__ = [
    "checker_throw_axis_error",
    "checker_throw_index_error",
    "checker_throw_runtime_error",
    "checker_throw_type_error",
    "checker_throw_value_error",
    # "copy_values_to_dparray",
    "dp2nd_array",
    "_get_linear_index",
    # "get_shape_dtype",
    "nd2dp_array",
    "normalize_axis",
    # "_normalize_order",
    "_object_to_tuple",
    "use_origin_backend"
]

cdef ERROR_PREFIX = "Intel NumPy error:"


cpdef checker_throw_axis_error(function_name, param_name, param, expected):
    err_msg = f"{ERROR_PREFIX} in function {function_name}()"
    err_msg += f" axes '{param_name}' expected `{expected}`, but '{param}' provided"
    raise numpy.AxisError(err_msg)


cpdef checker_throw_index_error(function_name, index, size):
    raise IndexError(
        f"{ERROR_PREFIX} in function {function_name}() index {index} is out of bounds. dimension size `{size}`")


cpdef checker_throw_runtime_error(function_name, message):
    raise RuntimeError(f"{ERROR_PREFIX} in function {function_name}(): '{message}'")


cpdef checker_throw_type_error(function_name, given_type):
    raise TypeError(f"{ERROR_PREFIX} in function {function_name}() type '{given_type}' is not supported")


cpdef checker_throw_value_error(function_name, param_name, param, expected):
    # import sys
    # sys.tracebacklimit = 0
    err_msg = f"{ERROR_PREFIX} in function {function_name}() paramenter '{param_name}'"
    err_msg += f" expected `{expected}`, but '{param}' provided"
    raise ValueError(err_msg)


cdef long copy_values_to_dparray(dparray dst, input_obj, size_t dst_idx=0) except -1:
    cdef elem_dtype = dst.dtype

    for elem_value in input_obj:
        if isinstance(elem_value, (list, tuple)):
            dst_idx = copy_values_to_dparray(dst, elem_value, dst_idx)
        elif issubclass(type(elem_value), (numpy.ndarray, dparray)):
            dst_idx = copy_values_to_dparray(dst, elem_value, dst_idx)
        else:
            if elem_dtype == numpy.float64:
                ( < double * > dst.get_data())[dst_idx] = elem_value
            elif elem_dtype == numpy.float32:
                ( < float * > dst.get_data())[dst_idx] = elem_value
            elif elem_dtype == numpy.int64:
                ( < long * > dst.get_data())[dst_idx] = elem_value
            elif elem_dtype == numpy.int32:
                ( < int * > dst.get_data())[dst_idx] = elem_value
            else:
                checker_throw_type_error("copy_values_to_dparray", elem_dtype)

            dst_idx += 1

    return dst_idx


cpdef dp2nd_array(arr):
    """Convert dparray to ndarray"""
    return dpnp.asnumpy(arr) if isinstance(arr, dparray) else arr


cpdef long _get_linear_index(key, tuple shape, int ndim):
    """
    Compute linear index of an element in memory from array indices
    """

    if isinstance(key, tuple):
        li = 0
        m = 1
        for i in range(ndim - 1, -1, -1):
            li += key[i] * m
            m *= shape[i]
    else:
        li = key
    return li


cdef tuple get_shape_dtype(object input_obj):
    cdef dparray_shape_type return_shape  # empty shape means scalar
    return_dtype = None

    if issubclass(type(input_obj), (numpy.ndarray, dparray)):
        return (input_obj.shape, input_obj.dtype)

    cdef dparray_shape_type elem_shape
    cdef dparray_shape_type list_shape
    if isinstance(input_obj, (list, tuple)):
        for elem in input_obj:
            elem_shape, elem_dtype = get_shape_dtype(elem)

            if return_shape.empty():
                return_shape = elem_shape
                return_dtype = elem_dtype

            # shape and dtype does not match with siblings.
            if ((return_shape != elem_shape) or (return_dtype != elem_dtype)):
                return (elem_shape, numpy.dtype(numpy.object))

        list_shape.push_back(len(input_obj))
        list_shape.insert(list_shape.end(), return_shape.begin(), return_shape.end())
        return (list_shape, return_dtype)

    # assume scalar or object
    return (return_shape, numpy.dtype(type(input_obj)))


cpdef nd2dp_array(arr):
    """Convert ndarray to dparray"""
    if not isinstance(arr, numpy.ndarray):
        return arr

    result = dparray(arr.shape, dtype=arr.dtype)
    for i in range(result.size):
        result._setitem_scalar(i, arr.item(i))

    return result


cpdef dparray_shape_type normalize_axis(dparray_shape_type axis, size_t shape_size_inp):
    """
    Conversion of the transformation shape axis [-1, 0, 1] into [2, 0, 1] where numbers are `id`s of array shape axis
    """

    cdef ssize_t shape_size = shape_size_inp  # convert type for comparison with axis id

    cdef size_t axis_size = axis.size()
    cdef dparray_shape_type result = dparray_shape_type(axis_size, 0)
    for i in range(axis_size):
        if (axis[i] >= shape_size) or (axis[i] < -shape_size):
            checker_throw_axis_error("normalize_axis", "axis", axis[i], shape_size - 1)

        if (axis[i] < 0):
            result[i] = shape_size + axis[i]
        else:
            result[i] = axis[i]

    return result


@cython.profile(False)
cdef inline int _normalize_order(order, cpp_bool allow_k=True) except? 0:
    """ Converts memory order letters to some common view

    """

    cdef int order_type
    order_type = b'C' if len(order) == 0 else ord(order[0])

    if order_type == b'K' or order_type == b'k':
        if not allow_k:
            raise ValueError("Intel NumPy _normalize_order(): order \'K\' is not permitted")
        order_type = b'K'
    elif order_type == b'A' or order_type == b'a':
        order_type = b'A'
    elif order_type == b'C' or order_type == b'c':
        order_type = b'C'
    elif order_type == b'F' or order_type == b'f':
        order_type = b'F'
    else:
        raise TypeError("Intel NumPy _normalize_order(): order is not understood")

    return order_type


@cython.profile(False)
cpdef inline tuple _object_to_tuple(object obj):
    """ Converts Python object into tuple

    """

    if obj is None:
        return ()

    if cpython.PySequence_Check(obj):
        return tuple(obj)

    if isinstance(obj, int):
        return obj,

    raise ValueError("Intel NumPy object_to_tuple(): 'obj' should be 'None', collections.abc.Sequence, or 'int'")


cpdef cpp_bool use_origin_backend(input1=None, size_t compute_size=0):
    """
    This function needs to redirect particular computation cases to original backend
    Parameters:
        input1: One of the input parameter of the API function
        compute_size: Some amount of total compute size of the task
    Return:
        True - computations are better to be executed on original backend
        False - it is better to use this SW to compute
    """

    if (config.__DPNP_ORIGIN__):
        return True

    return False
