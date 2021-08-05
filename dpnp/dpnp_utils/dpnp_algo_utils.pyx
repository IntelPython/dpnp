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

import numpy
import dpnp.config as config
import dpnp
from dpnp.dpnp_algo cimport dpnp_DPNPFuncType_to_dtype, dpnp_dtype_to_DPNPFuncType, get_dpnp_function_ptr
from libcpp cimport bool as cpp_bool
from libcpp.complex cimport complex as cpp_complex

cimport cpython
cimport cython
cimport numpy


"""
Python import functions
"""
__all__ = [
    "call_origin",
    "checker_throw_axis_error",
    "checker_throw_index_error",
    "checker_throw_runtime_error",
    "checker_throw_type_error",
    "checker_throw_value_error",
    "create_output_descriptor_py",
    "dp2nd_array",
    "dpnp_descriptor",
    "get_axis_indeces",
    "get_axis_offsets",
    "_get_linear_index",
    "nd2dp_array",
    "normalize_axis",
    "_object_to_tuple",
    "use_origin_backend"
]

cdef ERROR_PREFIX = "DPNP error:"


def call_origin(function, *args, **kwargs):
    """
    Call fallback function for unsupported cases
    """

    # print(f"DPNP call_origin(): Fallback called. \n\t function={function}, \n\t args={args}, \n\t kwargs={kwargs}")

    kwargs_out = kwargs.get("out", None)
    if (kwargs_out is not None):
        kwargs["out"] = dpnp.asnumpy(kwargs_out) if isinstance(kwargs_out, dparray) else kwargs_out

    args_new = []
    for arg in args:
        argx = dpnp.asnumpy(arg) if isinstance(arg, dparray) else arg
        args_new.append(argx)

    kwargs_new = {}
    for key, kwarg in kwargs.items():
        kwargx = dpnp.asnumpy(kwarg) if isinstance(kwarg, dparray) else kwarg
        kwargs_new[key] = kwargx

    # TODO need to put dparray memory into NumPy call
    result_origin = function(*args_new, **kwargs_new)
    result = result_origin
    if isinstance(result, numpy.ndarray):
        if (kwargs_out is None):
            result_dtype = result_origin.dtype
            kwargs_dtype = kwargs.get("dtype", None)
            if (kwargs_dtype is not None):
                result_dtype = kwargs_dtype

            result = dparray(result_origin.shape, dtype=result_dtype)
        else:
            result = kwargs_out

        for i in range(result.size):
            result._setitem_scalar(i, result_origin.item(i))

    return result


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


cpdef dpnp_descriptor create_output_descriptor_py(dparray_shape_type output_shape, object d_type, object requested_out):
    cdef DPNPFuncType c_type = dpnp_dtype_to_DPNPFuncType(d_type)

    return create_output_descriptor(output_shape, c_type, requested_out)


cpdef dp2nd_array(arr):
    """Convert dparray to ndarray"""
    return dpnp.asnumpy(arr) if isinstance(arr, dparray) else arr

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
            elif elem_dtype == numpy.bool_ or elem_dtype == numpy.bool:
                (< cpp_bool * > dst.get_data())[dst_idx] = elem_value
            elif elem_dtype == numpy.complex128:
                (< cpp_complex[double] * > dst.get_data())[dst_idx] = elem_value
            else:
                checker_throw_type_error("copy_values_to_dparray", elem_dtype)

            dst_idx += 1

    return dst_idx


cpdef tuple get_axis_indeces(idx, shape):
    """
    Compute axis indices of an element in array from array linear index
    """

    ids = []
    remainder = idx
    offsets = get_axis_offsets(shape)
    for i in offsets:
        quotient, remainder = divmod(remainder, i)
        ids.append(quotient)

    return _object_to_tuple(ids)


cpdef tuple get_axis_offsets(shape):
    """
    Compute axis offsets in the linear array memory
    """

    res_size = len(shape)
    result = [0] * res_size
    acc = 1
    for i in range(res_size - 1, -1, -1):
        result[i] = acc
        acc *= shape[i]

    return _object_to_tuple(result)


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


cpdef find_common_type(object x1_obj, object x2_obj):
    cdef bint x1_obj_is_dparray = isinstance(x1_obj, dparray)
    cdef bint x2_obj_is_dparray = isinstance(x2_obj, dparray)

    _, x1_dtype = get_shape_dtype(x1_obj)
    _, x2_dtype = get_shape_dtype(x2_obj)

    cdef list array_types = []
    cdef list scalar_types = []

    if x1_obj_is_dparray:
        array_types.append(x1_dtype)
    else:
        scalar_types.append(x1_dtype)

    if x2_obj_is_dparray:
        array_types.append(x2_dtype)
    else:
        scalar_types.append(x2_dtype)

    return numpy.find_common_type(array_types, scalar_types)


cdef dparray_shape_type get_common_shape(dparray_shape_type input1_shape, dparray_shape_type input2_shape):
    cdef dparray_shape_type result_shape

    # ex (8, 1, 6, 1) and (7, 1, 5) -> (8, 1, 6, 1) and (1, 7, 1, 5)
    cdef size_t max_shape_size = max(input1_shape.size(), input2_shape.size())
    input1_shape.insert(input1_shape.begin(), max_shape_size - input1_shape.size(), 1)
    input2_shape.insert(input2_shape.begin(), max_shape_size - input2_shape.size(), 1)

    # ex result (8, 7, 6, 5)
    for it in range(max_shape_size):
        if input1_shape[it] == input2_shape[it]:
            result_shape.push_back(input1_shape[it])
        elif input1_shape[it] == 1:
            result_shape.push_back(input2_shape[it])
        elif input2_shape[it] == 1:
            result_shape.push_back(input1_shape[it])
        else:
            err_msg = f"{ERROR_PREFIX} in function get_common_shape()"
            err_msg += f"operands could not be broadcast together with shapes {input1_shape} {input2_shape}"
            ValueError(err_msg)

    return result_shape


cdef dparray_shape_type get_reduction_output_shape(dparray_shape_type input_shape, object axis, cpp_bool keepdims):
    cdef dparray_shape_type result_shape
    cdef tuple axis_tuple = _object_to_tuple(axis)

    if axis is not None:
        for it in range(input_shape.size()):
            if it not in axis_tuple:
                result_shape.push_back(input_shape[it])
            elif keepdims is True:
                result_shape.push_back(1)
    elif keepdims is True:
        for it in range(input_shape.size()):
            result_shape.push_back(1)

    return result_shape


cdef DPNPFuncType get_output_c_type(DPNPFuncName funcID,
                                    DPNPFuncType input_array_c_type,
                                    object requested_out,
                                    object requested_dtype):

    if requested_out is None:
        if requested_dtype is None:
            """ get recommended result type by function ID """
            kernel_data = get_dpnp_function_ptr(funcID, input_array_c_type, input_array_c_type)
            return kernel_data.return_type
        else:
            """ return type by request """
            return dpnp_dtype_to_DPNPFuncType(requested_dtype)
    else:
        if requested_dtype is None:
            """ determined by 'out' parameter """
            return dpnp_dtype_to_DPNPFuncType(requested_out.dtype)

    checker_throw_value_error("get_output_c_type", "dtype and out", requested_dtype, requested_out)


cdef dparray create_output_array(dparray_shape_type output_shape, DPNPFuncType c_type, object requested_out):
    """
    TODO This function needs to be deleted. Replace with create_output_descriptor()
    """

    cdef dparray result

    if requested_out is None:
        """ Create DPNP array """
        result = dparray(output_shape, dtype=dpnp_DPNPFuncType_to_dtype( < size_t > c_type))
    else:
        """ Based on 'out' parameter """
        if (output_shape != requested_out.shape):
            checker_throw_value_error("create_output_array", "out.shape", requested_out.shape, output_shape)
        result = requested_out

    return result

cdef dpnp_descriptor create_output_descriptor(dparray_shape_type output_shape,
                                              DPNPFuncType c_type,
                                              dpnp_descriptor requested_out):
    cdef dpnp_descriptor result_desc

    if requested_out is None:
        """ Create DPNP array """
        result = dparray(output_shape, dtype=dpnp_DPNPFuncType_to_dtype( < size_t > c_type))
        result_desc = dpnp_descriptor(result)
    else:
        """ Based on 'out' parameter """
        if (output_shape != requested_out.shape):
            checker_throw_value_error("create_output_array", "out.shape", requested_out.shape, output_shape)

        if isinstance(requested_out, dpnp_descriptor):
            result_desc = requested_out
        else:
            result_desc = dpnp_descriptor(requested_out)

    return result_desc


cpdef nd2dp_array(arr):
    """Convert ndarray to dparray"""
    if not isinstance(arr, numpy.ndarray):
        return arr

    result = dparray(arr.shape, dtype=arr.dtype)
    for i in range(result.size):
        result._setitem_scalar(i, arr.item(i))

    return result


cpdef dparray_shape_type normalize_axis(object axis_obj, size_t shape_size_inp):
    """
    Conversion of the transformation shape axis [-1, 0, 1] into [2, 0, 1] where numbers are `id`s of array shape axis
    """

    cdef dparray_shape_type axis = _object_to_tuple(axis_obj)  # axis_obj might be a scalar
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
            raise ValueError("DPNP _normalize_order(): order \'K\' is not permitted")
        order_type = b'K'
    elif order_type == b'A' or order_type == b'a':
        order_type = b'A'
    elif order_type == b'C' or order_type == b'c':
        order_type = b'C'
    elif order_type == b'F' or order_type == b'f':
        order_type = b'F'
    else:
        raise TypeError("DPNP _normalize_order(): order is not understood")

    return order_type


@cython.profile(False)
cpdef inline tuple _object_to_tuple(object obj):
    """ Converts Python object into tuple

    """

    if obj is None:
        return ()

    if cpython.PySequence_Check(obj):
        return tuple(obj)

    if dpnp.isscalar(obj):
        return (obj, )

    raise ValueError("DPNP object_to_tuple(): 'obj' should be 'None', collections.abc.Sequence, or 'int'")


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


cdef class dpnp_descriptor:
    def __init__(self, obj):
        """ Initialze variables """
        self.origin_pyobj = None
        self.descriptor = None
        self.dpnp_descriptor_data_size = 0
        self.dpnp_descriptor_is_scalar = True

        """ Accure main data storage """
        self.descriptor = getattr(obj, "__array_interface__", None)
        if self.descriptor is None:
            return

        if self.descriptor["version"] != 3:
            return

        self.origin_pyobj = obj

        """ array size calculation """
        cdef Py_ssize_t shape_it = 0
        self.dpnp_descriptor_data_size = 1
        for shape_it in self.shape:
            # TODO need to use common procedure from utils to calculate array size by shape
            if shape_it < 0:
                raise ValueError(f"{ERROR_PREFIX} dpnp_descriptor::__init__() invalid value {shape_it} in 'shape'")
            self.dpnp_descriptor_data_size *= shape_it

        """ set scalar property """
        self.dpnp_descriptor_is_scalar = False

    @property
    def is_valid(self):
        if self.descriptor is None:
            return False
        return True

    @property
    def shape(self):
        if self.is_valid:
            return self.descriptor["shape"]
        return None

    @property
    def strides(self):
        if self.is_valid:
            return self.descriptor["strides"]
        return None

    @property
    def ndim(self):
        if self.is_valid:
            return len(self.shape)
        return 0

    @property
    def size(self):
        if self.is_valid:
            return self.dpnp_descriptor_data_size
        return 0

    @property
    def dtype(self):
        if self.is_valid:
            type_str = self.descriptor["typestr"]
            return dpnp.dtype(type_str)
        return None

    @property
    def is_scalar(self):
        return self.dpnp_descriptor_is_scalar

    @property
    def data(self):
        if self.is_valid:
            data_tuple = self.descriptor["data"]
            return data_tuple[0]
        return None

    @property
    def descr(self):
        if self.is_valid:
            return self.descriptor["descr"]
        return None

    @property
    def __array_interface__(self):
        # print(f"====dpnp_descriptor::__array_interface__====self.descriptor={ < size_t > self.descriptor}")
        if self.descriptor is None:
            return None

        # TODO need to think about feature compatibility
        interface_dict = {
            "data": self.descriptor["data"],
            "strides": self.descriptor["strides"],
            "descr": self.descriptor["descr"],
            "typestr": self.descriptor["typestr"],
            "shape": self.descriptor["shape"],
            "version": self.descriptor["version"]
        }

        return interface_dict

    def get_pyobj(self):
        return self.origin_pyobj

    cdef void * get_data(self):
        cdef long val = self.data
        return < void * > val

    def __bool__(self):
        return self.is_valid

    def __str__(self):
        return str(self.descriptor)
