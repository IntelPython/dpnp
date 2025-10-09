# cython: language_level=3
# cython: linetrace=True
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

This module contains different helpers and utilities

"""

import dpctl
import dpctl.utils as dpu
import numpy

import dpnp
import dpnp.config as config
import dpnp.dpnp_container as dpnp_container
from dpnp.dpnp_array import dpnp_array

cimport cpython
cimport cython
cimport numpy
from libcpp cimport bool as cpp_bool

from dpnp.dpnp_algo.dpnp_algo cimport (
    dpnp_DPNPFuncType_to_dtype,
)

"""
Python import functions
"""
__all__ = [
    "call_origin",
    "checker_throw_type_error",
    "checker_throw_value_error",
    "convert_item",
    "dpnp_descriptor",
    "get_usm_allocations",
    "map_dtype_to_device",
    "_object_to_tuple",
    "unwrap_array",
    "use_origin_backend"
]

cdef ERROR_PREFIX = "DPNP error:"


def convert_item(item):
    if hasattr(item, "__sycl_usm_array_interface__"):
        item_converted = dpnp.asnumpy(item)
    elif hasattr(item, "__array_interface__"):  # detect if it is a container (TODO any better way?)
        mod_name = getattr(item, "__module__", 'none')
        if (mod_name != 'numpy'):
            item_converted = dpnp.asnumpy(item)
        else:
            item_converted = item
    elif isinstance(item, list):
        item_converted = convert_list_args(item)
    elif isinstance(item, tuple):
        item_converted = tuple(convert_list_args(item))
    else:
        item_converted = item

    return item_converted


def convert_list_args(input_list):
    result_list = []
    for item in input_list:
        item_converted = convert_item(item)
        result_list.append(item_converted)

    return result_list


def copy_from_origin(dst, src):
    """Copy origin result to output result."""
    if hasattr(dst, "__sycl_usm_array_interface__"):
        if src.size:
            dst_dpt = unwrap_array(dst)
            dst_dpt[...] = src
    else:
        for i in range(dst.size):
            dst.flat[i] = src.item(i)


def call_origin(function, *args, **kwargs):
    """
    Call fallback function for unsupported cases
    """

    allow_fallback = kwargs.pop("allow_fallback", False)

    if not allow_fallback and config.__DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK__ == 1:
        raise NotImplementedError(f"Requested function={function.__name__} with args={args} and kwargs={kwargs} "
                                   "isn't currently supported and would fall back on NumPy implementation. "
                                   "Define environment variable `DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK` to `0` "
                                   "if the fall back is required to be supported without raising an exception.")

    dpnp_inplace = kwargs.pop("dpnp_inplace", False)
    sycl_queue = kwargs.pop("sycl_queue", None)
    # print(f"DPNP call_origin(): Fallback called. \n\t function={function}, \n\t args={args}, \n\t kwargs={kwargs}, \n\t dpnp_inplace={dpnp_inplace}")

    kwargs_out = kwargs.get("out", None)
    alloc_queues = [sycl_queue] if sycl_queue else []
    if (kwargs_out is not None):
        if isinstance(kwargs_out, numpy.ndarray):
            kwargs["out"] = kwargs_out
        else:
            if hasattr(kwargs_out, "sycl_queue"):
                alloc_queues.append(kwargs_out.sycl_queue)
            kwargs["out"] = dpnp.asnumpy(kwargs_out)

    args_new_list = []
    for arg in args:
        if hasattr(arg, "sycl_queue"):
            alloc_queues.append(arg.sycl_queue)
        argx = convert_item(arg)
        args_new_list.append(argx)
    args_new = tuple(args_new_list)

    kwargs_new = {}
    for key, kwarg in kwargs.items():
        if hasattr(kwarg, "sycl_queue"):
            alloc_queues.append(kwarg.sycl_queue)
        kwargx = convert_item(kwarg)
        kwargs_new[key] = kwargx

    exec_q = dpu.get_execution_queue(alloc_queues)
    if exec_q is None:
        exec_q = dpnp.get_normalized_queue_device(sycl_queue=sycl_queue)
    # print(f"DPNP call_origin(): backend called. \n\t function={function}, \n\t args_new={args_new}, \n\t kwargs_new={kwargs_new}, \n\t dpnp_inplace={dpnp_inplace}")
    # TODO need to put array memory into NumPy call
    result_origin = function(*args_new, **kwargs_new)
    # print(f"DPNP call_origin(): result from backend. \n\t result_origin={result_origin}, \n\t args_new={args_new}, \n\t kwargs_new={kwargs_new}, \n\t dpnp_inplace={dpnp_inplace}")
    result = result_origin
    if dpnp_inplace:
        # enough to modify only first argument in place
        if args and args_new:
            arg, arg_new = args[0], args_new[0]
            if isinstance(arg_new, numpy.ndarray):
                copy_from_origin(arg, arg_new)
            elif isinstance(arg_new, list):
                for i, val in enumerate(arg_new):
                    arg[i] = val

    elif isinstance(result, numpy.ndarray):
        if kwargs_out is None:
            # use dtype from input arguments if present or from the result otherwise
            result_dtype = kwargs.get("dtype", None) or result_origin.dtype

            if exec_q is not None:
                result_dtype = map_dtype_to_device(result_origin.dtype, exec_q.sycl_device)

            result = dpnp_container.empty(result_origin.shape, dtype=result_dtype, sycl_queue=exec_q)
        else:
            result = kwargs_out

        copy_from_origin(result, result_origin)

    elif isinstance(result, tuple):
        # convert tuple(fallback_array) to tuple(result_array)
        result_list = []
        for res_origin in result:
            res = res_origin
            if isinstance(res_origin, numpy.ndarray):
                if exec_q is not None:
                    result_dtype = map_dtype_to_device(res_origin.dtype, exec_q.sycl_device)
                else:
                    result_dtype = res_origin.d_type
                res = dpnp_container.empty(res_origin.shape, dtype=result_dtype, sycl_queue=exec_q)
                copy_from_origin(res, res_origin)
            result_list.append(res)

        result = tuple(result_list)

    return result


def unwrap_array(x1):
    """
    Get array from input object.
    """
    if isinstance(x1, dpnp_array):
        return x1.get_array()

    return x1


def _get_coerced_usm_type(objects):
    types_in_use = [obj.usm_type for obj in objects if hasattr(obj, "usm_type")]
    if len(types_in_use) == 0:
        return None
    elif len(types_in_use) == 1:
        return types_in_use[0]

    common_usm_type = dpu.get_coerced_usm_type(types_in_use)
    if common_usm_type is None:
        raise ValueError("Input arrays must have coerced USM types")
    return common_usm_type


def _get_common_allocation_queue(objects):
    queues_in_use = [obj.sycl_queue for obj in objects if hasattr(obj, "sycl_queue")]
    if len(queues_in_use) == 0:
        return None
    elif len(queues_in_use) == 1:
        return queues_in_use[0]

    common_queue = dpu.get_execution_queue(queues_in_use)
    if common_queue is None:
        raise ValueError("Input arrays must be allocated on the same SYCL queue")
    return common_queue


def get_usm_allocations(objects):
    """
    Given a list of objects returns a tuple of USM type and SYCL queue
    which can be used for a memory allocation and to follow compute follows data paradigm,
    or returns `(None, None)` if the default USM type and SYCL queue can be used.
    An exception will be raised, if the paradigm is broken for the given list of objects.

    """

    if not isinstance(objects, (list, tuple)):
        raise TypeError("Expected a list or a tuple, got {}".format(type(objects)))

    if len(objects) == 0:
        return (None, None)
    return (_get_coerced_usm_type(objects), _get_common_allocation_queue(objects))


def map_dtype_to_device(dtype, device):
    """
    Map an input ``dtype`` with type ``device`` may use
    """

    dtype = dpnp.dtype(dtype)
    if not hasattr(dtype, 'char'):
        raise TypeError(f"Invalid type of input dtype={dtype}")
    elif not isinstance(device, dpctl.SyclDevice):
        raise TypeError(f"Invalid type of input device={device}")

    dtc = dtype.char
    if dtc == "?" or dpnp.issubdtype(dtype, dpnp.integer):
        # bool or integer type
        return dtype

    if dpnp.issubdtype(dtype, dpnp.floating):
        if dtc == "f":
            # float32 type
            return dtype
        elif dtc == "d":
            # float64 type
            if device.has_aspect_fp64:
                return dtype
        elif dtc == "e":
            # float16 type
            if device.has_aspect_fp16:
                return dtype
        # float32 is default floating type
        return dpnp.dtype("f4")

    if dpnp.issubdtype(dtype, dpnp.complexfloating):
        if dtc == "F":
            # complex64 type
            return dtype
        elif dtc == "D":
            # complex128 type
            if device.has_aspect_fp64:
                return dtype
        # complex64 is default complex type
        return dpnp.dtype("c8")
    raise RuntimeError(f"Unrecognized type of input dtype={dtype}")


cpdef checker_throw_type_error(function_name, given_type):
    raise TypeError(f"{ERROR_PREFIX} in function {function_name}() type '{given_type}' is not supported")


cpdef checker_throw_value_error(function_name, param_name, param, expected):
    # import sys
    # sys.tracebacklimit = 0
    err_msg = f"{ERROR_PREFIX} in function {function_name}() parameter '{param_name}'"
    err_msg += f" expected `{expected}`, but '{param}' provided"
    raise ValueError(err_msg)


cdef dpnp_descriptor create_output_descriptor(shape_type_c output_shape,
                                              DPNPFuncType c_type,
                                              dpnp_descriptor requested_out,
                                              device=None,
                                              usm_type="device",
                                              sycl_queue=None):
    cdef dpnp_descriptor result_desc

    if requested_out is None:
        result = None
        if sycl_queue is not None:
            device = None
        result_dtype = dpnp_DPNPFuncType_to_dtype(< size_t > c_type)
        result_obj = dpnp_container.empty(output_shape,
                                          dtype=result_dtype,
                                          device=device,
                                          usm_type=usm_type,
                                          sycl_queue=sycl_queue)
        result_desc = dpnp_descriptor(result_obj)
    else:
        """ Based on 'out' parameter """
        if (output_shape != requested_out.shape):
            checker_throw_value_error("create_output_descriptor", "out.shape", requested_out.shape, output_shape)

        if isinstance(requested_out, dpnp_descriptor):
            result_desc = requested_out
        else:
            result_desc = dpnp_descriptor(requested_out)

    return result_desc


@cython.profile(False)
cpdef inline tuple _object_to_tuple(object obj):
    """ Converts Python object into tuple

    """

    if obj is None:
        return ()

    # dpnp.ndarray unconditionally succeeds in PySequence_Check as it implements __getitem__
    if cpython.PySequence_Check(obj) and not dpnp.is_supported_array_type(obj):
        if isinstance(obj, numpy.ndarray):
            obj = numpy.atleast_1d(obj)

        nd = len(obj)
        shape = []

        for i in range(0, nd):
            if cpython.PyBool_Check(obj[i]):
                raise TypeError("DPNP object_to_tuple(): no item in size can be bool")

            # Assumes each item is castable to Py_ssize_t,
            # otherwise TypeError will be raised
            shape.append(<Py_ssize_t> obj[i])
        return tuple(shape)

    if dpnp.isscalar(obj):
        if cpython.PyBool_Check(obj):
            raise TypeError("DPNP object_to_tuple(): 'obj' can't be bool")
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


cdef tuple get_common_usm_allocation(dpnp_descriptor x1, dpnp_descriptor x2):
    """Get common USM allocation in the form of (sycl_device, usm_type, sycl_queue)."""
    array1_obj = x1.get_array()
    array2_obj = x2.get_array()

    common_usm_type = dpctl.utils.get_coerced_usm_type((array1_obj.usm_type, array2_obj.usm_type))
    if common_usm_type is None:
        raise ValueError(
            "could not recognize common USM type for inputs of USM types {} and {}"
            "".format(array1_obj.usm_type, array2_obj.usm_type))

    common_sycl_queue = dpu.get_execution_queue((array1_obj.sycl_queue, array2_obj.sycl_queue))
    if common_sycl_queue is None:
        raise ValueError(
            "could not recognize common SYCL queue for inputs in SYCL queues {} and {}"
            "".format(array1_obj.sycl_queue, array2_obj.sycl_queue))

    return (common_sycl_queue.sycl_device, common_usm_type, common_sycl_queue)


cdef class dpnp_descriptor:
    def __init__(self, obj):
        """ Initialize variables """
        self.origin_pyobj = None
        self.descriptor = None
        self.dpnp_descriptor_data_size = 0
        self.dpnp_descriptor_is_scalar = True

        """ Acquire DPCTL data container storage """
        self.descriptor = getattr(obj, "__sycl_usm_array_interface__", None)
        if self.descriptor is None:

            """ Acquire main data storage """
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
    def offset(self):
        if self.is_valid:
            return self.descriptor.get('offset', 0)
        return 0

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

    def get_array(self):
        if isinstance(self.origin_pyobj, dpctl.tensor.usm_ndarray):
            return self.origin_pyobj
        if isinstance(self.origin_pyobj, dpnp_array):
            return self.origin_pyobj.get_array()

        raise TypeError(
            "expected either dpctl.tensor.usm_ndarray or dpnp.dpnp_array.dpnp_array, got {}"
            "".format(type(self.origin_pyobj)))

    cdef void * get_data(self):
        cdef Py_ssize_t item_size = 0
        cdef Py_ssize_t elem_offset = 0
        cdef char *data_ptr = NULL
        cdef size_t val = self.data

        if self.offset > 0:
            item_size = self.origin_pyobj.itemsize
            elem_offset = self.offset
            data_ptr = <char *>(val) + item_size * elem_offset
            return < void * > data_ptr

        return < void * > val

    def __bool__(self):
        return self.is_valid

    def __str__(self):
        return str(self.descriptor)
