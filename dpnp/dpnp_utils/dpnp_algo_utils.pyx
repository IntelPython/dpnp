# cython: language_level=3
# cython: linetrace=True
# cython: freethreading_compatible = True
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
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

import dpnp
import dpnp.config as config
import dpnp.dpnp_container as dpnp_container
from dpnp.dpnp_array import dpnp_array
from dpnp.tensor import get_coerced_usm_type, get_execution_queue

from libcpp cimport bool as cpp_bool

from dpnp.dpnp_algo.dpnp_algo cimport (
    dpnp_DPNPFuncType_to_dtype,
)

# Python import functions
__all__ = [
    "checker_throw_type_error",
    "checker_throw_value_error",
    "convert_item",
    "dpnp_descriptor",
    "get_usm_allocations",
    "map_dtype_to_device",
    "use_origin_backend"
]

cdef ERROR_PREFIX = "DPNP error:"


def convert_item(item):
    if hasattr(item, "__sycl_usm_array_interface__"):
        item_converted = dpnp.asnumpy(item)
    elif hasattr(item, "__array_interface__"):
        # detect if it is a container (TODO any better way?)
        mod_name = getattr(item, "__module__", "none")
        if (mod_name != "numpy"):
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


def _get_coerced_usm_type(objects):
    types_in_use = [
        obj.usm_type for obj in objects
        if hasattr(obj, "usm_type")
    ]
    if len(types_in_use) == 0:
        return None
    elif len(types_in_use) == 1:
        return types_in_use[0]

    common_usm_type = get_coerced_usm_type(types_in_use)
    if common_usm_type is None:
        raise ValueError(
            "Input arrays must have coerced USM types"
        )
    return common_usm_type


def _get_common_allocation_queue(objects):
    queues_in_use = [
        obj.sycl_queue for obj in objects
        if hasattr(obj, "sycl_queue")
    ]
    if len(queues_in_use) == 0:
        return None
    elif len(queues_in_use) == 1:
        return queues_in_use[0]

    common_queue = get_execution_queue(queues_in_use)
    if common_queue is None:
        raise ValueError(
            "Input arrays must be allocated "
            "on the same SYCL queue"
        )
    return common_queue


def get_usm_allocations(objects):
    """
    Given a list of objects returns a tuple of USM type
    and SYCL queue which can be used for a memory
    allocation and to follow compute follows data
    paradigm, or returns `(None, None)` if the default
    USM type and SYCL queue can be used.
    An exception will be raised, if the paradigm is
    broken for the given list of objects.

    """

    if not isinstance(objects, (list, tuple)):
        raise TypeError(
            f"Expected a list or a tuple, got {type(objects)}"
        )

    if len(objects) == 0:
        return (None, None)
    return (
        _get_coerced_usm_type(objects),
        _get_common_allocation_queue(objects),
    )


def map_dtype_to_device(dtype, device):
    """
    Map an input ``dtype`` with type ``device`` may use
    """

    dtype = dpnp.dtype(dtype)
    if not hasattr(dtype, "char"):
        raise TypeError(
            f"Invalid type of input dtype={dtype}"
        )
    elif not isinstance(device, dpctl.SyclDevice):
        raise TypeError(
            f"Invalid type of input device={device}"
        )

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
    raise RuntimeError(
        f"Unrecognized type of input dtype={dtype}"
    )


cpdef checker_throw_type_error(function_name, given_type):
    raise TypeError(
        f"{ERROR_PREFIX} in function {function_name}() "
        f"type '{given_type}' is not supported"
    )


cpdef checker_throw_value_error(
    function_name, param_name, param, expected
):
    # import sys
    # sys.tracebacklimit = 0
    err_msg = (
        f"{ERROR_PREFIX} in function {function_name}() "
        f"parameter '{param_name}'"
    )
    err_msg += f" expected `{expected}`, but '{param}' provided"
    raise ValueError(err_msg)


cdef dpnp_descriptor create_output_descriptor(
    shape_type_c output_shape,
    DPNPFuncType c_type,
    dpnp_descriptor requested_out,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    cdef dpnp_descriptor result_desc

    if requested_out is None:
        _result = None
        if sycl_queue is not None:
            device = None
        result_dtype = dpnp_DPNPFuncType_to_dtype(
            < size_t > c_type
        )
        result_obj = dpnp_container.empty(
            output_shape,
            dtype=result_dtype,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
        result_desc = dpnp_descriptor(result_obj)
    else:
        """ Based on 'out' parameter """
        if (output_shape != requested_out.shape):
            checker_throw_value_error(
                "create_output_descriptor",
                "out.shape",
                requested_out.shape,
                output_shape,
            )

        if isinstance(requested_out, dpnp_descriptor):
            result_desc = requested_out
        else:
            result_desc = dpnp_descriptor(requested_out)

    return result_desc


cpdef cpp_bool use_origin_backend(
    input1=None, size_t compute_size=0
):
    """
    This function needs to redirect particular
    computation cases to original backend.

    Parameters:
        input1: One of the input parameter of the API
            function
        compute_size: Some amount of total compute size
            of the task
    Return:
        True - computations are better to be executed on
            original backend
        False - it is better to use this SW to compute
    """

    if (config.__DPNP_ORIGIN__):
        return True

    return False


cdef class dpnp_descriptor:
    def __init__(self, obj):
        """ Initialize variables """
        self.origin_pyobj = None
        self.descriptor = None
        self.dpnp_descriptor_data_size = 0
        self.dpnp_descriptor_is_scalar = True

        """ Acquire DPCTL data container storage """
        self.descriptor = getattr(
            obj, "__sycl_usm_array_interface__", None
        )
        if self.descriptor is None:

            """ Acquire main data storage """
            self.descriptor = getattr(
                obj, "__array_interface__", None
            )
            if self.descriptor is None:
                return

            if self.descriptor["version"] != 3:
                return

        self.origin_pyobj = obj

        """ array size calculation """
        cdef Py_ssize_t shape_it = 0
        self.dpnp_descriptor_data_size = 1
        for shape_it in self.shape:
            # TODO need to use common procedure from
            # utils to calculate array size by shape
            if shape_it < 0:
                raise ValueError(
                    f"{ERROR_PREFIX} "
                    "dpnp_descriptor::__init__() "
                    f"invalid value {shape_it} "
                    "in 'shape'"
                )
            self.dpnp_descriptor_data_size *= shape_it

        # set scalar property
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
            return self.descriptor.get("offset", 0)
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
        # print("====dpnp_descriptor::"
        #       "__array_interface__===="
        #       f"self.descriptor="
        #       f"{<size_t>self.descriptor}")
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
        if isinstance(
            self.origin_pyobj, dpnp.tensor.usm_ndarray
        ):
            return self.origin_pyobj
        if isinstance(self.origin_pyobj, dpnp_array):
            return self.origin_pyobj.get_array()

        raise TypeError(
            "expected either dpnp.tensor.usm_ndarray "
            f"or dpnp.dpnp_array.dpnp_array, got {type(self.origin_pyobj)}"
        )

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
