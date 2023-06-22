# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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

"""
Interface of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import collections
import os

import dpctl
import dpctl.tensor as dpt
import numpy
import numpy.lib.stride_tricks as np_st

import dpnp.config as config
from dpnp.dpnp_algo import *
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import *
from dpnp.fft import *
from dpnp.linalg import *
from dpnp.random import *

__all__ = [
    "array_equal",
    "asnumpy",
    "astype",
    "convert_single_elem_array_to_scalar",
    "default_float_type",
    "dpnp_queue_initialize",
    "dpnp_queue_is_cpu",
    "from_dlpack",
    "get_dpnp_descriptor",
    "get_include",
    "get_normalized_queue_device",
    "get_usm_ndarray",
    "get_usm_ndarray_or_scalar",
    "is_supported_array_type",
]

from dpnp import float64, isscalar
from dpnp.dpnp_iface_arraycreation import *
from dpnp.dpnp_iface_arraycreation import __all__ as __all__arraycreation
from dpnp.dpnp_iface_bitwise import *
from dpnp.dpnp_iface_bitwise import __all__ as __all__bitwise
from dpnp.dpnp_iface_counting import *
from dpnp.dpnp_iface_counting import __all__ as __all__counting
from dpnp.dpnp_iface_indexing import *
from dpnp.dpnp_iface_indexing import __all__ as __all__indexing
from dpnp.dpnp_iface_libmath import *
from dpnp.dpnp_iface_libmath import __all__ as __all__libmath
from dpnp.dpnp_iface_linearalgebra import *
from dpnp.dpnp_iface_linearalgebra import __all__ as __all__linearalgebra
from dpnp.dpnp_iface_logic import *
from dpnp.dpnp_iface_logic import __all__ as __all__logic
from dpnp.dpnp_iface_manipulation import *
from dpnp.dpnp_iface_manipulation import __all__ as __all__manipulation
from dpnp.dpnp_iface_mathematical import *
from dpnp.dpnp_iface_mathematical import __all__ as __all__mathematical
from dpnp.dpnp_iface_searching import *
from dpnp.dpnp_iface_searching import __all__ as __all__searching
from dpnp.dpnp_iface_sorting import *
from dpnp.dpnp_iface_sorting import __all__ as __all__sorting
from dpnp.dpnp_iface_statistics import *
from dpnp.dpnp_iface_statistics import __all__ as __all__statistics
from dpnp.dpnp_iface_trigonometric import *
from dpnp.dpnp_iface_trigonometric import __all__ as __all__trigonometric

__all__ += __all__arraycreation
__all__ += __all__bitwise
__all__ += __all__counting
__all__ += __all__indexing
__all__ += __all__libmath
__all__ += __all__linearalgebra
__all__ += __all__logic
__all__ += __all__manipulation
__all__ += __all__mathematical
__all__ += __all__searching
__all__ += __all__sorting
__all__ += __all__statistics
__all__ += __all__trigonometric


def array_equal(a1, a2, equal_nan=False):
    """
    True if two arrays have the same shape and elements, False otherwise.

    For full documentation refer to :obj:`numpy.array_equal`.

    See Also
    --------
    :obj:`dpnp.allclose` : Returns True if two arrays are element-wise equal
                           within a tolerance.
    :obj:`dpnp.array_equiv` : Returns True if input arrays are shape consistent
                              and all elements equal.

    """

    return numpy.array_equal(a1, a2)


def asnumpy(input, order="C"):
    """
    Returns the NumPy array with input data.

    Notes
    -----
    This function works exactly the same as :obj:`numpy.asarray`.

    """
    if isinstance(input, dpnp_array):
        return input.asnumpy()

    if isinstance(input, dpt.usm_ndarray):
        return dpt.asnumpy(input)

    return numpy.asarray(input, order=order)


def astype(x1, dtype, order="K", casting="unsafe", subok=True, copy=True):
    """Copy the array with data type casting."""
    if isinstance(x1, dpnp_array):
        return x1.astype(dtype, order=order, casting=casting, copy=copy)

    if isinstance(x1, dpt.usm_ndarray):
        return dpt.astype(x1, dtype, order=order, casting=casting, copy=copy)

    x1_desc = get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if not x1_desc:
        pass
    elif order != "K":
        pass
    elif casting != "unsafe":
        pass
    elif not subok:
        pass
    elif not copy:
        pass
    elif x1_desc.dtype == numpy.complex128 or dtype == numpy.complex128:
        pass
    elif x1_desc.dtype == numpy.complex64 or dtype == numpy.complex64:
        pass
    else:
        return dpnp_astype(x1_desc, dtype).get_pyobj()

    return call_origin(
        numpy.ndarray.astype,
        x1,
        dtype,
        order=order,
        casting=casting,
        subok=subok,
        copy=copy,
    )


def convert_single_elem_array_to_scalar(obj, keepdims=False):
    """
    Convert array with single element to scalar
    """

    if (obj.ndim > 0) and (obj.size == 1) and (keepdims is False):
        return obj.dtype.type(obj[0])

    return obj


def default_float_type(device=None, sycl_queue=None):
    """
    Return a floating type used by default in DPNP depending on device capabilities.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where an array of default floating type might be created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
        The value ``None`` is interpreted as to use a default device.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue which might be used to create an array of default floating type.
        The `sycl_queue` can be ``None`` (the default), which is interpreted as
        to get the SYCL queue from `device` keyword if present or to use a default queue.

    Returns
    -------
    dt : dtype
        A default DPNP floating type.

    """

    _sycl_queue = get_normalized_queue_device(
        device=device, sycl_queue=sycl_queue
    )
    return map_dtype_to_device(float64, _sycl_queue.sycl_device)


def from_dlpack(obj, /):
    """
    Create a dpnp array from a Python object implementing the ``__dlpack__``
    protocol.

    See https://dmlc.github.io/dlpack/latest/ for more details.

    Parameters
    ----------
    obj : object
        A Python object representing an array that implements the ``__dlpack__``
        and ``__dlpack_device__`` methods.

    Returns
    -------
    out : dpnp_array
        Returns a new dpnp array containing the data from another array
        (obj) with the ``__dlpack__`` method on the same device as object.

    """

    usm_ary = dpt.from_dlpack(obj)
    return dpnp_array._create_from_usm_ndarray(usm_ary)


def get_dpnp_descriptor(
    ext_obj,
    copy_when_strides=True,
    copy_when_nondefault_queue=True,
    alloc_dtype=None,
    alloc_usm_type=None,
    alloc_queue=None,
):
    """
    Return True:
      never
    Return DPNP internal data discriptor object if:
      1. We can proceed with input data object with DPNP
      2. We want to handle input data object
    Return False if:
      1. We do not want to work with input data object
      2. We can not handle with input data object
    """

    # TODO need to allow "import dpnp" with no build procedure
    # if no_modules_load_doc_build();
    #    return False

    if use_origin_backend():
        return False

    # It's required to keep track of input object if a non-strided copy is going to be created.
    # Thus there will be an extra descriptor allocated to refer on original input.
    orig_desc = None

    # If input object is a scalar, it means it was allocated on host memory.
    # We need to copy it to USM memory according to compute follows data paradigm.
    if isscalar(ext_obj):
        ext_obj = array(
            ext_obj,
            dtype=alloc_dtype,
            usm_type=alloc_usm_type,
            sycl_queue=alloc_queue,
        )

    # while dpnp functions have no implementation with strides support
    # we need to create a non-strided copy
    # if function get implementation for strides case
    # then this behavior can be disabled with setting "copy_when_strides"
    if copy_when_strides and getattr(ext_obj, "strides", None) is not None:
        # TODO: replace this workaround when usm_ndarray will provide such functionality
        shape_offsets = tuple(
            numpy.prod(ext_obj.shape[i + 1 :], dtype=numpy.int64)
            for i in range(ext_obj.ndim)
        )

        if hasattr(ext_obj, "__sycl_usm_array_interface__"):
            ext_obj_offset = ext_obj.__sycl_usm_array_interface__.get(
                "offset", 0
            )
        else:
            ext_obj_offset = 0

        if ext_obj.strides != shape_offsets or ext_obj_offset != 0:
            orig_desc = dpnp_descriptor(ext_obj)
            ext_obj = array(ext_obj)

    # while dpnp functions are based on DPNP_QUEUE
    # we need to create a copy on device associated with DPNP_QUEUE
    # if function get implementation for different queue
    # then this behavior can be disabled with setting "copy_when_nondefault_queue"
    queue = getattr(ext_obj, "sycl_queue", None)
    if queue is not None and copy_when_nondefault_queue:
        default_queue = dpctl.SyclQueue()
        queue_is_default = (
            dpctl.utils.get_execution_queue([queue, default_queue]) is not None
        )
        if not queue_is_default:
            ext_obj = array(ext_obj, sycl_queue=default_queue)

    dpnp_desc = dpnp_descriptor(ext_obj, orig_desc)
    if dpnp_desc.is_valid:
        return dpnp_desc

    return False


def get_include():
    """
    Return the directory that contains the DPNP C++ backend \\*.h header files.
    """

    dpnp_path = os.path.join(os.path.dirname(__file__), "backend", "include")

    return dpnp_path


def get_normalized_queue_device(obj=None, device=None, sycl_queue=None):
    """
    Utility to process complementary keyword arguments 'device' and 'sycl_queue'
    in subsequent calls of functions from `dpctl.tensor` module.

    If both arguments 'device' and 'sycl_queue' have default value `None`
    and 'obj' has `sycl_queue` attribute, it assumes that Compute Follows Data
    approach has to be applied and so the resulting SYCL queue will be normalized
    based on the queue value from 'obj'.

    Args:
        obj (optional): A python object. Can be an instance of `dpnp_array`,
            `dpctl.tensor.usm_ndarray`, an object representing SYCL USM allocation
            and implementing `__sycl_usm_array_interface__` protocol,
            an instance of `numpy.ndarray`, an object supporting Python buffer protocol,
            a Python scalar, or a (possibly nested) sequence of Python scalars.
        sycl_queue (:class:`dpctl.SyclQueue`, optional):
            explicitly indicates where USM allocation is done
            and the population code (if any) is executed.
            Value `None` is interpreted as get the SYCL queue
            from `obj` parameter if not None, from `device` keyword,
            or use default queue.
            Default: None
        device (string, :class:`dpctl.SyclDevice`, :class:`dpctl.SyclQueue,
            :class:`dpctl.tensor.Device`, optional):
            array-API keyword indicating non-partitioned SYCL device
            where array is allocated.
    Returns
        :class:`dpctl.SyclQueue` object normalized by `normalize_queue_device` call
        of `dpctl.tensor` module invoked with 'device' and 'sycl_queue' values.
        If both incoming 'device' and 'sycl_queue' are None and 'obj' has `sycl_queue` attribute,
        the normalization will be performed for 'obj.sycl_queue' value.
    Raises:
        TypeError: if argument is not of the expected type, or keywords
            imply incompatible queues.
    """
    if (
        device is None
        and sycl_queue is None
        and obj is not None
        and hasattr(obj, "sycl_queue")
    ):
        sycl_queue = obj.sycl_queue

    # TODO: remove check dpt._device has attribute 'normalize_queue_device'
    if hasattr(dpt._device, "normalize_queue_device"):
        return dpt._device.normalize_queue_device(
            sycl_queue=sycl_queue, device=device
        )
    return sycl_queue


def get_usm_ndarray(a):
    """
    Return :class:`dpctl.tensor.usm_ndarray` from input array `a`.

    Parameters
    ----------
    a : {dpnp_array, usm_ndarray}
        Input array of supported type :class:`dpnp.ndarray`
        or :class:`dpctl.tensor.usm_ndarray`.

    Returns
    -------
    out : usm_ndarray
        A dpctl USM ndarray of input array `a`.

    Raises
    ------
    TypeError
        If input parameter `a` is of unsupported array type.

    """

    if isinstance(a, dpnp_array):
        return a.get_array()
    if isinstance(a, dpt.usm_ndarray):
        return a
    raise TypeError(
        "An array must be any of supported type, but got {}".format(type(a))
    )


def get_usm_ndarray_or_scalar(a):
    """
    Return scalar or :class:`dpctl.tensor.usm_ndarray` from input object `a`.

    Parameters
    ----------
    a : {scalar, dpnp_array, usm_ndarray}
        Input of any supported type: scalar, :class:`dpnp.ndarray`
        or :class:`dpctl.tensor.usm_ndarray`.

    Returns
    -------
    out : scalar, usm_ndarray
        A scalar if the input `a` is scalar.
        A dpctl USM ndarray if the input `a` is array.

    Raises
    ------
    TypeError
        If input parameter `a` is of unsupported object type.

    """

    return a if isscalar(a) else get_usm_ndarray(a)


def is_supported_array_type(a):
    """
    Return ``True`` if an array of either type :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray` type, ``False`` otherwise.

    Parameters
    ----------
    a : array
        An input array to check the type.

    Returns
    -------
    out : bool
        ``True`` if type of array `a` is supported array type,
        ``False`` otherwise.

    """

    return isinstance(a, (dpnp_array, dpt.usm_ndarray))
