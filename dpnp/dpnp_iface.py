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
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name

import os

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
import dpctl.utils as dpu
import numpy
from dpctl.tensor._device import normalize_queue_device

import dpnp
from dpnp.dpnp_algo import *
from dpnp.dpnp_array import dpnp_array
from dpnp.fft import *
from dpnp.linalg import *
from dpnp.random import *

__all__ = [
    "are_same_logical_tensors",
    "asnumpy",
    "as_usm_ndarray",
    "check_limitations",
    "check_supported_arrays_type",
    "default_float_type",
    "get_dpnp_descriptor",
    "get_include",
    "get_normalized_queue_device",
    "get_result_array",
    "get_usm_ndarray",
    "get_usm_ndarray_or_scalar",
    "is_cuda_backend",
    "is_supported_array_or_scalar",
    "is_supported_array_type",
    "synchronize_array_data",
]

from dpnp.dpnp_iface_arraycreation import *
from dpnp.dpnp_iface_arraycreation import __all__ as __all__arraycreation
from dpnp.dpnp_iface_bitwise import *
from dpnp.dpnp_iface_bitwise import __all__ as __all__bitwise
from dpnp.dpnp_iface_counting import *
from dpnp.dpnp_iface_counting import __all__ as __all__counting
from dpnp.dpnp_iface_functional import *
from dpnp.dpnp_iface_functional import __all__ as __all__functional
from dpnp.dpnp_iface_histograms import *
from dpnp.dpnp_iface_histograms import __all__ as __all__histograms
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
from dpnp.dpnp_iface_nanfunctions import *
from dpnp.dpnp_iface_nanfunctions import __all__ as __all__nanfunctions
from dpnp.dpnp_iface_searching import *
from dpnp.dpnp_iface_searching import __all__ as __all__searching
from dpnp.dpnp_iface_sorting import *
from dpnp.dpnp_iface_sorting import __all__ as __all__sorting
from dpnp.dpnp_iface_statistics import *
from dpnp.dpnp_iface_statistics import __all__ as __all__statistics
from dpnp.dpnp_iface_trigonometric import *
from dpnp.dpnp_iface_trigonometric import __all__ as __all__trigonometric
from dpnp.dpnp_iface_window import *
from dpnp.dpnp_iface_window import __all__ as __all__window

# pylint: disable=no-name-in-module
from .dpnp_utils import (
    dpnp_descriptor,
    map_dtype_to_device,
    use_origin_backend,
)

__all__ += __all__arraycreation
__all__ += __all__bitwise
__all__ += __all__counting
__all__ += __all__functional
__all__ += __all__histograms
__all__ += __all__indexing
__all__ += __all__libmath
__all__ += __all__linearalgebra
__all__ += __all__logic
__all__ += __all__manipulation
__all__ += __all__mathematical
__all__ += __all__nanfunctions
__all__ += __all__searching
__all__ += __all__sorting
__all__ += __all__statistics
__all__ += __all__trigonometric
__all__ += __all__window


def are_same_logical_tensors(ar1, ar2):
    """
    Check if two arrays are logical views into the same memory.

    Parameters
    ----------
    ar1 : {dpnp.ndarray, usm_ndarray}
        First input array.
    ar2 : {dpnp.ndarray, usm_ndarray}
        Second input array.

    Returns
    -------
    out : bool
        ``True`` if two arrays are logical views into the same memory,
        ``False`` otherwise.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> b = a[:]
    >>> a is b
    False
    >>> np.are_same_logical_tensors(a, b)
    True
    >>> b[0] = 0
    >>> a
    array([0, 2, 3])

    >>> c = a.copy()
    >>> np.are_same_logical_tensors(a, c)
    False

    """

    return ti._same_logical_tensors(
        dpnp.get_usm_ndarray(ar1), dpnp.get_usm_ndarray(ar2)
    )


def asnumpy(a, order="C"):
    """
    Returns the NumPy array with input data.

    Parameters
    ----------
    a : {array_like}
        Arbitrary object that can be converted to :obj:`numpy.ndarray`.
    order : {None, 'C', 'F', 'A', 'K'}, optional
        The desired memory layout of the converted array.
        When `order` is ``'A'``, it uses ``'F'`` if `a` is column-major and
        uses ``'C'`` otherwise. And when `order` is ``'K'``, it keeps strides
        as closely as possible.

        Default: ``'C'``.

    Returns
    -------
    out : numpy.ndarray
        NumPy interpretation of input array `a`.

    Notes
    -----
    This function works exactly the same as :obj:`numpy.asarray`.

    """

    if isinstance(a, dpnp_array):
        return a.asnumpy()

    if isinstance(a, dpt.usm_ndarray):
        return dpt.asnumpy(a)

    return numpy.asarray(a, order=order)


def as_usm_ndarray(a, dtype=None, device=None, usm_type=None, sycl_queue=None):
    """
    Return :class:`dpctl.tensor.usm_ndarray` from input object `a`.

    Parameters
    ----------
    a : {array_like, scalar}
        Input array or scalar.
    dtype : {None, str, dtype object}, optional
        The desired dtype for the result array if new array is creating. If not
        given, a default dtype will be used that can represent the values (by
        considering Promotion Type Rule and device capabilities when necessary).

        Default: ``None``.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpctl.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.
        If the value is ``None``, returned array is created on the same device
        as `a`.

        Default: ``None``.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the result array if new array
        is created.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for result array allocation if required.

        Default: ``None``.

    Returns
    -------
    out : usm_ndarray
        A dpctl USM ndarray from input array or scalar `a`.
        If `a` is instance of :class:`dpnp.ndarray`
        or :class:`dpctl.tensor.usm_ndarray`, no array allocation will be done
        and `dtype`, `device`, `usm_type`, `sycl_queue` keywords
        will be ignored.

    """

    if is_supported_array_type(a):
        return get_usm_ndarray(a)

    return dpt.asarray(
        a, dtype=dtype, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def check_limitations(
    subok=False,
    like=None,
    initial=None,
    where=True,
    subok_linalg=True,
    signature=None,
):
    """
    Checking limitation kwargs for their supported values.

    Parameter `subok` for array creation functions is only supported with
    default value ``False``.
    Parameter `like` is only supported with default value ``None``.
    Parameter `initial` is only supported with default value ``None``.
    Parameter `where` is only supported with default value ``True``.
    Parameter `subok` for linear algebra functions, named as `subok_linalg`
    here, and is only supported with default value ``True``.
    Parameter `signature` is only supported with default value ``None``.

    Raises
    ------
    NotImplementedError
        If any input kwargs is of unsupported value.

    """

    if like is not None:
        raise NotImplementedError(
            "Keyword argument `like` is supported only with "
            f"default value ``None``, but got {like}."
        )
    if subok is not False:
        raise NotImplementedError(
            "Keyword argument `subok` is supported only with "
            f"default value ``False``, but got {subok}."
        )
    if initial is not None:
        raise NotImplementedError(
            "Keyword argument `initial` is only supported with "
            f"default value ``None``, but got {initial}."
        )
    if where is not True:
        raise NotImplementedError(
            "Keyword argument `where` is supported only with "
            f"default value ``True``, but got {where}."
        )
    if not subok_linalg:
        raise NotImplementedError(
            "keyword argument `subok` is only supported with "
            f"default value ``True``, but got {subok_linalg}."
        )
    if signature is not None:
        raise NotImplementedError(
            "keyword argument `signature` is only supported with "
            f"default value ``None``, but got {signature}."
        )


def check_supported_arrays_type(*arrays, scalar_type=False, all_scalars=False):
    """
    Return ``True`` if each array has either type of scalar,
    :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    But if any array has unsupported type, ``TypeError`` will be raised.

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
        Input arrays to check for supported types.
    scalar_type : bool, optional
        A scalar type is also considered as supported if flag is ``True``.

        Default: ``False``.
    all_scalars : bool, optional
        All the input arrays can be scalar if flag is ``True``.

        Default: ``False``.

    Returns
    -------
    out : bool
        ``True`` if each type of input `arrays` is supported type,
        ``False`` otherwise.

    Raises
    ------
    TypeError
        If any input array from `arrays` is of unsupported array type.

    """

    any_is_array = False
    for a in arrays:
        if is_supported_array_type(a):
            any_is_array = True
            continue

        if scalar_type and dpnp.isscalar(a):
            continue

        raise TypeError(
            f"An array must be any of supported type, but got {type(a)}"
        )

    if len(arrays) > 0 and not (all_scalars or any_is_array):
        raise TypeError(
            "At least one input must be of supported array type, "
            "but got all scalars."
        )
    return True


def default_float_type(device=None, sycl_queue=None):
    """
    Return a floating type used by default in DPNP depending on device
    capabilities.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpctl.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.
        The value ``None`` is interpreted as to use a default device.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue which might be used to create an array of default floating
        type. The `sycl_queue` can be ``None`` (the default), which is
        interpreted as to get the SYCL queue from `device` keyword if present
        or to use a default queue.

        Default: ``None``.

    Returns
    -------
    dt : dtype
        A default DPNP floating type.

    """

    _sycl_queue = get_normalized_queue_device(
        device=device, sycl_queue=sycl_queue
    )
    return map_dtype_to_device(dpnp.float64, _sycl_queue.sycl_device)


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
    Return DPNP internal data descriptor object if:
      1. We can proceed with input data object with DPNP
      2. We want to handle input data object
    Return False if:
      1. We do not want to work with input data object
      2. We can not handle with input data object
    """

    if use_origin_backend():
        return False

    # If input object is a scalar, it means it was allocated on host memory.
    # We need to copy it to USM memory according to compute follows data.
    if dpnp.isscalar(ext_obj):
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
        # TODO: replace this workaround when usm_ndarray will provide
        # such functionality
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
            ext_obj = array(ext_obj, order="C")

    # while dpnp functions are based on DPNP_QUEUE
    # we need to create a copy on device associated with DPNP_QUEUE
    # if function get implementation for different queue
    # then this behavior can be disabled with setting
    # "copy_when_nondefault_queue"
    queue = getattr(ext_obj, "sycl_queue", None)
    if queue is not None and copy_when_nondefault_queue:
        default_queue = dpctl.SyclQueue()
        queue_is_default = (
            dpctl.utils.get_execution_queue([queue, default_queue]) is not None
        )
        if not queue_is_default:
            ext_obj = array(ext_obj, sycl_queue=default_queue)

    dpnp_desc = dpnp_descriptor(ext_obj)
    if dpnp_desc.is_valid:  # pylint: disable=using-constant-test
        return dpnp_desc

    return False


def get_include():
    r"""
    Return the directory that contains \*.h header files of dpnp C++ backend.

    An extension module that needs to be compiled against dpnp backend
    should use this function to locate the appropriate include directory.

    """

    return os.path.join(os.path.dirname(__file__), "backend", "include")


def get_normalized_queue_device(obj=None, device=None, sycl_queue=None):
    """
    Utility to process complementary keyword arguments 'device' and 'sycl_queue'
    in subsequent calls of functions from `dpctl.tensor` module.

    If both arguments 'device' and 'sycl_queue' have default value ``None``
    and 'obj' has `sycl_queue` attribute, it assumes that Compute Follows Data
    approach has to be applied and so the resulting SYCL queue will be
    normalized based on the queue value from 'obj'.

    Parameters
    ----------
    obj : object, optional
        A python object. Can be an instance of `dpnp_array`,
        `dpctl.tensor.usm_ndarray`, an object representing SYCL USM allocation
        and implementing `__sycl_usm_array_interface__` protocol, an instance
        of `numpy.ndarray`, an object supporting Python buffer protocol,
        a Python scalar, or a (possibly nested) sequence of Python scalars.
    sycl_queue : {None, class:`dpctl.SyclQueue`}, optional
        A queue which explicitly indicates where USM allocation is done
        and the population code (if any) is executed.
        Value ``None`` is interpreted as to get the SYCL queue from either
        `obj` parameter if not ``None`` or from `device` keyword,
        or to use default queue.

        Default: ``None``.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpctl.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.
        The value ``None`` is interpreted as to use the same device as `obj`.

        Default: ``None``.

    Returns
    -------
    sycl_queue: dpctl.SyclQueue
        A :class:`dpctl.SyclQueue` object normalized by
        `normalize_queue_device` call of `dpctl.tensor` module invoked with
        `device` and `sycl_queue` values. If both incoming `device` and
        `sycl_queue` are ``None`` and `obj` has `sycl_queue` attribute,
        the normalization will be performed for `obj.sycl_queue` value.

    """

    if (
        device is None
        and sycl_queue is None
        and obj is not None
        and hasattr(obj, "sycl_queue")
    ):
        sycl_queue = obj.sycl_queue

    return normalize_queue_device(sycl_queue=sycl_queue, device=device)


def get_result_array(a, out=None, casting="safe"):
    """
    If `out` is provided, value of `a` array will be copied into the
    `out` array according to ``safe`` casting rule.
    Otherwise, the input array `a` is returned.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    out : {None, dpnp.ndarray, usm_ndarray}
        If provided, value of `a` array will be copied into it
        according to ``safe`` casting rule.
        It should be of the appropriate shape.

        Default: ``None``.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

        Default: ``'safe'``.

    Returns
    -------
    out : dpnp.ndarray
        Return `out` if provided, otherwise return `a`.

    """

    if out is None:
        if isinstance(a, dpt.usm_ndarray):
            return dpnp_array._create_from_usm_ndarray(a)
        return a

    if isinstance(out, dpt.usm_ndarray):
        out = dpnp_array._create_from_usm_ndarray(out)

    if a is out or dpnp.are_same_logical_tensors(a, out):
        return out

    if out.shape != a.shape:
        raise ValueError(
            f"Output array of shape {a.shape} is needed, got {out.shape}."
        )

    dpnp.copyto(out, a, casting=casting)
    return out


def get_usm_ndarray(a):
    """
    Return :class:`dpctl.tensor.usm_ndarray` from input array `a`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
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
        f"An array must be any of supported type, but got {type(a)}"
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

    return a if dpnp.isscalar(a) else get_usm_ndarray(a)


def is_cuda_backend(obj=None):
    """
    Checks that object has a CUDA backend.

    Parameters
    ----------
    obj : {Device, SyclDevice, SyclQueue, dpnp.ndarray, usm_ndarray, None},
          optional
        An input object with sycl_device property to check device backend.
        If `obj` is ``None``, device backend will be checked for the default
        queue.

        Default: ``None``.

    Returns
    -------
    out : bool
        Return ``True`` if data of the input object resides on a CUDA backend,
        otherwise ``False``.

    """

    if obj is None:
        sycl_device = dpctl.select_default_device()
    elif isinstance(obj, dpctl.SyclDevice):
        sycl_device = obj
    else:
        sycl_device = getattr(obj, "sycl_device", None)
    if (
        sycl_device is not None
        and sycl_device.backend == dpctl.backend_type.cuda
    ):  # pragma: no cover
        return True
    return False


def is_supported_array_or_scalar(a):
    """
    Return ``True`` if `a` is a scalar or an array of either
    :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray` type,
    ``False`` otherwise.

    Parameters
    ----------
    a : {scalar, dpnp_array, usm_ndarray}
        An input scalar or an array to check the type of.

    Returns
    -------
    out : bool
        ``True`` if input `a` is a scalar or an array of supported type,
        ``False`` otherwise.

    """

    return dpnp.isscalar(a) or is_supported_array_type(a)


def is_supported_array_type(a):
    """
    Return ``True`` if an array of either type :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray` type, ``False`` otherwise.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        An input array to check the type.

    Returns
    -------
    out : bool
        ``True`` if type of array `a` is supported array type,
        ``False`` otherwise.

    """

    return isinstance(a, (dpnp_array, dpt.usm_ndarray))


def synchronize_array_data(a):
    """
    The dpctl interface was reworked to make asynchronous execution.
    That function makes a synchronization call to ensure array data is valid
    before exit from dpnp interface function.

    """

    check_supported_arrays_type(a)
    dpu.SequentialOrderManager[a.sycl_queue].wait()
