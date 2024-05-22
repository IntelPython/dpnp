# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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
Interface of the array creation function of the dpnp

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import operator

import numpy

import dpnp
from dpnp import dpnp_container

from .dpnp_algo.dpnp_arraycreation import (
    dpnp_geomspace,
    dpnp_linspace,
    dpnp_logspace,
    dpnp_nd_grid,
)

__all__ = [
    "arange",
    "array",
    "asanyarray",
    "asarray",
    "ascontiguousarray",
    "asfortranarray",
    "copy",
    "diag",
    "diagflat",
    "empty",
    "empty_like",
    "eye",
    "frombuffer",
    "fromfile",
    "fromfunction",
    "fromiter",
    "fromstring",
    "full",
    "full_like",
    "geomspace",
    "identity",
    "linspace",
    "loadtxt",
    "logspace",
    "meshgrid",
    "mgrid",
    "ogrid",
    "ones",
    "ones_like",
    "trace",
    "tri",
    "tril",
    "triu",
    "vander",
    "zeros",
    "zeros_like",
]


def arange(
    start,
    /,
    stop=None,
    step=1,
    *,
    dtype=None,
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Returns an array with evenly spaced values within a given interval.

    For full documentation refer to :obj:`numpy.arange`.

    Parameters
    ----------
    start : {int, real}, optional
        Start of interval. The interval includes this value.
        The default start value is 0.
    stop : {int, real}
        End of interval. The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of out.
    step : {int, real}, optional
        Spacing between values. The default `step` size is 1. If `step`
        is specified as a position argument, `start` must also be given.
    dtype : {None, dtype}, optional
        The desired dtype for the array. If not given, a default dtype will be
        used that can represent the values (by considering Promotion Type Rule
        and device capabilities when necessary).
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        The 1-D array containing evenly spaced values.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.linspace` : Evenly spaced numbers with careful handling of
                           endpoints.

    Examples
    --------
    >>> import dpnp as np
    >>> np.arange(3)
    array([0, 1, 2])
    >>> np.arange(3, 7)
    array([3, 4, 5, 6])
    >>> np.arange(3, 7, 2)
    array([3, 5])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.arange(3)  # default case
    >>> x, x.device, x.usm_type
    (array([0, 1, 2]), Device(level_zero:gpu:0), 'device')

    >>> y = np.arange(3, device="cpu")
    >>> y, y.device, y.usm_type
    (array([0, 1, 2]), Device(opencl:cpu:0), 'device')

    >>> z = np.arange(3, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([0, 1, 2]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)

    if usm_type is None:
        usm_type = "device"

    return dpnp_container.arange(
        start,
        stop=stop,
        step=step,
        dtype=dtype,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


# pylint: disable=redefined-outer-name
def array(
    a,
    dtype=None,
    *,
    copy=True,
    order="K",
    subok=False,
    ndmin=0,
    like=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Create an array.

    For full documentation refer to :obj:`numpy.array`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : {None, dtype}, optional
        The desired dtype for the array. If not given, a default dtype will be
        used that can represent the values (by considering Promotion Type Rule
        and device capabilities when necessary).
    copy : bool, optional
        If ``True`` (default), then the object is copied.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the newly output array. Default: "K".
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        An array object satisfying the specified requirements.

    Limitations
    -----------
    Parameter `subok` is supported only with default value ``False``.
    Parameter `ndmin` is supported only with default value ``0``.
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of
                             input.
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of
                            input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of
                             input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with
                            value.
    :obj:`dpnp.empty` : Return a new uninitialized array.
    :obj:`dpnp.ones` : Return a new array setting values to one.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2, 3])
    >>> x.ndim, x.size, x.shape
    (1, 3, (3,))
    >>> x
    array([1, 2, 3])

    More than one dimension:

    >>> x2 = np.array([[1, 2], [3, 4]])
    >>> x2.ndim, x2.size, x2.shape
    (2, 4, (2, 2))
    >>> x2
    array([[1, 2],
           [3, 4]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.array([1, 2, 3]) # default case
    >>> x, x.device, x.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'device')

    >>> y = np.array([1, 2, 3], device="cpu")
    >>> y, y.device, y.usm_type
    (array([1, 2, 3]), Device(opencl:cpu:0), 'device')

    >>> z = np.array([1, 2, 3], usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(subok=subok, like=like)
    if ndmin != 0:
        raise NotImplementedError(
            "Keyword argument `ndmin` is supported only with "
            f"default value ``0``, but got {ndmin}"
        )

    # `False`` in numpy means exactly the same like `None` in python array API:
    # that is to reuse existing memory buffer if possible or to copy otherwise.
    if copy is False:
        copy = None

    return dpnp_container.asarray(
        a,
        dtype=dtype,
        copy=copy,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def asanyarray(
    a,
    dtype=None,
    order=None,
    *,
    like=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Convert the input to an :class:`dpnp.ndarray`.

    For full documentation refer to :obj:`numpy.asanyarray`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : {None, dtype}, optional
        The desired dtype for the array. If not given, a default dtype will be
        used that can represent the values (by considering Promotion Type Rule
        and device capabilities when necessary).
    order : {None, "C", "F", "A", "K"}, optional
        Memory layout of the newly output array. Default: "K".
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array interpretation of `a`.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.asarray` : Similar function which always returns ndarrays.
    :obj:`dpnp.ascontiguousarray` : Convert input to a contiguous array.
    :obj:`dpnp.asfarray` : Convert input to a floating point ndarray.
    :obj:`dpnp.asfortranarray` : Convert input to an ndarray with column-major
                                 memory order.
    :obj:`dpnp.asarray_chkfinite` : Similar function which checks input
                                    for NaNs and Infs.
    :obj:`dpnp.fromiter` : Create an array from an iterator.
    :obj:`dpnp.fromfunction` : Construct an array by executing a function
                               on grid positions.

    Examples
    --------
    >>> import dpnp as np
    >>> np.asanyarray([1, 2, 3])
    array([1, 2, 3])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.asanyarray([1, 2, 3]) # default case
    >>> x, x.device, x.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'device')

    >>> y = np.asanyarray([1, 2, 3], device="cpu")
    >>> y, y.device, y.usm_type
    (array([1, 2, 3]), Device(opencl:cpu:0), 'device')

    >>> z = np.asanyarray([1, 2, 3], usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)

    return asarray(
        a,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def asarray(
    a,
    dtype=None,
    order=None,
    like=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Converts an input object into array.

    For full documentation refer to :obj:`numpy.asarray`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : {None, dtype}, optional
        The desired dtype for the array. If not given, a default dtype will be
        used that can represent the values (by considering Promotion Type Rule
        and device capabilities when necessary).
    order : {None, "C", "F", "A", "K"}, optional
        Memory layout of the newly output array. Default: "K".
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array interpretation of `a`. No copy is performed if the input
        is already an ndarray with matching dtype and order.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.asanyarray` : Similar function which passes through subclasses.
    :obj:`dpnp.ascontiguousarray` : Convert input to a contiguous array.
    :obj:`dpnp.asfarray` : Convert input to a floating point ndarray.
    :obj:`dpnp.asfortranarray` : Convert input to an ndarray with column-major
                                 memory order.
    :obj:`dpnp.asarray_chkfinite` : Similar function which checks input
                                    for NaNs and Infs.
    :obj:`dpnp.fromiter` : Create an array from an iterator.
    :obj:`dpnp.fromfunction` : Construct an array by executing a function
                               on grid positions.

    Examples
    --------
    >>> import dpnp as np
    >>> np.asarray([1, 2, 3])
    array([1, 2, 3])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.asarray([1, 2, 3]) # default case
    >>> x, x.device, x.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'device')

    >>> y = np.asarray([1, 2, 3], device="cpu")
    >>> y, y.device, y.usm_type
    (array([1, 2, 3]), Device(opencl:cpu:0), 'device')

    >>> z = np.asarray([1, 2, 3], usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)

    return dpnp_container.asarray(
        a,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def ascontiguousarray(
    a, dtype=None, *, like=None, device=None, usm_type=None, sycl_queue=None
):
    """
    Return a contiguous array in memory (C order).

    For full documentation refer to :obj:`numpy.ascontiguousarray`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : {None, dtype}, optional
        The desired dtype for the array. If not given, a default dtype will be
        used that can represent the values (by considering Promotion Type Rule
        and device capabilities when necessary).
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Contiguous array of same shape and content as `a`, with type `dtype`
        if specified.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.asfortranarray` : Convert input to an ndarray with column-major
                     memory order.
    :obj:`dpnp.require` : Return an ndarray that satisfies requirements.
    :obj:`dpnp.ndarray.flags` : Information about the memory layout
                                of the array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.ones((2, 3), order='F')
    >>> x.flags['F_CONTIGUOUS']
    True

    Calling ``ascontiguousarray`` makes a C-contiguous copy:

    >>> y = np.ascontiguousarray(x)
    >>> y.flags['F_CONTIGUOUS']
    True
    >>> x is y
    False

    Now, starting with a C-contiguous array:

    >>> x = np.ones((2, 3), order='C')
    >>> x.flags['C_CONTIGUOUS']
    True

    Then, calling ``ascontiguousarray`` returns the same object:

    >>> y = np.ascontiguousarray(x)
    >>> x is y
    True

    Creating an array on a different device or with a specified usm_type

    >>> x0 = np.asarray([1, 2, 3])
    >>> x = np.ascontiguousarray(x0) # default case
    >>> x, x.device, x.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'device')

    >>> y = np.ascontiguousarray(x0, device="cpu")
    >>> y, y.device, y.usm_type
    (array([1, 2, 3]), Device(opencl:cpu:0), 'device')

    >>> z = np.ascontiguousarray(x0, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)

    # at least 1-d array has to be returned
    if dpnp.isscalar(a) or hasattr(a, "ndim") and a.ndim == 0:
        a = [a]

    return asarray(
        a,
        dtype=dtype,
        order="C",
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def asfortranarray(
    a, dtype=None, *, like=None, device=None, usm_type=None, sycl_queue=None
):
    """
    Return an array ``(ndim >= 1)`` laid out in Fortran order in memory.

    For full documentation refer to :obj:`numpy.asfortranarray`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : {None, dtype}, optional
        The desired dtype for the array. If not given, a default dtype will be
        used that can represent the values (by considering Promotion Type Rule
        and device capabilities when necessary).
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        The input `a` in Fortran, or column-major, order.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.ascontiguousarray` : Convert input to a contiguous (C order)
                                    array.
    :obj:`dpnp.asanyarray` : Convert input to an ndarray with either row or
                             column-major memory order.
    :obj:`dpnp.require` : Return an ndarray that satisfies requirements.
    :obj:`dpnp.ndarray.flags` : Information about the memory layout
                                of the array.

    Examples
    --------
    >>> import dpnp as np

    Starting with a C-contiguous array:

    >>> x = np.ones((2, 3), order='C')
    >>> x.flags['C_CONTIGUOUS']
    True

    Calling ``asfortranarray`` makes a Fortran-contiguous copy:

    >>> y = np.asfortranarray(x)
    >>> y.flags['F_CONTIGUOUS']
    True
    >>> x is y
    False

    Now, starting with a Fortran-contiguous array:

    >>> x = np.ones((2, 3), order='F')
    >>> x.flags['F_CONTIGUOUS']
    True

    Then, calling ``asfortranarray`` returns the same object:

    >>> y = np.asfortranarray(x)
    >>> x is y
    True

    Creating an array on a different device or with a specified usm_type

    >>> x0 = np.asarray([1, 2, 3])
    >>> x = np.asfortranarray(x0) # default case
    >>> x, x.device, x.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'device')

    >>> y = np.asfortranarray(x0, device="cpu")
    >>> y, y.device, y.usm_type
    (array([1, 2, 3]), Device(opencl:cpu:0), 'device')

    >>> z = np.asfortranarray(x0, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)

    # at least 1-d array has to be returned
    if dpnp.isscalar(a) or hasattr(a, "ndim") and a.ndim == 0:
        a = [a]

    return asarray(
        a,
        dtype=dtype,
        order="F",
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def copy(
    a, order="K", subok=False, device=None, usm_type=None, sycl_queue=None
):
    """
    Return an array copy of the given object.

    For full documentation refer to :obj:`numpy.copy`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    order : {"C", "F", "A", "K"}, optional
        Memory layout of the newly output array. Default: "K".
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Limitations
    -----------
    Parameter `subok` is supported only with default value ``False``.
    Otherwise, the function raises `NotImplementedError` exception.

    Returns
    -------
    out : dpnp.ndarray
        Array interpretation of `a`.

    See Also
    --------
    :obj:`dpnp.ndarray.copy` : Preferred method for creating an array copy

    Notes
    -----
    This is equivalent to:

    >>> dpnp.array(a, copy=True)

    Examples
    --------
    Create an array `x`, with a reference `y` and a copy `z`:

    >>> import dpnp as np
    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)

    Note that, when we modify `x`, `y` will change, but not `z`:

    >>> x[0] = 10
    >>> x[0] == y[0]
    array(True)
    >>> x[0] == z[0]
    array(False)

    Creating an array on a different device or with a specified usm_type

    >>> x0 = np.array([1, 2, 3])
    >>> x = np.copy(x0) # default case
    >>> x, x.device, x.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'device')

    >>> y = np.copy(x0, device="cpu")
    >>> y, y.device, y.usm_type
    (array([1, 2, 3]), Device(opencl:cpu:0), 'device')

    >>> z = np.copy(x0, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(subok=subok)

    if dpnp.is_supported_array_type(a):
        sycl_queue_normalized = dpnp.get_normalized_queue_device(
            a, device=device, sycl_queue=sycl_queue
        )
        if (
            usm_type is None or usm_type == a.usm_type
        ) and sycl_queue_normalized == a.sycl_queue:
            return dpnp_container.copy(a, order=order)

    return array(
        a,
        order=order,
        subok=subok,
        copy=True,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def diag(v, /, k=0, *, device=None, usm_type=None, sycl_queue=None):
    """
    Extract a diagonal or construct a diagonal array.

    For full documentation refer to :obj:`numpy.diag`.

    Parameters
    ----------
    v : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
        If `v` is a 1-D array, return a 2-D array with `v`
        on the `k`-th diagonal.
        If `v` is a 2-D array and is an instance of
        {dpnp.ndarray, usm_ndarray}, then:
          - If `device`, `usm_type`, and `sycl_queue` are set to their
            default values, returns a read/write view of its k-th diagonal.
          - Otherwise, returns a copy of its k-th diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use k > 0 for diagonals above
        the main diagonal, and k < 0 for diagonals below the main diagonal.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    :obj:`diagonal` : Return specified diagonals.
    :obj:`diagflat` : Create a 2-D array with the flattened input as a diagonal.
    :obj:`trace` : Return the sum along diagonals of the array.
    :obj:`triu` : Upper triangle of an array.
    :obj:`tril` : Lower triangle of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(9).reshape((3, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])

    >>> np.diag(x)
    array([0, 4, 8])
    >>> np.diag(x, k=1)
    array([1, 5])
    >>> np.diag(x, k=-1)
    array([3, 7])

    >>> np.diag(np.diag(x))
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 8]])

    Creating an array on a different device or with a specified usm_type

    >>> res = np.diag(x) # default case
    >>> res, res.device, res.usm_type
    (array([0, 4, 8]), Device(level_zero:gpu:0), 'device')

    >>> res_cpu = np.diag(x, device="cpu")
    >>> res_cpu, res_cpu.device, res_cpu.usm_type
    (array([0, 4, 8]), Device(opencl:cpu:0), 'device')

    >>> res_host = np.diag(x, usm_type="host")
    >>> res_host, res_host.device, res_host.usm_type
    (array([0, 4, 8]), Device(level_zero:gpu:0), 'host')

    """

    if not isinstance(k, int):
        raise TypeError(f"An integer is required, but got {type(k)}")

    v = dpnp.asanyarray(
        v, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )

    if v.ndim == 1:
        size = v.shape[0] + abs(k)
        ret = dpnp.zeros_like(v, shape=(size, size))
        ret.diagonal(k)[:] = v
        return ret

    if v.ndim == 2:
        return v.diagonal(k)

    raise ValueError("Input must be a 1-D or 2-D array.")


def diagflat(v, /, k=0, *, device=None, usm_type=None, sycl_queue=None):
    """
    Create a two-dimensional array with the flattened input as a diagonal.

    For full documentation refer to :obj:`numpy.diagflat`.

    Parameters
    ----------
    v : array_like
        Input data, which is flattened and set as the k-th diagonal of the
        output, in any form that can be converted to an array. This includes
        scalars, lists, lists of tuples, tuples, tuples of tuples, tuples of
        lists, and ndarrays.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal,
        a positive (negative) k giving the number of the diagonal above (below)
        the main.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        The 2-D output array.

    See Also
    --------
    :obj:`dpnp.diag` : Extract a diagonal or construct a diagonal array.
    :obj:`dpnp.diagonal` : Return specified diagonals.
    :obj:`dpnp.trace` : Return sum along diagonals.

    Examples
    --------
    >>> import dpnp as np
    >>> x0 = np.array([[1, 2], [3, 4]])
    >>> np.diagflat(x0)
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> np.diagflat(x0, 1)
    array([[0, 1, 0, 0, 0],
           [0, 0, 2, 0, 0],
           [0, 0, 0, 3, 0],
           [0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.diagflat(x0) # default case
    >>> x, x.device, x.usm_type
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]]), Device(level_zero:gpu:0), 'device')

    >>> y = np.diagflat(x0, device="cpu")
    >>> y, y.device, y.usm_type
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]]), Device(opencl:cpu:0), 'device')

    >>> z = np.diagflat(x0, usm_type="host")
    >>> z, z.device, z.usm_type
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]]), Device(level_zero:gpu:0), 'host')

    """

    if not isinstance(k, int):
        raise TypeError(f"An integer is required, but got {type(k)}")

    v = dpnp.asarray(v, device=device, usm_type=usm_type, sycl_queue=sycl_queue)
    v = dpnp.ravel(v)
    return dpnp.diag(v, k, usm_type=v.usm_type, sycl_queue=v.sycl_queue)


def empty(
    shape,
    *,
    dtype=None,
    order="C",
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return a new array of given shape and type, without initializing entries.

    For full documentation refer to :obj:`numpy.empty`.

    Parameters
    ----------
    shape : {int, sequence of ints}
        Shape of the new array, e.g., (2, 3) or 2.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    order : {None, "C", "F"}, optional
        Memory layout of the newly output array. Default: "C".
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of uninitialized data of the given shape, dtype, and order.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"``, ``"F"`` and
    ``None``.
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of
                             input.
    :obj:`dpnp.ones` : Return a new array setting values to one.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import dpnp as np
    >>> np.empty(4)
    array([9.03088525e-312, 9.03088525e-312, 9.03088525e-312, 9.03088525e-312])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.empty((3, 3)) # default case
    >>> x.shape, x.device, x.usm_type
    ((3, 3), Device(level_zero:gpu:0), 'device')

    >>> y = np.empty((3, 3), device="cpu")
    >>> y.shape, y.device, y.usm_type
    ((3, 3), Device(opencl:cpu:0), 'device')

    >>> z = np.empty((3, 3), usm_type="host")
    >>> z.shape, z.device, z.usm_type
    ((3, 3), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(order=order, like=like)

    if usm_type is None:
        usm_type = "device"

    return dpnp_container.empty(
        shape,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def empty_like(
    a,
    /,
    *,
    dtype=None,
    order="C",
    subok=False,
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Return a new array with the same shape and type as a given array.

    For full documentation refer to :obj:`numpy.empty_like`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        The shape and dtype of `a` define these same attributes
        of the returned array.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    order : {None, "C", "F"}, optional
        Memory layout of the newly output array. Default: "C".
    shape : {None, int, sequence of ints}
        Overrides the shape of the result.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of uninitialized data with the same shape and type as prototype.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"``, ``"F"`` and
    ``None``.
    Parameter `subok` is supported only with default value ``False``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of
                            input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of
                             input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with
                            value.
    :obj:`dpnp.empty` : Return a new uninitialized array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> np.empty_like(a)
    array([1, 2, 3])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.empty_like(a) # default case
    >>> x.shape, x.device, x.usm_type
    ((3, ), Device(level_zero:gpu:0), 'device')

    >>> y = np.empty_like(a, device="cpu")
    >>> y.shape, y.device, y.usm_type
    ((3, ), Device(opencl:cpu:0), 'device')

    >>> z = np.empty_like(a, usm_type="host")
    >>> z.shape, z.device, z.usm_type
    ((3, ), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_supported_arrays_type(a)
    dpnp.check_limitations(order=order, subok=subok)

    _shape = a.shape if shape is None else shape
    _dtype = a.dtype if dtype is None else dtype
    _usm_type = a.usm_type if usm_type is None else usm_type
    _sycl_queue = dpnp.get_normalized_queue_device(
        a, sycl_queue=sycl_queue, device=device
    )
    return dpnp_container.empty(
        _shape,
        dtype=_dtype,
        order=order,
        usm_type=_usm_type,
        sycl_queue=_sycl_queue,
    )


# pylint: disable=invalid-name
def eye(
    N,
    /,
    M=None,
    k=0,
    dtype=None,
    order="C",
    *,
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    For full documentation refer to :obj:`numpy.eye`.

    Parameters
    ----------
    N : int
        Number of rows in the output.
    M : {None, int}, optional
        Number of columns in the output. If None, defaults to `N`.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value to
        a lower diagonal.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    order : {None, "C", "F"}, optional
        Memory layout of the newly output array. Default: "C".
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        An array where all elements are equal to zero, except for the k-th
        diagonal, whose values are equal to one.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"``, ``"F"`` and
    ``None``.
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.identity` : Return the identity array.
    :obj:`dpnp.diag` : Extract a diagonal or construct a diagonal array.

    Examples
    --------
    >>> import dpnp as np
    >>> np.eye(2, dtype=int)
    array([[1, 0],
           [0, 1]])

    >>> np.eye(3, k=1)
    array([[0.,  1.,  0.],
           [0.,  0.,  1.],
           [0.,  0.,  0.]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.eye(2, dtype=int) # default case
    >>> x, x.device, x.usm_type
    (array([[1, 0],
            [0, 1]]), Device(level_zero:gpu:0), 'device')

    >>> y = np.eye(2, dtype=int, device="cpu")
    >>> y, y.device, y.usm_type
    (array([[1, 0],
            [0, 1]]), Device(opencl:cpu:0), 'device')

    >>> z = np.eye(2, dtype=int, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([[1, 0],
            [0, 1]]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(order=order, like=like)

    if usm_type is None:
        usm_type = "device"

    return dpnp_container.eye(
        N,
        M,
        k=k,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def frombuffer(
    buffer,
    dtype=float,
    count=-1,
    offset=0,
    *,
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    r"""
    Interpret a buffer as a 1-dimensional array.

    For full documentation refer to :obj:`numpy.frombuffer`.

    Parameters
    ----------
    buffer : buffer_like
        An object that exposes the buffer interface.
    dtype : data-type, optional
        Data-type of the returned array.
        Default is the default floating point data type for the device where
        the returned array is allocated.
    count : int, optional
        Number of items to read. ``-1`` means all data in the buffer.
    offset : int, optional
        Start reading the buffer from this offset (in bytes); default: 0.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        A 1-dimensional array created from input buffer object.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    Notes
    -----
    This uses :obj:`numpy.frombuffer` and coerces the result to a DPNP array.

    See also
    --------
    :obj:`dpnp.fromfile` : Construct array from data in a text or binary file.
    :obj:`dpnp.fromiter` : Construct array from an iterable object.
    :obj:`dpnp.fromstring` : Construct array from the text data in a string.

    Examples
    --------
    >>> import dpnp as np
    >>> s = b'\x01\x02\x03\x04'
    >>> np.frombuffer(s, dtype=np.int32)
    array([67305985], dtype=int32)
    >>> np.frombuffer(b'\x01\x02\x03\x04\x05', dtype='u1', count=3)
    array([1, 2, 3], dtype=uint8)

    Creating an array on a different device or with a specified usm_type

    >>> x = np.frombuffer(s, dtype=np.int32) # default case
    >>> x.device, x.usm_type
    (Device(level_zero:gpu:0), 'device')

    >>> y = np.frombuffer(s, dtype=np.int32, device='cpu')
    >>> y.device, y.usm_type
    (Device(opencl:cpu:0), 'device')

    >>> z = np.frombuffer(s, dtype=np.int32, usm_type="host")
    >>> z.device, z.usm_type
    (Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)
    return asarray(
        numpy.frombuffer(buffer, dtype=dtype, count=count, offset=offset),
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def fromfile(
    file,
    dtype=float,
    count=-1,
    sep="",
    offset=0,
    *,
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    r"""
    Construct an array from data in a text or binary file.

    A highly efficient way of reading binary data with a known data-type,
    as well as parsing simply formatted text files. Data written using the
    `tofile` method can be read using this function.

    For full documentation refer to :obj:`numpy.fromfile`.

    Parameters
    ----------
    file : file or str or Path
        Open file object or filename.
    dtype : data-type, optional
        Data type of the returned array.
        For binary files, it is used to determine the size and byte-order
        of the items in the file.
        Default is the default floating point data type for the device where
        the returned array is allocated.
    count : int, optional
        Number of items to read. ``-1`` means all items (i.e., the complete
        file).
    sep : str, optional
        Separator between items if `file` is a text file.
        Empty ("") separator means the file should be treated as binary.
        Spaces (" ") in the separator match zero or more whitespace characters.
        A separator consisting only of spaces must match at least one
        whitespace.
    offset : int, optional
        The offset (in bytes) from the file's current position. Defaults to 0.
        Only permitted for binary files.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        A 1-dimensional array created from data in a text or binary file.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    Notes
    -----
    This uses :obj:`numpy.fromfile` and coerces the result to a DPNP array.

    See also
    --------
    :obj:`dpnp.frombuffer` : Construct array from the buffer data.
    :obj:`dpnp.fromiter` : Construct array from an iterable object.
    :obj:`dpnp.fromstring` : Construct array from the text data in a string.
    :obj:`dpnp.load` : Load arrays or pickled objects from files.
    :obj:`dpnp.save` : Save an array to a binary file.
    :obj:`dpnp.ndarray.tofile` : Write array to a file as text or binary.
    :obj:`dpnp.loadtxt` : More flexible way of loading data from a text file.

    Examples
    --------
    Save the data to a temporary file:

    >>> import tempfile
    >>> fh = tempfile.TemporaryFile()
    >>> fh.write(b"\x00\x01\x02\x03\x04")
    >>> fh.flush()
    >>> fh.seek(0)

    Construct an array:

    >>> import dpnp as np
    >>> np.fromfile(fh, dtype="u1")
    array([0, 1, 2, 3, 4], dtype=uint8)

    Creating an array on a different device or with a specified usm_type

    >>> fh.seek(0)
    >>> x = np.fromfile(fh, dtype="u1") # default case
    >>> x.device, x.usm_type
    (Device(level_zero:gpu:0), 'device')

    >>> fh.seek(0)
    >>> y = np.fromfile(fh, dtype="u1", device='cpu')
    >>> y.device, y.usm_type
    (Device(opencl:cpu:0), 'device')

    >>> fh.seek(0)
    >>> z = np.fromfile(fh, dtype="u1", usm_type="host")
    >>> z.device, z.usm_type
    (Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)
    return asarray(
        numpy.fromfile(file, dtype=dtype, count=count, sep=sep, offset=offset),
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def fromfunction(
    function,
    shape,
    *,
    dtype=float,
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
    **kwargs,
):
    """
    Construct an array by executing a function over each coordinate.

    The resulting array therefore has a value ``fn(x, y, z)`` at
    coordinate ``(x, y, z)``.

    For full documentation refer to :obj:`numpy.fromfunction`.

    Parameters
    ----------
    function : callable
        The function is called with N parameters, where N is the rank of
        `shape`. Each parameter represents the coordinates of the array varying
        along a specific axis. For example, if `shape` were ``(2, 2)``, then
        the parameters would be ``array([[0, 0], [1, 1]])`` and
        ``array([[0, 1], [0, 1]])``.
    shape : (N,) tuple of ints
        Shape of the output array, which also determines the shape of
        the coordinate arrays passed to `function`.
    dtype : data-type, optional
        Data-type of the coordinate arrays passed to `function`.
        Default is the default floating point data type for the device where
        the returned array is allocated.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        The result of the call to `function` is passed back directly.
        Therefore the shape of `fromfunction` is completely determined by
        `function`.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    Notes
    -----
    This uses :obj:`numpy.fromfunction` and coerces the result to a DPNP array.
    Keywords other than `dtype` and `like` are passed to `function`.

    See also
    --------
    :obj:`dpnp.indices` : Return an array representing the indices of a grid.
    :obj:`dpnp.meshgrid` : Return coordinate matrices from coordinate vectors.

    Examples
    --------
    >>> import dpnp as np
    >>> np.fromfunction(lambda i, j: i, (2, 2), dtype=float)
    array([[0., 0.],
           [1., 1.]])

    >>> np.fromfunction(lambda i, j: j, (2, 2), dtype=float)
    array([[0., 1.],
           [0., 1.]])

    >>> np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
    array([[ True, False, False],
           [False,  True, False],
           [False, False,  True]])

    >>> np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.fromfunction(lambda i, j: i - j, (3, 3)) # default case
    >>> x.device, x.usm_type
    (Device(level_zero:gpu:0), 'device')

    >>> y = np.fromfunction(lambda i, j: i - j, (3, 3), device='cpu')
    >>> y.device, y.usm_type
    (Device(opencl:cpu:0), 'device')

    >>> z = np.fromfunction(lambda i, j: i - j, (3, 3), usm_type="host")
    >>> z.device, z.usm_type
    (Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)
    return asarray(
        numpy.fromfunction(function, shape, dtype=dtype, **kwargs),
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def fromiter(
    iter,
    dtype,
    count=-1,
    *,
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Create a new 1-dimensional array from an iterable object.

    For full documentation refer to :obj:`numpy.fromiter`.

    Parameters
    ----------
    iter : iterable object
        An iterable object providing data for the array.
    dtype : data-type
        The data-type of the returned array.
    count : int, optional
        The number of items to read from *iterable*. The default is -1,
        which means all data is read.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        The output array.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    Notes
    -----
    This uses :obj:`numpy.fromiter` and coerces the result to a DPNP array.

    See also
    --------
    :obj:`dpnp.frombuffer` : Construct array from the buffer data.
    :obj:`dpnp.fromfile` : Construct array from data in a text or binary file.
    :obj:`dpnp.fromstring` : Construct array from the text data in a string.

    Examples
    --------
    >>> import dpnp as np
    >>> iterable = (a * a for a in range(5))
    >>> np.fromiter(iterable, float)
    array([  0.,   1.,   4.,   9.,  16.])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.fromiter(iterable, np.int32) # default case
    >>> x.device, x.usm_type
    (Device(level_zero:gpu:0), 'device')

    >>> y = np.fromiter(iterable, np.int32, device='cpu')
    >>> y.device, y.usm_type
    (Device(opencl:cpu:0), 'device')

    >>> z = np.fromiter(iterable, np.int32, usm_type="host")
    >>> z.device, z.usm_type
    (Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)
    return asarray(
        numpy.fromiter(iter, dtype, count=count),
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def fromstring(
    string,
    dtype=float,
    count=-1,
    *,
    sep,
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    A new 1-D array initialized from text data in a string.

    For full documentation refer to :obj:`numpy.fromstring`.

    Parameters
    ----------
    string : str
        A string containing the data.
    dtype : data-type, optional
        The data type of the array.
        For binary input data, the data must be in exactly this format.
        Default is the default floating point data type for the device where
        the returned array is allocated.
    count : int, optional
        Read this number of `dtype` elements from the data. If this is negative
        (the default), the count will be determined from the length of the data.
    sep : str, optional
        The string separating numbers in the data; extra whitespace between
        elements is also ignored.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        The constructed array.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    Notes
    -----
    This uses :obj:`numpy.fromstring` and coerces the result to a DPNP array.

    See also
    --------
    :obj:`dpnp.frombuffer` : Construct array from the buffer data.
    :obj:`dpnp.fromfile` : Construct array from data in a text or binary file.
    :obj:`dpnp.fromiter` : Construct array from an iterable object.

    Examples
    --------
    >>> import dpnp as np
    >>> np.fromstring('1 2', dtype=int, sep=' ')
    array([1, 2])
    >>> np.fromstring('1, 2', dtype=int, sep=',')
    array([1, 2])

    """

    dpnp.check_limitations(like=like)
    return asarray(
        numpy.fromstring(string, dtype=dtype, count=count, sep=sep),
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def full(
    shape,
    fill_value,
    *,
    dtype=None,
    order="C",
    like=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    For full documentation refer to :obj:`numpy.full`.

    Parameters
    ----------
    shape : {int, sequence of ints}
        Shape of the new array, e.g., (2, 3) or 2.
    fill_value : {scalar, array_like}
        Fill value, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    order : {None, "C", "F"}, optional
        Memory layout of the newly output array. Default: "C".
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of `fill_value` with the given shape, dtype, and order.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"``, ``"F"`` and
    ``None``.
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with
                            value.
    :obj:`dpnp.empty` : Return a new uninitialized array.
    :obj:`dpnp.ones` : Return a new array setting values to one.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> np.full(4, 10)
    array([10, 10, 10, 10])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.full(4, 10) # default case
    >>> x, x.device, x.usm_type
    (array([10, 10, 10, 10]), Device(level_zero:gpu:0), 'device')

    >>> y = np.full(4, 10, device="cpu")
    >>> y, y.device, y.usm_type
    (array([10, 10, 10, 10]), Device(opencl:cpu:0), 'device')

    >>> z = np.full(4, 10, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([10, 10, 10, 10]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(order=order, like=like)

    return dpnp_container.full(
        shape,
        fill_value,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def full_like(
    a,
    /,
    fill_value,
    *,
    dtype=None,
    order="C",
    subok=False,
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Return a full array with the same shape and type as a given array.

    For full documentation refer to :obj:`numpy.full_like`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        The shape and dtype of `a` define these same attributes
        of the returned array.
    fill_value : {scalar, array_like}
        Fill value, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    order : {None, "C", "F"}, optional
        Memory layout of the newly output array. Default: "C".
    shape : {None, int, sequence of ints}
        Overrides the shape of the result.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of `fill_value` with the same shape and type as `a`.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"``, ``"F"`` and
    ``None``.
    Parameter `subok` is supported only with default value ``False``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of
                             input.
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of
                            input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of
                             input.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(6)
    >>> np.full_like(a, 1)
    array([1, 1, 1, 1, 1, 1])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.full_like(a, 1) # default case
    >>> x, x.device, x.usm_type
    (array([1, 1, 1, 1, 1, 1]), Device(level_zero:gpu:0), 'device')

    >>> y = np.full_like(a, 1, device="cpu")
    >>> y, y.device, y.usm_type
    (array([1, 1, 1, 1, 1, 1]), Device(opencl:cpu:0), 'device')

    >>> z = np.full_like(a, 1, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1, 1, 1, 1, 1, 1]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_supported_arrays_type(a)
    dpnp.check_limitations(order=order, subok=subok)

    _shape = a.shape if shape is None else shape
    _dtype = a.dtype if dtype is None else dtype
    _usm_type = a.usm_type if usm_type is None else usm_type
    _sycl_queue = dpnp.get_normalized_queue_device(
        a, sycl_queue=sycl_queue, device=device
    )

    return dpnp_container.full(
        _shape,
        fill_value,
        dtype=_dtype,
        order=order,
        usm_type=_usm_type,
        sycl_queue=_sycl_queue,
    )


def geomspace(
    start,
    stop,
    /,
    num=50,
    *,
    dtype=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
    endpoint=True,
    axis=0,
):
    """
    Return numbers spaced evenly on a log scale (a geometric progression).

    For full documentation refer to :obj:`numpy.geomspace`.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence, in any form that can be converted
        to an array. This includes scalars, lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists, and ndarrays.
    stop : array_like
        The final value of the sequence, in any form that can be converted to
        an array. This includes scalars, lists, lists of tuples, tuples, tuples
        of tuples, tuples of lists, and ndarrays. If `endpoint` is ``False``
        ``num + 1`` values are spaced over the interval in log-space, of which
        all but the last (a sequence of length `num`) are returned.
    num : int, optional
        Number of samples to generate. Default is 50.
    dtype : {None, dtype}, optional
        The desired dtype for the array. If not given, a default dtype will be
        used that can represent the values (by considering Promotion Type Rule
        and device capabilities when necessary).
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.
    endpoint : bool, optional
        If ``True``, `stop` is the last sample. Otherwise, it is not included.
        Default is ``True``.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start or
        stop are array-like. By default (0), the samples will be along a new
        axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    out : dpnp.ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    :obj:`dpnp.logspace` : Similar to :obj:`dpnp.geomspace`, but with endpoints
                           specified using log and base.
    :obj:`dpnp.linspace` : Similar to :obj:`dpnp.geomspace`, but with
                           arithmetic instead of geometric progression.
    :obj:`dpnp.arange` : Similar to :obj:`dpnp.linspace`, with the step size
                         specified instead of the number of samples.

    Examples
    --------
    >>> import dpnp as np
    >>> np.geomspace(1, 1000, num=4)
    array([   1.,   10.,  100., 1000.])
    >>> np.geomspace(1, 1000, num=3, endpoint=False)
    array([  1.,  10., 100.])
    >>> np.geomspace(1, 1000, num=4, endpoint=False)
    array([  1.        ,   5.62341325,  31.6227766 , 177.827941  ])
    >>> np.geomspace(1, 256, num=9)
    array([  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256.])

    >>> np.geomspace(1, 256, num=9, dtype=int)
    array([  1,   2,   4,   7,  16,  32,  63, 127, 256])
    >>> np.around(np.geomspace(1, 256, num=9)).astype(int)
    array([  1,   2,   4,   8,  16,  32,  64, 128, 256])

    >>> np.geomspace(1000, 1, num=4)
    array([1000.,  100.,   10.,    1.])
    >>> np.geomspace(-1000, -1, num=4)
    array([-1000.,  -100.,   -10.,    -1.])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.geomspace(1000, 1, num=4) # default case
    >>> x, x.device, x.usm_type
    (array([1000.,  100.,   10.,    1.]), Device(level_zero:gpu:0), 'device')

    >>> y = np.geomspace(1000, 1, num=4, device="cpu")
    >>> y, y.device, y.usm_type
    (array([1000.,  100.,   10.,    1.]), Device(opencl:cpu:0), 'device')

    >>> z = np.geomspace(1000, 1, num=4, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1000.,  100.,   10.,    1.]), Device(level_zero:gpu:0), 'host')

    """

    return dpnp_geomspace(
        start,
        stop,
        num,
        dtype=dtype,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
        endpoint=endpoint,
        axis=axis,
    )


def identity(
    n,
    /,
    dtype=None,
    *,
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return the identity array.

    The identity array is a square array with ones on the main diagonal.

    For full documentation refer to :obj:`numpy.identity`.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        `n` x `n` array with its main diagonal set to one,
        and all other elements 0.

    Limitations
    -----------
    Parameter `like` is currently not supported.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.eye` : Return a 2-D array with ones on the diagonal and zeros
                      elsewhere.
    :obj:`dpnp.ones` : Return a new array setting values to one.
    :obj:`dpnp.diag` : Extract a diagonal or construct a diagonal array.

    Examples
    --------
    >>> import dpnp as np
    >>> np.identity(3)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.identity(3) # default case
    >>> x, x.device, x.usm_type
    (array([[1.,  0.,  0.],
            [0.,  1.,  0.],
            [0.,  0.,  1.]]), Device(level_zero:gpu:0), 'device')

    >>> y = np.identity(3, device="cpu")
    >>> y, y.device, y.usm_type
    (array([[1.,  0.,  0.],
            [0.,  1.,  0.],
            [0.,  0.,  1.]]), Device(opencl:cpu:0), 'device')

    >>> z = np.identity(3, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([[1.,  0.,  0.],
            [0.,  1.,  0.],
            [0.,  0.,  1.]]), Device(level_zero:gpu:0), 'host')

    """

    if n < 0:
        raise ValueError("negative dimensions are not allowed")

    dpnp.check_limitations(like=like)

    _dtype = dpnp.default_float_type() if dtype is None else dtype
    return dpnp.eye(
        n,
        dtype=_dtype,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def linspace(
    start,
    stop,
    /,
    num,
    *,
    dtype=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
    endpoint=True,
    retstep=False,
    axis=0,
):
    """
    Return evenly spaced numbers over a specified interval.

    For full documentation refer to :obj:`numpy.linspace`.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence, in any form that can be converted
        to an array. This includes scalars, lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists, and ndarrays.
    stop : array_like
        The end value of the sequence, in any form that can be converted to
        an array. This includes scalars, lists, lists of tuples, tuples, tuples
        of tuples, tuples of lists, and ndarrays. If `endpoint` is set to
        ``False`` the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.
    dtype : {None, dtype}, optional
        The desired dtype for the array. If not given, a default dtype will be
        used that can represent the values (by considering Promotion Type Rule
        and device capabilities when necessary).
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.
    endpoint : bool, optional
        If ``True``, `stop` is the last sample. Otherwise, it is not included.
        Default is ``True``.
    retstep : bool, optional
        If ``True``, return (samples, step), where step is the spacing between
        samples.
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start or
        stop are array-like. By default (0), the samples will be along a new
        axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    out : dpnp.ndarray
        There are `num` equally spaced samples in the closed interval
        [`start`, `stop`] or the half-open interval [`start`, `stop`)
        (depending on whether `endpoint` is ``True`` or ``False``).
    step : float, optional
        Only returned if `retstep` is ``True``.
        Size of spacing between samples.

    See Also
    --------
    :obj:`dpnp.arange` : Similar to :obj:`dpnp.linspace`, but uses a step size
                         (instead of the number of samples).
    :obj:`dpnp.geomspace` : Similar to :obj:`dpnp.linspace`, but with numbers
                        spaced evenly on a log scale (a geometric progression).
    :obj:`dpnp.logspace` : Similar to :obj:`dpnp.geomspace`, but with the end
                           points specified as logarithms.

    Examples
    --------
    >>> import dpnp as np
    >>> np.linspace(2.0, 3.0, num=5)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])

    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    array([2. , 2.2, 2.4, 2.6, 2.8])

    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    (array([2.  , 2.25, 2.5 , 2.75, 3.  ]), array(0.25))

    Creating an array on a different device or with a specified usm_type

    >>> x = np.linspace(2.0, 3.0, num=3) # default case
    >>> x, x.device, x.usm_type
    (array([2. , 2.5, 3. ]), Device(level_zero:gpu:0), 'device')

    >>> y = np.linspace(2.0, 3.0, num=3, device="cpu")
    >>> y, y.device, y.usm_type
    (array([2. , 2.5, 3. ]), Device(opencl:cpu:0), 'device')

    >>> z = np.linspace(2.0, 3.0, num=3, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([2. , 2.5, 3. ]), Device(level_zero:gpu:0), 'host')

    """

    return dpnp_linspace(
        start,
        stop,
        num,
        dtype=dtype,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
        endpoint=endpoint,
        retstep=retstep,
        axis=axis,
    )


def loadtxt(
    fname,
    dtype=float,
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
    **kwargs,
):
    r"""
    Load data from a text file.

    For full documentation refer to :obj:`numpy.loadtxt`.

    Parameters
    ----------
    fname : file, str, pathlib.Path, list of str, generator
        File, filename, list, or generator to read. If the filename extension
        is ``.gz`` or ``.bz2``, the file is first decompressed. Note that
        generators must return bytes or strings. The strings in a list or
        produced by a generator are treated as lines.
    dtype : data-type, optional
        Data-type of the resulting array.
        Default is the default floating point data type for the device where
        the returned array is allocated.
        A structured data-type is not supported.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Data read from the text file.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    Notes
    -----
    This uses :obj:`numpy.loadtxt` and coerces the result to a DPNP array.

    See also
    --------
    :obj:`dpnp.frombuffer` : Construct array from the buffer data.
    :obj:`dpnp.fromstring` : Construct array from the text data in a string.
    :obj:`dpnp.fromregex` : Construct an array from a text file,
                            using regular expression parsing.
    :obj:`dpnp.load` : Load arrays or pickled objects from files.
    :obj:`dpnp.genfromtxt` : Load data with missing values handled as specified.

    Examples
    --------
    >>> import dpnp as np
    >>> from io import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\n2 3")
    >>> np.loadtxt(c)
    array([[0., 1.],
           [2., 3.]])

    Creating an array on a different device or with a specified usm_type

    >>> c = StringIO("0 1\n2 3")
    >>> x = np.loadtxt(c, dtype=np.int32) # default case
    >>> x.device, x.usm_type
    (Device(level_zero:gpu:0), 'device')

    >>> c = StringIO("0 1\n2 3")
    >>> y = np.loadtxt(c, dtype=np.int32, device='cpu')
    >>> y.device, y.usm_type
    (Device(opencl:cpu:0), 'device')

    >>> c = StringIO("0 1\n2 3")
    >>> z = np.loadtxt(c, dtype=np.int32, usm_type="host")
    >>> z.device, z.usm_type
    (Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(like=like)
    return asarray(
        numpy.loadtxt(fname, dtype=dtype, **kwargs),
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def logspace(
    start,
    stop,
    /,
    num=50,
    *,
    device=None,
    usm_type=None,
    sycl_queue=None,
    endpoint=True,
    base=10.0,
    dtype=None,
    axis=0,
):
    """
    Return numbers spaced evenly on a log scale.

    For full documentation refer to :obj:`numpy.logspace`.

    Parameters
    ----------
    start : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays. `base` ** `start` is the starting value
        of the sequence.
    stop : array_like
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays. `base` ** `stop` is the final value of
        the sequence, unless `endpoint` is ``False``. In that case, ``num + 1``
        values are spaced over the interval in log-space, of which all but
        the last (a sequence of length `num`) are returned.
    num : int, optional
        Number of samples to generate. Default is 50.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.
    endpoint : {bool}, optional
        If ``True``, stop is the last sample. Otherwise, it is not included.
        Default is ``True``.
    base : {array_like}, optional
        Input data, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays. The base of the log space, in any form
        that can be converted to an array.This includes scalars, lists, lists
        of tuples, tuples, tuples of tuples, tuples of lists, and ndarrays.
        The `step` size between the elements in ``ln(samples) / ln(base)``
        (or log_base(samples)) is uniform. Default is 10.0.
    dtype : {None, dtype}, optional
        The desired dtype for the array. If not given, a default dtype will be
        used that can represent the values (by considering Promotion Type Rule
        and device capabilities when necessary).
    axis : int, optional
        The axis in the result to store the samples. Relevant only if start,
        stop, or base are array-like. By default (0), the samples will be along
        a new axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    out: dpnp.ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    :obj:`dpnp.arange` : Similar to :obj:`dpnp.linspace`, with the step size
        specified instead of the number of samples. Note that, when used with
        a float endpoint, the endpoint may or may not be included.
    :obj:`dpnp.linspace` : Similar to :obj:`dpnp.logspace`, but with the
        samples uniformly distributed in linear space, instead of log space.
    :obj:`dpnp.geomspace` : Similar to :obj:`dpnp.logspace`, but with endpoints
                            specified directly.

    Examples
    --------
    >>> import dpnp as np
    >>> np.logspace(2.0, 3.0, num=4)
    array([ 100.        ,  215.443469  ,  464.15888336, 1000.        ])

    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)
    array([100.        , 177.827941  , 316.22776602, 562.34132519])

    >>> np.logspace(2.0, 3.0, num=4, base=2.0)
    array([4.        , 5.0396842 , 6.34960421, 8.        ])

    >>> np.logspace(2.0, 3.0, num=4, base=[2.0, 3.0], axis=-1)
    array([[ 4.        ,  5.0396842 ,  6.34960421,  8.        ],
           [ 9.        , 12.98024613, 18.72075441, 27.        ]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.logspace(1.0, 3.0, num=3) # default case
    >>> x, x.device, x.usm_type
    (array([  10.,  100., 1000.]), Device(level_zero:gpu:0), 'device')

    >>> y = np.logspace(1.0, 3.0, num=3, device="cpu")
    >>> y, y.device, y.usm_type
    (array([  10.,  100., 1000.]), Device(opencl:cpu:0), 'device')

    >>> z = np.logspace(1.0, 3.0, num=3, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([  10.,  100., 1000.]), Device(level_zero:gpu:0), 'host')

    """

    return dpnp_logspace(
        start,
        stop,
        num=num,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
        endpoint=endpoint,
        base=base,
        dtype=dtype,
        axis=axis,
    )


# pylint: disable=redefined-outer-name
def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    """
    Return coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays ``x1, x2,..., xn``.

    For full documentation refer to :obj:`numpy.meshgrid`.

    Parameters
    ----------
    x1, x2,..., xn : {dpnp.ndarray, usm_ndarray}
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian (``'xy'``, default) or matrix (``'ij'``) indexing of output.
    sparse : bool, optional
        If ``True`` the shape of the returned coordinate array for dimension `i`
        is reduced from ``(N1, ..., Ni, ... Nn)`` to
        ``(1, ..., 1, Ni, 1, ..., 1)``.
        Default: ``False``.
    copy : bool, optional
        If ``False``, a view into the original arrays are returned in order to
        conserve memory.
        Default: ``True``.

    Returns
    -------
    X1, X2,..., XN : tuple of dpnp.ndarrays
        For vectors `x1`, `x2`,..., `xn` with lengths ``Ni=len(xi)``, returns
        ``(N1, N2, N3,..., Nn)`` shaped arrays if ``indexing='ij'`` or
        ``(N2, N1, N3,..., Nn)`` shaped arrays if ``indexing='xy'`` with
        the elements of `xi` repeated to fill the matrix along the first
        dimension for `x1`, the second for `x2` and so on.

    Examples
    --------
    >>> import dpnp as np
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = np.meshgrid(x, y)
    >>> xv
    array([[0. , 0.5, 1. ],
           [0. , 0.5, 1. ]])
    >>> yv
    array([[0.,  0.,  0.],
           [1.,  1.,  1.]])
    >>> xv, yv = np.meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[0. ,  0.5,  1. ]])
    >>> yv
    array([[0.],
           [1.]])

    `meshgrid` is very useful to evaluate functions on a grid.

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(-5, 5, 0.1)
    >>> y = np.arange(-5, 5, 0.1)
    >>> xx, yy = np.meshgrid(x, y, sparse=True)
    >>> z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)
    >>> plt.show()

    """

    dpnp.check_supported_arrays_type(*xi)

    ndim = len(xi)

    if indexing not in ["xy", "ij"]:
        raise ValueError(
            "Unrecognized indexing keyword value, expecting 'xy' or 'ij'."
        )

    s0 = (1,) * ndim
    output = [
        dpnp.reshape(x, s0[:i] + (-1,) + s0[i + 1 :]) for i, x in enumerate(xi)
    ]

    if indexing == "xy" and ndim > 1:
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])

    if not sparse:
        output = dpnp.broadcast_arrays(*output)

    if copy:
        output = [x.copy() for x in output]

    return output


class MGridClass:
    """
    Construct a dense multi-dimensional "meshgrid".

    For full documentation refer to :obj:`numpy.mgrid`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : one dpnp.ndarray or tuple of dpnp.ndarray
        Returns one array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.

    Examples
    --------
    >>> import dpnp as np
    >>> np.mgrid[0:5,0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.mgrid[-1:1:5j] # default case
    >>> x, x.device, x.usm_type
    (array([-1. , -0.5,  0. ,  0.5,  1. ]), Device(level_zero:gpu:0), 'device')

    >>> y = np.mgrid(device="cpu")[-1:1:5j]
    >>> y, y.device, y.usm_type
    (array([-1. , -0.5,  0. ,  0.5,  1. ]), Device(opencl:cpu:0), 'device')

    >>> z = np.mgrid(usm_type="host")[-1:1:5j]
    >>> z, z.device, z.usm_type
    (array([-1. , -0.5,  0. ,  0.5,  1. ]), Device(level_zero:gpu:0), 'host')

    """

    def __getitem__(self, key):
        return dpnp_nd_grid(sparse=False)[key]

    def __call__(self, device=None, usm_type="device", sycl_queue=None):
        return dpnp_nd_grid(
            sparse=False,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )


mgrid = MGridClass()


class OGridClass:
    """
    Construct an open multi-dimensional "meshgrid".

    For full documentation refer to :obj:`numpy.ogrid`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : one dpnp.ndarray or tuple of dpnp.ndarray
        Returns a tuple of arrays,
        with grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)
        with dimensions[i] in the i-th place.

    Examples
    --------
    >>> import dpnp as np
    >>> np.ogrid[0:5, 0:5]
    [array([[0],
            [1],
            [2],
            [3],
            [4]]), array([[0, 1, 2, 3, 4]])]

    Creating an array on a different device or with a specified usm_type

    >>> x = np.ogrid[-1:1:5j] # default case
    >>> x, x.device, x.usm_type
    (array([-1. , -0.5,  0. ,  0.5,  1. ]), Device(level_zero:gpu:0), 'device')

    >>> y = np.ogrid(device="cpu")[-1:1:5j]
    >>> y, y.device, y.usm_type
    (array([-1. , -0.5,  0. ,  0.5,  1. ]), Device(opencl:cpu:0), 'device')

    >>> z = np.ogrid(usm_type="host")[-1:1:5j]
    >>> z, z.device, z.usm_type
    (array([-1. , -0.5,  0. ,  0.5,  1. ]), Device(level_zero:gpu:0), 'host')

    """

    def __getitem__(self, key):
        return dpnp_nd_grid(sparse=True)[key]

    def __call__(self, device=None, usm_type="device", sycl_queue=None):
        return dpnp_nd_grid(
            sparse=True, device=device, usm_type=usm_type, sycl_queue=sycl_queue
        )


ogrid = OGridClass()


def ones(
    shape,
    *,
    dtype=None,
    order="C",
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return a new array of given shape and type, filled with ones.

    For full documentation refer to :obj:`numpy.ones`.

    Parameters
    ----------
    shape : {int, sequence of ints}
        Shape of the new array, e.g., (2, 3) or 2.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    order : {None, "C", "F"}, optional
        Memory layout of the newly output array. Default: "C".
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of ones with the given shape, dtype, and order.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"``, ``"F"`` and
    ``None``.
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of
                            input.
    :obj:`dpnp.empty` : Return a new uninitialized array.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import dpnp as np
    >>> np.ones(5)
    array([1., 1., 1., 1., 1.])
    >>> x = np.ones((2, 1))
    >>> x.ndim, x.size, x.shape
    (2, 2, (2, 1))
    >>> x
    array([[1.],
           [1.]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.ones(3) # default case
    >>> x, x.device, x.usm_type
    (array([1., 1., 1.]), Device(level_zero:gpu:0), 'device')

    >>> y = np.ones(3, device="cpu")
    >>> y, y.device, y.usm_type
    (array([1., 1., 1.]), Device(opencl:cpu:0), 'device')

    >>> z = np.ones(3, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1., 1., 1.]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(order=order, like=like)

    if usm_type is None:
        usm_type = "device"

    return dpnp_container.ones(
        shape,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def ones_like(
    a,
    /,
    *,
    dtype=None,
    order="C",
    subok=False,
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Return an array of ones with the same shape and type as a given array.

    For full documentation refer to :obj:`numpy.ones_like`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        The shape and dtype of `a` define these same attributes
        of the returned array.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    order : {None, "C", "F"}, optional
        Memory layout of the newly output array. Default: "C".
    shape : {None, int, sequence of ints}
        Overrides the shape of the result.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of ones with the same shape and type as `a`.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"``, ``"F"`` and
    ``None``.
    Parameter `subok` is supported only with default value ``False``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of
                             input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of
                             input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with
                            value.
    :obj:`dpnp.ones` : Return a new array setting values to one.

    Examples
    --------
    >>> import dpnp as np
    >>> x0 = np.arange(6)
    >>> x0
    array([0, 1, 2, 3, 4, 5])
    >>> np.ones_like(x0)
    array([1, 1, 1, 1, 1, 1])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.ones_like(x0) # default case
    >>> x, x.device, x.usm_type
    (array([1, 1, 1, 1, 1, 1]), Device(level_zero:gpu:0), 'device')

    >>> y = np.ones_like(x0, device="cpu")
    >>> y, y.device, y.usm_type
    (array([1, 1, 1, 1, 1, 1]), Device(opencl:cpu:0), 'device')

    >>> z = np.ones_like(x0, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1, 1, 1, 1, 1, 1]), Device(level_zero:gpu:0), 'host')

    """
    dpnp.check_supported_arrays_type(a)
    dpnp.check_limitations(order=order, subok=subok)

    _shape = a.shape if shape is None else shape
    _dtype = a.dtype if dtype is None else dtype
    _usm_type = a.usm_type if usm_type is None else usm_type
    _sycl_queue = dpnp.get_normalized_queue_device(
        a, sycl_queue=sycl_queue, device=device
    )
    return dpnp_container.ones(
        _shape,
        dtype=_dtype,
        order=order,
        usm_type=_usm_type,
        sycl_queue=_sycl_queue,
    )


def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """
    Return the sum along diagonals of the array.

    For full documentation refer to :obj:`numpy.trace`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array, from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both positive and
        negative.
        Default: ``0``.
    axis1, axis2 : int, optional
        Axes to be used as the first and second axis of the 2-D sub-arrays from
        which the diagonals should be taken. Defaults are the first two axes of
        `a`.
    dtype : dtype, optional
        Determines the data-type of the returned array and of the accumulator
        where the elements are summed. If `dtype` has the value ``None`` and
        `a` is of integer type of precision less than the default integer
        precision, then the default integer precision is used. Otherwise, the
        precision is the same as that of `a`.
        Default: ``None``.
    out : {dpnp.ndarray, usm_ndarray}, optional
        Array into which the output is placed. Its type is preserved and it
        must be of the right shape to hold the output.
        Default: ``None``.

    Returns
    -------
    sum_along_diagonals : dpnp.ndarray
        If `a` is 2-D, the sum along the diagonal is returned. If `a` has
        larger dimensions, then an array of sums along diagonals is returned.

    See Also
    --------
    :obj:`dpnp.diag` : Extract a diagonal or construct a diagonal array.
    :obj:`dpnp.diagonal` : Return specified diagonals.
    :obj:`dpnp.diagflat` : Create a 2-D array with the flattened input as
                           a diagonal.

    Examples
    --------
    >>> import dpnp as np
    >>> np.trace(np.eye(3))
    array(3.)
    >>> a = np.arange(8).reshape((2, 2, 2))
    >>> np.trace(a)
    array([6, 8])

    >>> a = np.arange(24).reshape((2, 2, 2, 3))
    >>> np.trace(a).shape
    (2, 3)

    """

    d = dpnp.diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
    return dpnp.sum(d, axis=-1, dtype=dtype, out=out, keepdims=False)


# pylint: disable=invalid-name
def tri(
    N,
    /,
    M=None,
    k=0,
    dtype=dpnp.float,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    For full documentation refer to :obj:`numpy.tri`.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : {None, int}, optional
        Number of columns in the array. By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled. k = 0 is
        the main diagonal, while k < 0 is below it, and k > 0 is above.
        The default is 0.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray of shape (N, M)
        Array with its lower triangle filled with ones and zeros elsewhere.

    See Also
    --------
    :obj:`dpnp.tril` : Return lower triangle of an array.
    :obj:`dpnp.triu` : Return upper triangle of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> np.tri(3, 5, 2, dtype=int)
    array([[1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1]])

    >>> np.tri(3, 5, -1)
    array([[0.,  0.,  0.,  0.,  0.],
           [1.,  0.,  0.,  0.,  0.],
           [1.,  1.,  0.,  0.,  0.]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.tri(3, 2) # default case
    >>> x, x.device, x.usm_type
    (array([[1., 0.],
            [1., 1.],
            [1., 1.]]), Device(level_zero:gpu:0), 'device')

    >>> y = np.tri(3, 2, device="cpu")
    >>> y, y.device, y.usm_type
    (array([[1., 0.],
            [1., 1.],
            [1., 1.]]), Device(opencl:cpu:0), 'device')

    >>> z = np.tri(3, 2, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([[1., 0.],
            [1., 1.],
            [1., 1.]]), Device(level_zero:gpu:0), 'host')

    """

    if not isinstance(N, int):
        raise TypeError(f"`N` must be a integer data type, but got {type(N)}")
    if N < 0:
        raise ValueError("negative dimensions are not allowed")

    if M is not None:
        if not isinstance(M, int):
            raise TypeError(
                f"`M` must be a integer data type, but got {type(M)}"
            )
        if M < 0:
            raise ValueError("negative dimensions are not allowed")
    else:
        M = N

    _k = None
    try:
        _k = operator.index(k)
    except TypeError:
        pass
    if _k is None:
        raise TypeError(f"`k` must be a integer data type, but got {type(k)}")

    _dtype = dpnp.default_float_type() if dtype in (dpnp.float, None) else dtype

    m = dpnp.ones(
        (N, M),
        dtype=_dtype,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )
    return dpnp.tril(m, k=_k)


def tril(m, /, *, k=0):
    """
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    For full documentation refer to :obj:`numpy.tril`.

    Parameters
    ----------
    m : {dpnp.ndarray, usm_ndarray}, shape (…, M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements. k = 0 (the default) is
        the main diagonal, k < 0 is below it and k > 0 is above.

    Returns
    -------
    out : dpnp.ndarray of shape (N, M)
        Lower triangle of `m`, of same shape and dtype as `m`.

    Examples
    --------
    >>> import dpnp as np
    >>> m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> np.tril(m, k=-1)
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])

    """

    _k = None
    try:
        _k = operator.index(k)
    except TypeError:
        pass
    if _k is None:
        raise TypeError(f"`k` must be a integer data type, but got {type(k)}")

    dpnp.check_supported_arrays_type(m)

    if m.ndim == 0:
        raise ValueError("Input array must have 1 or more dimensions")

    if m.ndim == 1:
        m = dpnp.broadcast_to(m, (m.shape[0], m.shape[0]))

    return dpnp_container.tril(m, k=_k)


def triu(m, /, *, k=0):
    """
    Upper triangle of an array.

    Return a copy of a matrix with the elements below the `k`-th diagonal
    zeroed.

    For full documentation refer to :obj:`numpy.triu`.

    Parameters
    ----------
    m : {dpnp.ndarray, usm_ndarray}, shape (…, M, N)
        Input array.
    k : int, optional
        Diagonal below which to zero elements. k = 0 (the default) is
        the main diagonal, k < 0 is below it and k > 0 is above.

    Returns
    -------
    out : dpnp.ndarray of shape (N, M)
        Upper triangle of `m`, of same shape and dtype as `m`.

    Examples
    --------
    >>> import dpnp as np
    >>> m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> np.triu(m, k=-1)
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])

    """

    _k = None
    try:
        _k = operator.index(k)
    except TypeError:
        pass
    if _k is None:
        raise TypeError(f"`k` must be a integer data type, but got {type(k)}")

    dpnp.check_supported_arrays_type(m)

    if m.ndim == 0:
        raise ValueError("Input array must have 1 or more dimensions")

    if m.ndim == 1:
        m = dpnp.broadcast_to(m, (m.shape[0], m.shape[0]))

    return dpnp_container.triu(m, k=_k)


# pylint: disable=invalid-name
def vander(
    x,
    /,
    N=None,
    increasing=False,
    *,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Generate a Vandermonde matrix.

    For full documentation refer to :obj:`numpy.vander`.

    Parameters
    ----------
    x : array_like
        1-D input array, in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    N : {None, int}, optional
        Number of columns in the output. If `N` is not specified, a square
        array is returned ``(N = len(x))``.
    increasing : bool, optional
        Order of the powers of the columns. If ``True,`` the powers increase
        from left to right, if ``False`` (the default) they are reversed.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Vandermonde matrix.

    Examples
    --------
    >>> import dpnp as np
    >>> x0 = np.array([1, 2, 3, 5])
    >>> N = 3
    >>> np.vander(x0, N)
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])

    >>> np.vander(x0)
    array([[  1,   1,   1,   1],
           [  8,   4,   2,   1],
           [ 27,   9,   3,   1],
           [125,  25,   5,   1]])

    >>> np.vander(x0, increasing=True)
    array([[  1,   1,   1,   1],
           [  1,   2,   4,   8],
           [  1,   3,   9,  27],
           [  1,   5,  25, 125]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.vander(x0) # default case
    >>> x, x.device, x.usm_type
    (array([[  1,   1,   1,   1],
            [  8,   4,   2,   1],
            [ 27,   9,   3,   1],
            [125,  25,   5,   1]]), Device(level_zero:gpu:0), 'device')

    >>> y = np.vander(x0, device="cpu")
    >>> y, y.device, y.usm_type
    (array([[  1,   1,   1,   1],
            [  8,   4,   2,   1],
            [ 27,   9,   3,   1],
            [125,  25,   5,   1]]), Device(opencl:cpu:0), 'device')

    >>> z = np.vander(x0, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([[  1,   1,   1,   1],
            [  8,   4,   2,   1],
            [ 27,   9,   3,   1],
            [125,  25,   5,   1]]), Device(level_zero:gpu:0), 'host')
    """

    x = dpnp.asarray(x, device=device, usm_type=usm_type, sycl_queue=sycl_queue)

    if N is not None and not isinstance(N, int):
        raise TypeError(f"An integer is required, but got {type(N)}")

    if x.ndim != 1:
        raise ValueError("`x` must be a one-dimensional array or sequence.")

    if N is None:
        N = x.size

    _dtype = int if x.dtype == bool else x.dtype
    m = empty(
        (x.size, N),
        dtype=_dtype,
        usm_type=x.usm_type,
        sycl_queue=x.sycl_queue,
    )
    tmp = m[:, ::-1] if not increasing else m
    dpnp.power(
        x.reshape(-1, 1),
        dpnp.arange(N, dtype=_dtype, sycl_queue=x.sycl_queue),
        out=tmp,
    )
    return m


def zeros(
    shape,
    *,
    dtype=None,
    order="C",
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return a new array of given shape and type, filled with zeros.

    For full documentation refer to :obj:`numpy.zeros`.

    Parameters
    ----------
    shape : {int, sequence of ints}
        Shape of the new array, e.g., (2, 3) or 2.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    order : {None, "C", "F"}, optional
        Memory layout of the newly output array. Default: "C".
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``"device"``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of zeros with the given shape, dtype, and order.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"``, ``"F"`` and
    ``None``.
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of
                             input.
    :obj:`dpnp.empty` : Return a new uninitialized array.
    :obj:`dpnp.ones` : Return a new array setting values to one.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import dpnp as np
    >>> np.zeros(5)
    array([0., 0., 0., 0., 0.])
    >>> x = np.zeros((2, 1))
    >>> x.ndim, x.size, x.shape
    (2, 2, (2, 1))
    >>> x
    array([[0.],
           [0.]])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.zeros(3) # default case
    >>> x, x.device, x.usm_type
    (array([0., 0., 0.]), Device(level_zero:gpu:0), 'device')

    >>> y = np.zeros(3, device="cpu")
    >>> y, y.device, y.usm_type
    (array([0., 0., 0.]), Device(opencl:cpu:0), 'device')

    >>> z = np.zeros(3, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([0., 0., 0.]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_limitations(order=order, like=like)

    if usm_type is None:
        usm_type = "device"

    return dpnp_container.zeros(
        shape,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def zeros_like(
    a,
    /,
    *,
    dtype=None,
    order="C",
    subok=False,
    shape=None,
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Return an array of zeros with the same shape and type as a given array.

    For full documentation refer to :obj:`numpy.zeros_like`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        The shape and dtype of `a` define these same attributes
        of the returned array.
    dtype : {None, dtype}, optional
        The desired dtype for the array, e.g., dpnp.int32.
        Default is the default floating point data type for the device where
        input array is allocated.
    order : {None, "C", "F"}, optional
        Memory layout of the newly output array. Default: "C".
    shape : {None, int, sequence of ints}
        Overrides the shape of the result.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of zeros with the same shape and type as `a`.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"``, ``"F"`` and
    ``None``.
    Parameter `subok` is supported only with default value ``False``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of
                             input.
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of
                            input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled
                            with value.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> x0 = np.arange(6)
    >>> x0
    array([0, 1, 2, 3, 4, 5])
    >>> np.zeros_like(x0)
    array([0, 0, 0, 0, 0, 0])

    Creating an array on a different device or with a specified usm_type

    >>> x = np.zeros_like(x0) # default case
    >>> x, x.device, x.usm_type
    (array([0, 0, 0, 0, 0, 0]), Device(level_zero:gpu:0), 'device')

    >>> y = np.zeros_like(x0, device="cpu")
    >>> y, y.device, y.usm_type
    (array([0, 0, 0, 0, 0, 0]), Device(opencl:cpu:0), 'device')

    >>> z = np.zeros_like(x0, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([0, 0, 0, 0, 0, 0]), Device(level_zero:gpu:0), 'host')

    """

    dpnp.check_supported_arrays_type(a)
    dpnp.check_limitations(order=order, subok=subok)

    _shape = a.shape if shape is None else shape
    _dtype = a.dtype if dtype is None else dtype
    _usm_type = a.usm_type if usm_type is None else usm_type
    _sycl_queue = dpnp.get_normalized_queue_device(
        a, sycl_queue=sycl_queue, device=device
    )
    return dpnp_container.zeros(
        _shape,
        dtype=_dtype,
        order=order,
        usm_type=_usm_type,
        sycl_queue=_sycl_queue,
    )
