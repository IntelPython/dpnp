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

import dpctl.tensor as dpt
import numpy

import dpnp
import dpnp.dpnp_container as dpnp_container
from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *

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
    "ptp",
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

    Returns
    -------
    out : dpnp.ndarray
        The 1-D array containing evenly spaced values.

    Limitations
    -----------
    Parameter ``like`` is supported only with default value ``None``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.linspace` : Evenly spaced numbers with careful handling of endpoints.

    Examples
    --------
    >>> import dpnp as np
    >>> [i for i in np.arange(3)]
    [0, 1, 2]
    >>> [i for i in np.arange(3, 7)]
    [3, 4, 5, 6]
    >>> [i for i in np.arange(3, 7, 2)]
    [3, 5]

    """

    if like is None:
        return dpnp_container.arange(
            start,
            stop=stop,
            step=step,
            dtype=dtype,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )

    return call_origin(
        numpy.arange, start, stop=stop, step=step, dtype=dtype, like=like
    )


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

    Returns
    -------
    out : dpnp.ndarray
        An array object satisfying the specified requirements.

    Limitations
    -----------
    Parameter `subok` is supported only with default value ``False``.
    Parameter `ndmin` is supported only with default value ``0``.
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `ValueError` exception.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of input.
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with value.
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
    >>> print(x)
    [1 2 3]

    More than one dimension:

    >>> x2 = np.array([[1, 2], [3, 4]])
    >>> x2.ndim, x2.size, x2.shape
    (2, 4, (2, 2))
    >>> print(x2)
    [[1 2]
     [3 4]]

    """

    if subok is not False:
        raise ValueError(
            "Keyword argument `subok` is supported only with "
            f"default value ``False``, but got {subok}"
        )
    elif ndmin != 0:
        raise ValueError(
            "Keyword argument `ndmin` is supported only with "
            f"default value ``0``, but got {ndmin}"
        )
    elif like is not None:
        raise ValueError(
            "Keyword argument `like` is supported only with "
            f"default value ``None``, but got {like}"
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

    Returns
    -------
    out : dpnp.ndarray
        Array interpretation of `a`.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `ValueError` exception.

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

    """

    if like is not None:
        raise ValueError(
            "Keyword argument `like` is supported only with "
            f"default value ``None``, but got {like}"
        )

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

    Returns
    -------
    out : dpnp.ndarray
        Array interpretation of `a`. No copy is performed if the input
        is already an ndarray with matching dtype and order.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `ValueError` exception.

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

    """

    if like is not None:
        raise ValueError(
            "Keyword argument `like` is supported only with "
            f"default value ``None``, but got {like}"
        )

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
    Return a contiguous array (ndim >= 1) in memory (C order).

    For full documentation refer to :obj:`numpy.ascontiguousarray`.

    Returns
    -------
    out : dpnp.ndarray
        Contiguous array of same shape and content as `a`, with type `dtype`
        if specified.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `ValueError` exception.

    See Also
    --------
    :obj:`dpnp.asfortranarray` : Convert input to an ndarray with column-major
                     memory order.
    :obj:`dpnp.require` : Return an ndarray that satisfies requirements.
    :obj:`dpnp.ndarray.flags` : Information about the memory layout of the array.

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

    """

    if like is not None:
        raise ValueError(
            "Keyword argument `like` is supported only with "
            f"default value ``None``, but got {like}"
        )

    # at least 1-d array has to be returned
    if a.ndim == 0:
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
    Return an array (ndim >= 1) laid out in Fortran order in memory.

    For full documentation refer to :obj:`numpy.asfortranarray`.

    Returns
    -------
    out : dpnp.ndarray
        The input `a` in Fortran, or column-major, order.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `ValueError` exception.

    See Also
    --------
    :obj:`dpnp.ascontiguousarray` : Convert input to a contiguous (C order) array.
    :obj:`dpnp.asanyarray` : Convert input to an ndarray with either row or column-major memory order.
    :obj:`dpnp.require` : Return an ndarray that satisfies requirements.
    :obj:`dpnp.ndarray.flags` : Information about the memory layout of the array.

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

    """

    if like is not None:
        raise ValueError(
            "Keyword argument `like` is supported only with "
            f"default value ``None``, but got {like}"
        )

    # at least 1-d array has to be returned
    if a.ndim == 0:
        a = [a]

    return asarray(
        a,
        dtype=dtype,
        order="F",
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def copy(a, order="K", subok=False):
    """
    Return an array copy of the given object.

    For full documentation refer to :obj:`numpy.copy`.

    Limitations
    -----------
    Parameter `subok` is supported only with default value ``False``.
    Otherwise, the function raises `ValueError` exception.

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

    """

    if subok is not False:
        raise ValueError(
            "Keyword argument `subok` is supported only with "
            f"default value ``False``, but got {subok}"
        )

    if dpnp.is_supported_array_type(a):
        return dpnp_container.copy(a, order=order)

    return array(a, order=order, subok=subok, copy=True)


def diag(x1, k=0):
    """
    Extract a diagonal or construct a diagonal array.

    For full documentation refer to :obj:`numpy.diag`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(9).reshape((3,3))
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

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if not isinstance(k, int):
            pass
        elif x1_desc.ndim != 1 and x1_desc.ndim != 2:
            pass
        else:
            return dpnp_diag(x1_desc, k).get_pyobj()

    return call_origin(numpy.diag, x1, k)


def diagflat(x1, k=0):
    """
    Create a two-dimensional array with the flattened input as a diagonal.

    For full documentation refer to :obj:`numpy.diagflat`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.diagflat([[1,2], [3,4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])

    >>> np.diagflat([1,2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        input_ravel = dpnp.ravel(x1)
        input_ravel_desc = dpnp.get_dpnp_descriptor(
            input_ravel, copy_when_nondefault_queue=False
        )

        return dpnp_diag(input_ravel_desc, k).get_pyobj()

    return call_origin(numpy.diagflat, x1, k)


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

    Limitations
    -----------
    Parameter ``order`` is supported only with values ``"C"`` and ``"F"``.
    Parameter ``like`` is supported only with default value ``None``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of input.
    :obj:`dpnp.ones` : Return a new array setting values to one.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.empty(4)
    >>> print(x)
    [0. 0. 0. 0.]

    """

    if like is not None:
        pass
    elif order not in ("C", "c", "F", "f", None):
        pass
    else:
        return dpnp_container.empty(
            shape,
            dtype=dtype,
            order=order,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )

    return call_origin(numpy.empty, shape, dtype=dtype, order=order, like=like)


def empty_like(
    x1,
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

    Limitations
    -----------
    Parameter ``x1`` is supported as :class:`dpnp.dpnp_array` or :class:`dpctl.tensor.usm_ndarray`
    Parameter ``order`` is supported with values ``"C"`` or ``"F"``.
    Parameter ``subok`` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with value.
    :obj:`dpnp.empty` : Return a new uninitialized array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> x = np.empty_like(a)
    >>> [i for i in x]
    [0, 0, 0]

    """

    if not isinstance(x1, (dpnp.ndarray, dpt.usm_ndarray)):
        pass
    elif order not in ("C", "c", "F", "f", None):
        pass
    elif subok is not False:
        pass
    else:
        _shape = x1.shape if shape is None else shape
        _dtype = x1.dtype if dtype is None else dtype
        _usm_type = x1.usm_type if usm_type is None else usm_type
        _sycl_queue = dpnp.get_normalized_queue_device(
            x1, sycl_queue=sycl_queue, device=device
        )
        return dpnp_container.empty(
            _shape,
            dtype=_dtype,
            order=order,
            usm_type=_usm_type,
            sycl_queue=_sycl_queue,
        )

    return call_origin(numpy.empty_like, x1, dtype, order, subok, shape)


def eye(
    N,
    M=None,
    /,
    *,
    k=0,
    dtype=None,
    order="C",
    like=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    For full documentation refer to :obj:`numpy.eye`.

    Limitations
    -----------
    Parameter ``order`` is supported only with values ``"C"`` and ``"F"``.
    Parameter ``like`` is supported only with default value ``None``.
    Otherwise the function will be executed sequentially on CPU.

    """
    if order not in ("C", "c", "F", "f", None):
        pass
    elif like is not None:
        pass
    else:
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

    return call_origin(
        numpy.eye, N, M, k=k, dtype=dtype, order=order, like=None
    )


def frombuffer(buffer, **kwargs):
    """
    Interpret a buffer as a 1-dimensional array.

    For full documentation refer to :obj:`numpy.frombuffer`.

    Limitations
    -----------
    Only float64, float32, int64, int32 types are supported.

    """

    return call_origin(numpy.frombuffer, buffer, **kwargs)


def fromfile(file, **kwargs):
    """
    Construct an array from data in a text or binary file.

    A highly efficient way of reading binary data with a known data-type,
    as well as parsing simply formatted text files.  Data written using the
    `tofile` method can be read using this function.

    For full documentation refer to :obj:`numpy.fromfile`.

    Limitations
    -----------
    Only float64, float32, int64, int32 types are supported.

    """

    return call_origin(numpy.fromfile, file, **kwargs)


def fromfunction(function, shape, **kwargs):
    """
    Construct an array by executing a function over each coordinate.

    The resulting array therefore has a value ``fn(x, y, z)`` at
    coordinate ``(x, y, z)``.

    For full documentation refer to :obj:`numpy.fromfunction`.

    Limitations
    -----------
    Only float64, float32, int64, int32 types are supported.

    """

    return call_origin(numpy.fromfunction, function, shape, **kwargs)


def fromiter(iterable, dtype, count=-1):
    """
    Create a new 1-dimensional array from an iterable object.

    For full documentation refer to :obj:`numpy.fromiter`.

    Limitations
    -----------
    Only float64, float32, int64, int32 types are supported.

    """

    return call_origin(numpy.fromiter, iterable, dtype, count)


def fromstring(string, **kwargs):
    """
    A new 1-D array initialized from text data in a string.

    For full documentation refer to :obj:`numpy.fromstring`.

    Limitations
    -----------
    Only float64, float32, int64, int32 types are supported.

    """

    return call_origin(numpy.fromstring, string, **kwargs)


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

    Limitations
    -----------
    Parameter ``order`` is supported only with values ``"C"`` and ``"F"``.
    Parameter ``like`` is supported only with default value ``None``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with value.
    :obj:`dpnp.empty` : Return a new uninitialized array.
    :obj:`dpnp.ones` : Return a new array setting values to one.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.full(4, 10)
    >>> [i for i in x]
    [10, 10, 10, 10]

    """
    if like is not None:
        pass
    elif order not in ("C", "c", "F", "f", None):
        pass
    else:
        return dpnp_container.full(
            shape,
            fill_value,
            dtype=dtype,
            order=order,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )

    return call_origin(numpy.full, shape, fill_value, dtype, order, like=like)


def full_like(
    x1,
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

    Limitations
    -----------
    Parameter ``x1`` is supported as :class:`dpnp.dpnp_array` or :class:`dpctl.tensor.usm_ndarray`
    Parameter ``order`` is supported only with values ``"C"`` and ``"F"``.
    Parameter ``subok`` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of input.
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of input.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(6)
    >>> x = np.full_like(a, 1)
    >>> [i for i in x]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    """
    if not isinstance(x1, (dpnp.ndarray, dpt.usm_ndarray)):
        pass
    elif order not in ("C", "c", "F", "f", None):
        pass
    elif subok is not False:
        pass
    else:
        _shape = x1.shape if shape is None else shape
        _dtype = x1.dtype if dtype is None else dtype
        _usm_type = x1.usm_type if usm_type is None else usm_type
        _sycl_queue = dpnp.get_normalized_queue_device(
            x1, sycl_queue=sycl_queue, device=device
        )

        return dpnp_container.full(
            _shape,
            fill_value,
            dtype=_dtype,
            order=order,
            usm_type=_usm_type,
            sycl_queue=_sycl_queue,
        )
    return numpy.full_like(x1, fill_value, dtype, order, subok, shape)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """
    Return numbers spaced evenly on a log scale (a geometric progression).

    For full documentation refer to :obj:`numpy.geomspace`.

    Limitations
    -----------
    Parameter ``axis`` is supported only with default value ``0``.

    See Also
    --------
    :obj:`dpnp.logspace` : Similar to geomspace, but with endpoints specified
                           using log and base.
    :obj:`dpnp.linspace` : Similar to geomspace, but with arithmetic instead of
                           geometric progression.
    :obj:`dpnp.arange` : Similar to linspace, with the step size specified
                         instead of the number of samples.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.geomspace(1, 1000, num=4)
    >>> [i for i in x]
    [1.0, 10.0, 100.0, 1000.0]
    >>> x2 = np.geomspace(1, 1000, num=4, endpoint=False)
    >>> [i for i in x2]
    [1.0, 5.62341325, 31.6227766, 177.827941]

    """

    if not use_origin_backend():
        if axis != 0:
            pass
        else:
            return dpnp_geomspace(
                start, stop, num, endpoint, dtype, axis
            ).get_pyobj()

    return call_origin(numpy.geomspace, start, stop, num, endpoint, dtype, axis)


def identity(n, dtype=None, *, like=None):
    """
    Return the identity array.

    The identity array is a square array with ones on the main diagonal.

    For full documentation refer to :obj:`numpy.identity`.

    Limitations
    -----------
    Parameter ``like`` is currently not supported .

    Examples
    --------
    >>> import dpnp as np
    >>> np.identity(3)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])

    """
    if not use_origin_backend():
        if like is not None:
            pass
        elif n < 0:
            pass
        else:
            _dtype = dpnp.default_float_type() if dtype is None else dtype
            return dpnp_identity(n, _dtype).get_pyobj()

    return call_origin(numpy.identity, n, dtype=dtype, like=like)


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

    Limitations
    -----------
    Parameter ``axis`` is supported only with default value ``0``.
    Parameter ``retstep`` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.arange` : Similar to `linspace`, but uses a step size (instead
                         of the number of samples).
    :obj:`dpnp.geomspace` : Similar to `linspace`, but with numbers spaced
                            evenly on a log scale (a geometric progression).
    :obj:`dpnp.logspace` : Similar to `geomspace`, but with the end points
                           specified as logarithms.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.linspace(2.0, 3.0, num=5)
    >>> [i for i in x]
    [2.0, 2.25, 2.5, 2.75, 3.0]
    >>> x2 = np.linspace(2.0, 3.0, num=5, endpoint=False)
    >>> [i for i in x2]
    [2.0, 2.2, 2.4, 2.6, 2.8]
    >>> x3, step = np.linspace(2.0, 3.0, num=5, retstep=True)
    >>> [i for i in x3], step
    ([2.0, 2.25, 2.5, 2.75, 3.0], 0.25)

    """

    if retstep is not False:
        pass
    elif axis != 0:
        pass
    else:
        return dpnp_linspace(
            start,
            stop,
            num,
            dtype=dtype,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
            endpoint=endpoint,
        )

    return call_origin(
        numpy.linspace, start, stop, num, endpoint, retstep, dtype, axis
    )


def loadtxt(fname, **kwargs):
    r"""
    Load data from a text file.

    Each row in the text file must have the same number of values.

    For full documentation refer to :obj:`numpy.loadtxt`.

    Limitations
    -----------
    Only float64, float32, int64, int32 types are supported.

    Examples
    --------
    >>> import dpnp as np
    >>> from io import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\n2 3")
    >>> np.loadtxt(c)
    array([[0., 1.],
           [2., 3.]])

    """

    return call_origin(numpy.loadtxt, fname, **kwargs)


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """
    Return numbers spaced evenly on a log scale.

    For full documentation refer to :obj:`numpy.logspace`.

    Limitations
    -----------
    Parameter ``axis`` is supported only with default value ``0``.

    See Also
    --------
    :obj:`dpnp.arange` : Similar to linspace, with the step size specified
                         instead of the number of samples. Note that, when used
                         with a float endpoint, the endpoint may or may not be
                         included.
    :obj:`dpnp.linspace` : Similar to logspace, but with the samples uniformly
                           distributed in linear space, instead of log space.
    :obj:`dpnp.geomspace` : Similar to logspace, but with endpoints specified
                            directly.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.logspace(2.0, 3.0, num=4)
    >>> [i for i in x]
    [100.0, 215.443469, 464.15888336, 1000.0]
    >>> x2 = np.logspace(2.0, 3.0, num=4, endpoint=False)
    >>> [i for i in x2]
    [100.0, 177.827941, 316.22776602, 562.34132519]
    >>> x3 = np.logspace(2.0, 3.0, num=4, base=2.0)
    >>> [i for i in x3]
    [4.0, 5.0396842, 6.34960421, 8.0]

    """

    if not use_origin_backend():
        if axis != 0:
            checker_throw_value_error("linspace", "axis", axis, 0)

        return dpnp_logspace(
            start, stop, num, endpoint, base, dtype, axis
        ).get_pyobj()

    return call_origin(
        numpy.logspace, start, stop, num, endpoint, base, dtype, axis
    )


def meshgrid(*xi, copy=True, sparse=False, indexing="xy"):
    """
    Return coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    For full documentation refer to :obj:`numpy.meshgrid`.

    Limitations
    -----------
    Each array instance from `xi` is supported as either :class:`dpnp.dpnp_array` or :class:`dpctl.tensor.usm_ndarray`.
    Parameter ``copy`` is supported only with default value ``True``.
    Parameter ``sparse`` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.

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

    if not all((isinstance(x, (dpnp.ndarray, dpt.usm_ndarray)) for x in xi)):
        pass
    elif indexing not in ["ij", "xy"]:
        pass
    elif copy is not True:
        pass
    elif sparse is not False:
        pass
    else:
        return dpnp_container.meshgrid(*xi, indexing=indexing)

    return call_origin(numpy.meshgrid, xi, copy, sparse, indexing)


class MGridClass:
    """
    Construct a dense multi-dimensional "meshgrid".

    For full documentation refer to :obj:`numpy.mgrid`.

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
    >>> np.mgrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    """

    def __getitem__(self, key):
        return dpnp.array(numpy.mgrid[key])


mgrid = MGridClass()


class OGridClass:
    """
    Construct an open multi-dimensional "meshgrid".

    For full documentation refer to :obj:`numpy.ogrid`.

    Examples
    --------
    >>> import dpnp as np
    >>> from numpy import ogrid
    >>> ogrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> ogrid[0:5,0:5]
    [array([[0],
            [1],
            [2],
            [3],
            [4]]), array([[0, 1, 2, 3, 4]])]

    """

    def __getitem__(self, key):
        return dpnp.array(numpy.ogrid[key])


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

    Limitations
    -----------
    Parameter ``order`` is supported only with values ``"C"`` and ``"F"``.
    Parameter ``like`` is supported only with default value ``None``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of input.
    :obj:`dpnp.empty` : Return a new uninitialized array.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import dpnp as np
    >>> [i for i in np.ones(5)]
    [1.0, 1.0, 1.0, 1.0, 1.0]
    >>> x = np.ones((2, 1))
    >>> x.ndim, x.size, x.shape
    (2, 2, (2, 1))
    >>> [i for i in x]
    [1.0, 1.0]

    """

    if like is not None:
        pass
    elif order not in ("C", "c", "F", "f", None):
        pass
    else:
        return dpnp_container.ones(
            shape,
            dtype=dtype,
            order=order,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )

    return call_origin(numpy.ones, shape, dtype=dtype, order=order, like=like)


def ones_like(
    x1,
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

    Limitations
    -----------
    Parameter ``x1`` is supported as :class:`dpnp.dpnp_array` or :class:`dpctl.tensor.usm_ndarray`
    Parameter ``order`` is supported with values ``"C"`` or ``"F"``.
    Parameter ``subok`` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with value.
    :obj:`dpnp.ones` : Return a new array setting values to one.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(6)
    >>> [i for i in x]
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    >>> [i for i in np.ones_like(x)]
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    """
    if not isinstance(x1, (dpnp.ndarray, dpt.usm_ndarray)):
        pass
    elif order not in ("C", "c", "F", "f", None):
        pass
    elif subok is not False:
        pass
    else:
        _shape = x1.shape if shape is None else shape
        _dtype = x1.dtype if dtype is None else dtype
        _usm_type = x1.usm_type if usm_type is None else usm_type
        _sycl_queue = dpnp.get_normalized_queue_device(
            x1, sycl_queue=sycl_queue, device=device
        )
        return dpnp_container.ones(
            _shape,
            dtype=_dtype,
            order=order,
            usm_type=_usm_type,
            sycl_queue=_sycl_queue,
        )

    return call_origin(numpy.ones_like, x1, dtype, order, subok, shape)


def ptp(arr, axis=None, out=None, keepdims=numpy._NoValue):
    """
    Range of values (maximum - minimum) along an axis.

    For full documentation refer to :obj:`numpy.ptp`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameters ``out`` and ``keepdims`` are supported only with default values.
    """
    arr_desc = dpnp.get_dpnp_descriptor(arr, copy_when_nondefault_queue=False)
    if not arr_desc:
        pass
    elif axis is not None and not isinstance(axis, int):
        pass
    elif out is not None:
        pass
    elif keepdims is not numpy._NoValue:
        pass
    else:
        result_obj = dpnp_ptp(arr_desc, axis=axis).get_pyobj()
        result = dpnp.convert_single_elem_array_to_scalar(result_obj)

        return result

    return call_origin(numpy.ptp, arr, axis, out, keepdims)


def trace(x1, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """
    Return the sum along diagonals of the array.

    For full documentation refer to :obj:`numpy.trace`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameters ``axis1``, ``axis2``, ``out`` and ``dtype`` are supported only with default values.
    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if x1_desc.size == 0:
            pass
        elif x1_desc.ndim < 2:
            pass
        elif axis1 != 0:
            pass
        elif axis2 != 1:
            pass
        elif out is not None:
            pass
        else:
            return dpnp_trace(
                x1_desc, offset, axis1, axis2, dtype, out
            ).get_pyobj()

    return call_origin(numpy.trace, x1, offset, axis1, axis2, dtype, out)


def tri(N, M=None, k=0, dtype=dpnp.float, **kwargs):
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    For full documentation refer to :obj:`numpy.tri`.

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

    """

    if not use_origin_backend():
        if len(kwargs) != 0:
            pass
        elif not isinstance(N, int):
            pass
        elif N < 0:
            pass
        elif M is not None and not isinstance(M, int):
            pass
        elif M is not None and M < 0:
            pass
        elif not isinstance(k, int):
            pass
        else:
            _dtype = (
                dpnp.default_float_type()
                if dtype in (dpnp.float, None)
                else dtype
            )
            return dpnp_tri(N, M, k, _dtype).get_pyobj()

    return call_origin(numpy.tri, N, M, k, dtype, **kwargs)


def tril(x1, /, *, k=0):
    """
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    For full documentation refer to :obj:`numpy.tril`.

    Limitations
    -----------
    Parameter `x1` is supported as :class:`dpnp.dpnp_array` or :class:`dpctl.tensor.usm_ndarray` with two or more dimensions.
    Parameter `k` is supported only of integer data type.
    Otherwise the function will be executed sequentially on CPU.

    Examples
    --------
    >>> import dpnp as np
    >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
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

    if not isinstance(x1, (dpnp.ndarray, dpt.usm_ndarray)):
        pass
    elif x1.ndim < 2:
        pass
    elif _k is None:
        pass
    else:
        return dpnp_container.tril(x1, k=_k)

    return call_origin(numpy.tril, x1, k)


def triu(x1, /, *, k=0):
    """
    Upper triangle of an array.

    Return a copy of a matrix with the elements below the `k`-th diagonal
    zeroed.

    For full documentation refer to :obj:`numpy.triu`.

    Limitations
    -----------
    Parameter `x1` is supported as :class:`dpnp.dpnp_array` or :class:`dpctl.tensor.usm_ndarray` with two or more dimensions.
    Parameter `k` is supported only of integer data type.
    Otherwise the function will be executed sequentially on CPU.

    Examples
    --------
    >>> import dpnp as np
    >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
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

    if not isinstance(x1, (dpnp.ndarray, dpt.usm_ndarray)):
        pass
    elif x1.ndim < 2:
        pass
    elif _k is None:
        pass
    else:
        return dpnp_container.triu(x1, k=_k)

    return call_origin(numpy.triu, x1, k)


def vander(x1, N=None, increasing=False):
    """
    Generate a Vandermonde matrix.

    For full documentation refer to :obj:`numpy.vander`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2, 3, 5])
    >>> N = 3
    >>> np.vander(x, N)
    array([[ 1,  1,  1],
           [ 4,  2,  1],
           [ 9,  3,  1],
           [25,  5,  1]])
    >>> x = np.array([1, 2, 3, 5])
    >>> np.vander(x)
    array([[  1,   1,   1,   1],
           [  8,   4,   2,   1],
           [ 27,   9,   3,   1],
           [125,  25,   5,   1]])
    >>> np.vander(x, increasing=True)
    array([[  1,   1,   1,   1],
           [  1,   2,   4,   8],
           [  1,   3,   9,  27],
           [  1,   5,  25, 125]])
    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if x1.ndim != 1:
            pass
        else:
            if N is None:
                N = x1.size

            return dpnp_vander(x1_desc, N, increasing).get_pyobj()

    return call_origin(numpy.vander, x1, N=N, increasing=increasing)


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

    Limitations
    -----------
    Parameter ``order`` is supported only with values ``"C"`` and ``"F"``.
    Parameter ``like`` is supported only with default value ``None``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of input.
    :obj:`dpnp.empty` : Return a new uninitialized array.
    :obj:`dpnp.ones` : Return a new array setting values to one.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> import dpnp as np
    >>> [i for i in np.zeros(5)]
    [0.0, 0.0, 0.0, 0.0, 0.0]
    >>> x = np.zeros((2, 1))
    >>> x.ndim, x.size, x.shape
    (2, 2, (2, 1))
    >>> [i for i in x]
    [0.0, 0.0]

    """
    if like is not None:
        pass
    elif order not in ("C", "c", "F", "f", None):
        pass
    else:
        return dpnp_container.zeros(
            shape,
            dtype=dtype,
            order=order,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )

    return call_origin(numpy.zeros, shape, dtype=dtype, order=order, like=like)


def zeros_like(
    x1,
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

    Limitations
    -----------
    Parameter ``x1`` is supported as :class:`dpnp.dpnp_array` or :class:`dpctl.tensor.usm_ndarray`
    Parameter ``order`` is supported with values ``"C"`` or ``"F"``.
    Parameter ``subok`` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of input.
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with value.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(6)
    >>> [i for i in x]
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    >>> [i for i in np.zeros_like(x)]
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    """
    if not isinstance(x1, (dpnp.ndarray, dpt.usm_ndarray)):
        pass
    elif order not in ("C", "c", "F", "f", None):
        pass
    elif subok is not False:
        pass
    else:
        _shape = x1.shape if shape is None else shape
        _dtype = x1.dtype if dtype is None else dtype
        _usm_type = x1.usm_type if usm_type is None else usm_type
        _sycl_queue = dpnp.get_normalized_queue_device(
            x1, sycl_queue=sycl_queue, device=device
        )
        return dpnp_container.zeros(
            _shape,
            dtype=_dtype,
            order=order,
            usm_type=_usm_type,
            sycl_queue=_sycl_queue,
        )

    return call_origin(numpy.zeros_like, x1, dtype, order, subok, shape)
