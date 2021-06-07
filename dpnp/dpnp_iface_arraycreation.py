# cython: language_level=3
# distutils: language = c++
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


import numpy
import dpnp

from dpnp.dpnp_algo import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *

__all__ = [
    "arange",
    "array",
    "asanyarray",
    "asarray",
    "ascontiguousarray",
    "copy",
    "diag",
    "diagflat",
    "empty",
    "empty_like",
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
    "zeros_like"
]


def arange(start, stop=None, step=1, dtype=None):
    """
    Returns an array with evenly spaced values within a given interval.

    For full documentation refer to :obj:`numpy.arange`.

    Returns
    -------
    arange : :obj:`dpnp.ndarray`
        The 1-D array of range values.

    Limitations
    -----------
    Parameter ``start`` is supported as integer only.
    Parameters ``stop`` and ``step`` are supported as either integer or ``None``.
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
    if not use_origin_backend():
        if not isinstance(start, int):
            pass
        elif not isinstance(stop, int) or stop is not None:
            pass
        elif not isinstance(step, int) or step is not None:
            pass
        else:
            if dtype is None:
                dtype = numpy.float64

            if stop is None:
                stop = start
                start = 0

            if step is None:
                step = 1

            return dpnp_arange(start, stop, step, dtype)

    return call_origin(numpy.arange, start, stop=stop, step=step, dtype=dtype)


def array(obj, dtype=None, copy=True, order='C', subok=False, ndmin=0):
    """
    Creates an array.

    For full documentation refer to :obj:`numpy.array`.

    Limitations
    -----------
    Parameter ``copy`` is supported only with default value ``True``.
    Parameter ``order`` is supported only with default value ``"C"``.
    Parameter ``subok`` is supported only with default value ``False``.
    Parameter ``ndmin`` is supported only with default value ``0``.

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
    >>> [i for i in x]
    [1, 2, 3]

    More than one dimension:

    >>> x2 = np.array([[1, 2], [3, 4]])
    >>> x2.ndim, x2.size, x2.shape
    (2, 4, (2, 2))
    >>> [i for i in x2]
    [1, 2, 3, 4]

    """

    if (use_origin_backend(obj)):
        return numpy.array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    # if not isinstance(obj, collections.abc.Sequence):
    #     return numpy.array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    # if isinstance(obj, numpy.object):
    #     return numpy.array(obj, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

    if subok is not False:
        checker_throw_value_error("array", "subok", subok, False)

    if copy is not True:
        checker_throw_value_error("array", "copy", copy, True)

    if order != 'C':
        checker_throw_value_error("array", "order", order, 'C')

    if ndmin != 0:
        checker_throw_value_error("array", "ndmin", ndmin, 0)

    return dpnp_array(obj, dtype)


def asanyarray(a, dtype=None, order='C'):
    """
    Convert the input to an ndarray, but pass ndarray subclasses through.

    For full documentation refer to :obj:`numpy.asanyarray`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.

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
    >>> x = np.asanyarray([1, 2, 3])
    >>> [i for i in x]
    [1, 2, 3]

    """

    if not use_origin_backend(a):
        # if it is already dpnp.ndarray then same object should be returned
        if isinstance(a, dpnp.ndarray):
            return a

        if order != 'C':
            checker_throw_value_error("asanyarray", "order", order, 'C')

        return array(a, dtype=dtype, order=order)

    return call_origin(numpy.asanyarray, a, dtype, order)


def asarray(input, dtype=None, order='C'):
    """
    Converts an input object into array.

    For full documentation refer to :obj:`numpy.asarray`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.

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
    >>> x = np.asarray([1, 2, 3])
    >>> [i for i in x]
    [1, 2, 3]

    """

    if (use_origin_backend(input)):
        return numpy.asarray(input, dtype=dtype, order=order)

    return array(input, dtype=dtype, order=order)


def ascontiguousarray(a, dtype=None):
    """
    Return a contiguous array (ndim >= 1) in memory (C order).

    For full documentation refer to :obj:`numpy.ascontiguousarray`.

    See Also
    --------
    :obj:`dpnp.asfortranarray` : Convert input to an ndarray with column-major
                     memory order.
    :obj:`dpnp.require` : Return an ndarray that satisfies requirements.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(6).reshape((2, 3))
    >>> out = np.ascontiguousarray(x)
    >>> out.shape
    (2, 3)
    >>> [i for i in out]
    [0, 1, 2, 3, 4, 5]

    """

    if not use_origin_backend(a):
        # we support only c-contiguous arrays for now
        # if type is the same then same object should be returned
        if isinstance(a, dpnp.ndarray) and a.dtype == dtype:
            return a

        return array(a, dtype=dtype)

    return call_origin(numpy.ascontiguousarray, a, dtype)


# numpy.copy(a, order='K', subok=False)
def copy(a, order='C', subok=False):
    """
    Return an array copy of the given object.

    For full documentation refer to :obj:`numpy.copy`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.
    Parameter ``subok`` is supported only with default value ``False``.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2, 3])
    >>> y = x
    >>> z = np.copy(x)
    >>> x[0] = 10
    >>> x[0] == y[0]
    True
    >>> x[0] == z[0]
    False

    """

    if not use_origin_backend(a):
        return dpnp_copy(a, order, subok)

    return call_origin(numpy.copy, a, order, subok)


def diag(v, k=0):
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

    if not use_origin_backend(v):
        if not isinstance(v, dparray):
            pass
        elif not isinstance(k, int):
            pass
        elif v.ndim != 1 and v.ndim != 2:
            pass
        else:
            return dpnp_diag(v, k)

    return call_origin(numpy.diag, v, k)


def diagflat(v, k=0):
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

    if not use_origin_backend(v):
        if not isinstance(v, dparray):
            pass
        else:
            return dpnp_diag(v.ravel(), k)

    return call_origin(numpy.diagflat, v, k)


# numpy.empty(shape, dtype=float, order='C')
def empty(shape, dtype=numpy.float64, order='C'):
    """
    Return a new array of given shape and type, without initializing entries.

    For full documentation refer to :obj:`numpy.empty`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.

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
    >>> [i for i in x]
    [0.0, 0.0, 1e-323, -3.5935729608842025e+22]

    """

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("empty", "order", order, 'C')

        return dparray(shape, dtype)

    return numpy.empty(shape, dtype, order)


# numpy.empty_like(prototype, dtype=None, order='K', subok=True, shape=None)
def empty_like(prototype, dtype=None, order='C', subok=False, shape=None):
    """
    Return a new array with the same shape and type as a given array.

    For full documentation refer to :obj:`numpy.empty_like`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.
    Parameter ``subok`` is supported only with default value ``False``.

    See Also
    --------
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with value.
    :obj:`dpnp.empty` : Return a new uninitialized array.

    Examples
    --------
    >>> import dpnp as np
    >>> prototype = np.array([1, 2, 3])
    >>> x = np.empty_like(prototype)
    >>> [i for i in x]
    [0, 0, 0]

    """

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("empty_like", "order", order, 'C')
        if subok is not False:
            checker_throw_value_error("empty_like", "subok", subok, False)

        _shape = shape if shape is not None else prototype.shape
        _dtype = dtype if dtype is not None else prototype.dtype.type

        return dparray(_shape, _dtype)

    return numpy.empty_like(prototype, dtype, order, subok, shape)


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


# numpy.full(shape, fill_value, dtype=None, order='C')
def full(shape, fill_value, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    For full documentation refer to :obj:`numpy.full`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.

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
    if not use_origin_backend():
        if order not in ('C', 'c', None):
            pass
        else:
            if dtype is None:
                dtype = numpy.array(fill_value).dtype.type

            return dpnp_full(shape, fill_value, dtype)

    return call_origin(numpy.full, shape, fill_value, dtype, order)


# numpy.full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None)
def full_like(x1, fill_value, dtype=None, order='C', subok=False, shape=None):
    """
    Return a full array with the same shape and type as a given array.

    For full documentation refer to :obj:`numpy.full_like`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.
    Parameter ``subok`` is supported only with default value ``False``.

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

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("full_like", "order", order, 'C')
        if subok is not False:
            checker_throw_value_error("full_like", "subok", subok, False)

        _shape = shape if shape is not None else x1.shape
        _dtype = dtype if dtype is not None else x1.dtype

        return dpnp_full_like(_shape, fill_value, _dtype)

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
            checker_throw_value_error("linspace", "axis", axis, 0)

        return dpnp_geomspace(start, stop, num, endpoint, dtype, axis)

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
            if dtype is None:
                dtype = dpnp.float64
            return dpnp_identity(n, dtype)

    return call_origin(numpy.identity, n, dtype=dtype, like=like)


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """
    Return evenly spaced numbers over a specified interval.

    For full documentation refer to :obj:`numpy.linspace`.

    Limitations
    -----------
    Parameter ``axis`` is supported only with default value ``0``.

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

    if not use_origin_backend():
        if axis != 0:
            checker_throw_value_error("linspace", "axis", axis, 0)

        res = dpnp_linspace(start, stop, num, endpoint, retstep, dtype, axis)

        if retstep:
            return res
        else:
            return res[0]

    return call_origin(numpy.linspace, start, stop, num, endpoint, retstep, dtype, axis)


def loadtxt(fname, **kwargs):
    """
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

        return dpnp_logspace(start, stop, num, endpoint, base, dtype, axis)

    return call_origin(numpy.logspace, start, stop, num, endpoint, base, dtype, axis)


def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    """
    Return coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    For full documentation refer to :obj:`numpy.meshgrid`.

    Limitations
    -----------
    Parameter ``copy`` is supported only with default value ``True``.
    Parameter ``sparse`` is supported only with default value ``False``.

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

    if not use_origin_backend():
        # original limitation
        if indexing not in ["ij", "xy"]:
            checker_throw_value_error("meshgrid", "indexing", indexing, "'ij' or 'xy'")

        if copy is not True:
            checker_throw_value_error("meshgrid", "copy", copy, True)
        if sparse is not False:
            checker_throw_value_error("meshgrid", "sparse", sparse, False)

        return dpnp_meshgrid(xi, copy, sparse, indexing)

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


def ones(shape, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with ones.

    For full documentation refer to :obj:`numpy.ones`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.

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

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("ones", "order", order, 'C')

        _dtype = dtype if dtype is not None else dpnp.float64

        return dpnp_ones(shape, _dtype)

    return numpy.ones(shape, dtype=dtype, order=order)


# numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
def ones_like(x1, dtype=None, order='C', subok=False, shape=None):
    """
    Return an array of ones with the same shape and type as a given array.

    For full documentation refer to :obj:`numpy.ones_like`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.
    Parameter ``subok`` is supported only with default value ``False``.

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

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("ones_like", "order", order, 'C')
        if subok is not False:
            checker_throw_value_error("ones_like", "subok", subok, False)

        _shape = shape if shape is not None else x1.shape
        _dtype = dtype if dtype is not None else x1.dtype

        return dpnp_ones_like(_shape, _dtype)

    return numpy.ones_like(x1, dtype, order, subok, shape)


def trace(arr, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """
       Return the sum along diagonals of the array.

       For full documentation refer to :obj:`numpy.trace`.

       Limitations
       -----------
       Input array is supported as :obj:`dpnp.ndarray`.
       Parameters ``axis1``, ``axis2``, ``out`` and ``dtype`` are supported only with default values.
       """
    if not use_origin_backend():
        if not isinstance(arr, dparray):
            pass
        elif arr.size == 0:
            pass
        elif arr.ndim < 2:
            pass
        elif axis1 != 0:
            pass
        elif axis2 != 1:
            pass
        elif out is not None and (not isinstance(out, dparray) or (isinstance(out, dparray) and out.shape != arr.shape)):
            pass
        else:
            return dpnp_trace(arr, offset, axis1, axis2, dtype, out)

    return call_origin(numpy.trace, arr, offset, axis1, axis2, dtype, out)


def tri(N, M=None, k=0, dtype=numpy.float, **kwargs):
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
            return dpnp_tri(N, M, k, dtype)

    return call_origin(numpy.tri, N, M, k, dtype, **kwargs)


def tril(m, k=0):
    """
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    For full documentation refer to :obj:`numpy.tril`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])

    """

    if not use_origin_backend(m):
        if not isinstance(m, dparray):
            pass
        else:
            return dpnp_tril(m, k)

    return call_origin(numpy.tril, m, k)


def triu(m, k=0):
    """
    Upper triangle of an array.

    Return a copy of a matrix with the elements below the `k`-th diagonal
    zeroed.

    For full documentation refer to :obj:`numpy.triu`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.triu([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 0,  8,  9],
           [ 0,  0, 12]])

    """

    if not use_origin_backend(m):
        if not isinstance(m, dparray):
            pass
        elif not isinstance(k, int):
            pass
        else:
            return dpnp_triu(m, k)

    return call_origin(numpy.triu, m, k)


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
    if (not use_origin_backend(x1)):
        if not isinstance(x1, dparray):
            pass
        elif x1.ndim != 1:
            pass
        else:
            if N is None:
                N = x1.size
            return dpnp_vander(x1, N, increasing)

    return call_origin(numpy.vander, x1, N=N, increasing=increasing)


def zeros(shape, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with zeros.

    For full documentation refer to :obj:`numpy.zeros`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.

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

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("zeros", "order", order, 'C')

        _dtype = dtype if dtype is not None else dpnp.float64

        return dpnp_zeros(shape, _dtype)

    return numpy.zeros(shape, dtype=dtype, order=order)


# numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
def zeros_like(x1, dtype=None, order='C', subok=False, shape=None):
    """
    Return an array of zeros with the same shape and type as a given array.

    For full documentation refer to :obj:`numpy.zeros_like`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value ``"C"``.
    Parameter ``subok`` is supported only with default value ``False``.

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

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("zeros_like", "order", order, 'C')
        if subok is not False:
            checker_throw_value_error("zeros_like", "subok", subok, False)

        _shape = shape if shape is not None else x1.shape
        _dtype = dtype if dtype is not None else x1.dtype

        return dpnp_zeros_like(_shape, _dtype)

    return numpy.zeros_like(x1, dtype, order, subok, shape)
