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

from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *

__all__ = [
    "arange",
    "array",
    "asarray",
    "empty",
    "empty_like",
    "full",
    "full_like",
    "geomspace",
    "linspace",
    "logspace",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like"
]


def arange(start, stop=None, step=1, dtype=None):
    """Returns an array with evenly spaced values within a given interval.

    Values are generated within the half-open interval [start, stop). The first
    three arguments are mapped like the ``range`` built-in function, i.e. start
    and step are optional.

    Parameters
    ----------
    start : number, optional
        Start of the interval.
    stop : number
        End of the interval.
    step : number, optional
        Step width between each pair of consecutive values.
    dtype : dtype
        Data type specifier. It is inferred from other arguments by default.

    Returns
    -------
    arange : :obj:`dpnp.ndarray`
        The 1-D array of range values.

    Limitations
    -----------
    Parameter ``start`` is supported as integer only.
    Parameters ``stop`` and ``step`` are supported as either integer or `None`.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`numpy.arange` : Return evenly spaced values within a given interval.
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
    if use_origin_backend():
        if not isinstance(start, int):
            pass
        if not isinstance(stop, int) or stop is not None:
            pass
        if not isinstance(step, int) or step is not None:
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

    Parameters
    ----------
    obj : array_like
        Array-like object.
    dtype: data-type, optional
        Data type specifier.
    copy : bool, optional
        If ``False``, this function returns ``obj`` if possible.
        Otherwise this function always returns a new array.
    order : {'K', 'A', 'C', 'F'}, optional
        Specify the memory layout of the array.
    subok : bool, optional
        If True, then sub-classes will be passed-through,
        otherwise the returned array will be forced to be a base-class
        array (default).
    ndmin : int, optional
        Minimum number of dimensions. Ones are inserted to the
        head of the shape if needed.

    Returns
    -------
        out : :obj:`dpnp.ndarray`
            An array object.

    Limitations
    -----------
    Parameter ``copy`` is supported only with default value `True`.
    Parameter ``order`` is supported only with default value `'C'`.
    Parameter ``subok`` is currently unsupported.
    Parameter ``ndmin`` is supported only with default value `0`.

    See Also
    --------
    :obj:`numpy.array` : Create an array.
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


def asarray(input, dtype=None, order='C'):
    """Converts an input object into array.

    Parameters
    ----------
    input : array_like
        Input data.
    dtype: data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major (C-style) or column-major (Fortran-style) memory representation.
        Defaults to 'C'.

    Returns
    -------
        out : :obj:`dpnp.ndarray`
            Array interpretation of `input`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value `'C'`.

    .. seealso:: :obj:`numpy.asarray`

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.asarray([1, 2])
    >>> [i for i in x]
    [1, 2]

    """

    if (use_origin_backend(input)):
        return numpy.asarray(input, dtype=dtype, order=order)

    return array(input, dtype=dtype, order=order)


# numpy.empty(shape, dtype=float, order='C')
def empty(shape, dtype=numpy.float64, order='C'):
    """
    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array.
    dtype : data-type, optional
        Data type specifier.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    out : :obj:`dpnp.ndarray`
        A new array with elements not initialized.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value `'C'`.

    See Also
    --------
    :obj:`numpy.empty` : Return a new array of given shape and type, without initializing entries.
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

    Parameters
    ----------
    prototype : array_like
        Base array.
    dtype : data-type, optional
        Data type specifier.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result.

    Returns
    -------
    out : :obj:`dpnp.ndarray`
        A new array with same shape and dtype of `prototype` with elements not initialized.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value `'C'`.
    Parameter ``subok`` is supported only with default value `False`.

    See Also
    --------
    :obj:`numpy.empty_like` : Return a new array with the same shape and type as a given array.
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


# numpy.full(shape, fill_value, dtype=None, order='C')
def full(shape, fill_value, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array.
    fill_value : scalar or array_like
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.

    Returns
    -------
    out : :obj:`dpnp.ndarray`
        Array of `fill_value` with the given shape, dtype, and order.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value `'C'`.

    See Also
    --------
    :obj:`numpy.full` : Return a new array of given shape and type, filled with `fill_value`.
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

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("full", "order", order, 'C')

        _dtype = dtype if dtype is not None else type(fill_value)

        return dpnp_init_val(shape, _dtype, fill_value)

    return numpy.full(shape, fill_value, dtype, order)


# numpy.full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None)
def full_like(x1, fill_value, dtype=None, order='C', subok=False, shape=None):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    x1 : array_like
        Base array.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `x1`, otherwise it will be a base-class array.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result.

    Returns
    -------
    out : :obj:`dpnp.ndarray`
        Array of `fill_value` with the same shape and type as `x1`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value `'C'`.
    Parameter ``subok`` is supported only with default value `False`.

    See Also
    --------
    :obj:`numpy.full_like` : Return a full array with the same shape and type as a given array.
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

        return dpnp_init_val(_shape, _dtype, fill_value)

    return numpy.full_like(x1, fill_value, dtype, order, subok, shape)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """
    Return numbers spaced evenly on a log scale (a geometric progression).

    This is similar to `logspace`, but with endpoints specified directly.
    Each output sample is a constant multiple of the previous.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The final value of the sequence, unless `endpoint` is False.
        In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate. Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : dtype
        The type of the output array.
    axis : int, optional
        The axis in the result to store the samples.

    Returns
    -------
    samples : :obj:`dpnp.ndarray`
        `num` samples, equally spaced on a log scale.

    Limitations
    -----------
    Parameter ``axis`` is supported only with default value `0`.

    See Also
    --------
    :obj:`numpy.geomspace` : Return numbers spaced evenly on a log scale
                             (a geometric progression).
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


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : dtype, optional
        The type of the output array.
    axis : int, optional
        The axis in the result to store the samples.

    Returns
    -------
    samples : :obj:`dpnp.ndarray`
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float, optional
        Only returned if `retstep` is True.
        Size of spacing between samples.

    Limitations
    -----------
    Parameter ``axis`` is supported only with default value `0`.

    See Also
    --------
    :obj:`numpy.linspace` : Return evenly spaced numbers over a specified interval.
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


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """
    Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

    Parameters
    ----------
    start : array_like
        ``base ** start`` is the starting value of the sequence.
    stop : array_like
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate. Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : float, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype
        The type of the output array.
    axis : int, optional
        The axis in the result to store the samples.

    Returns
    -------
    samples : :obj:`dpnp.ndarray`
        `num` samples, equally spaced on a log scale.

    Limitations
    -----------
    Parameter ``axis`` is supported only with default value `0`.

    See Also
    --------
    :obj:`numpy.logspace` : Return numbers spaced evenly on a log scale.
    :obj:`dpnp.arange` : Similar to linspace, with the step size specified
                         instead of the number of samples. Note that, when used
                         with a float endpoint, the endpoint may or may not be
                         included.
    :obj:`dpnp.inspace` : Similar to logspace, but with the samples uniformly
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


def ones(shape, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
        The shape of the output array.
    dtype : data-type, optional
        The type of the output array.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    out : :obj:`dpnp.ndarray`
        Array of ones with the given shape, dtype, and order.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value `'C'`.

    See Also
    --------
    :obj:`numpy.ones` : Return a new array of given shape and type, filled with ones.
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

        return dpnp_init_val(shape, dtype, 1)

    return numpy.ones(shape, dtype=dtype, order=order)


# numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
def ones_like(x1, dtype=None, order='C', subok=False, shape=None):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    x1 : array_like
        Base array.
    dtype : data-type, optional
        The type of the output array.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        The shape of the output array.

    Returns
    -------
    out : :obj:`dpnp.ndarray`
        Array of ones with the same shape and type as `x1`.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value `'C'`.
    Parameter ``subok`` is supported only with default value `False`.

    See Also
    --------
    :obj:`numpy.ones_like` : Return an array of ones
                             with the same shape and type as a given array.
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

        return dpnp_init_val(_shape, _dtype, 1)

    return numpy.ones_like(x1, dtype, order, subok, shape)


def zeros(shape, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        The shape of the output array.
    dtype : data-type, optional
        The type of the output array.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    out : :obj:`dpnp.ndarray`
        Array of zeros with the given shape, dtype, and order.

    Limitations
    -----------
    Parameter ``order`` is supported only with default value `'C'`.

    See Also
    --------
    :obj:`numpy.zeros` : Return a new array of given shape and type, filled with zeros.
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

        return dpnp_init_val(shape, dtype, 0)

    return numpy.zeros(shape, dtype=dtype, order=order)


# numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
def zeros_like(x1, dtype=None, order='C', subok=False, shape=None):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    x1 : array_like
        Base array.
    dtype : data-type, optional
        The type of the output array.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'x1', otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        The shape of the output array.

    Returns
    -------
    out : :obj:`dpnp.ndarray`
        Array of zeros with the same shape and type as `x1`.

    See Also
    --------
    :obj:`numpy.zeros_like` : Return an array of zeros
                              with the same shape and type as a given array.
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

        return dpnp_init_val(_shape, _dtype, 0)

    return numpy.zeros_like(x1, dtype, order, subok, shape)
