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


def arange(*args, **kwargs):
    """Returns an array with evenly spaced values within a given interval.

    Values are generated within the half-open interval [start, stop). The first
    three arguments are mapped like the ``range`` built-in function, i.e. start
    and step are optional.

    Parameters
    ----------
        start: Start of the interval.
        stop: End of the interval.
        step: Step width between each pair of consecutive values.
        dtype: Data type specifier. It is inferred from other arguments by
            default.

    Returns
    -------
        inumpy.dparray: The 1-D array of range values.

    .. seealso:: :obj:`numpy.arange`

    """

    if (use_origin_backend()):
        return numpy.arange(*args, **kwargs)

    if not isinstance(args[0], (int)):
        raise TypeError(f"DPNP arange(): scalar arguments expected. Given:{type(args[0])}")

    start_param = 0
    stop_param = 0
    step_param = 1
    dtype_param = kwargs.pop("dtype", None)
    if dtype_param is None:
        dtype_param = numpy.float64

    if kwargs:
        raise TypeError("DPNP arange(): unexpected keyword argument(s): %s" % ",".join(kwargs.keys()))

    args_len = len(args)
    if args_len == 1:
        stop_param = args[0]
    elif args_len == 2:
        start_param = args[0]
        stop_param = args[1]
    elif args_len == 3:
        start_param, stop_param, step_param = args
    else:
        raise TypeError("DPNP arange() takes 3 positional arguments: arange([start], stop, [step])")

    return dpnp_arange(start_param, stop_param, step_param, dtype_param)


def array(obj, dtype=None, copy=True, order='C', subok=False, ndmin=0):
    """
    Creates an array.

    This function currently does not support the ``subok`` option.

    Args:
        obj: :class:`inumpy.dparray` object or any other object that can be
            passed to :obj:`numpy.array`.
        dtype: Data type specifier.
        copy (bool): If ``False``, this function returns ``obj`` if possible.
            Otherwise this function always returns a new array.
        order ({'C', 'F', 'A', 'K'}): Row-major (C-style) or column-major
            (Fortran-style) order.
            When ``order`` is 'A', it uses 'F' if ``a`` is column-major and
            uses 'C' otherwise.
            And when ``order`` is 'K', it keeps strides as closely as
            possible.
            If ``obj`` is :class:`numpy.ndarray`, the function returns 'C' or
            'F' order array.
        subok (bool): If True, then sub-classes will be passed-through,
            otherwise the returned array will be forced to be a base-class
            array (default).
        ndmin (int): Minimum number of dimensions. Ones are inserted to the
            head of the shape if needed.

    Returns:
        inumpy.dparray: An array on the current device.



    .. note::
       This method currently does not support ``subok`` argument.

    .. seealso:: :obj:`numpy.array`

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
        checker_throw_value_error("array", "order", order, 'K')

    if ndmin != 0:
        checker_throw_value_error("array", "ndmin", ndmin, 0)

    return dpnp_array(obj, dtype)


def asarray(input, dtype=None, order='C'):
    """Converts an input object into array.

    This is equivalent to ``array(a, dtype, copy=False)``.

    Args:
        input: The source object.
        dtype: Data type specifier. It is inferred from the input by default.
        order{‘C’, ‘F’}, optional
            Whether to use row-major (C-style) or column-major (Fortran-style) memory representation.
            Defaults to ‘C’.

    Returns:
        inumpy.dparray populated with input data

    .. seealso:: :obj:`numpy.asarray`

    """

    if (use_origin_backend(input)):
        return numpy.asarray(input, dtype=dtype, order=order)

    return array(input, dtype=dtype, order=order)


# numpy.empty(shape, dtype=float, order='C')
def empty(shape, dtype=numpy.float64, order='C'):
    """Return a new matrix of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty matrix.
    dtype : data-type, optional
        Desired output data-type.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    See Also
    --------
    :obj:`dpnp.empty_like`, :obj:`dpnp.zeros`

    Notes
    -----
    :obj:`dpnp.empty`, unlike :obj:`dpnp.zeros`, does not set the matrix values
    to zero, and may therefore be marginally faster.  On the other hand, it
    requires the user to manually set all the values in the array, and should
    be used with caution.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.empty((2, 2))    # filled with random data
    matrix([[  6.76425276e-320,   9.79033856e-307], # random
            [  7.39337286e-309,   3.22135945e-309]])
    >>> np.matlib.empty((2, 2), dtype=int)
    matrix([[ 6600475,        0], # random
            [ 6586976, 22740995]])
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
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
        .. versionadded:: 1.6.0
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if ``prototype`` is Fortran
        contiguous, 'C' otherwise. 'K' means match the layout of ``prototype``
        as closely as possible.
        .. versionadded:: 1.6.0
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
        .. versionadded:: 1.17.0

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `prototype`.

    See Also
    --------
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with value.
    :obj:`dpnp.empty` : Return a new uninitialized array.

    Notes
    -----
    This function does *not* initialize the returned array; to do that use
    :obj:`dpnp.zeros_like` or :obj:`dpnp.ones_like` instead.  It may be marginally faster than
    the functions that do set the array values.

    Examples
    --------
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> np.empty_like(a)
    array([[-1073741821, -1073741821,           3],    # uninitialized
           [          0,           0, -1073741821]])
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    >>> np.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000], # uninitialized
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
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
def full_like(a, fill_value, dtype=None, order='C', subok=False, shape=None):
    """
    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        Base array.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
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
        Array of `fill_value` with the same shape and type as `a`.

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

        _shape = shape if shape is not None else a.shape
        _dtype = dtype if dtype is not None else a.dtype

        return dpnp_init_val(_shape, _dtype, fill_value)

    return numpy.full_like(a, fill_value, dtype, order, subok, shape)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """
    Return numbers spaced evenly on a log scale (a geometric progression).

    This is similar to `logspace`, but with endpoints specified directly.
    Each output sample is a constant multiple of the previous.

    .. versionchanged:: 1.16.0
        Non-scalar `start` and `stop` are now supported.

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
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

        .. versionadded:: 1.16.0

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    :obj:`dpnp.logspace` : Similar to geomspace, but with endpoints specified
                           using log and base.
    :obj:`dpnp.linspace` : Similar to geomspace, but with arithmetic instead of
                           geometric progression.
    :obj:`dpnp.arange` : Similar to linspace, with the step size specified
                         instead of the number of samples.

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

    .. versionchanged:: 1.16.0
        Non-scalar `start` and `stop` are now supported.

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
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

        .. versionadded:: 1.9.0

    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

        .. versionadded:: 1.16.0

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float, optional
        Only returned if `retstep` is True

        Size of spacing between samples.


    See Also
    --------
    :obj:`dpnp.arange` : Similar to `linspace`, but uses a step size (instead
                         of the number of samples).
    :obj:`dpnp.geomspace` : Similar to `linspace`, but with numbers spaced
                            evenly on a log scale (a geometric progression).
    :obj:`dpnp.logspace` : Similar to `geomspace`, but with the end points
                           specified as logarithms.
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

    .. versionchanged:: 1.16.0
        Non-scalar `start` and `stop` are now supported.

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
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : float, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

        .. versionadded:: 1.16.0


    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    :obj:`dpnp.arange` : Similar to linspace, with the step size specified
                         instead of the number of samples. Note that, when used
                         with a float endpoint, the endpoint may or may not be
                         included.
    :obj:`dpnp.inspace` : Similar to logspace, but with the samples uniformly
                          distributed in linear space, instead of log space.
    :obj:`dpnp.geomspace` : Similar to logspace, but with endpoints specified
                            directly.
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
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int64`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: C
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    out : dparray
        Array of ones with the given shape, dtype, and order.

    See Also
    --------
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of input.
    :obj:`dpnp.empty` : Return a new uninitialized array.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> np.ones(5)
    array([1., 1., 1., 1., 1.])
    >>> np.ones((5,), dtype=int64)
    array([1, 1, 1, 1, 1])
    >>> np.ones((2, 1))
    array([[1.],
           [1.]])
    >>> s = (2,2)
    >>> np.ones(s)
    array([[1.,  1.],
           [1.,  1.]])
    """

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("ones", "order", order, 'C')

        return dpnp_init_val(shape, dtype, 1)

    return numpy.ones(shape, dtype=dtype, order=order)


# numpy.ones_like(a, dtype=None, order='K', subok=True, shape=None)
def ones_like(prototype, dtype=None, order='C', subok=False, shape=None):
    """
    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
        .. versionadded:: 1.6.0
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
        .. versionadded:: 1.6.0
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
        .. versionadded:: 1.17.0

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of input.
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with value.
    :obj:`dpnp.ones` : Return a new array setting values to one.

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.ones_like(x)
    array([[1, 1, 1],
           [1, 1, 1]])
    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> np.ones_like(y)
    array([1.,  1.,  1.])
    """

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("ones_like", "order", order, 'C')
        if subok is not False:
            checker_throw_value_error("ones_like", "subok", subok, False)

        _shape = shape if shape is not None else prototype.shape
        _dtype = dtype if dtype is not None else prototype.dtype

        return dpnp_init_val(_shape, _dtype, 1)

    return numpy.ones_like(prototype, dtype, order, subok, shape)


def zeros(shape, dtype=None, order='C'):
    """
    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: 'C'
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and order.

    See Also
    --------
    :obj:`dpnp.zeros_like` : Return an array of zeros with shape and type of input.
    :obj:`dpnp.empty` : Return a new uninitialized array.
    :obj:`dpnp.ones` : Return a new array setting values to one.
    :obj:`dpnp.full` : Return a new array of given shape filled with value.

    Examples
    --------
    >>> np.zeros(5)
    array([ 0.,  0.,  0.,  0.,  0.])

    >>> np.zeros((5,), dtype=int)
    array([0, 0, 0, 0, 0])

    >>> np.zeros((2, 1))
    array([[ 0.],
           [ 0.]])

    >>> s = (2,2)
    >>> np.zeros(s)
    array([[ 0.,  0.],
           [ 0.,  0.]])

    >>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
    array([(0, 0), (0, 0)],
          dtype=[('x', '<i4'), ('y', '<i4')])
    """

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("zeros", "order", order, 'C')

        return dpnp_init_val(shape, dtype, 0)

    return numpy.zeros(shape, dtype=dtype, order=order)


# numpy.zeros_like(a, dtype=None, order='K', subok=True, shape=None)
def zeros_like(prototype, dtype=None, order='C', subok=False, shape=None):
    """
    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
        .. versionadded:: 1.6.0
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
        'C' otherwise. 'K' means match the layout of `a` as closely
        as possible.
        .. versionadded:: 1.6.0
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of 'a', otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.

        .. versionadded:: 1.17.0

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    :obj:`dpnp.empty_like` : Return an empty array with shape and type of input.
    :obj:`dpnp.ones_like` : Return an array of ones with shape and type of input.
    :obj:`dpnp.full_like` : Return a new array with shape of input filled with value.
    :obj:`dpnp.zeros` : Return a new array setting values to zero.

    Examples
    --------
    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])
    >>> y = np.arange(3, dtype=float)
    >>> y
    array([0., 1., 2.])
    >>> np.zeros_like(y)
    array([0.,  0.,  0.])
    """

    if (not use_origin_backend()):
        if order not in ('C', 'c', None):
            checker_throw_value_error("zeros_like", "order", order, 'C')
        if subok is not False:
            checker_throw_value_error("zeros_like", "subok", subok, False)

        _shape = shape if shape is not None else prototype.shape
        _dtype = dtype if dtype is not None else prototype.dtype

        return dpnp_init_val(_shape, _dtype, 0)

    return numpy.zeros_like(prototype, dtype, order, subok, shape)
