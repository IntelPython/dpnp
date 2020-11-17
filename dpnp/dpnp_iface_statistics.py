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
Interface of the statistics function of the DPNP

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
import dpnp


__all__ = [
    'amax',
    'amin',
    'average',
    'correlate',
    'cov',
    'max',
    'mean',
    'median',
    'min',
    'std',
    'var',
]


def amax(input, axis=None, out=None):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    input : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `ufuncs-output-type` for more details.

    Returns
    -------
    amax : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.
    """
    return max(input, axis=axis, out=out)


def amin(input, axis=None, out=None):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    input : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `ufuncs-output-type` for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
        If the default value is passed, then `keepdims` will not be
        passed through to the `amin` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.
        .. versionadded:: 1.15.0
    where : array_like of bool, optional
        Elements to compare for the minimum. See `~numpy.ufunc.reduce`
        for details.
        .. versionadded:: 1.17.0

    Returns
    -------
    amin : ndarray or scalar
        Minimum of `input`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``input.ndim - 1``.
    """

    return min(input, axis=axis, out=out)


def average(a, axis=None, weights=None, returned=False):
    """
    Compute the weighted average along the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing data to be averaged. If `a` is not an array, a
        conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        axis=None, will average over all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.

        .. versionadded:: 1.7.0

        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.  The 1-D calculation is::

            avg = sum(a * weights) / sum(weights)

        The only constraint on `weights` is that `sum(weights)` must not be 0.
    returned : bool, optional
        Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
        is returned, otherwise only the average is returned.
        If `weights=None`, `sum_of_weights` is equivalent to the number of
        elements over which the average is taken.

    Returns
    -------
    retval, [sum_of_weights] : array_type or double
        Return the average along the specified axis. When `returned` is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. `sum_of_weights` is of the
        same type as `retval`. The result dtype follows a genereal pattern.
        If `weights` is None, the result dtype will be that of `a` , or ``float64``
        if `a` is integral. Otherwise, if `weights` is not None and `a` is non-
        integral, the result type will be the type of lowest precision capable of
        representing values of both `a` and `weights`. If `a` happens to be
        integral, the previous rules still applies but the result dtype will
        at least be ``float64``.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When the length of 1D `weights` is not the same as the shape of `a`
        along axis.

    See Also
    --------
    mean

    ma.average : average for masked arrays -- useful if your data contains
                 "missing" values
    numpy.result_type : Returns the type that results from applying the
                        numpy type promotion rules to the arguments.

    """
    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif axis is not None:
            pass
        elif weights is not None:
            pass
        elif returned:
            pass
        else:
            return dpnp_average(a)

    return call_origin(numpy.average, a, axis, weights, returned)


def correlate(a, v, mode='valid'):
    """
    Cross-correlation of two 1-dimensional sequences.

    This function computes the correlation as generally defined in signal
    processing texts::

        c_{av}[k] = sum_n a[n+k] * conj(v[n])

    with a and v sequences being zero-padded where necessary and conj being
    the conjugate.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        Refer to the `convolve` docstring.  Note that the default
        is 'valid', unlike `convolve`, which uses 'full'.
    old_behavior : bool
        `old_behavior` was removed in NumPy 1.10. If you need the old
        behavior, use `multiarray.correlate`.

    Returns
    -------
    out : ndarray
        Discrete cross-correlation of `a` and `v`.

    See Also
    --------
    convolve : Discrete, linear convolution of two one-dimensional sequences.
    multiarray.correlate : Old, no conjugate, version of correlate.

    Notes
    -----
    The definition of correlation above is not unique and sometimes correlation
    may be defined differently. Another common definition is::

        c'_{av}[k] = sum_n a[n] conj(v[n+k])

    which is related to ``c_{av}[k]`` by ``c'_{av}[k] = c_{av}[-k]``.

    Examples
    --------
    >>> np.correlate([1, 2, 3], [0, 1, 0.5])
    array([3.5])
    >>> np.correlate([1, 2, 3], [0, 1, 0.5], "same")
    array([2. ,  3.5,  3. ])
    >>> np.correlate([1, 2, 3], [0, 1, 0.5], "full")
    array([0.5,  2. ,  3.5,  3. ,  0. ])

    Using complex sequences:

    >>> np.correlate([1+1j, 2, 3-1j], [0, 1, 0.5j], 'full')
    array([ 0.5-0.5j,  1.0+0.j ,  1.5-1.5j,  3.0-1.j ,  0.0+0.j ])

    Note that you get the time reversed, complex conjugated result
    when the two input sequences change places, i.e.,
    ``c_{va}[k] = c^{*}_{av}[-k]``:

    >>> np.correlate([0, 1, 0.5j], [1+1j, 2, 3-1j], 'full')
    array([ 0.0+0.j ,  3.0+1.j ,  1.5+1.5j,  1.0+0.j ,  0.5+0.5j])

    """
    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif not isinstance(v, dparray):
            pass
        elif a.size != v.size or a.size == 0:
            pass
        elif a.shape != v.shape:
            pass
        elif mode != 'valid':
            pass
        else:
            return dpnp_correlate(a, v)

    return call_origin(numpy.correlate, a, v, mode=mode)


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    """
    Estimate a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    See the notes for an outline of the algorithm.

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        `fweights` and `aweights` are specified, and ``ddof=0`` will return
        the simple average. See the notes for the details. The default value
        is ``None``.

        .. versionadded:: 1.5
    fweights : array_like, int, optional
        1-D array of integer frequency weights; the number of times each
        observation vector should be repeated.

        .. versionadded:: 1.10
    aweights : array_like, optional
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.

        .. versionadded:: 1.10

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    See Also
    --------
    corrcoef : Normalized covariance matrix

    Limitations
    -----------
    Some limitations.

    Notes
    -----
    Assume that the observations are in the columns of the observation
    array `m` and let ``f = fweights`` and ``a = aweights`` for brevity. The
    steps to compute the weighted covariance are as follows::

        >>> m = np.arange(10, dtype=np.float64)
        >>> f = np.arange(10) * 2
        >>> a = np.arange(10) ** 2.
        >>> ddof = 1
        >>> w = f * a
        >>> v1 = np.sum(w)
        >>> v2 = np.sum(w * a)
        >>> m -= np.sum(m * w, axis=None, keepdims=True) / v1
        >>> cov = np.dot(m * w, m.T) * v1 / (v1**2 - ddof * v2)

    Note that when ``a == 1``, the normalization factor
    ``v1 / (v1**2 - ddof * v2)`` goes over to ``1 / (np.sum(f) - ddof)``
    as it should.

    Examples
    --------
    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> np.cov(x)
    array([[ 1., -1.],
           [-1.,  1.]])

    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how `x` and `y` are combined:

    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> X = np.stack((x, y), axis=0)
    >>> np.cov(X)
    array([[11.71      , -4.286     ], # may vary
           [-4.286     ,  2.144133]])
    >>> np.cov(x, y)
    array([[11.71      , -4.286     ], # may vary
           [-4.286     ,  2.144133]])
    >>> np.cov(x)
    array(11.71)

    """
    if not use_origin_backend(m):
        if not isinstance(m, dparray):
            pass
        elif m.ndim > 2:
            pass
        elif y is not None:
            pass
        elif not rowvar:
            pass
        elif bias:
            pass
        elif ddof is not None:
            pass
        elif fweights is not None:
            pass
        elif aweights is not None:
            pass
        else:
            return dpnp_cov(m)

    return call_origin(numpy.cov, m, y, rowvar, bias, ddof, fweights, aweights)


def max(input, axis=None, out=None):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    input : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `ufuncs-output-type` for more details.

    Returns
    -------
    amax : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.
    """

    dim_input = input.ndim

    is_input_dparray = isinstance(input, dparray)

    if not use_origin_backend(input) and is_input_dparray:
        if out is not None:
            checker_throw_value_error("max", "out", type(out), None)

        result = dpnp_max(input, axis=axis)

        # scalar returned
        if result.shape == (1,):
            return result.dtype.type(result[0])

        return result

    input1 = dpnp.asnumpy(input) if is_input_dparray else input

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.max(input1, axis=axis)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def mean(a, axis=None, **kwargs):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `input` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    kwargs : dict
        Remaining input parameters of the function.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    """
    if not use_origin_backend(a) and not kwargs:
        if not isinstance(a, dparray):
            pass
        elif a.size == 0:
            pass
        else:
            result = dpnp_mean(a, axis=axis)

            # scalar returned
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.mean, a, axis=axis, **kwargs)


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.
    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default
        is to compute the median along a flattened version of the array.
        A sequence of axes is supported since version 1.9.0.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       `median`. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. If `overwrite_input` is ``True`` and `a` is not already an
       `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.
        .. versionadded:: 1.9.0

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``np.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean, percentile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the
    middle value of a sorted copy of ``V``, ``V_sorted`` - i
    e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
    two middle values of ``V_sorted`` when ``N`` is even.

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.median(a)
    3.5
    >>> np.median(a, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.median(a, axis=1)
    array([7.,  2.])
    >>> m = np.median(a, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.median(a, axis=0, out=m)
    array([6.5,  4.5,  2.5])
    >>> m
    array([6.5,  4.5,  2.5])
    >>> b = a.copy()
    >>> np.median(b, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.median(b, axis=None, overwrite_input=True)
    3.5
    >>> assert not np.all(a==b)
    """
    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif axis is not None:
            pass
        elif out is not None:
            pass
        elif overwrite_input:
            pass
        elif keepdims:
            pass
        else:
            result = dpnp_median(a)

            # scalar returned
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.median, a, axis, out, overwrite_input, keepdims)


def min(input, axis=None, out=None):
    """
    Return the minimum along a given axis.

    Parameters
    ----------
    input : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
        New in version 1.7.0.
        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result. Must be of the
        same shape and buffer length as the expected output.
        See ufuncs-output-type for more details.

    Returns
    -------
    m : ndarray, see dtype parameter above
        Minimum of a. If axis is None, the result is a scalar value.
        If axis is given, the result is an array of dimension a.ndim - 1.
    """

    dim_input = input.ndim

    is_input_dparray = isinstance(input, dparray)

    if not use_origin_backend(input) and is_input_dparray:
        if out is not None:
            checker_throw_value_error("min", "out", type(out), None)

        result = dpnp_min(input, axis=axis)

        # scalar returned
        if result.shape == (1,):
            return result.dtype.type(result[0])

        return result

    input1 = dpnp.asnumpy(input) if is_input_dparray else input

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.min(input1, axis=axis)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=numpy._NoValue):
    """
    Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    See Also
    --------
    var, mean, nanmean, nanstd, nanvar
    ufuncs-output-type

    """

    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif a.size == 0:
            pass
        elif axis is not None:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif keepdims is not numpy._NoValue:
            pass
        else:
            result = dpnp_std(a, ddof)
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.std, a, axis, dtype, out, ddof, keepdims)


def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=numpy._NoValue):
    """
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    See Also
    --------
    std, mean, nanmean, nanstd, nanvar
    ufuncs-output-type

    """

    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif a.size == 0:
            pass
        elif axis is not None:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif keepdims is not numpy._NoValue:
            pass
        else:
            result = dpnp_var(a, ddof)
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.var, a, axis, dtype, out, ddof, keepdims)
