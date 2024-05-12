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
Interface of the statistics function of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import dpctl.tensor as dpt
import numpy
from numpy.core.numeric import normalize_axis_index

import dpnp

# pylint: disable=no-name-in-module
from .dpnp_algo import (
    dpnp_correlate,
    dpnp_median,
)
from .dpnp_array import dpnp_array
from .dpnp_utils import (
    call_origin,
    get_usm_allocations,
)
from .dpnp_utils.dpnp_utils_reduction import dpnp_wrap_reduction_call
from .dpnp_utils.dpnp_utils_statistics import (
    dpnp_cov,
)

__all__ = [
    "amax",
    "amin",
    "average",
    "bincount",
    "correlate",
    "cov",
    "max",
    "mean",
    "median",
    "min",
    "ptp",
    "std",
    "var",
]


def _count_reduce_items(arr, axis, where=True):
    """
    Calculates the number of items used in a reduction operation
    along the specified axis or axes.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        axis or axes along which the number of items used in a reduction
        operation must be counted. If a tuple of unique integers is given,
        the items are counted over multiple axes. If ``None``, the variance
        is computed over the entire array.
        Default: `None`.

    Returns
    -------
    out : int
        The number of items should be used in a reduction operation.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.

    """
    if where is True:
        # no boolean mask given, calculate items according to axis
        if axis is None:
            axis = tuple(range(arr.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)
        items = 1
        for ax in axis:
            items *= arr.shape[normalize_axis_index(ax, arr.ndim)]
        items = dpnp.intp(items)
    else:
        raise NotImplementedError(
            "where keyword argument is only supported with its default value."
        )
    return items


def _get_comparison_res_dt(a, _dtype, _out):
    """Get a data type used by dpctl for result array in comparison function."""

    return a.dtype


def amax(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the maximum of an array or maximum along an axis.

    `amax` is an alias of :obj:`dpnp.max`.

    See Also
    --------
    :obj:`dpnp.max` : alias of this function
    :obj:`dpnp.ndarray.max` : equivalent method

    """

    return max(
        a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
    )


def amin(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the minimum of an array or minimum along an axis.

    `amin` is an alias of :obj:`dpnp.min`.

    See Also
    --------
    :obj:`dpnp.min` : alias of this function
    :obj:`dpnp.ndarray.min` : equivalent method

    """

    return min(
        a, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
    )


def average(a, axis=None, weights=None, returned=False, *, keepdims=False):
    """
    Compute the weighted average along the specified axis.

    For full documentation refer to :obj:`numpy.average`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which the averages must be computed. If
        a tuple of unique integers, the averages are computed over multiple
        axes. If ``None``, the average is computed over the entire array.
        Default: ``None``.
    weights : {array_like}, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.  The 1-D calculation is::

            avg = sum(a * weights) / sum(weights)

        The only constraint on `weights` is that `sum(weights)` must not be 0.
    returned : {bool}, optional
        Default is ``False``. If ``True``, the tuple (`average`,
        `sum_of_weights`) is returned, otherwise only the average is returned.
        If `weights=None`, `sum_of_weights` is equivalent to the number of
        elements over which the average is taken.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array. Default: ``False``.

    Returns
    -------
    out, [sum_of_weights] : dpnp.ndarray, dpnp.ndarray
        Return the average along the specified axis. When `returned` is
        ``True``, return a tuple with the average as the first element and
        the sum of the weights as the second element. `sum_of_weights` is of
        the same type as `out`. The result dtype follows a general pattern.
        If `weights` is ``None``, the result dtype will be that of `a` , or
        default floating point data type for the device where input array `a`
        is allocated. Otherwise, if `weights` is not ``None`` and `a` is
        non-integral, the result type will be the type of lowest precision
        capable of representing values of both `a` and `weights`. If `a`
        happens to be integral, the previous rules still applies but the result
        dtype will at least be default floating point data type for the device
        where input array `a` is allocated.

    See Also
    --------
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.sum` : Sum of array elements over a given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> data = np.arange(1, 5)
    >>> data
    array([1, 2, 3, 4])
    >>> np.average(data)
    array(2.5)
    >>> np.average(np.arange(1, 11), weights=np.arange(10, 0, -1))
    array(4.0)

    >>> data = np.arange(6).reshape((3, 2))
    >>> data
    array([[0, 1],
        [2, 3],
        [4, 5]])
    >>> np.average(data, axis=1, weights=[1./4, 3./4])
    array([0.75, 2.75, 4.75])
    >>> np.average(data, weights=[1./4, 3./4])
    TypeError: Axis must be specified when shapes of a and weights differ.

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.average(data, axis=1, keepdims=True)
    array([[0.5],
        [2.5],
        [4.5]])

    >>> a = np.ones(5, dtype=np.float64)
    >>> w = np.ones(5, dtype=np.complex64)
    >>> avg = np.average(a, weights=w)
    >>> print(avg.dtype)
    complex128

    """

    dpnp.check_supported_arrays_type(a)
    if weights is None:
        avg = dpnp.mean(a, axis=axis, keepdims=keepdims)
        scl = dpnp.asanyarray(
            avg.dtype.type(a.size / avg.size),
            usm_type=a.usm_type,
            sycl_queue=a.sycl_queue,
        )
    else:
        if not isinstance(weights, (dpnp_array, dpt.usm_ndarray)):
            wgt = dpnp.asanyarray(
                weights, usm_type=a.usm_type, sycl_queue=a.sycl_queue
            )
        else:
            get_usm_allocations([a, weights])
            wgt = weights

        if not dpnp.issubdtype(a.dtype, dpnp.inexact):
            default_dtype = dpnp.default_float_type(a.device)
            result_dtype = dpnp.result_type(a.dtype, wgt.dtype, default_dtype)
        else:
            result_dtype = dpnp.result_type(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of input array and "
                    "weights differ."
                )
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of input array and "
                    "weights differ."
                )
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis."
                )

            # setup wgt to broadcast along axis
            wgt = dpnp.broadcast_to(wgt, (a.ndim - 1) * (1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)

        scl = wgt.sum(axis=axis, dtype=result_dtype, keepdims=keepdims)
        if dpnp.any(scl == 0.0):
            raise ZeroDivisionError("Weights sum to zero, can't be normalized")

        # result_datatype
        avg = (
            dpnp.multiply(a, wgt).sum(
                axis=axis, dtype=result_dtype, keepdims=keepdims
            )
            / scl
        )

    if returned:
        if scl.shape != avg.shape:
            scl = dpnp.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    return avg


def bincount(x1, weights=None, minlength=0):
    """
    Count number of occurrences of each value in array of non-negative integers.

    For full documentation refer to :obj:`numpy.bincount`.

    See Also
    --------
    :obj:`dpnp.unique` : Find the unique elements of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> res = np.bincount(np.arange(5))
    >>> print(res)
    [1, 1, 1, 1, 1]

    """

    return call_origin(numpy.bincount, x1, weights=weights, minlength=minlength)


def correlate(x1, x2, mode="valid"):
    """
    Cross-correlation of two 1-dimensional sequences.

    For full documentation refer to :obj:`numpy.correlate`.

    Limitations
    -----------
    Input arrays are supported as :obj:`dpnp.ndarray`.
    Size and shape of input arrays are supported to be equal.
    Parameter `mode` is supported only with default value ``"valid``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.convolve` : Discrete, linear convolution of
                           two one-dimensional sequences.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.correlate([1, 2, 3], [0, 1, 0.5])
    >>> [i for i in x]
    [3.5]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    x2_desc = dpnp.get_dpnp_descriptor(x2, copy_when_nondefault_queue=False)
    if x1_desc and x2_desc:
        if x1_desc.size != x2_desc.size or x1_desc.size == 0:
            pass
        elif x1_desc.shape != x2_desc.shape:
            pass
        elif mode != "valid":
            pass
        else:
            return dpnp_correlate(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.correlate, x1, x2, mode=mode)


def cov(
    m,
    y=None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights=None,
    aweights=None,
    *,
    dtype=None,
):
    """
    Estimate a covariance matrix, given data and weights.

    For full documentation refer to :obj:`numpy.cov`.

    Returns
    -------
    out : dpnp.ndarray
        The covariance matrix of the variables.

    Limitations
    -----------
    Input array ``m`` is supported as :obj:`dpnp.ndarray`.
    Dimension of input array ``m`` is limited by ``m.ndim <= 2``.
    Size and shape of input arrays are supported to be equal.
    Parameter `y` is supported only with default value ``None``.
    Parameter `bias` is supported only with default value ``False``.
    Parameter `ddof` is supported only with default value ``None``.
    Parameter `fweights` is supported only with default value ``None``.
    Parameter `aweights` is supported only with default value ``None``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.corrcoef` : Normalized covariance matrix

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x.shape
    (2, 3)
    >>> [i for i in x]
    [0, 1, 2, 2, 1, 0]
    >>> out = np.cov(x)
    >>> out.shape
    (2, 2)
    >>> [i for i in out]
    [1.0, -1.0, -1.0, 1.0]

    """

    if not isinstance(m, (dpnp_array, dpt.usm_ndarray)):
        pass
    elif m.ndim > 2:
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
        return dpnp_cov(m, y=y, rowvar=rowvar, dtype=dtype)

    return call_origin(
        numpy.cov, m, y, rowvar, bias, ddof, fweights, aweights, dtype=dtype
    )


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the maximum of an array or maximum along an axis.

    For full documentation refer to :obj:`numpy.max`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which to operate. By default, flattened input is
        used. If this is a tuple of integers, the minimum is selected over
        multiple axes, instead of a single axis or all the axes as before.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. Must be of the
        same shape and buffer length as the expected output.
        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        Maximum of `a`. If `axis` is ``None``, the result is a zero-dimensional
        array. If `axis` is an integer, the result is an array of dimension
        ``a.ndim - 1``. If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    Limitations
    -----------.
    Parameters `where`, and `initial` are only supported with their default
    values. Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.min` : Return the minimum of an array.
    :obj:`dpnp.maximum` : Element-wise maximum of two arrays, propagates NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignores NaNs.
    :obj:`dpnp.amax` : The maximum value of an array along a given axis,
                       propagates NaNs.
    :obj:`dpnp.nanmax` : The maximum value of an array along a given axis,
                         ignores NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.max(a)
    array(3)

    >>> np.max(a, axis=0)   # Maxima along the first axis
    array([2, 3])
    >>> np.max(a, axis=1)   # Maxima along the second axis
    array([1, 3])

    >>> b = np.arange(5, dtype=float)
    >>> b[2] = np.NaN
    >>> np.max(b)
    array(nan)

    """

    dpnp.check_limitations(initial=initial, where=where)
    usm_a = dpnp.get_usm_ndarray(a)

    return dpnp_wrap_reduction_call(
        a,
        out,
        dpt.max,
        _get_comparison_res_dt,
        usm_a,
        axis=axis,
        keepdims=keepdims,
    )


def mean(a, /, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    """
    Compute the arithmetic mean along the specified axis.

    For full documentation refer to :obj:`numpy.mean`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which the arithmetic means must be computed. If
        a tuple of unique integers, the means are computed over multiple
        axes. If ``None``, the mean is computed over the entire array.
        Default: ``None``.
    dtype : {None, dtype}, optional
        Type to use in computing the mean. By default, if `a` has a
        floating-point data type, the returned array will have
        the same data type as `a`.
        If `a` has a boolean or integral data type, the returned array
        will have the default floating point data type for the device
        where input array `a` is allocated.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary. Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array. Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the arithmetic means along the specified axis(axes).
        If the input is a zero-size array, an array containing NaN values is
        returned.

    Limitations
    -----------
    Parameter `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.average` : Weighted average.
    :obj:`dpnp.std` : Compute the standard deviation along the specified axis.
    :obj:`dpnp.var` : Compute the variance along the specified axis.
    :obj:`dpnp.nanmean` : Compute the arithmetic mean along the specified axis,
                          ignoring NaNs.
    :obj:`dpnp.nanstd` : Compute the standard deviation along
                         the specified axis, while ignoring NaNs.
    :obj:`dpnp.nanvar` : Compute the variance along the specified axis,
                         while ignoring NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.mean(a)
    array(2.5)
    >>> np.mean(a, axis=0)
    array([2., 3.])
    >>> np.mean(a, axis=1)
    array([1.5, 3.5])

    """

    dpnp.check_limitations(where=where)

    dpt_array = dpnp.get_usm_ndarray(a)
    result = dpnp_array._create_from_usm_ndarray(
        dpt.mean(dpt_array, axis=axis, keepdims=keepdims)
    )
    result = result.astype(dtype) if dtype is not None else result

    return dpnp.get_result_array(result, out, casting="same_kind")


def median(x1, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.

    For full documentation refer to :obj:`numpy.median`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameter `axis` is supported only with default value ``None``.
    Parameter `out` is supported only with default value ``None``.
    Parameter `overwrite_input` is supported only with default value ``False``.
    Parameter `keepdims` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.percentile` : Compute the q-th percentile of the data
                             along the specified axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> np.median(a)
    3.5

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if axis is not None:
            pass
        elif out is not None:
            pass
        elif overwrite_input:
            pass
        elif keepdims:
            pass
        else:
            result_obj = dpnp_median(x1_desc).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.median, x1, axis, out, overwrite_input, keepdims)


def min(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the minimum of an array or maximum along an axis.

    For full documentation refer to :obj:`numpy.min`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which to operate. By default, flattened input is
        used. If this is a tuple of integers, the minimum is selected over
        multiple axes, instead of a single axis or all the axes as before.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. Must be of the
        same shape and buffer length as the expected output.
        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the input array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        Minimum of `a`. If `axis` is ``None``, the result is a zero-dimensional
        array. If `axis` is an integer, the result is an array of dimension
        ``a.ndim - 1``. If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    Limitations
    -----------
    Parameters `where`, and `initial` are only supported with their default
    values. Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.max` : Return the maximum of an array.
    :obj:`dpnp.minimum` : Element-wise minimum of two arrays, propagates NaNs.
    :obj:`dpnp.fmin` : Element-wise minimum of two arrays, ignores NaNs.
    :obj:`dpnp.amin` : The minimum value of an array along a given axis,
                       propagates NaNs.
    :obj:`dpnp.nanmin` : The minimum value of an array along a given axis,
                         ignores NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.min(a)
    array(0)

    >>> np.min(a, axis=0)   # Minima along the first axis
    array([0, 1])
    >>> np.min(a, axis=1)   # Minima along the second axis
    array([0, 2])

    >>> b = np.arange(5, dtype=float)
    >>> b[2] = np.NaN
    >>> np.min(b)
    array(nan)

    """

    dpnp.check_limitations(initial=initial, where=where)
    usm_a = dpnp.get_usm_ndarray(a)

    return dpnp_wrap_reduction_call(
        a,
        out,
        dpt.min,
        _get_comparison_res_dt,
        usm_a,
        axis=axis,
        keepdims=keepdims,
    )


def ptp(
    a,
    /,
    axis=None,
    out=None,
    keepdims=False,
):
    """
    Range of values (maximum - minimum) along an axis.

    For full documentation refer to :obj:`numpy.ptp`.

    Returns
    -------
    ptp : dpnp.ndarray
        The range of a given array.

    Limitations
    -----------
    Input array is supported as :class:`dpnp.dpnp.ndarray` or
    :class:`dpctl.tensor.usm_ndarray`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[4, 9, 2, 10],[6, 9, 7, 12]])
    >>> np.ptp(x, axis=1)
    array([8, 6])

    >>> np.ptp(x, axis=0)
    array([2, 0, 5, 2])

    >>> np.ptp(x)
    array(10)

    """

    return dpnp.subtract(
        dpnp.max(a, axis=axis, keepdims=keepdims, out=out),
        dpnp.min(a, axis=axis, keepdims=keepdims),
        out=out,
    )


def std(
    a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
):
    """
    Compute the standard deviation along the specified axis.

    For full documentation refer to :obj:`numpy.std`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which the standard deviations must be computed.
        If a tuple of unique integers is given, the standard deviations
        are computed over multiple axes. If ``None``, the standard deviation
        is computed over the entire array.
        Default: ``None``.
    dtype : {None, dtype}, optional
        Type to use in computing the standard deviation. By default,
        if `a` has a floating-point data type, the returned array
        will have the same data type as `a`.
        If `a` has a boolean or integral data type, the returned array
        will have the default floating point data type for the device
        where input array `a` is allocated.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` corresponds to the total
        number of elements over which the standard deviation is calculated.
        Default: `0.0`.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array. Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the standard deviations. If the standard
        deviation was computed over the entire array, a zero-dimensional
        array is returned.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    Notes
    -----
    Note that, for complex numbers, the absolute value is taken before squaring,
    so that the result is always real and non-negative.

    See Also
    --------
    :obj:`dpnp.ndarray.std` : corresponding function for ndarrays.
    :obj:`dpnp.var` : Compute the variance along the specified axis.
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.nanmean` : Compute the arithmetic mean along the specified axis,
                          ignoring NaNs.
    :obj:`dpnp.nanstd` : Compute the standard deviation along
                         the specified axis, while ignoring NaNs.
    :obj:`dpnp.nanvar` : Compute the variance along the specified axis,
                         while ignoring NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.std(a)
    array(1.118033988749895)
    >>> np.std(a, axis=0)
    array([1.,  1.])
    >>> np.std(a, axis=1)
    array([0.5,  0.5])

    """

    dpnp.check_supported_arrays_type(a)
    dpnp.check_limitations(where=where)

    if not isinstance(ddof, (int, float)):
        raise TypeError(
            f"An integer or float is required, but got {type(ddof)}"
        )

    if dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        result = dpnp.var(
            a,
            axis=axis,
            dtype=None,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
        )
        dpnp.sqrt(result, out=result)
    else:
        dpt_array = dpnp.get_usm_ndarray(a)
        result = dpnp_array._create_from_usm_ndarray(
            dpt.std(dpt_array, axis=axis, correction=ddof, keepdims=keepdims)
        )
        result = dpnp.get_result_array(result, out)

    if dtype is not None and out is None:
        result = result.astype(dtype, casting="same_kind")
    return result


def var(
    a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
):
    """
    Compute the variance along the specified axis.

    For full documentation refer to :obj:`numpy.var`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        axis or axes along which the variances must be computed. If a tuple
        of unique integers is given, the variances are computed over multiple
        axes. If ``None``, the variance is computed over the entire array.
        Default: ``None``.
    dtype : {None, dtype}, optional
        Type to use in computing the variance. By default, if `a` has a
        floating-point data type, the returned array will have
        the same data type as `a`.
        If `a` has a boolean or integral data type, the returned array
        will have the default floating point data type for the device
        where input array `a` is allocated.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` corresponds to the total
        number of elements over which the variance is calculated.
        Default: `0.0`.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array. Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the variances. If the variance was computed
        over the entire array, a zero-dimensional array is returned.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    Notes
    -----
    Note that, for complex numbers, the absolute value is taken before squaring,
    so that the result is always real and non-negative.

    See Also
    --------
    :obj:`dpnp.ndarray.var` : corresponding function for ndarrays.
    :obj:`dpnp.std` : Compute the standard deviation along the specified axis.
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.nanmean` : Compute the arithmetic mean along the specified axis,
                          ignoring NaNs.
    :obj:`dpnp.nanstd` : Compute the standard deviation along
                         the specified axis, while ignoring NaNs.
    :obj:`dpnp.nanvar` : Compute the variance along the specified axis,
                         while ignoring NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.var(a)
    array(1.25)
    >>> np.var(a, axis=0)
    array([1.,  1.])
    >>> np.var(a, axis=1)
    array([0.25,  0.25])

    """

    dpnp.check_supported_arrays_type(a)
    dpnp.check_limitations(where=where)

    if not isinstance(ddof, (int, float)):
        raise TypeError(
            f"An integer or float is required, but got {type(ddof)}"
        )

    if dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        # Note that if dtype is not of inexact type then arrmean
        # will not be either.
        arrmean = dpnp.mean(
            a, axis=axis, dtype=dtype, keepdims=True, where=where
        )
        x = dpnp.subtract(a, arrmean)
        x = dpnp.multiply(x, x.conj(), out=x).real
        result = dpnp.sum(
            x,
            axis=axis,
            dtype=a.real.dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

        cnt = _count_reduce_items(a, axis, where)
        cnt = numpy.max(cnt - ddof, 0).astype(result.dtype, casting="same_kind")
        if not cnt:
            cnt = dpnp.nan

        dpnp.divide(result, cnt, out=result)
    else:
        dpt_array = dpnp.get_usm_ndarray(a)
        result = dpnp_array._create_from_usm_ndarray(
            dpt.var(dpt_array, axis=axis, correction=ddof, keepdims=keepdims)
        )
        result = dpnp.get_result_array(result, out)

    if out is None and dtype is not None:
        result = result.astype(dtype, casting="same_kind")
    return result
