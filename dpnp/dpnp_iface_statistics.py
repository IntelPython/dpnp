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
from dpnp.dpnp_algo import *
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import *
from dpnp.dpnp_utils.dpnp_utils_statistics import dpnp_cov

__all__ = [
    "amax",
    "amin",
    "average",
    "bincount",
    "correlate",
    "cov",
    "histogram",
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
    Calculates the number of items used in a reduction operation along the specified axis or axes

    Parameters
    ----------
    arr : {dpnp_array, usm_ndarray}
        Input array.
    axis : int or tuple of ints, optional
        axis or axes along which the number of items used in a reduction operation must be counted.
        If a tuple of unique integers is given, the items are counted over multiple axes.
        If ``None``, the variance is computed over the entire array.
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


def average(x1, axis=None, weights=None, returned=False):
    """
    Compute the weighted average along the specified axis.

    For full documentation refer to :obj:`numpy.average`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameter `axis` is supported only with default value ``None``.
    Parameter `weights` is supported only with default value ``None``.
    Parameter `returned` is supported only with default value ``False``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.

    Examples
    --------
    >>> import dpnp as np
    >>> data = np.arange(1, 5)
    >>> [i for i in data]
    [1, 2, 3, 4]
    >>> np.average(data)
    2.5

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if axis is not None:
            pass
        elif weights is not None:
            pass
        elif returned:
            pass
        else:
            result_obj = dpnp_average(x1_desc)
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.average, x1, axis, weights, returned)


def bincount(x1, weights=None, minlength=0):
    """
    Count number of occurrences of each value in array of non-negative ints.

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
    cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *, dtype=None)

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


def histogram(a, bins=10, range=None, density=None, weights=None):
    """
    Compute the histogram of a dataset.

    For full documentation refer to :obj:`numpy.histogram`.

    Examples
    --------
    >>> import dpnp
    >>> dpnp.histogram([1, 2, 1], bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> dpnp.histogram(dpnp.arange(4), bins=dpnp.arange(5), density=True)
    (array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4]))
    >>> dpnp.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))
    >>> a = dpnp.arange(5)
    >>> hist, bin_edges = dpnp.histogram(a, density=True)
    >>> hist
    array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])
    >>> hist.sum()
    2.4999999999999996
    >>> res = dpnp.sum(hist * dpnp.diff(bin_edges))
    >>> print(res)
    1.0

    """

    return call_origin(
        numpy.histogram,
        a=a,
        bins=bins,
        range=range,
        density=density,
        weights=weights,
    )


def max(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the maximum of an array or maximum along an axis.

    For full documentation refer to :obj:`numpy.max`.

    Parameters
    ----------
    a :  {dpnp_array, usm_ndarray}
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes along which maximum values must be computed. By default,
        the maximum value must be computed over the entire array. If a tuple of integers,
        maximum values must be computed over multiple axes.
        Default: ``None``.
    out :  {dpnp_array, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool
        If ``True``, the reduced axes (dimensions) must be included in the
        result as singleton dimensions, and, accordingly, the result must be
        compatible with the input array. Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        If the maximum value was computed over the entire array, a zero-dimensional array
        containing the maximum value; otherwise, a non-zero-dimensional array
        containing the maximum values. The returned array must have
        the same data type as `a`.

    Limitations
    -----------
    Input array is only supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, and `initial` are only supported with their default values.
    Otherwise ``NotImplementedError`` exception will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.min` : Return the minimum of an array.
    :obj:`dpnp.maximum` : Element-wise maximum of two arrays, propagates NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignores NaNs.
    :obj:`dpnp.amax` : The maximum value of an array along a given axis, propagates NaNs.
    :obj:`dpnp.nanmax` : The maximum value of an array along a given axis, ignores NaNs.

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

    if initial is not None:
        raise NotImplementedError(
            "initial keyword argument is only supported with its default value."
        )
    elif where is not True:
        raise NotImplementedError(
            "where keyword argument is only supported with its default value."
        )
    else:
        dpt_array = dpnp.get_usm_ndarray(a)
        result = dpnp_array._create_from_usm_ndarray(
            dpt.max(dpt_array, axis=axis, keepdims=keepdims)
        )

        return dpnp.get_result_array(result, out)


def mean(a, /, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    """
    Compute the arithmetic mean along the specified axis.

    For full documentation refer to :obj:`numpy.mean`.

    Returns
    -------
    out : dpnp.ndarray
        an array containing the mean values of the elements along the specified axis(axes).
        If the input is a zero-size array, an array containing NaN values is returned.

    Limitations
    -----------
    Parameters `a` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameter `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

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

    if where is not True:
        raise NotImplementedError(
            "where keyword argument is only supported with its default value."
        )
    else:
        dpt_array = dpnp.get_usm_ndarray(a)
        result = dpnp_array._create_from_usm_ndarray(
            dpt.mean(dpt_array, axis=axis, keepdims=keepdims)
        )
        result = result.astype(dtype) if dtype is not None else result

        return dpnp.get_result_array(result, out)


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
    a :  {dpnp_array, usm_ndarray}
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes along which minimum values must be computed. By default,
        the minimum value must be computed over the entire array. If a tuple of integers,
        minimum values must be computed over multiple axes.
        Default: ``None``.
    out :  {dpnp_array, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If ``True``, the reduced axes (dimensions) must be included in the
        result as singleton dimensions, and, accordingly, the result must be
        compatible with the input array. Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        If the minimum value was computed over the entire array, a zero-dimensional array
        containing the minimum value; otherwise, a non-zero-dimensional array
        containing the minimum values. The returned array must have
        the same data type as `a`.

    Limitations
    -----------
    Input array is only supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, and `initial` are only supported with their default values.
    Otherwise ``NotImplementedError`` exception will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.max` : Return the maximum of an array.
    :obj:`dpnp.minimum` : Element-wise minimum of two arrays, propagates NaNs.
    :obj:`dpnp.fmin` : Element-wise minimum of two arrays, ignores NaNs.
    :obj:`dpnp.amin` : The minimum value of an array along a given axis, propagates NaNs.
    :obj:`dpnp.nanmin` : The minimum value of an array along a given axis, ignores NaNs.

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

    if initial is not None:
        raise NotImplementedError(
            "initial keyword argument is only supported with its default value."
        )
    elif where is not True:
        raise NotImplementedError(
            "where keyword argument is only supported with its default value."
        )
    else:
        dpt_array = dpnp.get_usm_ndarray(a)
        result = dpnp_array._create_from_usm_ndarray(
            dpt.min(dpt_array, axis=axis, keepdims=keepdims)
        )

        return dpnp.get_result_array(result, out)


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
    Input array is supported as :class:`dpnp.dpnp_array` or :class:`dpctl.tensor.usm_ndarray`.

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
    a : {dpnp_array, usm_ndarray}:
        nput array.
    axis : int or tuple of ints, optional
        Axis or axes along which the variances must be computed. If a tuple
        of unique integers is given, the variances are computed over multiple axes.
        If ``None``, the variance is computed over the entire array.
        Default: `None`.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default real-valued floating-point data type is used,
        for arrays of float types it is the same as the array type.
    out : {dpnp_array, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` corresponds to the total
        number of elements over which the variance is calculated.
        Default: `0.0`.
    keepdims : bool, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array. Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        an array containing the standard deviations. If the standard
        deviation was computed over the entire array, a zero-dimensional
        array is returned.

        If `a` has a real-valued floating-point data type, the returned
        array will have the same data type as `a`.
        If `a` has a boolean or integral data type, the returned array
        will have the default floating point data type for the device
        where input array `a` is allocated.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Notes
    -----
    Note that, for complex numbers, the absolute value is taken before squaring,
    so that the result is always real and nonnegative.

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

    if where is not True:
        raise NotImplementedError(
            "where keyword argument is only supported with its default value."
        )
    elif not isinstance(ddof, (int, float)):
        raise TypeError(
            "An integer or float is required, but got {}".format(type(ddof))
        )
    else:
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
                dpt.std(
                    dpt_array, axis=axis, correction=ddof, keepdims=keepdims
                )
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
    a : {dpnp_array, usm_ndarray}:
        Input array.
    axis : int or tuple of ints, optional
        axis or axes along which the variances must be computed. If a tuple
        of unique integers is given, the variances are computed over multiple axes.
        If ``None``, the variance is computed over the entire array.
        Default: `None`.
    dtype : dtype, optional
        Type to use in computing the variance. For arrays of integer type
        the default real-valued floating-point data type is used,
        for arrays of float types it is the same as the array type.
    out : {dpnp_array, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` corresponds to the total
        number of elements over which the variance is calculated.
        Default: `0.0`.
    keepdims : bool, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array. Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        an array containing the variances. If the variance was computed
        over the entire array, a zero-dimensional array is returned.

        If `a` has a real-valued floating-point data type, the returned
        array will have the same data type as `a`.
        If `a` has a boolean or integral data type, the returned array
        will have the default floating point data type for the device
        where input array `a` is allocated.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Notes
    -----
    Note that, for complex numbers, the absolute value is taken before squaring,
    so that the result is always real and nonnegative.

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
    if where is not True:
        raise NotImplementedError(
            "where keyword argument is only supported with its default value."
        )
    elif not isinstance(ddof, (int, float)):
        raise TypeError(
            "An integer or float is required, but got {}".format(type(ddof))
        )
    else:
        if dpnp.issubdtype(a.dtype, dpnp.complexfloating):
            # Note that if dtype is not of inexact type then arrmean will not be either.
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
            cnt = numpy.max(cnt - ddof, 0).astype(
                result.dtype, casting="same_kind"
            )
            if not cnt:
                cnt = dpnp.nan

            dpnp.divide(result, cnt, out=result)
        else:
            dpt_array = dpnp.get_usm_ndarray(a)
            result = dpnp_array._create_from_usm_ndarray(
                dpt.var(
                    dpt_array, axis=axis, correction=ddof, keepdims=keepdims
                )
            )
            result = dpnp.get_result_array(result, out)

        if out is None and dtype is not None:
            result = result.astype(dtype, casting="same_kind")
        return result
