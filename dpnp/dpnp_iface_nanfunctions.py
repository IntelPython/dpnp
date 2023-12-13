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
Interface of the nan functions of the DPNP

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

from .dpnp_algo import *
from .dpnp_utils import *

__all__ = [
    "nancumprod",
    "nancumsum",
    "nanprod",
    "nansum",
    "nanvar",
]


def _replace_nan(a, val):
    """
    Replace NaNs in array `a` with `val`.

    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.

    Parameters
    ----------
    a : {dpnp_array, usm_ndarray}
        Input array.
    val : float
        NaN values are set to `val` before doing the operation.

    Returns
    -------
    out : {dpnp_array}
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return ``None``.

    """

    dpnp.check_supported_arrays_type(a)
    if issubclass(a.dtype.type, dpnp.inexact):
        mask = dpnp.isnan(a)
        if not dpnp.any(mask):
            mask = None
        else:
            a = dpnp.array(a, copy=True)
            dpnp.copyto(a, val, where=mask)
    else:
        mask = None

    return a, mask


def nancumprod(x1, **kwargs):
    """
    Return the cumulative product of array elements over a given axis treating Not a Numbers (NaNs) as one.

    For full documentation refer to :obj:`numpy.nancumprod`.

    Limitations
    -----------
    Parameter `x` is supported as :class:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.cumprod` : Return the cumulative product of elements along a given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1., np.nan])
    >>> result = np.nancumprod(a)
    >>> [x for x in result]
    [1.0, 1.0]
    >>> b = np.array([[1., 2., np.nan], [4., np.nan, 6.]])
    >>> result = np.nancumprod(b)
    >>> [x for x in result]
    [1.0, 2.0, 2.0, 8.0, 8.0, 48.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc and not kwargs:
        return dpnp_nancumprod(x1_desc).get_pyobj()

    return call_origin(numpy.nancumprod, x1, **kwargs)


def nancumsum(x1, **kwargs):
    """
    Return the cumulative sum of the elements along a given axis.

    For full documentation refer to :obj:`numpy.nancumsum`.

    Limitations
    -----------
    Parameter `x` is supported as :class:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.cumsum` : Return the cumulative sum of the elements along a given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1., np.nan])
    >>> result = np.nancumsum(a)
    >>> [x for x in result]
    [1.0, 1.0]
    >>> b = np.array([[1., 2., np.nan], [4., np.nan, 6.]])
    >>> result = np.nancumprod(b)
    >>> [x for x in result]
    [1.0, 3.0, 3.0, 7.0, 7.0, 13.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc and not kwargs:
        return dpnp_nancumsum(x1_desc).get_pyobj()

    return call_origin(numpy.nancumsum, x1, **kwargs)


def nansum(x1, **kwargs):
    """
    Calculate sum() function treating 'Not a Numbers' (NaN) as zero.

    For full documentation refer to :obj:`numpy.nansum`.

    Limitations
    -----------
    Parameter `x1` is supported as :class:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.nansum(np.array([1, 2]))
    3
    >>> np.nansum(np.array([[1, 2], [3, 4]]))
    10

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc and not kwargs:
        result_obj = dpnp_nansum(x1_desc).get_pyobj()
        result = dpnp.convert_single_elem_array_to_scalar(result_obj)
        return result

    return call_origin(numpy.nansum, x1, **kwargs)


def nanprod(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    """
    Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.

    For full documentation refer to :obj:`numpy.nanprod`.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result is returned unless `out` is specified, in which case it is returned.

    See Also
    --------
    :obj:`dpnp.prod` : Returns product across array propagating NaNs.
    :obj:`dpnp.isnan` : Test element-wise for NaN and return result as a boolean array.

    Limitations
    -----------
    Input array is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `initial`, and `where` are only supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.nanprod(np.array(1))
    array(1)
    >>> np.nanprod(np.array([1]))
    array(1)
    >>> np.nanprod(np.array([1, np.nan]))
    array(1.0)
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanprod(a)
    array(6.0)
    >>> np.nanprod(a, axis=0)
    array([3., 2.])

    """

    a, mask = _replace_nan(a, 1)

    return dpnp.prod(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


def nanvar(
    a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True
):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    For full documentation refer to :obj:`numpy.nanvar`.

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
    Input array data types are limited by real valued data types.

    See Also
    --------
    :obj:`dpnp.var` : Compute the variance along the specified axis.
    :obj:`dpnp.std` : Compute the standard deviation along the specified axis.
    :obj:`dpnp.nanmean` : Compute the arithmetic mean along the specified axis,
                          ignoring NaNs.
    :obj:`dpnp.nanstd` : Compute the standard deviation along
                         the specified axis, while ignoring NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanvar(a)
    array(1.5555555555555554)
    >>> np.nanvar(a, axis=0)
    array([1.,  0.])
    >>> np.nanvar(a, axis=1)
    array([0.,  0.25])  # may vary

    """

    if where is not True:
        raise NotImplementedError(
            "where keyword argument is only supported with its default value."
        )
    elif not isinstance(ddof, (int, float)):
        raise TypeError(
            "An integer or float is required, but got {}".format(type(ddof))
        )
    else:
        arr, mask = _replace_nan(a, 0)
        if mask is None:
            return dpnp.var(
                arr,
                axis=axis,
                dtype=dtype,
                out=out,
                ddof=ddof,
                keepdims=keepdims,
                where=where,
            )

        if dtype is not None:
            dtype = dpnp.dtype(dtype)
            if not issubclass(dtype.type, dpnp.inexact):
                raise TypeError(
                    "If input is inexact, then dtype must be inexact."
                )
        if out is not None and not issubclass(out.dtype.type, dpnp.inexact):
            raise TypeError("If input is inexact, then out must be inexact.")

        # Compute mean
        cnt = dpnp.sum(
            ~mask, axis=axis, dtype=dpnp.intp, keepdims=True, where=where
        )
        avg = dpnp.sum(arr, axis=axis, dtype=dtype, keepdims=True, where=where)
        avg = dpnp.divide(avg, cnt)

        # Compute squared deviation from mean.
        arr = dpnp.subtract(arr, avg)
        dpnp.copyto(arr, 0.0, where=mask, casting="safe")
        if dpnp.issubdtype(arr.dtype, dpnp.complexfloating):
            sqr = dpnp.multiply(arr, arr.conj(), out=arr).real
        else:
            sqr = dpnp.multiply(arr, arr, out=arr)

        # Compute variance
        var_dtype = a.real.dtype if dtype is None else dtype
        var = dpnp.sum(
            sqr,
            axis=axis,
            dtype=var_dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

        if var.ndim < cnt.ndim:
            cnt = cnt.squeeze(axis)
        cnt = cnt.astype(var.dtype, casting="same_kind")
        cnt -= ddof
        dpnp.divide(var, cnt, out=var)

        isbad = cnt <= 0
        if dpnp.any(isbad):
            # NaN, inf, or negative numbers are all possible bad
            # values, so explicitly replace them with NaN.
            dpnp.copyto(var, dpnp.nan, where=isbad, casting="same_kind")

        return var
