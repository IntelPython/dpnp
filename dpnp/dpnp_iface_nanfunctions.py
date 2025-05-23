# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2023-2025, Intel Corporation
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

# pylint: disable=duplicate-code

import warnings

import dpnp
from dpnp.dpnp_utils.dpnp_utils_statistics import dpnp_median

__all__ = [
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanprod",
    "nanstd",
    "nansum",
    "nanvar",
]


def _replace_nan_no_mask(a, val):
    """
    Replace NaNs in array `a` with `val`.

    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy. If `a` is not of inexact type,
    do nothing and return `a`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    val : float
        NaN values are set to `val` before doing the operation.

    Returns
    -------
    out : dpnp.ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.

    """

    dpnp.check_supported_arrays_type(a)
    if dpnp.issubdtype(a.dtype, dpnp.inexact):
        return dpnp.nan_to_num(a, nan=val, posinf=dpnp.inf, neginf=-dpnp.inf)

    return a


def _replace_nan(a, val):
    """
    Replace NaNs in array `a` with `val`.

    If `a` is of inexact type, make a copy of `a`, replace NaNs with
    the `val` value, and return the copy together with a boolean mask
    marking the locations where NaNs were present. If `a` is not of
    inexact type, do nothing and return `a` together with a mask of None.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    val : float
        NaN values are set to `val` before doing the operation.

    Returns
    -------
    out : dpnp.ndarray
        If `a` is of inexact type, return a copy of `a` with the NaNs
        replaced by the fill value, otherwise return `a`.
    mask: {bool, None}
        If `a` is of inexact type, return a boolean mask marking locations of
        NaNs, otherwise return ``None``.

    """

    dpnp.check_supported_arrays_type(a)
    if dpnp.issubdtype(a.dtype, dpnp.inexact):
        mask = dpnp.isnan(a)
        if not dpnp.any(mask):
            mask = None
        else:
            a = dpnp.array(a, copy=True)
            dpnp.copyto(a, val, where=mask)
    else:
        mask = None

    return a, mask


def nanargmax(a, axis=None, out=None, *, keepdims=False):
    """
    Returns the indices of the maximum values along an axis ignoring NaNs.

    For full documentation refer to :obj:`numpy.nanargmax`.

    Warning
    -------
    This function synchronizes in order to test for all-NaN slices in the array.
    This may harm performance in some applications. To avoid synchronization,
    the user is recommended to filter NaNs themselves and use `dpnp.argmax`
    on the filtered array.

    Warning
    -------
    The results cannot be trusted if a slice contains only NaNs
    and -Infs.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int}, optional
        Axis along which to operate. By default flattened input is used.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should be
        of the appropriate shape and dtype.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the array.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        If `axis` is ``None``, a zero-dimensional array containing the index of
        the first occurrence of the maximum value ignoring NaNs; otherwise,
        a non-zero-dimensional array containing the indices of the minimum
        values ignoring NaNs. The returned array must have the default array
        index data type.
        For all-NaN slices ``ValueError`` is raised.

    Limitations
    -----------
    Input array is only supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.nanargmin` : Returns the indices of the minimum values along an
                            axis, ignoring NaNs.
    :obj:`dpnp.argmax` : Returns the indices of the maximum values along
                         an axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmax(a)
    array(0)
    >>> np.nanargmax(a)
    array(1)
    >>> np.nanargmax(a, axis=0)
    array([1, 0])
    >>> np.nanargmax(a, axis=1)
    array([1, 1])

    """

    a, mask = _replace_nan(a, -dpnp.inf)
    if mask is not None:
        mask = dpnp.all(mask, axis=axis)
        if dpnp.any(mask):
            raise ValueError("All-NaN slice encountered")
    return dpnp.argmax(a, axis=axis, out=out, keepdims=keepdims)


def nanargmin(a, axis=None, out=None, *, keepdims=False):
    """
    Returns the indices of the minimum values along an axis ignoring NaNs.

    For full documentation refer to :obj:`numpy.nanargmin`.

    Warning
    -------
    This function synchronizes in order to test for all-NaN slices in the array.
    This may harm performance in some applications. To avoid synchronization,
    the user is recommended to filter NaNs themselves and use `dpnp.argmax`
    on the filtered array.

    Warning
    -------
    The results cannot be trusted if a slice contains only NaNs
    and -Infs.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int}, optional
        Axis along which to operate. By default flattened input is used.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should be
        of the appropriate shape and dtype.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the array.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        If `axis` is ``None``, a zero-dimensional array containing the index of
        the first occurrence of the minimum value ignoring NaNs; otherwise,
        a non-zero-dimensional array containing the indices of the minimum
        values ignoring NaNs. The returned array must have the default array
        index data type.
        For all-NaN slices ``ValueError`` is raised.

    Limitations
    -----------
    Input and output arrays are only supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.nanargmax` : Returns the indices of the maximum values along
                            an axis, ignoring NaNs.
    :obj:`dpnp.argmin` : Returns the indices of the minimum values along
                         an axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> np.argmin(a)
    array(0)
    >>> np.nanargmin(a)
    array(2)
    >>> np.nanargmin(a, axis=0)
    array([1, 1])
    >>> np.nanargmin(a, axis=1)
    array([1, 0])

    """

    a, mask = _replace_nan(a, dpnp.inf)
    if mask is not None:
        mask = dpnp.all(mask, axis=axis)
        if dpnp.any(mask):
            raise ValueError("All-NaN slice encountered")
    return dpnp.argmin(a, axis=axis, out=out, keepdims=keepdims)


def nancumprod(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product of array elements over a given axis treating
    Not a Numbers (NaNs) as zero. The cumulative product does not change when
    NaNs are encountered and leading NaNs are replaced by ones.

    For full documentation refer to :obj:`numpy.nancumprod`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int}, optional
        Axis along which the cumulative product is computed. The default
        is to compute the cumulative product over the flattened array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Type of the returned array and of the accumulator in which the elements
        are summed. If `dtype` is not specified, it defaults to the dtype of
        `a`, unless `a` has an integer dtype with a precision less than that of
        the default platform integer. In that case, the default platform
        integer is used.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result is returned unless `out` is specified as
        :class:`dpnp.ndarray`, in which case a reference to `out` is returned.
        The result has the same size as `a`, and the same shape as `a` if `axis`
        is not ``None`` or `a` is a 1-d array.

    See Also
    --------
    :obj:`dpnp.cumprod` : Cumulative product across array propagating NaNs.
    :obj:`dpnp.isnan` : Show which elements are NaN.

    Examples
    --------
    >>> import dpnp as np
    >>> np.nancumprod(np.array(1))
    array(1)
    >>> np.nancumprod(np.array([1]))
    array([1])
    >>> np.nancumprod(np.array([1, np.nan]))
    array([1., 1.])
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nancumprod(a)
    array([1., 2., 6., 6.])
    >>> np.nancumprod(a, axis=0)
    array([[1., 2.],
           [3., 2.]])
    >>> np.nancumprod(a, axis=1)
    array([[1., 2.],
           [3., 3.]])

    """

    a = _replace_nan_no_mask(a, 1.0)
    return dpnp.cumprod(a, axis=axis, dtype=dtype, out=out)


def nancumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of array elements over a given axis treating
    Not a Numbers (NaNs) as zero. The cumulative sum does not change when NaNs
    are encountered and leading NaNs are replaced by zeros.

    For full documentation refer to :obj:`numpy.nancumsum`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int}, optional
        Axis along which the cumulative sum is computed. The default is to
        compute the cumulative sum over the flattened array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Type of the returned array and of the accumulator in which the elements
        are summed. If `dtype` is not specified, it defaults to the dtype of
        `a`, unless `a` has an integer dtype with a precision less than that of
        the default platform integer. In that case, the default platform
        integer is used.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result is returned unless `out` is specified as
        :class:`dpnp.ndarray`, in which case a reference to `out` is returned.
        The result has the same size as `a`, and the same shape as `a` if `axis`
        is not ``None`` or `a` is a 1-d array.

    See Also
    --------
    :obj:`dpnp.cumsum` : Cumulative sum across array propagating NaNs.
    :obj:`dpnp.isnan` : Show which elements are NaN.

    Examples
    --------
    >>> import dpnp as np
    >>> np.nancumsum(np.array(1))
    array(1)
    >>> np.nancumsum(np.array([1]))
    array([1])
    >>> np.nancumsum(np.array([1, np.nan]))
    array([1., 1.])
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nancumsum(a)
    array([1., 3., 6., 6.])
    >>> np.nancumsum(a, axis=0)
    array([[1., 2.],
           [4., 2.]])
    >>> np.nancumsum(a, axis=1)
    array([[1., 3.],
           [3., 3.]])

    """

    a = _replace_nan_no_mask(a, 0.0)
    return dpnp.cumsum(a, axis=axis, dtype=dtype, out=out)


def nanmax(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the maximum of an array or maximum along an axis, ignoring any NaNs.

    For full documentation refer to :obj:`numpy.nanmax`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which maximum values must be computed. By default,
        the maximum value must be computed over the entire array. If a tuple
        of integers, maximum values must be computed over multiple axes.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) must be included in the
        result as singleton dimensions, and, accordingly, the result must be
        compatible with the input array. Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        If the maximum value was computed over the entire array,
        a zero-dimensional array containing the maximum value ignoring NaNs;
        otherwise, a non-zero-dimensional array containing the maximum values
        ignoring NaNs. The returned array must have the same data type as `a`.
        When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
        NaN is returned for that slice.

    Limitations
    -----------
    Input array is only supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, and `initial` are only supported with their default
    values.
    Otherwise ``NotImplementedError`` exception will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.nanmin` : The minimum value of an array along a given axis,
                         ignoring any NaNs.
    :obj:`dpnp.max` : The maximum value of an array along a given axis,
                      propagating any NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignoring any NaNs.
    :obj:`dpnp.maximum` : Element-wise maximum of two arrays, propagating
                          any NaNs.
    :obj:`dpnp.isnan` : Shows which elements are Not a Number (NaN).
    :obj:`dpnp.isfinite` : Shows which elements are neither NaN nor infinity.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmax(a)
    array(3.)
    >>> np.nanmax(a, axis=0)
    array([3.,  2.])
    >>> np.nanmax(a, axis=1)
    array([2.,  3.])

    When positive infinity and negative infinity are present:

    >>> np.nanmax(np.array([1, 2, np.nan, -np.inf]))
    array(2.)
    >>> np.nanmax(np.array([1, 2, np.nan, np.inf]))
    array(inf)

    """

    dpnp.check_limitations(initial=initial, where=where)

    a, mask = _replace_nan(a, -dpnp.inf)
    res = dpnp.max(a, axis=axis, out=out, keepdims=keepdims)
    if mask is None:
        return res

    mask = dpnp.all(mask, axis=axis)
    if dpnp.any(mask):
        dpnp.copyto(res, dpnp.nan, where=mask)
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
    return res


def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    """
    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    For full documentation refer to :obj:`numpy.nanmean`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which the arithmetic means must be computed. If
        a tuple of unique integers, the means are computed over multiple
        axes. If ``None``, the mean is computed over the entire array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Type to use in computing the mean. By default, if `a` has a
        floating-point data type, the returned array will have
        the same data type as `a`.
        If `a` has a boolean or integral data type, the returned array
        will have the default floating point data type for the device
        where input array `a` is allocated.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the arithmetic means along the specified axis(axes).
        If the input is a zero-size array, an array containing NaN values is
        returned. In addition, NaN is returned for slices that contain only
        NaNs.

    Limitations
    -----------
    Parameter `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.average` : Weighted average.
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.var` : Compute the variance along the specified axis.
    :obj:`dpnp.nanvar` : Compute the variance along the specified axis,
                         while ignoring NaNs.
    :obj:`dpnp.std` : Compute the standard deviation along the specified axis.
    :obj:`dpnp.nanstd` : Compute the standard deviation along the specified
                         axis, while ignoring NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanmean(a)
    array(2.6666666666666665)
    >>> np.nanmean(a, axis=0)
    array([2., 4.])
    >>> np.nanmean(a, axis=1)
    array([1., 3.5]) # may vary

    """

    dpnp.check_limitations(where=where)

    arr, mask = _replace_nan(a, 0)
    if mask is None:
        return dpnp.mean(
            arr,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
        )

    if dtype is not None:
        dtype = dpnp.dtype(dtype)
        if not dpnp.issubdtype(dtype, dpnp.inexact):
            raise TypeError("If input is inexact, then dtype must be inexact.")
    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if not dpnp.issubdtype(out.dtype, dpnp.inexact):
            raise TypeError("If input is inexact, then out must be inexact.")

    cnt_dtype = a.real.dtype if dtype is None else dtype
    cnt = dpnp.sum(
        ~mask, axis=axis, dtype=cnt_dtype, keepdims=keepdims, where=where
    )
    var_dtype = a.dtype if dtype is None else dtype
    avg = dpnp.sum(
        arr,
        axis=axis,
        dtype=var_dtype,
        out=out,
        keepdims=keepdims,
        where=where,
    )
    dpnp.divide(avg, cnt, out=avg)

    return avg


def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis, while ignoring NaNs.

    For full documentation refer to :obj:`numpy.nanmedian`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple or list of ints}, optional
        Axis or axes along which the medians are computed. The default,
        ``axis=None``, will compute the median along a flattened version of
        the array. If a sequence of axes, the array is first flattened along
        the given axes, then the median is computed along the resulting
        flattened axis.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.

        Default: ``None``.
    overwrite_input : bool, optional
       If ``True``, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       :obj:`dpnp.nanmedian`. This will save memory when you do not need to
       preserve the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted.

       Default: ``False``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result. If `a` has a floating-point data type,
        the returned array will have the same data type as `a`. If `a` has a
        boolean or integral data type, the returned array will have the
        default floating point data type for the device where input array `a`
        is allocated.

    See Also
    --------
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.median` : Compute the median along the specified axis.
    :obj:`dpnp.percentile` : Compute the q-th percentile of the data
                             along the specified axis.

    Notes
    -----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the
    middle value of a sorted copy of ``V``, ``V_sorted`` - i.e.,
    ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
    two middle values of ``V_sorted`` when ``N`` is even.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[10.0, 7, 4], [3, 2, 1]])
    >>> a[0, 1] = np.nan
    >>> a
    array([[10., nan,  4.],
           [ 3.,  2.,  1.]])
    >>> np.median(a)
    array(nan)
    >>> np.nanmedian(a)
    array(3.)

    >>> np.nanmedian(a, axis=0)
    array([6.5, 2., 2.5])
    >>> np.nanmedian(a, axis=1)
    array([7., 2.])

    >>> b = a.copy()
    >>> np.nanmedian(b, axis=1, overwrite_input=True)
    array([7., 2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.nanmedian(b, axis=None, overwrite_input=True)
    array(3.)
    >>> assert not np.all(a==b)

    """

    dpnp.check_supported_arrays_type(a)
    ignore_nan = False
    if dpnp.issubdtype(a.dtype, dpnp.inexact):
        mask = dpnp.isnan(a)
        if dpnp.any(mask):
            ignore_nan = True

    return dpnp_median(
        a, axis, out, overwrite_input, keepdims, ignore_nan=ignore_nan
    )


def nanmin(a, axis=None, out=None, keepdims=False, initial=None, where=True):
    """
    Return the minimum of an array or minimum along an axis, ignoring any NaNs.

    For full documentation refer to :obj:`numpy.nanmin`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which minimum values must be computed. By default,
        the minimum value must be computed over the entire array. If a tuple
        of integers, minimum values must be computed over multiple axes.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) must be included in the
        result as singleton dimensions, and, accordingly, the result must be
        compatible with the input array. Otherwise, if ``False``, the reduced
        axes (dimensions) must not be included in the result.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        If the minimum value was computed over the entire array,
        a zero-dimensional array containing the minimum value ignoring NaNs;
        otherwise, a non-zero-dimensional array containing the minimum values
        ignoring NaNs. The returned array must have the same data type as `a`.
        When all-NaN slices are encountered a ``RuntimeWarning`` is raised and
        NaN is returned for that slice.

    Limitations
    -----------
    Input array is only supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, and `initial` are only supported with their default
    values.
    Otherwise ``NotImplementedError`` exception will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.nanmax` : The maximum value of an array along a given axis,
                         ignoring any NaNs.
    :obj:`dpnp.min` : The minimum value of an array along a given axis,
                      propagating any NaNs.
    :obj:`dpnp.fmin` : Element-wise minimum of two arrays, ignoring any NaNs.
    :obj:`dpnp.minimum` : Element-wise minimum of two arrays, propagating
                          any NaNs.
    :obj:`dpnp.isnan` : Shows which elements are Not a Number (NaN).
    :obj:`dpnp.isfinite` : Shows which elements are neither NaN nor infinity.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> np.nanmin(a)
    array(1.)
    >>> np.nanmin(a, axis=0)
    array([1.,  2.])
    >>> np.nanmin(a, axis=1)
    array([1.,  3.])

    When positive infinity and negative infinity are present:

    >>> np.nanmin(np.array([1, 2, np.nan, np.inf]))
    array(1.)
    >>> np.nanmin(np.array([1, 2, np.nan, -np.inf]))
    array(-inf)

    """

    dpnp.check_limitations(initial=initial, where=where)

    a, mask = _replace_nan(a, +dpnp.inf)
    res = dpnp.min(a, axis=axis, out=out, keepdims=keepdims)
    if mask is None:
        return res

    mask = dpnp.all(mask, axis=axis)
    if dpnp.any(mask):
        dpnp.copyto(res, dpnp.nan, where=mask)
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=2)
    return res


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
    Return the product of array elements over a given axis treating
    Not a Numbers (NaNs) as ones.

    For full documentation refer to :obj:`numpy.nanprod`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which the product is computed. The default is to
        compute the product of the flattened array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        The type of the returned array and of the accumulator in which the
        elements are multiplied. By default, the dtype of `a` is used. An
        exception is when `a` has an integer type with less precision than
        the platform (u)intp. In that case, the default will be either (u)int32
        or (u)int64 depending on whether the platform is 32 or 64 bits. For
        inexact inputs, dtype must be inexact.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternate output array in which to place the result. If provided, it
        must have the same shape as the expected output, but the type will be
        cast if necessary. The casting of NaN to integer
        can yield unexpected results.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If ``True``, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast
        correctly against the original `a`.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result is returned unless `out` is specified,
        in which case it is returned.

    See Also
    --------
    :obj:`dpnp.prod` : Returns product across array propagating NaNs.
    :obj:`dpnp.isnan` : Test element-wise for NaN and return result
                        as a boolean array.

    Limitations
    -----------
    Input array is only supported as either :class:`dpnp.ndarray` or
    :class:`dpctl.tensor.usm_ndarray`.
    Parameters `initial`, and `where` are only supported with their default
    values.
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

    a = _replace_nan_no_mask(a, 1.0)
    return dpnp.prod(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


def nansum(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    """
    Return the sum of array elements over a given axis treating
    Not a Numbers (NaNs) as zero.

    For full documentation refer to :obj:`numpy.nansum`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int or tuple of ints}, optional
        Axis or axes along which the sum is computed. The default is to compute
        the sum of the flattened array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        The type of the returned array and of the accumulator in which the
        elements are summed. By default, the dtype of `a` is used. An exception
        is when `a` has an integer type with less precision than the platform
        (u)intp. In that case, the default will be either (u)int32 or (u)int64
        depending on whether the platform is 32 or 64 bits. For inexact inputs,
        dtype must be inexact.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternate output array in which to place the result. If provided, it
        must have the same shape as the expected output, but the type will be
        cast if necessary. The casting of NaN to integer can yield unexpected
        results.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result will
        broadcast correctly against the original `a`.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result is returned unless `out` is specified,
        in which it is returned. The result has the same size as `a`, and the
        same shape as `a` if `axis` is not ``None`` or `a` is a 1-d array.

    Limitations
    -----------
    Parameters `initial` and `where` are supported with their default values.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.sum` : Sum across array propagating NaNs.
    :obj:`dpnp.isnan` : Show which elements are NaN.
    :obj:`dpnp.isfinite` : Show which elements are not NaN or +/-inf.

    Notes
    -----
    If both positive and negative infinity are present, the sum will be Not
    A Number (NaN).

    Examples
    --------
    >>> import dpnp as np
    >>> np.nansum(np.array([1]))
    array(1)
    >>> np.nansum(np.array([1, np.nan]))
    array(1.)
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> np.nansum(a)
    array(3.)
    >>> np.nansum(a, axis=0)
    array([2.,  1.])
    >>> np.nansum(np.array([1, np.nan, np.inf]))
    array(inf)
    >>> np.nansum(np.array([1, np.nan, -np.inf]))
    array(-inf)
    >>> # both +/- infinity present
    >>> np.nansum(np.array([1, np.nan, np.inf, -np.inf]))
    array(nan)

    """

    a = _replace_nan_no_mask(a, 0.0)
    return dpnp.sum(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


def nanstd(
    a,
    axis=None,
    dtype=None,
    out=None,
    ddof=0,
    keepdims=False,
    *,
    where=True,
    mean=None,
    correction=None,
):
    """
    Compute the standard deviation along the specified axis,
    while ignoring NaNs.

    For full documentation refer to :obj:`numpy.nanstd`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which the standard deviations must be computed.
        If a tuple of unique integers is given, the standard deviations are
        computed over multiple axes. If ``None``, the standard deviation is
        computed over the entire array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Type to use in computing the standard deviation. By default, if `a` has
        a floating-point data type, the returned array will have the same data
        type as `a`. If `a` has a boolean or integral data type, the returned
        array will have the default floating point data type for the device
        where input array `a` is allocated.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.

        Default: ``None``.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom. The divisor used in calculations is
        ``N - ddof``, where ``N`` the number of non-NaN elements.

        Default: ``0.0``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains compatible
        with the input array according to Array Broadcasting rules. Otherwise,
        if ``False``, the reduced axes are not included in the returned array.

        Default: ``False``.
    mean : {dpnp.ndarray, usm_ndarray}, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this `nanstd` function.

        Default: ``None``.

    correction : {None, int, float}, optional
        Array API compatible name for the `ddof` parameter. Only one of them
        can be provided at the same time.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the standard deviations. If the standard deviation
        was computed over the entire array, a zero-dimensional array is
        returned. If `ddof` is >= the number of non-NaN elements in a slice or
        the slice contains only NaNs, then the result for that slice is NaN.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    Notes
    -----
    The standard deviation is the square root of the average of the squared
    deviations from the mean: ``std = sqrt(mean(abs(x - x.mean())**2))``.

    The average squared deviation is normally calculated as ``x.sum() / N``,
    where ``N = len(x)``. If, however, `ddof` is specified, the divisor
    ``N - ddof`` is used instead. In standard statistical practice, ``ddof=1``
    provides an unbiased estimator of the variance of the infinite population.
    ``ddof=0`` provides a maximum likelihood estimate of the variance for
    normally distributed variables.
    The standard deviation computed in this function is the square root of
    the estimated variance, so even with ``ddof=1``, it will not be an unbiased
    estimate of the standard deviation per se.

    Note that, for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and non-negative.

    See Also
    --------
    :obj:`dpnp.var` : Compute the variance along the specified axis.
    :obj:`dpnp.mean` : Compute the arithmetic mean along the specified axis.
    :obj:`dpnp.std` : Compute the standard deviation along the specified axis.
    :obj:`dpnp.nanmean` : Compute the arithmetic mean along the specified axis,
                          ignoring NaNs.
    :obj:`dpnp.nanvar` : Compute the variance along the specified axis,
                         while ignoring NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, np.nan], [3, 4]])
    >>> np.nanstd(a)
    array(1.24721913)
    >>> np.nanstd(a, axis=0)
    array([1., 0.])
    >>> np.nanstd(a, axis=1)
    array([0. , 0.5])  # may vary

    Using the mean keyword to save computation time:

    >>> a = np.array([[14, 8, np.nan, 10], [7, 9, 10, 11], [np.nan, 15, 5, 10]])
    >>> mean = np.nanmean(a, axis=1, keepdims=True)
    >>> np.nanstd(a, axis=1, mean=mean)
    array([2.49443826, 1.47901995, 4.0824829 ])

    """

    dpnp.check_limitations(where=where)
    if not isinstance(ddof, (int, float)):
        raise TypeError(
            f"An integer or float is required, but got {type(ddof)}"
        )

    res = nanvar(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
        mean=mean,
        correction=correction,
    )
    return dpnp.sqrt(res, out=res)


def nanvar(
    a,
    axis=None,
    dtype=None,
    out=None,
    ddof=0,
    keepdims=False,
    *,
    where=True,
    mean=None,
    correction=None,
):
    """
    Compute the variance along the specified axis, while ignoring NaNs.

    For full documentation refer to :obj:`numpy.nanvar`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which the variances must be computed. If a tuple
        of unique integers is given, the variances are computed over multiple
        axes. If ``None``, the variance is computed over the entire array.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        Type to use in computing the variance. By default, if `a` has a
        floating-point data type, the returned array will have
        the same data type as `a`. If `a` has a boolean or integral data type,
        the returned array will have the default floating point data type for
        the device where input array `a` is allocated.

        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.

        Default: ``None``.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom. The divisor used in calculations is
        ``N - ddof``, where ``N`` represents the number of non-NaN elements.

        Default: ``0.0``.
    keepdims : {None, bool}, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains compatible
        with the input array according to Array Broadcasting rules. Otherwise,
        if ``False``, the reduced axes are not included in the returned array.

        Default: ``False``.
    mean : {dpnp.ndarray, usm_ndarray}, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this `nanvar` function.

        Default: ``None``.

    correction : {None, int, float}, optional
        Array API compatible name for the `ddof` parameter. Only one of them
        can be provided at the same time.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the variances. If the variance was computed over
        the entire array, a zero-dimensional array is returned. If `ddof` is >=
        the number of non-NaN elements in a slice or the slice contains only
        NaNs, then the result for that slice is NaN.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    Notes
    -----
    The variance is the average of the squared deviations from the mean,
    that is ``var = mean(abs(x - x.mean())**2)``.

    The mean is normally calculated as ``x.sum() / N``, where ``N = len(x)``.
    If, however, `ddof` is specified, the divisor ``N - ddof`` is used instead.
    In standard statistical practice, ``ddof=1`` provides an unbiased estimator
    of the variance of a hypothetical infinite population. ``ddof=0`` provides
    a maximum likelihood estimate of the variance for normally distributed
    variables.

    Note that, for complex numbers, the absolute value is taken before squaring,
    so that the result is always real and non-negative.

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
    array(1.55555556)
    >>> np.nanvar(a, axis=0)
    array([1., 0.])
    >>> np.nanvar(a, axis=1)
    array([0.  , 0.25])  # may vary

    Using the mean keyword to save computation time:

    >>> a = np.array([[14, 8, np.nan, 10], [7, 9, 10, 11], [np.nan, 15, 5, 10]])
    >>> mean = np.nanmean(a, axis=1, keepdims=True)
    >>> np.nanvar(a, axis=1, mean=mean)
    array([ 6.22222222,  2.1875    , 16.66666667])

    """

    dpnp.check_limitations(where=where)
    if not isinstance(ddof, (int, float)):
        raise TypeError(
            f"An integer or float is required, but got {type(ddof)}"
        )

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
            correction=correction,
        )

    if dtype is not None:
        dtype = dpnp.dtype(dtype)
        if not dpnp.issubdtype(dtype, dpnp.inexact):
            raise TypeError("If input is inexact, then dtype must be inexact.")

    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if not dpnp.issubdtype(out.dtype, dpnp.inexact):
            raise TypeError("If input is inexact, then out must be inexact.")

    if correction is not None:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously."
            )
        ddof = correction

    # Compute mean
    cnt = dpnp.sum(
        ~mask, axis=axis, dtype=dpnp.intp, keepdims=True, where=where
    )

    if mean is not None:
        avg = mean
    else:
        avg = dpnp.sum(arr, axis=axis, dtype=dtype, keepdims=True, where=where)
        avg = dpnp.divide(avg, cnt, out=avg)

    # Compute squared deviation from mean
    if arr.dtype == avg.dtype:
        arr = dpnp.subtract(arr, avg, out=arr)
    else:
        arr = dpnp.subtract(arr, avg)
    dpnp.copyto(arr, 0.0, where=mask)

    if dpnp.issubdtype(arr.dtype, dpnp.complexfloating):
        sqr = dpnp.multiply(arr, arr.conj(), out=arr).real
    else:
        sqr = dpnp.square(arr, out=arr)

    # Compute variance
    var = dpnp.sum(
        sqr,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        where=where,
    )

    if var.ndim < cnt.ndim:
        cnt = cnt.squeeze(axis)
    dof = cnt - ddof
    dpnp.divide(var, dof, out=var)

    isbad = dof <= 0
    if dpnp.any(isbad):
        # NaN, inf, or negative numbers are all possible bad
        # values, so explicitly replace them with NaN.
        dpnp.copyto(var, dpnp.nan, where=isbad)

    return var
