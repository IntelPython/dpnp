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
Interface of the Mathematical part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import dpctl.tensor as dpt
import dpctl.utils as du
import numpy
from numpy.core.numeric import (
    normalize_axis_index,
    normalize_axis_tuple,
)

import dpnp
from dpnp.dpnp_array import dpnp_array

from .dpnp_algo import *
from .dpnp_algo.dpnp_elementwise_common import (
    check_nd_call_func,
    dpnp_abs,
    dpnp_add,
    dpnp_ceil,
    dpnp_conj,
    dpnp_copysign,
    dpnp_divide,
    dpnp_floor,
    dpnp_floor_divide,
    dpnp_imag,
    dpnp_maximum,
    dpnp_minimum,
    dpnp_multiply,
    dpnp_negative,
    dpnp_positive,
    dpnp_power,
    dpnp_proj,
    dpnp_real,
    dpnp_remainder,
    dpnp_round,
    dpnp_sign,
    dpnp_signbit,
    dpnp_subtract,
    dpnp_trunc,
)
from .dpnp_utils import *

__all__ = [
    "abs",
    "absolute",
    "add",
    "around",
    "ceil",
    "clip",
    "conj",
    "conjugate",
    "convolve",
    "copysign",
    "cross",
    "cumprod",
    "cumsum",
    "diff",
    "divide",
    "ediff1d",
    "fabs",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gradient",
    "imag",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "negative",
    "positive",
    "power",
    "prod",
    "proj",
    "real",
    "remainder",
    "rint",
    "round",
    "sign",
    "signbit",
    "subtract",
    "sum",
    "trapz",
    "true_divide",
    "trunc",
]


def _append_to_diff_array(a, axis, combined, values):
    """
    Append `values` to `combined` list based on data of array `a`.

    Scalar value (including case with 0d array) is expanded to an array
    with length=1 in the direction of axis and the shape of the input array `a`
    along all other axes.
    Note, if `values` is a scalar, then it is converted to 0d array allocating
    on the same SYCL queue as the input array `a` and with the same USM type.

    """

    dpnp.check_supported_arrays_type(values, scalar_type=True)
    if dpnp.isscalar(values):
        values = dpnp.asarray(
            values, sycl_queue=a.sycl_queue, usm_type=a.usm_type
        )

    if values.ndim == 0:
        shape = list(a.shape)
        shape[axis] = 1
        values = dpnp.broadcast_to(values, tuple(shape))
    combined.append(values)


def absolute(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Calculate the absolute value element-wise.

    For full documentation refer to :obj:`numpy.absolute`.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the absolute value of each element in `x`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `out`, `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.fabs` : Calculate the absolute value element-wise excluding complex types.

    Notes
    -----
    ``dpnp.abs`` is a shorthand for this function.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.2, 1.2])
    >>> np.absolute(a)
    array([1.2, 1.2])

    >>> a = np.array(1.2 + 1j)
    >>> np.absolute(a)
    array(1.5620499351813308)

    """

    return check_nd_call_func(
        numpy.absolute,
        dpnp_abs,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


abs = absolute


def add(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Add arguments element-wise.

    For full documentation refer to :obj:`numpy.add`.

    Returns
    -------
    out : dpnp.ndarray
        The sum of `x1` and `x2`, element-wise.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([1, 2, 3])
    >>> np.add(a, b)
    array([2, 4, 6])

    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
        [  3.,   5.,   7.],
        [  6.,   8.,  10.]])

    The ``+`` operator can be used as a shorthand for ``add`` on
    :class:`dpnp.ndarray`.

    >>> x1 + x2
    array([[  0.,   2.,   4.],
        [  3.,   5.,   7.],
        [  6.,   8.,  10.]])
    """

    return check_nd_call_func(
        numpy.add,
        dpnp_add,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def around(x, /, decimals=0, out=None):
    """
    Round an array to the given number of decimals.

    For full documentation refer to :obj:`numpy.around`.

    Returns
    -------
    out : dpnp.ndarray
        The rounded value of elements of the array.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `decimals` is supported with its default value.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.round` : Equivalent function; see for details.
    :obj:`dpnp.ndarray.round` : Equivalent function.
    :obj:`dpnp.rint` : Round elements of the array to the nearest integer.
    :obj:`dpnp.ceil` : Compute the ceiling of the input, element-wise.
    :obj:`dpnp.floor` : Return the floor of the input, element-wise.
    :obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

    Notes
    -----
    This function works the same as :obj:`dpnp.round`.
    """

    return round(x, decimals, out)


def ceil(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the ceiling of the input, element-wise.

    For full documentation refer to :obj:`numpy.ceil`.

    Returns
    -------
    out : dpnp.ndarray
        The ceiling of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype`, and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by real-value data types.

    See Also
    --------
    :obj:`dpnp.floor` : Return the floor of the input, element-wise.
    :obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.ceil(a)
    array([-1.0, -1.0, -0.0, 1.0, 2.0, 2.0, 2.0])

    """

    return check_nd_call_func(
        numpy.ceil,
        dpnp_ceil,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def clip(a, a_min, a_max, *, out=None, order="K", **kwargs):
    """
    Clip (limit) the values in an array.

    For full documentation refer to :obj:`numpy.clip`.

    Parameters
    ----------
    a : {dpnp_array, usm_ndarray}
        Array containing elements to clip.
    a_min, a_max : {dpnp_array, usm_ndarray, None}
        Minimum and maximum value. If ``None``, clipping is not performed on the corresponding edge.
        Only one of `a_min` and `a_max` may be ``None``. Both are broadcast against `a`.
    out : {dpnp_array, usm_ndarray}, optional
        The results will be placed in this array. It may be the input array for in-place clipping.
        `out` must be of the right shape to hold the output. Its type is preserved.
    order : {"C", "F", "A", "K", None}, optional
        Memory layout of the newly output array, if parameter `out` is `None`.
        If `order` is ``None``, the default value "K" will be used.

    Returns
    -------
    out : dpnp_array
        An array with the elements of `a`, but where values < `a_min` are replaced with `a_min`,
        and those > `a_max` with `a_max`.

    Limitations
    -----------
    Keyword argument `kwargs` is currently unsupported.
    Otherwise ``NotImplementedError`` exception will be raised.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, 1, 8)
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> np.clip(a, 8, 1)
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> np.clip(a, 3, 6, out=a)
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])

    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> min = np.array([3, 4, 1, 1, 1, 4, 4, 4, 4, 4])
    >>> np.clip(a, min, 8)
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])

    """

    if kwargs:
        raise NotImplementedError(f"kwargs={kwargs} is currently not supported")

    if order is None:
        order = "K"

    usm_arr = dpnp.get_usm_ndarray(a)
    usm_min = None if a_min is None else dpnp.get_usm_ndarray_or_scalar(a_min)
    usm_max = None if a_max is None else dpnp.get_usm_ndarray_or_scalar(a_max)

    usm_out = None if out is None else dpnp.get_usm_ndarray(out)
    usm_res = dpt.clip(usm_arr, usm_min, usm_max, out=usm_out, order=order)
    if out is not None and isinstance(out, dpnp_array):
        return out
    return dpnp_array._create_from_usm_ndarray(usm_res)


def conjugate(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Return the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by changing the
    sign of its imaginary part.

    For full documentation refer to :obj:`numpy.conjugate`.

    Returns
    -------
    out : dpnp.ndarray
        The conjugate of each element of `x`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.conjugate(np.array(1+2j))
    (1-2j)

    >>> x = np.eye(2) + 1j * np.eye(2)
    >>> np.conjugate(x)
    array([[ 1.-1.j,  0.-0.j],
           [ 0.-0.j,  1.-1.j]])

    """

    return check_nd_call_func(
        numpy.conjugate,
        dpnp_conj,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


conj = conjugate


def convolve(a, v, mode="full"):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    For full documentation refer to :obj:`numpy.convolve`.

    Examples
    --------
    >>> ca = dpnp.convolve([1, 2, 3], [0, 1, 0.5])
    >>> print(ca)
    [0. , 1. , 2.5, 4. , 1.5]

    """

    return call_origin(numpy.convolve, a=a, v=v, mode=mode)


def copysign(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Change the sign of `x1` to that of `x2`, element-wise.

    For full documentation refer to :obj:`numpy.copysign`.

    Parameters
    ----------
    x1 : {dpnp.ndarray, usm_ndarray}
        First input array, expected to have a real floating-point data type.
    x2 : {dpnp.ndarray, usm_ndarray}
        Second input array, also expected to have a real floating-point data
        type.
    out : ({None, dpnp.ndarray, usm_ndarray}, optional):
        Output array to populate.
        Array must have the correct shape and the expected data type.
    order : ({'C', 'F', 'A', 'K'}, optional):
        Memory layout of the newly output array, if parameter `out` is `None`.
        Default: "K".

    Returns
    -------
    out : dpnp.ndarray
        The values of `x1` with the sign of `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported real data types.

    See Also
    --------
    :obj:`dpnp.negative` : Return the numerical negative of each element of `x`.
    :obj:`dpnp.positive` : Return the numerical positive of each element of `x`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.copysign(np.array(1.3), np.array(-1))
    array(-1.3)
    >>> 1 / np.copysign(np.array(0), 1)
    array(inf)
    >>> 1 / np.copysign(np.array(0), -1)
    array(-inf)

    >>> x = np.array([-1, 0, 1])
    >>> np.copysign(x, -1.1)
    array([-1., -0., -1.])
    >>> np.copysign(x, np.arange(3) - 1)
    array([-1., 0., 1.])

    """

    return check_nd_call_func(
        numpy.copysign,
        dpnp_copysign,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """
    Return the cross product of two (arrays of) vectors.

    For full documentation refer to :obj:`numpy.cross`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as :class:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Sizes of input arrays are limited by `x1.size == 3 and x2.size == 3`.
    Shapes of input arrays are limited by `x1.shape == (3,) and x2.shape == (3,)`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> result = np.cross(x, y)
    >>> [x for x in result]
    [-3,  6, -3]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    x2_desc = dpnp.get_dpnp_descriptor(x2, copy_when_nondefault_queue=False)

    if x1_desc and x2_desc:
        if x1_desc.size != 3 or x2_desc.size != 3:
            pass
        elif x1_desc.shape != (3,) or x2_desc.shape != (3,):
            pass
        elif axisa != -1:
            pass
        elif axisb != -1:
            pass
        elif axisc != -1:
            pass
        elif axis is not None:
            pass
        else:
            return dpnp_cross(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.cross, x1, x2, axisa, axisb, axisc, axis)


def cumprod(x1, **kwargs):
    """
    Return the cumulative product of elements along a given axis.

    For full documentation refer to :obj:`numpy.cumprod`.

    Limitations
    -----------
    Parameter `x` is supported as :class:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> result = np.cumprod(a)
    >>> [x for x in result]
    [1, 2, 6]
    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> result = np.cumprod(b)
    >>> [x for x in result]
    [1, 2, 6, 24, 120, 720]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc and not kwargs:
        return dpnp_cumprod(x1_desc).get_pyobj()

    return call_origin(numpy.cumprod, x1, **kwargs)


def cumsum(x1, **kwargs):
    """
    Return the cumulative sum of the elements along a given axis.

    For full documentation refer to :obj:`numpy.cumsum`.

    Limitations
    -----------
    Parameter `x` is supported as :obj:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.diff` : Calculate the n-th discrete difference along the given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 4])
    >>> result = np.cumsum(a)
    >>> [x for x in result]
    [1, 2, 7]
    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> result = np.cumsum(b)
    >>> [x for x in result]
    [1, 2, 6, 10, 15, 21]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc and not kwargs:
        return dpnp_cumsum(x1_desc).get_pyobj()

    return call_origin(numpy.cumsum, x1, **kwargs)


def diff(a, n=1, axis=-1, prepend=None, append=None):
    """
    Calculate the n-th discrete difference along the given axis.

    For full documentation refer to :obj:`numpy.diff`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend, append : {scalar, dpnp.ndarray, usm_ndarray}, optional
        Values to prepend or append to `a` along axis prior to
        performing the difference. Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes. Otherwise the
        dimension and shape must match `a` except along axis.

    Returns
    -------
    out : dpnp.ndarray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases.

    See Also
    --------
    :obj:`dpnp.gradient` : Return the gradient of an N-dimensional array.
    :obj:`dpnp.ediff1d` : Compute the differences between consecutive elements of an array.
    :obj:`dpnp.cumsum` : Return the cumulative sum of the elements along a given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.diff(x)
    array([ 1,  2,  3, -7])
    >>> np.diff(x, n=2)
    array([  1,   1, -10])

    >>> x = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> np.diff(x)
    array([[2, 3, 4],
           [5, 1, 2]])
    >>> np.diff(x, axis=0)
    array([[-1,  2,  0, -2]])

    """

    dpnp.check_supported_arrays_type(a)
    if n == 0:
        return a
    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")

    nd = a.ndim
    if nd == 0:
        raise ValueError("diff requires input that is at least one dimensional")
    axis = normalize_axis_index(axis, nd)

    combined = []
    if prepend is not None:
        _append_to_diff_array(a, axis, combined, prepend)

    combined.append(a)
    if append is not None:
        _append_to_diff_array(a, axis, combined, append)

    if len(combined) > 1:
        a = dpnp.concatenate(combined, axis=axis)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = dpnp.not_equal if a.dtype == numpy.bool_ else dpnp.subtract
    for _ in range(n):
        a = op(a[slice1], a[slice2])
    return a


def divide(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Divide arguments element-wise.

    For full documentation refer to :obj:`numpy.divide`.

    Returns
    -------
    out : dpnp.ndarray
        The quotient `x1/x2`, element-wise.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Notes
    -----
    Equivalent to `x1` / `x2` in terms of array-broadcasting.

    The ``true_divide(x1, x2)`` function is an alias for
    ``divide(x1, x2)``.

    Examples
    --------
    >>> import dpnp as np
    >>> np.divide(dp.array([1, -2, 6, -9]), np.array([-2, -2, -2, -2]))
    array([-0.5,  1. , -3. ,  4.5])

    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.divide(x1, x2)
    array([[nan, 1. , 1. ],
        [inf, 4. , 2.5],
        [inf, 7. , 4. ]])

    The ``/`` operator can be used as a shorthand for ``divide`` on
    :class:`dpnp.ndarray`.

    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = 2 * np.ones(3)
    >>> x1/x2
    array([[0. , 0.5, 1. ],
        [1.5, 2. , 2.5],
        [3. , 3.5, 4. ]])
    """

    return check_nd_call_func(
        numpy.divide,
        dpnp_divide,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def ediff1d(x1, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    For full documentation refer to :obj:`numpy.ediff1d`.

    Limitations
    -----------
    Parameter `x1`is supported as :class:`dpnp.ndarray`.
    Keyword arguments `to_end` and `to_begin` are currently supported only with default values `None`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.diff` : Calculate the n-th discrete difference along the given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 4, 7, 0])
    >>> result = np.ediff1d(a)
    >>> [x for x in result]
    [1, 2, 3, -7]
    >>> b = np.array([[1, 2, 4], [1, 6, 24]])
    >>> result = np.ediff1d(b)
    >>> [x for x in result]
    [1, 2, -3, 5, 18]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if to_begin is not None:
            pass
        elif to_end is not None:
            pass
        else:
            return dpnp_ediff1d(x1_desc).get_pyobj()

    return call_origin(numpy.ediff1d, x1, to_end=to_end, to_begin=to_begin)


def fabs(x1, **kwargs):
    """
    Compute the absolute values element-wise.

    For full documentation refer to :obj:`numpy.fabs`.

    Limitations
    -----------
    Parameter `x1` is supported as :class:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.absolute` : Calculate the absolute value element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> result = np.fabs(np.array([1, -2, 6, -9]))
    >>> [x for x in result]
    [1.0, 2.0, 6.0, 9.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_fabs(x1_desc).get_pyobj()

    return call_origin(numpy.fabs, x1, **kwargs)


def floor(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Round a number to the nearest integer toward minus infinity.

    For full documentation refer to :obj:`numpy.floor`.

    Returns
    -------
    out : dpnp.ndarray
        The floor of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype`, and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by real-value data types.

    See Also
    --------
    :obj:`dpnp.ceil` : Compute the ceiling of the input, element-wise.
    :obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

    Notes
    -----
    Some spreadsheet programs calculate the "floor-towards-zero", in other words floor(-2.5) == -2.
    DPNP instead uses the definition of floor where floor(-2.5) == -3.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.floor(a)
    array([-2.0, -2.0, -1.0, 0.0, 1.0, 1.0, 2.0])

    """

    return check_nd_call_func(
        numpy.floor,
        dpnp_floor,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def floor_divide(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the largest integer smaller or equal to the division of the inputs.

    For full documentation refer to :obj:`numpy.floor_divide`.

    Returns
    -------
    out : dpnp.ndarray
        The floordivide of each element of `x`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.remainder` : Remainder complementary to floor_divide.
    :obj:`dpnp.divide` : Standard division.
    :obj:`dpnp.floor` : Round a number to the nearest integer toward minus infinity.
    :obj:`dpnp.ceil` : Round a number to the nearest integer toward infinity.

    Examples
    --------
    >>> import dpnp as np
    >>> np.floor_divide(np.array([1, -1, -2, -9]), -2)
    array([-1,  0,  1,  4])

    >>> np.floor_divide(np.array([1., 2., 3., 4.]), 2.5)
    array([ 0.,  0.,  1.,  1.])

    The ``//`` operator can be used as a shorthand for ``floor_divide`` on
    :class:`dpnp.ndarray`.

    >>> x1 = np.array([1., 2., 3., 4.])
    >>> x1 // 2.5
    array([0., 0., 1., 1.])

    """

    return check_nd_call_func(
        numpy.floor_divide,
        dpnp_floor_divide,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def fmax(x1, x2, /, out=None, *, where=True, dtype=None, subok=True, **kwargs):
    """
    Element-wise maximum of array elements.

    For full documentation refer to :obj:`numpy.fmax`.

    Returns
    -------
    out : dpnp.ndarray
        The maximum of `x1` and `x2`, element-wise, ignoring NaNs.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by real-valued data types.

    See Also
    --------
    :obj:`dpnp.maximum` : Element-wise maximum of array elements, propagates NaNs.
    :obj:`dpnp.fmin` : Element-wise minimum of array elements, ignore NaNs.
    :obj:`dpnp.minimum` : Element-wise minimum of array elements, propagates NaNs.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([2, 3, 4])
    >>> x2 = np.array([1, 5, 2])
    >>> np.fmax(x1, x2)
    array([2, 5, 4])

    >>> x1 = np.eye(2)
    >>> x2 = np.array([0.5, 2])
    >>> np.fmax(x1, x2) # broadcasting
    array([[1. , 2. ],
           [0.5, 2. ]])

    >>> x1 = np.array([np.nan, 0, np.nan])
    >>> x2 = np.array([0, np.nan, np.nan])
    >>> np.fmax(x1, x2)
    array([ 0.,  0., nan])

    """

    if kwargs:
        pass
    elif where is not True:
        pass
    elif dtype is not None:
        pass
    elif subok is not True:
        pass
    elif dpnp.isscalar(x1) and dpnp.isscalar(x2):
        # at least either x1 or x2 has to be an array
        pass
    else:
        # get USM type and queue to copy scalar from the host memory into a USM allocation
        usm_type, queue = (
            get_usm_allocations([x1, x2])
            if dpnp.isscalar(x1) or dpnp.isscalar(x2)
            else (None, None)
        )

        x1_desc = dpnp.get_dpnp_descriptor(
            x1,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        x2_desc = dpnp.get_dpnp_descriptor(
            x2,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        if x1_desc and x2_desc:
            if out is not None:
                if not dpnp.is_supported_array_type(out):
                    raise TypeError(
                        "return array must be of supported array type"
                    )
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out, copy_when_nondefault_queue=False
                    )
                    or None
                )
            else:
                out_desc = None

            return dpnp_fmax(
                x1_desc, x2_desc, dtype=dtype, out=out_desc, where=where
            ).get_pyobj()

    return call_origin(
        numpy.fmax, x1, x2, dtype=dtype, out=out, where=where, **kwargs
    )


def fmin(x1, x2, /, out=None, *, where=True, dtype=None, subok=True, **kwargs):
    """
    Element-wise minimum of array elements.

    For full documentation refer to :obj:`numpy.fmin`.

    Returns
    -------
    out : dpnp.ndarray
        The minimum of `x1` and `x2`, element-wise, ignoring NaNs.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by real-valued data types.

    See Also
    --------
    :obj:`dpnp.minimum` : Element-wise minimum of array elements, propagates NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of array elements, ignore NaNs.
    :obj:`dpnp.maximum` : Element-wise maximum of array elements, propagates NaNs.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([2, 3, 4])
    >>> x2 = np.array([1, 5, 2])
    >>> np.fmin(x1, x2)
    array([1, 3, 2])

    >>> x1 = np.eye(2)
    >>> x2 = np.array([0.5, 2])
    >>> np.fmin(x1, x2) # broadcasting
    array([[0.5, 0. ],
           [0. , 1. ]]

    >>> x1 = np.array([np.nan, 0, np.nan])
    >>> x2 = np.array([0, np.nan, np.nan])
    >>> np.fmin(x1, x2)
    array([ 0.,  0., nan])

    """

    if kwargs:
        pass
    elif where is not True:
        pass
    elif dtype is not None:
        pass
    elif subok is not True:
        pass
    elif dpnp.isscalar(x1) and dpnp.isscalar(x2):
        # at least either x1 or x2 has to be an array
        pass
    else:
        # get USM type and queue to copy scalar from the host memory into a USM allocation
        usm_type, queue = (
            get_usm_allocations([x1, x2])
            if dpnp.isscalar(x1) or dpnp.isscalar(x2)
            else (None, None)
        )

        x1_desc = dpnp.get_dpnp_descriptor(
            x1,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        x2_desc = dpnp.get_dpnp_descriptor(
            x2,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        if x1_desc and x2_desc:
            if out is not None:
                if not dpnp.is_supported_array_type(out):
                    raise TypeError(
                        "return array must be of supported array type"
                    )
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out, copy_when_nondefault_queue=False
                    )
                    or None
                )
            else:
                out_desc = None

            return dpnp_fmin(
                x1_desc, x2_desc, dtype=dtype, out=out_desc, where=where
            ).get_pyobj()

    return call_origin(
        numpy.fmin, x1, x2, dtype=dtype, out=out, where=where, **kwargs
    )


def fmod(x1, x2, /, out=None, *, where=True, dtype=None, subok=True, **kwargs):
    """
    Returns the element-wise remainder of division.

    For full documentation refer to :obj:`numpy.fmod`.

    Returns
    -------
    out : dpnp.ndarray
        The remainder of the division of `x1` by `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.remainder` : Remainder complementary to floor_divide.
    :obj:`dpnp.divide` : Standard division.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-3, -2, -1, 1, 2, 3])
    >>> np.fmod(a, 2)
    array([-1,  0, -1,  1,  0,  1])
    >>> np.remainder(a, 2)
    array([1, 0, 1, 1, 0, 1])

    >>> a = np.array([5, 3])
    >>> b = np.array([2, 2.])
    >>> np.fmod(a, b)
    array([1., 1.])

    >>> a = np.arange(-3, 3).reshape(3, 2)
    >>> a
    array([[-3, -2],
           [-1,  0],
           [ 1,  2]])
    >>> b = np.array([2, 2])
    >>> np.fmod(a, b)
    array([[-1,  0],
           [-1,  0],
           [ 1,  0]])

    """

    if kwargs:
        pass
    elif where is not True:
        pass
    elif dtype is not None:
        pass
    elif subok is not True:
        pass
    elif dpnp.isscalar(x1) and dpnp.isscalar(x2):
        # at least either x1 or x2 has to be an array
        pass
    else:
        # get USM type and queue to copy scalar from the host memory into a USM allocation
        usm_type, queue = (
            get_usm_allocations([x1, x2])
            if dpnp.isscalar(x1) or dpnp.isscalar(x2)
            else (None, None)
        )

        x1_desc = dpnp.get_dpnp_descriptor(
            x1,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        x2_desc = dpnp.get_dpnp_descriptor(
            x2,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        if x1_desc and x2_desc:
            if out is not None:
                if not dpnp.is_supported_array_type(out):
                    raise TypeError(
                        "return array must be of supported array type"
                    )
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out, copy_when_nondefault_queue=False
                    )
                    or None
                )
            else:
                out_desc = None

            return dpnp_fmod(
                x1_desc, x2_desc, dtype=dtype, out=out_desc, where=where
            ).get_pyobj()

    return call_origin(
        numpy.fmod, x1, x2, dtype=dtype, out=out, where=where, **kwargs
    )


def gradient(x1, *varargs, **kwargs):
    """
    Return the gradient of an array.

    For full documentation refer to :obj:`numpy.gradient`.

    Limitations
    -----------
    Parameter `y1` is supported as :class:`dpnp.ndarray`.
    Argument `varargs[0]` is supported as `int`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.diff` : Calculate the n-th discrete difference along the given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> y = np.array([1, 2, 4, 7, 11, 16], dtype=float)
    >>> result = np.gradient(y)
    >>> [x for x in result]
    [1.0, 1.5, 2.5, 3.5, 4.5, 5.0]
    >>> result = np.gradient(y, 2)
    >>> [x for x in result]
    [0.5, 0.75, 1.25, 1.75, 2.25, 2.5]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc and not kwargs:
        if len(varargs) > 1:
            pass
        elif len(varargs) == 1 and not isinstance(varargs[0], int):
            pass
        else:
            if len(varargs) == 0:
                return dpnp_gradient(x1_desc).get_pyobj()

            return dpnp_gradient(x1_desc, varargs[0]).get_pyobj()

    return call_origin(numpy.gradient, x1, *varargs, **kwargs)


def imag(x):
    """
    Return the imaginary part of the complex argument.

    For full documentation refer to :obj:`numpy.imag`.

    Returns
    -------
    out : dpnp.ndarray
        The imaginary component of the complex argument.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.real` : Return the real part of the complex argument.
    :obj:`dpnp.conj` : Return the complex conjugate, element-wise.
    :obj:`dpnp.conjugate` : Return the complex conjugate, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1+2j, 3+4j, 5+6j])
    >>> a.imag
    array([2., 4., 6.])

    >>> a.imag = np.array([8, 10, 12])
    >>> a
    array([1. +8.j, 3.+10.j, 5.+12.j])

    >>> np.imag(np.array(1 + 1j))
    array(1.)

    """

    if dpnp.isscalar(x):
        # input has to be an array
        pass
    else:
        return dpnp_imag(x)
    return call_origin(numpy.imag, x)


def maximum(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Element-wise maximum of array elements.

    For full documentation refer to :obj:`numpy.maximum`.

    Returns
    -------
    out : dpnp.ndarray
        The maximum of `x1` and `x2`, element-wise, propagating NaNs.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.minimum` : Element-wise minimum of two arrays, propagates NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignores NaNs.
    :obj:`dpnp.max` : The maximum value of an array along a given axis, propagates NaNs.
    :obj:`dpnp.nanmax` : The maximum value of an array along a given axis, ignores NaNs.
    :obj:`dpnp.fmin` : Element-wise minimum of two arrays, ignores NaNs.
    :obj:`dpnp.min` : The minimum value of an array along a given axis, propagates NaNs.
    :obj:`dpnp.nanmin` : The minimum value of an array along a given axis, ignores NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([2, 3, 4])
    >>> x2 = np.array([1, 5, 2])
    >>> np.maximum(x1, x2)
    array([2, 5, 4])

    >>> x1 = np.eye(2)
    >>> x2 = np.array([0.5, 2])
    >>> np.maximum(x1, x2) # broadcasting
    array([[1. , 2. ],
           [0.5, 2. ]])

    >>> x1 = np.array([np.nan, 0, np.nan])
    >>> x2 = np.array([0, np.nan, np.nan])
    >>> np.maximum(x1, x2)
    array([nan, nan, nan])

    >>> np.maximum(np.array(np.Inf), 1)
    array(inf)

    """

    return check_nd_call_func(
        numpy.maximum,
        dpnp_maximum,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def minimum(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Element-wise minimum of array elements.

    For full documentation refer to :obj:`numpy.minimum`.

    Returns
    -------
    out : dpnp.ndarray
        The minimum of `x1` and `x2`, element-wise, propagating NaNs.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.maximum` : Element-wise maximum of two arrays, propagates NaNs.
    :obj:`dpnp.fmin` : Element-wise minimum of two arrays, ignores NaNs.
    :obj:`dpnp.min` : The minimum value of an array along a given axis, propagates NaNs.
    :obj:`dpnp.nanmin` : The minimum value of an array along a given axis, ignores NaNs.
    :obj:`dpnp.fmax` : Element-wise maximum of two arrays, ignores NaNs.
    :obj:`dpnp.max` : The maximum value of an array along a given axis, propagates NaNs.
    :obj:`dpnp.nanmax` : The maximum value of an array along a given axis, ignores NaNs.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([2, 3, 4])
    >>> x2 = np.array([1, 5, 2])
    >>> np.minimum(x1, x2)
    array([1, 3, 2])

    >>> x1 = np.eye(2)
    >>> x2 = np.array([0.5, 2])
    >>> np.minimum(x1, x2) # broadcasting
    array([[0.5, 0. ],
           [0. , 1. ]]

    >>> x1 = np.array([np.nan, 0, np.nan])
    >>> x2 = np.array([0, np.nan, np.nan])
    >>> np.minimum(x1, x2)
    array([nan, nan, nan])

    >>> np.minimum(np.array(-np.Inf), 1)
    array(-inf)

    """

    return check_nd_call_func(
        numpy.minimum,
        dpnp_minimum,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def mod(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute element-wise remainder of division.

    For full documentation refer to :obj:`numpy.mod`.

    Returns
    -------
    out : dpnp.ndarray
        The element-wise remainder of the quotient `floor_divide(x1, x2)`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division
    :obj:`dpnp.remainder` : Remainder complementary to floor_divide.
    :obj:`dpnp.divide` : Standard division.

    Notes
    -----
    This function works the same as :obj:`dpnp.remainder`.

    """

    return dpnp.remainder(
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def modf(x1, **kwargs):
    """
    Return the fractional and integral parts of an array, element-wise.

    For full documentation refer to :obj:`numpy.modf`.

    Limitations
    -----------
    Parameter `x` is supported as :obj:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2])
    >>> result = np.modf(a)
    >>> [[x for x in y] for y in result ]
    [[1.0, 2.0], [0.0, 0.0]]


    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc and not kwargs:
        return dpnp_modf(x1_desc)

    return call_origin(numpy.modf, x1, **kwargs)


def multiply(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Multiply arguments element-wise.

    For full documentation refer to :obj:`numpy.multiply`.

    Returns
    -------
    out : {dpnp.ndarray, scalar}
        The product of `x1` and `x2`, element-wise.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Notes
    -----
    Equivalent to `x1` * `x2` in terms of array broadcasting.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> np.multiply(a, a)
    array([ 1,  4,  9, 16, 25])]

    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.multiply(x1, x2)
    array([[  0.,   1.,   4.],
        [  0.,   4.,  10.],
        [  0.,   7.,  16.]])

    The ``*`` operator can be used as a shorthand for ``multiply`` on
    :class:`dpnp.ndarray`.

    >>> x1 * x2
    array([[  0.,   1.,   4.],
        [  0.,   4.,  10.],
        [  0.,   7.,  16.]])
    """

    return check_nd_call_func(
        numpy.multiply,
        dpnp_multiply,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def negative(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Numerical negative, element-wise.

    For full documentation refer to :obj:`numpy.negative`.

    Returns
    -------
    out : dpnp.ndarray
        The numerical negative of each element of `x`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.positive` : Return the numerical positive of each element of `x`.
    :obj:`dpnp.copysign` : Change the sign of `x1` to that of `x2`, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> np.negative(np.array([1, -1]))
    array([-1, 1])

    The ``-`` operator can be used as a shorthand for ``negative`` on
    :class:`dpnp.ndarray`.

    >>> x = np.array([1., -1.])
    >>> -x
    array([-1.,  1.])
    """

    return check_nd_call_func(
        numpy.negative,
        dpnp_negative,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def positive(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Numerical positive, element-wise.

    For full documentation refer to :obj:`numpy.positive`.

    Returns
    -------
    out : dpnp.ndarray
        The numerical positive of each element of `x`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.negative` : Return the numerical negative of each element of `x`.
    :obj:`dpnp.copysign` : Change the sign of `x1` to that of `x2`, element-wise.

    Note
    ----
    Equivalent to `x.copy()`, but only defined for types that support arithmetic.

    Examples
    --------
    >>> import dpnp as np
    >>> np.positive(np.array([1., -1.]))
    array([ 1., -1.])

    The ``+`` operator can be used as a shorthand for ``positive`` on
    :class:`dpnp.ndarray`.

    >>> x = np.array([1., -1.])
    >>> +x
    array([ 1., -1.])
    """

    return check_nd_call_func(
        numpy.positive,
        dpnp_positive,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def power(
    x1,
    x2,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    First array elements raised to powers from second array, element-wise.

    An integer type (of either negative or positive value, but not zero)
    raised to a negative integer power will return an array of zeroes.

    For full documentation refer to :obj:`numpy.power`.

    Returns
    -------
    out : dpnp.ndarray
        The bases in `x1` raised to the exponents in `x2`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.fmax` : Element-wise maximum of array elements.
    :obj:`dpnp.fmin` : Element-wise minimum of array elements.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.


    Examples
    --------
    >>> import dpnp as dp
    >>> a = dp.arange(6)
    >>> dp.power(a, 3)
    array([  0,   1,   8,  27,  64, 125])

    Raise the bases to different exponents.

    >>> b = dp.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
    >>> dp.power(a, b)
    array([ 0.,  1.,  8., 27., 16.,  5.])

    The effect of broadcasting.

    >>> c = dp.array([[1, 2, 3, 3, 2, 1], [1, 2, 3, 3, 2, 1]])
    >>> dp.power(a, c)
    array([[ 0,  1,  8, 27, 16,  5],
           [ 0,  1,  8, 27, 16,  5]])

    The ``**`` operator can be used as a shorthand for ``power`` on
    :class:`dpnp.ndarray`.

    >>> b = dp.array([1, 2, 3, 3, 2, 1])
    >>> a = dp.arange(6)
    >>> a ** b
    array([ 0,  1,  8, 27, 16,  5])

    Negative values raised to a non-integral value will result in ``nan``.

    >>> d = dp.array([-1.0, -4.0])
    >>> dp.power(d, 1.5)
    array([nan, nan])

    """

    return check_nd_call_func(
        numpy.power,
        dpnp_power,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def prod(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    """
    Return the product of array elements over a given axis.

    For full documentation refer to :obj:`numpy.prod`.

    Returns
    -------
    out : dpnp.ndarray
        A new array holding the result is returned unless `out` is specified, in which case it is returned.

    Limitations
    -----------
    Input array is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `initial`, and `where` are only supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.nanprod` : Return the product of array elements over a given axis treating Not a Numbers (NaNs) as ones.

    Examples
    --------
    >>> import dpnp as np
    >>> np.prod(np.array([1, 2]))
    array(2)

    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.prod(a)
    array(24)

    >>> np.prod(a, axis=1)
    array([ 2, 12])
    >>> np.prod(a, axis=0)
    array([3, 8])

    >>> x = np.array([1, 2, 3], dtype=np.int8)
    >>> np.prod(x).dtype == int
    True

    """

    # Product reduction for complex output are known to fail for Gen9 with 2024.0 compiler
    # TODO: get rid of this temporary work around when OneAPI 2024.1 is released
    dpnp.check_supported_arrays_type(a)
    _dtypes = (a.dtype, dtype)
    _any_complex = any(
        dpnp.issubdtype(dt, dpnp.complexfloating) for dt in _dtypes
    )
    device_mask = (
        du.intel_device_info(a.sycl_device).get("device_id", 0) & 0xFF00
    )
    if _any_complex and device_mask in [0x3E00, 0x9B00]:
        return call_origin(
            numpy.prod,
            a,
            axis=axis,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
        )
    elif initial is not None:
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
            dpt.prod(dpt_array, axis=axis, dtype=dtype, keepdims=keepdims)
        )

        return dpnp.get_result_array(result, out)


def proj(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Returns the projection of a number onto the Riemann sphere.

    For all infinite complex numbers (including the cases where one component is infinite and the other is `NaN`),
    the function returns `(inf, 0.0)` or `(inf, -0.0)`.
    For finite complex numbers, the input is returned.
    All real-valued numbers are treated as complex numbers with positive zero imaginary part.

    Returns
    -------
    out : dpnp.ndarray
        The projection of each element of `x`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.absolute` : Returns the magnitude of a complex number, element-wise.
    :obj:`dpnp.conj` : Return the complex conjugate, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> np.proj(np.array([1, -2.3, 2.1-1.7j]))
    array([ 1. +0.j, -2.3+0.j,  2.1-1.7.j])

    >>> np.proj(np.array([complex(1,np.inf), complex(1,-np.inf), complex(np.inf,-1),]))
    array([inf+0.j, inf-0.j, inf-0.j])

    """

    return check_nd_call_func(
        None,
        dpnp_proj,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def real(x):
    """
    Return the real part of the complex argument.

    For full documentation refer to :obj:`numpy.real`.

    Returns
    -------
    out : dpnp.ndarray
        The real component of the complex argument.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.imag` : Return the imaginary part of the complex argument.
    :obj:`dpnp.conj` : Return the complex conjugate, element-wise.
    :obj:`dpnp.conjugate` : Return the complex conjugate, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1+2j, 3+4j, 5+6j])
    >>> a.real
    array([1., 3., 5.])

    >>> a.real = 9
    >>> a
    array([9.+2.j, 9.+4.j, 9.+6.j])

    >>> a.real = np.array([9, 8, 7])
    >>> a
    array([9.+2.j, 8.+4.j, 7.+6.j])

    >>> np.real(np.array(1 + 1j))
    array(1.)

    """
    if dpnp.isscalar(x):
        # input has to be an array
        pass
    else:
        if dpnp.issubsctype(x.dtype, dpnp.complexfloating):
            return dpnp_real(x)
        else:
            return x
    return call_origin(numpy.real, x)


def remainder(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Return element-wise remainder of division.

    For full documentation refer to :obj:`numpy.remainder`.

    Returns
    -------
    out : dpnp.ndarray
        The element-wise remainder of the quotient `floor_divide(x1, x2)`.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.
    :obj:`dpnp.divide` : Standard division.
    :obj:`dpnp.floor` : Round a number to the nearest integer toward minus infinity.
    :obj:`dpnp.floor_divide` : Compute the largest integer smaller or equal to the division of the inputs.
    :obj:`dpnp.mod` : Calculate the element-wise remainder of division.

    Examples
    --------
    >>> import dpnp as np
    >>> np.remainder(np.array([4, 7]), np.array([2, 3]))
    array([0, 1])

    >>> np.remainder(np.arange(7), 5)
    array([0, 1, 2, 3, 4, 0, 1])

    The ``%`` operator can be used as a shorthand for ``remainder`` on
    :class:`dpnp.ndarray`.

    >>> x1 = np.arange(7)
    >>> x1 % 5
    array([0, 1, 2, 3, 4, 0, 1])
    """

    return check_nd_call_func(
        numpy.remainder,
        dpnp_remainder,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def rint(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Round elements of the array to the nearest integer.

    For full documentation refer to :obj:`numpy.rint`.

    Returns
    -------
    out : dpnp.ndarray
        The rounded value of elements of the array to the nearest integer.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.round` : Evenly round to the given number of decimals.
    :obj:`dpnp.ceil` : Compute the ceiling of the input, element-wise.
    :obj:`dpnp.floor` : Return the floor of the input, element-wise.
    :obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.rint(a)
    array([-2., -2., -0.,  0.,  2.,  2.,  2.])

    """

    return check_nd_call_func(
        numpy.rint,
        dpnp_round,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def round(x, decimals=0, out=None):
    """
    Evenly round to the given number of decimals.

    For full documentation refer to :obj:`numpy.round`.

    Returns
    -------
    out : dpnp.ndarray
        The rounded value of elements of the array.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `decimals` is supported with its default value.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.around` : Equivalent function; see for details.
    :obj:`dpnp.ndarray.round` : Equivalent function.
    :obj:`dpnp.rint` : Round elements of the array to the nearest integer.
    :obj:`dpnp.ceil` : Compute the ceiling of the input, element-wise.
    :obj:`dpnp.floor` : Return the floor of the input, element-wise.
    :obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> np.round(np.array([0.37, 1.64]))
    array([0.,  2.])
    >>> np.round(np.array([0.37, 1.64]), decimals=1)
    array([0.4,  1.6])
    >>> np.round(np.array([.5, 1.5, 2.5, 3.5, 4.5])) # rounds to nearest even value
    array([0.,  2.,  2.,  4.,  4.])
    >>> np.round(np.array([1,2,3,11]), decimals=1) # ndarray of ints is returned
    array([ 1,  2,  3, 11])
    >>> np.round(np.array([1,2,3,11]), decimals=-1)
    array([ 0,  0,  0, 10])

    """

    if decimals != 0:
        pass
    elif dpnp.isscalar(x):
        # input has to be an array
        pass
    else:
        return dpnp_round(x, out=out)
    return call_origin(numpy.round, x, decimals=decimals, out=out)


def sign(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Returns an element-wise indication of the sign of a number.

    For full documentation refer to :obj:`numpy.sign`.

    Returns
    -------
    out : dpnp.ndarray
        The indication of the sign of each element of `x`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    However, if the input array data type is complex, the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.signbit` : Returns element-wise `True` where signbit is set (less than zero).

    Examples
    --------
    >>> import dpnp as np
    >>> np.sign(np.array([-5., 4.5]))
    array([-1.0, 1.0])
    >>> np.sign(np.array(0))
    array(0)
    >>> np.sign(np.array(5-2j))
    array([1+0j])

    """

    if numpy.iscomplexobj(x):
        return call_origin(
            numpy.sign,
            x,
            out=out,
            where=where,
            order=order,
            dtype=dtype,
            subok=subok,
            **kwargs,
        )
    else:
        return check_nd_call_func(
            numpy.sign,
            dpnp_sign,
            x,
            out=out,
            where=where,
            order=order,
            dtype=dtype,
            subok=subok,
            **kwargs,
        )


def signbit(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Returns element-wise `True` where signbit is set (less than zero).

    For full documentation refer to :obj:`numpy.signbit`.

    Returns
    -------
    out : dpnp.ndarray
        A boolean array with indication of the sign of each element of `x`.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported real-valued data types.

    See Also
    --------
    :obj:`dpnp.sign` : Returns an element-wise indication of the sign of a number.

    Examples
    --------
    >>> import dpnp as np
    >>> np.signbit(np.array([-1.2]))
    array([True])

    >>> np.signbit(np.array([1, -2.3, 2.1]))
    array([False,  True, False])

    """

    return check_nd_call_func(
        numpy.signbit,
        dpnp_signbit,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def subtract(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    order="K",
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Subtract arguments, element-wise.

    For full documentation refer to :obj:`numpy.subtract`.

    Returns
    -------
    out : dpnp.ndarray
        The difference of `x1` and `x2`, element-wise.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Notes
    -----
    Equivalent to `x1` - `x2` in terms of array broadcasting.

    Examples
    --------
    >>> import dpnp as np
    >>> np.subtract(dp.array([4, 3]), np.array([2, 7]))
    array([ 2, -4])

    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.subtract(x1, x2)
    array([[ 0.,  0.,  0.],
        [ 3.,  3.,  3.],
        [ 6.,  6.,  6.]])

    The ``-`` operator can be used as a shorthand for ``subtract`` on
    :class:`dpnp.ndarray`.

    >>> x1 - x2
    array([[ 0.,  0.,  0.],
        [ 3.,  3.,  3.],
        [ 6.,  6.,  6.]])
    """

    return check_nd_call_func(
        numpy.subtract,
        dpnp_subtract,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def sum(
    x,
    /,
    *,
    axis=None,
    dtype=None,
    keepdims=False,
    out=None,
    initial=0,
    where=True,
):
    """
    Sum of array elements over a given axis.

    For full documentation refer to :obj:`numpy.sum`.

    Returns
    -------
    out : dpnp.ndarray
        an array containing the sums. If the sum was computed over the
        entire array, a zero-dimensional array is returned. The returned
        array has the data type as described in the `dtype` parameter
        of the Python Array API standard for the `sum` function.

    Limitations
    -----------
    Parameters `x` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `initial` and `where` are supported with their default values.
    Otherwise ``NotImplementedError`` exception will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.sum(np.array([1, 2, 3, 4, 5]))
    array(15)
    >>> np.sum(np.array(5))
    array(5)
    >>> result = np.sum(np.array([[0, 1], [0, 5]]), axis=0)
    array([0, 6])

    """

    if axis is not None:
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)

        axis = normalize_axis_tuple(axis, x.ndim, "axis")

    if initial != 0:
        raise NotImplementedError(
            "initial keyword argument is only supported with its default value."
        )
    elif where is not True:
        raise NotImplementedError(
            "where keyword argument is only supported with its default value."
        )
    else:
        if (
            len(x.shape) == 2
            and x.itemsize == 4
            and (
                (
                    axis == (0,)
                    and x.flags.c_contiguous
                    and 32 <= x.shape[1] <= 1024
                    and x.shape[0] > x.shape[1]
                )
                or (
                    axis == (1,)
                    and x.flags.f_contiguous
                    and 32 <= x.shape[0] <= 1024
                    and x.shape[1] > x.shape[0]
                )
            )
        ):
            from dpctl.tensor._reduction import _default_reduction_dtype

            from dpnp.backend.extensions.sycl_ext import _sycl_ext_impl

            input = x
            if axis == (1,):
                input = input.T
            input = dpnp.get_usm_ndarray(input)

            queue = input.sycl_queue
            out_dtype = (
                _default_reduction_dtype(input.dtype, queue)
                if dtype is None
                else dtype
            )
            output = dpt.empty(
                input.shape[1], dtype=out_dtype, sycl_queue=queue
            )

            get_sum = _sycl_ext_impl._get_sum_over_axis_0
            sum = get_sum(input, output)

            if sum:
                sum(input, output, []).wait()
                result = dpnp_array._create_from_usm_ndarray(output)

                if keepdims:
                    if axis == (0,):
                        res_sh = (1,) + output.shape
                    else:
                        res_sh = output.shape + (1,)
                    result = result.reshape(res_sh)

                return result

        y = dpt.sum(
            dpnp.get_usm_ndarray(x), axis=axis, dtype=dtype, keepdims=keepdims
        )
        result = dpnp_array._create_from_usm_ndarray(y)
        return dpnp.get_result_array(result, out, casting="same_kind")


def trapz(y1, x1=None, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    For full documentation refer to :obj:`numpy.trapz`.

    Limitations
    -----------
    Parameters `y` and `x` are supported as :class:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 6, 8])
    >>> np.trapz(a)
    4.0
    >>> np.trapz(a, x=b)
    8.0
    >>> np.trapz(a, dx=2)
    8.0

    """

    y_desc = dpnp.get_dpnp_descriptor(y1, copy_when_nondefault_queue=False)
    if y_desc:
        if y_desc.ndim > 1:
            pass
        else:
            y_obj = y_desc.get_array()
            if x1 is None:
                x_obj = dpnp.empty(
                    y_desc.shape,
                    dtype=y_desc.dtype,
                    device=y_obj.sycl_device,
                    usm_type=y_obj.usm_type,
                    sycl_queue=y_obj.sycl_queue,
                )
            else:
                x_obj = x1

            x_desc = dpnp.get_dpnp_descriptor(
                x_obj, copy_when_nondefault_queue=False
            )
            # TODO: change to "not x_desc"
            if x_desc:
                pass
            elif y_desc.size != x_desc.size:
                pass
            elif y_desc.shape != x_desc.shape:
                pass
            else:
                return dpnp_trapz(y_desc, x_desc, dx).get_pyobj()

    return call_origin(numpy.trapz, y1, x1, dx, axis)


true_divide = divide


def trunc(
    x,
    /,
    out=None,
    *,
    order="K",
    where=True,
    dtype=None,
    subok=True,
    **kwargs,
):
    """
    Compute the truncated value of the input, element-wise.

    For full documentation refer to :obj:`numpy.trunc`.

    Returns
    -------
    out : dpnp.ndarray
        The truncated value of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype`, and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by real-value data types.

    See Also
    --------
    :obj:`dpnp.floor` : Round a number to the nearest integer toward minus infinity.
    :obj:`dpnp.ceil` : Round a number to the nearest integer toward infinity.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> np.trunc(a)
    array([-1.0, -1.0, -0.0, 0.0, 1.0, 1.0, 2.0])

    """

    return check_nd_call_func(
        numpy.trunc,
        dpnp_trunc,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )
