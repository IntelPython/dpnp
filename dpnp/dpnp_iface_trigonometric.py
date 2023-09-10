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
Interface of the Trigonometric part of the DPNP

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
from dpnp.dpnp_utils import *

from .dpnp_algo.dpnp_elementwise_common import (
    check_nd_call_func,
    dpnp_acos,
    dpnp_acosh,
    dpnp_asin,
    dpnp_asinh,
    dpnp_atan,
    dpnp_atan2,
    dpnp_atanh,
    dpnp_cos,
    dpnp_cosh,
    dpnp_log,
    dpnp_sin,
    dpnp_sinh,
    dpnp_sqrt,
    dpnp_square,
    dpnp_tan,
    dpnp_tanh,
)

__all__ = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "cbrt",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp",
    "exp2",
    "expm1",
    "hypot",
    "log",
    "log10",
    "log1p",
    "log2",
    "rad2deg",
    "radians",
    "reciprocal",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "unwrap",
]


def arccos(
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
    Trigonometric inverse cosine, element-wise.

    For full documentation refer to :obj:`numpy.arccos`.

    Returns
    -------
    out : dpnp.ndarray
        The inverse cosine of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.cos` : Trigonometric cosine, element-wise.
    :obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
    :obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.
    :obj:`dpnp.arccosh` : Hyperbolic inverse cosine, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, -1])
    >>> np.arccos(x)
    array([0.0,  3.14159265])

    """

    return check_nd_call_func(
        numpy.arccos,
        dpnp_acos,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def arccosh(
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
    Inverse hyperbolic cosine, element-wise.

    For full documentation refer to :obj:`numpy.arccosh`.

    Returns
    -------
    out : dpnp.ndarray
        The hyperbolic inverse cosine of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.
    :obj:`dpnp.arctanh` : Hyperbolic inverse tangent, element-wise.
    :obj:`dpnp.arcsinh` : Hyperbolic inverse sine, element-wise.
    :obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1.0, np.e, 10.0])
    >>> np.arccosh(x)
    array([0.0, 1.65745445, 2.99322285])

    """

    return check_nd_call_func(
        numpy.arccosh,
        dpnp_acosh,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def arcsin(
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
    Inverse sine, element-wise.

    For full documentation refer to :obj:`numpy.arcsin`.

    Returns
    -------
    out : dpnp.ndarray
        The inverse sine of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.sin` : Trigonometric sine, element-wise.
    :obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
    :obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.
    :obj:`dpnp.arcsinh` : Hyperbolic inverse sine, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, 1, -1])
    >>> np.arcsin(x)
    array([0.0, 1.5707963267948966, -1.5707963267948966])

    """

    return check_nd_call_func(
        numpy.arcsin,
        dpnp_asin,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def arcsinh(
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
    Inverse hyperbolic sine, element-wise.

    For full documentation refer to :obj:`numpy.arcsinh`.

    Returns
    -------
    out : dpnp.ndarray
        The hyperbolic inverse sine of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
    :obj:`dpnp.arctanh` : Hyperbolic inverse tangent, element-wise.
    :obj:`dpnp.arccosh` : Hyperbolic inverse cosine, element-wise.
    :obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([np.e, 10.0])
    >>> np.arcsinh(x)
    array([1.72538256, 2.99822295])

    """

    return check_nd_call_func(
        numpy.arcsinh,
        dpnp_asinh,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def arctan(
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
    Trigonometric inverse tangent, element-wise.

    For full documentation refer to :obj:`numpy.arctan`.

    Returns
    -------
    out : dpnp.ndarray
        The inverse tangent of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported real-valued floating-point data type.

    See Also
    --------
    :obj:`dpnp.arctan2` : Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.
    :obj:`dpnp.angle` : Argument of complex values.
    :obj:`dpnp.tan` : Trigonometric tangent, element-wise.
    :obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.
    :obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.
    :obj:`dpnp.arctanh` : Inverse hyperbolic tangent, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, 1])
    >>> np.arctan(x)
    array([0.0, 0.78539816])

    """

    return check_nd_call_func(
        numpy.arctan,
        dpnp_atan,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def arctan2(
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
    Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

    For full documentation refer to :obj:`numpy.arctan2`.

    Returns
    -------
    out : dpnp.ndarray
        The inverse tangent of `x1/x2`, element-wise.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword arguments `kwargs` are currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
    :obj:`dpnp.tan` : Compute tangent element-wise.
    :obj:`dpnp.angle` : Return the angle of the complex argument.
    :obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.
    :obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.
    :obj:`dpnp.arctanh` : Inverse hyperbolic tangent, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([1., -1.])
    >>> x2 = np.array([0., 0.])
    >>> np.arctan2(x1, x2)
    array([1.57079633, -1.57079633])

    >>> x1 = np.array([0., 0., np.inf])
    >>> x2 = np.array([+0., -0., np.inf])
    >>> np.arctan2(x1, x2)
    array([0.0 , 3.14159265, 0.78539816])

    >>> x1 = np.array([-1, +1, +1, -1])
    >>> x2 = np.array([-1, -1, +1, +1])
    >>> np.arctan2(x1, x2) * 180 / np.pi
    array([-135.,  -45.,   45.,  135.])

    """

    return check_nd_call_func(
        numpy.arctan2,
        dpnp_atan2,
        x1,
        x2,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def arctanh(
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
    Hyperbolic inverse tangent, element-wise.

    For full documentation refer to :obj:`numpy.arctanh`.

    Returns
    -------
    out : dpnp.ndarray
        The hyperbolic inverse tangent of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.
    :obj:`dpnp.arcsinh` : Hyperbolic inverse sine, element-wise.
    :obj:`dpnp.arccosh` : Hyperbolic inverse cosine, element-wise.
    :obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, -0.5])
    >>> np.arctanh(x)
    array([0.0, -0.54930614])

    """

    return check_nd_call_func(
        numpy.arctanh,
        dpnp_atanh,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def cbrt(x1):
    """
    Return the cube-root of an array, element-wise.

    For full documentation refer to :obj:`numpy.cbrt`.

    Limitations
    -----------
    Input array is supported as :class:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 8, 27])
    >>> out = np.cbrt(x)
    >>> [i for i in out]
    [1.0, 2.0, 3.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_cbrt(x1_desc).get_pyobj()

    return call_origin(numpy.cbrt, x1, **kwargs)


def cos(
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
    Trigonometric cosine, element-wise.

    For full documentation refer to :obj:`numpy.cos`.

    Returns
    -------
    out : dpnp.ndarray
        The cosine of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.
    :obj:`dpnp.sin` : Trigonometric sine, element-wise.
    :obj:`dpnp.tan` : Trigonometric tangent, element-wise.
    :obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, np.pi/2, np.pi])
    >>> np.cos(x)
    array([ 1.000000e+00, -4.371139e-08, -1.000000e+00])

    """

    return check_nd_call_func(
        numpy.cos,
        dpnp_cos,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def cosh(
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
    Hyperbolic cosine, element-wise.

    For full documentation refer to :obj:`numpy.cosh`.

    Returns
    -------
    out : dpnp.ndarray
        The hyperbolic cosine of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arccosh` : Hyperbolic inverse cosine, element-wise.
    :obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
    :obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.
    :obj:`dpnp.cos` : Trigonometric cosine, element-wise.


    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, np.pi/2, np.pi])
    >>> np.cosh(x)
    array([1.0, 2.5091786, 11.591953])

    """

    return check_nd_call_func(
        numpy.cosh,
        dpnp_cosh,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def deg2rad(x1):
    """
    Convert angles from degrees to radians.

    For full documentation refer to :obj:`numpy.deg2rad`.

    See Also
    --------
    :obj:`dpnp.rad2deg` : Convert angles from radians to degrees.
    :obj:`dpnp.unwrap` : Remove large jumps in angle by wrapping.

    Notes
    -----
    This function works exactly the same as :obj:`dpnp.radians`.

    """

    return radians(x1)


def degrees(x1):
    """
    Convert angles from radians to degrees.

    For full documentation refer to :obj:`numpy.degrees`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.rad2deg` convert angles from radians to degrees.

    Examples
    --------
    >>> import dpnp as np
    >>> rad = np.arange(6.) * np.pi/6
    >>> out = np.degrees(rad)
    >>> [i for i in out]
    [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_degrees(x1_desc).get_pyobj()

    return call_origin(numpy.degrees, x1, **kwargs)


def exp(x1, out=None, **kwargs):
    """
    Calculate the exponential, element-wise.

    For full documentation refer to :obj:`numpy.exp`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.expm1` : Calculate ``exp(x) - 1`` for all elements in the array.
    :obj:`dpnp.exp2` : Calculate `2**x` for all elements in the array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(3.)
    >>> out = np.exp(x)
    >>> [i for i in out]
    [1.0, 2.718281828, 7.389056099]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        out_desc = (
            dpnp.get_dpnp_descriptor(out, copy_when_nondefault_queue=False)
            if out is not None
            else None
        )
        return dpnp_exp(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.exp, x1, out=out, **kwargs)


def exp2(x1):
    """
    Calculate `2**p` for all `p` in the input array.

    For full documentation refer to :obj:`numpy.exp2`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.power` : First array elements raised to powers from
                        second array, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(3.)
    >>> out = np.exp2(x)
    >>> [i for i in out]
    [1.0, 2.0, 4.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_exp2(x1_desc).get_pyobj()

    return call_origin(numpy.exp2, x1)


def expm1(x1):
    """
    Return the exponential of the input array minus one, element-wise.

    For full documentation refer to :obj:`numpy.expm1`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.log1p` ``log(1 + x)``, the inverse of expm1.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(3.)
    >>> out = np.expm1(x)
    >>> [i for i in out]
    [0.0, 1.718281828, 6.389056099]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_expm1(x1_desc).get_pyobj()

    return call_origin(numpy.expm1, x1)


def hypot(x1, x2, /, out=None, *, where=True, dtype=None, subok=True, **kwargs):
    """
    Given the "legs" of a right triangle, return its hypotenuse.

    For full documentation refer to :obj:`numpy.hypot`.

    Returns
    -------
    out : dpnp.ndarray
        The hypotenuse of the triangle(s).

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = 3 * np.ones((3, 3))
    >>> x2 = 4 * np.ones((3, 3))
    >>> np.hypot(x1, x2)
    array([[5., 5., 5.],
           [5., 5., 5.],
           [5., 5., 5.]])

    Example showing broadcast of scalar argument:

    >>> np.hypot(x1, 4)
    array([[ 5.,  5.,  5.],
           [ 5.,  5.,  5.],
           [ 5.,  5.,  5.]])

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

            return dpnp_hypot(
                x1_desc, x2_desc, dtype=dtype, out=out_desc, where=where
            ).get_pyobj()

    return call_origin(
        numpy.hypot, x1, x2, dtype=dtype, out=out, where=where, **kwargs
    )


def log(
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
    Natural logarithm, element-wise.

    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base `e`.

    For full documentation refer to :obj:`numpy.log`.

    Returns
    -------
    out : dpnp.ndarray
        The natural logarithm of `x`, element-wise.

    Limitations
    -----------
    Parameters `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.log10` : Return the base 10 logarithm of the input array,
                        element-wise.
    :obj:`dpnp.log2` : Base-2 logarithm of x.
    :obj:`dpnp.log1p` : Return the natural logarithm of one plus
                        the input array, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, np.e, np.e**2, 0])
    >>> np.log(x)
    array([  0.,   1.,   2., -inf])

    """

    return check_nd_call_func(
        numpy.log,
        dpnp_log,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def log10(x1):
    """
    Return the base 10 logarithm of the input array, element-wise.

    For full documentation refer to :obj:`numpy.log10`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(3.)
    >>> out = np.log10(x)
    >>> [i for i in out]
    [-inf, 0.0, 0.30102999566]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_log10(x1_desc).get_pyobj()

    return call_origin(numpy.log10, x1)


def log1p(x1):
    """
    Return the natural logarithm of one plus the input array, element-wise.

    For full documentation refer to :obj:`numpy.log1p`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.expm1` : ``exp(x) - 1``, the inverse of :obj:`dpnp.log1p`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(3.)
    >>> out = np.log1p(x)
    >>> [i for i in out]
    [0.0, 0.69314718, 1.09861229]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_log1p(x1_desc).get_pyobj()

    return call_origin(numpy.log1p, x1)


def log2(x1):
    """
    Return the base 2 logarithm of the input array, element-wise.

    For full documentation refer to :obj:`numpy.log2`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.log` : Natural logarithm, element-wise.
    :obj:`dpnp.log10` : Return the base 10 logarithm of the input array,
                        element-wise.
    :obj:`dpnp.log1p` : Return the natural logarithm of one plus
                        the input array, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, 1, 2, 2**4])
    >>> out = np.log2(x)
    >>> [i for i in out]
    [-inf, 0.0, 1.0, 4.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_log2(x1_desc).get_pyobj()

    return call_origin(numpy.log2, x1)


def reciprocal(x1, **kwargs):
    """
    Return the reciprocal of the argument, element-wise.

    For full documentation refer to :obj:`numpy.reciprocal`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2., 3.33])
    >>> out = np.reciprocal(x)
    >>> [i for i in out]
    [1.0, 0.5, 0.3003003]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc and not kwargs:
        return dpnp_recip(x1_desc).get_pyobj()

    return call_origin(numpy.reciprocal, x1, **kwargs)


def rad2deg(x1):
    """
    Convert angles from radians to degrees.

    For full documentation refer to :obj:`numpy.rad2deg`.

    See Also
    --------
    :obj:`dpnp.deg2rad` : Convert angles from degrees to radians.
    :obj:`dpnp.unwrap` : Remove large jumps in angle by wrapping.

    Notes
    -----
    This function works exactly the same as :obj:`dpnp.degrees`.

    """

    return degrees(x1)


def radians(x1):
    """
    Convert angles from degrees to radians.

    For full documentation refer to :obj:`numpy.radians`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.deg2rad` equivalent function.

    Examples
    --------
    >>> import dpnp as np
    >>> deg = np.arange(6.) * 30.
    >>> out = np.radians(deg)
    >>> [i for i in out]
    [0.0, 0.52359878, 1.04719755, 1.57079633, 2.0943951, 2.61799388]

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    if x1_desc:
        return dpnp_radians(x1_desc).get_pyobj()

    return call_origin(numpy.radians, x1, **kwargs)


def sin(
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
    Trigonometric sine, element-wise.

    For full documentation refer to :obj:`numpy.sin`.

    Returns
    -------
    out : dpnp.ndarray
        The sine of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arcsin` : Trigonometric inverse sine, element-wise.
    :obj:`dpnp.cos` : Trigonometric cosine, element-wise.
    :obj:`dpnp.tan` : Trigonometric tangent, element-wise.
    :obj:`dpnp.sinh` : Hyperbolic sine, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, np.pi/2, np.pi])
    >>> np.sin(x)
    array([ 0.000000e+00,  1.000000e+00, -8.742278e-08])

    """

    return check_nd_call_func(
        numpy.sin,
        dpnp_sin,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def sinh(
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
    Hyperbolic sine, element-wise.

    For full documentation refer to :obj:`numpy.sinh`.

    Returns
    -------
    out : dpnp.ndarray
        The hyperbolic sine of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arcsinh` : Hyperbolic inverse sine, element-wise.
    :obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.
    :obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.
    :obj:`dpnp.sin` : Trigonometric sine, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, np.pi/2, np.pi])
    >>> np.sinh(x)
    array([0.0, 2.3012989, 11.548739])

    """

    return check_nd_call_func(
        numpy.sinh,
        dpnp_sinh,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def sqrt(
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
    Return the non-negative square-root of an array, element-wise.

    For full documentation refer to :obj:`numpy.sqrt`.

    Returns
    -------
    out : dpnp.ndarray
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`.  If any element in `x` is
        complex, a complex array is returned (and the square-roots of
        negative reals are calculated).  If all of the elements in `x`
        are real, so is `y`, with negative elements returning ``nan``.

    Limitations
    -----------
    Input array is supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameter `out` is supported as class:`dpnp.ndarray`, class:`dpctl.tensor.usm_ndarray` or
    with default value ``None``.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 4, 9])
    >>> np.sqrt(x)
    array([1., 2., 3.])

    >>> x2 = np.array([4, -1, np.inf])
    >>> np.sqrt(x2)
    array([ 2., nan, inf])

    """

    return check_nd_call_func(
        numpy.sqrt,
        dpnp_sqrt,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def square(
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
    Return the element-wise square of the input.

    For full documentation refer to :obj:`numpy.square`.

    Returns
    -------
    out : dpnp.ndarray
        Element-wise `x * x`, of the same shape and dtype as `x`.

    Limitations
    -----------
    Input array is supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameter `out` is supported as class:`dpnp.ndarray`, class:`dpctl.tensor.usm_ndarray` or
    with default value ``None``.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp..linalg.matrix_power` : Raise a square matrix
                                       to the (integer) power `n`.
    :obj:`dpnp.sqrt` : Return the positive square-root of an array,
                       element-wise.
    :obj:`dpnp.power` : First array elements raised to powers
                        from second array, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([-1j, 1])
    >>> np.square(x)
    array([-1.+0.j,  1.+0.j])

    """

    return check_nd_call_func(
        numpy.square,
        dpnp_square,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def tan(
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
    Trigonometric tangent, element-wise.

    For full documentation refer to :obj:`numpy.tan`.

    Returns
    -------
    out : dpnp.ndarray
        The tangent of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
    :obj:`dpnp.sin` : Trigonometric sine, element-wise.
    :obj:`dpnp.cos` : Trigonometric cosine, element-wise.
    :obj:`dpnp.tanh` : Hyperbolic tangent, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([-np.pi, np.pi/2, np.pi])
    >>> np.tan(x)
    array([1.22460635e-16, 1.63317787e+16, -1.22460635e-16])

    """

    return check_nd_call_func(
        numpy.tan,
        dpnp_tan,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def tanh(
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
    Compute hyperbolic tangent element-wise.

    For full documentation refer to :obj:`numpy.tanh`.

    Returns
    -------
    out : dpnp.ndarray
        The hyperbolic tangent of each element of `x`.

    Limitations
    -----------
    Parameter `x` is only supported as either :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Parameters `where`, `dtype` and `subok` are supported with their default values.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arctanh` : Hyperbolic inverse tangent, element-wise.
    :obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
    :obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.
    :obj:`dpnp.tan` : Trigonometric tangent, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, -np.pi, np.pi/2, np.pi])
    >>> np.tanh(x)
    array([0.0, -0.996272, 0.917152, 0.996272])

    """

    return check_nd_call_func(
        numpy.tanh,
        dpnp_tanh,
        x,
        out=out,
        where=where,
        order=order,
        dtype=dtype,
        subok=subok,
        **kwargs,
    )


def unwrap(x1):
    """
    Unwrap by changing deltas between values to 2*pi complement.

    For full documentation refer to :obj:`numpy.unwrap`.

    Limitations
    -----------
    Input array is supported as :class:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.rad2deg` : Convert angles from radians to degrees.
    :obj:`dpnp.deg2rad` : Convert angles from degrees to radians.

    Examples
    --------
    >>> import dpnp as np
    >>> phase = np.linspace(0, np.pi, num=5)
    >>> for i in range(3, 5):
    >>>     phase[i] += np.pi
    >>> out = np.unwrap(phase)
    >>> [i for i in out]
    [0.0, 0.78539816, 1.57079633, 5.49778714, 6.28318531]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        return dpnp_unwrap(x1_desc).get_pyobj()

    return call_origin(numpy.unwrap, x1, **kwargs)
