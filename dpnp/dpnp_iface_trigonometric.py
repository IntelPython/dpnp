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

from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *
import dpnp


__all__ = [
    'arccos',
    'arccosh',
    'arcsin',
    'arcsinh',
    'arctan',
    'arctan2',
    'arctanh',
    'cbrt',
    'cos',
    'cosh',
    'deg2rad',
    'degrees',
    'exp',
    'exp2',
    'expm1',
    'hypot',
    'log',
    'log10',
    'log1p',
    'log2',
    'rad2deg',
    'radians',
    'reciprocal',
    'sin',
    'sinh',
    'sqrt',
    'square',
    'tan',
    'tanh',
    'unwrap'
]


def arccos(x1):
    """
    Trigonometric inverse cosine, element-wise.

    For full documentation refer to :obj:`numpy.arccos`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.cos` : Cosine element-wise.
    :obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
    :obj:`dpnp.arcsin` : Inverse sine, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, -1])
    >>> out = np.arccos(x)
    >>> [i for i in out]
    [0.0,  3.14159265]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_arccos(x1_desc).get_pyobj()

    return call_origin(numpy.arccos, x1, **kwargs)


def arccosh(x1):
    """
    Trigonometric inverse hyperbolic cosine, element-wise.

    For full documentation refer to :obj:`numpy.arccosh`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.cosh` : Hyperbolic cosine, element-wise.
    :obj:`dpnp.arcsinh` : Inverse hyperbolic sine element-wise.
    :obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
    :obj:`dpnp.arctanh` : Inverse hyperbolic tangent element-wise.
    :obj:`dpnp.tanh` : Compute hyperbolic tangent element-wise.

    Examples
    --------
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([numpy.e, 10.0])
    >>> out = np.arccosh(x)
    >>> [i for i in out]
    [1.65745445, 2.99322285]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_arccosh(x1_desc).get_pyobj()

    return call_origin(numpy.arccosh, x1, **kwargs)


def arcsin(x1, out=None, **kwargs):
    """
    Inverse sine, element-wise.

    For full documentation refer to :obj:`numpy.arcsin`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Keyword arguments ``kwargs`` are currently unsupported.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.sin` : Trigonometric sine, element-wise.
    :obj:`dpnp.cos` : Cosine element-wise.
    :obj:`dpnp.arccos` : Trigonometric inverse cosine, element-wise.
    :obj:`dpnp.tan` : Compute tangent element-wise.
    :obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
    :obj:`dpnp.arctan2` : Element-wise arc tangent of ``x1/x2``
                          choosing the quadrant correctly.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, 1, -1])
    >>> out = np.arcsin(x)
    >>> [i for i in out]
    [0.0, 1.5707963267948966, -1.5707963267948966]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_arcsin(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.arcsin, x1, out=out, **kwargs)


def arcsinh(x1):
    """
    Inverse hyperbolic sine, element-wise.

    For full documentation refer to :obj:`numpy.arcsinh`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([numpy.e, 10.0])
    >>> out = np.arcsinh(x)
    >>> [i for i in out]
    [1.72538256, 2.99822295]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_arcsinh(x1_desc).get_pyobj()

    return call_origin(numpy.arcsinh, x1, **kwargs)


def arctan(x1, out=None, **kwargs):
    """
    Trigonometric inverse tangent, element-wise.

    For full documentation refer to :obj:`numpy.arctan`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Keyword arguments ``kwargs`` are currently unsupported.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arctan2` : Element-wise arc tangent of ``x1/x2``
                          choosing the quadrant correctly.
    :obj:`dpnp.angle` : Argument of complex values.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, 1])
    >>> out = np.arctan(x)
    >>> [i for i in out]
    [0.0, 0.78539816]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_arctan(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.arctan, x1, out=out, **kwargs)


def arctanh(x1):
    """
    Trigonometric hyperbolic inverse tangent, element-wise.

    For full documentation refer to :obj:`numpy.arctanh`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0, -0.5])
    >>> out = np.arctanh(x)
    >>> [i for i in out]
    [0.0, -0.54930614]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_arctanh(x1_desc).get_pyobj()

    return call_origin(numpy.arctanh, x1, **kwargs)


def cbrt(x1):
    """
    Return the cube-root of an array, element-wise.

    For full documentation refer to :obj:`numpy.cbrt`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 8, 27])
    >>> out = np.cbrt(x)
    >>> [i for i in out]
    [1.0, 2.0, 3.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_cbrt(x1_desc).get_pyobj()

    return call_origin(numpy.cbrt, x1, **kwargs)


def arctan2(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Element-wise arc tangent of ``x1/x2`` choosing the quadrant correctly.

    For full documentation refer to :obj:`numpy.arctan2`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arctan` : Trigonometric inverse tangent, element-wise.
    :obj:`dpnp.tan` : Compute tangent element-wise.
    :obj:`dpnp.angle` : Return the angle of the complex argument.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([1., -1.])
    >>> x2 = np.array([0., 0.])
    >>> out = np.arctan2(x1, x2)
    >>> [i for i in out]
    [1.57079633, -1.57079633]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif not where:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            return dpnp_arctan2(x1_desc, x2_desc, dtype, out_desc, where).get_pyobj()

    return call_origin(numpy.arctan2, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def cos(x1, out=None, **kwargs):
    """
    Trigonometric cosine, element-wise.

    For full documentation refer to :obj:`numpy.cos`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([0, numpy.pi/2, numpy.pi])
    >>> out = np.cos(x)
    >>> [i for i in out]
    [1.0, 6.123233995736766e-17, -1.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_cos(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.cos, x1, out=out, **kwargs)


def cosh(x1):
    """
    Trigonometric hyperbolic cosine, element-wise.

    For full documentation refer to :obj:`numpy.cosh`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0])
    >>> out = np.cosh(x)
    >>> [i for i in out]
    [1.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_cosh(x1_desc).get_pyobj()

    return call_origin(numpy.cosh, x1, **kwargs)


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
    >>> import numpy
    >>> import dpnp as np
    >>> rad = np.arange(6.) * numpy.pi/6
    >>> out = np.degrees(rad)
    >>> [i for i in out]
    [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_degrees(x1_desc).get_pyobj()

    return call_origin(numpy.degrees, x1, **kwargs)


def exp(x1, out=None, **kwargs):
    """
    Trigonometric exponent, element-wise.

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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_exp(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.exp, x1, out=out, **kwargs)


def exp2(x1):
    """
    Trigonometric exponent2, element-wise.

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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_exp2(x1_desc).get_pyobj()

    return call_origin(numpy.exp2, x1)


def expm1(x1):
    """
    Trigonometric exponent minus 1, element-wise.

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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_expm1(x1_desc).get_pyobj()

    return call_origin(numpy.expm1, x1)


def hypot(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Given the "legs" of a right triangle, return its hypotenuse.

    For full documentation refer to :obj:`numpy.hypot`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = 3 * np.ones(3)
    >>> x2 = 4 * np.ones(3)
    >>> out = np.hypot(x1, x2)
    >>> [i for i in out]
    [5.0, 5.0, 5.0]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            return dpnp_hypot(x1_desc, x2_desc, dtype, out_desc, where).get_pyobj()

    return call_origin(numpy.hypot, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def log(x1, out=None, **kwargs):
    """
    Trigonometric logarithm, element-wise.

    For full documentation refer to :obj:`numpy.log`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
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
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([1.0, numpy.e, numpy.e**2, 0.0])
    >>> out = np.log(x)
    >>> [i for i in out]
    [0.0, 1.0, 2.0, -inf]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_log(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.log, x1, out=out, **kwargs)


def log10(x1):
    """
    Trigonometric logarithm, element-wise.

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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_log10(x1_desc).get_pyobj()

    return call_origin(numpy.log10, x1)


def log1p(x1):
    """
    Trigonometric logarithm, element-wise.

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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_log1p(x1_desc).get_pyobj()

    return call_origin(numpy.log1p, x1)


def log2(x1):
    """
    Trigonometric logarithm, element-wise.

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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
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
    Keyword arguments ``kwargs`` are currently unsupported.
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_radians(x1_desc).get_pyobj()

    return call_origin(numpy.radians, x1, **kwargs)


def sin(x1, out=None, **kwargs):
    """
    Trigonometric sine, element-wise.

    For full documentation refer to :obj:`numpy.sin`.

    Limitations
    -----------
    Parameters ``x1`` is supported as :obj:`dpnp.ndarray`.
    Parameter ``out`` is supported as default value ``None``.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.arcsin` : Inverse sine, element-wise.
    :obj:`dpnp.sinh` : Hyperbolic sine, element-wise.
    :obj:`dpnp.cos` : Cosine element-wise.

    Examples
    --------
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([0, numpy.pi/2, numpy.pi])
    >>> out = np.sin(x)
    >>> [i for i in out]
    [0.0, 1.0, 1.2246467991473532e-16]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_sin(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.sin, x1, out=out, **kwargs)


def sinh(x1):
    """
    Trigonometric hyperbolic sine, element-wise.

    For full documentation refer to :obj:`numpy.sinh`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([0, numpy.pi/2, numpy.pi])
    >>> out = np.sinh(x)
    >>> [i for i in out]
    [0.0, 2.3012989, 11.548739]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_sinh(x1_desc).get_pyobj()

    return call_origin(numpy.sinh, x1, **kwargs)


def sqrt(x1):
    """
    Return the positive square-root of an array, element-wise.

    For full documentation refer to :obj:`numpy.sqrt`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 4, 9])
    >>> out = np.sqrt(x)
    >>> [i for i in out]
    [1.0, 2.0, 3.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_sqrt(x1_desc).get_pyobj()

    return call_origin(numpy.sqrt, x1)


def square(x1):
    """
    Return the element-wise square of the input.

    For full documentation refer to :obj:`numpy.square`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.sqrt` : Return the positive square-root of an array,
                       element-wise.
    :obj:`dpnp.power` : First array elements raised to powers
                        from second array, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2, 3])
    >>> out = np.square(x)
    >>> [i for i in out]
    [1, 4, 9]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_square(x1_desc).get_pyobj()

    return call_origin(numpy.square, x1, **kwargs)


def tan(x1, out=None, **kwargs):
    """
    Compute tangent element-wise.

    For full documentation refer to :obj:`numpy.tan`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Keyword arguments ``kwargs`` are currently unsupported.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([-numpy.pi, numpy.pi/2, numpy.pi])
    >>> out = np.tan(x)
    >>> [i for i in out]
    [1.22460635e-16, 1.63317787e+16, -1.22460635e-16]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_tan(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.tan, x1, out=out, **kwargs)


def tanh(x1):
    """
    Compute hyperbolic tangent element-wise.

    For full documentation refer to :obj:`numpy.tanh`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import numpy
    >>> import dpnp as np
    >>> x = np.array([-numpy.pi, numpy.pi/2, numpy.pi])
    >>> out = np.tanh(x)
    >>> [i for i in out]
    [-0.996272, 0.917152, 0.996272]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_tanh(x1_desc).get_pyobj()

    return call_origin(numpy.tanh, x1, **kwargs)


def unwrap(x1):
    """
    Unwrap by changing deltas between values to 2*pi complement.

    For full documentation refer to :obj:`numpy.unwrap`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.rad2deg` : Convert angles from radians to degrees.
    :obj:`dpnp.deg2rad` : Convert angles from degrees to radians.

    Examples
    --------
    >>> import numpy
    >>> import dpnp as np
    >>> phase = np.linspace(0, numpy.pi, num=5)
    >>> for i in range(3, 5):
    >>>     phase[i] += numpy.pi
    >>> out = np.unwrap(phase)
    >>> [i for i in out]
    [0.0, 0.78539816, 1.57079633, 5.49778714, 6.28318531]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_unwrap(x1_desc).get_pyobj()

    return call_origin(numpy.unwrap, x1, **kwargs)
