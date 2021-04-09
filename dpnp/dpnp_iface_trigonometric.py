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
from dpnp.dparray import dparray
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

    if (use_origin_backend(x1)):
        return numpy.arccos(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arccos(): Unsupported x1={type(x1)}")

    return dpnp_arccos(x1)


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

    if (use_origin_backend(x1)):
        return numpy.arccosh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arccosh(): Unsupported x1={type(x1)}")

    return dpnp_arccosh(x1)


def arcsin(x1):
    """
    Inverse sine, element-wise.

    For full documentation refer to :obj:`numpy.arcsin`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
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

    if (use_origin_backend(x1)):
        return numpy.arcsin(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arcsin(): Unsupported x1={type(x1)}")

    return dpnp_arcsin(x1)


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

    if (use_origin_backend(x1)):
        return numpy.arcsinh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arcsinh(): Unsupported x1={type(x1)}")

    return dpnp_arcsinh(x1)


def arctan(x1):
    """
    Trigonometric inverse tangent, element-wise.

    For full documentation refer to :obj:`numpy.arctan`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
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

    if (use_origin_backend(x1)):
        return numpy.arctan(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arctan(): Unsupported x1={type(x1)}")

    return dpnp_arctan(x1)


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

    if (use_origin_backend(x1)):
        return numpy.arctanh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arctanh(): Unsupported x1={type(x1)}")

    return dpnp_arctanh(x1)


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

    if (use_origin_backend(x1)):
        return numpy.cbrt(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP cbrt(): Unsupported x1={type(x1)}")

    return dpnp_cbrt(x1)


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
    x1_is_scalar, x2_is_scalar = dpnp.isscalar(x1), dpnp.isscalar(x2)
    x1_is_dparray, x2_is_dparray = isinstance(x1, dparray), isinstance(x2, dparray)

    if not use_origin_backend(x1) and not kwargs:
        if not x1_is_dparray and not x1_is_scalar:
            pass
        elif not x2_is_dparray and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_is_dparray and x1.ndim == 0:
            pass
        elif x2_is_dparray and x2.ndim == 0:
            pass
        elif out is not None and not isinstance(out, dparray):
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            return dpnp_arctan2(x1, x2, dtype=dtype, out=out, where=where)

    return call_origin(numpy.arctan2, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def cos(x1):
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

    if (use_origin_backend(x1)):
        return numpy.cos(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP cos(): Unsupported x1={type(x1)}")

    return dpnp_cos(x1)


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

    if (use_origin_backend(x1)):
        return numpy.cosh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP cosh(): Unsupported x1={type(x1)}")

    return dpnp_cosh(x1)


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

    if (use_origin_backend(x1)):
        return numpy.radians(x1)

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

    if (use_origin_backend(x1)):
        return numpy.degrees(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP degrees(): Unsupported x1={type(x1)}")

    return dpnp_degrees(x1)


def exp(x1):
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
    if not use_origin_backend(x1):
        if not isinstance(x1, dparray):
            pass
        else:
            return dpnp_exp(x1)

    return call_origin(numpy.exp, x1)


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

    if (use_origin_backend(x1)):
        return numpy.exp2(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP exp2(): Unsupported x1={type(x1)}")

    return dpnp_exp2(x1)


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

    if (use_origin_backend(x1)):
        return numpy.expm1(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP expm1(): Unsupported x1={type(x1)}")

    return dpnp_expm1(x1)


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
    x1_is_scalar, x2_is_scalar = dpnp.isscalar(x1), dpnp.isscalar(x2)
    x1_is_dparray, x2_is_dparray = isinstance(x1, dparray), isinstance(x2, dparray)

    if not use_origin_backend(x1) and not kwargs:
        if not x1_is_dparray and not x1_is_scalar:
            pass
        elif not x2_is_dparray and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_is_dparray and x1.ndim == 0:
            pass
        elif x2_is_dparray and x2.ndim == 0:
            pass
        elif out is not None and not isinstance(out, dparray):
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            return dpnp_hypot(x1, x2, dtype=dtype, out=out, where=where)

    return call_origin(numpy.hypot, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def log(x1):
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
    if not use_origin_backend(x1):
        if not isinstance(x1, dparray):
            pass
        else:
            return dpnp_log(x1)

    return call_origin(numpy.log, x1)


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

    if (use_origin_backend(x1)):
        return numpy.log10(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP log10(): Unsupported x1={type(x1)}")

    return dpnp_log10(x1)


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

    if (use_origin_backend(x1)):
        return numpy.log1p(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP log1p(): Unsupported x1={type(x1)}")

    return dpnp_log1p(x1)


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

    if (use_origin_backend(x1)):
        return numpy.log2(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP log2(): Unsupported x1={type(x1)}")

    return dpnp_log2(x1)


def reciprocal(x, **kwargs):
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
    if not use_origin_backend(x) and not kwargs:
        if not isinstance(x, dparray):
            pass
        else:
            return dpnp_recip(x)

    return call_origin(numpy.reciprocal, x, **kwargs)


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

    if (use_origin_backend(x1)):
        return numpy.degrees(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP rad2deg(): Unsupported x1={type(x1)}")

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

    if (use_origin_backend(x1)):
        return numpy.radians(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP radians(): Unsupported x1={type(x1)}")

    return dpnp_radians(x1)


def sin(x1):
    """
    Trigonometric sine, element-wise.

    For full documentation refer to :obj:`numpy.sin`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
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

    if (use_origin_backend(x1)):
        return numpy.sin(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP sin(): Unsupported x1={type(x1)}")

    return dpnp_sin(x1)


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

    if (use_origin_backend(x1)):
        return numpy.sinh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP sinh(): Unsupported x1={type(x1)}")

    return dpnp_sinh(x1)


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

    if (use_origin_backend(x1)):
        return numpy.sqrt(x1)

    if not isinstance(x1, dparray):
        return numpy.sqrt(x1)

    return dpnp_sqrt(x1)


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

    if (use_origin_backend(x1)):
        return numpy.square(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP square(): Unsupported x1={type(x1)}")

    return dpnp_square(x1)


def tan(x1):
    """
    Compute tangent element-wise.

    For full documentation refer to :obj:`numpy.tan`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
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

    if (use_origin_backend(x1)):
        return numpy.tan(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP tan(): Unsupported x1={type(x1)}")

    return dpnp_tan(x1)


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

    if (use_origin_backend(x1)):
        return numpy.tanh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP tanh(): Unsupported x1={type(x1)}")

    return dpnp_tanh(x1)


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

    if (use_origin_backend(x1)):
        return numpy.unwrap(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP unwrap(): Unsupported x1={type(x1)}")

    return dpnp_unwrap(x1)
