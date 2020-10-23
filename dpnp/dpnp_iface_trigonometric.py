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

from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *


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

    The inverse of cos so that, if y = cos(x), then x = arccos(y).

    Parameters
    ----------
    x1 : x-coordinate on the unit circle. For real arguments, the domain is [-1, 1].

    Returns
    -------
    out : The angle of the ray intersecting the unit circle at the given x-coordinate in radians [0, pi]. This is a scalar if x is a scalar.

    See Also
    --------
    cos, arctan, arcsin, emath.arccos

    """

    if (use_origin_backend(x1)):
        return numpy.arccos(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arccos(): Unsupported x1={type(x1)}")

    return dpnp_arccos(x1)


def arccosh(x1):
    """
    Trigonometric inverse hyperbolic cosine, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.arccosh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arccosh(): Unsupported x1={type(x1)}")

    return dpnp_arccosh(x1)


def arcsin(x1):
    """
    Inverse sine, element-wise.

    Parameters
    ----------
    x1 : y-coordinate on the unit circle.

    Returns
    -------
    out : The inverse sine of each element in x, in radians and in the closed interval [-pi/2, pi/2]. This is a scalar if x is a scalar.

    See Also
    --------
    sin, cos, arccos, tan, arctan, arctan2, emath.arcsin

    """

    if (use_origin_backend(x1)):
        return numpy.arcsin(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arcsin(): Unsupported x1={type(x1)}")

    return dpnp_arcsin(x1)


def arcsinh(x1):
    """
    Inverse hyperbolic sine, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.arcsinh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arcsinh(): Unsupported x1={type(x1)}")

    return dpnp_arcsinh(x1)


def arctan(x1):
    """
    Trigonometric inverse tangent, element-wise.

    The inverse of tan, so that if y = tan(x) then x = arctan(y).

    Parameters
    ----------
    x1 : array_like

    Returns
    -------
    out : Out has the same shape as x. Its real part is in [-pi/2, pi/2] (arctan(+/-inf) returns +/-pi/2). This is a scalar if x is a scalar.

    See Also
    --------
    arctan2, angle

    """

    if (use_origin_backend(x1)):
        return numpy.arctan(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arctan(): Unsupported x1={type(x1)}")

    return dpnp_arctan(x1)


def arctanh(x1):
    """
    Trigonometric hyperbolic inverse tangent, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.arctanh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP arctanh(): Unsupported x1={type(x1)}")

    return dpnp_arctanh(x1)


def cbrt(x1):
    """
    Return the cube-root of an array, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.cbrt(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP cbrt(): Unsupported x1={type(x1)}")

    return dpnp_cbrt(x1)


def arctan2(x1, x2, out=None):
    """
    Element-wise arc tangent of x1/x2 choosing the quadrant correctly.

    The quadrant (i.e., branch) is chosen so that arctan2(x1, x2) is the signed angle in radians between the ray ending at the origin and passing through the point (1,0), and the ray ending at the origin and passing through the point (x2, x1). (Note the role reversal: the “y-coordinate” is the first function parameter, the “x-coordinate” is the second.) By IEEE convention, this function is defined for x2 = +/-0 and for either or both of x1 and x2 = +/-inf (see Notes for specific values).

    This function is not defined for complex-valued arguments; for the so-called argument of complex values, use angle.

    Parameters
    ----------
    x1 : y-coordinates.

    x2 : x-coordinates. If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Returns
    -------
    out : Array of angles in radians, in the range [-pi, pi]. This is a scalar if both x1 and x2 are scalars.

    See Also
    --------
    arctan, tan, angle

    """

    if (use_origin_backend(x1)):
        return numpy.arctan2(x1, x2)

    if not (isinstance(x1, dparray) or isinstance(x2, dparray)):
        return numpy.arctan2(x1, x2, out=out)

    if out is not None:
        checker_throw_value_error("arctan2", "out", type(out), None)

    if (x1.size != x2.size):
        checker_throw_value_error("arctan2", "size", x1.size, x2.size)

    if (x1.shape != x2.shape):
        checker_throw_value_error("arctan2", "shape", x1.shape, x2.shape)

    return dpnp_arctan2(x1, x2)


def cos(x1):
    """
    Trigonometric cosine, element-wise.

    Parameters
    ----------
    x1 : Angle, in radians (2 \pi rad equals 360 degrees).

    Returns
    -------
    out : ndarray, None, or tuple of ndarray and None, optional
        The cosine of each element of x. This is a scalar if x is a scalar

    See Also
    --------
    arccos, cosh, sin

    """

    if (use_origin_backend(x1)):
        return numpy.cos(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP cos(): Unsupported x1={type(x1)}")

    return dpnp_cos(x1)


def cosh(x1):
    """
    Trigonometric hyperbolic cosine, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.cosh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP cosh(): Unsupported x1={type(x1)}")

    return dpnp_cosh(x1)


def deg2rad(x1):
    """
    Convert angles from degrees to radians.
    `radians` equivalent function

    Parameters
    ----------
    x1 : Angles in degrees.

    Returns
    -------
    out : The corresponding angle in radians. This is a scalar if x is a scalar.

    See Also
    --------
    rad2deg, unwrap

    """

    if (use_origin_backend(x1)):
        return numpy.radians(x1)

    return radians(x1)


def degrees(x1):
    """
    Convert angles from radians to degrees.

    Parameters
    ----------
    x1 : Input array in radians.

    Returns
    -------
    out : The corresponding degree values; if out was supplied this is a reference to it. This is a scalar if x is a scalar.

    See Also
    --------
    rad2deg

    """

    if (use_origin_backend(x1)):
        return numpy.degrees(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP degrees(): Unsupported x1={type(x1)}")

    return dpnp_degrees(x1)


def exp(x1):
    """
    Trigonometric exponent, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.exp(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP exp(): Unsupported x1={type(x1)}")

    return dpnp_exp(x1)


def exp2(x1):
    """
    Trigonometric exponent2, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.exp2(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP exp2(): Unsupported x1={type(x1)}")

    return dpnp_exp2(x1)


def expm1(x1):
    """
    Trigonometric exponent minus 1, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.expm1(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP expm1(): Unsupported x1={type(x1)}")

    return dpnp_expm1(x1)


def hypot(x1, x2, out=None):
    """
    Given the “legs” of a right triangle, return its hypotenuse.

    Equivalent to sqrt(x1**2 + x2**2), element-wise. If x1 or x2 is scalar_like (i.e., unambiguously cast-able to a scalar type), it is broadcast for use with each element of the other argument. (See Examples)

    Parameters
    ----------
    x1, x2 : Leg of the triangle(s). If x1.shape != x2.shape, they must be broadcastable to a common shape (which becomes the shape of the output).

    Returns
    -------
    out : The hypotenuse of the triangle(s). This is a scalar if both x1 and x2 are scalars.

    """

    if (use_origin_backend(x1)):
        return numpy.hypot(x1, x2)

    if not (isinstance(x1, dparray) or isinstance(x2, dparray)):
        return numpy.hypot(x1, x2, out=out)

    if out is not None:
        checker_throw_value_error("hypot", "out", type(out), None)

    if (x1.size != x2.size):
        checker_throw_value_error("hypot", "size", x1.size, x2.size)

    if (x1.shape != x2.shape):
        checker_throw_value_error("hypot", "shape", x1.shape, x2.shape)

    return dpnp_hypot(x1, x2)


def log(x1):
    """
    Trigonometric logarithm, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.log(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP log(): Unsupported x1={type(x1)}")

    return dpnp_log(x1)


def log10(x1):
    """
    Trigonometric logarithm, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.log10(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP log10(): Unsupported x1={type(x1)}")

    return dpnp_log10(x1)


def log1p(x1):
    """
    Trigonometric logarithm, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.log1p(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP log1p(): Unsupported x1={type(x1)}")

    return dpnp_log1p(x1)


def log2(x1):
    """
    Trigonometric logarithm, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.log2(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP log2(): Unsupported x1={type(x1)}")

    return dpnp_log2(x1)


def reciprocal(x, **kwargs):
    """
    Return the reciprocal of the argument, element-wise.

    Calculates 1/x.

    Parameters
    ----------
    x : array_like
        Input array.
    kwargs : dict
        Remaining input parameters of the function.

    Returns
    -------
    y : ndarray
        Return array. This is a scalar if x is a scalar.
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
    `degrees` equivalent function

    Parameters
    ----------
    x1 : Angle in radians.

    Returns
    -------
    out : The corresponding angle in degrees. This is a scalar if x is a scalar.

    See Also
    --------
    deg2rad, unwrap

    """

    if (use_origin_backend(x1)):
        return numpy.degrees(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP rad2deg(): Unsupported x1={type(x1)}")

    return degrees(x1)


def radians(x1):
    """
    Convert angles from degrees to radians.

    Parameters
    ----------
    x1 : Input array in degrees.

    Returns
    -------
    out : The corresponding radian values. This is a scalar if x is a scalar.

    See Also
    --------
    deg2rad

    """

    if (use_origin_backend(x1)):
        return numpy.radians(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP radians(): Unsupported x1={type(x1)}")

    return dpnp_radians(x1)


def sin(x1):
    """
    Trigonometric sine, element-wise.

    Parameters
    ----------
    x1 : Angle, in radians (2 \pi rad equals 360 degrees).

    Returns
    -------
    out : ndarray, None, or tuple of ndarray and None, optional
        The sine of each element of x. This is a scalar if x is a scalar

    See Also
    --------
    arcsin, sinh, cos

    """

    if (use_origin_backend(x1)):
        return numpy.sin(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP sin(): Unsupported x1={type(x1)}")

    return dpnp_sin(x1)


def sinh(x1):
    """
    Trigonometric hyperbolic sine, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.sinh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP sinh(): Unsupported x1={type(x1)}")

    return dpnp_sinh(x1)


def sqrt(x1):
    """
    Return the positive square-root of an array, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.sqrt(x1)

    if not isinstance(x1, dparray):
        return numpy.sqrt(x1)

    return dpnp_sqrt(x1)


def square(x1):
    """
    Return X1 * x1, element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.square(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP square(): Unsupported x1={type(x1)}")

    return dpnp_square(x1)


def tan(x1):
    """
    Compute tangent element-wise.

    Equivalent to np.sin(x)/np.cos(x) element-wise.

    Parameters
    ----------
    x1 : Input array.

    Returns
    -------
    out : The corresponding tangent values. This is a scalar if x is a scalar.

    """

    if (use_origin_backend(x1)):
        return numpy.tan(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP tan(): Unsupported x1={type(x1)}")

    return dpnp_tan(x1)


def tanh(x1):
    """
    Compute hyperbolic tangent element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.tanh(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP tanh(): Unsupported x1={type(x1)}")

    return dpnp_tanh(x1)


def unwrap(x1):
    """
    Unwrap by changing deltas between values to 2*pi complement.

    Unwrap radian phase p by changing absolute jumps greater than discont to their 2*pi complement along the given axis.

    Parameters
    ----------
    x1 : Input array.

    Returns
    -------
    out : Output array.

    See Also
    --------
    rad2deg, deg2rad

    """

    if (use_origin_backend(x1)):
        return numpy.unwrap(x1)

    if not isinstance(x1, dparray):
        raise TypeError(f"DPNP unwrap(): Unsupported x1={type(x1)}")

    return dpnp_unwrap(x1)
