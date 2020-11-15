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
Interface of the Mathematical part of the DPNP

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
    "abs",
    "absolute",
    "add",
    "ceil",
    "copysign",
    "divide",
    "fabs",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "nanprod",
    "nansum",
    "negative",
    "power",
    "prod",
    "remainder",
    "sign",
    "subtract",
    "sum",
    "true_divide",
    "trunc"
]


def abs(*args, **kwargs):
    """
    Calculate the absolute value element-wise.

    .. seealso:: :func:`numpy.add`

    """

    return dpnp.absolute(*args, **kwargs)


def absolute(x1, **kwargs):
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------
    x1 : array_like
        Input array.

    Returns
    -------
    absolute : ndarray
        An ndarray containing the absolute value of each element in x.
    """

    is_input_dparray = isinstance(x1, dparray)

    if not use_origin_backend(x1) and is_input_dparray and x1.ndim != 0 and not kwargs:
        result = dpnp_absolute(x1)

        return result

    return call_origin(numpy.absolute, x1, **kwargs)


def add(x1, x2, **kwargs):
    """
    Add arguments element-wise.

    .. note::
        The 'out' parameter is currently not supported.

    Args:
        x1  (dpnp.dparray): The left argument.
        x2  (dpnp.dparray): The right argument.
        out (dpnp.dparray): Output array.

    Returns:
        dpnp.dparray: The sum of x1 and x2, element-wise.
        This is a scalar if both x1 and x2 are scalars.

    .. seealso:: :func:`numpy.add`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        if (x1.size != x2.size):
            checker_throw_value_error("add", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("add", "shape", x1.shape, x2.shape)

        return dpnp_add(x1, x2)

    return call_origin(numpy.add, x1, x2, **kwargs)


def ceil(x1, **kwargs):
    """
    Compute  the ceiling of the input, element-wise.

    .. seealso:: :func:`numpy.ceil`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and not kwargs):
        return dpnp_ceil(x1)

    return call_origin(numpy.ceil, x1, **kwargs)


def copysign(x1, x2, **kwargs):
    """
    Change the sign of x1 to that of x2, element-wise.

    Parameters
    ----------
    x1 : array_like
        Values to change the sign of.
    x2 : array_like
        The sign of x2 is copied to x1.
    kwargs : dict
        Remaining input parameters of the function.

    Returns
    -------
    out: ndarray or scalar
        The values of x1 with the sign of x2.
    """
    if not use_origin_backend(x1) and not kwargs:
        if not isinstance(x1, dparray):
            pass
        elif not isinstance(x2, dparray):
            pass
        elif x1.size != x2.size:
            pass
        elif x1.shape != x2.shape:
            pass
        else:
            return dpnp_copysign(x1, x2)

    return call_origin(numpy.copysign, x1, x2, **kwargs)


def divide(x1, x2, **kwargs):
    """
    Divide arguments element-wise.

    .. note::
        The 'out' parameter is currently not supported.

    Args:
        x1  (dpnp.dparray): The left argument.
        x2  (dpnp.dparray): The right argument.
        out (dpnp.dparray): Output array.

    Returns:
        dpnp.dparray: The division of x1 and x2, element-wise.
        This is a scalar if both x1 and x2 are scalars.

    .. seealso:: :func:`numpy.divide`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        if (x1.size != x2.size):
            checker_throw_value_error("divide", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("divide", "shape", x1.shape, x2.shape)

        return dpnp_divide(x1, x2)

    return call_origin(numpy.divide, x1, x2, **kwargs)


def fabs(x1, **kwargs):
    """
    Compute the absolute values element-wise.

    .. seealso:: :func:`numpy.fabs`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray):
        return dpnp_fabs(x1)

    return call_origin(numpy.fabs, x1, **kwargs)


def floor(x1, **kwargs):
    """
    Compute the floor of the input, element-wise.

    Some spreadsheet programs calculate the “floor-towards-zero”, in other words floor(-2.5) == -2.
    dpNP instead uses the definition of floor where floor(-2.5) == -3.

    .. seealso:: :func:`numpy.floor`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and not kwargs):
        return dpnp_floor(x1)

    return call_origin(numpy.floor, x1, **kwargs)


def floor_divide(x1, x2, **kwargs):
    """
    Compute the largest integer smaller or equal to the division of the inputs.

    .. seealso:: :func:`numpy.floor_divide`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs:
        return dpnp_floor_divide(x1, x2)

    return call_origin(numpy.floor_divide, x1, x2, **kwargs)


def fmax(*args, **kwargs):
    """
    Element-wise maximum of array elements.

    .. seealso:: :func:`numpy.fmax`

    """

    return dpnp.maximum(*args, **kwargs)


def fmin(*args, **kwargs):
    """
    Element-wise minimum of array elements.

    .. seealso:: :func:`numpy.fmin`

    """

    return dpnp.minimum(*args, **kwargs)


def fmod(x1, x2, **kwargs):
    """
    Calculate the element-wise remainder of division.

    .. seealso:: :func:`numpy.fmod`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    # "dtype is None" is important here because we have no kernels with runtime dependent output type
    # kernels are (use "python example4.py" to investigate):
    # input1:float64   : input2:float64   : output:float64   : name:<ufunc 'fmod'>
    # input1:float64   : input2:float32   : output:float64   : name:<ufunc 'fmod'>
    # input1:float64   : input2:int64     : output:float64   : name:<ufunc 'fmod'>
    # input1:float64   : input2:int32     : output:float64   : name:<ufunc 'fmod'>
    # input1:float64   : input2:bool      : output:float64   : name:<ufunc 'fmod'>
    # input1:float32   : input2:float64   : output:float64   : name:<ufunc 'fmod'>
    # input1:float32   : input2:float32   : output:float32   : name:<ufunc 'fmod'>
    # input1:float32   : input2:int64     : output:float64   : name:<ufunc 'fmod'>
    # input1:float32   : input2:int32     : output:float64   : name:<ufunc 'fmod'>
    # input1:float32   : input2:bool      : output:float32   : name:<ufunc 'fmod'>
    # input1:int64     : input2:float64   : output:float64   : name:<ufunc 'fmod'>
    # input1:int64     : input2:float32   : output:float64   : name:<ufunc 'fmod'>
    # input1:int64     : input2:int64     : output:int64     : name:<ufunc 'fmod'>
    # input1:int64     : input2:int32     : output:int64     : name:<ufunc 'fmod'>
    # input1:int64     : input2:bool      : output:int64     : name:<ufunc 'fmod'>
    # input1:int32     : input2:float64   : output:float64   : name:<ufunc 'fmod'>
    # input1:int32     : input2:float32   : output:float64   : name:<ufunc 'fmod'>
    # input1:int32     : input2:int64     : output:int64     : name:<ufunc 'fmod'>
    # input1:int32     : input2:int32     : output:int32     : name:<ufunc 'fmod'>
    # input1:int32     : input2:bool      : output:int32     : name:<ufunc 'fmod'>
    # input1:bool      : input2:float64   : output:float64   : name:<ufunc 'fmod'>
    # input1:bool      : input2:float32   : output:float32   : name:<ufunc 'fmod'>
    # input1:bool      : input2:int64     : output:int64     : name:<ufunc 'fmod'>
    # input1:bool      : input2:int32     : output:int32     : name:<ufunc 'fmod'>
    # input1:bool      : input2:bool      : output:int8      : name:<ufunc 'fmod'>
    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        if (x1.size != x2.size):
            checker_throw_value_error("fmod", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("fmod", "shape", x1.shape, x2.shape)

        return dpnp_fmod(x1, x2)

    return call_origin(numpy.fmod, x1, x2, **kwargs)


def maximum(x1, x2, **kwargs):
    """
    Element-wise maximum of array elements.

    .. seealso:: :func:`numpy.maximum`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        if (x1.size != x2.size):
            checker_throw_value_error("maximum", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("maximum", "shape", x1.shape, x2.shape)

        return dpnp_maximum(x1, x2)

    return call_origin(numpy.maximum, x1, x2, **kwargs)


def minimum(x1, x2, **kwargs):
    """
    Element-wise minimum of array elements.

    .. seealso:: :func:`numpy.minimum`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        if (x1.size != x2.size):
            checker_throw_value_error("minimum", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("minimum", "shape", x1.shape, x2.shape)

        return dpnp_minimum(x1, x2)

    return call_origin(numpy.minimum, x1, x2, **kwargs)


def mod(*args, **kwargs):
    """
    Compute element-wise remainder of division.

    Alias for :func:`dpnp.remainder`

    .. seealso:: :func:`numpy.mod`

    """

    return dpnp.remainder(*args, **kwargs)


def modf(x, **kwargs):
    """
    Return the fractional and integral parts of an array, element-wise.

    The fractional and integral parts are negative if the given number is negative.

    Parameters
    ----------
    x : array_like
        Input array.
    kwargs : dict
        Remaining input parameters of the function.

    Returns
    -------
    y1 : ndarray or scalar
        Fractional part of x. This is a scalar if x is a scalar.
    y2 : ndarray or scalar
        Integral part of x. This is a scalar if x is a scalar.

    See Also
    --------
    divmod
    """
    if not use_origin_backend(x) and not kwargs:
        if not isinstance(x, dparray):
            pass
        else:
            return dpnp_modf(x)

    return call_origin(numpy.modf, x, **kwargs)


def multiply(x1, x2, **kwargs):
    """
    Multiply arguments element-wise.

    .. note::
        The 'out' parameter is currently not supported.

    Args:
        x1  (dpnp.dparray): The left argument.
        x2  (dpnp.dparray): The right argument.
        out (dpnp.dparray): Output array.

    Returns:
        dpnp.dparray: The product of x1 and x2, element-wise.
        This is a scalar if both x1 and x2 are scalars.

    .. seealso:: :func:`numpy.multiply`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        if (x1.size != x2.size):
            checker_throw_value_error("multiply", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("multiply", "shape", x1.shape, x2.shape)

        return dpnp_multiply(x1, x2)

    return call_origin(numpy.multiply, x1, x2, **kwargs)


def nanprod(x1, **kwargs):
    """
    Calculate prod() function treating 'Not a Numbers' (NaN) as ones.

    .. seealso:: :func:`numpy.nanprod`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and not kwargs):
        return dpnp_nanprod(x1)

    return call_origin(numpy.nanprod, x1, **kwargs)


def nansum(x1, **kwargs):
    """
    Calculate sum() function treating 'Not a Numbers' (NaN) as zero.

    .. seealso:: :func:`numpy.nansum`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and not kwargs):
        return dpnp_nansum(x1)

    return call_origin(numpy.nansum, x1, **kwargs)


def negative(x1, **kwargs):
    """
    Negative element-wise.

    .. seealso:: :func:`numpy.negative`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        return dpnp_negative(x1, x2)

    return call_origin(numpy.negative, x1, **kwargs)


def power(x1, x2, **kwargs):
    """
    First array elements raised to powers from second array, element-wise.

    .. note::
        The 'out' parameter is currently not supported.

    Args:
        x1  (dpnp.dparray): array.
        x2  (dpnp.dparray): array.
        out (dpnp.dparray): Output array.

    Returns:
        dpnp.dparray: The bases in x1 raised to the exponents in x2.
        This is a scalar if both x1 and x2 are scalars.

    .. seealso:: :func:`numpy.power`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        if (x1.size != x2.size):
            checker_throw_value_error("power", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("power", "shape", x1.shape, x2.shape)

        return dpnp_power(x1, x2)

    return call_origin(numpy.power, x1, x2, **kwargs)


def prod(x1, **kwargs):
    """
    Calculate product of array elements over a given axis.

    .. seealso:: :func:`numpy.prod`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and not kwargs):
        return dpnp_prod(x1)

    return call_origin(numpy.prod, x1, **kwargs)


def remainder(x1, x2, **kwargs):
    """
    Return element-wise remainder of division.
    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        if (x1.size != x2.size):
            checker_throw_value_error("remainder", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("remainder", "shape", x1.shape, x2.shape)

        return dpnp_remainder(x1, x2)

    return call_origin(numpy.remainder, x1, x2, **kwargs)


def sign(x1, **kwargs):
    """
    Compute the absolute values element-wise.

    .. seealso:: :func:`numpy.sign`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and not kwargs):
        return dpnp_sign(x1)

    return call_origin(numpy.sign, x1, **kwargs)


def subtract(x1, x2, **kwargs):
    """
    Subtract arguments, element-wise.

    .. note::
        The 'out' parameter is currently not supported.

    Args:
        x1  (dpnp.dparray): array.
        x2  (dpnp.dparray): array.
        out (dpnp.dparray): Output array.

    Returns:
        dpnp.dparray: The difference of x1 and x2, element-wise.
        This is a scalar if both x1 and x2 are scalars.

    .. seealso:: :func:`numpy.subtract`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and not kwargs):
        if (x1.dtype != numpy.bool) and (x2.dtype != numpy.bool):
            if (x1.size != x2.size):
                checker_throw_value_error("subtract", "size", x1.size, x2.size)

            if (x1.shape != x2.shape):
                checker_throw_value_error("subtract", "shape", x1.shape, x2.shape)

            return dpnp_subtract(x1, x2)

    return call_origin(numpy.subtract, x1, x2, **kwargs)


def sum(x1, **kwargs):
    """
    Sum of array elements over a given axis.

    .. seealso:: :func:`numpy.sum`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray):
        axis = kwargs.get('axis')

        result = dpnp_sum(x1, axis)

        # scalar returned
        if result.shape == (1,):
            return result.dtype.type(result[0])

        return result

    return call_origin(numpy.sum, x1, **kwargs)


def true_divide(*args, **kwargs):
    """
    Provide a true division of the inputs, element-wise.

    .. seealso:: :func:`numpy.true_divide`

    """

    return dpnp.divide(*args, **kwargs)


def trunc(x1, **kwargs):
    """
    Compute the truncated value of the input, element-wise.

    .. seealso:: :func:`numpy.trunc`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and not kwargs):
        return dpnp_trunc(x1)

    return call_origin(numpy.trunc, x1, **kwargs)
