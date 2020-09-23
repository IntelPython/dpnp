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
Interface of the Mathematical part of the Intel NumPy

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
from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import checker_throw_value_error, checker_throw_type_error, use_origin_backend


__all__ = [
    "abs",
    "absolute",
    "add",
    "divide",
    "fabs",
    "floor",
    "fmax",
    "fmin",
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "power",
    "sign",
    "subtract",
    "sum",
    "true_divide"
]


def abs(x):
    return dpnp.absolute(x)


def absolute(input):
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------
    input : array_like
        Input array.

    Returns
    -------
    absolute : ndarray
        An ndarray containing the absolute value of each element in x.
    """

    dim_input = input.ndim

    if dim_input == 0:
        return numpy.abs(input)

    is_input_dparray = isinstance(input, dparray)

    if not use_origin_backend(input) and is_input_dparray:
        result = dpnp_absolute(input)

        # scalar returned
        if result.shape == (1,):
            return result.dtype.type(result[0])

        return result

    input1 = dpnp.asnumpy(input) if is_input_dparray else input

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.abs(input1)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def add(x1, x2, out=None):
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

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray):
        if out is not None:
            checker_throw_value_error("add", "out", out, None)

        if (x1.size != x2.size):
            checker_throw_value_error("add", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("add", "shape", x1.shape, x2.shape)

        return dpnp_add(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.add(input1, input2, out=out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def divide(x1, x2, out=None):
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

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray):
        if out is not None:
            checker_throw_value_error("divide", "out", out, None)

        if (x1.size != x2.size):
            checker_throw_value_error("divide", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("divide", "shape", x1.shape, x2.shape)

        return dpnp_divide(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.divide(input1, input2, out=out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def fabs(x1, out=None):
    """
    Compute the absolute values element-wise.

    .. seealso:: :func:`numpy.fabs`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray):
        if out is not None:
            checker_throw_value_error("fabs", "out", out, None)

        return dpnp_fabs(x1)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.fabs(input1, out=out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def floor(x1, out=None):
    """
    Compute the floor of the input, element-wise.

    Some spreadsheet programs calculate the “floor-towards-zero”, in other words floor(-2.5) == -2.
    dpNP instead uses the definition of floor where floor(-2.5) == -3.

    .. seealso:: :func:`numpy.floor`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray):
        if out is not None:
            checker_throw_value_error("floor", "out", out, None)

        return dpnp_floor(x1)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.floor(input1, out=out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def fmax(x1, x2, out=None):
    """
    Element-wise maximum of array elements.

    .. seealso:: :func:`numpy.fmax`

    """

    return dpnp.maximum(x1, x2, out)


def fmin(x1, x2, out=None):
    """
    Element-wise minimum of array elements.

    .. seealso:: :func:`numpy.fmin`

    """

    return dpnp.minimum(x1, x2, out)


def maximum(x1, x2, out=None):
    """
    Element-wise maximum of array elements.

    .. seealso:: :func:`numpy.maximum`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray):
        if out is not None:
            checker_throw_value_error("maximum", "out", out, None)

        if (x1.size != x2.size):
            checker_throw_value_error("maximum", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("maximum", "shape", x1.shape, x2.shape)

        return dpnp_maximum(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.maximum(input1, input2, out=out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def minimum(x1, x2, out=None):
    """
    Element-wise minimum of array elements.

    .. seealso:: :func:`numpy.minimum`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray):
        if out is not None:
            checker_throw_value_error("minimum", "out", out, None)

        if (x1.size != x2.size):
            checker_throw_value_error("minimum", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("minimum", "shape", x1.shape, x2.shape)

        return dpnp_minimum(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.minimum(input1, input2, out=out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def multiply(x1, x2, out=None):
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

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray):
        if out is not None:
            checker_throw_value_error("multiply", "out", out, None)

        if (x1.size != x2.size):
            checker_throw_value_error("multiply", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("multiply", "shape", x1.shape, x2.shape)

        return dpnp_multiply(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.multiply(input1, input2, out=out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def negative(x1):
    """
    Negative element-wise.

    .. seealso:: :func:`numpy.negative`

    """

    if (use_origin_backend(x1)):
        return numpy.negative(x1)

    if not isinstance(x1, dparray):
        return numpy.negative(x1)

    return dpnp_negative(x1)


def power(x1, x2, out=None, modulo=None):
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

    if (use_origin_backend(x1)):
        return numpy.power(x1, x2, out=out)

    if not (isinstance(x1, dparray) or isinstance(x2, dparray)):
        return numpy.power(x1, x2, out=out)

    if out is not None:
        checker_throw_value_error("power", "out", type(out), None)

    # modulo parameter is not supported at now
    if modulo is not None:
        checker_throw_value_error("__pow__", "modulo", modulo, None)

    if (x1.size != x2.size):
        checker_throw_value_error("power", "size", x1.size, x2.size)

    if (x1.shape != x2.shape):
        checker_throw_value_error("power", "shape", x1.shape, x2.shape)

    return dpnp_power(x1, x2)


def sign(x1, out=None):
    """
    Compute the absolute values element-wise.

    .. seealso:: :func:`numpy.sign`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray):
        if out is not None:
            checker_throw_value_error("sign", "out", out, None)

        return dpnp_sign(x1)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.sign(input1, out=out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def subtract(x1, x2, out=None):
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

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray):
        if out is not None:
            checker_throw_value_error("subtract", "out", out, None)

        if (x1.size != x2.size):
            checker_throw_value_error("subtract", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("subtract", "shape", x1.shape, x2.shape)

        return dpnp_subtract(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.subtract(input1, input2, out=out)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def sum(x1, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
    """
    Sum of array elements over a given axis.

    .. seealso:: :func:`numpy.sum`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1)
        and is_x1_dparray
        and (axis is None)
        and (dtype is None)
        and (out is None)
        and (keepdims is False)
        and (initial is 0)
        and (where is True)
        ):
        return dpnp_sum(x1)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.sum(input1, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def true_divide(x1, x2, out=None):
    """
    Provide a true division of the inputs, element-wise.

    .. seealso:: :func:`numpy.true_divide`

    """

    return dpnp.divide(x1, x2, out)
