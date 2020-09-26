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
    "ceil",
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
    "multiply",
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


def _fallback(function, *args, **kwargs):
    """
    Call fallback function for unsupported cases
    """

    args_new = []
    for arg in args:
        argx = dpnp.asnumpy(arg) if isinstance(arg, dparray) else arg
        args_new.append(argx)

    # TODO need to put dparray memory into NumPy call
    result_fallback = function(*args_new, **kwargs)
    result = result_fallback
    if isinstance(result, numpy.ndarray):
        result = dparray(result_fallback.shape, dtype=result_fallback.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_fallback.item(i))

    return result


def abs(*args, **kwargs):
    """
    Calculate the absolute value element-wise.

    .. seealso:: :func:`numpy.add`

    """

    return dpnp.absolute(*args, **kwargs)


def absolute(input, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    result_numpy = numpy.absolute(input1, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def add(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    result_numpy = numpy.add(input1, input2, out=out, where=where, casting=casting,
                             order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def ceil(x1, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    """
    Compute  the ceiling of the input, element-wise.

    .. seealso:: :func:`numpy.ceil`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray):
        if out is not None:
            checker_throw_value_error("ceil", "out", out, None)

        return dpnp_ceil(x1)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.ceil(input1, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def divide(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    result_numpy = numpy.divide(input1, input2, out=out, where=where, casting=casting,
                                order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def fabs(x1, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    result_numpy = numpy.fabs(input1, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def floor(x1, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    result_numpy = numpy.floor(input1, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def floor_divide(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    """
    Compute the largest integer smaller or equal to the division of the inputs.

    .. seealso:: :func:`numpy.floor_divide`

    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and dtype is None):
        if out is not None:
            checker_throw_value_error("floor_divide", "out", out, None)

        return dpnp_floor_divide(x1)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.floor_divide(input1, input2, out=out, where=where,
                                      casting=casting, order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


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


def fmod(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and dtype is None):
        if out is not None:
            checker_throw_value_error("fmod", "out", out, None)

        if (x1.size != x2.size):
            checker_throw_value_error("fmod", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("fmod", "shape", x1.shape, x2.shape)

        return dpnp_fmod(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.fmod(input1, input2, out=out, where=where, casting=casting,
                              order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result_dtype = result_numpy.dtype
        if (dtype is not None):
            result_dtype = dtype
        result = dparray(result_numpy.shape, dtype=result_dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def maximum(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    result_numpy = numpy.maximum(input1, input2, out=out, where=where, casting=casting,
                                 order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def minimum(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    result_numpy = numpy.minimum(input1, input2, out=out, where=where, casting=casting,
                                 order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def mod(*args, **kwargs):
    """
    Compute element-wise remainder of division.

    Alias for :func:`dpnp.remainder`

    .. seealso:: :func:`numpy.mod`

    """

    return dpnp.remainder(*args, **kwargs)


def multiply(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    result_numpy = numpy.multiply(input1, input2, out=out, where=where, casting=casting,
                                  order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def negative(x1, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    """
    Negative element-wise.

    .. seealso:: :func:`numpy.negative`

    """

    if (use_origin_backend(x1)):
        return numpy.negative(x1, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

    if not isinstance(x1, dparray):
        return numpy.negative(x1, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

    return dpnp_negative(x1)


def power(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
        return numpy.power(x1, x2, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

    if not (isinstance(x1, dparray) or isinstance(x2, dparray)):
        return numpy.power(x1, x2, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

    if out is not None:
        checker_throw_value_error("power", "out", type(out), None)

    if (x1.size != x2.size):
        checker_throw_value_error("power", "size", x1.size, x2.size)

    if (x1.shape != x2.shape):
        checker_throw_value_error("power", "shape", x1.shape, x2.shape)

    return dpnp_power(x1, x2)


def prod(x1, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True):
    """
    Calculate product of array elements over a given axis.

    .. seealso:: :func:`numpy.prod`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1)
                and is_x1_dparray
                and (axis is None)
                and (dtype is None)
                and (out is None)
                and (keepdims is False)
                and (initial is 1)
                and (where is True)
            ):
        return dpnp_prod(x1)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.prod(input1, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def remainder(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    """
    Return element-wise remainder of division.
    """

    is_x1_dparray = isinstance(x1, dparray)
    is_x2_dparray = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and dtype is None):
        if out is not None:
            checker_throw_value_error("remainder", "out", out, None)

        if (x1.size != x2.size):
            checker_throw_value_error("remainder", "size", x1.size, x2.size)

        if (x1.shape != x2.shape):
            checker_throw_value_error("remainder", "shape", x1.shape, x2.shape)

        return dpnp_remainder(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1
    input2 = dpnp.asnumpy(x2) if is_x2_dparray else x2

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.remainder(input1, input2, out=out, where=where, casting=casting,
                                   order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result_dtype = result_numpy.dtype
        if (dtype is not None):
            result_dtype = dtype
        result = dparray(result_numpy.shape, dtype=result_dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def sign(x1, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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
    result_numpy = numpy.sign(input1, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def subtract(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
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

    if (not use_origin_backend(x1) and is_x1_dparray and is_x2_dparray and x1.dtype != numpy.bool and x2.dtype != numpy.bool):
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
    result_numpy = numpy.subtract(input1, input2, out=out, where=where, casting=casting,
                                  order=order, dtype=dtype, subok=subok)
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


def true_divide(*args, **kwargs):
    """
    Provide a true division of the inputs, element-wise.

    .. seealso:: :func:`numpy.true_divide`

    """

    return dpnp.divide(*args, **kwargs)


def trunc(x1, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    """
    Compute the truncated value of the input, element-wise.

    .. seealso:: :func:`numpy.trunc`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray):
        if out is not None:
            checker_throw_value_error("trunc", "out", out, None)

        return dpnp_trunc(x1)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.trunc(input1, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result
