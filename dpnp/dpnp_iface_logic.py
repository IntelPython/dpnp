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
Interface of the Logic part of the Intel NumPy

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
    "equal",
    "greater",
    "greater_equal",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "not_equal"
]


def equal(x1, x2):
    """
    Return (x1 == x2) element-wise.

    Unlike `numpy.equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray or bool
        Output array of bools, or a single bool if x1 and x2 are scalars.

    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less

    """

    if (use_origin_backend(x1)):
        return numpy.equal(x1, x2)

    if isinstance(x1, dparray) and isinstance(x2, int):  # hack to satisfy current test system requirements
        return dpnp_equal(x1, x2)

    for input in (x1, x2):
        if not isinstance(input, dparray):
            raise TypeError(f"Intel NumPy equal(): Unsupported input={type(input)}")

    if x1.size != x2.size:
        utils.checker_throw_value_error("equal", "array sizes", x1.size, x2.size)

    if not x1.dtype == x2.dtype:
        raise TypeError(f"Intel NumPy equal(): Input types must be equal ({x1.dtype} != {x2.dtype})")

    if x1.shape != x2.shape:
        utils.checker_throw_value_error("equal", "array shapes", x1.shape, x2.shape)

    return dpnp_equal(x1, x2)


def greater(x1, x2):
    """
    Return (x1 > x2) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.greater(x1, x2)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_greater(x1, x2)

    return numpy.greater(x1, x2)


def greater_equal(x1, x2):
    """
    Return (x1 >= x2) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.greater_equal(x1, x2)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_greater_equal(x1, x2)

    return numpy.greater_equal(x1, x2)


def isclose(x1, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within a tolerance.

    """

    if (use_origin_backend(x1)):
        return numpy.greater_equal(x1, x2)

    if isinstance(x1, dparray) and isinstance(x2, int):  # hack to satisfy current test system requirements
        return dpnp_isclose(x1, x2, rtol, atol, equal_nan)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_isclose(x1, x2, rtol, atol, equal_nan)

    return numpy.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isfinite(in_array1, out=None, where=True, **kwargs):
    """
    Test element-wise for finiteness (not infinity or not Not a Number).

    The result is returned as a boolean array.

    Parameters
    ----------
    x : array_like
        Input values.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray, bool
        True where ``x`` is not positive infinity, negative infinity,
        or NaN; false otherwise.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    isinf, isneginf, isposinf, isnan

    Notes
    -----
    Not a Number, positive infinity and negative infinity are considered
    to be non-finite.

    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Also that positive infinity is not equivalent to negative infinity. But
    infinity is equivalent to positive infinity.  Errors result if the
    second argument is also supplied when `x` is a scalar input, or if
    first and second arguments have different shapes.

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if out is not None:
            checker_throw_value_error("isfinite", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("isfinite", "where", where, True)

        return dpnp_isfinite(in_array1)

    input1 = dpnp.asnumpy(in_array1) if is_dparray1 else in_array1

    return numpy.isfinite(input1, out, where, **kwargs)


def isinf(in_array1, out=None, where=True, **kwargs):
    """
    Test element-wise for positive or negative infinity.

    Returns a boolean array of the same shape as `x`, True where ``x ==
    +/-inf``, otherwise False.

    Parameters
    ----------
    x : array_like
        Input values
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : bool (scalar) or boolean ndarray
        True where ``x`` is positive or negative infinity, false otherwise.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    isneginf, isposinf, isnan, isfinite

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is supplied when the first
    argument is a scalar, or if the first and second arguments have
    different shapes.

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if out is not None:
            checker_throw_value_error("isinf", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("isinf", "where", where, True)

        return dpnp_isinf(in_array1)

    input1 = dpnp.asnumpy(in_array1) if is_dparray1 else in_array1

    return numpy.isinf(input1, out, where, **kwargs)


def isnan(in_array1, out=None, where=True, **kwargs):
    """
    Test element-wise for NaN and return result as a boolean array.

    Parameters
    ----------
    x : array_like
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or bool
        True where ``x`` is NaN, false otherwise.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    isinf, isneginf, isposinf, isfinite, isnat

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if out is not None:
            checker_throw_value_error("isnan", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("isnan", "where", where, True)

        return dpnp_isnan(in_array1)

    input1 = dpnp.asnumpy(in_array1) if is_dparray1 else in_array1

    return numpy.isnan(input1, out, where, **kwargs)


def less(x1, x2):
    """
    Return (x1 < x2) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.less(x1, x2)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_less(x1, x2)

    return numpy.less(x1, x2)


def less_equal(x1, x2):
    """
    Return (x1 <= x2) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.less_equal(x1, x2)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_less_equal(x1, x2)

    return numpy.less_equal(x1, x2)


def logical_and(x1, x2, out=None, where=True, **kwargs):
    """
    Compute the truth value of x1 AND x2 element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical AND operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    logical_or, logical_not, logical_xor
    bitwise_and

    """

    is_dparray1 = isinstance(x1, dparray)
    is_dparray2 = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_dparray1 and is_dparray2):
        if out is not None:
            checker_throw_value_error("logical_and", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("logical_and", "where", where, True)

        return dpnp_logical_and(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_dparray1 else x1
    input2 = dpnp.asnumpy(x2) if is_dparray1 else x2

    return numpy.logical_and(input1, input2, out, where, **kwargs)


def logical_not(x1, out=None, where=True, **kwargs):
    """
    Compute the truth value of NOT x element-wise.

    Parameters
    ----------
    x : array_like
        Logical NOT is applied to the elements of `x`.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : bool or ndarray of bool
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    logical_and, logical_or, logical_xor

    """

    is_dparray1 = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_dparray1):
        if out is not None:
            checker_throw_value_error("logical_not", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("logical_not", "where", where, True)

        return dpnp_logical_not(x1)

    input1 = dpnp.asnumpy(x1) if is_dparray1 else x1

    return numpy.logical_not(input1, out, where, **kwargs)


def logical_or(x1, x2, out=None, where=True, **kwargs):
    """
    Compute the truth value of x1 OR x2 element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical OR is applied to the elements of `x1` and `x2`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical OR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    logical_and, logical_not, logical_xor
    bitwise_or

    """

    is_dparray1 = isinstance(x1, dparray)
    is_dparray2 = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_dparray1 and is_dparray2):
        if out is not None:
            checker_throw_value_error("logical_or", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("logical_or", "where", where, True)

        return dpnp_logical_or(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_dparray1 else x1
    input2 = dpnp.asnumpy(x2) if is_dparray1 else x2

    return numpy.logical_or(input1, input2, out, where, **kwargs)


def logical_xor(x1, x2, out=None, where=True, **kwargs):
    """
    Compute the truth value of x1 XOR x2, element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical XOR is applied to the elements of `x1` and `x2`. If ``x1.shape != x2.shape``, they must be broadcastable to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : bool or ndarray of bool
        Boolean result of the logical XOR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    logical_and, logical_or, logical_not, bitwise_xor

    """

    is_dparray1 = isinstance(x1, dparray)
    is_dparray2 = isinstance(x2, dparray)

    if (not use_origin_backend(x1) and is_dparray1 and is_dparray2):
        if out is not None:
            checker_throw_value_error("logical_xor", "out", type(out), None)
        if where is not True:
            checker_throw_value_error("logical_xor", "where", where, True)

        return dpnp_logical_xor(x1, x2)

    input1 = dpnp.asnumpy(x1) if is_dparray1 else x1
    input2 = dpnp.asnumpy(x2) if is_dparray1 else x2

    return numpy.logical_xor(input1, input2, out, where, **kwargs)


def not_equal(x1, x2):
    """
    Return (x1 != x2) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.not_equal(x1, x2)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_not_equal(x1, x2)

    return numpy.not_equal(x1, x2)
