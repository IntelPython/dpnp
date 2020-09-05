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
from dpnp.dpnp_utils import checker_throw_value_error, use_origin_backend


__all__ = [
    "equal",
    "greater",
    "greater_equal",
    "isclose",
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


def logical_and(x1, x2):
    """
    Return (x1 AND x2) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.logical_and(x1, x2)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_logical_and(x1, x2)

    return numpy.logical_and(x1, x2)


def logical_not(x1):
    """
    Return (NOT x1) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.logical_not(x1)

    if isinstance(x1, dparray):
        return dpnp_logical_not(x1)

    return numpy.logical_not(x1)


def logical_or(x1, x2):
    """
    Return (x1 OR x2) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.logical_or(x1, x2)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_logical_or(x1, x2)

    return numpy.logical_or(x1, x2)


def logical_xor(x1, x2):
    """
    Return (x1 XOR x2) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.logical_xor(x1, x2)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_logical_xor(x1, x2)

    return numpy.logical_xor(x1, x2)


def not_equal(x1, x2):
    """
    Return (x1 != x2) element-wise.

    """

    if (use_origin_backend(x1)):
        return numpy.not_equal(x1, x2)

    if isinstance(x1, dparray) or isinstance(x2, dparray):
        return dpnp_not_equal(x1, x2)

    return numpy.not_equal(x1, x2)
