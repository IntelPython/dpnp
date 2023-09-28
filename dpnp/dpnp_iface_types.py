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
Type interface of the DPNP

Notes
-----
This module provides public type interface file for the library
"""

import dpctl.tensor as dpt
import numpy

from dpnp.dpnp_array import dpnp_array

__all__ = [
    "bool",
    "bool_",
    "cdouble",
    "complex_",
    "complex128",
    "complex64",
    "complexfloating",
    "cfloat",
    "csingle",
    "double",
    "dtype",
    "e",
    "euler_gamma",
    "finfo",
    "float",
    "float_",
    "float16",
    "float32",
    "float64",
    "floating",
    "iinfo",
    "inexact",
    "Inf",
    "inf",
    "Infinity",
    "infty",
    "int",
    "int_",
    "int32",
    "int64",
    "integer",
    "intc",
    "isdtype",
    "isscalar",
    "issubdtype",
    "issubsctype",
    "is_type_supported",
    "NAN",
    "NaN",
    "nan",
    "newaxis",
    "NINF",
    "NZERO",
    "number",
    "pi",
    "PINF",
    "PZERO",
    "signedinteger",
    "single",
    "singlecomplex",
]


# =============================================================================
# Data types (borrowed from NumPy)
# =============================================================================
bool = numpy.bool_
bool_ = numpy.bool_
cdouble = numpy.cdouble
complex_ = numpy.complex_
complex128 = numpy.complex128
complex64 = numpy.complex64
complexfloating = numpy.complexfloating
cfloat = numpy.cfloat
csingle = numpy.csingle
double = numpy.double
dtype = numpy.dtype
float = numpy.float_
float_ = numpy.float_
float16 = numpy.float16
float32 = numpy.float32
float64 = numpy.float64
floating = numpy.floating
inexact = numpy.inexact
int = numpy.int_
int_ = numpy.int_
int32 = numpy.int32
int64 = numpy.int64
integer = numpy.integer
intc = numpy.intc
number = numpy.number
signedinteger = numpy.signedinteger
single = numpy.single
singlecomplex = numpy.singlecomplex


# =============================================================================
# Constants (borrowed from NumPy)
# =============================================================================
e = numpy.e
euler_gamma = numpy.euler_gamma
Inf = numpy.Inf
inf = numpy.inf
Infinity = numpy.Infinity
infty = numpy.infty
NAN = numpy.NAN
NaN = numpy.NaN
nan = numpy.nan
newaxis = None
NINF = numpy.NINF
NZERO = numpy.NZERO
pi = numpy.pi
PINF = numpy.PINF
PZERO = numpy.PZERO


def finfo(dtype):
    """
    Returns machine limits for floating-point data types.

    For full documentation refer to :obj:`numpy.finfo`.

    Parameters
    ----------
    dtype : dtype, dpnp_array)
        Floating-point dtype or an array with floating point data type.
        If complex, the information is about its component data type.

    Returns
    -------
    out : finfo_object
        An object have the following attributes
        * bits: int
            number of bits occupied by dtype.
        * eps: float
            difference between 1.0 and the next smallest representable
            real-valued floating-point number larger than 1.0 according
            to the IEEE-754 standard.
        * max: float
            largest representable real-valued number.
        * min: float
            smallest representable real-valued number.
        * smallest_normal: float
            smallest positive real-valued floating-point number with
            full precision.
        * dtype: dtype
            real-valued floating-point data type.

    """
    if isinstance(dtype, dpnp_array):
        dtype = dtype.dtype
    return dpt.finfo(dtype)


def isdtype(dtype_, kind):
    """Returns a boolean indicating whether a provided `dtype` is of a specified data type `kind`."""
    return dpt.isdtype(dtype_, kind)


def iinfo(dtype):
    """
    Returns machine limits for integer data types.

    For full documentation refer to :obj:`numpy.iinfo`.

    Parameters
    ----------
    dtype : dtype, dpnp_array
        Integer dtype or an array with integer dtype.

    Returns
    -------
    out : iinfo_object
        An object with the following attributes
        * bits: int
            number of bits occupied by the data type
        * max: int
            largest representable number.
        * min: int
            smallest representable number.
        * dtype: dtype
            integer data type.
    """
    if isinstance(dtype, dpnp_array):
        dtype = dtype.dtype
    return dpt.iinfo(dtype)


def isscalar(obj):
    """
    Returns True if the type of `obj` is a scalar type.

    For full documentation refer to :obj:`numpy.isscalar`.

    """
    return numpy.isscalar(obj)


def issubdtype(arg1, arg2):
    """
    Returns True if first argument is a typecode lower/equal in type hierarchy.

    For full documentation refer to :obj:`numpy.issubdtype`.

    """

    return numpy.issubdtype(arg1, arg2)


def issubsctype(arg1, arg2):
    """
    Determine if the first argument is a subclass of the second argument.

    For full documentation refer to :obj:`numpy.issubsctype`.

    """

    return numpy.issubsctype(arg1, arg2)


def is_type_supported(obj_type):
    """Return True if type is supported by DPNP python level."""

    if (
        obj_type == float64
        or obj_type == float32
        or obj_type == int64
        or obj_type == int32
    ):
        return True

    return False
