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

import numpy


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
    "float",
    "float_",
    "float16",
    "float32",
    "float64",
    "floating",
    "inexact",
    "int",
    "int_",
    "int32",
    "int64",
    "integer",
    "intc",
    "isscalar",
    "issubdtype",
    "issubsctype",
    "is_type_supported",
    "nan",
    "newaxis",
    "number",
    "signedinteger",
    "single",
    "singlecomplex",
    "void"
]

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


nan = numpy.nan
newaxis = None


def is_type_supported(obj_type):
    """
    Return True if type is supported by DPNP python level.
    """

    if obj_type == float64 or obj_type == float32 or obj_type == int64 or obj_type == int32:
        return True

    return False
