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
Type interface of the DPNP

Notes
-----
This module provides public type interface file for the library
"""

import numpy

__all__ = [
    "bool",
    "bool_",
    "complex128",
    "dtype",
    "float",
    "float32",
    "float64",
    "int",
    "int32",
    "int64",
    "integer",
    "isscalar",
    "nan",
    "newaxis"
]

bool = numpy.bool
bool_ = numpy.bool_
complex128 = numpy.complex128
dtype = numpy.dtype
float = numpy.float
float32 = numpy.float32
float64 = numpy.float64
int = numpy.int
int32 = numpy.int32
int64 = numpy.int64
integer = numpy.integer


def isscalar(obj):
    """
    Returns True if the type of `obj` is a scalar type.

    For full documentation refer to :obj:`numpy.isscalar`.

    """
    return numpy.isscalar(obj)


nan = numpy.nan
newaxis = None
