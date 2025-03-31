# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
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

import functools

import dpctl
import dpctl.tensor as dpt
import numpy

import dpnp

from .dpnp_array import dpnp_array

# pylint: disable=no-name-in-module
from .dpnp_utils import get_usm_allocations

__all__ = [
    "bool",
    "bool_",
    "byte",
    "cdouble",
    "common_type",
    "complex128",
    "complex64",
    "complexfloating",
    "csingle",
    "double",
    "dtype",
    "e",
    "euler_gamma",
    "finfo",
    "float16",
    "float32",
    "float64",
    "floating",
    "iinfo",
    "inexact",
    "inf",
    "int_",
    "int8",
    "int16",
    "int32",
    "int64",
    "integer",
    "intc",
    "intp",
    "isdtype",
    "issubdtype",
    "is_type_supported",
    "longlong",
    "nan",
    "newaxis",
    "number",
    "pi",
    "short",
    "signedinteger",
    "single",
    "ubyte",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uintc",
    "uintp",
    "unsignedinteger",
    "ushort",
    "ulonglong",
]


# pylint: disable=invalid-name
# =============================================================================
# Data types (borrowed from NumPy)
# =============================================================================
bool = numpy.bool_
bool_ = numpy.bool_
byte = numpy.byte
cdouble = numpy.cdouble
complex128 = numpy.complex128
complex64 = numpy.complex64
complexfloating = numpy.complexfloating
csingle = numpy.csingle
double = numpy.double
dtype = numpy.dtype
float16 = numpy.float16
float32 = numpy.float32
float64 = numpy.float64
floating = numpy.floating
inexact = numpy.inexact
int_ = numpy.int_
int8 = numpy.int8
int16 = numpy.int16
int32 = numpy.int32
int64 = numpy.int64
integer = numpy.integer
intc = numpy.intc
intp = numpy.intp
longlong = numpy.longlong
number = numpy.number
short = numpy.short
signedinteger = numpy.signedinteger
single = numpy.single
ubyte = numpy.ubyte
uint8 = numpy.uint8
uint16 = numpy.uint16
uint32 = numpy.uint32
uint64 = numpy.uint64
uintc = numpy.uintc
uintp = numpy.uintp
unsignedinteger = numpy.unsignedinteger
ushort = numpy.ushort
ulonglong = numpy.ulonglong


# =============================================================================
# Constants (borrowed from NumPy)
# =============================================================================
e = numpy.e
euler_gamma = numpy.euler_gamma
inf = numpy.inf
nan = numpy.nan
newaxis = None
pi = numpy.pi


def common_type(*arrays):
    """
    Return a scalar type which is common to the input arrays.

    The return type will always be an inexact (i.e. floating point or complex)
    scalar type, even if all the arrays are integer arrays.
    If one of the inputs is an integer array, the minimum precision type
    that is returned is the default floating point data type for the device
    where the input arrays are allocated.

    For full documentation refer to :obj:`numpy.common_type`.

    Parameters
    ----------
    arrays: {dpnp.ndarray, usm_ndarray}
        Input arrays.

    Returns
    -------
    out: data type
        Data type object.

    See Also
    --------
    :obj:`dpnp.dtype` : Create a data type object.

    Examples
    --------
    >>> import dpnp as np
    >>> np.common_type(np.arange(2, dtype=np.float32))
    numpy.float32
    >>> np.common_type(np.arange(2, dtype=np.float32), np.arange(2))
    numpy.float64 # may vary
    >>> np.common_type(np.arange(4), np.array([45, 6.j]), np.array([45.0]))
    numpy.complex128 # may vary

    """

    if len(arrays) == 0:
        return (
            dpnp.float16
            if dpctl.select_default_device().has_aspect_fp16
            else dpnp.float32
        )

    dpnp.check_supported_arrays_type(*arrays)

    _, exec_q = get_usm_allocations(arrays)
    default_float_dtype = dpnp.default_float_type(sycl_queue=exec_q)
    dtypes = []
    for a in arrays:
        if not dpnp.issubdtype(a.dtype, dpnp.number):
            raise TypeError("can't get common type for non-numeric array")
        if dpnp.issubdtype(a.dtype, dpnp.integer):
            dtypes.append(default_float_dtype)
        else:
            dtypes.append(a.dtype)

    return functools.reduce(numpy.promote_types, dtypes).type


# pylint: disable=redefined-outer-name
def finfo(dtype):
    """
    Returns machine limits for floating-point data types.

    For full documentation refer to :obj:`numpy.finfo`.

    Parameters
    ----------
    dtype : dtype, dpnp_array
        Floating-point dtype or an array with floating point data type.
        If complex, the information is about its component data type.

    Returns
    -------
    out : finfo_object
        An object have the following attributes

        * bits: int
            number of bits occupied by dtype.
        * dtype: dtype
            real-valued floating-point data type.
        * eps: float
            difference between 1.0 and the next smallest representable
            real-valued floating-point number larger than 1.0 according
            to the IEEE-754 standard.
        * epsneg: float
            difference between 1.0 and the next smallest representable
            real-valued floating-point number smaller than 1.0 according to
            the IEEE-754 standard.
        * max: float
            largest representable real-valued number.
        * min: float
            smallest representable real-valued number.
        * precision: float
            the approximate number of decimal digits to which this kind of
            floating point type is precise.
        * resolution: float
            the approximate decimal resolution of this type.
        * tiny: float
            an alias for `smallest_normal`
        * smallest_normal: float
            smallest positive real-valued floating-point number with
            full precision.

    """
    if isinstance(dtype, dpnp_array):
        dtype = dtype.dtype
    return dpt.finfo(dtype)


# pylint: disable=redefined-outer-name
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
        * dtype: dtype
            integer data type.
        * max: int
            largest representable number.
        * min: int
            smallest representable number.

    """

    if isinstance(dtype, dpnp_array):
        dtype = dtype.dtype
    return dpt.iinfo(dtype)


def isdtype(dtype, kind):
    """
    Returns a boolean indicating whether a provided `dtype` is
    of a specified data type `kind`.

    Parameters
    ----------
    dtype : dtype
        The input dtype.
    kind : {dtype, str, tuple of dtypes or strs}
        The input dtype or dtype kind. Allowed dtype kinds are:

        * ``'bool'`` : boolean kind
        * ``'signed integer'`` : signed integer data types
        * ``'unsigned integer'`` : unsigned integer data types
        * ``'integral'`` : integer data types
        * ``'real floating'`` : real-valued floating-point data types
        * ``'complex floating'`` : complex floating-point data types
        * ``'numeric'`` : numeric data types

    Returns
    -------
    out : bool
        A boolean indicating whether a provided `dtype` is of a specified data
        type `kind`.

    See Also
    --------
    :obj:`dpnp.issubdtype` : Test if the first argument is a type code
                             lower/equal in type hierarchy.

    Examples
    --------
    >>> import dpnp as np
    >>> np.isdtype(np.float32, np.float64)
    False
    >>> np.isdtype(np.float32, "real floating")
    True
    >>> np.isdtype(np.complex128, ("real floating", "complex floating"))
    True

    """

    if isinstance(dtype, type):
        dtype = dpt.dtype(dtype)

    if isinstance(kind, type):
        kind = dpt.dtype(kind)
    elif isinstance(kind, tuple):
        kind = tuple(dpt.dtype(k) if isinstance(k, type) else k for k in kind)

    return dpt.isdtype(dtype, kind)


def issubdtype(arg1, arg2):
    """
    Returns ``True`` if the first argument is a type code lower/equal
    in type hierarchy.

    For full documentation refer to :obj:`numpy.issubdtype`.

    """

    return numpy.issubdtype(arg1, arg2)


def is_type_supported(obj_type):
    """Return True if type is supported by DPNP python level."""

    if obj_type in (float64, float32, int64, int32):
        return True
    return False
