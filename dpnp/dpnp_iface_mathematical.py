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


from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *

import dpnp
import numpy


__all__ = [
    "abs",
    "absolute",
    "add",
    "around",
    "ceil",
    "conj",
    "conjugate",
    "convolve",
    "copysign",
    "cross",
    "cumprod",
    "cumsum",
    "diff",
    "divide",
    "ediff1d",
    "fabs",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gradient",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "nancumprod",
    "nancumsum",
    "nanprod",
    "nansum",
    "negative",
    "power",
    "prod",
    "remainder",
    "round_",
    "sign",
    "subtract",
    "sum",
    "trapz",
    "true_divide",
    "trunc"
]


def abs(*args, **kwargs):
    """
    Calculate the absolute value element-wise.

    For full documentation refer to :obj:`numpy.absolute`.

    Notes
    -----
    "obj:`dpnp.abs` is a shorthand for :obj:`dpnp.absolute`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.2, 1.2])
    >>> result = np.abs(a)
    >>> [x for x in result]
    [1.2, 1.2]

    """

    return dpnp.absolute(*args, **kwargs)


def absolute(x1, **kwargs):
    """
    Calculate the absolute value element-wise.

    For full documentation refer to :obj:`numpy.absolute`.

    .. seealso:: :obj:`dpnp.abs` : Calculate the absolute value element-wise.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
        Dimension of input array is limited by ``x1.ndim != 0``.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.2, 1.2])
    >>> result = np.absolute(a)
    >>> [x for x in result]
    [1.2, 1.2]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        if not x1_desc.ndim:
            pass
        else:
            result = dpnp_absolute(x1_desc).get_pyobj()

            return result

    return call_origin(numpy.absolute, x1, **kwargs)


def add(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Add arguments element-wise.

    For full documentation refer to :obj:`numpy.add`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([1, 2, 3])
    >>> result = np.add(a, b)
    >>> [x for x in result]
    [2, 4, 6]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            return dpnp_add(x1_desc, x2_desc, dtype, out_desc, where).get_pyobj()

    return call_origin(numpy.add, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def around(x1, decimals=0, out=None):
    """
    Evenly round to the given number of decimals.

    For full documentation refer to :obj:`numpy.around`.

    Limitations
    -----------
    Parameters ``x1`` is supported as :obj:`dpnp.ndarray`.
    Parameters ``decimals`` and ``out`` are supported with their default values.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.around([0.37, 1.64])
    array([0.,  2.])
    >>> np.around([0.37, 1.64], decimals=1)
    array([0.4,  1.6])
    >>> np.around([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
    array([0.,  2.,  2.,  4.,  4.])
    >>> np.around([1,2,3,11], decimals=1) # ndarray of ints is returned
    array([ 1,  2,  3, 11])
    >>> np.around([1,2,3,11], decimals=-1)
    array([ 0,  0,  0, 10])

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if out is not None:
            pass
        elif decimals != 0:
            pass
        else:
            return dpnp_around(x1_desc, decimals).get_pyobj()

    return call_origin(numpy.around, x1, decimals=decimals, out=out)


def ceil(x1, out=None, **kwargs):
    """
    Compute  the ceiling of the input, element-wise.

    For full documentation refer to :obj:`numpy.ceil`.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.floor` : Return the floor of the input, element-wise.
    :obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> result = np.ceil(a)
    >>> [x for x in result]
    [-1.0, -1.0, -0.0, 1.0, 2.0, 2.0, 2.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_ceil(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.ceil, x1, out=out, **kwargs)


def conjugate(x1, **kwargs):
    """
    Return the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by changing the
    sign of its imaginary part.

    For full documentation refer to :obj:`numpy.conjugate`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.conjugate(1+2j)
    (1-2j)

    >>> x = np.eye(2) + 1j * np.eye(2)
    >>> np.conjugate(x)
    array([[ 1.-1.j,  0.-0.j],
           [ 0.-0.j,  1.-1.j]])

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        return dpnp_conjugate(x1_desc)

    return call_origin(numpy.conjugate, x1, **kwargs)


conj = conjugate


def convolve(a, v, mode='full'):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    For full documentation refer to :obj:`numpy.convolve`.

    Examples
    --------
    >>> ca = dpnp.convolve([1, 2, 3], [0, 1, 0.5])
    >>> print(ca)
    [0. , 1. , 2.5, 4. , 1.5]

    """

    return call_origin(numpy.convolve, a=a, v=v, mode=mode)


def copysign(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Change the sign of x1 to that of x2, element-wise.

    For full documentation refer to :obj:`numpy.copysign`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> result = np.copysign(np.array([1, -2, 6, -9]), np.array([-1, -1, 1, 1]))
    >>> [x for x in result]
    [-1.0, -2.0, 6.0, 9.0]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            return dpnp_copysign(x1_desc, x2_desc, dtype=dtype, out=out, where=where).get_pyobj()

    return call_origin(numpy.copysign, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """
    Return the cross product of two (arrays of) vectors.

    For full documentation refer to :obj:`numpy.cross`.

    Limitations
    -----------
        Parameters ``x1`` and ``x2`` are supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Sizes of input arrays are limited by ``x1.size == 3 and x2.size == 3``.
        Shapes of input arrays are limited by ``x1.shape == (3,) and x2.shape == (3,)``.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> result = np.cross(x, y)
    >>> [x for x in result]
    [-3,  6, -3]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc:
        if x1_desc.size != 3 or x2_desc.size != 3:
            pass
        elif x1_desc.shape != (3,) or x2_desc.shape != (3,):
            pass
        elif axisa != -1:
            pass
        elif axisb != -1:
            pass
        elif axisc != -1:
            pass
        elif axis is not None:
            pass
        else:
            return dpnp_cross(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.cross, x1, x2, axisa, axisb, axisc, axis)


def cumprod(x1, **kwargs):
    """
    Return the cumulative product of elements along a given axis.

    For full documentation refer to :obj:`numpy.cumprod`.

    Limitations
    -----------
        Parameter ``x`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> result = np.cumprod(a)
    >>> [x for x in result]
    [1, 2, 6]
    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> result = np.cumprod(b)
    >>> [x for x in result]
    [1, 2, 6, 24, 120, 720]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        return dpnp_cumprod(x1_desc).get_pyobj()

    return call_origin(numpy.cumprod, x1, **kwargs)


def cumsum(x1, **kwargs):
    """
    Return the cumulative sum of the elements along a given axis.

    For full documentation refer to :obj:`numpy.cumsum`.

    Limitations
    -----------
        Parameter ``x`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 4])
    >>> result = np.cumsum(a)
    >>> [x for x in result]
    [1, 2, 7]
    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> result = np.cumsum(b)
    >>> [x for x in result]
    [1, 2, 6, 10, 15, 21]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        return dpnp_cumsum(x1_desc).get_pyobj()

    return call_origin(numpy.cumsum, x1, **kwargs)


def diff(x1, n=1, axis=-1, prepend=None, append=None):
    """
    Calculate the n-th discrete difference along the given axis.

    For full documentation refer to :obj:`numpy.diff`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameters ``axis``, ``prepend`` and ``append`` are supported only with default values.
    Otherwise the function will be executed sequentially on CPU.
    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if not isinstance(n, int):
            pass
        elif n < 1:
            pass
        elif axis != -1:
            pass
        elif prepend is not None:
            pass
        elif append is not None:
            pass
        else:
            return dpnp_diff(x1_desc, n)

    return call_origin(numpy.diff, x1, n, axis, prepend, append)


def divide(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Divide arguments element-wise.

    For full documentation refer to :obj:`numpy.divide`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> result = np.divide(np.array([1, -2, 6, -9]), np.array([-2, -2, -2, -2]))
    >>> [x for x in result]
    [-0.5, 1.0, -3.0, 4.5]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            return dpnp_divide(x1_desc, x2_desc, dtype=dtype, out=out, where=where).get_pyobj()

    return call_origin(numpy.divide, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def ediff1d(x1, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    For full documentation refer to :obj:`numpy.ediff1d`.

    Limitations
    -----------
        Parameter ``x1``is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``to_end`` and ``to_begin`` are currently supported only with default values `None`.
        Otherwise the function will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    ..seealso:: :obj:`dpnp.diff` : Calculate the n-th discrete difference along the given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 4, 7, 0])
    >>> result = np.ediff1d(a)
    >>> [x for x in result]
    [1, 2, 3, -7]
    >>> b = np.array([[1, 2, 4], [1, 6, 24]])
    >>> result = np.ediff1d(b)
    >>> [x for x in result]
    [1, 2, -3, 5, 18]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if to_begin is not None:
            pass
        elif to_end is not None:
            pass
        else:
            return dpnp_ediff1d(x1_desc).get_pyobj()

    return call_origin(numpy.ediff1d, x1, to_end=to_end, to_begin=to_begin)


def fabs(x1, **kwargs):
    """
    Compute the absolute values element-wise.

    For full documentation refer to :obj:`numpy.fabs`.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.abs` : Calculate the absolute value element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> result = np.fabs(np.array([1, -2, 6, -9]))
    >>> [x for x in result]
    [1.0, 2.0, 6.0, 9.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_fabs(x1_desc).get_pyobj()

    return call_origin(numpy.fabs, x1, **kwargs)


def floor(x1, out=None, **kwargs):
    """
    Round a number to the nearest integer toward minus infinity.

    For full documentation refer to :obj:`numpy.floor`.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
        :obj:`dpnp.ceil` : Compute  the ceiling of the input, element-wise.
        :obj:`dpnp.trunc` : Return the truncated value of the input, element-wise.

    Notes
    -----
        Some spreadsheet programs calculate the "floor-towards-zero", in other words floor(-2.5) == -2.
        dpNP instead uses the definition of floor where floor(-2.5) == -3.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> result = np.floor(a)
    >>> [x for x in result]
    [-2.0, -2.0, -1.0, 0.0, 1.0, 1.0, 2.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_floor(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.floor, x1, out=out, **kwargs)


def floor_divide(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Compute the largest integer smaller or equal to the division of the inputs.

    For full documentation refer to :obj:`numpy.floor_divide`.

    Limitations
    -----------
        Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
        Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.reminder` : Remainder complementary to floor_divide.
    :obj:`dpnp.divide` : Standard division.
    :obj:`dpnp.floor` : Round a number to the nearest integer toward minus infinity.
    :obj:`dpnp.ceil` : Round a number to the nearest integer toward infinity.

    Examples
    --------
    >>> import dpnp as np
    >>> result = np.floor_divide(np.array([1, -1, -2, -9]), np.array([-2, -2, -2, -2]))
    >>> [x for x in result]
    [-1, 0, 1, 4]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif x2_is_scalar and not x2_desc:
            pass
        elif x1_desc and x2_desc and x1_desc.size != x2_desc.size:
            pass
        elif x1_desc and x2_desc and x1_desc.shape != x2_desc.shape:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        elif x1_is_scalar and x2_desc.ndim > 1:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            return dpnp_floor_divide(x1_desc, x2_desc, dtype, out_desc, where).get_pyobj()

    return call_origin(numpy.floor_divide, x1, x2, out=out, where=where, dtype=dtype, **kwargs)


def fmax(*args, **kwargs):
    """
    Element-wise maximum of array elements.

    For full documentation refer to :obj:`numpy.fmax`.

    See Also
    --------
    :obj:`dpnp.maximum` : Element-wise maximum of array elements.
    :obj:`dpnp.fmin` : Element-wise minimum of array elements.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.

    Notes
    -----
    This function works the same as :obj:`dpnp.maximum`

    """

    return dpnp.maximum(*args, **kwargs)


def fmin(*args, **kwargs):
    """
    Element-wise minimum of array elements.

    For full documentation refer to :obj:`numpy.fmin`.

    See Also
    --------
    :obj:`dpnp.maximum` : Element-wise maximum of array elements.
    :obj:`dpnp.fmax` : Element-wise maximum of array elements.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.

    Notes
    -----
    This function works the same as :obj:`dpnp.minimum`

    """

    return dpnp.minimum(*args, **kwargs)


def fmod(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Calculate the element-wise remainder of division.

    For full documentation refer to :obj:`numpy.fmod`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.reminder` : Remainder complementary to floor_divide.
    :obj:`dpnp.divide` : Standard division.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([2, -3, 4, 5, -4.5])
    >>> b = np.array([2, 2, 2, 2, 2])
    >>> result = np.fmod(a, b)
    >>> [x for x in result]
    [0.0, -1.0, 0.0, 1.0, -0.5]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1.ndim == 0:
            pass
        elif x2_desc and x2.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            return dpnp_fmod(x1_desc, x2_desc, dtype, out_desc, where).get_pyobj()

    return call_origin(numpy.fmod, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def gradient(x1, *varargs, **kwargs):
    """
    Return the gradient of an array.

    For full documentation refer to :obj:`numpy.gradient`.

    Limitations
    -----------
        Parameter ``y1`` is supported as :obj:`dpnp.ndarray`.
        Argument ``varargs[0]`` is supported as `int`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Example
    -------
    >>> import dpnp as np
    >>> y = np.array([1, 2, 4, 7, 11, 16], dtype=float)
    >>> result = np.gradient(y)
    >>> [x for x in result]
    [1.0, 1.5, 2.5, 3.5, 4.5, 5.0]
    >>> result = np.gradient(y, 2)
    >>> [x for x in result]
    [0.5, 0.75, 1.25, 1.75, 2.25, 2.5]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        if len(varargs) > 1:
            pass
        elif len(varargs) == 1 and not isinstance(varargs[0], int):
            pass
        else:
            if len(varargs) == 0:
                return dpnp_gradient(x1_desc).get_pyobj()

            return dpnp_gradient(x1_desc, varargs[0]).get_pyobj()

    return call_origin(numpy.gradient, x1, *varargs, **kwargs)


def maximum(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Element-wise maximum of array elements.

    For full documentation refer to :obj:`numpy.maximum`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.fmax` : Element-wise maximum of array elements.
    :obj:`dpnp.fmin` : Element-wise minimum of array elements.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.

    Example
    -------
    >>> import dpnp as np
    >>> result = np.fmax(np.array([-2, 3, 4]), np.array([1, 5, 2]))
    >>> [x for x in result]
    [1, 5, 4]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            return dpnp_maximum(x1_desc, x2_desc, dtype=dtype, out=out, where=where).get_pyobj()

    return call_origin(numpy.maximum, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def minimum(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Element-wise minimum of array elements.

    For full documentation refer to :obj:`numpy.minimum`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.fmax` : Element-wise maximum of array elements.
    :obj:`dpnp.fmin` : Element-wise minimum of array elements.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.

    Example
    -------
    >>> import dpnp as np
    >>> result = np.fmin(np.array([-2, 3, 4]), np.array([1, 5, 2]))
    >>> [x for x in result]
    [-2, 3, 2]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            return dpnp_minimum(x1_desc, x2_desc, dtype=dtype, out=out, where=where).get_pyobj()

    return call_origin(numpy.minimum, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def mod(*args, **kwargs):
    """
    Compute element-wise remainder of division.

    For full documentation refer to :obj:`numpy.mod`.

    See Also
    --------
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division
    :obj:`dpnp.reminder` : Remainder complementary to floor_divide.
    :obj:`dpnp.divide` : Standard division.

    Notes
    -----
    This function works the same as :obj:`dpnp.remainder`.

    """

    return dpnp.remainder(*args, **kwargs)


def modf(x1, **kwargs):
    """
    Return the fractional and integral parts of an array, element-wise.

    For full documentation refer to :obj:`numpy.modf`.

    Limitations
    -----------
        Parameter ``x`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2])
    >>> result = np.modf(a)
    >>> [[x for x in y] for y in result ]
    [[1.0, 2.0], [0.0, 0.0]]


    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        return dpnp_modf(x1_desc)

    return call_origin(numpy.modf, x1, **kwargs)


def multiply(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Multiply arguments element-wise.

    For full documentation refer to :obj:`numpy.multiply`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> result = np.multiply(a, a)
    >>> [x for x in result]
    [1, 4, 9, 16, 25]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            return dpnp_multiply(x1_desc, x2_desc, dtype=dtype, out=out, where=where).get_pyobj()

    return call_origin(numpy.multiply, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def nancumprod(x1, **kwargs):
    """
    Return the cumulative product of array elements over a given axis treating Not a Numbers (NaNs) as one.

    For full documentation refer to :obj:`numpy.nancumprod`.

    Limitations
    -----------
        Parameter ``x`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.cumprod` : Return the cumulative product of elements along a given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1., np.nan])
    >>> result = np.nancumprod(a)
    >>> [x for x in result]
    [1.0, 1.0]
    >>> b = np.array([[1., 2., np.nan], [4., np.nan, 6.]])
    >>> result = np.nancumprod(b)
    >>> [x for x in result]
    [1.0, 2.0, 2.0, 8.0, 8.0, 48.0]


    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        return dpnp_nancumprod(x1_desc).get_pyobj()

    return call_origin(numpy.nancumprod, x1, **kwargs)


def nancumsum(x1, **kwargs):
    """
    Return the cumulative sum of the elements along a given axis.

    For full documentation refer to :obj:`numpy.nancumsum`.

    Limitations
    -----------
        Parameter ``x`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`dpnp.cumsum` : Return the cumulative sum of the elements along a given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1., np.nan])
    >>> result = np.nancumsum(a)
    >>> [x for x in result]
    [1.0, 1.0]
    >>> b = np.array([[1., 2., np.nan], [4., np.nan, 6.]])
    >>> result = np.nancumprod(b)
    >>> [x for x in result]
    [1.0, 3.0, 3.0, 7.0, 7.0, 13.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        return dpnp_nancumsum(x1_desc).get_pyobj()

    return call_origin(numpy.nancumsum, x1, **kwargs)


def nanprod(x1, **kwargs):
    """
    Calculate prod() function treating 'Not a Numbers' (NaN) as ones.

    For full documentation refer to :obj:`numpy.nanprod`.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.nanprod(np.array([1, 2]))
    2
    >>> np.nanprod(np.array([[1, 2], [3, 4]]))
    24

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        return dpnp_nanprod(x1_desc).get_pyobj()

    return call_origin(numpy.nanprod, x1, **kwargs)


def nansum(x1, **kwargs):
    """
    Calculate sum() function treating 'Not a Numbers' (NaN) as zero.

    For full documentation refer to :obj:`numpy.nansum`.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.nansum(np.array([1, 2]))
    3
    >>> np.nansum(np.array([[1, 2], [3, 4]]))
    10

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        result_obj = dpnp_nansum(x1_desc).get_pyobj()
        result = dpnp.convert_single_elem_array_to_scalar(result_obj)
        return result

    return call_origin(numpy.nansum, x1, **kwargs)


def negative(x1, **kwargs):
    """
    Negative element-wise.

    For full documentation refer to :obj:`numpy.negative`.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    .. see also: :obj:`dpnp.copysign` : Change the sign of x1 to that of x2, element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> result = np.negative([1, -1])
    >>> [x for x in result]
    [-1, 1]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        return dpnp_negative(x1_desc).get_pyobj()

    return call_origin(numpy.negative, x1, **kwargs)


def power(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    First array elements raised to powers from second array, element-wise.

    For full documentation refer to :obj:`numpy.power`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.fmax` : Element-wise maximum of array elements.
    :obj:`dpnp.fmin` : Element-wise minimum of array elements.
    :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.


    Example
    -------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> b = np.array([2, 2, 2, 2, 2])
    >>> result = np.power(a, b)
    >>> [x for x in result]
    [1, 4, 9, 16, 25]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif dtype is not None:
            pass
        elif not where:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            return dpnp_power(x1_desc, x2_desc, dtype, out_desc, where).get_pyobj()

    return call_origin(numpy.power, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def prod(x1, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    """
    Calculate product of array elements over a given axis.

    For full documentation refer to :obj:`numpy.prod`.

    Limitations
    -----------
        Parameter ``where`` is unsupported.
        Input array data types are limited by DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.prod(np.array([[1, 2], [3, 4]]))
    24
    >>> np.prod(np.array([1, 2]))
    2

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if where is not True:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            result_obj = dpnp_prod(x1_desc, axis, dtype, out_desc, keepdims, initial, where).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj, keepdims)

            return result

    return call_origin(numpy.prod, x1, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)


def remainder(x1, x2, out=None, where=True, dtype=None, **kwargs):
    """
    Return element-wise remainder of division.

    For full documentation refer to :obj:`numpy.remainder`.

    Limitations
    -----------
        Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
        Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.
        Parameters ``x1`` and ``x2`` are supported with equal sizes and shapes.

    See Also
    --------
        :obj:`dpnp.fmod` : Calculate the element-wise remainder of division.
        :obj:`dpnp.divide` : Standard division.
        :obj:`dpnp.floor` : Round a number to the nearest integer toward minus infinity.

    Example
    -------
    >>> import dpnp as np
    >>> result = np.remainder(np.array([4, 7]), np.array([2, 3]))
    >>> [x for x in result]
    [0, 1]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)

    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif x2_is_scalar and not x2_desc:
            pass
        elif x1_desc and x2_desc and x1_desc.size != x2_desc.size:
            pass
        elif x1_desc and x2_desc and x1_desc.shape != x2_desc.shape:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        elif x1_is_scalar and x2_desc.ndim > 1:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            return dpnp_remainder(x1_desc, x2_desc, dtype, out_desc, where).get_pyobj()

    return call_origin(numpy.remainder, x1, x2, out=out, where=where, dtype=dtype, **kwargs)


def round_(a, decimals=0, out=None):
    """
    Round an array to the given number of decimals.

    For full documentation refer to :obj:`numpy.round_`.

    See Also
    --------
        :obj:`dpnp.around` : equivalent function; see for details.

    """

    return around(a, decimals, out)


def sign(x1, **kwargs):
    """
    Returns an element-wise indication of the sign of a number.

    For full documentation refer to :obj:`numpy.sign`.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> result = np.sign(np.array([-5., 4.5]))
    >>> [x for x in result]
    [-1.0, 1.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        return dpnp_sign(x1_desc).get_pyobj()

    return call_origin(numpy.sign, x1, **kwargs)


def subtract(x1, x2, dtype=None, out=None, where=True, **kwargs):
    """
    Subtract arguments, element-wise.

    For full documentation refer to :obj:`numpy.subtract`.

    Limitations
    -----------
    Parameters ``x1`` and ``x2`` are supported as either :obj:`dpnp.ndarray` or scalar.
    Parameters ``dtype``, ``out`` and ``where`` are supported with their default values.
    Keyword arguments ``kwargs`` are currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Example
    -------
    >>> import dpnp as np
    >>> result = np.subtract(np.array([4, 3]), np.array([2, 7]))
    >>> [x for x in result]
    [2, -4]

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)
    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)
    if x1_desc and x2_desc and not kwargs:
        if not x1_desc and not x1_is_scalar:
            pass
        elif not x2_desc and not x2_is_scalar:
            pass
        elif x1_is_scalar and x2_is_scalar:
            pass
        elif x1_desc and x1_desc.ndim == 0:
            pass
        elif x1_desc and x1_desc.dtype == numpy.bool:
            pass
        elif x2_desc and x2_desc.ndim == 0:
            pass
        elif x2_desc and x2_desc.dtype == numpy.bool:
            pass
        elif dtype is not None:
            pass
        elif out is not None:
            pass
        elif not where:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            return dpnp_subtract(x1_desc, x2_desc, dtype=dtype, out=out_desc, where=where).get_pyobj()

    return call_origin(numpy.subtract, x1, x2, dtype=dtype, out=out, where=where, **kwargs)


def sum(x1, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=True):
    """
    Sum of array elements over a given axis.

    For full documentation refer to :obj:`numpy.sum`.

    Limitations
    -----------
        Parameter `where`` is unsupported.
        Input array data types are limited by DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> np.sum(np.array([1, 2, 3, 4, 5]))
    15
    >>> result = np.sum([[0, 1], [0, 5]], axis=0)
    [0, 6]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if where is not True:
            pass
        else:
            out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
            result_obj = dpnp_sum(x1_desc, axis, dtype, out_desc, keepdims, initial, where).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj, keepdims)

            return result

    return call_origin(numpy.sum, x1, axis=axis, dtype=dtype, out=out, keepdims=keepdims, initial=initial, where=where)


def trapz(y1, x1=None, dx=1.0, axis=-1):
    """
    Integrate along the given axis using the composite trapezoidal rule.

    For full documentation refer to :obj:`numpy.trapz`.

    Limitations
    -----------
        Parameters ``y`` and ``x`` are supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 6, 8])
    >>> np.trapz(a)
    4.0
    >>> np.trapz(a, x=b)
    8.0
    >>> np.trapz(a, dx=2)
    8.0

    """

    y_desc = dpnp.get_dpnp_descriptor(y1)
    if y_desc:
        if y_desc.ndim > 1:
            pass
        else:
            x_obj = dpnp.empty(y_desc.shape, dtype=y_desc.dtype) if x1 is None else x1
            x_desc = dpnp.get_dpnp_descriptor(x_obj)
            if x_desc:
                pass
            elif y_desc.size != x_desc.size:
                pass
            elif y_desc.shape != x_desc.shape:
                pass
            else:
                return dpnp_trapz(y_desc, x_desc, dx).get_pyobj()

    return call_origin(numpy.trapz, y1, x1, dx, axis)


def true_divide(*args, **kwargs):
    """
    Provide a true division of the inputs, element-wise.

    For full documentation refer to :obj:`numpy.true_divide`.

    See Also
    --------
    .. seealso:: :obj:`dpnp.divide` : Standard division.

    Notes
    -----
    This function works the same as :obj:`dpnp.divide`.


    """

    return dpnp.divide(*args, **kwargs)


def trunc(x1, out=None, **kwargs):
    """
    Compute the truncated value of the input, element-wise.

    For full documentation refer to :obj:`numpy.trunc`.

    Limitations
    -----------
        Parameter ``x1`` is supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
        :obj:`dpnp.floor` : Round a number to the nearest integer toward minus infinity.
        :obj:`dpnp.ceil` : Round a number to the nearest integer toward infinity.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
    >>> result = np.trunc(a)
    >>> [x for x in result]
    [-1.0, -1.0, -0.0, 0.0, 1.0, 1.0, 2.0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
        return dpnp_trunc(x1_desc, out_desc).get_pyobj()

    return call_origin(numpy.trunc, x1, out=out, **kwargs)
