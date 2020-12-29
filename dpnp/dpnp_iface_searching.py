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
Interface of the searching function of the dpnp

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import numpy

from dpnp.dpnp_algo import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *


__all__ = [
    'argmax',
    'argmin'
]


def argmax(in_array1, axis=None, out=None):
    """
    Returns the indices of the maximum values along an axis.

    For full documentation refer to :obj:`numpy.argmax`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Parameter ``axis`` is supported only with default value ``None``.
    Parameter ``out`` is supported only with default value ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.argmin` : Returns the indices of the minimum values along an axis.
    :obj:`dpnp.amax` : The maximum value along a given axis.
    :obj:`dpnp.unravel_index` : Convert a flat index into an index tuple.
    :obj:`dpnp.take_along_axis` : Apply ``np.expand_dims(index_array, axis)``
                                  from argmax to an array as if by calling max.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(6).reshape((2, 3)) + 10
    >>> a.shape
    (2, 3)
    >>> [i for i in a]
    [10, 11, 12, 13, 14, 15]
    >>> np.argmax(a)
    5

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if axis is not None:
            checker_throw_value_error("argmax", "axis", type(axis), None)
        if out is not None:
            checker_throw_value_error("argmax", "out", type(out), None)

        result = dpnp_argmax(in_array1)

        # scalar returned
        if result.shape == (1,):
            return result.dtype.type(result[0])

        return result

    return numpy.argmax(in_array1, axis, out)


def argmin(in_array1, axis=None, out=None):
    """
    Returns the indices of the minimum values along an axis.

    For full documentation refer to :obj:`numpy.argmin`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Parameter ``axis`` is supported only with default value ``None``.
    Parameter ``out`` is supported only with default value ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.argmax` : Returns the indices of the maximum values along an axis.
    :obj:`dpnp.amin` : The minimum value along a given axis.
    :obj:`dpnp.unravel_index` : Convert a flat index into an index tuple.
    :obj:`dpnp.take_along_axis` : Apply ``np.expand_dims(index_array, axis)``
                                  from argmin to an array as if by calling min.

    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(6).reshape((2, 3)) + 10
    >>> a.shape
    (2, 3)
    >>> [i for i in a]
    [10, 11, 12, 13, 14, 15]
    >>> np.argmin(a)
    0

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if axis is not None:
            checker_throw_value_error("argmin", "axis", type(axis), None)
        if out is not None:
            checker_throw_value_error("argmin", "out", type(out), None)

        result = dpnp_argmin(in_array1)

        # scalar returned
        if result.shape == (1,):
            return result.dtype.type(result[0])

        return result

    return numpy.argmin(in_array1, axis, out)
