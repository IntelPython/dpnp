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
Interface of the statistics function of the Intel NumPy

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
    'argsort'
]


def argsort(in_array1, axis=-1, kind=None, order=None):
    """
    Return an ndarray of indices that sort the array along the
    specified axis.  Masked values are filled beforehand to
    `fill_value`.
    Parameters
    ----------
    axis : int, optional
        Axis along which to sort. If None, the default, the flattened array
        is used.
        ..  versionchanged:: 1.13.0
            Previously, the default was documented to be -1, but that was
            in error. At some future date, the default will change to -1, as
            originally intended.
            Until then, the axis should be given explicitly when
            ``arr.ndim > 1``, to avoid a FutureWarning.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        The sorting algorithm used.
    order : list, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  Not all fields need be
        specified.
    endwith : {True, False}, optional
        Whether missing values (if any) should be treated as the largest values
        (True) or the smallest values (False)
        When the array contains unmasked values at the same extremes of the
        datatype, the ordering of these values and the masked values is
        undefined.
    fill_value : {var}, optional
        Value used internally for the masked values.
        If ``fill_value`` is not None, it supersedes ``endwith``.
    Returns
    -------
    index_array : ndarray, int
        Array of indices that sort `a` along the specified axis.
        In other words, ``a[index_array]`` yields a sorted `a`.
    See Also
    --------
    MaskedArray.sort : Describes sorting algorithms used.
    lexsort : Indirect stable sort with multiple keys.
    numpy.ndarray.sort : Inplace sort.
    Notes
    -----
    See `sort` for notes on the different sorting algorithms.
    Examples
    --------
    >>> a = np.ma.array([3,2,1], mask=[False, False, True])
    >>> a
    masked_array(data=[3, 2, --],
                 mask=[False, False,  True],
           fill_value=999999)
    >>> a.argsort()
    array([1, 0, 2])

    """

    is_dparray1 = isinstance(in_array1, dparray)

    if (not use_origin_backend(in_array1) and is_dparray1):
        if axis != -1:
            checker_throw_value_error("argsort", "axis", axis, -1)
        if kind is not None:
            checker_throw_value_error("argsort", "kind", type(kind), None)
        if order is not None:
            checker_throw_value_error("argsort", "order", type(order), None)

        return dpnp_argsort(in_array1)

    return numpy.argsort(in_array1)
