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
Interface of the sorting function of the dpnp

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
from dpnp.dpnp_utils import *

import dpnp


__all__ = [
    'argsort',
    'partition',
    'searchsorted',
    'sort'
]


def argsort(in_array1, axis=-1, kind=None, order=None):
    """
    Returns the indices that would sort an array.

    For full documentation refer to :obj:`numpy.argsort`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Prameters ``axis`` is supported only with default value ``-1``.
    Prameters ``kind`` is supported only with default value ``None``.
    Prameters ``order`` is supported only with default value ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.sort` : Describes sorting algorithms used.
    :obj:`dpnp.lexsort` : Indirect stable sort with multiple keys.
    :obj:`dpnp.argpartition` : Indirect partial sort.
    :obj:`dpnp.take_along_axis` : Apply ``index_array`` from argsort to
                                  an array as if by calling sort.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([3, 1, 2])
    >>> out = np.argsort(x)
    >>> [i for i in out]
    [1, 2, 0]

    """

    x1_desc = dpnp.get_dpnp_descriptor(in_array1)
    if x1_desc:
        if axis != -1:
            pass
        elif kind is not None:
            pass
        elif order is not None:
            pass
        else:
            return dpnp_argsort(x1_desc).get_pyobj()

    return call_origin(numpy.argsort, in_array1, axis, kind, order)


def partition(x1, kth, axis=-1, kind='introselect', order=None):
    """
    Return a partitioned copy of an array.
    For full documentation refer to :obj:`numpy.partition`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input kth is supported as :obj:`int`.
    Parameters ``axis``, ``kind`` and ``order`` are supported only with default values.
    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if not isinstance(kth, int):
            pass
        elif x1_desc.ndim == 0:
            pass
        elif kth >= x1_desc.shape[x1_desc.ndim - 1] or x1_desc.ndim + kth < 0:
            pass
        elif axis != -1:
            pass
        elif kind != 'introselect':
            pass
        elif order is not None:
            pass
        else:
            return dpnp_partition(x1_desc, kth, axis, kind, order).get_pyobj()

    return call_origin(numpy.partition, x1, kth, axis, kind, order)


def searchsorted(x1, x2, side='left', sorter=None):
    """
    Find indices where elements should be inserted to maintain order.
    For full documentation refer to :obj:`numpy.searchsorted`.

    Limitations
    -----------
    Input arrays is supported as :obj:`dpnp.ndarray`.
    Input array is supported only sorted.
    Input side is supported only values ``left``, ``right``.
    Parameters ``sorter`` is supported only with default values.
    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)
    if 0 and x1_desc and x2_desc:
        if x1_desc.ndim != 1:
            pass
        elif x1_desc.dtype != x2_desc.dtype:
            pass
        elif side not in ['left', 'right']:
            pass
        elif sorter is not None:
            pass
        elif x1_desc.size < 2:
            pass
        else:
            return dpnp_searchsorted(x1_desc, x2_desc, side=side).get_pyobj()

    return call_origin(numpy.searchsorted, x1, x2, side=side, sorter=sorter)


def sort(x1, **kwargs):
    """
    Return a sorted copy of an array.

    For full documentation refer to :obj:`numpy.sort`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Keyword arguments ``kwargs`` are currently unsupported.
    Dimension of input array is supported to be equal to ``1``.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.argsort` : Indirect sort.
    :obj:`dpnp.lexsort` : Indirect stable sort on multiple keys.
    :obj:`dpnp.searchsorted` : Find elements in a sorted array.
    :obj:`dpnp.partition` : Partial sort.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 4, 3, 1])
    >>> out = np.sort(a)
    >>> [i for i in out]
    [1, 1, 3, 4]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc and not kwargs:
        if x1_desc.ndim != 1:
            pass
        else:
            return dpnp_sort(x1_desc).get_pyobj()

    return call_origin(numpy.sort, x1, **kwargs)
