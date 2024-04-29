# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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


import dpctl.tensor as dpt
import numpy
from numpy.core.numeric import normalize_axis_index

import dpnp

# pylint: disable=no-name-in-module
from .dpnp_algo import (
    dpnp_partition,
)
from .dpnp_array import dpnp_array
from .dpnp_utils import (
    call_origin,
)

__all__ = ["argsort", "partition", "sort"]


def argsort(a, axis=-1, kind=None, order=None):
    """
    Returns the indices that would sort an array.

    For full documentation refer to :obj:`numpy.argsort`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort. If ``None``, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {None, "stable"}, optional
        Default is ``None``, which is equivalent to `"stable"`.
        Unlike in NumPy any other options are not accepted here.

    Returns
    -------
    out : dpnp.ndarray
        Array of indices that sort `a` along the specified `axis`.
        If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.
        More generally, ``dpnp.take_along_axis(a, index_array, axis=axis)``
        always yields the sorted `a`, irrespective of dimensionality.
        The return array has default array index data type.

    Notes
    -----
    For zero-dimensional arrays, if `axis=None`, output is a one-dimensional
    array with a single zero element. Otherwise, an ``AxisError`` is raised.

    Limitations
    -----------
    Parameters `order` is only supported with its default value.
    Parameters `kind` can only be ``None`` or ``"stable"`` which
    are equivalent.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.ndarray.argsort` : Equivalent method.
    :obj:`dpnp.sort` : Return a sorted copy of an array.
    :obj:`dpnp.lexsort` : Indirect stable sort with multiple keys.
    :obj:`dpnp.argpartition` : Indirect partial sort.
    :obj:`dpnp.take_along_axis` : Apply ``index_array`` from obj:`dpnp.argsort`
                                  to an array as if by calling sort.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])

    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])

    >>> ind = np.argsort(x, axis=0)  # sorts along first axis
    >>> ind
    array([[0, 1],
           [1, 0]])
    >>> np.take_along_axis(x, ind, axis=0)  # same as np.sort(x, axis=0)
    array([[0, 2],
           [2, 3]])

    >>> ind = np.argsort(x, axis=1)  # sorts along last axis
    >>> ind
    array([[0, 1],
           [0, 1]])
    >>> np.take_along_axis(x, ind, axis=1)  # same as np.sort(x, axis=1)
    array([[0, 3],
           [2, 2]])

    """

    if order is not None:
        raise NotImplementedError(
            "order keyword argument is only supported with its default value."
        )
    if kind is not None and kind != "stable":
        raise NotImplementedError(
            "kind keyword argument can only be None or 'stable'."
        )

    dpnp.check_supported_arrays_type(a)
    if axis is None:
        a = a.flatten()
        axis = -1

    axis = normalize_axis_index(axis, ndim=a.ndim)
    return dpnp_array._create_from_usm_ndarray(
        dpt.argsort(dpnp.get_usm_ndarray(a), axis=axis)
    )


def partition(x1, kth, axis=-1, kind="introselect", order=None):
    """
    Return a partitioned copy of an array.

    For full documentation refer to :obj:`numpy.partition`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Input `kth` is supported as :obj:`int`.
    Parameters `axis`, `kind` and `order` are supported only with default
    values.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if not isinstance(kth, int):
            pass
        elif x1_desc.ndim == 0:
            pass
        elif kth >= x1_desc.shape[x1_desc.ndim - 1] or x1_desc.ndim + kth < 0:
            pass
        elif axis != -1:
            pass
        elif kind != "introselect":
            pass
        elif order is not None:
            pass
        else:
            return dpnp_partition(x1_desc, kth, axis, kind, order).get_pyobj()

    return call_origin(numpy.partition, x1, kth, axis, kind, order)


def sort(a, axis=-1, kind=None, order=None):
    """
    Return a sorted copy of an array.

    For full documentation refer to :obj:`numpy.sort`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort. If ``None``, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {None, "stable"}, optional
        Default is ``None``, which is equivalent to `"stable"`.
        Unlike in NumPy any other options are not accepted here.

    Returns
    -------
    out : dpnp.ndarray
        Sorted array with the same type and shape as `a`.

    Notes
    -----
    For zero-dimensional arrays, if `axis=None`, output is the input array
    returned as a one-dimensional array. Otherwise, an ``AxisError`` is raised.

    Limitations
    -----------
    Parameters `order` is only supported with its default value.
    Parameters `kind` can only be ``None`` or ``"stable"`` which
    are equivalent.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.ndarray.sort` : Sort an array in-place.
    :obj:`dpnp.argsort` : Return the indices that would sort an array.
    :obj:`dpnp.lexsort` : Indirect stable sort on multiple keys.
    :obj:`dpnp.searchsorted` : Find elements in a sorted array.
    :obj:`dpnp.partition` : Partial sort.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1,4],[3,1]])
    >>> np.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> np.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> np.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    """

    if order is not None:
        raise NotImplementedError(
            "order keyword argument is only supported with its default value."
        )
    if kind is not None and kind != "stable":
        raise NotImplementedError(
            "kind keyword argument can only be None or 'stable'."
        )

    dpnp.check_supported_arrays_type(a)
    if axis is None:
        a = a.flatten()
        axis = -1

    axis = normalize_axis_index(axis, ndim=a.ndim)
    return dpnp_array._create_from_usm_ndarray(
        dpt.sort(dpnp.get_usm_ndarray(a), axis=axis)
    )
