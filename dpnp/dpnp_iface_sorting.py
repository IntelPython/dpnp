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
from dpctl.tensor._numpy_helper import normalize_axis_index

import dpnp

# pylint: disable=no-name-in-module
from .dpnp_algo import (
    dpnp_partition,
)
from .dpnp_array import dpnp_array
from .dpnp_utils import (
    call_origin,
    map_dtype_to_device,
)

__all__ = ["argsort", "partition", "sort", "sort_complex"]


def _wrap_sort_argsort(
    a,
    _sorting_fn,
    axis=-1,
    kind=None,
    order=None,
    descending=False,
    stable=True,
):
    """Wrap a sorting call from dpctl.tensor interface."""

    if order is not None:
        raise NotImplementedError(
            "`order` keyword argument is only supported with its default value."
        )
    if stable is not None:
        if stable not in [True, False]:
            raise ValueError(
                "`stable` parameter should be None, True, or False."
            )
        if kind is not None:
            raise ValueError(
                "`kind` and `stable` parameters can't be provided at"
                " the same time. Use only one of them."
            )

    usm_a = dpnp.get_usm_ndarray(a)
    if axis is None:
        usm_a = dpt.reshape(usm_a, -1)
        axis = -1

    axis = normalize_axis_index(axis, ndim=usm_a.ndim)
    usm_res = _sorting_fn(
        usm_a, axis=axis, descending=descending, stable=stable, kind=kind
    )
    return dpnp_array._create_from_usm_ndarray(usm_res)


def argsort(
    a, axis=-1, kind=None, order=None, *, descending=False, stable=None
):
    """
    Returns the indices that would sort an array.

    For full documentation refer to :obj:`numpy.argsort`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to be sorted.
    axis : {None, int}, optional
        Axis along which to sort. If ``None``, the array is flattened before
        sorting. The default is ``-1``, which sorts along the last axis.

        Default: ``-1``.
    kind : {None, "stable", "mergesort", "radixsort"}, optional
        Sorting algorithm. The default is ``None``, which uses parallel
        merge-sort or parallel radix-sort algorithms depending on the array
        data type.

        Default: ``None``.
    descending : bool, optional
        Sort order. If ``True``, the array must be sorted in descending order
        (by value). If ``False``, the array must be sorted in ascending order
        (by value).

        Default: ``False``.
    stable : {None, bool}, optional
        Sort stability. If ``True``, the returned array will maintain the
        relative order of `a` values which compare as equal. The same behavior
        applies when set to ``False`` or ``None``.
        Internally, this option selects ``kind="stable"``.

        Default: ``None``.

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
    For zero-dimensional arrays, if ``axis=None``, output is a one-dimensional
    array with a single zero element. Otherwise, an ``AxisError`` is raised.

    Limitations
    -----------
    Parameters `order` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.
    Sorting algorithms ``"quicksort"`` and ``"heapsort"`` are not supported.

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

    return _wrap_sort_argsort(
        a,
        dpt.argsort,
        axis=axis,
        kind=kind,
        order=order,
        descending=descending,
        stable=stable,
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
        if dpnp.is_cuda_backend(x1_desc.get_array()):  # pragma: no cover
            raise NotImplementedError(
                "Running on CUDA is currently not supported"
            )

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


def sort(a, axis=-1, kind=None, order=None, *, descending=False, stable=None):
    """
    Return a sorted copy of an array.

    For full documentation refer to :obj:`numpy.sort`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to be sorted.
    axis : {None, int}, optional
        Axis along which to sort. If ``None``, the array is flattened before
        sorting. The default is ``-1``, which sorts along the last axis.

        Default: ``-1``.
    kind : {None, "stable", "mergesort", "radixsort"}, optional
        Sorting algorithm. The default is ``None``, which uses parallel
        merge-sort or parallel radix-sort algorithms depending on the array
        data type.

        Default: ``None``.
    descending : bool, optional
        Sort order. If ``True``, the array must be sorted in descending order
        (by value). If ``False``, the array must be sorted in ascending order
        (by value).

        Default: ``False``.
    stable : {None, bool}, optional
        Sort stability. If ``True``, the returned array will maintain the
        relative order of `a` values which compare as equal. The same behavior
        applies when set to ``False`` or ``None``.
        Internally, this option selects ``kind="stable"``.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Sorted array with the same type and shape as `a`.

    Notes
    -----
    For zero-dimensional arrays, if ``axis=None``, output is the input array
    returned as a one-dimensional array. Otherwise, an ``AxisError`` is raised.

    Limitations
    -----------
    Parameters `order` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.
    Sorting algorithms ``"quicksort"`` and ``"heapsort"`` are not supported.

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
    >>> a = np.array([[1, 4], [3, 1]])
    >>> np.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> np.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> np.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    """

    return _wrap_sort_argsort(
        a,
        dpt.sort,
        axis=axis,
        kind=kind,
        order=order,
        descending=descending,
        stable=stable,
    )


def sort_complex(a):
    """
    Sort a complex array using the real part first, then the imaginary part.

    For full documentation refer to :obj:`numpy.sort_complex`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.

    Returns
    -------
    out : dpnp.ndarray of complex dtype
        Always returns a sorted complex array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([5, 3, 6, 2, 1])
    >>> np.sort_complex(a)
    array([1.+0.j, 2.+0.j, 3.+0.j, 5.+0.j, 6.+0.j])

    >>> a = np.array([1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j])
    >>> np.sort_complex(a)
    array([1.+2.j, 2.-1.j, 3.-3.j, 3.-2.j, 3.+5.j])

    """

    b = dpnp.sort(a)
    if not dpnp.issubdtype(b.dtype, dpnp.complexfloating):
        if b.dtype.char in "bhBH":
            b_dt = dpnp.complex64
        else:
            b_dt = map_dtype_to_device(dpnp.complex128, b.sycl_device)
        return b.astype(b_dt)
    return b
