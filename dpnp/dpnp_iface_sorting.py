# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

from collections.abc import Sequence

import dpctl.utils as dpu

import dpnp
import dpnp.libtensor._tensor_impl as ti

# pylint: disable=no-name-in-module
import dpnp.libtensor._tensor_sorting_impl as sorting_ext
import dpnp.tensor as dpt

from .dpnp_array import dpnp_array
from .dpnp_utils import (
    map_dtype_to_device,
)
from .tensor._numpy_helper import normalize_axis_index


def _normalize_kth(kth, length, check_bounds=True):
    """Validate `kth` and return it as a list of non-negative integers."""

    if isinstance(kth, int):
        kth = (kth,)
    elif not isinstance(kth, Sequence):
        raise TypeError(
            f"kth must be int or sequence of ints, but got {type(kth)}"
        )
    elif not all(isinstance(k, int) for k in kth):
        raise TypeError("kth is a sequence, but not all elements are integers")

    kth = list(kth)
    if not check_bounds:
        return kth
    for i, k in enumerate(kth):
        if k < 0:
            k += length
        if not 0 <= k < length:
            raise ValueError(f"kth(={kth[i]}) out of bounds {length}")
        kth[i] = k
    return kth


def _wrap_partition_argpartition(a, kth, axis, kind, order, return_indices):
    """
    Partition `a` along `axis` about a single `kth` by radix select, falling
    back to a full sort for several `kth` and for complex data types.

    """

    dpnp.check_supported_arrays_type(a)

    if kind != "introselect":
        raise NotImplementedError(
            "`kind` keyword argument is only supported with its default value."
        )
    if order is not None:
        raise NotImplementedError(
            "`order` keyword argument is only supported with its default value."
        )

    if axis is None:
        a = dpnp.ravel(a)
        axis = -1

    nd = a.ndim
    axis = normalize_axis_index(axis, nd)
    # an empty array is partitioned whatever kth is
    kth = _normalize_kth(kth, a.shape[axis], check_bounds=a.size > 0)

    if len(kth) == 0 and not return_indices:
        # nothing is to be put in its sorted position
        return dpnp.copy(a)
    if (
        a.size == 0
        or len(kth) != 1
        or dpnp.issubdtype(a.dtype, dpnp.complexfloating)
    ):
        sort_fn = dpnp.argsort if return_indices else dpnp.sort
        return sort_fn(a, axis=axis)

    usm_a = dpnp.get_usm_ndarray(a)
    if axis != nd - 1:
        perm = [i for i in range(nd) if i != axis] + [axis]
        usm_a = dpt.permute_dims(usm_a, perm)

    exec_q = usm_a.sycl_queue
    _manager = dpu.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events

    # the rows are partitioned along the last axis of C-contiguous arrays
    if not usm_a.flags.c_contiguous:
        usm_tmp = dpt.empty_like(usm_a, order="C")
        ht_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=usm_a, dst=usm_tmp, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        usm_a = usm_tmp
        dep_evs = [copy_ev]

    if return_indices:
        partition_fn = sorting_ext._argpartition
        res_dt = dpt.int64
    else:
        partition_fn = sorting_ext._partition
        res_dt = usm_a.dtype

    usm_res = dpt.empty(
        usm_a.shape,
        dtype=res_dt,
        usm_type=usm_a.usm_type,
        order="C",
        sycl_queue=exec_q,
    )
    ht_ev, part_ev = partition_fn(
        usm_a, kth[0], usm_res, sycl_queue=exec_q, depends=dep_evs
    )
    _manager.add_event_pair(ht_ev, part_ev)

    if axis != nd - 1:
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        usm_res = dpt.permute_dims(usm_res, inv_perm)
    return dpnp_array._create_from_usm_ndarray(usm_res)


def _wrap_sort_argsort(
    a,
    _sorting_fn,
    axis=-1,
    kind=None,
    order=None,
    descending=False,
    stable=True,
):
    """Wrap a sorting call from dpnp.tensor interface."""

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


def argpartition(a, kth, axis=-1, kind="introselect", order=None):
    """
    Perform an indirect partition along the given axis using the algorithm
    specified by the `kind` keyword. It returns an array of indices of the
    same shape as `a` that index data along the given axis in partitioned
    order.

    For full documentation refer to :obj:`numpy.argpartition`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to sort.
    kth : {int, sequence of ints}
        Element index to partition by. The k-th element will be in its final
        sorted position and all smaller elements will be moved before it and
        all larger elements behind it. The order of all elements in the
        partitions is undefined. If provided with a sequence of k-th it will
        partition all of them into their sorted position at once.
    axis : {None, int}, optional
        Axis along which to sort. If ``None``, the array is flattened before
        sorting. The default is ``-1``, which sorts along the last axis.

        Default: ``-1``.

    Returns
    -------
    index_array : dpnp.ndarray
        Array of indices that partition `a` along the specified axis. If `a`
        is one-dimensional, ``a[index_array]`` yields a partitioned `a`. More
        generally, ``dpnp.take_along_axis(a, index_array, axis=axis)`` always
        yields the partitioned `a`, irrespective of dimensionality.

    Limitations
    -----------
    Parameters `kind` and `order` are only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.partition` : Describes partition algorithms used.
    :obj:`dpnp.ndarray.partition` : Inplace partition.
    :obj:`dpnp.argsort` : Full indirect sort.
    :obj:`dpnp.take_along_axis` : Apply ``index_array`` from argpartition to
                                  an array as if by calling partition.

    Examples
    --------
    One dimensional array:

    >>> import dpnp as np
    >>> x = np.array([3, 4, 2, 1])
    >>> x[np.argpartition(x, 3)]
    array([2, 1, 3, 4]) # may vary
    >>> x[np.argpartition(x, (1, 3))]
    array([1, 2, 3, 4]) # may vary

    Multi-dimensional array:

    >>> x = np.array([[3, 4, 2], [1, 3, 1]])
    >>> index_array = np.argpartition(x, kth=1, axis=-1)
    >>> # below is the same as np.partition(x, kth=1)
    >>> np.take_along_axis(x, index_array, axis=-1)
    array([[2, 3, 4],
           [1, 1, 3]])

    """

    return _wrap_partition_argpartition(
        a, kth, axis, kind, order, return_indices=True
    )


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
        (by value). NaN values (and complex values with a NaN component) are
        ordered to the end regardless of `descending`.

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
    Parameter `order` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.
    Sorting algorithms ``"quicksort"`` and ``"heapsort"`` are not supported.

    See Also
    --------
    :obj:`dpnp.ndarray.argsort` : Equivalent method.
    :obj:`dpnp.sort` : Return a sorted copy of an array.
    :obj:`dpnp.lexsort` : Indirect stable sort with multiple keys.
    :obj:`dpnp.argpartition` : Indirect partial sort.
    :obj:`dpnp.take_along_axis` : Apply ``index_array`` from :obj:`dpnp.argsort`
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


def partition(a, kth, axis=-1, kind="introselect", order=None):
    """
    Return a partitioned copy of an array.

    For full documentation refer to :obj:`numpy.partition`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to be sorted.
    kth : {int, sequence of ints}
        Element index to partition by. The k-th value of the element will be in
        its final sorted position and all smaller elements will be moved before
        it and all equal or greater elements behind it. The order of all
        elements in the partitions is undefined. If provided with a sequence of
        k-th it will partition all elements indexed by k-th of them into their
        sorted position at once.
    axis : {None, int}, optional
        Axis along which to sort. If ``None``, the array is flattened before
        sorting. The default is ``-1``, which sorts along the last axis.

        Default: ``-1``.

    Returns
    -------
    out : dpnp.ndarray
        Array of the same type and shape as `a`.

    Limitations
    -----------
    Parameters `kind` and `order` are only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.ndarray.partition` : Equivalent method.
    :obj:`dpnp.argpartition` : Indirect partition.
    :obj:`dpnp.sort` : Full sorting.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([7, 1, 7, 7, 1, 5, 7, 2, 3, 2, 6, 2, 3, 0])
    >>> p = np.partition(a, 4)
    >>> p
    array([0, 1, 1, 2, 2, 2, 3, 3, 5, 7, 7, 7, 7, 6]) # may vary

    ``p[4]`` is 2;  all elements in ``p[:4]`` are less than or equal to
    ``p[4]``, and all elements in ``p[5:]`` are greater than or equal to
    ``p[4]``. The partition is::

        [0, 1, 1, 2], [2], [2, 3, 3, 5, 7, 7, 7, 7, 6]

    The next example shows the use of multiple values passed to `kth`.

    >>> p2 = np.partition(a, (4, 8))
    >>> p2
    array([0, 1, 1, 2, 2, 2, 3, 3, 5, 6, 7, 7, 7, 7])

    ``p2[4]`` is 2  and ``p2[8]`` is 5. All elements in ``p2[:4]`` are less
    than or equal to ``p2[4]``, all elements in ``p2[5:8]`` are greater than or
    equal to ``p2[4]`` and less than or equal to ``p2[8]``, and all elements in
    ``p2[9:]`` are greater than or equal to ``p2[8]``. The partition is::

        [0, 1, 1, 2], [2], [2, 3, 3], [5], [6, 7, 7, 7, 7]

    """

    return _wrap_partition_argpartition(
        a, kth, axis, kind, order, return_indices=False
    )


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
        (by value). NaN values (and complex values with a NaN component) are
        ordered to the end regardless of `descending`.

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
