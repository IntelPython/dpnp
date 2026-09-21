# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

import math
import operator
from typing import NamedTuple

import dpctl.utils as du

import dpnp.libtensor._tensor_impl as ti
import dpnp.tensor as dpt
from dpnp.libtensor._tensor_sorting_impl import (
    _argsort_ascending,
    _argsort_descending,
    _bitonic_argsort_ascending,
    _bitonic_argsort_descending,
    _bitonic_argsort_local_memory_nelems,
    _bitonic_sort_ascending,
    _bitonic_sort_descending,
    _bitonic_sort_local_memory_nelems,
    _radix_argsort_ascending,
    _radix_argsort_descending,
    _radix_sort_ascending,
    _radix_sort_descending,
    _radix_sort_dtype_supported,
    _sort_ascending,
    _sort_descending,
    _topk,
)

from ._numpy_helper import normalize_axis_index

__all__ = ["sort", "argsort", "top_k"]

# data types for which the default `kind` sorts with the radix sort, rather
# than with the merge sort
_radixsort_by_default_dtypes = [
    dpt.bool,
    dpt.uint8,
    dpt.int8,
    dpt.int16,
    dpt.uint16,
]

# Parameters of the heuristic that lets the default `kind` reach for the
# bitonic network. They were tuned on a sweep of row lengths, row counts and
# data types on Intel(R) Graphics [0x7d41], and are a starting point for other
# devices rather than a portable optimum.

# The radix sort makes a pass over the row per four bits of the key, so the
# wider the key, the longer the row over which the network still competes
_bitonic_max_padded_nelems_per_key_byte = 128

# A row of a handful of elements leaves the network too little work per row to
# pay for the comparators its padding adds, unless there are enough rows of it
# to keep the device busy on their own
_bitonic_min_padded_nelems = 16
_bitonic_min_nelems_for_short_rows = 1 << 20

# A row too long for local memory is sorted through global memory, one pass per
# few steps of the network, and the number of those passes grows with the square
# of how many blocks of local memory the padded row spans. The network keeps its
# edge while there are only a few passes, and only for a row of a power of two
# length, which needs no padding and is therefore sorted in the destination
# itself rather than in a temporary, and only when sorting values, since the
# `{key, index}` pair an argsort carries widens everything a pass moves.
_bitonic_max_local_memory_blocks = 32

# comparators the network may spend per element-comparison of the merge argsort
# it replaces. Sorting a padded row of 2**L elements spends L * (L + 1) / 2
# comparators per element against about log2 of the row length of them, and an
# argsort has a thinner margin to spend them out of than a sort of values, since
# the `{key, index}` pair it carries through local memory is wider than the key
# the merge sort moves. Because the comparator count is that of the padded row,
# this is also what turns down rows sitting just above a power of two, where the
# padding nearly doubles the work while the merge sort sees no extra elements at
# all.
_bitonic_max_work_ratio_vs_mergeargsort = 6.5
_bitonic_max_work_ratio_slow_compare = 4.4

# data types this device compares several times more slowly than the rest, here
# because it has no native double precision and because ordering a pair of
# complex numbers compares the real parts and then the imaginary ones. The
# network does far more comparisons than the merge sort, so it has less of a
# margin to spend on a row long enough for their number to matter.
_bitonic_slow_compare_dtypes = [dpt.complex128]


def _bitonic_padded_nelems(sort_nelems):
    """Row length the bitonic network pads `sort_nelems` up to"""
    return 1 << (sort_nelems - 1).bit_length()


def _bitonic_comparator_count(padded_nelems):
    """Comparators of a bitonic network sorting `padded_nelems` elements"""
    n_stages = padded_nelems.bit_length() - 1
    return (padded_nelems * n_stages * (n_stages + 1)) // 2


def _use_bitonicsort_by_default(
    dt, sort_nelems, nelems, local_memory_nelems, sorting_values
):
    """Whether the bitonic network is expected to be faster than the algorithm
    the default `kind` would otherwise use for this data type, row length, and
    number of elements to sort in total"""
    if sort_nelems < 2:
        return False

    padded_nelems = _bitonic_padded_nelems(sort_nelems)

    if dt in _radixsort_by_default_dtypes:
        return (
            padded_nelems
            <= _bitonic_max_padded_nelems_per_key_byte * dt.itemsize
        )

    slow_compare = dt in _bitonic_slow_compare_dtypes

    if padded_nelems > local_memory_nelems:
        # the network runs through global memory
        return (
            sorting_values
            and padded_nelems == sort_nelems
            and padded_nelems
            <= _bitonic_max_local_memory_blocks * local_memory_nelems
        )

    # the padded row is held in local memory, where a single kernel sorts it
    if sorting_values:
        return padded_nelems >= _bitonic_min_padded_nelems or (
            not slow_compare and nelems >= _bitonic_min_nelems_for_short_rows
        )

    max_work_ratio = (
        _bitonic_max_work_ratio_slow_compare
        if slow_compare
        else _bitonic_max_work_ratio_vs_mergeargsort
    )

    return _bitonic_comparator_count(padded_nelems) <= (
        max_work_ratio * sort_nelems * math.log2(sort_nelems)
    )


def _get_mergesort_impl_fn(descending):
    return _sort_descending if descending else _sort_ascending


def _get_radixsort_impl_fn(descending):
    return _radix_sort_descending if descending else _radix_sort_ascending


def _get_bitonicsort_impl_fn(descending):
    return _bitonic_sort_descending if descending else _bitonic_sort_ascending


def sort(x, /, *, axis=-1, descending=False, stable=True, kind=None):
    """sort(x, axis=-1, descending=False, stable=True)

    Returns a sorted copy of an input array `x`.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int]):
            axis along which to sort. If set to `-1`, the function
            must sort along the last axis. Default: `-1`.
        descending (Optional[bool]):
            sort order. If `True`, the array must be sorted in descending
            order (by value). If `False`, the array must be sorted in
            ascending order (by value). NaN values (and complex values with
            a NaN component) are ordered to the end regardless of
            `descending`. Default: `False`.
        stable (Optional[bool]):
            sort stability. If `True`, the returned array must maintain the
            relative order of `x` values which compare as equal. If `False`,
            the returned array may or may not maintain the relative order of
            `x` values which compare as equal. Default: `True`.
        kind (Optional[Literal["stable", "mergesort", "radixsort",
            "bitonicsort"]]):
            Sorting algorithm. The default is `"stable"`, which uses a
            parallel merge-sort, a parallel radix-sort, or a bitonic sorting
            network, whichever is expected to be the fastest for the array
            data type, the length of the sorted rows, and the device.
            `"bitonicsort"` always uses the bitonic sorting network, which
            tends to be faster when many short rows are sorted at once, and
            slower than the alternatives for rows too long to be held in the
            local memory of the device.
    Returns:
        usm_ndarray:
            a sorted array. The returned array has the same data type and
            the same shape as the input array `x`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected type dpnp.tensor.usm_ndarray, got {type(x)}")
    nd = x.ndim
    if nd == 0:
        axis = normalize_axis_index(axis, ndim=1, msg_prefix="axis")
        return dpt.copy(x, order="C")
    else:
        axis = normalize_axis_index(axis, ndim=nd, msg_prefix="axis")
    a1 = axis + 1
    if a1 == nd:
        perm = list(range(nd))
        arr = x
    else:
        perm = [i for i in range(nd) if i != axis] + [
            axis,
        ]
        arr = dpt.permute_dims(x, perm)
    if kind is None:
        kind = "stable"
    if not isinstance(kind, str) or kind not in [
        "stable",
        "radixsort",
        "mergesort",
        "bitonicsort",
    ]:
        raise ValueError(
            "Unsupported kind value. Expected 'stable', 'mergesort', "
            f"'radixsort', or 'bitonicsort', but got '{kind}'"
        )
    if kind == "mergesort":
        impl_fn = _get_mergesort_impl_fn(descending)
    elif kind == "bitonicsort":
        impl_fn = _get_bitonicsort_impl_fn(descending)
    elif kind == "radixsort":
        if _radix_sort_dtype_supported(x.dtype.num):
            impl_fn = _get_radixsort_impl_fn(descending)
        else:
            raise ValueError(f"Radix sort is not supported for {x.dtype}")
    else:
        dt = x.dtype
        if _use_bitonicsort_by_default(
            dt,
            arr.shape[-1],
            arr.size,
            _bitonic_sort_local_memory_nelems(
                typenum=dt.num, sycl_queue=x.sycl_queue
            ),
            sorting_values=True,
        ):
            impl_fn = _get_bitonicsort_impl_fn(descending)
        elif dt in _radixsort_by_default_dtypes:
            impl_fn = _get_radixsort_impl_fn(descending)
        else:
            impl_fn = _get_mergesort_impl_fn(descending)
    exec_q = x.sycl_queue
    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    if arr.flags.c_contiguous:
        res = dpt.empty_like(arr, order="C")
        ht_ev, impl_ev = impl_fn(
            src=arr,
            trailing_dims_to_sort=1,
            dst=res,
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_ev, impl_ev)
    else:
        tmp = dpt.empty_like(arr, order="C")
        ht_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arr, dst=tmp, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        res = dpt.empty_like(arr, order="C")
        ht_ev, impl_ev = impl_fn(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=res,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        _manager.add_event_pair(ht_ev, impl_ev)
    if a1 != nd:
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt.permute_dims(res, inv_perm)
    return res


def _get_mergeargsort_impl_fn(descending):
    return _argsort_descending if descending else _argsort_ascending


def _get_radixargsort_impl_fn(descending):
    return _radix_argsort_descending if descending else _radix_argsort_ascending


def _get_bitonicargsort_impl_fn(descending):
    return (
        _bitonic_argsort_descending
        if descending
        else _bitonic_argsort_ascending
    )


def argsort(x, axis=-1, descending=False, stable=True, kind=None):
    """argsort(x, axis=-1, descending=False, stable=True)

    Returns the indices that sort an array `x` along a specified axis.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int]):
            axis along which to sort. If set to `-1`, the function
            must sort along the last axis. Default: `-1`.
        descending (Optional[bool]):
            sort order. If `True`, the array must be sorted in descending
            order (by value). If `False`, the array must be sorted in
            ascending order (by value). NaN values (and complex values with
            a NaN component) are ordered to the end regardless of
            `descending`. Default: `False`.
        stable (Optional[bool]):
            sort stability. If `True`, the returned array must maintain the
            relative order of `x` values which compare as equal. If `False`,
            the returned array may or may not maintain the relative order of
            `x` values which compare as equal. Default: `True`.
        kind (Optional[Literal["stable", "mergesort", "radixsort",
            "bitonicsort"]]):
            Sorting algorithm. The default is `"stable"`, which uses a
            parallel merge-sort, a parallel radix-sort, or a bitonic sorting
            network, whichever is expected to be the fastest for the array
            data type, the length of the sorted rows, and the device.
            `"bitonicsort"` always uses the bitonic sorting network, which
            tends to be faster when many short rows are sorted at once, and
            slower than the alternatives for rows too long to be held in the
            local memory of the device.

    Returns:
        usm_ndarray:
            an array of indices. The returned array has the  same shape as
            the input array `x`. The return array has default array index
            data type.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected type dpnp.tensor.usm_ndarray, got {type(x)}")
    nd = x.ndim
    if nd == 0:
        axis = normalize_axis_index(axis, ndim=1, msg_prefix="axis")
        return dpt.zeros_like(
            x, dtype=ti.default_device_index_type(x.sycl_queue), order="C"
        )
    else:
        axis = normalize_axis_index(axis, ndim=nd, msg_prefix="axis")
    a1 = axis + 1
    if a1 == nd:
        perm = list(range(nd))
        arr = x
    else:
        perm = [i for i in range(nd) if i != axis] + [
            axis,
        ]
        arr = dpt.permute_dims(x, perm)
    if kind is None:
        kind = "stable"
    if not isinstance(kind, str) or kind not in [
        "stable",
        "radixsort",
        "mergesort",
        "bitonicsort",
    ]:
        raise ValueError(
            "Unsupported kind value. Expected 'stable', 'mergesort', "
            f"'radixsort', or 'bitonicsort', but got '{kind}'"
        )
    exec_q = x.sycl_queue
    index_dt = ti.default_device_index_type(exec_q)
    if kind == "mergesort":
        impl_fn = _get_mergeargsort_impl_fn(descending)
    elif kind == "bitonicsort":
        impl_fn = _get_bitonicargsort_impl_fn(descending)
    elif kind == "radixsort":
        if _radix_sort_dtype_supported(x.dtype.num):
            impl_fn = _get_radixargsort_impl_fn(descending)
        else:
            raise ValueError(f"Radix sort is not supported for {x.dtype}")
    else:
        dt = x.dtype
        if _use_bitonicsort_by_default(
            dt,
            arr.shape[-1],
            arr.size,
            _bitonic_argsort_local_memory_nelems(
                typenum=dt.num,
                index_typenum=dpt.dtype(index_dt).num,
                sycl_queue=exec_q,
            ),
            sorting_values=False,
        ):
            impl_fn = _get_bitonicargsort_impl_fn(descending)
        elif dt in _radixsort_by_default_dtypes:
            impl_fn = _get_radixargsort_impl_fn(descending)
        else:
            impl_fn = _get_mergeargsort_impl_fn(descending)
    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    if arr.flags.c_contiguous:
        res = dpt.empty_like(arr, dtype=index_dt, order="C")
        ht_ev, impl_ev = impl_fn(
            src=arr,
            trailing_dims_to_sort=1,
            dst=res,
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_ev, impl_ev)
    else:
        tmp = dpt.empty_like(arr, order="C")
        ht_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arr, dst=tmp, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        res = dpt.empty_like(arr, dtype=index_dt, order="C")
        ht_ev, impl_ev = impl_fn(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=res,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        _manager.add_event_pair(ht_ev, impl_ev)
    if a1 != nd:
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        res = dpt.permute_dims(res, inv_perm)
    return res


def _get_top_k_largest(mode):
    modes = {"largest": True, "smallest": False}
    try:
        return modes[mode]
    except KeyError:
        raise ValueError(
            f"`mode` must be `largest` or `smallest`. Got `{mode}`."
        )


class TopKResult(NamedTuple):
    values: dpt.usm_ndarray
    indices: dpt.usm_ndarray


def top_k(x, k, /, *, axis=None, mode="largest"):
    """top_k(x, k, axis=None, mode="largest")

    Returns the `k` largest or smallest values and their indices in the input
    array `x` along the specified axis `axis`.

    Args:
        x (usm_ndarray):
            input array.
        k (int):
            number of elements to find. Must be a positive integer value.
        axis (Optional[int]):
            axis along which to search. If `None`, the search will be performed
            over the flattened array. Default: ``None``.
        mode (Literal["largest", "smallest"]):
            search mode. Must be one of the following modes:

            - `"largest"`: return the `k` largest elements.
            - `"smallest"`: return the `k` smallest elements.

            NaN values (and complex values with a NaN component) are ordered
            last for both modes, so they are not returned ahead of finite
            values.

            Default: `"largest"`.

    Returns:
        tuple[usm_ndarray, usm_ndarray]
            a namedtuple `(values, indices)` whose

            * first element `values` will be an array containing the `k`
              largest or smallest elements of `x`. The array has the same data
              type as `x`. If `axis` was `None`, `values` will be a
              one-dimensional array with shape `(k,)` and otherwise, `values`
              will have shape `x.shape[:axis] + (k,) + x.shape[axis+1:]`
            * second element `indices` will be an array containing indices of
              `x` that result in `values`. The array will have the same shape
              as `values` and will have the default array index data type.

    .. note::
        The order of the `k` elements along `axis` in `values` and `indices`
        is unspecified, and may depend on the data type, the shape of `x` and
        the device. Sort the result if a particular order is needed. Among
        elements that compare equal, those with lower indices are selected.
    """
    largest = _get_top_k_largest(mode)
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected type dpnp.tensor.usm_ndarray, got {type(x)}")

    k = operator.index(k)
    if k < 0:
        raise ValueError("`k` must be a positive integer value")

    nd = x.ndim
    if axis is None:
        sz = x.size
        if nd == 0:
            if k > 1:
                raise ValueError(f"`k`={k} is out of bounds 1")
            return TopKResult(
                dpt.copy(x, order="C"),
                dpt.zeros_like(
                    x, dtype=ti.default_device_index_type(x.sycl_queue)
                ),
            )
        arr = x
        n_search_dims = None
        res_sh = k
    else:
        axis = normalize_axis_index(axis, ndim=nd, msg_prefix="axis")
        sz = x.shape[axis]
        a1 = axis + 1
        if a1 == nd:
            perm = list(range(nd))
            arr = x
        else:
            perm = [i for i in range(nd) if i != axis] + [
                axis,
            ]
            arr = dpt.permute_dims(x, perm)
        n_search_dims = 1
        res_sh = arr.shape[: nd - 1] + (k,)

    if k > sz:
        raise ValueError(f"`k`={k} is out of bounds {sz}")

    exec_q = x.sycl_queue
    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events

    res_usm_type = arr.usm_type
    if arr.flags.c_contiguous:
        vals = dpt.empty(
            res_sh,
            dtype=arr.dtype,
            usm_type=res_usm_type,
            order="C",
            sycl_queue=exec_q,
        )
        inds = dpt.empty(
            res_sh,
            dtype=ti.default_device_index_type(exec_q),
            usm_type=res_usm_type,
            order="C",
            sycl_queue=exec_q,
        )
        ht_ev, impl_ev = _topk(
            src=arr,
            trailing_dims_to_search=n_search_dims,
            k=k,
            largest=largest,
            vals=vals,
            inds=inds,
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_ev, impl_ev)
    else:
        tmp = dpt.empty_like(arr, order="C")
        ht_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arr, dst=tmp, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        vals = dpt.empty(
            res_sh,
            dtype=arr.dtype,
            usm_type=res_usm_type,
            order="C",
            sycl_queue=exec_q,
        )
        inds = dpt.empty(
            res_sh,
            dtype=ti.default_device_index_type(exec_q),
            usm_type=res_usm_type,
            order="C",
            sycl_queue=exec_q,
        )
        ht_ev, impl_ev = _topk(
            src=tmp,
            trailing_dims_to_search=n_search_dims,
            k=k,
            largest=largest,
            vals=vals,
            inds=inds,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        _manager.add_event_pair(ht_ev, impl_ev)
    if axis is not None and a1 != nd:
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        vals = dpt.permute_dims(vals, inv_perm)
        inds = dpt.permute_dims(inds, inv_perm)

    return TopKResult(vals, inds)
