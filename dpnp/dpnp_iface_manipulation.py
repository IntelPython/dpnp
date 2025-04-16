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
Interface of the array manipulation routines part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import math
import operator
import warnings
from typing import NamedTuple

import dpctl
import dpctl.tensor as dpt
import numpy
from dpctl.tensor._numpy_helper import AxisError, normalize_axis_index

import dpnp

from .dpnp_array import dpnp_array

# pylint: disable=no-name-in-module
from .dpnp_utils import get_usm_allocations
from .dpnp_utils.dpnp_utils_pad import dpnp_pad


class InsertDeleteParams(NamedTuple):
    """Parameters used for ``dpnp.delete`` and ``dpnp.insert``."""

    a: dpnp_array
    a_ndim: int
    order: str
    axis: int
    slobj: list
    n: int
    a_shape: list
    exec_q: dpctl.SyclQueue
    usm_type: str


# pylint:disable=missing-class-docstring
class UniqueAllResult(NamedTuple):
    values: dpnp.ndarray
    indices: dpnp.ndarray
    inverse_indices: dpnp.ndarray
    counts: dpnp.ndarray


class UniqueCountsResult(NamedTuple):
    values: dpnp.ndarray
    counts: dpnp.ndarray


class UniqueInverseResult(NamedTuple):
    values: dpnp.ndarray
    inverse_indices: dpnp.ndarray


__all__ = [
    "append",
    "array_split",
    "asarray_chkfinite",
    "asfarray",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "can_cast",
    "column_stack",
    "concat",
    "concatenate",
    "copyto",
    "delete",
    "dsplit",
    "dstack",
    "expand_dims",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "insert",
    "matrix_transpose",
    "moveaxis",
    "ndim",
    "pad",
    "permute_dims",
    "ravel",
    "repeat",
    "require",
    "reshape",
    "resize",
    "result_type",
    "roll",
    "rollaxis",
    "rot90",
    "row_stack",
    "shape",
    "size",
    "split",
    "squeeze",
    "stack",
    "swapaxes",
    "tile",
    "transpose",
    "trim_zeros",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unstack",
    "vsplit",
    "vstack",
]


def _check_stack_arrays(arrays):
    """Validate a sequence type of arrays to stack."""

    if not hasattr(arrays, "__getitem__"):
        raise TypeError(
            'arrays to stack must be passed as a "sequence" type '
            "such as list or tuple."
        )


def _delete_with_slice(params, obj, axis):
    """Utility function for ``dpnp.delete`` when obj is slice."""

    a, a_ndim, order, axis, slobj, n, newshape, exec_q, usm_type = params

    start, stop, step = obj.indices(n)
    xr = range(start, stop, step)
    num_del = len(xr)

    if num_del <= 0:
        return a.copy(order=order)

    # Invert if step is negative:
    if step < 0:
        step = -step
        start = xr[-1]
        stop = xr[0] + 1

    newshape[axis] -= num_del
    new = dpnp.empty(
        newshape,
        order=order,
        dtype=a.dtype,
        sycl_queue=exec_q,
        usm_type=usm_type,
    )
    # copy initial chunk
    if start == 0:
        pass
    else:
        slobj[axis] = slice(None, start)
        new[tuple(slobj)] = a[tuple(slobj)]
    # copy end chunk
    if stop == n:
        pass
    else:
        slobj[axis] = slice(stop - num_del, None)
        slobj2 = [slice(None)] * a_ndim
        slobj2[axis] = slice(stop, None)
        new[tuple(slobj)] = a[tuple(slobj2)]
    # copy middle pieces
    if step == 1:
        pass
    else:  # use array indexing.
        keep = dpnp.ones(
            stop - start,
            dtype=dpnp.bool,
            sycl_queue=exec_q,
            usm_type=usm_type,
        )
        keep[: stop - start : step] = False
        slobj[axis] = slice(start, stop - num_del)
        slobj2 = [slice(None)] * a_ndim
        slobj2[axis] = slice(start, stop)
        a = a[tuple(slobj2)]
        slobj2[axis] = keep
        new[tuple(slobj)] = a[tuple(slobj2)]

    return new


def _delete_without_slice(params, obj, axis, single_value):
    """Utility function for ``dpnp.delete`` when obj is int or array of int."""

    a, a_ndim, order, axis, slobj, n, newshape, exec_q, usm_type = params

    if single_value:
        # optimization for a single value
        if obj < -n or obj >= n:
            raise IndexError(
                f"index {obj} is out of bounds for axis {axis} with "
                f"size {n}"
            )
        if obj < 0:
            obj += n
        newshape[axis] -= 1
        new = dpnp.empty(
            newshape,
            order=order,
            dtype=a.dtype,
            sycl_queue=exec_q,
            usm_type=usm_type,
        )
        slobj[axis] = slice(None, obj)
        new[tuple(slobj)] = a[tuple(slobj)]
        slobj[axis] = slice(obj, None)
        slobj2 = [slice(None)] * a_ndim
        slobj2[axis] = slice(obj + 1, None)
        new[tuple(slobj)] = a[tuple(slobj2)]
    else:
        if obj.dtype == dpnp.bool:
            if obj.shape != (n,):
                raise ValueError(
                    "boolean array argument `obj` to delete must be "
                    f"one-dimensional and match the axis length of {n}"
                )

            # optimization, the other branch is slower
            keep = ~obj
        else:
            keep = dpnp.ones(
                n, dtype=dpnp.bool, sycl_queue=exec_q, usm_type=usm_type
            )
            keep[obj,] = False

        slobj[axis] = keep
        new = a[tuple(slobj)]

    return new


def _calc_parameters(a, axis, obj, values=None):
    """Utility function for ``dpnp.delete`` and ``dpnp.insert``."""

    a_ndim = a.ndim
    order = "F" if a.flags.fnc else "C"
    if axis is None:
        if a_ndim != 1:
            a = dpnp.ravel(a)
        a_ndim = 1
        axis = 0
    else:
        axis = normalize_axis_index(axis, a_ndim)

    slobj = [slice(None)] * a_ndim
    n = a.shape[axis]
    a_shape = list(a.shape)

    usm_type, exec_q = get_usm_allocations([a, obj, values])

    return InsertDeleteParams(
        a, a_ndim, order, axis, slobj, n, a_shape, exec_q, usm_type
    )


def _insert_array_indices(parameters, indices, values, obj):
    """
    Utility function for ``dpnp.insert`` when indices is an array with
    multiple elements.

    """

    a, a_ndim, order, axis, slobj, n, newshape, exec_q, usm_type = parameters

    is_array = isinstance(obj, (dpnp_array, numpy.ndarray, dpt.usm_ndarray))
    if indices.size == 0 and not is_array:
        # Can safely cast the empty list to intp
        indices = indices.astype(dpnp.intp)

    indices[indices < 0] += n

    numnew = len(indices)
    ind_sort = indices.argsort(kind="stable")
    indices[ind_sort] += dpnp.arange(
        numnew, dtype=indices.dtype, sycl_queue=exec_q, usm_type=usm_type
    )

    newshape[axis] += numnew
    old_mask = dpnp.ones(
        newshape[axis], dtype=dpnp.bool, sycl_queue=exec_q, usm_type=usm_type
    )
    old_mask[indices] = False

    new = dpnp.empty(
        newshape,
        order=order,
        dtype=a.dtype,
        sycl_queue=exec_q,
        usm_type=usm_type,
    )
    slobj2 = [slice(None)] * a_ndim
    slobj[axis] = indices
    slobj2[axis] = old_mask
    new[tuple(slobj)] = values
    new[tuple(slobj2)] = a

    return new


def _insert_singleton_index(parameters, indices, values, obj):
    """
    Utility function for ``dpnp.insert`` when indices is an array with
    one element.

    """

    a, a_ndim, order, axis, slobj, n, newshape, exec_q, usm_type = parameters

    # In dpnp, `.item()` calls `.wait()`, so it is preferred to avoid it
    # When possible (i.e. for numpy arrays, lists, etc), it is preferred
    # to use `.item()` on a NumPy array
    if dpnp.is_supported_array_type(obj):
        index = indices.item()
    else:
        if isinstance(obj, slice):
            obj = numpy.arange(*obj.indices(n), dtype=dpnp.intp)
        index = numpy.asarray(obj).item()

    if index < -n or index > n:
        raise IndexError(
            f"index {index} is out of bounds for axis {axis} with size {n}"
        )
    if index < 0:
        index += n

    # Need to change the dtype of values to input array dtype and update
    # its shape to make ``input_arr[..., index, ...] = values`` legal
    values = dpnp.array(
        values,
        copy=None,
        ndmin=a_ndim,
        dtype=a.dtype,
        sycl_queue=exec_q,
        usm_type=usm_type,
    )
    if indices.ndim == 0:
        # numpy.insert behave differently if obj is an scalar or an array
        # with one element, so, this change is needed to align with NumPy
        values = dpnp.moveaxis(values, 0, axis)

    numnew = values.shape[axis]
    newshape[axis] += numnew
    new = dpnp.empty(
        newshape,
        order=order,
        dtype=a.dtype,
        sycl_queue=exec_q,
        usm_type=usm_type,
    )

    slobj[axis] = slice(None, index)
    new[tuple(slobj)] = a[tuple(slobj)]
    slobj[axis] = slice(index, index + numnew)
    new[tuple(slobj)] = values
    slobj[axis] = slice(index + numnew, None)
    slobj2 = [slice(None)] * a_ndim
    slobj2[axis] = slice(index, None)
    new[tuple(slobj)] = a[tuple(slobj2)]

    return new


def _unique_1d(
    ar,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    equal_nan=True,
):
    """Find the unique elements of a 1D array."""

    def _get_first_nan_index(usm_a):
        """
        Find the first index of NaN in the input array with at least two NaNs.

        Assume the input array sorted where the NaNs are always at the end.
        Return None if the input array does not have at least two NaN values or
        data type of the array is not inexact.

        """

        if (
            usm_a.size > 2
            and dpnp.issubdtype(usm_a.dtype, dpnp.inexact)
            and dpnp.isnan(usm_a[-2])
        ):
            if dpnp.issubdtype(usm_a.dtype, dpnp.complexfloating):
                # for complex all NaNs are considered equivalent
                true_val = dpt.asarray(
                    True, sycl_queue=usm_a.sycl_queue, usm_type=usm_a.usm_type
                )
                return dpt.searchsorted(dpt.isnan(usm_a), true_val, side="left")
            return dpt.searchsorted(usm_a, usm_a[-1], side="left")
        return None

    usm_ar = dpnp.get_usm_ndarray(ar)

    num_of_flags = (return_index, return_inverse, return_counts).count(True)
    if num_of_flags == 0:
        usm_res = dpt.unique_values(usm_ar)
        usm_res = (usm_res,)  # cast to a tuple to align with other cases
    elif num_of_flags == 1 and return_inverse:
        usm_res = dpt.unique_inverse(usm_ar)
    elif num_of_flags == 1 and return_counts:
        usm_res = dpt.unique_counts(usm_ar)
    else:
        usm_res = dpt.unique_all(usm_ar)

    first_nan = None
    if equal_nan:
        first_nan = _get_first_nan_index(usm_res[0])

    # collapse multiple NaN values in an array into one NaN value if applicable
    result = (
        usm_res[0][: first_nan + 1] if first_nan is not None else usm_res[0],
    )
    if return_index:
        result += (
            (
                usm_res.indices[: first_nan + 1]
                if first_nan is not None
                else usm_res.indices
            ),
        )
    if return_inverse:
        if first_nan is not None:
            # all NaNs are collapsed, so need to replace the indices with
            # the index of the first NaN value in result array of unique values
            dpt.place(
                usm_res.inverse_indices,
                usm_res.inverse_indices > first_nan,
                dpt.reshape(first_nan, 1),
            )

        result += (usm_res.inverse_indices,)
    if return_counts:
        if first_nan is not None:
            # all NaNs are collapsed, so need to put a count of all NaNs
            # at the last index
            dpt.sum(usm_res.counts[first_nan:], out=usm_res.counts[first_nan])
            result += (usm_res.counts[: first_nan + 1],)
        else:
            result += (usm_res.counts,)

    result = tuple(dpnp_array._create_from_usm_ndarray(x) for x in result)
    return _unpack_tuple(result)


def _unique_build_sort_indices(a, index_sh):
    """
    Build the indices of an input array (when axis is provided) which result
    in the unique array.

    """

    is_inexact = dpnp.issubdtype(a, dpnp.inexact)
    if dpnp.issubdtype(a.dtype, numpy.unsignedinteger):
        ar_cmp = a.astype(dpnp.intp)
    elif dpnp.issubdtype(a.dtype, dpnp.bool):
        ar_cmp = a.astype(numpy.int8)
    else:
        ar_cmp = a

    def compare_axis_elems(idx1, idx2):
        comp = dpnp.trim_zeros(ar_cmp[idx1] - ar_cmp[idx2], "f")
        if comp.shape[0] > 0:
            diff = comp[0]
            if is_inexact and dpnp.isnan(diff):
                isnan1 = dpnp.isnan(ar_cmp[idx1])
                if not isnan1.any():  # no NaN in ar_cmp[idx1]
                    return True  # ar_cmp[idx1] goes to left

                isnan2 = dpnp.isnan(ar_cmp[idx2])
                if not isnan2.any():  # no NaN in ar_cmp[idx2]
                    return False  # ar_cmp[idx1] goes to right

                # for complex all NaNs are considered equivalent
                if (isnan1 & isnan2).all():  # NaNs at the same places
                    return False  # ar_cmp[idx1] goes to right

                xor_nan_idx = dpnp.where(isnan1 ^ isnan2)[0]
                if xor_nan_idx.size == 0:
                    return False

                if dpnp.isnan(ar_cmp[idx2][xor_nan_idx[0]]):
                    # first NaN in XOR mask is from ar_cmp[idx2]
                    return True  # ar_cmp[idx1] goes to left
                return False
            return diff < 0
        return False

    # sort the array `a` lexicographically using the first item
    # of each element on the axis
    sorted_indices = dpnp.empty_like(a, shape=index_sh, dtype=dpnp.intp)
    queue = [(numpy.arange(0, index_sh, dtype=numpy.intp).tolist(), 0)]
    while len(queue) != 0:
        current, off = queue.pop(0)
        if len(current) == 0:
            continue

        mid_elem = current[0]
        left = []
        right = []
        for i in range(1, len(current)):
            if compare_axis_elems(current[i], mid_elem):
                left.append(current[i])
            else:
                right.append(current[i])

        elem_pos = off + len(left)
        queue.append((left, off))
        queue.append((right, elem_pos + 1))

        sorted_indices[elem_pos] = mid_elem
    return sorted_indices


def _unpack_tuple(a):
    """Unpacks one-element tuples for use as return values."""

    if len(a) == 1:
        return a[0]
    return a


def append(arr, values, axis=None):
    """
    Append values to the end of an array.

    For full documentation refer to :obj:`numpy.append`.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        Values are appended to a copy of this array.
    values : {scalar, array_like}
        These values are appended to a copy of `arr`. It must be of the correct
        shape (the same shape as `arr`, excluding `axis`). If `axis` is not
        specified, `values` can be any shape and will be flattened before use.
        These values can be in any form that can be converted to an array. This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    axis : {None, int}, optional
        The axis along which `values` are appended. If `axis` is not given,
        both `arr` and `values` are flattened before use.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        A copy of `arr` with `values` appended to `axis`. Note that
        `append` does not occur in-place: a new array is allocated and
        filled. If `axis` is ``None``, `out` is a flattened array.

    See Also
    --------
    :obj:`dpnp.insert` : Insert elements into an array.
    :obj:`dpnp.delete` : Delete elements from an array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> np.append(a, [[4, 5, 6], [7, 8, 9]])
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    When `axis` is specified, `values` must have the correct shape.

    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.append(b, [[7, 8, 9]], axis=0)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])
    >>> np.append(b, [7, 8, 9], axis=0)
    Traceback (most recent call last):
        ...
    ValueError: all the input arrays must have same number of dimensions, but
    the array at index 0 has 2 dimension(s) and the array at index 1 has 1
    dimension(s)

    """

    dpnp.check_supported_arrays_type(arr)
    if not dpnp.is_supported_array_type(values):
        values = dpnp.array(
            values, usm_type=arr.usm_type, sycl_queue=arr.sycl_queue
        )

    if axis is None:
        if arr.ndim != 1:
            arr = dpnp.ravel(arr)
        if values.ndim != 1:
            values = dpnp.ravel(values)
        axis = 0
    return dpnp.concatenate((arr, values), axis=axis)


def array_split(ary, indices_or_sections, axis=0):
    """
    Split an array into multiple sub-arrays.

    Please refer to the :obj:`dpnp.split` documentation. The only difference
    between these functions is that ``dpnp.array_split`` allows
    `indices_or_sections` to be an integer that does *not* equally divide the
    axis. For an array of length l that should be split into n sections, it
    returns ``l % n`` sub-arrays of size ``l//n + 1`` and the rest of size
    ``l//n``.

    For full documentation refer to :obj:`numpy.array_split`.

    Parameters
    ----------
    ary : {dpnp.ndarray, usm_ndarray}
        Array to be divided into sub-arrays.
    indices_or_sections : {int, sequence of ints}
        If `indices_or_sections` is an integer, N, and array length is l, it
        returns ``l % n`` sub-arrays of size ``l//n + 1`` and the rest of size
        ``l//n``.
        If `indices_or_sections` is a sequence of sorted integers, the entries
        indicate where along `axis` the array is split.
    axis : int, optional
        The axis along which to split.

        Default: ``0``.

    Returns
    -------
    sub-arrays : list of dpnp.ndarray
        A list of sub arrays. Each array is a view of the corresponding input
        array.

    See Also
    --------
    :obj:`dpnp.split` : Split array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(8.0)
    >>> np.array_split(x, 3)
    [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7.])]

    >>> x = np.arange(9)
    >>> np.array_split(x, 4)
    [array([0, 1, 2]), array([3, 4]), array([5, 6]), array([7, 8])]

    """

    dpnp.check_supported_arrays_type(ary)
    n_tot = ary.shape[axis]
    try:
        # handle array case.
        n_sec = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [n_tot]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        n_sec = int(indices_or_sections)
        if n_sec <= 0:
            raise ValueError("number sections must be larger than 0.") from None
        n_each_sec, extras = numpy.divmod(n_tot, n_sec)
        section_sizes = (
            [0] + extras * [n_each_sec + 1] + (n_sec - extras) * [n_each_sec]
        )
        div_points = dpnp.array(
            section_sizes,
            dtype=dpnp.intp,
            usm_type=ary.usm_type,
            sycl_queue=ary.sycl_queue,
        ).cumsum()

    sub_arys = []
    sary = dpnp.swapaxes(ary, axis, 0)
    for i in range(n_sec):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(dpnp.swapaxes(sary[st:end], axis, 0))

    return sub_arys


def asarray_chkfinite(
    a, dtype=None, order=None, *, device=None, usm_type=None, sycl_queue=None
):
    """
    Convert the input to an array, checking for NaNs or Infs.

    For full documentation refer to :obj:`numpy.asarray_chkfinite`.

    Parameters
    ----------
    arr : array_like
        Input data, in any form that can be converted to an array. This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays. Success requires no NaNs or Infs.
    dtype : {None, str, dtype object}, optional
        By default, the data-type is inferred from the input data.

        Default: ``None``.
    order : {None, "C", "F", "A", "K"}, optional
        Memory layout of the newly output array.

        Default: ``"K"``.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpctl.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.

        Default: ``None``.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Array interpretation of `a`. No copy is performed if the input is
        already an ndarray.

    Raises
    -------
    ValueError
        Raises ``ValueError`` if `a` contains NaN (Not a Number) or
        Inf (Infinity).

    See Also
    --------
    :obj:`dpnp.asarray` : Create an array.
    :obj:`dpnp.asanyarray` : Converts an input object into array.
    :obj:`dpnp.ascontiguousarray` : Convert input to a c-contiguous array.
    :obj:`dpnp.asfortranarray` : Convert input to an array with column-major
                        memory order.
    :obj:`dpnp.fromiter` : Create an array from an iterator.
    :obj:`dpnp.fromfunction` : Construct an array by executing a function
                        on grid positions.

    Examples
    --------
    >>> import dpnp as np

    Convert a list into an array. If all elements are finite,
    ``asarray_chkfinite`` is identical to ``asarray``.

    >>> a = [1, 2]
    >>> np.asarray_chkfinite(a, dtype=np.float32)
    array([1., 2.])

    Raises ``ValueError`` if array_like contains NaNs or Infs.

    >>> a = [1, 2, np.inf]
    >>> try:
    ...     np.asarray_chkfinite(a)
    ... except ValueError:
    ...     print('ValueError')
    ValueError

    Creating an array on a different device or with a specified usm_type

    >>> x = np.asarray_chkfinite([1, 2, 3]) # default case
    >>> x, x.device, x.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'device')

    >>> y = np.asarray_chkfinite([1, 2, 3], device="cpu")
    >>> y, y.device, y.usm_type
    (array([1, 2, 3]), Device(opencl:cpu:0), 'device')

    >>> z = np.asarray_chkfinite([1, 2, 3], usm_type="host")
    >>> z, z.device, z.usm_type
    (array([1, 2, 3]), Device(level_zero:gpu:0), 'host')

    """

    a = dpnp.asarray(
        a,
        dtype=dtype,
        order=order,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )
    if dpnp.issubdtype(a.dtype, dpnp.inexact) and not dpnp.isfinite(a).all():
        raise ValueError("array must not contain infs or NaNs")
    return a


def asfarray(a, dtype=None, *, device=None, usm_type=None, sycl_queue=None):
    """
    Return an array converted to a float type.

    For full documentation refer to :obj:`numpy.asfarray`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.
        This includes an instance of :class:`dpnp.ndarray` or
        :class:`dpctl.tensor.usm_ndarray`, an object representing
        SYCL USM allocation and implementing `__sycl_usm_array_interface__`
        protocol, an instance of :class:`numpy.ndarray`, an object supporting
        Python buffer protocol, a Python scalar, or a (possibly nested)
        sequence of Python scalars.
    dtype : {None, str, dtype object}, optional
        Float type code to coerce input array `a`.  If `dtype` is ``None``,
        :obj:`dpnp.bool` or one of the `int` dtypes, it is replaced with
        the default floating type (:obj:`dpnp.float64` if a device supports it,
        or :obj:`dpnp.float32` type otherwise).

        Default: ``None``.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpctl.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.

        Default: ``None``.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The input `a` as a float ndarray.

    Examples
    --------
    >>> import dpnp as np
    >>> np.asfarray([2, 3])
    array([2.,  3.])
    >>> np.asfarray([2, 3], dtype=dpnp.float32)
    array([2., 3.], dtype=float32)
    >>> np.asfarray([2, 3], dtype=dpnp.int32)
    array([2.,  3.])

    """

    _sycl_queue = dpnp.get_normalized_queue_device(
        a, sycl_queue=sycl_queue, device=device
    )

    if dtype is None or not dpnp.issubdtype(dtype, dpnp.inexact):
        dtype = dpnp.default_float_type(sycl_queue=_sycl_queue)

    return dpnp.asarray(
        a, dtype=dtype, usm_type=usm_type, sycl_queue=_sycl_queue
    )


def atleast_1d(*arys):
    """
    Convert inputs to arrays with at least one dimension.

    For full documentation refer to :obj:`numpy.atleast_1d`.

    Parameters
    ----------
    arys : {dpnp.ndarray, usm_ndarray}
        One or more array-like sequences. Arrays that already have one or more
        dimensions are preserved.

    Returns
    -------
    out : dpnp.ndarray
        An array, or list of arrays, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    :obj:`dpnp.atleast_2d` : View inputs as arrays with at least two dimensions.
    :obj:`dpnp.atleast_3d` : View inputs as arrays with at least three
                             dimensions.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array(1.0)
    >>> np.atleast_1d(x)
    array([1.])

    >>> y = np.array([3, 4])
    >>> np.atleast_1d(x, y)
    [array([1.]), array([3, 4])]

    >>> x = np.arange(9.0).reshape(3, 3)
    >>> np.atleast_1d(x)
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> np.atleast_1d(x) is x
    True

    """

    res = []
    dpnp.check_supported_arrays_type(*arys)
    for ary in arys:
        if ary.ndim == 0:
            # 0-d arrays cannot be empty
            # 0-d arrays always have a size of 1, so
            # reshape(1) is guaranteed to succeed
            result = ary.reshape(1)
        else:
            result = ary
        if isinstance(result, dpt.usm_ndarray):
            result = dpnp_array._create_from_usm_ndarray(result)
        res.append(result)
    if len(res) == 1:
        return res[0]
    return res


def atleast_2d(*arys):
    """
    View inputs as arrays with at least two dimensions.

    For full documentation refer to :obj:`numpy.atleast_2d`.

    Parameters
    ----------
    arys : {dpnp.ndarray, usm_ndarray}
        One or more array-like sequences. Arrays that already have two or more
        dimensions are preserved.

    Returns
    -------
    out : dpnp.ndarray
        An array, or list of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    See Also
    --------
    :obj:`dpnp.atleast_1d` : Convert inputs to arrays with at least one
                             dimension.
    :obj:`dpnp.atleast_3d` : View inputs as arrays with at least three
                             dimensions.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array(3.0)
    >>> np.atleast_2d(x)
    array([[3.]])

    >>> x = np.arange(3.0)
    >>> np.atleast_2d(x)
    array([[0., 1., 2.]])

    """

    res = []
    dpnp.check_supported_arrays_type(*arys)
    for ary in arys:
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[dpnp.newaxis, :]
        else:
            result = ary
        if isinstance(result, dpt.usm_ndarray):
            result = dpnp_array._create_from_usm_ndarray(result)
        res.append(result)
    if len(res) == 1:
        return res[0]
    return res


def atleast_3d(*arys):
    """
    View inputs as arrays with at least three dimensions.

    For full documentation refer to :obj:`numpy.atleast_3d`.

    Parameters
    ----------
    arys : {dpnp.ndarray, usm_ndarray}
        One or more array-like sequences. Arrays that already have three or more
        dimensions are preserved.

    Returns
    -------
    out : dpnp.ndarray
        An array, or list of arrays, each with ``a.ndim >= 3``. Copies are
        avoided where possible, and views with three or more dimensions are
        returned.

    See Also
    --------
    :obj:`dpnp.atleast_1d` : Convert inputs to arrays with at least one
                             dimension.
    :obj:`dpnp.atleast_2d` : View inputs as arrays with at least three
                             dimensions.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array(3.0)
    >>> np.atleast_3d(x)
    array([[[3.]]])

    >>> x = np.arange(3.0)
    >>> np.atleast_3d(x).shape
    (1, 3, 1)

    >>> x = np.arange(12.0).reshape(4, 3)
    >>> np.atleast_3d(x).shape
    (4, 3, 1)

    """

    res = []
    dpnp.check_supported_arrays_type(*arys)
    for ary in arys:
        if ary.ndim == 0:
            result = ary.reshape(1, 1, 1)
        elif ary.ndim == 1:
            result = ary[dpnp.newaxis, :, dpnp.newaxis]
        elif ary.ndim == 2:
            result = ary[:, :, dpnp.newaxis]
        else:
            result = ary
        if isinstance(result, dpt.usm_ndarray):
            result = dpnp_array._create_from_usm_ndarray(result)
        res.append(result)
    if len(res) == 1:
        return res[0]
    return res


def broadcast_arrays(*args, subok=False):
    """
    Broadcast any number of arrays against each other.

    For full documentation refer to :obj:`numpy.broadcast_arrays`.

    Parameters
    ----------
    args : {dpnp.ndarray, usm_ndarray}
        A list of arrays to broadcast.

    Returns
    -------
    out : list of dpnp.ndarray
        A list of arrays which are views on the original arrays from `args`.

    Limitations
    -----------
    Parameter `subok` is supported with default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.broadcast_to` : Broadcast an array to a new shape.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[1, 2, 3]])
    >>> y = np.array([[4], [5]])
    >>> np.broadcast_arrays(x, y)
    [array([[1, 2, 3],
            [1, 2, 3]]), array([[4, 4, 4],
            [5, 5, 5]])]

    """

    if subok is not False:
        raise NotImplementedError(f"subok={subok} is currently not supported")

    if len(args) == 0:
        return []

    usm_arrays = dpt.broadcast_arrays(*[dpnp.get_usm_ndarray(a) for a in args])
    return [dpnp_array._create_from_usm_ndarray(a) for a in usm_arrays]


def broadcast_shapes(*args):
    """
    Broadcast the input shapes into a single shape.

    For full documentation refer to :obj:`numpy.broadcast_shapes`.

    Parameters
    ----------
    *args : tuples of ints, or ints
        The shapes to be broadcast against each other.

    Returns
    -------
    out : tuple
        Broadcasted shape.

    See Also
    --------
    :obj:`dpnp.broadcast_arrays` : Broadcast any number of arrays against
                                   each other.
    :obj:`dpnp.broadcast_to` : Broadcast an array to a new shape.

    Examples
    --------
    >>> import dpnp as np
    >>> np.broadcast_shapes((1, 2), (3, 1), (3, 2))
    (3, 2)
    >>> np.broadcast_shapes((6, 7), (5, 6, 1), (7,), (5, 1, 7))
    (5, 6, 7)

    """

    return numpy.broadcast_shapes(*args)


# pylint: disable=redefined-outer-name
def broadcast_to(array, /, shape, subok=False):
    """
    Broadcast an array to a new shape.

    For full documentation refer to :obj:`numpy.broadcast_to`.

    Parameters
    ----------
    array : {dpnp.ndarray, usm_ndarray}
        The array to broadcast.
    shape : {int, tuple of ints}
        The shape of the desired array. A single integer ``i`` is interpreted
        as ``(i,)``.

    Returns
    -------
    out : dpnp.ndarray
        An array having a specified shape.
        Must have the same data type as `array`.

    Limitations
    -----------
    Parameter `subok` is supported with default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.broadcast_arrays` : Broadcast any number of arrays against
                                   each other.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2, 3])
    >>> np.broadcast_to(x, (3, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])

    """

    if subok is not False:
        raise NotImplementedError(f"subok={subok} is currently not supported")

    usm_array = dpnp.get_usm_ndarray(array)
    new_array = dpt.broadcast_to(usm_array, shape)
    return dpnp_array._create_from_usm_ndarray(new_array)


def can_cast(from_, to, casting="safe"):
    """
    Returns ``True`` if cast between data types can occur according
    to the casting rule.

    For full documentation refer to :obj:`numpy.can_cast`.

    Parameters
    ----------
    from_ : {dpnp.ndarray, usm_ndarray, dtype, dtype specifier}
        Source data type.
    to : dtype
        Target data type.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

        Default: ``"safe"``.

    Returns
    -------
    out: bool
        ``True`` if cast can occur according to the casting rule,
        ``False`` otherwise.

    See Also
    --------
    :obj:`dpnp.result_type` : Returns the type that results from applying
                              the NumPy type promotion rules to the arguments.

    Examples
    --------
    Basic examples

    >>> import dpnp as np
    >>> np.can_cast(np.int32, np.int64)
    True
    >>> np.can_cast(np.float64, complex)
    True
    >>> np.can_cast(complex, float)
    False

    >>> np.can_cast('i8', 'f8')
    True
    >>> np.can_cast('i8', 'f4')
    False

    Array scalar checks the value, array does not

    >>> np.can_cast(np.array(1000.0), np.float32)
    True
    >>> np.can_cast(np.array([1000.0]), np.float32)
    False

    Using the casting rules

    >>> np.can_cast('i8', 'i8', 'no')
    True
    >>> np.can_cast('<i8', '>i8', 'no')
    False

    >>> np.can_cast('<i8', '>i8', 'equiv')
    True
    >>> np.can_cast('<i4', '>i8', 'equiv')
    False

    >>> np.can_cast('<i4', '>i8', 'safe')
    True
    >>> np.can_cast('<i8', '>i4', 'safe')
    False

    >>> np.can_cast('<i8', '>i4', 'same_kind')
    True
    >>> np.can_cast('<i8', '>u4', 'same_kind')
    False

    >>> np.can_cast('<i8', '>u4', 'unsafe')
    True

    """

    if dpnp.is_supported_array_type(to):
        raise TypeError("Cannot construct a dtype from an array")

    dtype_from = (
        from_.dtype
        if dpnp.is_supported_array_type(from_)
        else dpnp.dtype(from_)
    )
    return dpt.can_cast(dtype_from, to, casting=casting)


def column_stack(tup):
    """
    Stacks 1-D and 2-D arrays as columns into a 2-D array.

    Take a sequence of 1-D arrays and stack them as columns to make a single
    2-D array. 2-D arrays are stacked as-is, just like with :obj:`dpnp.hstack`.
    1-D arrays are turned into 2-D columns first.

    For full documentation refer to :obj:`numpy.column_stack`.

    Parameters
    ----------
    tup : {dpnp.ndarray, usm_ndarray}
        A sequence of 1-D or 2-D arrays to stack. All of them must have
        the same first dimension.

    Returns
    -------
    out : dpnp.ndarray
        The array formed by stacking the given arrays.

    See Also
    --------
    :obj:`dpnp.stack` : Join a sequence of arrays along a new axis.
    :obj:`dpnp.dstack` : Stack arrays in sequence depth wise (along third axis).
    :obj:`dpnp.hstack` : Stack arrays in sequence horizontally (column wise).
    :obj:`dpnp.vstack` : Stack arrays in sequence vertically (row wise).
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array((1, 2, 3))
    >>> b = np.array((2, 3, 4))
    >>> np.column_stack((a, b))
    array([[1, 2],
           [2, 3],
           [3, 4]])

    """

    _check_stack_arrays(tup)

    arrays = []
    for v in tup:
        dpnp.check_supported_arrays_type(v)

        if v.ndim == 1:
            v = v[:, dpnp.newaxis]
        elif v.ndim != 2:
            raise ValueError(
                "Only 1 or 2 dimensional arrays can be column stacked"
            )

        arrays.append(v)
    return dpnp.concatenate(arrays, axis=1)


def concatenate(
    arrays, /, *, axis=0, out=None, dtype=None, casting="same_kind"
):
    """
    Join a sequence of arrays along an existing axis.

    Note that :obj:`dpnp.concat` is an alias of :obj:`dpnp.concatenate`.

    For full documentation refer to :obj:`numpy.concatenate`.

    Parameters
    ----------
    arrays : {Sequence of dpnp.ndarray or usm_ndarray}
        The arrays must have the same shape, except in the dimension
        corresponding to axis (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined. If axis is ``None``,
        arrays are flattened before use.

        Default: ``0``.
    out : dpnp.ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned
        if no out argument were specified.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        Default: ``None``.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

        Default: ``"same_kind"``.

    Returns
    -------
    out : dpnp.ndarray
        The concatenated array.

    See Also
    --------
    :obj:`dpnp.array_split` : Split an array into multiple sub-arrays of equal
                              or near-equal size.
    :obj:`dpnp.split` : Split array into a list of multiple sub-arrays of equal
                        size.
    :obj:`dpnp.hsplit` : Split array into multiple sub-arrays horizontally
                         (column wise).
    :obj:`dpnp.vsplit` : Split array into multiple sub-arrays vertically
                         (row wise).
    :obj:`dpnp.dsplit` : Split array into multiple sub-arrays along
                         the 3rd axis (depth).
    :obj:`dpnp.stack` : Stack a sequence of arrays along a new axis.
    :obj:`dpnp.block` : Assemble arrays from blocks.
    :obj:`dpnp.hstack` : Stack arrays in sequence horizontally (column wise).
    :obj:`dpnp.vstack` : Stack arrays in sequence vertically (row wise).
    :obj:`dpnp.dstack` : Stack arrays in sequence depth wise
                         (along third dimension).
    :obj:`dpnp.column_stack` : Stack 1-D arrays as columns into a 2-D array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6]])
    >>> np.concatenate((a, b), axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> np.concatenate((a, b.T), axis=1)
    array([[1, 2, 5],
           [3, 4, 6]])
    >>> np.concatenate((a, b), axis=None)
    array([1, 2, 3, 4, 5, 6])

    """

    if dtype is not None and out is not None:
        raise TypeError(
            "concatenate() only takes `out` or `dtype` as an argument, "
            "but both were provided."
        )

    usm_arrays = [dpnp.get_usm_ndarray(x) for x in arrays]
    usm_res = dpt.concat(usm_arrays, axis=axis)

    res = dpnp_array._create_from_usm_ndarray(usm_res)
    if dtype is not None:
        res = res.astype(dtype, casting=casting, copy=False)
    elif out is not None:
        dpnp.copyto(out, res, casting=casting)
        return out
    return res


concat = concatenate  # concat is an alias of concatenate


def copyto(dst, src, casting="same_kind", where=True):
    """
    Copies values from one array to another, broadcasting as necessary.

    Raises a ``TypeError`` if the `casting` rule is violated, and if
    `where` is provided, it selects which elements to copy.

    For full documentation refer to :obj:`numpy.copyto`.

    Parameters
    ----------
    dst : {dpnp.ndarray, usm_ndarray}
        The array into which values are copied.
    src : array_like
        The array from which values are copied.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when copying.

        Default: ``"same_kind"``.
    where : {dpnp.ndarray, usm_ndarray, scalar} of bool, optional
        A boolean array or a scalar which is broadcasted to match
        the dimensions of `dst`, and selects elements to copy
        from `src` to `dst` wherever it contains the value ``True``.

        Default: ``True``.

    Examples
    --------
    >>> import dpnp as np
    >>> A = np.array([4, 5, 6])
    >>> B = [1, 2, 3]
    >>> np.copyto(A, B)
    >>> A
    array([1, 2, 3])

    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> B = [[4, 5, 6], [7, 8, 9]]
    >>> np.copyto(A, B)
    >>> A
    array([[4, 5, 6],
           [7, 8, 9]])

    """

    if not dpnp.is_supported_array_type(dst):
        raise TypeError(
            "Destination array must be any of supported type, "
            f"but got {type(dst)}"
        )
    if not dpnp.is_supported_array_type(src):
        python_sc = dpnp.isscalar(src) and not isinstance(src, numpy.generic)
        src = dpnp.array(src, sycl_queue=dst.sycl_queue)
        if python_sc:
            # Python scalar needs special handling to behave similar to NumPy
            if dpnp.issubdtype(src, dpnp.integer) and dpnp.issubdtype(
                dst, dpnp.unsignedinteger
            ):
                if dpnp.any(src < 0):
                    raise OverflowError(
                        "Cannot copy negative values to an unsigned int array"
                    )

                src = src.astype(dst.dtype)

    if not dpnp.can_cast(src.dtype, dst.dtype, casting=casting):
        raise TypeError(
            f"Cannot cast from {src.dtype} to {dst.dtype} "
            f"according to the rule {casting}."
        )

    if where is True:
        dst[...] = src
    elif where is False:
        # nothing to copy
        pass
    else:
        if dpnp.isscalar(where):
            where = dpnp.array(
                where, dtype=dpnp.bool, sycl_queue=dst.sycl_queue
            )
        elif not dpnp.is_supported_array_type(where):
            raise TypeError(
                "`where` array must be any of supported type, "
                f"but got {type(where)}"
            )
        elif where.dtype != dpnp.bool:
            raise TypeError(
                "`where` keyword argument must be of boolean type, "
                f"but got {where.dtype}"
            )

        dst_usm, src_usm, mask_usm = dpt.broadcast_arrays(
            dpnp.get_usm_ndarray(dst),
            dpnp.get_usm_ndarray(src),
            dpnp.get_usm_ndarray(where),
        )
        dst_usm[mask_usm] = src_usm[mask_usm]


def delete(arr, obj, axis=None):
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by
    ``arr[obj]``.

    For full documentation refer to :obj:`numpy.delete`.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        Input array.
    obj : {slice, int, array-like of ints or boolean}
        Indicate indices of sub-arrays to remove along the specified axis.
        Boolean indices are treated as a mask of elements to remove.
    axis : {None, int}, optional
        The axis along which to delete the subarray defined by `obj`.
        If `axis` is ``None``, `obj` is applied to the flattened array.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is ``None``, `out` is
        a flattened array.

    See Also
    --------
    :obj:`dpnp.insert` : Insert elements into an array.
    :obj:`dpnp.append` : Append elements at the end of an array.

    Notes
    -----
    Often it is preferable to use a boolean mask. For example:

    >>> import dpnp as np
    >>> arr = np.arange(12) + 1
    >>> mask = np.ones(len(arr), dtype=np.bool)
    >>> mask[0] = mask[2] = mask[4] = False
    >>> result = arr[mask,...]

    is equivalent to ``np.delete(arr, [0, 2, 4], axis=0)``, but allows further
    use of `mask`.

    Examples
    --------
    >>> import dpnp as np
    >>> arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    >>> arr
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> np.delete(arr, 1, 0)
    array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])

    >>> np.delete(arr, slice(None, None, 2), 1)
    array([[ 2,  4],
           [ 6,  8],
           [10, 12]])
    >>> np.delete(arr, [1, 3, 5], None)
    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])

    """

    dpnp.check_supported_arrays_type(arr)
    params = _calc_parameters(arr, axis, obj)

    if isinstance(obj, slice):
        return _delete_with_slice(params, obj, axis)

    if isinstance(obj, (int, dpnp.integer)) and not isinstance(obj, bool):
        single_value = True
        indices = obj
    else:
        single_value = False
        is_array = isinstance(obj, (dpnp_array, numpy.ndarray, dpt.usm_ndarray))
        indices = dpnp.asarray(
            obj, sycl_queue=params.exec_q, usm_type=params.usm_type
        )
        # if `obj` is originally an empty list, after converting it into
        # an array, it will have float dtype, so we need to change its dtype
        # to integer. However, if `obj` is originally an empty array with
        # float dtype, it is a mistake by user and it will raise an error later
        if indices.size == 0 and not is_array:
            indices = indices.astype(dpnp.intp)
        elif indices.size == 1 and indices.dtype.kind in "ui":
            # For a size 1 integer array we can use the single-value path
            # (most dtypes, except boolean, should just fail later).
            single_value = True
            # In dpnp, `.item()` calls `.wait()`, so it is preferred to avoid it
            # When possible (i.e. for numpy arrays, lists, etc), it is
            # preferred to use `.item()` on a NumPy array
            if dpnp.is_supported_array_type(obj):
                indices = indices.item()
            else:
                indices = numpy.asarray(obj).item()

    return _delete_without_slice(params, indices, axis, single_value)


def dsplit(ary, indices_or_sections):
    """
    Split array into multiple sub-arrays along the 3rd axis (depth).

    Please refer to the :obj:`dpnp.split` documentation. ``dsplit``
    is equivalent to ``split`` with ``axis=2``, the array is always
    split along the third axis provided the array dimension is greater than
    or equal to 3.

    For full documentation refer to :obj:`numpy.dsplit`.

    Parameters
    ----------
    ary : {dpnp.ndarray, usm_ndarray}
        Array to be divided into sub-arrays.
    indices_or_sections : {int, sequence of ints}
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along the third axis. If such a split is not
        possible, an error is raised.
        If `indices_or_sections` is a sequence of sorted integers, the entries
        indicate where along the third axis the array is split.

    Returns
    -------
    sub-arrays : list of dpnp.ndarray
        A list of sub arrays. Each array is a view of the corresponding input
        array.

    See Also
    --------
    :obj:`dpnp.split` : Split array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(16.0).reshape(2, 2, 4)
    >>> x
    array([[[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.]],
           [[ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]]])
    >>> np.dsplit(x, 2)
    [array([[[ 0.,  1.],
             [ 4.,  5.]],
            [[ 8.,  9.],
             [12., 13.]]]),
     array([[[ 2.,  3.],
             [ 6.,  7.]],
            [[10., 11.],
             [14., 15.]]])]
    >>> np.dsplit(x, np.array([3, 6]))
    [array([[[ 0.,  1.,  2.],
             [ 4.,  5.,  6.]],
            [[ 8.,  9., 10.],
             [12., 13., 14.]]]),
     array([[[ 3.],
             [ 7.]],
            [[11.],
             [15.]]]),
     array([])]

    """

    dpnp.check_supported_arrays_type(ary)
    if ary.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return split(ary, indices_or_sections, 2)


def dstack(tup):
    """
    Stack arrays in sequence depth wise (along third axis).

    This is equivalent to concatenation along the third axis after 2-D arrays
    of shape `(M, N)` have been reshaped to `(M, N, 1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1, N, 1)`. Rebuilds arrays divided by
    :obj:`dpnp.dsplit`.

    For full documentation refer to :obj:`numpy.dstack`.

    Parameters
    ----------
    tup : {dpnp.ndarray, usm_ndarray}
        One or more array-like sequences. The arrays must have the same shape
        along all but the third axis. 1-D or 2-D arrays must have the same
        shape.

    Returns
    -------
    out : dpnp.ndarray
        The array formed by stacking the given arrays, will be at least 3-D.

    See Also
    --------
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.
    :obj:`dpnp.vstack` : Stack arrays in sequence vertically (row wise).
    :obj:`dpnp.hstack` : Stack arrays in sequence horizontally (column wise).
    :obj:`dpnp.column_stack` : Stack 1-D arrays as columns into a 2-D array.
    :obj:`dpnp.stack` : Join a sequence of arrays along a new axis.
    :obj:`dpnp.block` : Assemble an ndarray from nested lists of blocks.
    :obj:`dpnp.dsplit` : Split array along third axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array((1, 2, 3))
    >>> b = np.array((2, 3, 4))
    >>> np.dstack((a, b))
    array([[[1, 2],
            [2, 3],
            [3, 4]]])

    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[2], [3], [4]])
    >>> np.dstack((a, b))
    array([[[1, 2]],
           [[2, 3]],
           [[3, 4]]])

    """

    _check_stack_arrays(tup)

    arrs = atleast_3d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return dpnp.concatenate(arrs, axis=2)


def expand_dims(a, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    For full documentation refer to :obj:`numpy.expand_dims`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {int, tuple of ints}
        Position in the expanded axes where the new axis (or axes) is placed.

    Returns
    -------
    out : dpnp.ndarray
        An array with the number of dimensions increased.
        A view is returned whenever possible.

    Notes
    -----
    If `a` has rank (i.e, number of dimensions) `N`, a valid `axis` must reside
    in the closed-interval `[-N-1, N]`.
    If provided a negative `axis`, the `axis` position at which to insert a
    singleton dimension is computed as `N + axis + 1`.
    Hence, if provided `-1`, the resolved axis position is `N` (i.e.,
    a singleton dimension must be appended to the input array `a`).
    If provided `-N-1`, the resolved axis position is `0` (i.e., a
    singleton dimension is added to the input array `a`).

    See Also
    --------
    :obj:`dpnp.squeeze` : The inverse operation, removing singleton dimensions
    :obj:`dpnp.reshape` : Insert, remove, and combine dimensions, and resize
                          existing ones
    :obj:`dpnp.atleast_1d` : Convert inputs to arrays with at least one
                             dimension.
    :obj:`dpnp.atleast_2d` : View inputs as arrays with at least two dimensions.
    :obj:`dpnp.atleast_3d` : View inputs as arrays with at least three
                             dimensions.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 2])
    >>> x.shape
    (2,)

    The following is equivalent to ``x[np.newaxis, :]`` or ``x[np.newaxis]``:

    >>> y = np.expand_dims(x, axis=0)
    >>> y
    array([[1, 2]])
    >>> y.shape
    (1, 2)

    The following is equivalent to ``x[:, np.newaxis]``:

    >>> y = np.expand_dims(x, axis=1)
    >>> y
    array([[1],
           [2]])
    >>> y.shape
    (2, 1)

    ``axis`` may also be a tuple:

    >>> y = np.expand_dims(x, axis=(0, 1))
    >>> y
    array([[[1, 2]]])

    >>> y = np.expand_dims(x, axis=(2, 0))
    >>> y
    array([[[1],
            [2]]])

    Note that some examples may use ``None`` instead of ``np.newaxis``.  These
    are the same objects:

    >>> np.newaxis is None
    True

    """

    usm_a = dpnp.get_usm_ndarray(a)
    usm_res = dpt.expand_dims(usm_a, axis=axis)
    return dpnp_array._create_from_usm_ndarray(usm_res)


def flip(m, axis=None):
    """
    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    For full documentation refer to :obj:`numpy.flip`.

    Parameters
    ----------
    m : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
         Axis or axes along which to flip over. The default,
         ``axis=None``, will flip over all of the axes of the input array.
         If `axis` is negative it counts from the last to the first axis.
         If `axis` is a tuple of integers, flipping is performed on all of
         the axes specified in the tuple.

         Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        A view of `m` with the entries of axis reversed.

    See Also
    --------
    :obj:`dpnp.flipud` : Flip an array vertically (axis=0).
    :obj:`dpnp.fliplr` : Flip an array horizontally (axis=1).

    Examples
    --------
    >>> import dpnp as np
    >>> A = np.arange(8).reshape((2, 2, 2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> np.flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> np.flip(A)
    array([[[7, 6],
            [5, 4]],
           [[3, 2],
            [1, 0]]])
    >>> np.flip(A, (0, 2))
    array([[[5, 4],
            [7, 6]],
           [[1, 0],
            [3, 2]]])
    >>> A = np.random.randn(3, 4, 5)
    >>> np.all(np.flip(A, 2) == A[:, :, ::-1, ...])
    array(True)

    """

    m_usm = dpnp.get_usm_ndarray(m)
    return dpnp_array._create_from_usm_ndarray(dpt.flip(m_usm, axis=axis))


def fliplr(m):
    """
    Reverse the order of elements along axis 1 (left/right).

    For a 2-D array, this flips the entries in each row in the left/right
    direction. Columns are preserved, but appear in a different order than
    before.

    For full documentation refer to :obj:`numpy.fliplr`.

    Parameters
    ----------
    m : {dpnp.ndarray, usm_ndarray}
        Input array, must be at least 2-D.

    Returns
    -------
    out : dpnp.ndarray
        A view of `m` with the columns reversed.

    See Also
    --------
    :obj:`dpnp.flipud` : Flip an array vertically (axis=0).
    :obj:`dpnp.flip` : Flip array in one or more dimensions.
    :obj:`dpnp.rot90` : Rotate array counterclockwise.

    Examples
    --------
    >>> import dpnp as np
    >>> A = np.diag(np.array([1., 2., 3.]))
    >>> A
    array([[1.,  0.,  0.],
           [0.,  2.,  0.],
           [0.,  0.,  3.]])
    >>> np.fliplr(A)
    array([[0.,  0.,  1.],
           [0.,  2.,  0.],
           [3.,  0.,  0.]])

    >>> A = np.random.randn(2, 3, 5)
    >>> np.all(np.fliplr(A) == A[:, ::-1, ...])
    array(True)

    """

    dpnp.check_supported_arrays_type(m)

    if m.ndim < 2:
        raise ValueError(f"Input must be >= 2-d, but got {m.ndim}")
    return m[:, ::-1]


def flipud(m):
    """
    Reverse the order of elements along axis 0 (up/down).

    For a 2-D array, this flips the entries in each column in the up/down
    direction. Rows are preserved, but appear in a different order than before.

    For full documentation refer to :obj:`numpy.flipud`.

    Parameters
    ----------
    m : {dpnp.ndarray, usm_ndarray}
        Input array.

    Returns
    -------
    out : dpnp.ndarray
        A view of `m` with the rows reversed.

    See Also
    --------
    :obj:`dpnp.fliplr` : Flip array in the left/right direction.
    :obj:`dpnp.flip` : Flip array in one or more dimensions.
    :obj:`dpnp.rot90` : Rotate array counterclockwise.

    Examples
    --------
    >>> import dpnp as np
    >>> A = np.diag(np.array([1., 2., 3.]))
    >>> A
    array([[1.,  0.,  0.],
           [0.,  2.,  0.],
           [0.,  0.,  3.]])
    >>> np.flipud(A)
    array([[0.,  0.,  3.],
           [0.,  2.,  0.],
           [1.,  0.,  0.]])

    >>> A = np.random.randn(2, 3, 5)
    >>> np.all(np.flipud(A) == A[::-1, ...])
    array(True)

    >>> np.flipud(np.array([1, 2]))
    array([2, 1])

    """

    dpnp.check_supported_arrays_type(m)

    if m.ndim < 1:
        raise ValueError(f"Input must be >= 1-d, but got {m.ndim}")
    return m[::-1, ...]


def hsplit(ary, indices_or_sections):
    """
    Split an array into multiple sub-arrays horizontally (column-wise).

    Please refer to the :obj:`dpnp.split` documentation. ``hsplit``
    is equivalent to ``dpnp.split`` with ``axis=1``, the array is always
    split along the second axis except for 1-D arrays, where it is split at
    ``axis=0``.

    For full documentation refer to :obj:`numpy.hsplit`.

    Parameters
    ----------
    ary : {dpnp.ndarray, usm_ndarray}
        Array to be divided into sub-arrays.
    indices_or_sections : {int, sequence of ints}
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along the second axis except for 1-D arrays, where
        it is split at the first axis. If such a split is not possible,
        an error is raised.
        If `indices_or_sections` is a sequence of sorted integers, the entries
        indicate where along the second axis the array is split. For 1-D arrays,
        the entries indicate where along the first axis the array is split.

    Returns
    -------
    sub-arrays : list of dpnp.ndarray
        A list of sub arrays. Each array is a view of the corresponding input
        array.

    See Also
    --------
    :obj:`dpnp.split` : Split array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.],
           [12., 13., 14., 15.]])
    >>> np.hsplit(x, 2)
    [array([[ 0.,  1.],
            [ 4.,  5.],
            [ 8.,  9.],
            [12., 13.]]),
     array([[ 2.,  3.],
            [ 6.,  7.],
            [10., 11.],
            [14., 15.]])]
    >>> np.hsplit(x, np.array([3, 6]))
    [array([[ 0.,  1.,  2.],
            [ 4.,  5.,  6.],
            [ 8.,  9., 10.],
            [12., 13., 14.]]),
     array([[ 3.],
            [ 7.],
            [11.],
            [15.]]),
     array([])]

    With a higher dimensional array the split is still along the second axis.

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[0., 1.],
            [2., 3.]],
           [[4., 5.],
            [6., 7.]]])
    >>> np.hsplit(x, 2)
    [array([[[0., 1.]],
            [[4., 5.]]]),
     array([[[2., 3.]],
            [[6., 7.]]])]

    With a 1-D array, the split is along axis 0.

    >>> x = np.array([0, 1, 2, 3, 4, 5])
    >>> np.hsplit(x, 2)
    [array([0, 1, 2]), array([3, 4, 5])]

    """

    dpnp.check_supported_arrays_type(ary)
    if ary.ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    if ary.ndim > 1:
        return split(ary, indices_or_sections, 1)
    return split(ary, indices_or_sections, 0)


def hstack(tup, *, dtype=None, casting="same_kind"):
    """
    Stack arrays in sequence horizontally (column wise).

    For full documentation refer to :obj:`numpy.hstack`.

    Parameters
    ----------
    tup : {dpnp.ndarray, usm_ndarray}
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.
    dtype : {None, str, dtype object}, optional
        If provided, the destination array will have this dtype.

        Default: ``None``.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        Default: ``"same_kind"``.

    Returns
    -------
    out : dpnp.ndarray
        The stacked array which has one more dimension than the input arrays.

    See Also
    --------
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.
    :obj:`dpnp.stack` : Join a sequence of arrays along a new axis.
    :obj:`dpnp.vstack` : Stack arrays in sequence vertically (row wise).
    :obj:`dpnp.dstack` : Stack arrays in sequence depth wise
                         (along third dimension).
    :obj:`dpnp.column_stack` : Stack 1-D arrays as columns into a 2-D array.
    :obj:`dpnp.block` : Assemble an ndarray from nested lists of blocks.
    :obj:`dpnp.split` : Split array into a list of multiple sub-arrays of equal
                        size.
    :obj:`dpnp.unstack` : Split an array into a tuple of sub-arrays along
                          an axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array((1, 2, 3))
    >>> b = np.array((4, 5, 6))
    >>> np.hstack((a, b))
    array([1, 2, 3, 4, 5, 6])

    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[4], [5], [6]])
    >>> np.hstack((a, b))
    array([[1, 4],
           [2, 5],
           [3, 6]])

    """

    _check_stack_arrays(tup)

    arrs = dpnp.atleast_1d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]

    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs and arrs[0].ndim == 1:
        return dpnp.concatenate(arrs, axis=0, dtype=dtype, casting=casting)
    return dpnp.concatenate(arrs, axis=1, dtype=dtype, casting=casting)


def insert(arr, obj, values, axis=None):
    """
    Insert values along the given axis before the given indices.

    For full documentation refer to :obj:`numpy.insert`.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : {slice, int, array-like of ints or bools}
        Object that defines the index or indices before which `values` is
        inserted. It supports multiple insertions when `obj` is a single
        scalar or a sequence with one element (similar to calling insert
        multiple times).
        Boolean indices are treated as a mask of elements to insert.
    values : array_like
        Values to insert into `arr`. If the type of `values` is different
        from that of `arr`, `values` is converted to the type of `arr`.
        `values` should be shaped so that ``arr[..., obj, ...] = values``
        is legal.
    axis : {None, int}, optional
        Axis along which to insert `values`. If `axis` is ``None`` then `arr`
        is flattened first.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        A copy of `arr` with `values` inserted. Note that :obj:`dpnp.insert`
        does not occur in-place: a new array is returned. If
        `axis` is ``None``, `out` is a flattened array.

    See Also
    --------
    :obj:`dpnp.append` : Append elements at the end of an array.
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.
    :obj:`dpnp.delete` : Delete elements from an array.

    Notes
    -----
    Note that for higher dimensional inserts ``obj=0`` behaves very different
    from ``obj=[0]`` just like ``arr[:, 0, :] = values`` is different from
    ``arr[:, [0], :] = values``.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 1], [2, 2], [3, 3]])
    >>> a
    array([[1, 1],
           [2, 2],
           [3, 3]])
    >>> np.insert(a, 1, 5)
    array([1, 5, 1, 2, 2, 3, 3])
    >>> np.insert(a, 1, 5, axis=1)
    array([[1, 5, 1],
           [2, 5, 2],
           [3, 5, 3]])

    Difference between sequence and scalars:

    >>> np.insert(a, [1], [[1],[2],[3]], axis=1)
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])
    >>> np.array_equal(np.insert(a, 1, [1, 2, 3], axis=1),
    ...                np.insert(a, [1], [[1],[2],[3]], axis=1))
    array(True)

    >>> b = a.flatten()
    >>> b
    array([1, 1, 2, 2, 3, 3])
    >>> np.insert(b, [2, 2], [5, 6])
    array([1, 1, 5, 6, 2, 2, 3, 3])

    >>> np.insert(b, slice(2, 4), [5, 6])
    array([1, 1, 5, 2, 6, 2, 3, 3])

    >>> np.insert(b, [2, 2], [7.13, False]) # dtype casting
    array([1, 1, 7, 0, 2, 2, 3, 3])

    >>> x = np.arange(8).reshape(2, 4)
    >>> idx = (1, 3)
    >>> np.insert(x, idx, 999, axis=1)
    array([[  0, 999,   1,   2, 999,   3],
           [  4, 999,   5,   6, 999,   7]])

    """

    dpnp.check_supported_arrays_type(arr)
    params = _calc_parameters(arr, axis, obj, values)

    if isinstance(obj, slice):
        # turn it into a range object
        indices = dpnp.arange(
            *obj.indices(params.n),
            dtype=dpnp.intp,
            sycl_queue=params.exec_q,
            usm_type=params.usm_type,
        )
    else:
        # need to copy obj, because indices will be changed in-place
        indices = dpnp.copy(
            obj, sycl_queue=params.exec_q, usm_type=params.usm_type
        )
        if indices.dtype == dpnp.bool:
            if indices.ndim != 1:
                raise ValueError(
                    "boolean array argument obj to insert "
                    "must be one dimensional"
                )
            indices = dpnp.flatnonzero(indices)
        elif indices.ndim > 1:
            raise ValueError(
                "index array argument `obj` to insert must be one-dimensional "
                "or scalar"
            )

    if indices.size == 1:
        return _insert_singleton_index(params, indices, values, obj)

    return _insert_array_indices(params, indices, values, obj)


def matrix_transpose(x, /):
    """
    Transposes a matrix (or a stack of matrices) `x`.

    For full documentation refer to :obj:`numpy.matrix_transpose`.

    Parameters
    ----------
    x : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Input array with ``x.ndim >= 2`` and whose two innermost
        dimensions form ``MxN`` matrices.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the transpose for each matrix and having shape
        (..., N, M).

    See Also
    --------
    :obj:`dpnp.transpose` : Returns an array with axes transposed.
    :obj:`dpnp.linalg.matrix_transpose` : Equivalent function.
    :obj:`dpnp.ndarray.mT` : Equivalent method.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.matrix_transpose(a)
    array([[1, 3],
           [2, 4]])

    >>> b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> np.matrix_transpose(b)
    array([[[1, 3],
            [2, 4]],
           [[5, 7],
            [6, 8]]])

    """

    usm_x = dpnp.get_usm_ndarray(x)
    if usm_x.ndim < 2:
        raise ValueError(
            "Input array must be at least 2-dimensional, "
            f"but it is {usm_x.ndim}"
        )

    usm_res = dpt.matrix_transpose(usm_x)
    return dpnp_array._create_from_usm_ndarray(usm_res)


def moveaxis(a, source, destination):
    """
    Move axes of an array to new positions. Other axes remain in their original
    order.

    For full documentation refer to :obj:`numpy.moveaxis`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        The array whose axes should be reordered.
    source : int or sequence of int
        Original positions of the axes to move. These must be unique.
    destination : int or sequence of int
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    out : dpnp.ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
    :obj:`dpnp.transpose` : Permute the dimensions of an array.
    :obj:`dpnp.swapaxes` : Interchange two axes of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.zeros((3, 4, 5))
    >>> np.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> np.moveaxis(x, -1, 0).shape
    (5, 3, 4)

    """

    usm_array = dpnp.get_usm_ndarray(a)
    return dpnp_array._create_from_usm_ndarray(
        dpt.moveaxis(usm_array, source, destination)
    )


def ndim(a):
    """
    Return the number of dimensions of array-like input.

    For full documentation refer to :obj:`numpy.ndim`.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`. Scalars are zero-dimensional.

    See Also
    --------
    :obj:`dpnp.ndarray.ndim` : Equivalent method for `dpnp.ndarray`
                        or `usm_ndarray` input.
    :obj:`dpnp.shape` : Return the shape of an array.
    :obj:`dpnp.ndarray.shape` : Return the shape of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = [[1, 2, 3], [4, 5, 6]]
    >>> np.ndim(a)
    2
    >>> a = np.asarray(a)
    >>> np.ndim(a)
    2
    >>> np.ndim(1)
    0

    """

    if dpnp.is_supported_array_type(a):
        return a.ndim
    return numpy.ndim(a)


def pad(array, pad_width, mode="constant", **kwargs):
    """
    Pad an array.

    For full documentation refer to :obj:`numpy.pad`.

    Parameters
    ----------
    array : {dpnp.ndarray, usm_ndarray}
        The array of rank ``N`` to pad.
    pad_width : {sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ``((before_1, after_1), ... (before_N, after_N))`` unique pad widths
        for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after pad for each axis.
        ``(pad,)`` or ``int`` is a shortcut for ``before = after = pad`` width
        for all axes.
    mode : {str, function}, optional
        One of the following string values or a user supplied function.

        "constant"
            Pads with a constant value.
        "edge"
            Pads with the edge values of array.
        "linear_ramp"
            Pads with the linear ramp between `end_value` and the
            array edge value.
        "maximum"
            Pads with the maximum value of all or part of the
            vector along each axis.
        "mean"
            Pads with the mean value of all or part of the
            vector along each axis.
        "median"
            Pads with the median value of all or part of the
            vector along each axis.
        "minimum"
            Pads with the minimum value of all or part of the
            vector along each axis.
        "reflect"
            Pads with the reflection of the vector mirrored on
            the first and last values of the vector along each
            axis.
        "symmetric"
            Pads with the reflection of the vector mirrored
            along the edge of the array.
        "wrap"
            Pads with the wrap of the vector along the axis.
            The first values are used to pad the end and the
            end values are used to pad the beginning.
        "empty"
            Pads with undefined values.
        <function>
            Padding function, see Notes.

        Default: ``"constant"``.
    stat_length : {None, int, sequence of ints}, optional
        Used in ``"maximum"``, ``"mean"``, ``"median"``, and ``"minimum"``.
        Number of values at edge of each axis used to calculate the statistic
        value. ``((before_1, after_1), ... (before_N, after_N))`` unique
        statistic lengths for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after statistic lengths for each axis.
        ``(stat_length,)`` or ``int`` is a shortcut for
        ``before = after = statistic`` length for all axes.

        Default: ``None``, to use the entire axis.
    constant_values : {sequence, scalar}, optional
        Used in ``"constant"``. The values to set the padded values for each
        axis.
        ``((before_1, after_1), ... (before_N, after_N))`` unique pad constants
        for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after constants for each axis.
        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default: ``0``.
    end_values : {sequence, scalar}, optional
        Used in ``"linear_ramp"``. The values used for the ending value of the
        linear_ramp and that will form the edge of the padded array.
        ``((before_1, after_1), ... (before_N, after_N))`` unique end values
        for each axis.
        ``(before, after)`` or ``((before, after),)`` yields same before
        and after end values for each axis.
        ``(constant,)`` or ``constant`` is a shortcut for
        ``before = after = constant`` for all axes.

        Default: ``0``.
    reflect_type : {"even", "odd"}, optional
        Used in ``"reflect"``, and ``"symmetric"``. The ``"even"`` style is the
        default with an unaltered reflection around the edge value. For
        the ``"odd"`` style, the extended part of the array is created by
        subtracting the reflected values from two times the edge value.

        Default: ``"even"``.

    Returns
    -------
    padded array : dpnp.ndarray
        Padded array of rank equal to `array` with shape increased
        according to `pad_width`.

    Notes
    -----
    For an array with rank greater than 1, some of the padding of later
    axes is calculated from padding of previous axes. This is easiest to
    think about with a rank 2 array where the corners of the padded array
    are calculated by using padded values from the first axis.

    The padding function, if used, should modify a rank 1 array in-place. It
    has the following signature::

        padding_func(vector, iaxis_pad_width, iaxis, kwargs)

    where

    vector : dpnp.ndarray
        A rank 1 array already padded with zeros. Padded values are
        vector[:iaxis_pad_width[0]] and vector[-iaxis_pad_width[1]:].
    iaxis_pad_width : tuple
        A 2-tuple of ints, iaxis_pad_width[0] represents the number of
        values padded at the beginning of vector where
        iaxis_pad_width[1] represents the number of values padded at
        the end of vector.
    iaxis : int
        The axis currently being calculated.
    kwargs : dict
        Any keyword arguments the function requires.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> np.pad(a, (2, 3), 'constant', constant_values=(4, 6))
    array([4, 4, 1, 2, 3, 4, 5, 6, 6, 6])

    >>> np.pad(a, (2, 3), 'edge')
    array([1, 1, 1, 2, 3, 4, 5, 5, 5, 5])

    >>> np.pad(a, (2, 3), 'linear_ramp', end_values=(5, -4))
    array([ 5,  3,  1,  2,  3,  4,  5,  2, -1, -4])

    >>> np.pad(a, (2,), 'maximum')
    array([5, 5, 1, 2, 3, 4, 5, 5, 5])

    >>> np.pad(a, (2,), 'mean')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> np.pad(a, (2,), 'median')
    array([3, 3, 1, 2, 3, 4, 5, 3, 3])

    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.pad(a, ((3, 2), (2, 3)), 'minimum')
    array([[1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1],
           [3, 3, 3, 4, 3, 3, 3],
           [1, 1, 1, 2, 1, 1, 1],
           [1, 1, 1, 2, 1, 1, 1]])

    >>> a = np.array([1, 2, 3, 4, 5])
    >>> np.pad(a, (2, 3), 'reflect')
    array([3, 2, 1, 2, 3, 4, 5, 4, 3, 2])

    >>> np.pad(a, (2, 3), 'reflect', reflect_type='odd')
    array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    >>> np.pad(a, (2, 3), 'symmetric')
    array([2, 1, 1, 2, 3, 4, 5, 5, 4, 3])

    >>> np.pad(a, (2, 3), 'symmetric', reflect_type='odd')
    array([0, 1, 1, 2, 3, 4, 5, 5, 6, 7])

    >>> np.pad(a, (2, 3), 'wrap')
    array([4, 5, 1, 2, 3, 4, 5, 1, 2, 3])

    >>> def pad_width(vector, pad_width, iaxis, kwargs):
    ...     pad_value = kwargs.get('padder', 10)
    ...     vector[:pad_width[0]] = pad_value
    ...     vector[-pad_width[1]:] = pad_value
    >>> a = np.arange(6)
    >>> a = a.reshape((2, 3))
    >>> np.pad(a, 2, pad_width)
    array([[10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10,  0,  1,  2, 10, 10],
           [10, 10,  3,  4,  5, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10]])
    >>> np.pad(a, 2, pad_width, padder=100)
    array([[100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100,   0,   1,   2, 100, 100],
           [100, 100,   3,   4,   5, 100, 100],
           [100, 100, 100, 100, 100, 100, 100],
           [100, 100, 100, 100, 100, 100, 100]])

    """

    dpnp.check_supported_arrays_type(array)
    return dpnp_pad(array, pad_width, mode=mode, **kwargs)


def ravel(a, order="C"):
    """
    Return a contiguous flattened array.

    For full documentation refer to :obj:`numpy.ravel`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array. The elements in `a` are read in the order specified by
        order, and packed as a 1-D array.
    order : {None, "C", "F", "A"}, optional
        The elements of `a` are read using this index order. ``"C"`` means to
        index the elements in row-major, C-style order, with the last axis
        index changing fastest, back to the first axis index changing slowest.
        ``"F"`` means to index the elements in column-major, Fortran-style
        order, with the first index changing fastest, and the last index
        changing slowest. Note that the "C" and "F" options take no account of
        the memory layout of the underlying array, and only refer to
        the order of axis indexing. "A" means to read the elements in
        Fortran-like index order if `a` is Fortran *contiguous* in
        memory, C-like order otherwise. ``order=None`` is an alias for
        ``order="C"``.

        Default: ``"C"``.

    Returns
    -------
    out : dpnp.ndarray
        A contiguous 1-D array of the same subtype as `a`, with shape (a.size,).

    Limitations
    -----------
    `order="K"` is not supported and the function raises `NotImplementedError`
    exception.

    See Also
    --------
    :obj:`dpnp.ndarray.flat` : 1-D iterator over an array.
    :obj:`dpnp.ndarray.flatten` : 1-D array copy of the elements of an array
                    in row-major order.
    :obj:`dpnp.ndarray.reshape` : Change the shape of an array without
                    changing its data.
    :obj:`dpnp.reshape` : The same as :obj:`dpnp.ndarray.reshape`.

    Notes
    -----
    In row-major, C-style order, in two dimensions, the row index
    varies the slowest, and the column index the quickest. This can
    be generalized to multiple dimensions, where row-major order
    implies that the index along the first axis varies slowest, and
    the index along the last quickest. The opposite holds for
    column-major, Fortran-style index ordering.

    When a view is desired in as many cases as possible, ``arr.reshape(-1)``
    may be preferable.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.ravel(x)
    array([1, 2, 3, 4, 5, 6])

    >>> x.reshape(-1)
    array([1, 2, 3, 4, 5, 6])

    >>> np.ravel(x, order='F')
    array([1, 4, 2, 5, 3, 6])

    When `order` is ``"A"``, it will preserve the array's
    ``"C"`` or ``"F"`` ordering:

    >>> np.ravel(x.T)
    array([1, 4, 2, 5, 3, 6])
    >>> np.ravel(x.T, order='A')
    array([1, 2, 3, 4, 5, 6])

    """

    if order in "kK":
        raise NotImplementedError(
            "Keyword argument `order` is supported only with "
            f"values None, 'C', 'F', and 'A', but got '{order}'"
        )

    result = dpnp.reshape(a, -1, order=order)
    if result.flags.c_contiguous:
        return result

    return dpnp.ascontiguousarray(result)


def repeat(a, repeats, axis=None):
    """
    Repeat elements of an array.

    For full documentation refer to :obj:`numpy.repeat`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.
    repeats : {int, tuple, list, range, dpnp.ndarray, usm_ndarray}
        The number of repetitions for each element. `repeats` is broadcasted to
        fit the shape of the given axis.
        If `repeats` is an array, it must have an integer data type.
        Otherwise, `repeats` must be a Python integer or sequence of Python
        integers (i.e., a tuple, list, or range).
    axis : {None, int}, optional
        The axis along which to repeat values. By default, use the flattened
        input array, and return a flat output array.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Output array which has the same shape as `a`, except along the given
        axis.

    See Also
    --------
    :obj:`dpnp.tile` : Tile an array.
    :obj:`dpnp.unique` : Find the unique elements of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([3])
    >>> np.repeat(x, 4)
    array([3, 3, 3, 3])

    >>> x = np.array([[1, 2], [3, 4]])
    >>> np.repeat(x, 2)
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> np.repeat(x, 3, axis=1)
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> np.repeat(x, [1, 2], axis=0)
    array([[1, 2],
           [3, 4],
           [3, 4]])

    """

    dpnp.check_supported_arrays_type(a)
    if not isinstance(repeats, (int, tuple, list, range)):
        repeats = dpnp.get_usm_ndarray(repeats)

    if axis is None and a.ndim > 1:
        a = dpnp.ravel(a)

    usm_arr = dpnp.get_usm_ndarray(a)
    usm_res = dpt.repeat(usm_arr, repeats, axis=axis)
    return dpnp_array._create_from_usm_ndarray(usm_res)


def require(a, dtype=None, requirements=None, *, like=None):
    """
    Return a :class:`dpnp.ndarray` of the provided type that satisfies
    requirements.

    This function is useful to be sure that an array with the correct flags
    is returned for passing to compiled code (perhaps through ctypes).

    For full documentation refer to :obj:`numpy.require`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
       The input array to be converted to a type-and-requirement-satisfying
       array.
    dtype : {None, str, dtype object}, optional
       The required data-type. If ``None`` preserve the current dtype.

       Default: ``None``.
    requirements : {None, str, sequence of str}, optional
       The requirements list can be any of the following:

       * 'F_CONTIGUOUS' ('F') - ensure a Fortran-contiguous array
       * 'C_CONTIGUOUS' ('C') - ensure a C-contiguous array
       * 'WRITABLE' ('W') - ensure a writable array

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Array with specified requirements and type if given.

    Limitations
    -----------
    Parameter `like` is supported only with default value ``None``.
    Otherwise, the function raises `NotImplementedError` exception.

    See Also
    --------
    :obj:`dpnp.asarray` : Convert input to an ndarray.
    :obj:`dpnp.asanyarray` : Convert to an ndarray, but pass through
                        ndarray subclasses.
    :obj:`dpnp.ascontiguousarray` : Convert input to a contiguous array.
    :obj:`dpnp.asfortranarray` : Convert input to an ndarray with
                        column-major memory order.
    :obj:`dpnp.ndarray.flags` : Information about the memory layout
                        of the array.

    Notes
    -----
    The returned array will be guaranteed to have the listed requirements
    by making a copy if needed.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(6).reshape(2, 3)
    >>> x.flags
      C_CONTIGUOUS : True
      F_CONTIGUOUS : False
      WRITEABLE : True

    >>> y = np.require(x, dtype=np.float32, requirements=['W', 'F'])
    >>> y.flags
      C_CONTIGUOUS : False
      F_CONTIGUOUS : True
      WRITEABLE : True

    """

    dpnp.check_limitations(like=like)
    dpnp.check_supported_arrays_type(a)

    possible_flags = {
        "C": "C",
        "C_CONTIGUOUS": "C",
        "F": "F",
        "F_CONTIGUOUS": "F",
        "W": "W",
        "WRITEABLE": "W",
    }

    if not requirements:
        return dpnp.asanyarray(a, dtype=dtype)

    try:
        requirements = {possible_flags[x.upper()] for x in requirements}
    except KeyError as exc:
        incorrect_flag = (set(requirements) - set(possible_flags.keys())).pop()
        raise ValueError(
            f"Incorrect flag {incorrect_flag} in requirements"
        ) from exc

    order = "A"
    if requirements.issuperset({"C", "F"}):
        raise ValueError("Cannot specify both 'C' and 'F' order")
    if "F" in requirements:
        order = "F"
        requirements.remove("F")
    elif "C" in requirements:
        order = "C"
        requirements.remove("C")

    arr = dpnp.array(a, dtype=dtype, order=order, copy=None)
    if not arr.flags["W"]:
        return arr.copy(order)

    return arr


def reshape(a, /, shape=None, order="C", *, newshape=None, copy=None):
    """
    Gives a new shape to an array without changing its data.

    For full documentation refer to :obj:`numpy.reshape`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to be reshaped.
    shape : {int, tuple of ints}, optional
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.

        Default: ``None``.
    order : {None, "C", "F", "A"}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order. ``"C"``
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. ``"F"`` means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the ``"C"`` and ``"F"`` options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        ``order=None`` is an alias for ``order="C"``. ``"A"`` means to
        read / write the elements in Fortran-like index order if ``a`` is
        Fortran *contiguous* in memory, C-like order otherwise.

        Default: ``"C"``.
    newshape : int or tuple of ints
        Replaced by `shape` argument. Retained for backward compatibility.

        Default: ``None``.
    copy : {None, bool}, optional
        If ``True``, then the array data is copied. If ``None``, a copy will
        only be made if it's required by ``order``. For ``False`` it raises
        a ``ValueError`` if a copy cannot be avoided.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        This will be a new view object if possible; otherwise, it will
        be a copy. Note there is no guarantee of the *memory layout* (C- or
        Fortran- contiguous) of the returned array.

    See Also
    --------
    :obj:`dpnp.ndarray.reshape` : Equivalent method.

    Notes
    -----
    It is not always possible to change the shape of an array without copying
    the data.

    The `order` keyword gives the index ordering both for *fetching*
    the values from ``a``, and then *placing* the values into the output
    array. For example, let's say you have an array:

    >>> import dpnp as np
    >>> a = np.arange(6).reshape((3, 2))
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5]])

    You can think of reshaping as first raveling the array (using the given
    index order), then inserting the elements from the raveled array into the
    new array using the same kind of index ordering as was used for the
    raveling.

    >>> np.reshape(a, (2, 3)) # C-like index ordering
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
    array([[0, 4, 3],
           [2, 1, 5]])
    >>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
    array([[0, 4, 3],
           [2, 1, 5]])

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.reshape(a, 6)
    array([1, 2, 3, 4, 5, 6])
    >>> np.reshape(a, 6, order='F')
    array([1, 4, 2, 5, 3, 6])

    >>> np.reshape(a, (3, -1))       # the unspecified value is inferred to be 2
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """

    if newshape is None and shape is None:
        raise TypeError(
            "reshape() missing 1 required positional argument: 'shape'"
        )

    if newshape is not None:
        if shape is not None:
            raise TypeError(
                "You cannot specify 'newshape' and 'shape' arguments "
                "at the same time."
            )
        # Deprecated in dpnp 0.17.0
        warnings.warn(
            "`newshape` keyword argument is deprecated, "
            "use `shape=...` or pass shape positionally instead. "
            "(deprecated in dpnp 0.17.0)",
            DeprecationWarning,
            stacklevel=2,
        )
        shape = newshape

    if order is None:
        order = "C"
    elif order in "aA":
        order = "F" if a.flags.fnc else "C"
    elif order not in "cfCF":
        raise ValueError(
            f"order must be None, 'C', 'F', or 'A' (got '{order}')"
        )

    usm_a = dpnp.get_usm_ndarray(a)
    usm_res = dpt.reshape(usm_a, shape=shape, order=order, copy=copy)
    return dpnp_array._create_from_usm_ndarray(usm_res)


def resize(a, new_shape):
    """
    Return a new array with the specified shape.

    If the new array is larger than the original array, then the new array is
    filled with repeated copies of `a`. Note that this behavior is different
    from ``a.resize(new_shape)`` which fills with zeros instead of repeated
    copies of `a`.

    For full documentation refer to :obj:`numpy.resize`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to be resized.
    new_shape : {int, tuple or list of ints}
        Shape of resized array.

    Returns
    -------
    out : dpnp.ndarray
        The new array is formed from the data in the old array, repeated
        if necessary to fill out the required number of elements. The
        data are repeated iterating over the array in C-order.

    See Also
    --------
    :obj:`dpnp.ndarray.resize` : Resize an array in-place.
    :obj:`dpnp.reshape` : Reshape an array without changing the total size.
    :obj:`dpnp.pad` : Enlarge and pad an array.
    :obj:`dpnp.repeat` : Repeat elements of an array.

    Notes
    -----
    When the total size of the array does not change :obj:`dpnp.reshape` should
    be used. In most other cases either indexing (to reduce the size) or
    padding (to increase the size) may be a more appropriate solution.

    Warning: This functionality does **not** consider axes separately,
    i.e. it does not apply interpolation/extrapolation.
    It fills the return array with the required number of elements, iterating
    over `a` in C-order, disregarding axes (and cycling back from the start if
    the new shape is larger). This functionality is therefore not suitable to
    resize images, or data where each axis represents a separate and distinct
    entity.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[0, 1], [2, 3]])
    >>> np.resize(a, (2, 3))
    array([[0, 1, 2],
           [3, 0, 1]])
    >>> np.resize(a, (1, 4))
    array([[0, 1, 2, 3]])
    >>> np.resize(a, (2, 4))
    array([[0, 1, 2, 3],
           [0, 1, 2, 3]])

    """

    dpnp.check_supported_arrays_type(a)
    if a.ndim == 0:
        return dpnp.full_like(a, a, shape=new_shape)

    if isinstance(new_shape, (int, numpy.integer)):
        new_shape = (new_shape,)

    new_size = 1
    for dim_length in new_shape:
        if dim_length < 0:
            raise ValueError("all elements of `new_shape` must be non-negative")
        new_size *= dim_length

    a_size = a.size
    if a_size == 0 or new_size == 0:
        # First case must zero fill. The second would have repeats == 0.
        return dpnp.zeros_like(a, shape=new_shape)

    repeats = -(-new_size // a_size)  # ceil division
    a = dpnp.concatenate((dpnp.ravel(a),) * repeats)[:new_size]

    return a.reshape(new_shape)


def result_type(*arrays_and_dtypes):
    """
    result_type(*arrays_and_dtypes)

    Returns the type that results from applying the NumPy
    type promotion rules to the arguments.

    For full documentation refer to :obj:`numpy.result_type`.

    Parameters
    ----------
    arrays_and_dtypes : list of {dpnp.ndarray, usm_ndarray, dtype}
        An arbitrary length sequence of arrays or dtypes.

    Returns
    -------
    out : dtype
        The result type.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(3, dtype=np.int64)
    >>> b = np.arange(7, dtype=np.int32)
    >>> np.result_type(a, b)
    dtype('int64')

    >>> np.result_type(np.int64, np.complex128)
    dtype('complex128')

    >>> np.result_type(np.ones(10, dtype=np.float32), np.float64)
    dtype('float64')

    """

    usm_arrays_and_dtypes = [
        (
            dpnp.get_usm_ndarray(X)
            if isinstance(X, (dpnp_array, dpt.usm_ndarray))
            else X
        )
        for X in arrays_and_dtypes
    ]
    return dpt.result_type(*usm_arrays_and_dtypes)


def roll(x, shift, axis=None):
    """
    Roll the elements of an array by a number of positions along a given axis.

    Array elements that roll beyond the last position are re-introduced
    at the first position. Array elements that roll beyond the first position
    are re-introduced at the last position.

    For full documentation refer to :obj:`numpy.roll`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    shift : {int, tuple of ints}
        The number of places by which elements are shifted. If a tuple, then
        `axis` must be a tuple of the same size, and each of the given axes
        is shifted by the corresponding number. If an integer while `axis` is
        a tuple of integers, then the same value is used for all given axes.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which elements are shifted. By default, the
        array is flattened before shifting, after which the original
        shape is restored.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        An array with the same data type as `x`
        and whose elements, relative to `x`, are shifted.

    See Also
    --------
    :obj:`dpnp.moveaxis` : Move array axes to new positions.
    :obj:`dpnp.rollaxis` : Roll the specified axis backwards
                       until it lies in a given position.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.arange(10)
    >>> np.roll(x1, 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    >>> np.roll(x1, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x2 = np.reshape(x1, (2, 5))
    >>> np.roll(x2, 1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 1, 2, 3, 4]])

    >>> np.roll(x2, (2, 1), axis=(1, 0))
    array([[8, 9, 5, 6, 7],
           [3, 4, 0, 1, 2]])

    """

    usm_x = dpnp.get_usm_ndarray(x)
    if dpnp.is_supported_array_type(shift):
        shift = dpnp.asnumpy(shift)

    if axis is None:
        return roll(dpt.reshape(usm_x, -1), shift, 0).reshape(x.shape)

    usm_res = dpt.roll(usm_x, shift=shift, axis=axis)
    return dpnp_array._create_from_usm_ndarray(usm_res)


def rollaxis(x, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    For full documentation refer to :obj:`numpy.rollaxis`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : int
        The axis to be rolled. The positions of the other axes do not
        change relative to one another.
    start : int, optional
        When ``start <= axis``, the axis is rolled back until it lies in
        this position. When ``start > axis``, the axis is rolled until it
        lies before this position. The default, ``0``, results in a "complete"
        roll.

        Default: ``0``.

    Returns
    -------
    out : dpnp.ndarray
        An array with the same data type as `x` where the specified axis
        has been moved to the requested position.

    See Also
    --------
    :obj:`dpnp.moveaxis` : Move array axes to new positions.
    :obj:`dpnp.roll` : Roll the elements of an array
                       by a number of positions along a given axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.ones((3, 4, 5, 6))
    >>> np.rollaxis(a, 3, 1).shape
    (3, 6, 4, 5)
    >>> np.rollaxis(a, 2).shape
    (5, 3, 4, 6)
    >>> np.rollaxis(a, 1, 4).shape
    (3, 5, 6, 4)

    """

    n = x.ndim
    axis = normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not 0 <= start < n + 1:
        raise ValueError(msg % ("start", -n, "start", n + 1, start))
    if axis < start:
        start -= 1
    if axis == start:
        return x
    usm_array = dpnp.get_usm_ndarray(x)
    return dpnp.moveaxis(usm_array, source=axis, destination=start)


def rot90(m, k=1, axes=(0, 1)):
    """
    Rotate an array by 90 degrees in the plane specified by axes.

    Rotation direction is from the first towards the second axis.
    This means for a 2D array with the default `k` and `axes`, the
    rotation will be counterclockwise.

    For full documentation refer to :obj:`numpy.rot90`.

    Parameters
    ----------
    m : {dpnp.ndarray, usm_ndarray}
        Array of two or more dimensions.
    k : integer, optional
        Number of times the array is rotated by 90 degrees.

        Default: ``1``.
    axes : (2,) array_like of ints, optional
        The array is rotated in the plane defined by the axes.
        Axes must be different.

        Default: ``(0, 1)``.

    Returns
    -------
    out : dpnp.ndarray
        A rotated view of `m`.

    See Also
    --------
    :obj:`dpnp.flip` : Reverse the order of elements in an array along
                    the given axis.
    :obj:`dpnp.fliplr` : Flip an array horizontally.
    :obj:`dpnp.flipud` : Flip an array vertically.

    Notes
    -----
    ``rot90(m, k=1, axes=(1, 0))`` is the reverse of
    ``rot90(m, k=1, axes=(0, 1))``.

    ``rot90(m, k=1, axes=(1, 0))`` is equivalent to
    ``rot90(m, k=-1, axes=(0, 1))``.

    Examples
    --------
    >>> import dpnp as np
    >>> m = np.array([[1, 2], [3, 4]])
    >>> m
    array([[1, 2],
           [3, 4]])
    >>> np.rot90(m)
    array([[2, 4],
           [1, 3]])
    >>> np.rot90(m, 2)
    array([[4, 3],
           [2, 1]])
    >>> m = np.arange(8).reshape((2, 2, 2))
    >>> np.rot90(m, 1, (1, 2))
    array([[[1, 3],
            [0, 2]],
           [[5, 7],
            [4, 6]]])

    """

    dpnp.check_supported_arrays_type(m)
    k = operator.index(k)

    m_ndim = m.ndim
    if m_ndim < 2:
        raise ValueError("Input must be at least 2-d.")

    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")

    if axes[0] == axes[1] or abs(axes[0] - axes[1]) == m_ndim:
        raise ValueError("Axes must be different.")

    if not (-m_ndim <= axes[0] < m_ndim and -m_ndim <= axes[1] < m_ndim):
        raise ValueError(
            f"Axes={axes} out of range for array of ndim={m_ndim}."
        )

    k %= 4
    if k == 0:
        return m[:]
    if k == 2:
        return dpnp.flip(dpnp.flip(m, axes[0]), axes[1])

    axes_list = list(range(0, m_ndim))
    (axes_list[axes[0]], axes_list[axes[1]]) = (
        axes_list[axes[1]],
        axes_list[axes[0]],
    )

    if k == 1:
        return dpnp.transpose(dpnp.flip(m, axes[1]), axes_list)

    # k == 3
    return dpnp.flip(dpnp.transpose(m, axes_list), axes[1])


def shape(a):
    """
    Return the shape of an array.

    For full documentation refer to :obj:`numpy.shape`.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : {int, tuple of ints}
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    See Also
    --------
    len : ``len(a)`` is equivalent to ``np.shape(a)[0]`` for N-D arrays with
          ``N>=1``.
    :obj:`dpnp.ndarray.shape` : Equivalent array method.

    Examples
    --------
    >>> import dpnp as np
    >>> np.shape(np.eye(3))
    (3, 3)
    >>> np.shape([[1, 3]])
    (1, 2)
    >>> np.shape([0])
    (1,)
    >>> np.shape(0)
    ()

    """

    if dpnp.is_supported_array_type(a):
        return a.shape
    return numpy.shape(a)


def size(a, axis=None):
    """
    Return the number of elements along a given axis.

    For full documentation refer to :obj:`numpy.size`.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : {None, int}, optional
        Axis along which the elements are counted.
        By default, give the total number of elements.

        Default: ``None``.

    Returns
    -------
    element_count : int
        Number of elements along the specified axis.

    See Also
    --------
    :obj:`dpnp.ndarray.size` : number of elements in array.
    :obj:`dpnp.shape` : Return the shape of an array.
    :obj:`dpnp.ndarray.shape` : Return the shape of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = [[1, 2, 3], [4, 5, 6]]
    >>> np.size(a)
    6
    >>> np.size(a, 1)
    3
    >>> np.size(a, 0)
    2

    >>> a = np.asarray(a)
    >>> np.size(a)
    6
    >>> np.size(a, 1)
    3

    """

    if dpnp.is_supported_array_type(a):
        if axis is None:
            return a.size
        return a.shape[axis]

    return numpy.size(a, axis)


def split(ary, indices_or_sections, axis=0):
    """
    Split an array into multiple sub-arrays as views into `ary`.

    For full documentation refer to :obj:`numpy.split`.

    Parameters
    ----------
    ary : {dpnp.ndarray, usm_ndarray}
        Array to be divided into sub-arrays.
    indices_or_sections : {int, sequence of ints}
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`. If such a split is not possible,
        an error is raised.

        If `indices_or_sections` is a sequence of sorted integers, the entries
        indicate where along `axis` the array is split. For example,
        ``[2, 3]`` would, for ``axis=0``, result in

        - ary[:2]
        - ary[2:3]
        - ary[3:]

        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split.

        Default: ``0``.

    Returns
    -------
    sub-arrays : list of dpnp.ndarray
        A list of sub arrays. Each array is a view of the corresponding input
        array.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division.

    See Also
    --------
    :obj:`dpnp.array_split` : Split an array into multiple sub-arrays of equal
                        or near-equal size. Does not raise an exception if an
                        equal division cannot be made.
    :obj:`dpnp.hsplit` : Split array into multiple sub-arrays horizontally
                    (column-wise).
    :obj:`dpnp.vsplit` : Split array into multiple sub-arrays vertically
                    (row wise).
    :obj:`dpnp.dsplit` : Split array into multiple sub-arrays along the 3rd
                    axis (depth).
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.
    :obj:`dpnp.stack` : Join a sequence of arrays along a new axis.
    :obj:`dpnp.hstack` : Stack arrays in sequence horizontally (column wise).
    :obj:`dpnp.vstack` : Stack arrays in sequence vertically (row wise).
    :obj:`dpnp.dstack` : Stack arrays in sequence depth wise
                    (along third dimension).

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(9.0)
    >>> np.split(x, 3)
    [array([0., 1., 2.]), array([3., 4., 5.]), array([6., 7., 8.])]

    >>> x = np.arange(8.0)
    >>> np.split(x, [3, 5, 6, 10])
    [array([0., 1., 2.]), array([3., 4.]), array([5.]), array([6., 7.]), \
    array([])]

    """

    dpnp.check_supported_arrays_type(ary)
    if ary.ndim <= axis:
        raise IndexError("Axis exceeds ndim")

    try:
        len(indices_or_sections)
    except TypeError:
        if ary.shape[axis] % indices_or_sections != 0:
            raise ValueError(
                "indices_or_sections must divide the size along the axes.\n"
                "If you want to split the array into non-equally-sized "
                "arrays, use array_split instead."
            ) from None

    return array_split(ary, indices_or_sections, axis)


def squeeze(a, /, axis=None):
    """
    Removes singleton dimensions (axes) from array `a`.

    For full documentation refer to :obj:`numpy.squeeze`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input data.
    axis : {None, int, tuple of ints}, optional
        Selects a subset of the entries of length one in the shape.
        If an axis is selected with shape entry greater than one,
        an error is raised.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Output array is a view, if possible, and a copy otherwise, but with all
        or a subset of the dimensions of length 1 removed. Output has the same
        data type as the input, is allocated on the same device as the input
        and has the same USM allocation type as the input array `a`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> np.squeeze(x).shape
    (3,)
    >>> np.squeeze(x, axis=0).shape
    (3, 1)
    >>> np.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: Cannot select an axis to squeeze out which has size not equal
    to one.
    >>> np.squeeze(x, axis=2).shape
    (1, 3)

    """

    usm_a = dpnp.get_usm_ndarray(a)
    usm_res = dpt.squeeze(usm_a, axis=axis)
    return dpnp_array._create_from_usm_ndarray(usm_res)


def stack(arrays, /, *, axis=0, out=None, dtype=None, casting="same_kind"):
    """
    Join a sequence of arrays along a new axis.

    For full documentation refer to :obj:`numpy.stack`.

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
        Each array must have the same shape.
    axis : int, optional
        The axis in the result array along which the input arrays are stacked.

        Default: ``0``.
    out : dpnp.ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what stack would have returned if no out
        argument were specified.

        Default: ``None``.
    dtype : {None, str, dtype object}, optional
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        Default: ``None``.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        Default: ``"same_kind"``.

    Returns
    -------
    out : dpnp.ndarray
        The stacked array which has one more dimension than the input arrays.

    See Also
    --------
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.
    :obj:`dpnp.hstack` : Stack arrays in sequence horizontally (column wise).
    :obj:`dpnp.vstack` : Stack arrays in sequence vertically (row wise).
    :obj:`dpnp.dstack` : Stack arrays in sequence depth wise
                         (along third dimension).
    :obj:`dpnp.column_stack` : Stack 1-D arrays as columns into a 2-D array.
    :obj:`dpnp.block` : Assemble an ndarray from nested lists of blocks.
    :obj:`dpnp.split` : Split array into a list of multiple sub-arrays of equal
                        size.
    :obj:`dpnp.unstack` : Split an array into a tuple of sub-arrays along
                          an axis.

    Examples
    --------
    >>> import dpnp as np
    >>> arrays = [np.random.randn(3, 4) for _ in range(10)]
    >>> np.stack(arrays, axis=0).shape
    (10, 3, 4)

    >>> np.stack(arrays, axis=1).shape
    (3, 10, 4)

    >>> np.stack(arrays, axis=2).shape
    (3, 4, 10)

    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.stack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> np.stack((a, b), axis=-1)
    array([[1, 4],
           [2, 5],
           [3, 6]])

    """

    _check_stack_arrays(arrays)

    if dtype is not None and out is not None:
        raise TypeError(
            "stack() only takes `out` or `dtype` as an argument, "
            "but both were provided."
        )

    usm_arrays = [dpnp.get_usm_ndarray(x) for x in arrays]
    usm_res = dpt.stack(usm_arrays, axis=axis)

    res = dpnp_array._create_from_usm_ndarray(usm_res)
    if dtype is not None:
        res = res.astype(dtype, casting=casting, copy=False)
    elif out is not None:
        dpnp.copyto(out, res, casting=casting)
        return out
    return res


def swapaxes(a, axis1, axis2):
    """
    Interchange two axes of an array.

    For full documentation refer to :obj:`numpy.swapaxes`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    out : dpnp.ndarray
        An array with with swapped axes.
        A view is returned whenever possible.

    Notes
    -----
    If `a` has rank (i.e., number of dimensions) `N`,
    a valid `axis` must be in the half-open interval `[-N, N)`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[1, 2, 3]])
    >>> np.swapaxes(x, 0, 1)
    array([[1],
           [2],
           [3]])

    >>> x = np.array([[[0, 1],[2, 3]],[[4, 5],[6, 7]]])
    >>> x
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.swapaxes(x, 0, 2)
    array([[[0, 4],
            [2, 6]],
           [[1, 5],
            [3, 7]]])

    """

    usm_a = dpnp.get_usm_ndarray(a)
    usm_res = dpt.swapaxes(usm_a, axis1=axis1, axis2=axis2)
    return dpnp_array._create_from_usm_ndarray(usm_res)


# pylint: disable=invalid-name
def tile(A, reps):
    """
    Construct an array by repeating `A` the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of
    ``max(d, A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by prepending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Note : Although tile may be used for broadcasting, it is strongly
    recommended to use dpnp's broadcasting operations and functions.

    For full documentation refer to :obj:`numpy.tile`.

    Parameters
    ----------
    A : {dpnp.ndarray, usm_ndarray}
        The input array.
    reps : int or tuple of ints
        The number of repetitions of `A` along each axis.

    Returns
    -------
    out : dpnp.ndarray
        The tiled output array.

    See Also
    --------
    :obj:`dpnp.repeat` : Repeat elements of an array.
    :obj:`dpnp.broadcast_to` : Broadcast an array to a new shape

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([0, 1, 2])
    >>> np.tile(a, 2)
    array([0, 1, 2, 0, 1, 2])

    >>> np.tile(a, (2, 2))
    array([[0, 1, 2, 0, 1, 2],
           [0, 1, 2, 0, 1, 2]])

    >>> np.tile(a, (2, 1, 2))
    array([[[0, 1, 2, 0, 1, 2]],
           [[0, 1, 2, 0, 1, 2]]])

    >>> b = np.array([[1, 2], [3, 4]])
    >>> np.tile(b, 2)
    array([[1, 2, 1, 2],
           [3, 4, 3, 4]])

    >>> np.tile(b, (2, 1))
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> c = np.array([1, 2, 3, 4])
    >>> np.tile(c, (4, 1))
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])

    """

    usm_a = dpnp.get_usm_ndarray(A)
    usm_res = dpt.tile(usm_a, reps)
    return dpnp_array._create_from_usm_ndarray(usm_res)


def transpose(a, axes=None):
    """
    Returns an array with axes transposed.

    Note that :obj:`dpnp.permute_dims` is an alias of :obj:`dpnp.transpose`.

    For full documentation refer to :obj:`numpy.transpose`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axes : {None, tuple or list of ints}, optional
        If specified, it must be a tuple or list which contains a permutation
        of [0, 1, ..., N-1] where N is the number of axes of `a`.
        The `i`'th axis of the returned array will correspond to the axis
        numbered ``axes[i]`` of the input. If not specified or ``None``,
        defaults to ``range(a.ndim)[::-1]``, which reverses the order of
        the axes.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        `a` with its axes permuted. A view is returned whenever possible.

    See Also
    --------
    :obj:`dpnp.ndarray.transpose` : Equivalent method.
    :obj:`dpnp.moveaxis` : Move array axes to new positions.
    :obj:`dpnp.argsort` : Returns the indices that would sort an array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> np.transpose(a)
    array([[1, 3],
           [2, 4]])

    >>> a = np.array([1, 2, 3, 4])
    >>> a
    array([1, 2, 3, 4])
    >>> np.transpose(a)
    array([1, 2, 3, 4])

    >>> a = np.ones((1, 2, 3))
    >>> np.transpose(a, (1, 0, 2)).shape
    (2, 1, 3)

    >>> a = np.ones((2, 3, 4, 5))
    >>> np.transpose(a).shape
    (5, 4, 3, 2)

    """

    dpnp.check_supported_arrays_type(a)
    if isinstance(a, dpt.usm_ndarray):
        a = dpnp_array._create_from_usm_ndarray(a)

    if axes is None:
        return a.transpose()
    return a.transpose(*axes)


permute_dims = transpose  # permute_dims is an alias for transpose


def trim_zeros(filt, trim="fb", axis=None):
    """
    Remove values along a dimension which are zero along all other.

    For full documentation refer to :obj:`numpy.trim_zeros`.

    Parameters
    ----------
    filt : {dpnp.ndarray, usm_ndarray}
        Input array.
    trim : {"fb", "f", "b"}, optional
        A string with `"f"` representing trim from front and `"b"` to trim from
        back. By default, zeros are trimmed on both sides. Front and back refer
        to the edges of a dimension, with "front" referring to the side with
        the lowest index 0, and "back" referring to the highest index
        (or index -1).

        Default: ``"fb"``.
    axis : {None, int}, optional
        If ``None``, `filt` is cropped such that the smallest bounding box is
        returned that still contains all values which are not zero.
        If an `axis` is specified, `filt` will be sliced in that dimension only
        on the sides specified by `trim`. The remaining area will be the
        smallest that still contains all values which are not zero.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The result of trimming the input. The number of dimensions and the
        input data type are preserved.

    Notes
    -----
    For all-zero arrays, the first axis is trimmed first.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
    >>> np.trim_zeros(a)
    array([1, 2, 3, 0, 2, 1])

    >>> np.trim_zeros(a, trim='b')
    array([0, 0, 0, 1, 2, 3, 0, 2, 1])

    Multiple dimensions are supported:

    >>> b = np.array([[0, 0, 2, 3, 0, 0],
    ...               [0, 1, 0, 3, 0, 0],
    ...               [0, 0, 0, 0, 0, 0]])
    >>> np.trim_zeros(b)
    array([[0, 2, 3],
           [1, 0, 3]])

    >>> np.trim_zeros(b, axis=-1)
    array([[0, 2, 3],
           [1, 0, 3],
           [0, 0, 0]])

    """

    dpnp.check_supported_arrays_type(filt)

    if not isinstance(trim, str):
        raise TypeError("only string trim is supported")

    trim = trim.lower()
    if trim not in ["fb", "bf", "f", "b"]:
        raise ValueError(f"unexpected character(s) in `trim`: {trim!r}")

    nd = filt.ndim
    if axis is not None:
        axis = normalize_axis_index(axis, nd)

    if filt.size == 0:
        return filt  # no trailing zeros in empty array

    non_zero = dpnp.argwhere(filt)
    if non_zero.size == 0:
        # `filt` has all zeros, so assign `start` and `stop` to the same value,
        # then the resulting slice will be empty
        start = stop = dpnp.zeros_like(filt, shape=nd, dtype=dpnp.intp)
    else:
        if "f" in trim:
            start = non_zero.min(axis=0)
        else:
            start = (None,) * nd

        if "b" in trim:
            stop = non_zero.max(axis=0)
            stop += 1  # Adjust for slicing
        else:
            stop = (None,) * nd

    if axis is None:
        # trim all axes
        sl = tuple(slice(*x) for x in zip(start, stop))
    else:
        # only trim single axis
        sl = (slice(None),) * axis + (slice(start[axis], stop[axis]),) + (...,)

    return filt[sl]


def unique(
    ar,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    *,
    equal_nan=True,
):
    """
    Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    For full documentation refer to :obj:`numpy.unique`.

    Parameters
    ----------
    ar : {dpnp.ndarray, usm_ndarray}
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If ``True``, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.

        Default: ``False``.
    return_inverse : bool, optional
        If ``True``, also return the indices of the unique array (for the
        specified axis, if provided) that can be used to reconstruct `ar`.

        Default: ``False``.
    return_counts : bool, optional
        If ``True``, also return the number of times each unique item appears
        in `ar`.

        Default: ``False``.
    axis : {int, None}, optional
        The axis to operate on. If ``None``, `ar` will be flattened. If an
        integer, the subarrays indexed by the given axis will be flattened and
        treated as the elements of a 1-D array with the dimension of the given
        axis, see the notes for more details.

        Default: ``None``.
    equal_nan : bool, optional
        If ``True``, collapses multiple NaN values in the return array into one.

        Default: ``True``.

    Returns
    -------
    unique : dpnp.ndarray
        The sorted unique values.
    unique_indices : dpnp.ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is ``True``.
    unique_inverse : dpnp.ndarray, optional
        The indices to reconstruct the original array from the unique array.
        Only provided if `return_inverse` is ``True``.
    unique_counts : dpnp.ndarray, optional
        The number of times each of the unique values comes up in the original
        array. Only provided if `return_counts` is ``True``.

    See Also
    --------
    :obj:`dpnp.repeat` : Repeat elements of an array.

    Notes
    -----
    When an axis is specified the subarrays indexed by the axis are sorted.
    This is done by making the specified axis the first dimension of the array
    (move the axis to the first dimension to keep the order of the other axes)
    and then flattening the subarrays in C order.
    For complex arrays all NaN values are considered equivalent (no matter
    whether the NaN is in the real or imaginary part). As the representative for
    the returned array the smallest one in the lexicographical order is chosen.
    For multi-dimensional inputs, `unique_inverse` is reshaped such that the
    input can be reconstructed using
    ``dpnp.take(unique, unique_inverse, axis=axis)``.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 1, 2, 2, 3, 3])
    >>> np.unique(a)
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])

    Return the unique rows of a 2D array

    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0],
           [2, 3, 4]])

    Reconstruct the input array from the unique values and inverse:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])

    Reconstruct the input values from the unique values and counts:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> values, counts = np.unique(a, return_counts=True)
    >>> values
    array([1, 2, 3, 4, 6])
    >>> counts
    array([1, 3, 1, 1, 1])
    >>> np.repeat(values, counts)
    array([1, 2, 2, 2, 3, 4, 6])    # original order not preserved

    """

    if axis is None:
        return _unique_1d(
            ar, return_index, return_inverse, return_counts, equal_nan
        )

    # axis was specified and not None
    try:
        ar = dpnp.moveaxis(ar, axis, 0)
    except AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise AxisError(axis, ar.ndim) from None

    # reshape input array into a contiguous 2D array
    orig_sh = ar.shape
    ar = ar.reshape(orig_sh[0], math.prod(orig_sh[1:]))
    ar = dpnp.ascontiguousarray(ar)

    # build the indices for result array with unique values
    sorted_indices = _unique_build_sort_indices(ar, orig_sh[0])
    ar = ar[sorted_indices]

    if ar.size > 0:
        mask = dpnp.empty_like(ar, dtype=dpnp.bool)
        mask[:1] = True
        mask[1:] = ar[1:] != ar[:-1]

        mask = mask.any(axis=1)
    else:
        # if the array is empty, then the mask should grab the first empty
        # array as the unique one
        mask = dpnp.ones_like(ar, shape=(ar.shape[0]), dtype=dpnp.bool)
        mask[1:] = False

    # index the input array with the unique elements and reshape it into the
    # original size and dimension order
    ar = ar[mask]
    ar = ar.reshape(mask.sum().asnumpy(), *orig_sh[1:])
    ar = dpnp.moveaxis(ar, 0, axis)

    result = (ar,)
    if return_index:
        result += (sorted_indices[mask],)
    if return_inverse:
        imask = dpnp.cumsum(mask) - 1
        inv_idx = dpnp.empty_like(mask, dtype=dpnp.intp)
        inv_idx[sorted_indices] = imask
        result += (inv_idx,)
    if return_counts:
        nonzero = dpnp.nonzero(mask)[0]
        idx = dpnp.empty_like(
            nonzero, shape=(nonzero.size + 1,), dtype=nonzero.dtype
        )
        idx[:-1] = nonzero
        idx[-1] = mask.size
        result += (idx[1:] - idx[:-1],)

    return _unpack_tuple(result)


def unique_all(x, /):
    """
    Find the unique elements of an array, and counts, inverse, and indices.

    For full documentation refer to :obj:`numpy.unique_all`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    A namedtuple with the following attributes:

    values : dpnp.ndarray
        The unique elements of an input array.
    indices : dpnp.ndarray
        The first occurring indices for each unique element.
    inverse_indices : dpnp.ndarray
        The indices from the set of unique elements that reconstruct `x`.
    counts : dpnp.ndarray
        The corresponding counts for each unique element.

    See Also
    --------
    :obj:`dpnp.unique` : Find the unique elements of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 1, 2])
    >>> uniq = np.unique_all(x)
    >>> uniq.values
    array([1, 2])
    >>> uniq.indices
    array([0, 2])
    >>> uniq.inverse_indices
    array([0, 0, 1])
    >>> uniq.counts
    array([2, 1])

    """

    result = dpnp.unique(
        x,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        equal_nan=False,
    )
    return UniqueAllResult(*result)


def unique_counts(x, /):
    """
    Find the unique elements and counts of an input array `x`.

    For full documentation refer to :obj:`numpy.unique_counts`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    A namedtuple with the following attributes:

    values : dpnp.ndarray
        The unique elements of an input array.
    counts : dpnp.ndarray
        The corresponding counts for each unique element.

    See Also
    --------
    :obj:`dpnp.unique` : Find the unique elements of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 1, 2])
    >>> uniq = np.unique_counts(x)
    >>> uniq.values
    array([1, 2])
    >>> uniq.counts
    array([2, 1])

    """

    result = dpnp.unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=True,
        equal_nan=False,
    )
    return UniqueCountsResult(*result)


def unique_inverse(x, /):
    """
    Find the unique elements of `x` and indices to reconstruct `x`.

    For full documentation refer to :obj:`numpy.unique_inverse`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    A namedtuple with the following attributes:

    values : dpnp.ndarray
        The unique elements of an input array.
    inverse_indices : dpnp.ndarray
        The indices from the set of unique elements that reconstruct `x`.

    See Also
    --------
    :obj:`dpnp.unique` : Find the unique elements of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 1, 2])
    >>> uniq = np.unique_inverse(x)
    >>> uniq.values
    array([1, 2])
    >>> uniq.inverse_indices
    array([0, 0, 1])

    """

    result = dpnp.unique(
        x,
        return_index=False,
        return_inverse=True,
        return_counts=False,
        equal_nan=False,
    )
    return UniqueInverseResult(*result)


def unique_values(x, /):
    """
    Returns the unique elements of an input array `x`.

    For full documentation refer to :obj:`numpy.unique_values`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : dpnp.ndarray
        The unique elements of an input array.

    See Also
    --------
    :obj:`dpnp.unique` : Find the unique elements of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> np.unique_values(np.array([1, 1, 2]))
    array([1, 2])

    """

    return dpnp.unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=False,
        equal_nan=False,
    )


def unstack(x, /, *, axis=0):
    """
    Split an array into a sequence of arrays along the given axis.

    The `axis` parameter specifies the dimension along which the array will
    be split. For example, if ``axis=0`` (the default) it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    The result is a tuple of arrays split along `axis`.

    For full documentation refer to :obj:`numpy.unstack`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        The array to be unstacked.
    axis : int, optional
        Axis along which the array will be split.

        Default: ``0``.

    Returns
    -------
    unstacked : tuple of dpnp.ndarray
        The unstacked arrays.

    See Also
    --------
    :obj:`dpnp.stack` : Join a sequence of arrays along a new axis.
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.
    :obj:`dpnp.block` : Assemble an ndarray from nested lists of blocks.
    :obj:`dpnp.split` : Split array into a list of multiple sub-arrays of equal
                        size.

    Notes
    -----
    :obj:`dpnp.unstack` serves as the reverse operation of :obj:`dpnp.stack`,
    i.e., ``dpnp.stack(dpnp.unstack(x, axis=axis), axis=axis) == x``.

    This function is equivalent to ``tuple(dpnp.moveaxis(x, axis, 0))``, since
    iterating on an array iterates along the first axis.

    Examples
    --------
    >>> import dpnp as np
    >>> arr = np.arange(24).reshape((2, 3, 4))
    >>> np.unstack(arr)
    (array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]),
     array([[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]))

    >>> np.unstack(arr, axis=1)
    (array([[ 0,  1,  2,  3],
            [12, 13, 14, 15]]),
     array([[ 4,  5,  6,  7],
            [16, 17, 18, 19]]),
     array([[ 8,  9, 10, 11],
            [20, 21, 22, 23]]))

    >>> arr2 = np.stack(np.unstack(arr, axis=1), axis=1)
    >>> arr2.shape
    (2, 3, 4)
    >>> np.all(arr == arr2)
    array(True)

    """

    usm_x = dpnp.get_usm_ndarray(x)

    if usm_x.ndim == 0:
        raise ValueError("Input array must be at least 1-d.")

    res = dpt.unstack(usm_x, axis=axis)
    return tuple(dpnp_array._create_from_usm_ndarray(a) for a in res)


def vsplit(ary, indices_or_sections):
    """
    Split an array into multiple sub-arrays vertically (row-wise).

    Please refer to the :obj:`dpnp.split` documentation. ``vsplit``
    is equivalent to ``split`` with ``axis=0`` (default), the array
    is always split along the first axis regardless of the array dimension.

    For full documentation refer to :obj:`numpy.vsplit`.

    Parameters
    ----------
    ary : {dpnp.ndarray, usm_ndarray}
        Array to be divided into sub-arrays.
    indices_or_sections : {int, sequence of ints}
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along the first axis. If such a split is not
        possible, an error is raised.
        If `indices_or_sections` is a sequence of sorted integers, the entries
        indicate where along the first axis the array is split.

    Returns
    -------
    sub-arrays : list of dpnp.ndarray
        A list of sub arrays. Each array is a view of the corresponding input
        array.

    See Also
    --------
    :obj:`dpnp.split` : Split array into multiple sub-arrays of equal size.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(16.0).reshape(4, 4)
    >>> x
    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.],
           [12., 13., 14., 15.]])
    >>> np.vsplit(x, 2)
    [array([[0., 1., 2., 3.],
            [4., 5., 6., 7.]]),
     array([[ 8.,  9., 10., 11.],
            [12., 13., 14., 15.]])]
    >>> np.vsplit(x, np.array([3, 6]))
    [array([[ 0.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]]),
     array([[12., 13., 14., 15.]]),
     array([], shape=(0, 4), dtype=float64)]

    With a higher dimensional array the split is still along the first axis.

    >>> x = np.arange(8.0).reshape(2, 2, 2)
    >>> x
    array([[[0., 1.],
            [2., 3.]],
           [[4., 5.],
            [6., 7.]]])
    >>> np.vsplit(x, 2)
    [array([[[0., 1.],
             [2., 3.]]]),
     array([[[4., 5.],
             [6., 7.]]])]

    """

    dpnp.check_supported_arrays_type(ary)
    if ary.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return split(ary, indices_or_sections, 0)


def vstack(tup, *, dtype=None, casting="same_kind"):
    """
    Stack arrays in sequence vertically (row wise).

    :obj:`dpnp.row_stack` is an alias for :obj:`dpnp.vstack`.
    They are the same function.

    For full documentation refer to :obj:`numpy.vstack`.

    Parameters
    ----------
    tup : {dpnp.ndarray, usm_ndarray}
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.
    dtype : {None, str, dtype object}, optional
        If provided, the destination array will have this dtype.

        Default: ``None``.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

        Default: ``"same_kind"``.

    Returns
    -------
    out : dpnp.ndarray
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.
    :obj:`dpnp.stack` : Join a sequence of arrays along a new axis.
    :obj:`dpnp.hstack` : Stack arrays in sequence horizontally (column wise).
    :obj:`dpnp.dstack` : Stack arrays in sequence depth wise (along third axis).
    :obj:`dpnp.column_stack` : Stack 1-D arrays as columns into a 2-D array.
    :obj:`dpnp.block` : Assemble an ndarray from nested lists of blocks.
    :obj:`dpnp.split` : Split array into a list of multiple sub-arrays of equal
                        size.
    :obj:`dpnp.unstack` : Split an array into a tuple of sub-arrays along
                          an axis.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> np.vstack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> a = np.array([[1], [2], [3]])
    >>> b = np.array([[4], [5], [6]])
    >>> np.vstack((a, b))
    array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]])

    """

    _check_stack_arrays(tup)

    arrs = dpnp.atleast_2d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    return dpnp.concatenate(arrs, axis=0, dtype=dtype, casting=casting)


row_stack = vstack
