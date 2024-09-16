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
Interface of the Array manipulation routines part of the DPNP

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

import dpctl.tensor as dpt
import numpy
from dpctl.tensor._numpy_helper import AxisError, normalize_axis_index

import dpnp

from .dpnp_array import dpnp_array

__all__ = [
    "append",
    "array_split",
    "asarray_chkfinite",
    "asfarray",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "column_stack",
    "concat",
    "concatenate",
    "copyto",
    "dsplit",
    "dstack",
    "expand_dims",
    "flip",
    "fliplr",
    "flipud",
    "hsplit",
    "hstack",
    "moveaxis",
    "ndim",
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
        These values are appended to a copy of `arr`. It must be of the
        correct shape (the same shape as `arr`, excluding `axis`). If
        `axis` is not specified, `values` can be any shape and will be
        flattened before use.
        These values can be in any form that can be converted to an array.
        This includes scalars, lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists, and ndarrays.
    axis : {None, int}, optional
        The axis along which `values` are appended. If `axis` is not
        given, both `arr` and `values` are flattened before use.
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
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
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
    dtype : str or dtype object, optional
        Float type code to coerce input array `a`.  If `dtype` is ``None``,
        :obj:`dpnp.bool` or one of the `int` dtypes, it is replaced with
        the default floating type (:obj:`dpnp.float64` if a device supports it,
        or :obj:`dpnp.float32` type otherwise).
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by :obj:`dpnp.ndarray.device` property.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
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

    >>> x = np.arange(9.0).reshape(3,3)
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


# pylint: disable=redefined-outer-name
def broadcast_to(array, /, shape, subok=False):
    """
    Broadcast an array to a new shape.

    For full documentation refer to :obj:`numpy.broadcast_to`.

    Parameters
    ----------
    array : {dpnp.ndarray, usm_ndarray}
        The array to broadcast.
    shape : tuple or int
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
    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

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
    where : {dpnp.ndarray, usm_ndarray, scalar} of bool, optional
        A boolean array or a scalar which is broadcasted to match
        the dimensions of `dst`, and selects elements to copy
        from `src` to `dst` wherever it contains the value ``True``.

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
        src = dpnp.array(src, sycl_queue=dst.sycl_queue)

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
    axis : int or tuple of ints
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
    axis : None or int or tuple of ints, optional
         Axis or axes along which to flip over. The default,
         ``axis=None``, will flip over all of the axes of the input array.
         If `axis` is negative it counts from the last to the first axis.
         If `axis` is a tuple of integers, flipping is performed on all of
         the axes specified in the tuple.

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
    dtype : str or dtype
        If provided, the destination array will have this dtype.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

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


def ravel(a, order="C"):
    """
    Return a contiguous flattened array.

    For full documentation refer to :obj:`numpy.ravel`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array. The elements in `a` are read in the order specified by
        order, and packed as a 1-D array.
    order : {"C", "F"}, optional
        The elements of `a` are read using this index order. ``"C"`` means to
        index the elements in row-major, C-style order, with the last axis
        index changing fastest, back to the first axis index changing slowest.
        ``"F"`` means to index the elements in column-major, Fortran-style
        order, with the first index changing fastest, and the last index
        changing slowest. By default, ``"C"`` index order is used.

    Returns
    -------
    out : dpnp.ndarray
        A contiguous 1-D array of the same subtype as `a`, with shape (a.size,).

    See Also
    --------
    :obj:`dpnp.reshape` : Change the shape of an array without changing its
                          data.

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

    """

    return dpnp.reshape(a, -1, order=order)


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
    dtype : {None, data-type}, optional
       The required data-type. If ``None`` preserve the current dtype.
    requirements : {None, str, sequence of str}, optional
       The requirements list can be any of the following:

       * 'F_CONTIGUOUS' ('F') - ensure a Fortran-contiguous array
       * 'C_CONTIGUOUS' ('C') - ensure a C-contiguous array
       * 'WRITABLE' ('W') - ensure a writable array

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


def reshape(a, /, newshape, order="C", copy=None):
    """
    Gives a new shape to an array without changing its data.

    For full documentation refer to :obj:`numpy.reshape`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {"C", "F"}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order. ``"C"``
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. ``"F"`` means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the ``"C"`` and ``"F"`` options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
    copy : {None, bool}, optional
        Boolean indicating whether or not to copy the input array.
        If ``True``, the result array will always be a copy of input `a`.
        If ``False``, the result array can never be a copy
        and a ValueError exception will be raised in case the copy is necessary.
        If ``None``, the result array will reuse existing memory buffer of `a`
        if possible and copy otherwise.
        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        This will be a new view object if possible; otherwise, it will
        be a copy.  Note there is no guarantee of the *memory layout* (C- or
        Fortran- contiguous) of the returned array.

    Limitations
    -----------
    Parameter `order` is supported only with values ``"C"`` and ``"F"``.

    See Also
    --------
    :obj:`dpnp.ndarray.reshape` : Equivalent method.

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

    if newshape is None:
        newshape = a.shape

    if order is None:
        order = "C"
    elif order not in "cfCF":
        raise ValueError(f"order must be one of 'C' or 'F' (got {order})")

    usm_a = dpnp.get_usm_ndarray(a)
    usm_res = dpt.reshape(usm_a, shape=newshape, order=order, copy=copy)
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
    if axis is None:
        return roll(x.reshape(-1), shift, 0).reshape(x.shape)

    usm_x = dpnp.get_usm_ndarray(x)
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
    >>> a = np.ones((3,4,5,6))
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
    ``rot90(m, k=1, axes=(1,0))`` is the reverse of
    ``rot90(m, k=1, axes=(0,1))``.

    ``rot90(m, k=1, axes=(1,0))`` is equivalent to
    ``rot90(m, k=-1, axes=(0,1))``.

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
    shape : tuple of integers
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
    axis : None or int or tuple of ints, optional
        Selects a subset of the entries of length one in the shape.
        If an axis is selected with shape entry greater than one,
        an error is raised.

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
    out : dpnp.ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what stack would have returned if no out
        argument were specified.
    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

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

    >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.swapaxes(x,0,2)
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
    axes : None, tuple or list of ints, optional
        If specified, it must be a tuple or list which contains a permutation
        of [0, 1, ..., N-1] where N is the number of axes of `a`.
        The `i`'th axis of the returned array will correspond to the axis
        numbered ``axes[i]`` of the input. If not specified or ``None``,
        defaults to ``range(a.ndim)[::-1]``, which reverses the order of
        the axes.

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

    if isinstance(a, dpnp_array):
        array = a
    elif isinstance(a, dpt.usm_ndarray):
        array = dpnp_array._create_from_usm_ndarray(a)
    else:
        raise TypeError(
            f"An array must be any of supported type, but got {type(a)}"
        )

    if axes is None:
        return array.transpose()
    return array.transpose(*axes)


permute_dims = transpose  # permute_dims is an alias for transpose


def trim_zeros(filt, trim="fb"):
    """
    Trim the leading and/or trailing zeros from a 1-D array.

    For full documentation refer to :obj:`numpy.trim_zeros`.

    Parameters
    ----------
    filt : {dpnp.ndarray, usm_ndarray}
        Input 1-D array.
    trim : str, optional
        A string with 'f' representing trim from front and 'b' to trim from
        back. By defaults, trim zeros from both front and back of the array.
        Default: ``"fb"``.

    Returns
    -------
    out : dpnp.ndarray
        The result of trimming the input.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
    >>> np.trim_zeros(a)
    array([1, 2, 3, 0, 2, 1])

    >>> np.trim_zeros(a, 'b')
    array([0, 0, 0, 1, 2, 3, 0, 2, 1])

    """

    dpnp.check_supported_arrays_type(filt)
    if filt.ndim == 0:
        raise TypeError("0-d array cannot be trimmed")
    if filt.ndim > 1:
        raise ValueError("Multi-dimensional trim is not supported")

    if not isinstance(trim, str):
        raise TypeError("only string trim is supported")

    trim = trim.upper()
    if not any(x in trim for x in "FB"):
        return filt  # no trim rule is specified

    if filt.size == 0:
        return filt  # no trailing zeros in empty array

    a = dpnp.nonzero(filt)[0]
    a_size = a.size
    if a_size == 0:
        # 'filt' is array of zeros
        return dpnp.empty_like(filt, shape=(0,))

    first = 0
    if "F" in trim:
        first = a[0]

    last = filt.size
    if "B" in trim:
        last = a[-1] + 1

    return filt[first:last]


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


def vsplit(ary, indices_or_sections):
    """
    Split an array into multiple sub-arrays vertically (row-wise).

    Please refer to the :obj:`dpnp.split` documentation. ``vsplit``
    is equivalent to ``split`` with ``axis=0``(default), the array
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
    dtype : str or dtype
        If provided, the destination array will have this dtype.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

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
