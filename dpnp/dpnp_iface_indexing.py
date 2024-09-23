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
Interface of the Indexing part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

import operator

import dpctl.tensor as dpt
import numpy
from dpctl.tensor._numpy_helper import normalize_axis_index

import dpnp

# pylint: disable=no-name-in-module
from .dpnp_algo import (
    dpnp_choose,
    dpnp_putmask,
)
from .dpnp_array import dpnp_array
from .dpnp_utils import (
    call_origin,
    get_usm_allocations,
)

__all__ = [
    "choose",
    "diag_indices",
    "diag_indices_from",
    "diagonal",
    "extract",
    "fill_diagonal",
    "flatnonzero",
    "indices",
    "ix_",
    "mask_indices",
    "nonzero",
    "place",
    "put",
    "put_along_axis",
    "putmask",
    "ravel_multi_index",
    "select",
    "take",
    "take_along_axis",
    "tril_indices",
    "tril_indices_from",
    "triu_indices",
    "triu_indices_from",
    "unravel_index",
]


def _build_along_axis_index(a, ind, axis):
    """
    Build a fancy index used by a family of `_along_axis` functions.

    The fancy index consists of orthogonal arranges, with the
    requested index inserted at the right location.

    The resulting index is going to be used inside `dpnp.put_along_axis`
    and `dpnp.take_along_axis` implementations.

    """

    if not dpnp.issubdtype(ind.dtype, dpnp.integer):
        raise IndexError("`indices` must be an integer array")

    # normalize array shape and input axis
    if axis is None:
        a_shape = (a.size,)
        axis = 0
    else:
        a_shape = a.shape
        axis = normalize_axis_index(axis, a.ndim)

    if len(a_shape) != ind.ndim:
        raise ValueError(
            "`indices` and `a` must have the same number of dimensions"
        )

    # compute dimensions to iterate over
    dest_dims = list(range(axis)) + [None] + list(range(axis + 1, ind.ndim))
    shape_ones = (1,) * ind.ndim

    # build the index
    fancy_index = []
    for dim, n in zip(dest_dims, a_shape):
        if dim is None:
            fancy_index.append(ind)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
            fancy_index.append(
                dpnp.arange(
                    n,
                    dtype=ind.dtype,
                    usm_type=ind.usm_type,
                    sycl_queue=ind.sycl_queue,
                ).reshape(ind_shape)
            )

    return tuple(fancy_index)


def _ravel_multi_index_checks(multi_index, dims, order):
    dpnp.check_supported_arrays_type(*multi_index)
    ndim = len(dims)
    if len(multi_index) != ndim:
        raise ValueError(
            f"parameter multi_index must be a sequence of length {ndim}"
        )
    dim_mul = 1.0
    for d in dims:
        if not isinstance(d, int):
            raise TypeError(
                f"{type(d)} object cannot be interpreted as an integer"
            )
        dim_mul *= d

    if dim_mul > dpnp.iinfo(dpnp.int64).max:
        raise ValueError(
            "invalid dims: array size defined by dims is larger than the "
            "maximum possible size"
        )
    if order not in ("C", "c", "F", "f", None):
        raise ValueError(
            "Unrecognized `order` keyword value, expecting "
            f"'C' or 'F', but got '{order}'"
        )


def choose(x1, choices, out=None, mode="raise"):
    """
    Construct an array from an index array and a set of arrays to choose from.

    For full documentation refer to :obj:`numpy.choose`.

    See also
    --------
    :obj:`dpnp.take_along_axis` : Preferable if choices is an array.

    """
    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)

    choices_list = []
    for choice in choices:
        choices_list.append(
            dpnp.get_dpnp_descriptor(choice, copy_when_nondefault_queue=False)
        )

    if x1_desc:
        if any(not desc for desc in choices_list):
            pass
        elif out is not None:
            pass
        elif mode != "raise":
            pass
        elif any(not choices[0].dtype == choice.dtype for choice in choices):
            pass
        elif not choices_list:
            pass
        else:
            size = x1_desc.size
            choices_size = choices_list[0].size
            if any(
                choice.size != choices_size or choice.size != size
                for choice in choices
            ):
                pass
            elif any(x >= choices_size for x in dpnp.asnumpy(x1)):
                pass
            else:
                return dpnp_choose(x1_desc, choices_list).get_pyobj()

    return call_origin(numpy.choose, x1, choices, out, mode)


def diag_indices(n, ndim=2, device=None, usm_type="device", sycl_queue=None):
    """
    Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape
    (n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for
    ``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``
    for ``i = [0..n-1]``.

    For full documentation refer to :obj:`numpy.diag_indices`.

    Parameters
    ----------
    n : int
        The size, along each dimension, of the arrays for which the returned
        indices can be used.
    ndim : int, optional
        The number of dimensions. Default: ``2``.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.
        Default: ``None``.

    Returns
    -------
    out : tuple of dpnp.ndarray
        The indices to access the main diagonal of an array.

    See also
    --------
    :obj:`dpnp.diag_indices_from` : Return the indices to access the main
                                    diagonal of an n-dimensional array.

    Examples
    --------
    Create a set of indices to access the diagonal of a (4, 4) array:

    >>> import dpnp as np
    >>> di = np.diag_indices(4)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> a[di] = 100
    >>> a
    array([[100,   1,   2,   3],
           [  4, 100,   6,   7],
           [  8,   9, 100,  11],
           [ 12,  13,  14, 100]])

    Now, we create indices to manipulate a 3-D array:

    >>> d3 = np.diag_indices(2, 3)
    >>> d3
    (array([0, 1]), array([0, 1]), array([0, 1]))

    And use it to set the diagonal of an array of zeros to 1:

    >>> a = np.zeros((2, 2, 2), dtype=int)
    >>> a[d3] = 1
    >>> a
    array([[[1, 0],
            [0, 0]],
           [[0, 0],
            [0, 1]]])

    """

    idx = dpnp.arange(
        n,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )
    return (idx,) * ndim


def diag_indices_from(arr):
    """
    Return the indices to access the main diagonal of an n-dimensional array.

    For full documentation refer to :obj:`numpy.diag_indices_from`.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        Array at least 2-D.

    Returns
    -------
    out : tuple of dpnp.ndarray
        The indices to access the main diagonal of an n-dimensional array.

    See also
    --------
    :obj:`dpnp.diag_indices` : Return the indices to access the main diagonal
                               of an array.

    Examples
    --------
    Create a 4 by 4 array.

    >>> import dpnp as np
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Get the indices of the diagonal elements.

    >>> di = np.diag_indices_from(a)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    >>> a[di]
    array([ 0,  5, 10, 15])

    This is simply syntactic sugar for diag_indices.

    >>> np.diag_indices(a.shape[0])
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    """

    dpnp.check_supported_arrays_type(arr)

    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")

    if not numpy.all(numpy.diff(arr.shape) == 0):
        raise ValueError("All dimensions of input must be of equal length")

    return diag_indices(
        arr.shape[0],
        arr.ndim,
        usm_type=arr.usm_type,
        sycl_queue=arr.sycl_queue,
    )


def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    This function always returns a read/write view, and writing to
    the returned array will alter your original array.

    If you need to modify the array returned by this function without affecting
    the original array, we suggest copying the returned array explicitly, i.e.,
    use ``dpnp.diagonal(a).copy()`` instead of ``dpnp.diagonal(a)``.

    For full documentation refer to :obj:`numpy.diagonal`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be positive or
        negative. Defaults to main diagonal (``0``).
    axis1 : int, optional
        Axis to be used as the first axis of the 2-D sub-arrays from which
        the diagonals should be taken. Defaults to first axis (``0``).
    axis2 : int, optional
        Axis to be used as the second axis of the 2-D sub-arrays from
        which the diagonals should be taken. Defaults to second axis (``1``).

    Returns
    -------
    array_of_diagonals : dpnp.ndarray
        Array is a read/write view.
        If `a` is 2-D, then a 1-D array containing the diagonal and of the
        same type as `a` is returned.
        If ``a.ndim > 2``, then the dimensions specified by `axis1` and `axis2`
        are removed, and a new axis inserted at the end corresponding to the
        diagonal.

    See Also
    --------
    :obj:`dpnp.diag` : Extract a diagonal or construct a diagonal array.
    :obj:`dpnp.diagflat` : Create a two-dimensional array
                           with the flattened input as a diagonal.
    :obj:`dpnp.trace` : Return the sum along diagonals of the array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> a.diagonal()
    array([0, 3])
    >>> a.diagonal(1)
    array([1])

    A 3-D example:

    >>> a = np.arange(8).reshape(2,2,2)
    >>> a
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> a.diagonal(0,  # Main diagonals of two arrays created by skipping
    ...            0,  # across the outer(left)-most axis last and
    ...            1)  # the "middle" (row) axis first.
    array([[0, 6],
           [1, 7]])

    The sub-arrays whose main diagonals we just obtained; note that each
    corresponds to fixing the right-most (column) axis, and that the
    diagonals are "packed" in rows.

    >>> a[:,:,0]  # main diagonal is [0 6]
    array([[0, 2],
           [4, 6]])
    >>> a[:,:,1]  # main diagonal is [1 7]
    array([[1, 3],
           [5, 7]])

    The anti-diagonal can be obtained by reversing the order of elements
    using either `dpnp.flipud` or `dpnp.fliplr`.

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.fliplr(a).diagonal()  # Horizontal flip
    array([2, 4, 6])
    >>> np.flipud(a).diagonal()  # Vertical flip
    array([6, 4, 2])

    Note that the order in which the diagonal is retrieved varies depending
    on the flip function.

    """

    dpnp.check_supported_arrays_type(a)
    a_ndim = a.ndim

    if a_ndim < 2:
        raise ValueError("diag requires an array of at least two dimensions")

    if not isinstance(offset, int):
        raise TypeError(
            f"`offset` must be an integer data type, but got {type(offset)}"
        )

    axis1 = normalize_axis_index(axis1, a_ndim)
    axis2 = normalize_axis_index(axis2, a_ndim)

    if axis1 == axis2:
        raise ValueError("`axis1` and `axis2` cannot be the same")

    # get list of the order of all axes excluding the two target axes
    axes_order = [i for i in range(a_ndim) if i not in [axis1, axis2]]

    # transpose the input array to put the target axes at the end
    # to simplify diagonal extraction
    if offset >= 0:
        a = dpnp.transpose(a, axes_order + [axis1, axis2])
    else:
        a = dpnp.transpose(a, axes_order + [axis2, axis1])
        offset = -offset

    a_shape = a.shape
    a_straides = a.strides
    n, m = a_shape[-2:]
    st_n, st_m = a_straides[-2:]
    # pylint: disable=W0212
    a_element_offset = a.get_array()._element_offset

    # Compute shape, strides and offset of the resulting diagonal array
    # based on the input offset
    if offset == 0:
        out_shape = a_shape[:-2] + (min(n, m),)
        out_strides = a_straides[:-2] + (st_n + st_m,)
        out_offset = a_element_offset
    elif 0 < offset < m:
        out_shape = a_shape[:-2] + (min(n, m - offset),)
        out_strides = a_straides[:-2] + (st_n + st_m,)
        out_offset = a_element_offset + st_m * offset
    else:
        out_shape = a_shape[:-2] + (0,)
        out_strides = a_straides[:-2] + (1,)
        out_offset = a_element_offset

    return dpnp_array._create_from_usm_ndarray(
        dpt.usm_ndarray(
            out_shape,
            dtype=a.dtype,
            buffer=a.get_array(),
            strides=out_strides,
            offset=out_offset,
        )
    )


def extract(condition, a):
    """
    Return the elements of an array that satisfy some condition.

    This is equivalent to
    ``dpnp.compress(dpnp.ravel(condition), dpnp.ravel(a))``. If `condition`
    is boolean :obj:`dpnp.extract` is equivalent to ``a[condition]``.

    Note that :obj:`dpnp.place` does the exact opposite of :obj:`dpnp.extract`.

    For full documentation refer to :obj:`numpy.extract`.

    Parameters
    ----------
    condition : {array_like, scalar}
        An array whose non-zero or ``True`` entries indicate the element of `a`
        to extract.
    a : {dpnp.ndarray, usm_ndarray}
        Input array of the same size as `condition`.

    Returns
    -------
    out : dpnp.ndarray
        Rank 1 array of values from `a` where `condition` is ``True``.

    See Also
    --------
    :obj:`dpnp.take` : Take elements from an array along an axis.
    :obj:`dpnp.put` : Replaces specified elements of an array with given values.
    :obj:`dpnp.copyto` : Copies values from one array to another, broadcasting
                         as necessary.
    :obj:`dpnp.compress` : Return selected slices of an array along given axis.
    :obj:`dpnp.place` : Change elements of an array based on conditional and
                        input values.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(12).reshape((3, 4))
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> condition = np.mod(a, 3) == 0
    >>> condition
    array([[ True, False, False,  True],
           [False, False,  True, False],
           [False,  True, False, False]])
    >>> np.extract(condition, a)
    array([0, 3, 6, 9])

    If `condition` is boolean:

    >>> a[condition]
    array([0, 3, 6, 9])

    """

    usm_a = dpnp.get_usm_ndarray(a)
    usm_cond = dpnp.as_usm_ndarray(
        condition,
        dtype=dpnp.bool,
        usm_type=usm_a.usm_type,
        sycl_queue=usm_a.sycl_queue,
    )

    if usm_cond.size != usm_a.size:
        usm_a = dpt.reshape(usm_a, -1)
        usm_cond = dpt.reshape(usm_cond, -1)

        usm_res = dpt.take(usm_a, dpt.nonzero(usm_cond)[0])
    else:
        if usm_cond.shape != usm_a.shape:
            usm_a = dpt.reshape(usm_a, -1)
            usm_cond = dpt.reshape(usm_cond, -1)

        usm_res = dpt.extract(usm_cond, usm_a)

    return dpnp_array._create_from_usm_ndarray(usm_res)


def fill_diagonal(a, val, wrap=False):
    """
    Fill the main diagonal of the given array of any dimensionality.

    For an array `a` with ``a.ndim >= 2``, the diagonal is the list of values
    ``a[i, ..., i]`` with indices ``i`` all identical. This function modifies
    the input array in-place without returning a value.

    For full documentation refer to :obj:`numpy.fill_diagonal`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array whose diagonal is to be filled in-place. It must be at least 2-D.
    val : {dpnp.ndarray, usm_ndarray, scalar}
        Value(s) to write on the diagonal. If `val` is scalar, the value is
        written along the diagonal. If array, the flattened `val` is
        written along the diagonal, repeating if necessary to fill all
        diagonal entries.
    wrap : bool
        It enables the diagonal "wrapped" after N columns. This affects only
        tall matrices. Default: ``False``.

    See Also
    --------
    :obj:`dpnp.diag_indices` : Return the indices to access the main diagonal
                               of an array.
    :obj:`dpnp.diag_indices_from` : Return the indices to access the main
                                    diagonal of an n-dimensional array.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.zeros((3, 3), dtype=int)
    >>> np.fill_diagonal(a, 5)
    >>> a
    array([[5, 0, 0],
           [0, 5, 0],
           [0, 0, 5]])

    The same function can operate on a 4-D array:

    >>> a = np.zeros((3, 3, 3, 3), dtype=int)
    >>> np.fill_diagonal(a, 4)

    We only show a few blocks for clarity:

    >>> a[0, 0]
    array([[4, 0, 0],
           [0, 0, 0],
           [0, 0, 0]])
    >>> a[1, 1]
    array([[0, 0, 0],
           [0, 4, 0],
           [0, 0, 0]])
    >>> a[2, 2]
    array([[0, 0, 0],
           [0, 0, 0],
           [0, 0, 4]])

    The `wrap` option affects only tall matrices:

    >>> # tall matrices no wrap
    >>> a = np.zeros((5, 3), dtype=int)
    >>> np.fill_diagonal(a, 4)
    >>> a
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [0, 0, 0]])

    >>> # tall matrices wrap
    >>> a = np.zeros((5, 3), dtype=int)
    >>> np.fill_diagonal(a, 4, wrap=True)
    >>> a
    array([[4, 0, 0],
           [0, 4, 0],
           [0, 0, 4],
           [0, 0, 0],
           [4, 0, 0]])

    >>> # wide matrices
    >>> a = np.zeros((3, 5), dtype=int)
    >>> np.fill_diagonal(a, 4, wrap=True)
    >>> a
    array([[4, 0, 0, 0, 0],
           [0, 4, 0, 0, 0],
           [0, 0, 4, 0, 0]])

    The anti-diagonal can be filled by reversing the order of elements
    using either `dpnp.flipud` or `dpnp.fliplr`.

    >>> a = np.zeros((3, 3), dtype=int)
    >>> val = np.array([1, 2, 3])
    >>> np.fill_diagonal(np.fliplr(a), val)  # Horizontal flip
    >>> a
    array([[0, 0, 1],
           [0, 2, 0],
           [3, 0, 0]])
    >>> np.fill_diagonal(np.flipud(a), val)  # Vertical flip
    >>> a
    array([[0, 0, 3],
           [0, 2, 0],
           [1, 0, 0]])

    """

    usm_a = dpnp.get_usm_ndarray(a)
    usm_val = dpnp.get_usm_ndarray_or_scalar(val)

    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")

    end = a.size
    if a.ndim == 2:
        step = a.shape[1] + 1
        if not wrap and a.shape[0] > a.shape[1]:
            end = a.shape[1] * a.shape[1]
    else:
        if not numpy.all(numpy.diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        step = sum(a.shape[0] ** x for x in range(a.ndim))

    # TODO: implement flatiter for slice key
    # a.flat[:end:step] = val
    # but need to consider use case when `a` is usm_ndarray also
    a_sh = a.shape
    tmp_a = dpt.reshape(usm_a, -1)
    if dpnp.isscalar(usm_val):
        tmp_a[:end:step] = usm_val
    else:
        usm_val = dpt.reshape(usm_val, -1)

        # Setitem can work only if index size equal val size.
        # Using loop for general case without dependencies of val size.
        for i in range(0, usm_val.size):
            tmp_a[step * i : end : step * (i + 1)] = usm_val[i]

    tmp_a = dpt.reshape(tmp_a, a_sh)
    usm_a[:] = tmp_a


def flatnonzero(a):
    """
    Return indices that are non-zero in the flattened version of `a`.

    This is equivalent to ``dpnp.nonzero(dpnp.ravel(a))[0]``.

    For full documentation refer to :obj:`numpy.flatnonzero`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input data.

    Returns
    -------
    out : dpnp.ndarray
        Output array, containing the indices of the elements of ``a.ravel()``
        that are non-zero.

    See Also
    --------
    :obj:`dpnp.nonzero` : Return the indices of the non-zero elements of
                          the input array.
    :obj:`dpnp.ravel` : Return a 1-D array containing the elements of
                        the input array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(-2, 3)
    >>> x
    array([-2, -1,  0,  1,  2])
    >>> np.flatnonzero(x)
    array([0, 1, 3, 4])

    Use the indices of the non-zero elements as an index array to extract
    these elements:

    >>> x.ravel()[np.flatnonzero(x)]
    array([-2, -1,  1,  2])

    """

    return dpnp.nonzero(dpnp.ravel(a))[0]


def indices(
    dimensions,
    dtype=int,
    sparse=False,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return an array representing the indices of a grid.

    Compute an array where the subarrays contain index values 0, 1, …
    varying only along the corresponding axis.

    For full documentation refer to :obj:`numpy.indices`.

    Parameters
    ----------
    dimensions : sequence of ints
        The shape of the grid.
    dtype : {None, dtype}, optional
        Data type of the result.
    sparse : {None, boolean}, optional
        Return a sparse representation of the grid instead of a dense
        representation. Default is ``False``.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.
        Default: ``None``.

    Returns
    -------
    out : one dpnp.ndarray or tuple of dpnp.ndarray
        If sparse is ``False``:
        Returns one array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.

        If sparse is ``True``:
        Returns a tuple of arrays,
        with grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)
        with dimensions[i] in the i-th place.

    See Also
    --------
    :obj:`dpnp.mgrid` : Return a dense multi-dimensional “meshgrid”.
    :obj:`dpnp.ogrid` : Return an open multi-dimensional “meshgrid”.
    :obj:`dpnp.meshgrid` : Return a tuple of coordinate matrices from
                           coordinate vectors.

    Examples
    --------
    >>> import dpnp as np
    >>> grid = np.indices((2, 3))
    >>> grid.shape
    (2, 2, 3)
    >>> grid[0]
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> grid[1]
    array([[0, 1, 2],
           [0, 1, 2]])

    The indices can be used as an index into an array.

    >>> x = np.arange(20).reshape(5, 4)
    >>> row, col = np.indices((2, 3))
    >>> x[row, col]
    array([[0, 1, 2],
           [4, 5, 6]])

    Note that it would be more straightforward in the above example to
    extract the required elements directly with ``x[:2, :3]``.
    If sparse is set to ``True``, the grid will be returned in a sparse
    representation.

    >>> i, j = np.indices((2, 3), sparse=True)
    >>> i.shape
    (2, 1)
    >>> j.shape
    (1, 3)
    >>> i
    array([[0],
           [1]])
    >>> j
    array([[0, 1, 2]])

    """

    dimensions = tuple(dimensions)
    n = len(dimensions)
    shape = (1,) * n

    if sparse:
        res = ()
    else:
        res = dpnp.empty(
            (n,) + dimensions,
            dtype=dtype,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )

    for i, dim in enumerate(dimensions):
        idx = dpnp.arange(
            dim,
            dtype=dtype,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        ).reshape(shape[:i] + (dim,) + shape[i + 1 :])

        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


def ix_(*args):
    """Construct an open mesh from multiple sequences.

    This function takes N 1-D sequences and returns N outputs with N
    dimensions each, such that the shape is 1 in all but one dimension
    and the dimension with the non-unit shape value cycles through all
    N dimensions.

    Using :obj:`dpnp.ix_` one can quickly construct index arrays that will
    index the cross product. ``a[dpnp.ix_([1,3],[2,5])]`` returns the array
    ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.

    Parameters
    ----------
    x1, x2,..., xn : {dpnp.ndarray, usm_ndarray}
        1-D sequences. Each sequence should be of integer or boolean type.
        Boolean sequences will be interpreted as boolean masks for the
        corresponding dimension (equivalent to passing in
        ``dpnp.nonzero(boolean_sequence)``).

    Returns
    -------
    out : tuple of dpnp.ndarray
        N arrays with N dimensions each, with N the number of input sequences.
        Together these arrays form an open mesh.

    See Also
    --------
    :obj:`dpnp.mgrid` : Return a dense multi-dimensional “meshgrid”.
    :obj:`dpnp.ogrid` : Return an open multi-dimensional “meshgrid”.
    :obj:`dpnp.meshgrid` : Return a tuple of coordinate matrices from
                           coordinate vectors.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(10).reshape(2, 5)
    >>> a
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> x1 = np.array([0, 1])
    >>> x2 = np.array([2, 4])
    >>> ixgrid = np.ix_(x1, x2)
    >>> ixgrid
    (array([[0],
           [1]]), array([[2, 4]]))
    >>> ixgrid[0].shape, ixgrid[1].shape
    ((2, 1), (1, 2))
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])

    >>> x1 = np.array([True, True])
    >>> x2 = np.array([2, 4])
    >>> ixgrid = np.ix_(x1, x2)
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])
    >>> x1 = np.array([True, True])
    >>> x2 = np.array([False, False, True, False, True])
    >>> ixgrid = np.ix_(x1, x2)
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])

    """

    dpnp.check_supported_arrays_type(*args)

    out = []
    nd = len(args)
    for k, new in enumerate(args):
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        if dpnp.issubdtype(new.dtype, dpnp.bool):
            (new,) = dpnp.nonzero(new)
        new = dpnp.reshape(new, (1,) * k + (new.size,) + (1,) * (nd - k - 1))
        out.append(new)
    return tuple(out)


def mask_indices(
    n,
    mask_func,
    k=0,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return the indices to access (n, n) arrays, given a masking function.

    Assume `mask_func` is a function that, for a square array a of size
    ``(n, n)`` with a possible offset argument `k`, when called as
    ``mask_func(a, k=k)`` returns a new array with zeros in certain locations
    (functions like :obj:`dpnp.triu` or :obj:`dpnp.tril` do precisely this).
    Then this function returns the indices where the non-zero values would be
    located.

    Parameters
    ----------
    n : int
        The returned indices will be valid to access arrays of shape (n, n).
    mask_func : callable
        A function whose call signature is similar to that of :obj:`dpnp.triu`,
        :obj:`dpnp.tril`. That is, ``mask_func(x, k=k)`` returns a boolean
        array, shaped like `x`.`k` is an optional argument to the function.
    k : scalar
        An optional argument which is passed through to `mask_func`. Functions
        like :obj:`dpnp.triu`, :obj:`dpnp.tril` take a second argument that is
        interpreted as an offset. Default: ``0``.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.
        Default: ``None``.

    Returns
    -------
    indices : tuple of dpnp.ndarray
        The `n` arrays of indices corresponding to the locations where
        ``mask_func(np.ones((n, n)), k)`` is True.

    See Also
    --------
    :obj:`dpnp.tril` : Return lower triangle of an array.
    :obj:`dpnp.triu` : Return upper triangle of an array.
    :obj:`dpnp.triu_indices` : Return the indices for the upper-triangle of an
                               (n, m) array.
    :obj:`dpnp.tril_indices` : Return the indices for the lower-triangle of an
                               (n, m) array.

    Examples
    --------
    These are the indices that would allow you to access the upper triangular
    part of any 3x3 array:

    >>> import dpnp as np
    >>> iu = np.mask_indices(3, np.triu)

    For example, if `a` is a 3x3 array:

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> a[iu]
    array([0, 1, 2, 4, 5, 8])

    An offset can be passed also to the masking function. This gets us the
    indices starting on the first diagonal right of the main one:

    >>> iu1 = np.mask_indices(3, np.triu, 1)

    with which we now extract only three elements:

    >>> a[iu1]
    array([1, 2, 5])

    """

    m = dpnp.ones(
        (n, n),
        dtype=int,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )
    a = mask_func(m, k=k)
    return nonzero(a != 0)


def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `a`, containing
    the indices of the non-zero elements in that dimension. The values in `a`
    are always tested and returned in row-major, C-style order.

    To group the indices by element, rather than dimension, use
    :obj:`dpnp.argwhere`, which returns a row for each non-zero element.

    For full documentation refer to :obj:`numpy.nonzero`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.

    Returns
    -------
    out : tuple[dpnp.ndarray]
        Indices of elements that are non-zero.

    See Also
    --------
    :obj:`dpnp.flatnonzero` : Return indices that are non-zero in
                              the flattened version of the input array.
    :obj:`dpnp.ndarray.nonzero` : Equivalent ndarray method.
    :obj:`dpnp.count_nonzero` : Counts the number of non-zero elements
                                in the input array.

    Notes
    -----
    While the nonzero values can be obtained with ``a[nonzero(a)]``, it is
    recommended to use ``a[a.astype(bool)]`` or ``a[a != 0]`` instead, which
    will correctly handle 0-d arrays.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> x
    array([[3, 0, 0],
           [0, 4, 0],
           [5, 6, 0]])
    >>> np.nonzero(x)
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))

    >>> x[np.nonzero(x)]
    array([3, 4, 5, 6])
    >>> np.stack(np.nonzero(x)).T
    array([[0, 0],
           [1, 1],
           [2, 0],
           [2, 1]])

    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is ``True``. Given an array `a`, the condition `a` > 3 is
    a boolean array and since ``False`` is interpreted as ``0``,
    ``np.nonzero(a > 3)`` yields the indices of the `a` where the condition is
    true.

    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a > 3
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> np.nonzero(a > 3)
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    Using this result to index `a` is equivalent to using the mask directly:

    >>> a[np.nonzero(a > 3)]
    array([4, 5, 6, 7, 8, 9])
    >>> a[a > 3]  # prefer this spelling
    array([4, 5, 6, 7, 8, 9])

    ``nonzero`` can also be called as a method of the array.

    >>> (a > 3).nonzero()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    """

    usm_a = dpnp.get_usm_ndarray(a)
    return tuple(
        dpnp_array._create_from_usm_ndarray(y) for y in dpt.nonzero(usm_a)
    )


def place(a, mask, vals):
    """
    Change elements of an array based on conditional and input values.

    Similar to ``dpnp.copyto(a, vals, where=mask)``, the difference is that
    :obj:`dpnp.place` uses the first N elements of `vals`, where N is
    the number of ``True`` values in `mask`, while :obj:`dpnp.copyto` uses
    the elements where `mask` is ``True``.

    Note that :obj:`dpnp.extract` does the exact opposite of :obj:`dpnp.place`.

    For full documentation refer to :obj:`numpy.place`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Array to put data into.
    mask : {array_like, scalar}
        Boolean mask array. Must have the same size as `a`.
    vals : {array_like, scalar}
        Values to put into `a`. Only the first N elements are used, where N is
        the number of ``True`` values in `mask`. If `vals` is smaller than N,
        it will be repeated, and if elements of `a` are to be masked, this
        sequence must be non-empty.

    See Also
    --------
    :obj:`dpnp.copyto` : Copies values from one array to another.
    :obj:`dpnp.put` : Replaces specified elements of an array with given values.
    :obj:`dpnp.take` : Take elements from an array along an axis.
    :obj:`dpnp.extract` : Return the elements of an array that satisfy some
                         condition.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(6).reshape(2, 3)
    >>> np.place(a, a > 2, [44, 55])
    >>> a
    array([[ 0,  1,  2],
           [44, 55, 44]])

    """

    usm_a = dpnp.get_usm_ndarray(a)
    usm_mask = dpnp.as_usm_ndarray(
        mask,
        dtype=dpnp.bool,
        usm_type=usm_a.usm_type,
        sycl_queue=usm_a.sycl_queue,
    )
    usm_vals = dpnp.as_usm_ndarray(
        vals,
        dtype=usm_a.dtype,
        usm_type=usm_a.usm_type,
        sycl_queue=usm_a.sycl_queue,
    )

    if usm_vals.ndim != 1:
        # dpt.place supports only 1-D array of values
        usm_vals = dpt.reshape(usm_vals, -1)

    if usm_vals.dtype != usm_a.dtype:
        # dpt.place casts values to a.dtype with "unsafe" rule,
        # while numpy.place does that with "safe" casting rule
        usm_vals = dpt.astype(usm_vals, usm_a.dtype, casting="safe", copy=False)

    dpt.place(usm_a, usm_mask, usm_vals)


def put(a, ind, v, /, *, axis=None, mode="wrap"):
    """
    Puts values of an array into another array along a given axis.

    For full documentation refer to :obj:`numpy.put`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        The array the values will be put into.
    ind : {array_like}
        Target indices, interpreted as integers.
    v : {scalar, array_like}
         Values to be put into `a`. Must be broadcastable to the result shape
         ``a.shape[:axis] + ind.shape + a.shape[axis+1:]``.
    axis : {None, int}, optional
        The axis along which the values will be placed. If `a` is 1-D array,
        this argument is optional.
        Default: ``None``.
    mode : {'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        - 'wrap': clamps indices to (``-n <= i < n``), then wraps negative
          indices.
        - 'clip': clips indices to (``0 <= i < n``).

        Default: ``'wrap'``.

    See Also
    --------
    :obj:`dpnp.putmask` : Changes elements of an array based on conditional
                          and input values.
    :obj:`dpnp.place` : Change elements of an array based on conditional and
                        input values.
    :obj:`dpnp.put_along_axis` : Put values into the destination array
                                 by matching 1d index and data slices.

    Notes
    -----
    In contrast to :obj:`numpy.put` `wrap` mode which wraps indices around
    the array for cyclic operations, :obj:`dpnp.put` `wrap` mode clamps indices
    to a fixed range within the array boundaries (-n <= i < n).

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(5)
    >>> np.put(a, [0, 2], [-44, -55])
    >>> a
    array([-44,   1, -55,   3,   4])

    >>> a = np.arange(5)
    >>> np.put(a, 22, -5, mode='clip')
    >>> a
    array([ 0,  1,  2,  3, -5])

    """

    usm_a = dpnp.get_usm_ndarray(a)

    if not (axis is None or isinstance(axis, int)):
        raise TypeError(f"`axis` must be of integer type, got {type(axis)}")

    if mode not in ("wrap", "clip"):
        raise ValueError(
            f"clipmode must be one of 'clip' or 'wrap' (got '{mode}')"
        )

    usm_v = dpnp.as_usm_ndarray(
        v,
        dtype=usm_a.dtype,
        usm_type=usm_a.usm_type,
        sycl_queue=usm_a.sycl_queue,
    )
    if usm_v.size == 0:
        return

    usm_ind = dpnp.as_usm_ndarray(
        ind,
        dtype=dpnp.intp,
        usm_type=usm_a.usm_type,
        sycl_queue=usm_a.sycl_queue,
    )

    if usm_ind.ndim != 1:
        # dpt.put supports only 1-D array of indices
        usm_ind = dpt.reshape(usm_ind, -1, copy=False)

    if not dpnp.issubdtype(usm_ind.dtype, dpnp.integer):
        # dpt.put supports only integer dtype for array of indices
        usm_ind = dpt.astype(usm_ind, dpnp.intp, casting="safe")

    in_usm_a = usm_a
    if axis is None and usm_a.ndim > 1:
        usm_a = dpt.reshape(usm_a, -1)

    dpt.put(usm_a, usm_ind, usm_v, axis=axis, mode=mode)
    if in_usm_a._pointer != usm_a._pointer:  # pylint: disable=protected-access
        in_usm_a[:] = dpt.reshape(usm_a, in_usm_a.shape, copy=False)


def put_along_axis(a, ind, values, axis):
    """
    Put values into the destination array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to place values into the
    latter. These slices can be different lengths.

    Functions returning an index along an `axis`, like :obj:`dpnp.argsort` and
    :obj:`dpnp.argpartition`, produce suitable indices for this function.

    For full documentation refer to :obj:`numpy.put_along_axis`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}, (Ni..., M, Nk...)
        Destination array.
    ind : {dpnp.ndarray, usm_ndarray}, (Ni..., J, Nk...)
        Indices to change along each 1d slice of `a`. This must match the
        dimension of input array, but dimensions in ``Ni`` and ``Nj``
        may be 1 to broadcast against `a`.
    values : {scalar, array_like}, (Ni..., J, Nk...)
        Values to insert at those indices. Its shape and dimension are
        broadcast to match that of `ind`.
    axis : int
        The axis to take 1d slices along. If axis is ``None``, the destination
        array is treated as if a flattened 1d view had been created of it.

    See Also
    --------
    :obj:`dpnp.put` : Put values along an axis, using the same indices
                      for every 1d slice.
    :obj:`dpnp.take_along_axis` : Take values from the input array
                                  by matching 1d index and data slices.

    Examples
    --------
    For this sample array

    >>> import dpnp as np
    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    We can replace the maximum values with:

    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> ai
    array([[1],
           [0]])
    >>> np.put_along_axis(a, ai, 99, axis=1)
    >>> a
    array([[10, 99, 20],
           [99, 40, 50]])

    """

    dpnp.check_supported_arrays_type(a, ind)

    if axis is None:
        a = a.ravel()

    a[_build_along_axis_index(a, ind, axis)] = values


def putmask(x1, mask, values):
    """
    Changes elements of an array based on conditional and input values.

    For full documentation refer to :obj:`numpy.putmask`.

    Limitations
    -----------
    Input arrays ``arr``, ``mask`` and ``values`` are supported
    as :obj:`dpnp.ndarray`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(
        x1, copy_when_strides=False, copy_when_nondefault_queue=False
    )
    mask_desc = dpnp.get_dpnp_descriptor(mask, copy_when_nondefault_queue=False)
    values_desc = dpnp.get_dpnp_descriptor(
        values, copy_when_nondefault_queue=False
    )
    if x1_desc and mask_desc and values_desc:
        return dpnp_putmask(x1_desc, mask_desc, values_desc)

    return call_origin(numpy.putmask, x1, mask, values, dpnp_inplace=True)


def ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    """
    Converts a tuple of index arrays into an array of flat indices, applying
    boundary modes to the multi-index.

    For full documentation refer to :obj:`numpy.ravel_multi_index`.

    Parameters
    ----------
    multi_index : tuple of {dpnp.ndarray, usm_ndarray}
        A tuple of integer arrays, one array for each dimension.
    dims : tuple or list of ints
        The shape of array into which the indices from `multi_index` apply.
    mode : {"raise", "wrap" or "clip'}, optional
        Specifies how out-of-bounds indices are handled. Can specify either
        one mode or a tuple of modes, one mode per index:

        - "raise" -- raise an error
        - "wrap" -- wrap around
        - "clip" -- clip to the range

        In ``"clip"`` mode, a negative index which would normally wrap will
        clip to 0 instead.
        Default: ``"raise"``.
    order : {None, "C", "F"}, optional
        Determines whether the multi-index should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.
        Default: ``"C"``.

    Returns
    -------
    raveled_indices : dpnp.ndarray
        An array of indices into the flattened version of an array of
        dimensions `dims`.

    See Also
    --------
    :obj:`dpnp.unravel_index` : Converts array of flat indices into a tuple of
                                coordinate arrays.

    Examples
    --------
    >>> import dpnp as np
    >>> arr = np.array([[3, 6, 6], [4, 5, 1]])
    >>> np.ravel_multi_index(arr, (7, 6))
    array([22, 41, 37])
    >>> np.ravel_multi_index(arr, (7, 6), order="F")
    array([31, 41, 13])
    >>> np.ravel_multi_index(arr, (4, 6), mode="clip")
    array([22, 23, 19])
    >>> np.ravel_multi_index(arr, (4, 4), mode=("clip", "wrap"))
    array([12, 13, 13])

    >>> arr = np.array([3, 1, 4, 1])
    >>> np.ravel_multi_index(arr, (6, 7, 8, 9))
    array(1621)

    """

    _ravel_multi_index_checks(multi_index, dims, order)

    ndim = len(dims)
    if isinstance(mode, str):
        mode = (mode,) * ndim

    s = 1
    ravel_strides = [1] * ndim

    multi_index = tuple(multi_index)
    usm_type_alloc, sycl_queue_alloc = get_usm_allocations(multi_index)

    order = "C" if order is None else order.upper()
    if order == "C":
        for i in range(ndim - 2, -1, -1):
            s = s * dims[i + 1]
            ravel_strides[i] = s
    else:
        for i in range(1, ndim):
            s = s * dims[i - 1]
            ravel_strides[i] = s

    multi_index = dpnp.broadcast_arrays(*multi_index)
    raveled_indices = dpnp.zeros(
        multi_index[0].shape,
        dtype=dpnp.int64,
        usm_type=usm_type_alloc,
        sycl_queue=sycl_queue_alloc,
    )
    for d, stride, idx, _mode in zip(dims, ravel_strides, multi_index, mode):
        if not dpnp.can_cast(idx, dpnp.int64, "same_kind"):
            raise TypeError(
                f"multi_index entries could not be cast from dtype({idx.dtype})"
                f" to dtype({dpnp.int64}) according to the rule 'same_kind'"
            )
        idx = idx.astype(dpnp.int64, copy=False)

        if _mode == "raise":
            if dpnp.any(dpnp.logical_or(idx >= d, idx < 0)):
                raise ValueError("invalid entry in coordinates array")
        elif _mode == "clip":
            idx = dpnp.clip(idx, 0, d - 1)
        elif _mode == "wrap":
            idx = idx % d
        else:
            raise ValueError(f"Unrecognized mode: {_mode}")
        raveled_indices += stride * idx
    return raveled_indices


def select(condlist, choicelist, default=0):
    """
    Return an array drawn from elements in `choicelist`, depending on
    conditions.

    For full documentation refer to :obj:`numpy.select`.

    Parameters
    ----------
    condlist : list of bool dpnp.ndarray or usm_ndarray
        The list of conditions which determine from which array in `choicelist`
        the output elements are taken. When multiple conditions are satisfied,
        the first one encountered in `condlist` is used.
    choicelist : list of dpnp.ndarray or usm_ndarray
        The list of arrays from which the output elements are taken. It has
        to be of the same length as `condlist`.
    default : {scalar, dpnp.ndarray, usm_ndarray}, optional
        The element inserted in `output` when all conditions evaluate to
        ``False``. Default: ``0``.

    Returns
    -------
    out : dpnp.ndarray
        The output at position m is the m-th element of the array in
        `choicelist` where the m-th element of the corresponding array in
        `condlist` is ``True``.

    See Also
    --------
    :obj:`dpnp.where` : Return elements from one of two arrays depending on
                       condition.
    :obj:`dpnp.take` : Take elements from an array along an axis.
    :obj:`dpnp.choose` : Construct an array from an index array and a set of
                         arrays to choose from.
    :obj:`dpnp.compress` : Return selected slices of an array along given axis.
    :obj:`dpnp.diag` : Extract a diagonal or construct a diagonal array.
    :obj:`dpnp.diagonal` : Return specified diagonals.

    Examples
    --------
    >>> import dpnp as np

    Beginning with an array of integers from 0 to 5 (inclusive),
    elements less than ``3`` are negated, elements greater than ``3``
    are squared, and elements not meeting either of these conditions
    (exactly ``3``) are replaced with a `default` value of ``42``.

    >>> x = np.arange(6)
    >>> condlist = [x<3, x>3]
    >>> choicelist = [x, x**2]
    >>> np.select(condlist, choicelist, 42)
    array([ 0,  1,  2, 42, 16, 25])

    When multiple conditions are satisfied, the first one encountered in
    `condlist` is used.

    >>> condlist = [x<=4, x>3]
    >>> choicelist = [x, x**2]
    >>> np.select(condlist, choicelist, 55)
    array([ 0,  1,  2,  3,  4, 25])

    """

    if len(condlist) != len(choicelist):
        raise ValueError(
            "list of cases must be same length as list of conditions"
        )

    if len(condlist) == 0:
        raise ValueError("select with an empty condition list is not possible")

    dpnp.check_supported_arrays_type(*condlist)
    dpnp.check_supported_arrays_type(*choicelist)

    if not dpnp.isscalar(default) and not (
        dpnp.is_supported_array_type(default) and default.ndim == 0
    ):
        raise TypeError(
            "A default value must be any of scalar or 0-d supported array type"
        )

    dtype = dpnp.result_type(*choicelist, default)

    usm_type_alloc, sycl_queue_alloc = get_usm_allocations(
        condlist + choicelist + [default]
    )

    for i, cond in enumerate(condlist):
        if not dpnp.issubdtype(cond, dpnp.bool):
            raise TypeError(
                f"invalid entry {i} in condlist: should be boolean ndarray"
            )

    # Convert conditions to arrays and broadcast conditions and choices
    # as the shape is needed for the result
    condlist = dpnp.broadcast_arrays(*condlist)
    choicelist = dpnp.broadcast_arrays(*choicelist)

    result_shape = dpnp.broadcast_arrays(condlist[0], choicelist[0])[0].shape

    result = dpnp.full(
        result_shape,
        default,
        dtype=dtype,
        usm_type=usm_type_alloc,
        sycl_queue=sycl_queue_alloc,
    )

    # Do in reverse order since the first choice should take precedence.
    choicelist = choicelist[::-1]
    condlist = condlist[::-1]
    for choice, cond in zip(choicelist, condlist):
        dpnp.where(cond, choice, result, out=result)

    return result


# pylint: disable=redefined-outer-name
def take(a, indices, /, *, axis=None, out=None, mode="wrap"):
    """
    Take elements from an array along an axis.

    When `axis` is not ``None``, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``dpnp.take(a, indices, axis=3)`` is equivalent to
    ``a[:, :, :, indices, ...]``.

    For full documentation refer to :obj:`numpy.take`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}, (Ni..., M, Nk...)
        The source array.
    indices : {array_like, scalars}, (Nj...)
        The indices of the values to extract.
        Also allow scalars for `indices`.
    axis : {None, int, bool, 0-d array of integer dtype}, optional
        The axis over which to select values. By default, the flattened
        input array is used.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional (Ni..., Nj..., Nk...)
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype.
        Default: ``None``.
    mode : {"wrap", "clip"}, optional
        Specifies how out-of-bounds indices will be handled. Possible values
        are:

        - ``"wrap"``: clamps indices to (``-n <= i < n``), then wraps
          negative indices.
        - ``"clip"``: clips indices to (``0 <= i < n``).

        Default: ``"wrap"``.

    Returns
    -------
    out : dpnp.ndarray, (Ni..., Nj..., Nk...)
        The returned array has the same type as `a`.

    See Also
    --------
    :obj:`dpnp.compress` : Take elements using a boolean mask.
    :obj:`dpnp.ndarray.take` : Equivalent method.
    :obj:`dpnp.take_along_axis` : Take elements by matching the array and
                                  the index arrays.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([4, 3, 5, 7, 6, 8])
    >>> indices = np.array([0, 1, 4])
    >>> np.take(x, indices)
    array([4, 3, 6])

    In this example "fancy" indexing can be used.

    >>> x[indices]
    array([4, 3, 6])

    >>> indices = dpnp.array([-1, -6, -7, 5, 6])
    >>> np.take(x, indices)
    array([8, 4, 4, 8, 8])

    >>> np.take(x, indices, mode="clip")
    array([4, 4, 4, 8, 8])

    If `indices` is not one dimensional, the output also has these dimensions.

    >>> np.take(x, [[0, 1], [2, 3]])
    array([[4, 3],
           [5, 7]])

    """

    if mode not in ("wrap", "clip"):
        raise ValueError(f"`mode` must be 'wrap' or 'clip', but got `{mode}`.")

    usm_a = dpnp.get_usm_ndarray(a)
    if not dpnp.is_supported_array_type(indices):
        usm_ind = dpt.asarray(
            indices, usm_type=a.usm_type, sycl_queue=a.sycl_queue
        )
    else:
        usm_ind = dpnp.get_usm_ndarray(indices)

    a_ndim = a.ndim
    if axis is None:
        res_shape = usm_ind.shape

        if a_ndim > 1:
            # dpt.take requires flattened input array
            usm_a = dpt.reshape(usm_a, -1)
    elif a_ndim == 0:
        axis = normalize_axis_index(operator.index(axis), 1)
        res_shape = usm_ind.shape
    else:
        axis = normalize_axis_index(operator.index(axis), a_ndim)
        a_sh = a.shape
        res_shape = a_sh[:axis] + usm_ind.shape + a_sh[axis + 1 :]

    if usm_ind.ndim != 1:
        # dpt.take supports only 1-D array of indices
        usm_ind = dpt.reshape(usm_ind, -1)

    if not dpnp.issubdtype(usm_ind.dtype, dpnp.integer):
        # dpt.take supports only integer dtype for array of indices
        usm_ind = dpt.astype(usm_ind, dpnp.intp, copy=False, casting="safe")

    usm_res = dpt.take(usm_a, usm_ind, axis=axis, mode=mode)

    # need to reshape the result if shape of indices array was changed
    result = dpnp.reshape(usm_res, res_shape)
    return dpnp.get_result_array(result, out)


def take_along_axis(a, indices, axis, mode="wrap"):
    """
    Take values from the input array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to look up values in the
    latter. These slices can be different lengths.

    Functions returning an index along an `axis`, like :obj:`dpnp.argsort` and
    :obj:`dpnp.argpartition`, produce suitable indices for this function.

    For full documentation refer to :obj:`numpy.take_along_axis`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}, (Ni..., M, Nk...)
        Source array
    indices : {dpnp.ndarray, usm_ndarray}, (Ni..., J, Nk...)
        Indices to take along each 1d slice of `a`. This must match the
        dimension of the input array, but dimensions ``Ni`` and ``Nj``
        only need to broadcast against `a`.
    axis : {None, int}
        The axis to take 1d slices along. If axis is ``None``, the input
        array is treated as if it had first been flattened to 1d,
        for consistency with :obj:`dpnp.sort` and :obj:`dpnp.argsort`.
    mode : {"wrap", "clip"}, optional
        Specifies how out-of-bounds indices will be handled. Possible values
        are:

        - ``"wrap"``: clamps indices to (``-n <= i < n``), then wraps
          negative indices.
        - ``"clip"``: clips indices to (``0 <= i < n``).

        Default: ``"wrap"``.

    Returns
    -------
    out : dpnp.ndarray
        The indexed result of the same data type as `a`.

    See Also
    --------
    :obj:`dpnp.take` : Take along an axis, using the same indices for
                       every 1d slice.
    :obj:`dpnp.put_along_axis` : Put values into the destination array
                                 by matching 1d index and data slices.
    :obj:`dpnp.argsort` : Return the indices that would sort an array.

    Examples
    --------
    For this sample array

    >>> import dpnp as np
    >>> a = np.array([[10, 30, 20], [60, 40, 50]])

    We can sort either by using :obj:`dpnp.sort` directly, or
    :obj:`dpnp.argsort` and this function:

    >>> np.sort(a, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])
    >>> ai = np.argsort(a, axis=1)
    >>> ai
    array([[0, 2, 1],
           [1, 2, 0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 20, 30],
           [40, 50, 60]])

    The same works for max and min, if you maintain the trivial dimension
    with ``keepdims``:

    >>> np.max(a, axis=1, keepdims=True)
    array([[30],
           [60]])
    >>> ai = np.argmax(a, axis=1, keepdims=True)
    >>> ai
    array([[1],
           [0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[30],
           [60]])

    If we want to get the max and min at the same time, we can stack the
    indices first:

    >>> ai_min = np.argmin(a, axis=1, keepdims=True)
    >>> ai_max = np.argmax(a, axis=1, keepdims=True)
    >>> ai = np.concatenate([ai_min, ai_max], axis=1)
    >>> ai
    array([[0, 1],
           [1, 0]])
    >>> np.take_along_axis(a, ai, axis=1)
    array([[10, 30],
           [40, 60]])

    """

    if axis is None:
        dpnp.check_supported_arrays_type(indices)
        if indices.ndim != 1:
            raise ValueError(
                "when axis=None, `indices` must have a single dimension."
            )

        a = dpnp.ravel(a)
        axis = 0

    usm_a = dpnp.get_usm_ndarray(a)
    usm_ind = dpnp.get_usm_ndarray(indices)

    usm_res = dpt.take_along_axis(usm_a, usm_ind, axis=axis, mode=mode)
    return dpnp_array._create_from_usm_ndarray(usm_res)


def tril_indices(
    n,
    k=0,
    m=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return the indices for the lower-triangle of an (n, m) array.

    For full documentation refer to :obj:`numpy.tril_indices`.

    Parameters
    ----------
    n : int
        The row dimension of the arrays for which the returned
        indices will be valid.
    k : int, optional
        Diagonal offset (see :obj:`dpnp.tril` for details). Default: ``0``.
    m : {None, int}, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`. Default: ``None``.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.
        Default: ``None``.

    Returns
    -------
    inds : tuple of dpnp.ndarray
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.

    See Also
    --------
    :obj:`dpnp.triu_indices` : similar function, for upper-triangular.
    :obj:`dpnp.mask_indices` : generic function accepting an arbitrary mask
                               function.
    :obj:`dpnp.tril` : Return lower triangle of an array.
    :obj:`dpnp.triu` : Return upper triangle of an array.

    Examples
    --------
    Compute two different sets of indices to access 4x4 arrays, one for the
    lower triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> import dpnp as np
    >>> il1 = np.tril_indices(4)
    >>> il2 = np.tril_indices(4, 2)

    Here is how they can be used with a sample array:

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Both for indexing:

    >>> a[il1]
    array([ 0,  4,  5, ..., 13, 14, 15])

    And for assigning values:

    >>> a[il1] = -1
    >>> a
    array([[-1,  1,  2,  3],
           [-1, -1,  6,  7],
           [-1, -1, -1, 11],
           [-1, -1, -1, -1]])

    These cover almost the whole array (two diagonals right of the main one):

    >>> a[il2] = -10
    >>> a
    array([[-10, -10, -10,   3],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10],
           [-10, -10, -10, -10]])

    """

    tri_ = dpnp.tri(
        n,
        m,
        k=k,
        dtype=bool,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )

    return tuple(
        dpnp.broadcast_to(inds, tri_.shape)[tri_]
        for inds in indices(
            tri_.shape,
            sparse=True,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
    )


def tril_indices_from(arr, k=0):
    """
    Return the indices for the lower-triangle of arr.

    For full documentation refer to :obj:`numpy.tril_indices_from`.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        The indices will be valid for square arrays whose dimensions are
        the same as arr.
    k : int, optional
        Diagonal offset (see :obj:`dpnp.tril` for details). Default: ``0``.

    Returns
    -------
    inds : tuple of dpnp.ndarray
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array.

    See Also
    --------
    :obj:`dpnp.tril_indices` : Return the indices for the lower-triangle of an
                               (n, m) array.
    :obj:`dpnp.tril` : Return lower triangle of an array.
    :obj:`dpnp.triu_indices_from` : similar function, for upper-triangular.

    Examples
    --------
    Create a 4 by 4 array.

    >>> import dpnp as np
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Pass the array to get the indices of the lower triangular elements.

    >>> trili = np.tril_indices_from(a)
    >>> trili
    (array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
     array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    >>> a[trili]
    array([ 0,  4,  5,  8,  9, 10, 12, 13, 14, 15])

    This is syntactic sugar for tril_indices().

    >>> np.tril_indices(a.shape[0])
    (array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
     array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))

    Use the `k` parameter to return the indices for the lower triangular array
    up to the k-th diagonal.

    >>> trili1 = np.tril_indices_from(a, k=1)
    >>> a[trili1]
    array([ 0,  1,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15])

    """

    dpnp.check_supported_arrays_type(arr)

    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")

    return tril_indices(
        arr.shape[-2],
        k=k,
        m=arr.shape[-1],
        usm_type=arr.usm_type,
        sycl_queue=arr.sycl_queue,
    )


def triu_indices(
    n,
    k=0,
    m=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return the indices for the upper-triangle of an (n, m) array.

    For full documentation refer to :obj:`numpy.triu_indices`.

    Parameters
    ----------
    n : int
        The size of the arrays for which the returned indices will
        be valid.
    k : int, optional
        Diagonal offset (see :obj:`dpnp.triu` for details). Default: ``0``.
    m : int, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`. Default: ``None``.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector
        string, an instance of :class:`dpctl.SyclDevice` corresponding to
        a non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.
        Default: ``None``.

    Returns
    -------
    inds : tuple of dpnp.ndarray
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array. Can be used
        to slice a ndarray of shape(`n`, `n`).

    See Also
    --------
    :obj:`dpnp.tril_indices` : similar function, for lower-triangular.
    :obj:`dpnp.mask_indices` : generic function accepting an arbitrary mask
                               function.
    :obj:`dpnp.tril` : Return lower triangle of an array.
    :obj:`dpnp.triu` : Return upper triangle of an array.

    Examples
    --------
    Compute two different sets of indices to access 4x4 arrays, one for the
    upper triangular part starting at the main diagonal, and one starting two
    diagonals further right:

    >>> import dpnp as np
    >>> iu1 = np.triu_indices(4)
    >>> iu2 = np.triu_indices(4, 2)

    Here is how they can be used with a sample array:

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Both for indexing:

    >>> a[iu1]
    array([ 0,  1,  2, ..., 10, 11, 15])

    And for assigning values:

    >>> a[iu1] = -1
    >>> a
    array([[-1, -1, -1, -1],
           [ 4, -1, -1, -1],
           [ 8,  9, -1, -1],
           [12, 13, 14, -1]])

    These cover only a small part of the whole array (two diagonals right
    of the main one):

    >>> a[iu2] = -10
    >>> a
    array([[ -1,  -1, -10, -10],
           [  4,  -1,  -1, -10],
           [  8,   9,  -1,  -1],
           [ 12,  13,  14,  -1]])

    """

    tri_ = ~dpnp.tri(
        n,
        m,
        k=k - 1,
        dtype=bool,
        device=device,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )

    return tuple(
        dpnp.broadcast_to(inds, tri_.shape)[tri_]
        for inds in indices(
            tri_.shape,
            sparse=True,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
    )


def triu_indices_from(arr, k=0):
    """
    Return the indices for the lower-triangle of arr.

    For full documentation refer to :obj:`numpy.triu_indices_from`.

    Parameters
    ----------
    arr : {dpnp.ndarray, usm_ndarray}
        The indices will be valid for square arrays whose dimensions are
        the same as arr.
    k : int, optional
        Diagonal offset (see :obj:`dpnp.triu` for details). Default: ``0``.

    Returns
    -------
    inds : tuple of dpnp.ndarray
        The indices for the triangle. The returned tuple contains two arrays,
        each with the indices along one dimension of the array. Can be used
        to slice a ndarray of shape(`n`, `n`).

    See Also
    --------
    :obj:`dpnp.triu_indices` : Return the indices for the upper-triangle of an
                               (n, m) array.
    :obj:`dpnp.triu` : Return upper triangle of an array.
    :obj:`dpnp.tril_indices_from` : similar function, for lower-triangular.

    Examples
    --------
    Create a 4 by 4 array.

    >>> import dpnp as np
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])

    Pass the array to get the indices of the upper triangular elements.

    >>> triui = np.triu_indices_from(a)
    >>> triui
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
     array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    >>> a[triui]
    array([ 0,  1,  2,  3,  5,  6,  7, 10, 11, 15])

    This is syntactic sugar for triu_indices().

    >>> np.triu_indices(a.shape[0])
    (array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3]),
     array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]))

    Use the `k` parameter to return the indices for the upper triangular array
    from the k-th diagonal.

    >>> triuim1 = np.triu_indices_from(a, k=1)
    >>> a[triuim1]
    array([ 1,  2,  3,  6,  7, 11])

    """

    dpnp.check_supported_arrays_type(arr)

    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")

    return triu_indices(
        arr.shape[-2],
        k=k,
        m=arr.shape[-1],
        usm_type=arr.usm_type,
        sycl_queue=arr.sycl_queue,
    )


def unravel_index(indices, shape, order="C"):
    """
    Converts array of flat indices into a tuple of coordinate arrays.

    For full documentation refer to :obj:`numpy.unravel_index`.

    Parameters
    ----------
    indices : {dpnp.ndarray, usm_ndarray}
        An integer array whose elements are indices into the flattened version
        of an array of dimensions `shape`.
    shape : tuple or list of ints
        The shape of the array to use for unraveling `indices`.
    order : {None, "C", "F"}, optional
        Determines whether the indices should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.
        Default: ``"C"``.

    Returns
    -------
    unraveled_coords : tuple of dpnp.ndarray
        Each array in the tuple has the same shape as the indices array.

    See Also
    --------
    :obj:`dpnp.ravel_multi_index` : Converts a tuple of index arrays into an
                                    array of flat indices.

    Examples
    --------
    >>> import dpnp as np
    >>> np.unravel_index(np.array([22, 41, 37]), (7, 6))
    (array([3, 6, 6]), array([4, 5, 1]))
    >>> np.unravel_index(np.array([31, 41, 13]), (7, 6), order="F")
    (array([3, 6, 6]), array([4, 5, 1]))

    >>> np.unravel_index(np.array(1621), (6, 7, 8, 9))
    (array(3), array(1), array(4), array(1))

    """

    dpnp.check_supported_arrays_type(indices)

    if order not in ("C", "c", "F", "f", None):
        raise ValueError(
            "Unrecognized `order` keyword value, expecting "
            f"'C' or 'F', but got '{order}'"
        )
    order = "C" if order is None else order.upper()
    if order == "C":
        shape = reversed(shape)

    if not dpnp.can_cast(indices, dpnp.int64, "same_kind"):
        raise TypeError(
            "Iterator operand 0 dtype could not be cast from dtype("
            f"{indices.dtype}) to dtype({dpnp.int64}) according to the rule "
            "'same_kind'"
        )

    if (indices < 0).any():
        raise ValueError("invalid entry in index array")

    unraveled_coords = []
    for dim in shape:
        unraveled_coords.append(indices % dim)
        indices = indices // dim

    if (indices > 0).any():
        raise ValueError("invalid entry in index array")

    if order == "C":
        unraveled_coords = reversed(unraveled_coords)
    return tuple(unraveled_coords)
