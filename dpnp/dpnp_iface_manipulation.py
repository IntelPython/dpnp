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


import dpctl.tensor as dpt
import numpy
from numpy.core.numeric import normalize_axis_index

import dpnp

from .dpnp_array import dpnp_array

# pylint: disable=no-name-in-module
from .dpnp_utils import (
    call_origin,
)

__all__ = [
    "asfarray",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_arrays",
    "broadcast_to",
    "can_cast",
    "column_stack",
    "concatenate",
    "copyto",
    "dstack",
    "expand_dims",
    "flip",
    "fliplr",
    "flipud",
    "hstack",
    "moveaxis",
    "ravel",
    "repeat",
    "reshape",
    "result_type",
    "roll",
    "rollaxis",
    "row_stack",
    "shape",
    "squeeze",
    "stack",
    "swapaxes",
    "tile",
    "transpose",
    "unique",
    "vstack",
]


def _check_stack_arrays(arrays):
    """Validate a sequence type of arrays to stack."""

    if not hasattr(arrays, "__getitem__"):
        raise TypeError(
            'arrays to stack must be passed as a "sequence" type '
            "such as list or tuple."
        )


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
        A SYCL queue to use for output array allocation and copying.

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
    for ary in arys:
        if not dpnp.is_supported_array_type(ary):
            raise TypeError(
                "Each input array must be any of supported type, "
                f"but got {type(ary)}"
            )
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
    for ary in arys:
        if not dpnp.is_supported_array_type(ary):
            raise TypeError(
                "Each input array must be any of supported type, "
                f"but got {type(ary)}"
            )
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
    for ary in arys:
        if not dpnp.is_supported_array_type(ary):
            raise TypeError(
                "Each input array must be any of supported type, "
                f"but got {type(ary)}"
            )
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

    For full documentation refer to :obj:`numpy.concatenate`.

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
        The arrays must have the same shape, except in the dimension
        corresponding to axis (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined. If axis is ``None``,
        arrays are flattened before use. Default is 0.
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

    usm_array = dpnp.get_usm_ndarray(a)
    return dpnp_array._create_from_usm_ndarray(
        dpt.expand_dims(usm_array, axis=axis)
    )


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
    repeat : int or array of int
        The number of repetitions for each element. `repeats` is broadcasted to
        fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values. By default, use the flattened
        input array, and return a flat output array.

    Returns
    -------
    out : dpnp.ndarray
        Output array which has the same shape as `a`, except along the given
        axis.

    See Also
    --------
    :obj:`dpnp.tile` : Construct an array by repeating A the number of times
                       given by reps.

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

    rep = repeats
    if isinstance(repeats, dpnp_array):
        rep = dpnp.get_usm_ndarray(repeats)
    if axis is None and a.ndim > 1:
        usm_arr = dpnp.get_usm_ndarray(a.flatten())
    else:
        usm_arr = dpnp.get_usm_ndarray(a)
    usm_arr = dpt.repeat(usm_arr, rep, axis=axis)
    return dpnp_array._create_from_usm_ndarray(usm_arr)


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
        if possible and copy otherwise. Default: None.

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
    >>> import dpnp as dp
    >>> a = dp.array([[1, 2, 3], [4, 5, 6]])
    >>> dp.reshape(a, 6)
    array([1, 2, 3, 4, 5, 6])
    >>> dp.reshape(a, 6, order='F')
    array([1, 4, 2, 5, 3, 6])

    >>> dp.reshape(a, (3, -1))       # the unspecified value is inferred to be 2
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

    usm_arr = dpnp.get_usm_ndarray(a)
    usm_arr = dpt.reshape(usm_arr, shape=newshape, order=order, copy=copy)
    return dpnp_array._create_from_usm_ndarray(usm_arr)


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
    >>> import dpnp as dp
    >>> a = dp.arange(3, dtype=dp.int64)
    >>> b = dp.arange(7, dtype=dp.int32)
    >>> dp.result_type(a, b)
    dtype('int64')

    >>> dp.result_type(dp.int64, dp.complex128)
    dtype('complex128')

    >>> dp.result_type(dp.ones(10, dtype=dp.float32), dp.float64)
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
    usm_array = dpnp.get_usm_ndarray(x)
    return dpnp_array._create_from_usm_ndarray(
        dpt.roll(usm_array, shift=shift, axis=axis)
    )


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
    >>> import dpnp as dp
    >>> dp.shape(dp.eye(3))
    (3, 3)
    >>> dp.shape([[1, 3]])
    (1, 2)
    >>> dp.shape([0])
    (1,)
    >>> dp.shape(0)
    ()

    """

    if dpnp.is_supported_array_type(a):
        return a.shape
    return numpy.shape(a)


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

    usm_array = dpnp.get_usm_ndarray(a)
    return dpnp_array._create_from_usm_ndarray(
        dpt.squeeze(usm_array, axis=axis)
    )


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

    usm_array = dpnp.get_usm_ndarray(a)
    return dpnp_array._create_from_usm_ndarray(
        dpt.swapaxes(usm_array, axis1=axis1, axis2=axis2)
    )


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

    usm_array = dpnp.get_usm_ndarray(A)
    return dpnp_array._create_from_usm_ndarray(dpt.tile(usm_array, reps))


def transpose(a, axes=None):
    """
    Returns an array with axes transposed.

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


def unique(ar, **kwargs):
    """
    Find the unique elements of an array.

    For full documentation refer to :obj:`numpy.unique`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([1, 1, 2, 2, 3, 3])
    >>> res = np.unique(x)
    >>> print(res)
    [1, 2, 3]

    """

    return call_origin(numpy.unique, ar, **kwargs)


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
