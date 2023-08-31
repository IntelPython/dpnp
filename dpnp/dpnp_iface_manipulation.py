# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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
from dpnp.dpnp_algo import *
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_iface_arraycreation import array
from dpnp.dpnp_utils import *

__all__ = [
    "asfarray",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_to",
    "concatenate",
    "copyto",
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
    "shape",
    "squeeze",
    "stack",
    "swapaxes",
    "transpose",
    "unique",
    "vstack",
]


def asfarray(a, dtype=None):
    """
    Return an array converted to a float type.

    For full documentation refer to :obj:`numpy.asfarray`.

    Notes
    -----
    This function works exactly the same as :obj:`dpnp.array`.
    If dtype is `None`, `bool` or one of the `int` dtypes, it is replaced with
    the default floating type in DPNP depending on device capabilities.

    """

    a_desc = dpnp.get_dpnp_descriptor(a, copy_when_nondefault_queue=False)
    if a_desc:
        if dtype is None or not numpy.issubdtype(dtype, dpnp.inexact):
            dtype = dpnp.default_float_type(sycl_queue=a.sycl_queue)

        # if type is the same then same object should be returned
        if a_desc.dtype == dtype:
            return a

        return array(a, dtype=dtype)

    return call_origin(numpy.asfarray, a, dtype)


def atleast_1d(*arys):
    """
    Convert inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    For full documentation refer to :obj:`numpy.atleast_1d`.

    Parameters
    ----------
    arys : {dpnp_array, usm_ndarray}
        One or more input arrays.

    Returns
    -------
    out : dpnp.ndarray
        An array, or list of arrays, each with ``a.ndim >= 1``.
        Copies are made only if necessary.

    See Also
    --------
    atleast_2d, atleast_3d

    Examples
    --------
    >>> import dpnp as np
    >>> np.atleast_1d(1.0)
    array([1.])

    >>> x = np.arange(9.0).reshape(3,3)
    >>> np.atleast_1d(x)
    array([[0., 1., 2.],
           [3., 4., 5.],
           [6., 7., 8.]])
    >>> np.atleast_1d(x) is x
    True

    >>> np.atleast_1d(1, [3, 4])
    [array([1]), array([3, 4])]

    """

    res = []
    for ary in arys:
        ary = dpnp.asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1)
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def atleast_2d(*arys):
    """
    View inputs as arrays with at least two dimensions.

    For full documentation refer to :obj:`numpy.atleast_2d`.

    Limitations
    -----------
    Input arrays is supported as :obj:`dpnp.ndarray`.
    """

    all_is_array = True
    arys_desc = []
    for ary in arys:
        if not dpnp.isscalar(ary):
            ary_desc = dpnp.get_dpnp_descriptor(
                ary, copy_when_nondefault_queue=False
            )
            if ary_desc:
                arys_desc.append(ary_desc)
                continue
        all_is_array = False
        break

    if not use_origin_backend(arys[0]) and all_is_array:
        result = []
        for ary_desc in arys_desc:
            res = dpnp_atleast_2d(ary_desc).get_pyobj()
            result.append(res)

        if len(result) == 1:
            return result[0]
        else:
            return result

    return call_origin(numpy.atleast_2d, *arys)


def atleast_3d(*arys):
    """
    View inputs as arrays with at least three dimensions.

    For full documentation refer to :obj:`numpy.atleast_3d`.

    Limitations
    -----------
    Input arrays is supported as :obj:`dpnp.ndarray`.
    """

    all_is_array = True
    arys_desc = []
    for ary in arys:
        if not dpnp.isscalar(ary):
            ary_desc = dpnp.get_dpnp_descriptor(
                ary, copy_when_nondefault_queue=False
            )
            if ary_desc:
                arys_desc.append(ary_desc)
                continue
        all_is_array = False
        break

    if not use_origin_backend(arys[0]) and all_is_array:
        result = []
        for ary_desc in arys_desc:
            res = dpnp_atleast_3d(ary_desc).get_pyobj()
            result.append(res)

        if len(result) == 1:
            return result[0]
        else:
            return result

    return call_origin(numpy.atleast_3d, *arys)


def broadcast_to(array, /, shape, subok=False):
    """
    Broadcast an array to a new shape.

    For full documentation refer to :obj:`numpy.broadcast_to`.

    Returns
    -------
    y : dpnp.ndarray
        An array having a specified shape. Must have the same data type as `array`.

    Limitations
    -----------
    Parameter `array` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Parameter `subok` is supported with default value.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types of `array` is limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as dp
    >>> x = dp.array([1, 2, 3])
    >>> dp.broadcast_to(x, (3, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])

    """

    if subok is not False:
        pass
    elif dpnp.is_supported_array_type(array):
        dpt_array = dpnp.get_usm_ndarray(array)
        new_array = dpt.broadcast_to(dpt_array, shape)
        return dpnp_array._create_from_usm_ndarray(new_array)

    return call_origin(numpy.broadcast_to, array, shape=shape, subok=subok)


def concatenate(
    arrays, /, *, axis=0, out=None, dtype=None, casting="same_kind", **kwargs
):
    """
    Join a sequence of arrays along an existing axis.

    For full documentation refer to :obj:`numpy.concatenate`.

    Returns
    -------
    out : dpnp.ndarray
        The concatenated array.

    Limitations
    -----------
    Each array in `arrays` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`. Otherwise ``TypeError`` exception
    will be raised.
    Parameters `out` and `dtype are supported with default value.
    Keyword argument ``kwargs`` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.array_split` : Split an array into multiple sub-arrays of equal or near-equal size.
    :obj:`dpnp.split` : Split array into a list of multiple sub-arrays of equal size.
    :obj:`dpnp.hsplit` : Split array into multiple sub-arrays horizontally (column wise).
    :obj:`dpnp.vsplit` : Split array into multiple sub-arrays vertically (row wise).
    :obj:`dpnp.dsplit` : Split array into multiple sub-arrays along the 3rd axis (depth).
    :obj:`dpnp.stack` : Stack a sequence of arrays along a new axis.
    :obj:`dpnp.block` : Assemble arrays from blocks.
    :obj:`dpnp.hstack` : Stack arrays in sequence horizontally (column wise).
    :obj:`dpnp.vstack` : Stack arrays in sequence vertically (row wise).
    :obj:`dpnp.dstack` : Stack arrays in sequence depth wise (along third dimension).
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

    if kwargs:
        pass
    elif out is not None:
        pass
    elif dtype is not None:
        pass
    elif casting != "same_kind":
        pass
    else:
        usm_arrays = [dpnp.get_usm_ndarray(x) for x in arrays]
        usm_res = dpt.concat(usm_arrays, axis=axis)
        return dpnp_array._create_from_usm_ndarray(usm_res)

    return call_origin(
        numpy.concatenate,
        arrays,
        axis=axis,
        out=out,
        dtype=dtype,
        casting=casting,
        **kwargs,
    )


def copyto(dst, src, casting="same_kind", where=True):
    """
    Copies values from one array to another, broadcasting as necessary.

    Raises a ``TypeError`` if the `casting` rule is violated, and if
    `where` is provided, it selects which elements to copy.

    For full documentation refer to :obj:`numpy.copyto`.

    Limitations
    -----------
    The `dst` parameter is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    The `where` parameter is supported as either :class:`dpnp.ndarray`,
    :class:`dpctl.tensor.usm_ndarray` or scalar.
    Otherwise ``TypeError`` exception will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

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
    elif not dpnp.is_supported_array_type(src):
        src = dpnp.array(src, sycl_queue=dst.sycl_queue)

    if not dpt.can_cast(src.dtype, dst.dtype, casting=casting):
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


def expand_dims(a, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    For full documentation refer to :obj:`numpy.expand_dims`.

    Returns
    -------
    out : dpnp.ndarray
        An array with the number of dimensions increased.
        A view is returned whenever possible.

    Limitations
    -----------
    Parameters `a` is supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Otherwise ``TypeError`` exception will be raised.

    Notes
    -----
    If `a` has rank (i.e, number of dimensions) `N`, a valid `axis` must reside
    in the closed-interval `[-N-1, N]`.
    If provided a negative `axis`, the `axis` position at which to insert a
    singleton dimension is computed as `N + axis + 1`.
    Hence, if provided `-1`, the resolved axis position is `N` (i.e.,
    a singleton dimension must be appended to the input array `a`).
    If provided `-N-1`, the resolved axis position is `0` (i.e., a
    singleton dimension is prepended to the input array `a`).

    See Also
    --------
    :obj:`dpnp.squeeze` : The inverse operation, removing singleton dimensions
    :obj:`dpnp.reshape` : Insert, remove, and combine dimensions, and resize existing ones
    :obj:`dpnp.atleast_1d`, :obj:`dpnp.atleast_2d`, :obj:`dpnp.atleast_3d`

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

    dpt_array = dpnp.get_usm_ndarray(a)
    return dpnp_array._create_from_usm_ndarray(
        dpt.expand_dims(dpt_array, axis=axis)
    )


def flip(m, axis=None):
    """
    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    For full documentation refer to :obj:`numpy.flip`.

    Returns
    -------
    out : dpnp.ndarray
        A view of `m` with the entries of axis reversed.

    Limitations
    -----------
    Parameters `m` is supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Otherwise ``TypeError`` exception will be raised.

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

    Returns
    -------
    out : dpnp.ndarray
        A view of `m` with the columns reversed.

    Limitations
    -----------
    Parameters `m` is supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Otherwise ``TypeError`` exception will be raised.

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

    if not dpnp.is_supported_array_type(m):
        raise TypeError(
            "An array must be any of supported type, but got {}".format(type(m))
        )

    if m.ndim < 2:
        raise ValueError(f"Input must be >= 2-d, but got {m.ndim}")
    return m[:, ::-1]


def flipud(m):
    """
    Reverse the order of elements along axis 0 (up/down).

    For a 2-D array, this flips the entries in each column in the up/down
    direction. Rows are preserved, but appear in a different order than before.

    For full documentation refer to :obj:`numpy.flipud`.

    Returns
    -------
    out : dpnp.ndarray
        A view of `m` with the rows reversed.

    Limitations
    -----------
    Parameters `m` is supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Otherwise ``TypeError`` exception will be raised.

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

    if not dpnp.is_supported_array_type(m):
        raise TypeError(
            "An array must be any of supported type, but got {}".format(type(m))
        )

    if m.ndim < 1:
        raise ValueError(f"Input must be >= 1-d, but got {m.ndim}")
    return m[::-1, ...]


def hstack(tup, *, dtype=None, casting="same_kind"):
    """
    Stack arrays in sequence horizontally (column wise).

    For full documentation refer to :obj:`numpy.hstack`.

    Returns
    -------
    out : dpnp.ndarray
        The stacked array which has one more dimension than the input arrays.

    Limitations
    -----------
    Each array in `tup` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`. Otherwise ``TypeError`` exception
    will be raised.
    Parameters `dtype` and `casting` are supported with default value.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.
    :obj:`dpnp.stack` : Join a sequence of arrays along a new axis.
    :obj:`dpnp.vstack` : Stack arrays in sequence vertically (row wise).
    :obj:`dpnp.block` : Assemble an nd-array from nested lists of blocks.
    :obj:`dpnp.split` : Split array into a list of multiple sub-arrays of equal size.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array((1,2,3))
    >>> b = np.array((4,5,6))
    >>> np.hstack((a,b))
    array([1, 2, 3, 4, 5, 6])

    >>> a = np.array([[1],[2],[3]])
    >>> b = np.array([[4],[5],[6]])
    >>> np.hstack((a,b))
    array([[1, 4],
           [2, 5],
           [3, 6]])

    """

    if not hasattr(tup, "__getitem__"):
        raise TypeError(
            "Arrays to stack must be passed as a sequence type such as list or tuple."
        )
    arrs = dpnp.atleast_1d(*tup)
    if not isinstance(arrs, list):
        arrs = [arrs]
    # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
    if arrs and arrs[0].ndim == 1:
        return dpnp.concatenate(arrs, axis=0, dtype=dtype, casting=casting)
    else:
        return dpnp.concatenate(arrs, axis=1, dtype=dtype, casting=casting)


def moveaxis(a, source, destination):
    """
    Move axes of an array to new positions. Other axes remain in their original order.

    For full documentation refer to :obj:`numpy.moveaxis`.

    Returns
    -------
    out : dpnp.ndarray
        Array with moved axes.
        The returned array will have the same data and
        the same USM allocation type as `a`.

    Limitations
    -----------
    Parameters `a` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`. Otherwise ``TypeError`` exception
    will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Otherwise ``TypeError`` exception will be raised.

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

    dpt_array = dpnp.get_usm_ndarray(a)
    return dpnp_array._create_from_usm_ndarray(
        dpt.moveaxis(dpt_array, source, destination)
    )


def ravel(a, order="C"):
    """
    Return a contiguous flattened array.

    For full documentation refer to :obj:`numpy.ravel`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> out = np.ravel(x)
    >>> [i for i in out]
    [1, 2, 3, 4, 5, 6]

    """

    a_desc = dpnp.get_dpnp_descriptor(a, copy_when_nondefault_queue=False)
    if a_desc:
        return dpnp_flatten(a_desc).get_pyobj()

    return call_origin(numpy.ravel, a, order=order)


def repeat(a, repeats, axis=None):
    """
    Repeat elements of an array.

    For full documentation refer to :obj:`numpy.repeat`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameter ``axis`` is supported with value either ``None`` or ``0``.
    Dimension of input array are supported to be less than ``2``.
    Otherwise the function will be executed sequentially on CPU.
    If ``repeats`` is ``tuple`` or ``list``, should be ``len(repeats) > 1``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    .. seealso:: :obj:`numpy.tile` tile an array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.repeat(3, 4)
    >>> [i for i in x]
    [3, 3, 3, 3]

    """

    a_desc = dpnp.get_dpnp_descriptor(a, copy_when_nondefault_queue=False)
    if a_desc:
        if axis is not None and axis != 0:
            pass
        elif a_desc.ndim >= 2:
            pass
        elif not dpnp.isscalar(repeats) and len(repeats) > 1:
            pass
        else:
            repeat_val = repeats if dpnp.isscalar(repeats) else repeats[0]
            return dpnp_repeat(a_desc, repeat_val, axis).get_pyobj()

    return call_origin(numpy.repeat, a, repeats, axis)


def reshape(a, /, newshape, order="C", copy=None):
    """
    Gives a new shape to an array without changing its data.

    For full documentation refer to :obj:`numpy.reshape`.

    Parameters
    ----------
    a : {dpnp_array, usm_ndarray}
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {'C', 'F'}, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
    copy : bool, optional
        Boolean indicating whether or not to copy the input array.
        If ``True``, the result array will always be a copy of input `a`.
        If ``False``, the result array can never be a copy
        and a ValueError exception will be raised in case the copy is necessary.
        If ``None``, the result array will reuse existing memory buffer of `a`
        if possible and copy otherwise. Default: None.

    Returns
    -------
    y : dpnp.ndarray
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
    arrays_and_dtypes : list of arrays and dtypes
        An arbitrary length sequence of arrays or dtypes.

    Returns
    -------
    out : dtype
        The result type.

    Limitations
    -----------
    An array in the input list is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.

    Examples
    --------
    >>> import dpnp as dp
    >>> dp.result_type(dp.arange(3, dtype=dp.int64), dp.arange(7, dtype=dp.int32))
    dtype('int64')

    >>> dp.result_type(dp.int64, dp.complex128)
    dtype('complex128')

    >>> dp.result_type(dp.ones(10, dtype=dp.float32), dp.float64)
    dtype('float64')

    """

    usm_arrays_and_dtypes = [
        X.dtype if isinstance(X, (dpnp_array, dpt.usm_ndarray)) else X
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

    Returns
    -------
    dpnp.ndarray
        An array with the same data type as `x`
        and whose elements, relative to `x`, are shifted.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`. Otherwise ``TypeError`` exception
    will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.


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
    dpt_array = dpnp.get_usm_ndarray(x)
    return dpnp_array._create_from_usm_ndarray(
        dpt.roll(dpt_array, shift=shift, axis=axis)
    )


def rollaxis(x, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    For full documentation refer to :obj:`numpy.rollaxis`.

    Returns
    -------
    dpnp.ndarray
        An array with the same data type as `x` where the specified axis
        has been repositioned to the desired position.

    Limitations
    -----------
    Parameter `x` is supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`. Otherwise ``TypeError`` exception
    will be raised.
    Input array data types are limited by supported DPNP :ref:`Data types`.

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
    if not (0 <= start < n + 1):
        raise ValueError(msg % ("start", -n, "start", n + 1, start))
    if axis < start:
        start -= 1
    if axis == start:
        return x
    dpt_array = dpnp.get_usm_ndarray(x)
    return dpnp.moveaxis(dpt_array, source=axis, destination=start)


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
    shape : tuple of ints
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
    else:
        return numpy.shape(a)


def squeeze(a, /, axis=None):
    """
    Removes singleton dimensions (axes) from array `a`.

    For full documentation refer to :obj:`numpy.squeeze`.

    Returns
    -------
    out : dpnp.ndarray
        Output array is a view, if possible,
        and a copy otherwise, but with all or a subset of the
        dimensions of length 1 removed. Output has the same data
        type as the input, is allocated on the same device as the
        input and has the same USM allocation type as the input
        array `a`.

    Limitations
    -----------
    Parameters `a` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Otherwise ``TypeError`` exception will be raised.

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
    ValueError: Cannot select an axis to squeeze out which has size not equal to one.
    >>> np.squeeze(x, axis=2).shape
    (1, 3)

    """

    dpt_array = dpnp.get_usm_ndarray(a)
    return dpnp_array._create_from_usm_ndarray(
        dpt.squeeze(dpt_array, axis=axis)
    )


def stack(arrays, /, *, axis=0, out=None, dtype=None, **kwargs):
    """
    Join a sequence of arrays along a new axis.

    For full documentation refer to :obj:`numpy.stack`.

    Returns
    -------
    out : dpnp.ndarray
        The stacked array which has one more dimension than the input arrays.

    Limitations
    -----------
    Each array in `arrays` is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`. Otherwise ``TypeError`` exception
    will be raised.
    Parameters `out` and `dtype` are supported with default value.
    Keyword argument ``kwargs`` is currently unsupported.
    Otherwise the function will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.concatenate` : Join a sequence of arrays along an existing axis.
    :obj:`dpnp.block` : Assemble an nd-array from nested lists of blocks.
    :obj:`dpnp.split` : Split array into a list of multiple sub-arrays of equal size.

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

    if kwargs:
        pass
    elif out is not None:
        pass
    elif dtype is not None:
        pass
    else:
        usm_arrays = [dpnp.get_usm_ndarray(x) for x in arrays]
        usm_res = dpt.stack(usm_arrays, axis=axis)
        return dpnp_array._create_from_usm_ndarray(usm_res)

    return call_origin(
        numpy.stack,
        arrays,
        axis=axis,
        out=out,
        dtype=dtype,
        **kwargs,
    )


def swapaxes(a, axis1, axis2):
    """
    Interchange two axes of an array.

    For full documentation refer to :obj:`numpy.swapaxes`.

    Returns
    -------
    out : dpnp.ndarray
        An array with with swapped axes.
        A view is returned whenever possible.

    Limitations
    -----------
    Parameters `a` is supported either as :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.
    Otherwise ``TypeError`` exception will be raised.

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

    dpt_array = dpnp.get_usm_ndarray(a)
    return dpnp_array._create_from_usm_ndarray(
        dpt.swapaxes(dpt_array, axis1=axis1, axis2=axis2)
    )


def transpose(a, axes=None):
    """
    Returns an array with axes transposed.

    For full documentation refer to :obj:`numpy.transpose`.

    Returns
    -------
    y : dpnp.ndarray
        `a` with its axes permuted. A view is returned whenever possible.

    Limitations
    -----------
    Input array is supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.

    See Also
    --------
    :obj:`dpnp.ndarray.transpose` : Equivalent method.
    :obj:`dpnp.moveaxis` : Move array axes to new positions.
    :obj:`dpnp.argsort` : Returns the indices that would sort an array.

    Examples
    --------
    >>> import dpnp as dp
    >>> a = dp.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> dp.transpose(a)
    array([[1, 3],
           [2, 4]])

    >>> a = dp.array([1, 2, 3, 4])
    >>> a
    array([1, 2, 3, 4])
    >>> dp.transpose(a)
    array([1, 2, 3, 4])

    >>> a = dp.ones((1, 2, 3))
    >>> dp.transpose(a, (1, 0, 2)).shape
    (2, 1, 3)

    >>> a = dp.ones((2, 3, 4, 5))
    >>> dp.transpose(a).shape
    (5, 4, 3, 2)

    """

    if isinstance(a, dpnp_array):
        array = a
    elif isinstance(a, dpt.usm_ndarray):
        array = dpnp_array._create_from_usm_ndarray(a.get_array())
    else:
        raise TypeError(
            "An array must be any of supported type, but got {}".format(type(a))
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


def vstack(tup):
    """
    Stack arrays in sequence vertically (row wise).

    For full documentation refer to :obj:`numpy.vstack`.

    """

    # TODO:
    # `call_origin` cannot convert sequence of array to sequence of
    # nparray
    tup_new = []
    for tp in tup:
        tpx = dpnp.asnumpy(tp) if not isinstance(tp, numpy.ndarray) else tp
        tup_new.append(tpx)

    return call_origin(numpy.vstack, tup_new)
