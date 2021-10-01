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
Interface of the Array manipulation routines part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import collections.abc

from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *
from dpnp.dpnp_iface_arraycreation import array

import dpnp
import numpy


__all__ = [
    "asfarray",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "concatenate",
    "copyto",
    "expand_dims",
    "hstack",
    "moveaxis",
    "ravel",
    "repeat",
    "reshape",
    "rollaxis",
    "squeeze",
    "stack",
    "swapaxes",
    "transpose",
    "unique",
    "vstack"
]


def asfarray(x1, dtype=numpy.float64):
    """
    Return an array converted to a float type.

    For full documentation refer to :obj:`numpy.asfarray`.

    Notes
    -----
    This function works exactly the same as :obj:`dpnp.array`.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        # behavior of original function: int types replaced with float64
        if numpy.issubdtype(dtype, numpy.integer):
            dtype = numpy.float64

        # if type is the same then same object should be returned
        if x1_desc.dtype == dtype:
            return x1

        return array(x1, dtype=dtype)

    return call_origin(numpy.asfarray, x1, dtype)


def atleast_1d(*arys):
    """
    Convert inputs to arrays with at least one dimension.
    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    For full documentation refer to :obj:`numpy.atleast_1d`.

    Limitations
    -----------
    Input arrays is supported as :obj:`dpnp.ndarray`.

    """

    return call_origin(numpy.atleast_1d, *arys)


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
        ary_desc = dpnp.get_dpnp_descriptor(ary)
        if ary_desc:
            arys_desc.append(ary_desc)
        else:
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
        ary_desc = dpnp.get_dpnp_descriptor(ary)
        if ary_desc:
            arys_desc.append(ary_desc)
        else:
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


def concatenate(arrs, axis=0, out=None, dtype=None, casting="same_kind"):
    """
    Join a sequence of arrays along an existing axis.

    For full documentation refer to :obj:`numpy.concatenate`.

    Examples
    --------
    >>> import dpnp
    >>> a = dpnp.array([[1, 2], [3, 4]])
    >>> b = dpnp.array([[5, 6]])
    >>> res = dpnp.concatenate((a, b), axis=0)
    >>> print(res)
    [[1 2]
     [3 4]
     [5 6]]
    >>> res = dpnp.concatenate((a, b.T), axis=1)
    >>> print(res)
    [[1 2 5]
     [3 4 6]]
    >>> res = dpnp.concatenate((a, b), axis=None)
    >>> print(res)
    [1 2 3 4 5 6]

    """
    return call_origin(numpy.concatenate, arrs, axis=axis, out=out, dtype=dtype, casting=casting)


def copyto(dst, src, casting='same_kind', where=True):
    """
    Copies values from one array to another, broadcasting as necessary.

    For full documentation refer to :obj:`numpy.copyto`.

    Limitations
    -----------
    Input arrays are supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Parameter ``casting`` is supported only with default value ``"same_kind"``.
    Parameter ``where`` is supported only with default value ``True``.
    Shapes of input arrays are supported to be equal.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    """

    dst_desc = dpnp.get_dpnp_descriptor(dst, copy_when_strides=False)
    src_desc = dpnp.get_dpnp_descriptor(src)
    if dst_desc and src_desc:
        if casting != 'same_kind':
            pass
        elif (dst_desc.dtype == dpnp.bool and  # due to 'same_kind' casting
              src_desc.dtype in [dpnp.int32, dpnp.int64, dpnp.float32, dpnp.float64, dpnp.complex128]):
            pass
        elif (dst_desc.dtype in [dpnp.int32, dpnp.int64] and  # due to 'same_kind' casting
              src_desc.dtype in [dpnp.float32, dpnp.float64, dpnp.complex128]):
            pass
        elif dst_desc.dtype in [dpnp.float32, dpnp.float64] and src_desc.dtype == dpnp.complex128:  # due to 'same_kind' casting
            pass
        elif where is not True:
            pass
        elif dst_desc.shape != src_desc.shape:
            pass
        elif dst_desc.strides != src_desc.strides:
            pass
        else:
            return dpnp_copyto(dst_desc, src_desc, where=where)

    return call_origin(numpy.copyto, dst, src, casting, where, dpnp_inplace=True)


def expand_dims(x1, axis):
    """
    Expand the shape of an array.

    Insert a new axis that will appear at the `axis` position in the expanded
    array shape.

    For full documentation refer to :obj:`numpy.expand_dims`.

    See Also
    --------
    :obj:`dpnp.squeeze` : The inverse operation, removing singleton dimensions
    :obj:`dpnp.reshape` : Insert, remove, and combine dimensions, and resize existing ones
    :obj:`dpnp.indexing`, :obj:`dpnp.atleast_1d`, :obj:`dpnp.atleast_2d`, :obj:`dpnp.atleast_3d`

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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_expand_dims(x1_desc, axis).get_pyobj()

    return call_origin(numpy.expand_dims, x1, axis)


def hstack(tup):
    """
    Stack arrays in sequence horizontally (column wise).

    For full documentation refer to :obj:`numpy.hstack`.

    """

    # TODO:
    # `call_origin` cannot convert sequence of array to sequence of
    # nparrays
    tup_new = []
    for tp in tup:
        tpx = dpnp.asnumpy(tp) if not isinstance(tp, numpy.ndarray) else tp
        tup_new.append(tpx)

    return call_origin(numpy.hstack, tup_new)


def moveaxis(x1, source, destination):
    """
    Move axes of an array to new positions. Other axes remain in their original order.

    For full documentation refer to :obj:`numpy.moveaxis`.

    Limitations
    -----------
    Input array ``x1`` is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Sizes of normalized input arrays are supported to be equal.
    Input array data types are limited by supported DPNP :ref:`Data types`.

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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        source_norm = normalize_axis(source, x1_desc.ndim)
        destination_norm = normalize_axis(destination, x1_desc.ndim)

        if len(source_norm) != len(destination_norm):
            pass
        else:
            # 'do nothing' pattern for transpose() with no elements in 'source'
            input_permute = []
            for i in range(x1_desc.ndim):
                if i not in source_norm:
                    input_permute.append(i)

            # insert moving axes into proper positions
            for destination_id, source_id in sorted(zip(destination_norm, source_norm)):
                # if destination_id in input_permute:
                # pytest tests/third_party/cupy/manipulation_tests/test_transpose.py::TestTranspose::test_moveaxis_invalid5_3
                # checker_throw_value_error("swapaxes", "source_id exists", source_id, input_permute)
                input_permute.insert(destination_id, source_id)

            return transpose(x1_desc.get_pyobj(), axes=input_permute)

    return call_origin(numpy.moveaxis, x1, source, destination)


def ravel(x1, order='C'):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_flatten(x1_desc).get_pyobj()

    return call_origin(numpy.ravel, x1, order=order)


def repeat(x1, repeats, axis=None):
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axis is not None and axis != 0:
            pass
        elif x1_desc.ndim >= 2:
            pass
        elif not dpnp.isscalar(repeats) and len(repeats) > 1:
            pass
        else:
            repeat_val = repeats if dpnp.isscalar(repeats) else repeats[0]
            return dpnp_repeat(x1_desc, repeat_val, axis).get_pyobj()

    return call_origin(numpy.repeat, x1, repeats, axis)


def reshape(x1, newshape, order='C'):
    """
    Gives a new shape to an array without changing its data.

    For full documentation refer to :obj:`numpy.reshape`.

    Limitations
    -----------
    Only 'C' order is supported.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if order != 'C':
            pass
        else:
            return dpnp_reshape(x1_desc, newshape, order).get_pyobj()

    return call_origin(numpy.reshape, x1, newshape, order)


def rollaxis(x1, axis, start=0):
    """
    Roll the specified axis backwards, until it lies in a given position.

    For full documentation refer to :obj:`numpy.rollaxis`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameter ``axis`` is supported as integer only.
    Parameter ``start`` is limited by ``-a.ndim <= start <= a.ndim``.
    Otherwise the function will be executed sequentially on CPU.
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if not isinstance(axis, int):
            pass
        elif start < -x1_desc.ndim or start > x1_desc.ndim:
            pass
        else:
            start_norm = start + x1_desc.ndim if start < 0 else start
            destination = start_norm - 1 if start_norm > axis else start_norm

            return dpnp.moveaxis(x1_desc.get_pyobj(), axis, destination)

    return call_origin(numpy.rollaxis, x1, axis, start)


def squeeze(x1, axis=None):
    """
    Remove single-dimensional entries from the shape of an array.

    For full documentation refer to :obj:`numpy.squeeze`.

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
    ValueError: cannot select an axis to squeeze out which has size not equal to one
    >>> np.squeeze(x, axis=2).shape
    (1, 3)
    >>> x = np.array([[1234]])
    >>> x.shape
    (1, 1)
    >>> np.squeeze(x)
    array(1234)  # 0d array
    >>> np.squeeze(x).shape
    ()
    >>> np.squeeze(x)[()]
    1234

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        return dpnp_squeeze(x1_desc, axis).get_pyobj()

    return call_origin(numpy.squeeze, x1, axis)


def stack(arrays, axis=0, out=None):
    """
    Join a sequence of arrays along a new axis.

    For full documentation refer to :obj:`numpy.stack`.

    """

    return call_origin(numpy.stack, arrays, axis, out)


def swapaxes(x1, axis1, axis2):
    """
    Interchange two axes of an array.

    For full documentation refer to :obj:`numpy.swapaxes`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Parameter ``axis1`` is limited by ``axis1 < x1.ndim``.
    Parameter ``axis2`` is limited by ``axis2 < x1.ndim``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[1, 2, 3]])
    >>> out = np.swapaxes(x, 0, 1)
    >>> out.shape
    (3, 1)
    >>> [i for i in out]
    [1, 2, 3]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axis1 >= x1_desc.ndim:
            pass
        elif axis2 >= x1_desc.ndim:
            pass
        else:
            # 'do nothing' pattern for transpose()
            input_permute = [i for i in range(x1.ndim)]
            # swap axes
            input_permute[axis1], input_permute[axis2] = input_permute[axis2], input_permute[axis1]

            return transpose(x1_desc.get_pyobj(), axes=input_permute)

    return call_origin(numpy.swapaxes, x1, axis1, axis2)


def transpose(x1, axes=None):
    """
    Reverse or permute the axes of an array; returns the modified array.

    For full documentation refer to :obj:`numpy.transpose`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Value of the parameter ``axes`` likely to be replaced with ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.moveaxis` : Move array axes to new positions.
    :obj:`dpnp.argsort` : Returns the indices that would sort an array.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.arange(4).reshape((2,2))
    >>> x.shape
    (2, 2)
    >>> [i for i in x]
    [0, 1, 2, 3]
    >>> out = np.transpose(x)
    >>> out.shape
    (2, 2)
    >>> [i for i in out]
    [0, 2, 1, 3]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if axes is not None:
            if not any(axes):
                """
                pytest tests/third_party/cupy/manipulation_tests/test_transpose.py
                """
                axes = None

        result = dpnp_transpose(x1_desc, axes).get_pyobj()

        return result

    return call_origin(numpy.transpose, x1, axes=axes)


def unique(x1, **kwargs):
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

    return call_origin(numpy.unique, x1, **kwargs)


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
