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
import numpy

from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *
import dpnp


__all__ = [
    "copyto",
    "moveaxis",
    "ravel",
    "repeat",
    "rollaxis",
    "swapaxes",
    "transpose"
]


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

    is_input_dparray1 = isinstance(dst, dparray)
    is_input_dparray2 = isinstance(src, dparray)

    if not use_origin_backend(dst) and is_input_dparray1 and is_input_dparray2:
        if casting != 'same_kind':
            checker_throw_value_error("copyto", "casting", type(casting), 'same_kind')

        if not isinstance(where, bool) and not isinstance(where, dparray):
            checker_throw_type_error('copyto', type(where))

        if where is not True:
            checker_throw_value_error("copyto", "where", where, True)

        if dst.shape != src.shape:
            raise NotImplemented

        result = dpnp_copyto(dst, src, where=where)

        return result

    input1 = dpnp.asnumpy(dst) if is_input_dparray1 else dst
    input2 = dpnp.asnumpy(src) if is_input_dparray2 else src

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.copyto(dst, src, where=where)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


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

    if (use_origin_backend(x1)):
        return numpy.swapaxes(x1, source, destination)

    if (not isinstance(x1, dparray)):
        return numpy.swapaxes(x1, source, destination)

    if not isinstance(source, collections.abc.Sequence):  # assume scalar
        source = (source,)

    if not isinstance(destination, collections.abc.Sequence):  # assume scalar
        destination = (destination,)

    source_norm = normalize_axis(source, x1.ndim)
    destination_norm = normalize_axis(destination, x1.ndim)

    if len(source_norm) != len(destination_norm):
        checker_throw_axis_error(
            "swapaxes",
            "source_norm.size() != destination_norm.size()",
            source_norm,
            destination_norm)

    # 'do nothing' pattern for transpose() with no elements in 'source'
    input_permute = []
    for i in range(x1.ndim):
        if i not in source_norm:
            input_permute.append(i)

    # insert moving axes into proper positions
    for destination_id, source_id in sorted(zip(destination_norm, source_norm)):
        # if destination_id in input_permute:
        # pytest tests/third_party/cupy/manipulation_tests/test_transpose.py::TestTranspose::test_moveaxis_invalid5_3
        #checker_throw_value_error("swapaxes", "source_id exists", source_id, input_permute)
        input_permute.insert(destination_id, source_id)

    return transpose(x1, axes=input_permute)


def ravel(a, order='C'):
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

    if not use_origin_backend(a) and isinstance(a, dparray):
        return a.ravel(order=order)

    result = numpy.rollaxis(dp2nd_array(a), order=order)

    return nd2dp_array(result)


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

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1)
        and is_x1_dparray
        and (axis is None or axis == 0)
        and (x1.ndim < 2)
        ):

        repeat_val = repeats
        if isinstance(repeats, (tuple, list)):
            if (len(repeats) > 1):
                checker_throw_value_error("repeat", "len(repeats)", len(repeats), 1)

            repeat_val = repeats[0]

        return dpnp_repeat(x1, repeat_val, axis)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.repeat(input1, repeats, axis=axis)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def rollaxis(a, axis, start=0):
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

    if not use_origin_backend(a):
        if not isinstance(a, dparray):
            pass
        elif not isinstance(axis, int):
            pass
        elif start < -a.ndim or start > a.ndim:
            pass
        else:
            start_norm = start + a.ndim if start < 0 else start
            destination = start_norm - 1 if start_norm > axis else start_norm

            return dpnp.moveaxis(a, axis, destination)

    result = numpy.rollaxis(dp2nd_array(a), axis, start)

    return nd2dp_array(result)


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

    if (use_origin_backend(x1)):
        return numpy.swapaxes(x1, axis1, axis2)

    if (not isinstance(x1, dparray)):
        return numpy.swapaxes(x1, axis1, axis2)

    if not (axis1 < x1.ndim):
        checker_throw_value_error("swapaxes", "axis1", axis1, x1.ndim - 1)

    if not (axis2 < x1.ndim):
        checker_throw_value_error("swapaxes", "axis2", axis2, x1.ndim - 1)

    # 'do nothing' pattern for transpose()
    input_permute = [i for i in range(x1.ndim)]
    # swap axes
    input_permute[axis1], input_permute[axis2] = input_permute[axis2], input_permute[axis1]

    return transpose(x1, axes=input_permute)


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

    if (use_origin_backend(x1)):
        return numpy.transpose(x1, axes=axes)

    if (not isinstance(x1, dparray)):
        return numpy.transpose(x1, axes=axes)

    if (axes is not None):
        if (not any(axes)):
            """
            pytest tests/third_party/cupy/manipulation_tests/test_transpose.py
            """
            axes = None

    return dpnp_transpose(x1, axes=axes)
