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
Interface of the Linear Algebra part of the DPNP

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

import dpnp
from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *

__all__ = [
    "dot",
    "einsum",
    "einsum_path",
    "inner",
    "kron",
    "matmul",
    "outer",
    "tensordot",
    "vdot",
]


def dot(x1, x2, out=None, **kwargs):
    """
    Dot product of `x1` and `x2`.

    For full documentation refer to :obj:`numpy.dot`.

    Returns
    -------
    y : dpnp.ndarray
        Returns the dot product of `x1` and `x2`.
        If `out` is given, then it is returned.

    Limitations
    -----------
    Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
    Keyword argument ``kwargs`` is currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.tensordot` : Sum products over arbitrary axes.
    :obj:`dpnp.vdot` : Complex-conjugating dot product.

    Examples
    --------
    >>> import dpnp as dp
    >>> a = dp.array([1, 2, 3])
    >>> b = dp.array([1, 2, 3])
    >>> dp.dot(a, b)
    14

    """

    if kwargs:
        pass
    elif dpnp.isscalar(x1) and dpnp.isscalar(x2):
        # at least either x1 or x2 has to be an array
        pass
    else:
        # get USM type and queue to copy scalar from the host memory into a USM allocation
        usm_type, queue = (
            get_usm_allocations([x1, x2])
            if dpnp.isscalar(x1) or dpnp.isscalar(x2)
            else (None, None)
        )

        x1_desc = dpnp.get_dpnp_descriptor(
            x1,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        x2_desc = dpnp.get_dpnp_descriptor(
            x2,
            copy_when_strides=False,
            copy_when_nondefault_queue=False,
            alloc_usm_type=usm_type,
            alloc_queue=queue,
        )
        if x1_desc and x2_desc:
            if out is not None:
                if not isinstance(out, (dpnp.ndarray, dpt.usm_ndarray)):
                    raise TypeError(
                        "return array must be of supported array type"
                    )
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out,
                        copy_when_strides=False,
                        copy_when_nondefault_queue=False,
                    )
                    or None
                )
            else:
                out_desc = None
            return dpnp_dot(x1_desc, x2_desc, out=out_desc).get_pyobj()

    return call_origin(numpy.dot, x1, x2, out=out, **kwargs)


def einsum(*args, **kwargs):
    """
    Evaluates the Einstein summation convention on the operands.

    For full documentation refer to :obj:`numpy.einsum`.

    Limitations
    -----------
    Function is executed sequentially on CPU.

    See Also
    -------
    :obj:`dpnp.einsum_path` : Evaluates the lowest cost contraction order for an einsum expression.
    :obj:`dpnp.dot` : Returns the dot product of two arrays.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.
    :obj:`dpnp.outer` : Returns the outer product of two arrays.

    """

    return call_origin(numpy.einsum, *args, **kwargs)


def einsum_path(*args, **kwargs):
    """
    einsum_path(subscripts, *operands, optimize='greedy')

    Evaluates the lowest cost contraction order for an einsum expression
    by considering the creation of intermediate arrays.

    For full documentation refer to :obj:`numpy.einsum_path`.

    Limitations
    -----------
    Function is executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention on the operands.
    :obj:`dpnp.dot` : Returns the dot product of two arrays.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.
    :obj:`dpnp.outer` : Returns the outer product of two arrays.

    """

    return call_origin(numpy.einsum_path, *args, **kwargs)


def inner(x1, x2, **kwargs):
    """
    Returns the inner product of two arrays.

    For full documentation refer to :obj:`numpy.inner`.

    Limitations
    -----------
        Parameters ``x1`` and ``x2`` are supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.

    See Also
    --------
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention on the operands.
    :obj:`dpnp.dot` : Returns the dot product of two arrays.
    :obj:`dpnp.tensordot` : Compute tensor dot product along specified axes.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1,2,3])
    >>> b = np.array([0, 1, 0])
    >>> result = np.inner(a, b)
    >>> [x for x in result]
    [2]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    x2_desc = dpnp.get_dpnp_descriptor(x2, copy_when_nondefault_queue=False)
    if x1_desc and x2_desc and not kwargs:
        return dpnp_inner(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.inner, x1, x2, **kwargs)


def kron(x1, x2):
    """
    Returns the kronecker product of two arrays.

    For full documentation refer to :obj:`numpy.kron`.

    .. seealso:: :obj:`dpnp.outer` returns the outer product of two arrays.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    x2_desc = dpnp.get_dpnp_descriptor(x2, copy_when_nondefault_queue=False)
    if x1_desc and x2_desc:
        return dpnp_kron(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.kron, x1, x2)


def matmul(x1, x2, out=None, **kwargs):
    """
    Matrix product of two arrays.

    For full documentation refer to :obj:`numpy.matmul`.

    Limitations
    -----------
    Input arrays are supported as :obj:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Parameter ``out`` is supported as :obj:`dpnp.ndarray` and as default value ``None``.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.vdot` : Complex-conjugating dot product.
    :obj:`dpnp.tensordot` : Sum products over arbitrary axes.
    :obj:`dpnp.einsum` : Einstein summation convention.
    :obj:`dpnp.dot` : Alternative matrix product with
                      different broadcasting rules.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.ones([9, 5, 7, 4])
    >>> c = np.ones([9, 5, 4, 3])
    >>> np.matmul(a, c).shape
    (9, 5, 7, 3)
    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.matmul(a, b)
    array([[4, 1],
           [2, 2]])

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    x2_desc = dpnp.get_dpnp_descriptor(x2, copy_when_nondefault_queue=False)
    if x1_desc and x2_desc and not kwargs:
        if x1_desc.ndim != 2 or x2_desc.ndim != 2:
            pass
        elif not x1_desc.ndim:
            pass
        elif not x2_desc.ndim:
            pass
        elif not x1_desc.size:
            pass
        elif not x2_desc.size:
            pass
        else:
            if 0:
                """
                Cost model checks
                """

                array1_size = x1_desc.size
                array2_size = x2_desc.size
                cost_size = 4096  # 2D array shape(64, 64)

                if (x1_desc.dtype == dpnp.float64) or (
                    x1_desc.dtype == dpnp.float32
                ):
                    """
                    Floating point types are handled via original math library better than SYCL math library
                    """
                    cost_size = 262144  # 2D array shape(512, 512)

                if (array1_size > cost_size) and (array2_size > cost_size):
                    return dpnp_matmul(x1_desc, x2_desc, out)
            else:
                out_desc = (
                    dpnp.get_dpnp_descriptor(
                        out, copy_when_nondefault_queue=False
                    )
                    if out is not None
                    else None
                )
                return dpnp_matmul(x1_desc, x2_desc, out_desc).get_pyobj()

    return call_origin(numpy.matmul, x1, x2, out=out, **kwargs)


def outer(x1, x2, out=None):
    """
    Returns the outer product of two arrays.

    For full documentation refer to :obj:`numpy.outer`.

    Limitations
    -----------
        Parameters `x1` and `x2` are supported as either scalar, :class:`dpnp.ndarray`
        or :class:`dpctl.tensor.usm_ndarray`, but both `x1` and `x2` can not be scalars at the same time.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention on the operands.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 1, 1])
    >>> b = np.array([1, 2, 3])
    >>> result = np.outer(a, b)
    >>> [x for x in result]
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])

    """

    x1_is_scalar = dpnp.isscalar(x1)
    x2_is_scalar = dpnp.isscalar(x2)

    if x1_is_scalar and x2_is_scalar:
        pass
    elif not dpnp.is_supported_array_or_scalar(x1):
        pass
    elif not dpnp.is_supported_array_or_scalar(x2):
        pass
    else:
        x1_in = (
            x1
            if x1_is_scalar
            else (x1.reshape(-1) if x1.ndim > 1 else x1)[:, None]
        )
        x2_in = (
            x2
            if x2_is_scalar
            else (x2.reshape(-1) if x2.ndim > 1 else x2)[None, :]
        )
        return dpnp.multiply(x1_in, x2_in, out=out)

    return call_origin(numpy.outer, x1, x2, out=out)


def tensordot(x1, x2, axes=2):
    """
    Compute tensor dot product along specified axes.

    For full documentation refer to :obj:`numpy.tensordot`.

    Limitations
    -----------
        Parameters ``x1`` and ``x2`` are supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
        Parameter ``axes`` is supported only with value ``1``.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.dot` : Returns the dot product.
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention on the operands.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b = np.array([1, 2, 3])
    >>> result = np.tensordot(a, b, 1)
    >>> [x for x in result]
    [14, 32, 50]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    x2_desc = dpnp.get_dpnp_descriptor(x2, copy_when_nondefault_queue=False)
    if x1_desc and x2_desc and (axes == 1):
        return dpnp_tensordot_not_implemented(x1_desc, x2_desc)  # dpnp_matmul

    return call_origin(numpy.tensordot, x1, x2, axes)


def vdot(*args, **kwargs):
    """
    Return the dot product of two vectors.

    For full documentation refer to :obj:`numpy.vdot`.

    See Also
    --------
    :obj:`dpnp.dot` : Returns the dot product.

    Notes
    -----
    This function works the same as :obj:`dpnp.dot`.

    """
    return dpnp.dot(*args, **kwargs)
