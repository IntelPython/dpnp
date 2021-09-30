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
Interface of the Linear Algebra part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *
import dpnp
import dpnp.config as config

import numpy


__all__ = [
    "dot",
    "einsum",
    "einsum_path",
    "inner",
    "kron",
    "matmul",
    "outer",
    "tensordot",
    "vdot"
]


def dot(x1, x2, **kwargs):
    """
    Returns the dot product of `x1` and `x2`.

    For full documentation refer to :obj:`numpy.dot`.

    Limitations
    -----------
        Parameters ``x1`` and ``x2`` are supported as :obj:`dpnp.ndarray` of the same type.
        Keyword arguments ``kwargs`` are currently unsupported.
        Otherwise the functions will be executed sequentially on CPU.
        Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.tensordot` : Sum products over arbitrary axes.
    :obj:`dpnp.vdot` : Complex-conjugating dot product.

    Examples
    --------
    >>> import dpnp as np
    >>> np.dot(3, 4)
    12
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([1, 2, 3])
    >>> np.dot(a, b)
    14

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_strides=False)
    x2_desc = dpnp.get_dpnp_descriptor(x2, copy_when_strides=False)
    if x1_desc and x2_desc and not kwargs:
        # TODO: remove fallback with scalars when muliply backend func will support strides
        if(x1_desc.ndim == 0 and x2_desc.strides is not None
                or x2_desc.ndim == 0 and x1_desc.strides is not None):
            pass
        else:
            return dpnp_dot(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.dot, x1, x2, **kwargs)


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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)
    if x1_desc and x2_desc and not kwargs:
        return dpnp_inner(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.inner, x1, x2, **kwargs)


def kron(x1, x2):
    """
    Returns the kronecker product of two arrays.

    For full documentation refer to :obj:`numpy.kron`.

    .. seealso:: :obj:`dpnp.outer` returns the outer product of two arrays.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)
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

                if ((x1_desc.dtype == numpy.float64) or (x1_desc.dtype == numpy.float32)):
                    """
                    Floating point types are handled via original math library better than SYCL math library
                    """
                    cost_size = 262144  # 2D array shape(512, 512)

                if (array1_size > cost_size) and (array2_size > cost_size):
                    return dpnp_matmul(x1_desc, x2_desc, out)
            else:
                out_desc = dpnp.get_dpnp_descriptor(out) if out is not None else None
                return dpnp_matmul(x1_desc, x2_desc, out_desc).get_pyobj()

    return call_origin(numpy.matmul, x1, x2, out=out, **kwargs)


def outer(x1, x2, **kwargs):
    """
    Returns the outer product of two arrays.

    For full documentation refer to :obj:`numpy.outer`.

    Limitations
    -----------
        Parameters ``x1`` and ``x2`` are supported as :obj:`dpnp.ndarray`.
        Keyword arguments ``kwargs`` are currently unsupported.
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
    [1, 2, 3, 1, 2, 3, 1, 2, 3]

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)
    if x1_desc and x2_desc and not kwargs:
        return dpnp_outer(x1_desc, x2_desc).get_pyobj()

    return call_origin(numpy.outer, x1, x2, **kwargs)


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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    x2_desc = dpnp.get_dpnp_descriptor(x2)
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
