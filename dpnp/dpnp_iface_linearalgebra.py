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


import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti
import numpy

import dpnp
import dpnp.backend.extensions.blas._blas_impl as bi
from dpnp.dpnp_algo import *
from dpnp.dpnp_array import dpnp_array
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
    Parameters `x1` and `x2` are supported as :obj:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
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
    Parameter `out` is supported as :obj:`dpnp.ndarray` and as default value ``None``.
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

    x1_ndim = x1.ndim
    x2_ndim = x2.ndim

    if x1_ndim == 0 or x2_ndim == 0:
        raise ValueError(
            "matmul: Input operand does not have enough dimensions"
        )

    exec_q = dpctl.utils.get_execution_queue((x1.sycl_queue, x2.sycl_queue))
    if exec_q is None:
        raise ValueError(
            "Execution placement can not be unambiguously inferred "
            "from input arguments."
        )

    squeeze_flag = x1_ndim == 1 or x2_ndim == 1
    if x1_ndim == 1:
        x1 = x1[dpnp.newaxis, :]
        x1_ndim = x1.ndim

    if x2_ndim == 1:
        x2 = x2[:, dpnp.newaxis]
        x2_ndim = x2.ndim

    x1_shape = x1.shape
    x2_shape = x2.shape
    if x1_shape[-1] != x2_shape[-2]:
        raise ValueError(
            "Input operand 1 has a mismatch in its core dimension 0, "
            "with gufunc signature (n?,k),(k,m?)->(n?,m?) "
            f"(size {x1_shape[1]} is different from {x2_shape[0]})"
        )

    # Determine the result data type # should be corrected for integer data type # VAHID
    res_dtype = _common_type(x1, x2)
    if x1.dtype != res_dtype:
        x1 = dpnp.astype(x1, res_dtype)
    if x2.dtype != res_dtype:
        x2 = dpnp.astype(x2, res_dtype)

    if x1_ndim == 2 and x2_ndim == 2:
        res_shape = (x1.shape[0], x2.shape[1])
    else:
        if x1_ndim != x2_ndim:
            diff = abs(x1_ndim - x2_ndim)

            if x1_ndim < x2_ndim:
                x1 = x1.reshape((1,) * diff + x1.shape)
                x1_ndim = x1.ndim
                x1_shape = x1.shape
                res_shape = x2_shape[:-2] + (x1_shape[-2], x2_shape[-1])
            else:
                x2 = x2.reshape((1,) * diff + x2.shape)
                x2_ndim = x2.ndim
                x2_shape = x2.shape
                res_shape = x1_shape[:-2] + (x1_shape[-2], x2_shape[-1])
        else:
            for i in range(x1_ndim - 2):
                if x1_shape[i] != x2_shape[i]:
                    if x1_shape[i] == 1:
                        x1 = dpnp.repeat(x1, x2_shape[i], axis=i)
                    elif x2_shape[i] == 1:
                        x2 = dpnp.repeat(x2, x1_shape[i], axis=i)
                    else:
                        raise ValueError(
                            "operands could not be broadcast together with remapped shapes."
                        )
            x1_shape = x1.shape
            x2_shape = x2.shape
            res_shape = x1_shape[:-1] + (x2_shape[-1],)

    result = dpnp.empty(res_shape, dtype=res_dtype, sycl_queue=exec_q)
    # Is it necessary to do a copy of the input arrays?!
    isRowMajor = True
    if result.size == 0:
        pass
    else:
        if x1.size == 0 or x2.size == 0:
            result = dpnp.zeros(res_shape, dtype=res_dtype, sycl_queue=exec_q)
        else:
            if x1_ndim == 2 and x2_ndim == 2:
                ht_blas_ev, _ = bi._gemm(
                    exec_q,
                    dpnp.get_usm_ndarray(x1),
                    dpnp.get_usm_ndarray(x2),
                    dpnp.get_usm_ndarray(result),
                    isRowMajor,
                    [],
                )
            else:
                # if_a_f_contig = a.flags["F_CONTIGUOUS"]
                # if_b_f_contig = b.flags["F_CONTIGUOUS"]
                # if_out_f_contig = out.flags["F_CONTIGUOUS"]

                # x1_strides = a.strides if not if_a_f_contig else a.strides[::-1]
                # x2_strides = b.strides if not if_b_f_contig else b.strides[::-1]
                # res_strides = out.strides if not if_out_f_contig else out.strides[::-1]

                x1_strides = x1.strides
                x2_strides = x2.strides
                res_strides = result.strides

                is_support_gemm(x1_strides, x1_ndim)
                is_support_gemm(x2_strides, x2_ndim)

                transa = is_row(x1_strides, x1_ndim)
                transb = is_row(x2_strides, x2_ndim)

                batch_size = res_shape[:-2][0]  # VAHID
                m = x1_shape[-2]
                n = x2_shape[-1]
                k = x1_shape[-1]

                # lda = max(x1_shape[-2:])
                # ldb = max(x2_shape[-2:])
                # ldc = max(res_shape[-2:])
                lda = k if transa else m
                ldb = n if transb else k
                ldc = n  # column major m, row major n # VAHID

                stridea = x1_strides[0]
                strideb = x2_strides[0]
                stridec = res_strides[-3]

                if x1_ndim > 3:
                    iter = ti._contract_iter2(
                        res_shape[:-2], x1_strides[:-2], x2_strides[:-2]
                    )
                    if len(iter[0]) != 1:
                        raise ValueError(
                            "Input arrays cannot be used in gemm_batch"
                        )
                    batch_size = iter[0][0]
                    stridea = iter[1][0]
                    strideb = iter[3][0]

                ht_blas_ev, _ = bi._gemm_batch(
                    exec_q,
                    dpnp.get_usm_ndarray(x1),
                    dpnp.get_usm_ndarray(x2),
                    dpnp.get_usm_ndarray(result),
                    m,
                    n,
                    k,
                    batch_size,
                    lda,
                    ldb,
                    ldc,
                    stridea,
                    strideb,
                    stridec,
                    transa,
                    transb,
                    [],
                )

            ht_blas_ev.wait()

    if squeeze_flag:
        result = dpnp.squeeze(result)

    if out is None:
        return result
    else:
        if out.shape != result.shape:
            raise ValueError(
                f"Output array of shape {result.shape} is needed, got {out.shape}."
            )
        elif not isinstance(out, dpnp_array):
            if isinstance(out, dpt.usm_ndarray):
                out = dpnp_array._create_from_usm_ndarray(out)
            else:
                raise TypeError(
                    "Output array must be any of supported type, but got {}".format(
                        type(out)
                    )
                )

        dpnp.copyto(out, result, casting="safe")

        return out


def is_support_gemm(strides, ndim):
    if strides[ndim - 1] != 1 and strides[ndim - 2] != 1:
        raise ValueError(
            "The input matrices must be contiguous on inner dimension."
        )


def is_row(strides, ndim):
    return strides[ndim - 1] == 1


def _common_type(*arrays):
    dtypes = [arr.dtype for arr in arrays]

    default = dpnp.default_float_type().name
    dtype_common = _common_type_internal(default, *dtypes)

    return dtype_common


def _common_type_internal(default_dtype, *dtypes):
    inexact_dtypes = [
        dtype if dtype.kind in "fc" else default_dtype for dtype in dtypes
    ]
    return dpnp.result_type(*inexact_dtypes)


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
    Parameters `x1` and `x2` are supported as :obj:`dpnp.ndarray`.
    Keyword argument `kwargs` is currently unsupported.
    Parameter `axes` is supported only with value ``1``.
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
