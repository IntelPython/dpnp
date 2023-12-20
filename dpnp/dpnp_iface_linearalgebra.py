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


def _gemm_res_dtype(*arrays, casting):
    """
    Determines the data types for matmul operation and the output array of matmul operation.

    The output array data type is determined based on the Promotion Type Rule
    and device capibilities. The data type used in matmul operation is an 'inexact' data type
    determined based on the output data type and device capabilities.
    Both data types are determined based on the fact that the output array data type can be cast
    to the other data type according to casting rule specified, otherwise a ``TypeError`` is raised.

    Parameters
    ----------
    arrays : {dpnp_array, usm_ndarray}
        Input arrays.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

    Returns
    -------
    gemm_dtype, res_dtype :
        The appropriate data types for performing matmul operation and presenting output array.

    """

    res_dtype = dpnp.result_type(*arrays)
    gemm_dtype = dpnp.default_float_type(device=arrays[0].device)
    if dpnp.issubdtype(res_dtype, dpnp.complexfloating):
        gemm_dtype = (
            dpnp.complex64 if gemm_dtype == dpnp.float32 else dpnp.complex128
        )

    if dpnp.can_cast(res_dtype, gemm_dtype, casting):
        if res_dtype in [
            dpnp.float64,
            dpnp.complex128,
        ]:  # in case device does not support fp64
            return gemm_dtype, gemm_dtype
        elif res_dtype in [
            dpnp.float32,
            dpnp.complex64,
        ]:  # needed dtype is fp32 but device supports fp64
            return res_dtype, res_dtype
        else:
            return gemm_dtype, res_dtype
    else:
        raise TypeError(
            f"Cannot cast ufunc 'matmul' output from dtype({res_dtype}) to dtype({gemm_dtype}) with casting rule {casting}"
        )
    

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


def matmul(
    x1,
    x2,
    /,
    out=None,
    *,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    """
    Matrix product of two arrays.

    For full documentation refer to :obj:`numpy.matmul`.

    Limitations
    -----------
    Input arrays and parameter `out` are supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Keyword argument `subok` is currently unsupported.
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
    For 2-D arrays it is the matrix product:

    >>> import dpnp as np
    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.matmul(a, b)
    array([[4, 1],
           [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([1, 2])
    >>> np.matmul(a, b)
    array([1, 2])
    >>> np.matmul(b, a)
    array([1, 2])

    Broadcasting is conventional for stacks of arrays

    >>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
    >>> b = np.arange(2 * 2 * 4).reshape((2, 4, 2))
    >>> np.matmul(a,b).shape
    (2, 2, 2)
    >>> np.matmul(a, b)[0, 1, 1]
    array(98)
    >>> np.sum(a[0, 1, :] * b[0 , :, 1])
    array(98)

    Vector, vector returns the scalar inner product, but neither argument is complex-conjugated:

    >>> x1 = np.array([2j, 3j])
    >>> x2 = np.array([2j, 3j])
    >>> np.matmul(x1, x2)
    array(-13+0j)

    The ``@`` operator can be used as a shorthand for ``matmul`` on
    :class:`dpnp.ndarray`.

    >>> x1 @ x2
    array(-13+0j)

    """

    dpnp.check_supported_arrays_type(x1)
    dpnp.check_supported_arrays_type(x2)
    if subok is False:
        raise NotImplementedError(
            "subok keyword argument is only supported by its default value."
        )
    else:
        x1_ndim = x1.ndim
        x2_ndim = x2.ndim

        if x1_ndim == 0 or x2_ndim == 0:
            raise ValueError(
                "matmul: Input operand does not have enough dimensions"
            )

        res_usm_type, exec_q = get_usm_allocations([x1, x2])

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

        # Determine the appropriate data types
        if dtype is not None:
            res_dtype = dtype
            gemm_dtype, _ = _gemm_res_dtype(x1, x2, casting=casting)
        else:
            gemm_dtype, res_dtype = _gemm_res_dtype(x1, x2, casting=casting)

        # find the result shape
        if x1_ndim == 2 and x2_ndim == 2:
            res_shape = (x1.shape[0], x2.shape[1])
        else:
            x1_is_2D = numpy.prod(x1_shape[:-2]) == 1  # inherently 2D
            x2_is_2D = numpy.prod(x2_shape[:-2]) == 1

            # makes the dimension of input the same by adding new axis
            if x1_ndim != x2_ndim:
                diff = abs(x1_ndim - x2_ndim)
                if x1_ndim < x2_ndim:
                    x1 = x1.reshape((1,) * diff + x1.shape)
                    x1_ndim = x1.ndim
                    x1_shape = x1.shape
                else:
                    x2 = x2.reshape((1,) * diff + x2.shape)
                    x2_ndim = x2.ndim
                    x2_shape = x2.shape

            # examining the option to align inputs
            # when their shapes differ but they are 1-D in some dimensions.
            tmp_shape = list(x1_shape[:-2])
            for i in range(x1_ndim - 2):
                if x1_shape[i] != x2_shape[i]:
                    if x1_shape[i] == 1:
                        tmp_shape[i] = x2_shape[i]
                        # If the `x1` array is inherently 2D, there's no need to
                        # duplicate the data for the 1-D dimension;
                        # GEMM handles it automatically.
                        if not x1_is_2D:
                            x1 = dpnp.repeat(x1, x2_shape[i], axis=i)
                    elif x2_shape[i] == 1:
                        tmp_shape[i] = x1_shape[i]
                        if not x2_is_2D:
                            x2 = dpnp.repeat(x2, x1_shape[i], axis=i)
                    else:
                        raise ValueError(
                            "operands could not be broadcast together with remapped shapes."
                        )
            x1_shape = x1.shape
            x2_shape = x2.shape
            res_shape = tuple(tmp_shape) + (x1_shape[-2], x2_shape[-1])

        is_x1_c_f_contig = x1.flags.c_contiguous or x1.flags.f_contiguous
        is_x2_c_f_contig = x2.flags.c_contiguous or x2.flags.f_contiguous
        list_of_events = []

        # input arrays should have the proper data type
        # and be C_CONTIGUOUS or F_CONTIGUOUS
        if not is_x1_c_f_contig or x1.dtype != gemm_dtype:
            x1_copy = dpnp.empty_like(x1, dtype=gemm_dtype, order="C")
            ht_copy_ev_x1, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=dpnp.get_usm_ndarray(x1),
                dst=x1_copy.get_array(),
                sycl_queue=x1.sycl_queue,
            )
            x1 = x1_copy
            list_of_events.append(ht_copy_ev_x1)

        if not is_x2_c_f_contig or x2.dtype != gemm_dtype:
            x2_copy = dpnp.empty_like(x2, dtype=gemm_dtype, order="C")
            ht_copy_ev_x2, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=dpnp.get_usm_ndarray(x2),
                dst=x2_copy.get_array(),
                sycl_queue=x2.sycl_queue,
            )
            x2 = x2_copy
            list_of_events.append(ht_copy_ev_x2)

        if list_of_events:
            for ev in list_of_events:
                ev.wait()

        # calculate results
        result = dpnp.empty(
            res_shape,
            dtype=gemm_dtype,
            usm_type=res_usm_type,
            sycl_queue=exec_q,
        )
        if result.size == 0:
            pass
        else:
            if x1.size == 0 or x2.size == 0:
                result = dpnp.zeros(
                    res_shape,
                    dtype=gemm_dtype,
                    usm_type=res_usm_type,
                    sycl_queue=exec_q,
                )
            else:
                ht_copy_ev_x1 = dpctl.SyclEvent()
                ht_copy_ev_x2 = dpctl.SyclEvent()
                if x1_ndim == 2 and x2_ndim == 2:
                    ht_blas_ev, _ = bi._gemm(
                        exec_q,
                        dpnp.get_usm_ndarray(x1),
                        dpnp.get_usm_ndarray(x2),
                        dpnp.get_usm_ndarray(result),
                        [],
                    )
                else:
                    (
                        ht_blas_ev,
                        ht_copy_ev_x1,
                        ht_copy_ev_x2,
                        result,
                    ) = dpnp_matmul_batch(
                        exec_q,
                        x1,
                        x2,
                        result,
                        x1_is_2D,
                        x2_is_2D,
                        ht_copy_ev_x1,
                        ht_copy_ev_x2,
                    )

                ht_blas_ev.wait()
                ht_copy_ev_x1.wait()
                ht_copy_ev_x2.wait()

        if squeeze_flag:
            result = dpnp.squeeze(result)

        if gemm_dtype != res_dtype:
            result = dpnp.astype(result, res_dtype)

        if out is not None:
            return dpnp.get_result_array(result, out, casting=casting)

        # If `order` was not passed as default
        # we must copy `result` to match the passed `order`.
        if order not in ["k", "K"]:
            return dpnp.copy(result, order=order)

        return result


def dpnp_matmul_batch(
    exec_q, x1, x2, res, x1_is_2D, x2_is_2D, ht_copy_ev_x1, ht_copy_ev_x2
):
    # If input array is F-contiguous, we need to change the order to C-contiguous.
    # because mkl::gemm_bacth needs each 2D array to be F-contiguous but 
    # when the input array is F-contiguous, the data of 2D array 
    # that needs to be called in mkl::gemm_batch are not contiguous.
    copy_ev_x1 = dpctl.SyclEvent()
    if not x1.flags.c_contiguous:
        x1_copy  = dpnp.empty_like(x1, order="C")
        (
            ht_copy_ev_x1,
            copy_ev_x1,
        ) = ti._copy_usm_ndarray_into_usm_ndarray(
            src=dpnp.get_usm_ndarray(x1),
            dst=x1_copy.get_array(),
            sycl_queue=x1.sycl_queue,
        )
        x1 = x1_copy 

    copy_ev_x2 = dpctl.SyclEvent()
    if not x2.flags.c_contiguous:
        x2_copy  = dpnp.empty_like(x2, order="C")
        (
            ht_copy_ev_x2,
            copy_ev_x2,
        ) = ti._copy_usm_ndarray_into_usm_ndarray(
            src=dpnp.get_usm_ndarray(x2),
            dst=x2_copy.get_array(),
            sycl_queue=x2.sycl_queue,
        )
        x2 = x2_copy

    x1_strides = x1.strides
    x2_strides = x2.strides
    res_strides = res.strides

    # When shape along any particular dimension is 1,
    # the stride along that dimension is not a 
    # meaningful number and is undefined. Here, we 
    # standardizing strides before continuing,
    # setting stride to 0 if the shape along that axis is <=1
    if x1_is_2D:
        x1_strides = tuple(
            str_i if sh_i > 1 else 0
            for sh_i, str_i in zip(x1.shape, x1_strides)
        )
    if x2_is_2D:
        x2_strides = tuple(
            str_i if sh_i > 1 else 0
            for sh_i, str_i in zip(x2.shape, x2_strides)
        )

    batch_size = res.shape[:-2][0]
    m = x1.shape[-2]
    n = x2.shape[-1]
    k = x1.shape[-1]

    stridea = x1_strides[0]
    strideb = x2_strides[0]
    stridec = res_strides[-3]

    if x1.ndim > 3:
        iter = ti._contract_iter2(
            res.shape[:-2], x1_strides[:-2], x2_strides[:-2]
        )

        if len(iter[0]) != 1:
            raise ValueError("Input arrays cannot be used in gemm_batch")
        batch_size = iter[0][0]
        stridea = iter[1][0]
        strideb = iter[3][0]

    ht_blas_ev, _ = bi._gemm_batch(
        exec_q,
        dpnp.get_usm_ndarray(x1),
        dpnp.get_usm_ndarray(x2),
        dpnp.get_usm_ndarray(res),
        m,
        n,
        k,
        batch_size,
        k,  # lda
        n,  # ldb
        n,  # ldc
        stridea,
        strideb,
        stridec,
        True,  # transa
        True,  # transb
        [copy_ev_x1, copy_ev_x2],
    )

    return ht_blas_ev, ht_copy_ev_x1, ht_copy_ev_x2, res


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
