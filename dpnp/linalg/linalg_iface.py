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


import dpnp
import numpy

from dpnp.dparray import dparray
from dpnp.dpnp_utils import *
from dpnp.linalg.linalg import *


__all__ = [
    "cholesky",
    "det",
    "eig",
    "eigvals",
    "inv",
    "matrix_power",
    "matrix_rank",
    "multi_dot",
    "norm"
]


def cholesky(input):
    """
    Cholesky decomposition.
    Return the Cholesky decomposition, `L * L.H`, of the square matrix `input`,
    where `L` is lower-triangular and .H is the conjugate transpose operator
    (which is the ordinary transpose if `input` is real-valued).  `input` must be
    Hermitian (symmetric if real-valued) and positive-definite. No
    checking is performed to verify whether `a` is Hermitian or not.
    In addition, only the lower-triangular and diagonal elements of `input`
    are used. Only `L` is actually returned.

    Parameters
    ----------
    input : (..., M, M) array_like
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.

    Returns
    -------
    L : (..., M, M) array_like
        Upper or lower-triangular Cholesky factor of `input`.  Returns a
        matrix object if `input` is a matrix object.
    """
    is_input_dparray = isinstance(input, dparray)

    if not use_origin_backend(input) and is_input_dparray and input.ndim == 2 and \
            input.shape[0] == input.shape[1] and input.shape[0] > 0:
        result = dpnp_cholesky(input)

        return result

    return call_origin(numpy.linalg.cholesky, input)


def det(input):
    """
    Compute the determinant of an array.

    Parameters
    ----------
    input : (..., M, M) array_like
        Input array to compute determinants for.

    Returns
    -------
    det : (...) array_like
        Determinant of `input`.
    """
    is_input_dparray = isinstance(input, dparray)

    if not use_origin_backend(input) and is_input_dparray:
        if input.shape[-1] == input.shape[-2]:
            result = dpnp_det(input)

            # scalar returned
            if result.shape == (1,):
                return result.dtype.type(result[0])

            return result

    return call_origin(numpy.linalg.det, input)


def eig(x1):
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    .. seealso:: :obj:`numpy.linalg.eig`

    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1) and is_x1_dparray):
        if (x1.size > 0):
            return dpnp_eig(x1)

    return call_origin(numpy.linalg.eig, x1)


def eigvals(input):
    """
    Compute the eigenvalues of a general matrix.
    Main difference between `eigvals` and `eig`: the eigenvectors aren't
    returned.

    Parameters
    ----------
    input : (..., M, M) array_like
        A complex- or real-valued matrix whose eigenvalues will be computed.

    Returns
    -------
    w : (..., M,) ndarray
        The eigenvalues, each repeated according to its multiplicity.
        They are not necessarily ordered, nor are they necessarily
        real for real matrices.
    """

    is_input_dparray = isinstance(input, dparray)

    if (not use_origin_backend(input) and is_input_dparray):
        if (input.size > 0):
            return dpnp_eigvals(input)

    return call_origin(numpy.linalg.eigvals, input)


def inv(input):
    """
    Divide arguments element-wise.

    For full documentation refer to :obj:`numpy.linalg.inv`.

    Limitations
    -----------
        Input array is supported as :obj:`dpnp.ndarray`.
        Input array support with ndim = 2, shape[0] = shape[1], shape[0] >= 2.
    """

    is_input_dparray = isinstance(input, dparray)

    if (not use_origin_backend(input) and is_input_dparray):
        if input.ndim == 2 and input.shape[0] == input.shape[1] and input.shape[0] >= 2:
            return dpnp_inv(input)

    return call_origin(numpy.linalg.inv, input)


def matrix_power(input, count):
    """
    Raise a square matrix to the (integer) power `count`.

    Parameters
    ----------
    input : sequence of array_like

    Returns
    -------
    output : dparray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    :obj:`numpy.linalg.matrix_power`

    """

    is_input_dparray = isinstance(input, dparray)

    if not use_origin_backend(input) and is_input_dparray and count > 0:
        result = input
        for id in range(count - 1):
            result = dpnp.matmul(result, input)

        return result

    input1 = dpnp.asnumpy(input) if is_input_dparray else input

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.linalg.matrix_power(input1, count)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def matrix_rank(input, tol=None, hermitian=False):
    """
    Return matrix rank of array
    Rank of the array is the number of singular values of the array that are
    greater than `tol`.

    Parameters
    ----------
    M : {(M,), (..., M, N)} array_like
        Input vector or stack of matrices.
    tol : (...) array_like, float, optional
        Threshold below which SVD values are considered zero. If `tol` is
        None, and ``S`` is an array with singular values for `M`, and
        ``eps`` is the epsilon value for datatype of ``S``, then `tol` is
        set to ``S.max() * max(M.shape) * eps``.
    hermitian : bool, optional
        If True, `M` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Defaults to False.

    Returns
    -------
    rank : (...) array_like
        Rank of M.

    """

    is_input_dparray = isinstance(input, dparray)

    if not use_origin_backend(input) and is_input_dparray:
        if tol is not None:
            checker_throw_value_error("matrix_rank", "tol", type(tol), None)
        if hermitian is not False:
            checker_throw_value_error("matrix_rank", "hermitian", hermitian, False)

        result = dpnp_matrix_rank(input)

        # scalar returned
        if result.shape == (1,):
            return result.dtype.type(result[0])

        return result

    return call_origin(numpy.linalg.matrix_rank, input, tol, hermitian)


def multi_dot(arrays, out=None):
    """
    Compute the dot product of two or more arrays in a single function call

    Parameters
    ----------
    arrays : sequence of array_like
        If the first argument is 1-D it is treated as row vector.
        If the last argument is 1-D it is treated as column vector.
        The other arguments must be 2-D.
    out : ndarray, optional
        unsupported

    Returns
    -------
    output : ndarray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    :obj:`numpy.multi_dot`

    """

    n = len(arrays)

    if n < 2:
        checker_throw_value_error("multi_dot", "arrays", n, ">1")

    result = arrays[0]
    for id in range(1, n):
        result = dpnp.dot(result, arrays[id])

    return result


def norm(input, ord=None, axis=None, keepdims=False):
    """
    Matrix or vector norm.
    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    input : array_like
        Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``x.ravel`` will be returned.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.
    axis : {None, int, 2-tuple of ints}, optional.
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
        is None.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).
    """

    is_input_dparray = isinstance(input, dparray)

    if not use_origin_backend(input) and is_input_dparray:
        if keepdims is not False:
            checker_throw_value_error("norm", "keepdims", keepdims, False)
        if not isinstance(axis, int) and not isinstance(axis, tuple) and axis is not None:
            raise TypeError("'axis' must be None, an integer or a tuple of integers")

        result = dpnp_norm(input, ord=ord, axis=axis)

        # scalar returned
        if result.shape == (1,) and axis is None:
            return result.dtype.type(result[0])

        return result

    return call_origin(numpy.linalg.norm, input, ord, axis, keepdims)
