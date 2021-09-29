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

from dpnp.dpnp_utils import *
from dpnp.dpnp_algo import *
from dpnp.linalg.dpnp_algo_linalg import *


__all__ = [
    "cholesky",
    "cond",
    "det",
    "eig",
    "eigvals",
    "inv",
    "matrix_power",
    "matrix_rank",
    "multi_dot",
    "norm",
    "qr",
    "svd",
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

    x1_desc = dpnp.get_dpnp_descriptor(input)
    if x1_desc:
        if x1_desc.shape[-1] != x1_desc.shape[-2]:
            pass
        else:
            if input.dtype == dpnp.int32 or input.dtype == dpnp.int64:
                # TODO memory copy. needs to move into DPNPC
                input_ = dpnp.get_dpnp_descriptor(dpnp.astype(input, dpnp.float64))
            else:
                input_ = x1_desc
            return dpnp_cholesky(input_).get_pyobj()

    return call_origin(numpy.linalg.cholesky, input)


def cond(input, p=None):
    """
    Compute the condition number of a matrix.
    For full documentation refer to :obj:`numpy.linalg.cond`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameter p=[None, 1, -1, 2, -2, numpy.inf, -numpy.inf, 'fro'] is supported.

    See Also
    --------
    :obj:`dpnp.norm` : Matrix or vector norm.
    """

    if (not use_origin_backend(input)):
        if p in [None, 1, -1, 2, -2, numpy.inf, -numpy.inf, 'fro']:
            result_obj = dpnp_cond(input, p)
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result
        else:
            pass

    return call_origin(numpy.linalg.cond, input, p)


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

    x1_desc = dpnp.get_dpnp_descriptor(input)
    if x1_desc:
        if x1_desc.shape[-1] == x1_desc.shape[-2]:
            result_obj = dpnp_det(x1_desc).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.linalg.det, input)


def eig(x1):
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    .. seealso:: :obj:`numpy.linalg.eig`

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if (x1_desc.size > 0):
            return dpnp_eig(x1_desc)

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

    x1_desc = dpnp.get_dpnp_descriptor(input)
    if x1_desc:
        if x1_desc.size > 0:
            return dpnp_eigvals(x1_desc).get_pyobj()

    return call_origin(numpy.linalg.eigvals, input)


def inv(input):
    """
    Divide arguments element-wise.

    For full documentation refer to :obj:`numpy.linalg.inv`.

    Limitations
    -----------
        Input array is supported as :obj:`dpnp.ndarray`.
        Dimension of input array is supported to be equal to ``2``.
        Shape of input array is limited by ``input.shape[0] == input.shape[1]``, ``input.shape[0] >= 2``.
        Otherwise the function will be executed sequentially on CPU.
    """

    x1_desc = dpnp.get_dpnp_descriptor(input)
    if x1_desc:
        if x1_desc.ndim == 2 and x1_desc.shape[0] == x1_desc.shape[1] and x1_desc.shape[0] >= 2:
            return dpnp_inv(x1_desc).get_pyobj()

    return call_origin(numpy.linalg.inv, input)


def matrix_power(input, count):
    """
    Raise a square matrix to the (integer) power `count`.

    Parameters
    ----------
    input : sequence of array_like

    Returns
    -------
    output : array
        Returns the dot product of the supplied arrays.

    See Also
    --------
    :obj:`numpy.linalg.matrix_power`

    """

    if not use_origin_backend() and count > 0:
        result = input
        for id in range(count - 1):
            result = dpnp.matmul(result, input)

        return result

    return call_origin(numpy.linalg.matrix_power, input, count)


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

    x1_desc = dpnp.get_dpnp_descriptor(input)
    if x1_desc:
        if tol is not None:
            pass
        elif hermitian:
            pass
        else:
            result_obj = dpnp_matrix_rank(x1_desc).get_pyobj()
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

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


def norm(x1, ord=None, axis=None, keepdims=False):
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
    ord : optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.
    axis : optional.
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

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if not isinstance(axis, int) and not isinstance(axis, tuple) and axis is not None:
            pass
        elif keepdims is not False:
            pass
        elif ord not in [None, 0, 3, 'fro', 'f']:
            pass
        else:
            result_obj = dpnp_norm(x1, ord=ord, axis=axis)
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.linalg.norm, x1, ord, axis, keepdims)


def qr(x1, mode='reduced'):
    """
    Compute the qr factorization of a matrix.

    Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
    upper-triangular.

    For full documentation refer to :obj:`numpy.linalg.qr`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameter mode='complete' is supported.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if mode != 'reduced':
            pass
        else:
            result_tup = dpnp_qr(x1, mode)

            return result_tup

    return call_origin(numpy.linalg.qr, x1, mode)


def svd(x1, full_matrices=True, compute_uv=True, hermitian=False):
    """
    Singular Value Decomposition.

    For full documentation refer to :obj:`numpy.linalg.svd`.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
    >>> b = np.random.randn(2, 7, 8, 3) + 1j*np.random.randn(2, 7, 8, 3)

    Reconstruction based on full SVD, 2D case:

    >>> u, s, vh = np.linalg.svd(a, full_matrices=True)
    >>> u.shape, s.shape, vh.shape
    ((9, 9), (6,), (6, 6))
    >>> np.allclose(a, np.dot(u[:, :6] * s, vh))
    True
    >>> smat = np.zeros((9, 6), dtype=complex)
    >>> smat[:6, :6] = np.diag(s)
    >>> np.allclose(a, np.dot(u, np.dot(smat, vh)))
    True

    Reconstruction based on reduced SVD, 2D case:

    >>> u, s, vh = np.linalg.svd(a, full_matrices=False)
    >>> u.shape, s.shape, vh.shape
    ((9, 6), (6,), (6, 6))
    >>> np.allclose(a, np.dot(u * s, vh))
    True
    >>> smat = np.diag(s)
    >>> np.allclose(a, np.dot(u, np.dot(smat, vh)))
    True

    Reconstruction based on full SVD, 4D case:

    >>> u, s, vh = np.linalg.svd(b, full_matrices=True)
    >>> u.shape, s.shape, vh.shape
    ((2, 7, 8, 8), (2, 7, 3), (2, 7, 3, 3))
    >>> np.allclose(b, np.matmul(u[..., :3] * s[..., None, :], vh))
    True
    >>> np.allclose(b, np.matmul(u[..., :3], s[..., None] * vh))
    True

    Reconstruction based on reduced SVD, 4D case:

    >>> u, s, vh = np.linalg.svd(b, full_matrices=False)
    >>> u.shape, s.shape, vh.shape
    ((2, 7, 8, 3), (2, 7, 3), (2, 7, 3, 3))
    >>> np.allclose(b, np.matmul(u * s[..., None, :], vh))
    True
    >>> np.allclose(b, np.matmul(u, s[..., None] * vh))
    True

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1)
    if x1_desc:
        if not x1_desc.ndim == 2:
            pass
        elif not full_matrices == True:
            pass
        elif not compute_uv == True:
            pass
        elif not hermitian == False:
            pass
        else:
            result_tup = dpnp_svd(x1_desc, full_matrices, compute_uv, hermitian)

            return result_tup

    return call_origin(numpy.linalg.svd, x1, full_matrices, compute_uv, hermitian)
