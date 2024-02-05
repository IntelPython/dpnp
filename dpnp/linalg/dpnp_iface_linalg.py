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
Interface of the Linear Algebra part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import numpy

import dpnp
from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *
from dpnp.linalg.dpnp_algo_linalg import *

from .dpnp_utils_linalg import (
    check_stacked_2d,
    check_stacked_square,
    dpnp_cholesky,
    dpnp_det,
    dpnp_eigh,
    dpnp_inv,
    dpnp_qr,
    dpnp_slogdet,
    dpnp_solve,
    dpnp_svd,
)

__all__ = [
    "cholesky",
    "cond",
    "det",
    "eig",
    "eigh",
    "eigvals",
    "inv",
    "matrix_power",
    "matrix_rank",
    "multi_dot",
    "norm",
    "qr",
    "solve",
    "svd",
    "slogdet",
]


def cholesky(a, upper=False):
    """
    Cholesky decomposition.

    Return the lower or upper Cholesky decomposition, ``L * L.H`` or
    ``U.H * U``, of the square matrix ``a``, where ``L`` is lower-triangular,
    ``U`` is upper-triangular, and ``.H`` is the conjugate transpose operator
    (which is the ordinary transpose if ``a`` is real-valued). ``a`` must be
    Hermitian (symmetric if real-valued) and positive-definite. No checking is
    performed to verify whether ``a`` is Hermitian or not. In addition, only
    the lower or upper-triangular and diagonal elements of ``a`` are used.
    Only ``L`` or ``U`` is actually returned.

    For full documentation refer to :obj:`numpy.linalg.cholesky`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Hermitian (symmetric if all elements are real), positive-definite
        input matrix.
    upper : bool, optional
        If ``True``, the result must be the upper-triangular Cholesky factor.
        If ``False``, the result must be the lower-triangular Cholesky factor.
        Default: ``False``.

    Returns
    -------
    L : (..., M, M) dpnp.ndarray
        Lower or upper-triangular Cholesky factor of `a`.

    Examples
    --------
    >>> import dpnp as np
    >>> A = np.array([[1.0, 2.0],[2.0, 5.0]])
    >>> A
    array([[1., 2.],
           [2., 5.]])
    >>> L = np.linalg.cholesky(A)
    >>> L
    array([[1., 0.],
           [2., 1.]])
    >>> np.dot(L, L.T.conj()) # verify that L * L.H = A
    array([[1., 2.],
           [2., 5.]])

    The upper-triangular Cholesky factor can also be obtained:

    >>> np.linalg.cholesky(A, upper=True)
    array([[ 1.+0.j, -0.-2.j],
           [ 0.+0.j,  1.+0.j]]

    """

    dpnp.check_supported_arrays_type(a)
    check_stacked_2d(a)
    check_stacked_square(a)

    return dpnp_cholesky(a, upper=upper)


def cond(input, p=None):
    """
    Compute the condition number of a matrix.

    For full documentation refer to :obj:`numpy.linalg.cond`.

    Limitations
    -----------
    Input array is supported as :obj:`dpnp.ndarray`.
    Parameter p=[None, 1, -1, 2, -2, dpnp.inf, -dpnp.inf, 'fro'] is supported.

    See Also
    --------
    :obj:`dpnp.norm` : Matrix or vector norm.
    """

    if not use_origin_backend(input):
        if p in [None, 1, -1, 2, -2, dpnp.inf, -dpnp.inf, "fro"]:
            result_obj = dpnp_cond(input, p)
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result
        else:
            pass

    return call_origin(numpy.linalg.cond, input, p)


def det(a):
    """
    Compute the determinant of an array.

    For full documentation refer to :obj:`numpy.linalg.det`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Input array to compute determinants for.

    Returns
    -------
    det : (...) dpnp.ndarray
        Determinant of `a`.

    See Also
    --------
    :obj:`dpnp.linalg.slogdet` : Returns sign and logarithm of the determinant of an array.

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

    >>> import dpnp as dp
    >>> a = dp.array([[1, 2], [3, 4]])
    >>> dp.linalg.det(a)
    array(-2.)

    Computing determinants for a stack of matrices:

    >>> a = dp.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)
    >>> dp.linalg.det(a)
    array([-2., -3., -8.])

    """

    dpnp.check_supported_arrays_type(a)
    check_stacked_2d(a)
    check_stacked_square(a)

    return dpnp_det(a)


def eig(x1):
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    .. seealso:: :obj:`numpy.linalg.eig`

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if x1_desc.size > 0:
            return dpnp_eig(x1_desc)

    return call_origin(numpy.linalg.eig, x1)


def eigh(a, UPLO="L"):
    """
    eigh(a, UPLO="L")

    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    For full documentation refer to :obj:`numpy.linalg.eigh`.

    Returns
    -------
    w : (..., M) dpnp.ndarray
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.
    v : (..., M, M) dpnp.ndarray
        The column ``v[:, i]`` is the normalized eigenvector corresponding
        to the eigenvalue ``w[i]``.

    Limitations
    -----------
    Parameter `a` is supported as :class:`dpnp.ndarray` or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.eig` : eigenvalues and right eigenvectors for non-symmetric arrays.
    :obj:`dpnp.eigvals` : eigenvalues of non-symmetric arrays.

    Examples
    --------
    >>> import dpnp as dp
    >>> a = dp.array([[1, -2j], [2j, 5]])
    >>> a
    array([[ 1.+0.j, -0.-2.j],
           [ 0.+2.j,  5.+0.j]])
    >>> w, v = dp.linalg.eigh(a)
    >>> w; v
    array([0.17157288, 5.82842712]),
    array([[-0.92387953-0.j        , -0.38268343+0.j        ], # may vary
           [ 0.        +0.38268343j,  0.        -0.92387953j]]))

    """

    dpnp.check_supported_arrays_type(a)

    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    if a.ndim < 2:
        raise ValueError(
            "%d-dimensional array given. Array must be "
            "at least two-dimensional" % a.ndim
        )

    m, n = a.shape[-2:]
    if m != n:
        raise ValueError("Last 2 dimensions of the array must be square")

    return dpnp_eigh(a, UPLO=UPLO)


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

    x1_desc = dpnp.get_dpnp_descriptor(input, copy_when_nondefault_queue=False)
    if x1_desc:
        if x1_desc.size > 0:
            return dpnp_eigvals(x1_desc).get_pyobj()

    return call_origin(numpy.linalg.eigvals, input)


def inv(a):
    """
    Compute the (multiplicative) inverse of a matrix.

    Given a square matrix a, return the matrix ainv
    satisfying ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.

    For full documentation refer to :obj:`numpy.linalg.inv`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Matrix to be inverted.

    Returns
    -------
    out : (..., M, M) dpnp.ndarray
        (Multiplicative) inverse of the matrix a.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> ainv = np.linalg.inv(a)
    >>> np.allclose(np.dot(a, ainv), np.eye(2))
    array([ True])
    >>> np.allclose(np.dot(ainv, a), np.eye(2))
    array([ True])

    Inverses of several matrices can be computed at once:
    >>> a = np.array([[[1., 2.], [3., 4.]], [[1, 3], [3, 5]]])
    >>> np.linalg.inv(a)
    array([[[-2.  ,  1.  ],
            [ 1.5 , -0.5 ]],

          [[-1.25,  0.75],
            [ 0.75, -0.25]]])

    """

    dpnp.check_supported_arrays_type(a)
    check_stacked_2d(a)
    check_stacked_square(a)

    return dpnp_inv(a)


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
        for _ in range(count - 1):
            result = dpnp.matmul(result, input)

        return result

    return call_origin(numpy.linalg.matrix_power, input, count)


def matrix_rank(input, tol=None, hermitian=False):
    """
    Return matrix rank of array.

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

    x1_desc = dpnp.get_dpnp_descriptor(input, copy_when_nondefault_queue=False)
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

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_nondefault_queue=False)
    if x1_desc:
        if (
            not isinstance(axis, int)
            and not isinstance(axis, tuple)
            and axis is not None
        ):
            pass
        elif keepdims is not False:
            pass
        elif ord not in [None, 0, 3, "fro", "f"]:
            pass
        else:
            result_obj = dpnp_norm(x1, ord=ord, axis=axis)
            result = dpnp.convert_single_elem_array_to_scalar(result_obj)

            return result

    return call_origin(numpy.linalg.norm, x1, ord, axis, keepdims)


def qr(a, mode="reduced"):
    """
    Compute the qr factorization of a matrix.

    Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
    upper-triangular.

    For full documentation refer to :obj:`numpy.linalg.qr`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        The input array with the dimensionality of at least 2.
    mode : {"reduced", "complete", "r", "raw"}, optional
        If K = min(M, N), then
        - "reduced" : returns Q, R with dimensions (…, M, K), (…, K, N)
        - "complete" : returns Q, R with dimensions (…, M, M), (…, M, N)
        - "r" : returns R only with dimensions (…, K, N)
        - "raw" : returns h, tau with dimensions (…, N, M), (…, K,)
        Default: "reduced".

    Returns
    -------
    When mode is "reduced" or "complete", the result will be a namedtuple with
    the attributes Q and R.
    Q : dpnp.ndarray
        A matrix with orthonormal columns.
        When mode = "complete" the result is an orthogonal/unitary matrix
        depending on whether or not a is real/complex.
        The determinant may be either +/- 1 in that case.
        In case the number of dimensions in the input array is greater
        than 2 then a stack of the matrices with above properties is returned.
    R : dpnp.ndarray
        The upper-triangular matrix or a stack of upper-triangular matrices
        if the number of dimensions in the input array is greater than 2.
    (h, tau) : tuple of dpnp.ndarray
        The h array contains the Householder reflectors that generate Q along with R.
        The tau array contains scaling factors for the reflectors.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.random.randn(9, 6)
    >>> Q, R = np.linalg.qr(a)
    >>> np.allclose(a, np.dot(Q, R))  # a does equal QR
    array([ True])
    >>> R2 = np.linalg.qr(a, mode='r')
    >>> np.allclose(R, R2)  # mode='r' returns the same R as mode='full'
    array([ True])
    >>> a = np.random.normal(size=(3, 2, 2)) # Stack of 2 x 2 matrices as input
    >>> Q, R = np.linalg.qr(a)
    >>> Q.shape
    (3, 2, 2)
    >>> R.shape
    (3, 2, 2)
    >>> np.allclose(a, np.matmul(Q, R))
    array([ True])

    """

    dpnp.check_supported_arrays_type(a)
    check_stacked_2d(a)

    if mode not in ("reduced", "complete", "r", "raw"):
        raise ValueError(f"Unrecognized mode {mode}")

    return dpnp_qr(a, mode)


def solve(a, b):
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    For full documentation refer to :obj:`numpy.linalg.solve`.

    Returns
    -------
    out : {(…, M,), (…, M, K)} dpnp.ndarray
        Solution to the system ax = b. Returned shape is identical to b.

    Limitations
    -----------
    Parameters `a` and `b` are supported as either :class:`dpnp.ndarray`
    or :class:`dpctl.tensor.usm_ndarray`.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.dot` : Returns the dot product of two arrays.

    Examples
    --------
    >>> import dpnp as dp
    >>> a = dp.array([[1, 2], [3, 5]])
    >>> b = dp.array([1, 2])
    >>> x = dp.linalg.solve(a, b)
    >>> x
    array([-1.,  1.])

    Check that the solution is correct:

    >>> dp.allclose(dp.dot(a, x), b)
    array([ True])

    """

    dpnp.check_supported_arrays_type(a, b)
    check_stacked_2d(a)
    check_stacked_square(a)

    if not (
        (a.ndim == b.ndim or a.ndim == b.ndim + 1)
        and a.shape[:-1] == b.shape[: a.ndim - 1]
    ):
        raise dpnp.linalg.LinAlgError(
            "a must have (..., M, M) shape and b must have (..., M) "
            "or (..., M, K)"
        )

    return dpnp_solve(a, b)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """
    Singular Value Decomposition.

    For full documentation refer to :obj:`numpy.linalg.svd`.

    Parameters
    ----------
    a : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Input array with ``a.ndim >= 2``.
    full_matrices : bool, optional
        If ``True``, it returns `u` and `Vh` with full-sized matrices.
        If ``False``, the matrices are reduced in size.
        Default: ``True``.
    compute_uv : bool, optional
        If ``False``, it only returns singular values.
        Default: ``True``.
    hermitian : bool, optional
        If True, a is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.
        Default: ``False``.

    Returns
    -------
    u : { (…, M, M), (…, M, K) } dpnp.ndarray
        Unitary matrix, where M is the number of rows of the input array `a`.
        The shape of the matrix `u` depends on the value of `full_matrices`.
        If `full_matrices` is ``True``, `u` has the shape (…, M, M).
        If `full_matrices` is ``False``, `u` has the shape (…, M, K),
        where K = min(M, N), and N is the number of columns of the input array `a`.
        If `compute_uv` is ``False``, neither `u` or `Vh` are computed.
    s : (…, K) dpnp.ndarray
        Vector containing the singular values of `a`, sorted in descending order.
        The length of `s` is min(M, N).
    Vh : { (…, N, N), (…, K, N) } dpnp.ndarray
        Unitary matrix, where N is the number of columns of the input array `a`.
        The shape of the matrix `Vh` depends on the value of `full_matrices`.
        If `full_matrices` is ``True``, `Vh` has the shape (…, N, N).
        If `full_matrices` is ``False``, `Vh` has the shape (…, K, N).
        If `compute_uv` is ``False``, neither `u` or `Vh` are computed.

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
    array([ True])
    >>> smat = np.zeros((9, 6), dtype=complex)
    >>> smat[:6, :6] = np.diag(s)
    >>> np.allclose(a, np.dot(u, np.dot(smat, vh)))
    array([ True])

    Reconstruction based on reduced SVD, 2D case:

    >>> u, s, vh = np.linalg.svd(a, full_matrices=False)
    >>> u.shape, s.shape, vh.shape
    ((9, 6), (6,), (6, 6))
    >>> np.allclose(a, np.dot(u * s, vh))
    array([ True])
    >>> smat = np.diag(s)
    >>> np.allclose(a, np.dot(u, np.dot(smat, vh)))
    array([ True])

    Reconstruction based on full SVD, 4D case:

    >>> u, s, vh = np.linalg.svd(b, full_matrices=True)
    >>> u.shape, s.shape, vh.shape
    ((2, 7, 8, 8), (2, 7, 3), (2, 7, 3, 3))
    >>> np.allclose(b, np.matmul(u[..., :3] * s[..., None, :], vh))
    array([ True])
    >>> np.allclose(b, np.matmul(u[..., :3], s[..., None] * vh))
    array([ True])

    Reconstruction based on reduced SVD, 4D case:

    >>> u, s, vh = np.linalg.svd(b, full_matrices=False)
    >>> u.shape, s.shape, vh.shape
    ((2, 7, 8, 3), (2, 7, 3), (2, 7, 3, 3))
    >>> np.allclose(b, np.matmul(u * s[..., None, :], vh))
    array([ True])
    >>> np.allclose(b, np.matmul(u, s[..., None] * vh))
    array([ True])

    """

    dpnp.check_supported_arrays_type(a)
    check_stacked_2d(a)

    return dpnp_svd(a, full_matrices, compute_uv, hermitian)


def slogdet(a):
    """
    Compute the sign and (natural) logarithm of the determinant of an array.

    For full documentation refer to :obj:`numpy.linalg.slogdet`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Input array, has to be a square 2-D array.

    Returns
    -------
    sign : (...) dpnp.ndarray
        A number representing the sign of the determinant. For a real matrix,
        this is 1, 0, or -1. For a complex matrix, this is a complex number
        with absolute value 1 (i.e., it is on the unit circle), or else 0.
    logabsdet : (...) dpnp.ndarray
        The natural log of the absolute value of the determinant.

    See Also
    --------
    :obj:`dpnp.det` : Returns the determinant of an array.

    Examples
    --------
    The determinant of a 2-D array [[a, b], [c, d]] is ad - bc:

    >>> import dpnp as dp
    >>> a = dp.array([[1, 2], [3, 4]])
    >>> (sign, logabsdet) = dp.linalg.slogdet(a)
    >>> (sign, logabsdet)
    (array(-1.), array(0.69314718))
    >>> sign * dp.exp(logabsdet)
    array(-2.)

    Computing log-determinants for a stack of matrices:

    >>> a = dp.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)
    >>> sign, logabsdet = dp.linalg.slogdet(a)
    >>> (sign, logabsdet)
    (array([-1., -1., -1.]), array([0.69314718, 1.09861229, 2.07944154]))
    >>> sign * dp.exp(logabsdet)
    array([-2., -3., -8.])

    """

    dpnp.check_supported_arrays_type(a)
    check_stacked_2d(a)
    check_stacked_square(a)

    return dpnp_slogdet(a)
