# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
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

# pylint: disable=invalid-name
# pylint: disable=no-member

from typing import NamedTuple

import numpy
from dpctl.tensor._numpy_helper import normalize_axis_tuple

import dpnp

from .dpnp_utils_linalg import (
    assert_2d,
    assert_stacked_2d,
    assert_stacked_square,
    dpnp_cholesky,
    dpnp_cond,
    dpnp_det,
    dpnp_eigh,
    dpnp_inv,
    dpnp_lstsq,
    dpnp_matrix_power,
    dpnp_matrix_rank,
    dpnp_multi_dot,
    dpnp_norm,
    dpnp_pinv,
    dpnp_qr,
    dpnp_slogdet,
    dpnp_solve,
    dpnp_svd,
)

__all__ = [
    "cholesky",
    "cond",
    "cross",
    "det",
    "diagonal",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "inv",
    "lstsq",
    "matmul",
    "matrix_norm",
    "matrix_power",
    "matrix_rank",
    "matrix_transpose",
    "multi_dot",
    "norm",
    "outer",
    "pinv",
    "qr",
    "solve",
    "svd",
    "svdvals",
    "slogdet",
    "tensordot",
    "tensorinv",
    "tensorsolve",
    "trace",
    "vecdot",
    "vector_norm",
]


# pylint:disable=missing-class-docstring
class EigResult(NamedTuple):
    eigenvalues: dpnp.ndarray
    eigenvectors: dpnp.ndarray


def cholesky(a, /, *, upper=False):
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
    upper : {bool}, optional
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
    assert_stacked_2d(a)
    assert_stacked_square(a)

    return dpnp_cholesky(a, upper=upper)


def cond(x, p=None):
    """
    Compute the condition number of a matrix.

    For full documentation refer to :obj:`numpy.linalg.cond`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        The matrix whose condition number is sought.
    p : {None, 1, -1, 2, -2, inf, -inf, "fro"}, optional
        Order of the norm used in the condition number computation.
        ``inf`` means the `dpnp.inf` object, and the Frobenius norm is
        the root-of-sum-of-squares norm.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The condition number of the matrix. May be infinite.

    See Also
    --------
    :obj:`dpnp.linalg.norm` : Matrix or vector norm.

    Notes
    -----
    This function will raise :class:`dpnp.linalg.LinAlgError` on singular input
    when using any of the norm: ``1``, ``-1``, ``inf``, ``-inf``, or ``'fro'``.
    In contrast, :obj:`numpy.linalg.cond` will fill the result array with
    ``inf`` values for each 2D batch in the input array that is singular
    when using these norms.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 0, -1], [0, 1, 0], [1, 0, 1]])
    >>> a
    array([[ 1,  0, -1],
           [ 0,  1,  0],
           [ 1,  0,  1]])
    >>> np.linalg.cond(a)
    array(1.41421356)
    >>> np.linalg.cond(a, 'fro')
    array(3.16227766)
    >>> np.linalg.cond(a, np.inf)
    array(2.)
    >>> np.linalg.cond(a, -np.inf)
    array(1.)
    >>> np.linalg.cond(a, 1)
    array(2.)
    >>> np.linalg.cond(a, -1)
    array(1.)
    >>> np.linalg.cond(a, 2)
    array(1.41421356)
    >>> np.linalg.cond(a, -2)
    array(0.70710678) # may vary
    >>> x = min(np.linalg.svd(a, compute_uv=False))
    >>> y = min(np.linalg.svd(np.linalg.inv(a), compute_uv=False))
    >>> x * y
    array(0.70710678) # may vary

    """

    dpnp.check_supported_arrays_type(x)
    return dpnp_cond(x, p)


def cross(x1, x2, /, *, axis=-1):
    """
    Returns the cross product of 3-element vectors.

    If `x1` and/or `x2` are multi-dimensional arrays, then
    the cross-product of each pair of corresponding 3-element vectors
    is independently computed.

    This function is Array API compatible, contrary to :obj:`dpnp.cross`.

    For full documentation refer to :obj:`numpy.linalg.cross`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        First input array.
    b : {dpnp.ndarray, usm_ndarray}
        Second input array. Must be compatible with `x1` for all
        non-compute axes. The size of the axis over which to compute
        the cross-product must be the same size as the respective axis
        in `x1`.
    axis : int, optional
        The axis (dimension) of `x1` and `x2` containing the vectors for
        which to compute the cross-product.

        Default: ``-1``.

    Returns
    -------
    out : dpnp.ndarray
         An array containing the cross products.

    See Also
    --------
    :obj:`dpnp.cross` : Similar function with support for more
                    keyword arguments.

    Examples
    --------
    Vector cross-product.

    >>> import dpnp as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> np.linalg.cross(x, y)
    array([-3,  6, -3])

    Multiple vector cross-products. Note that the direction of the cross
    product vector is defined by the *right-hand rule*.

    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = np.array([[4, 5, 6], [1, 2, 3]])
    >>> np.linalg.cross(x, y)
    array([[-3,  6, -3],
           [ 3, -6,  3]])

    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([[4, 5], [6, 1], [2, 3]])
    >>> np.linalg.cross(x, y, axis=0)
    array([[-24,  6],
           [ 18, 24],
           [-6,  -18]])

    """

    dpnp.check_supported_arrays_type(x1, x2)
    if x1.shape[axis] != 3 or x2.shape[axis] != 3:
        raise ValueError(
            "Both input arrays must be (arrays of) 3-dimensional vectors, "
            f"but they are {x1.shape[axis]} and {x2.shape[axis]} "
            "dimensional instead."
        )

    return dpnp.cross(x1, x2, axis=axis)


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
    :obj:`dpnp.linalg.slogdet` : Returns sign and logarithm of the determinant
                                 of an array.

    Examples
    --------
    The determinant of a 2-D array ``[[a, b], [c, d]]`` is ``ad - bc``:

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
    assert_stacked_2d(a)
    assert_stacked_square(a)

    return dpnp_det(a)


def diagonal(x, /, *, offset=0):
    """
    Returns specified diagonals of a matrix (or a stack of matrices) `x`.

    This function is Array API compatible, contrary to :obj:`dpnp.diagonal`
    the matrix is assumed to be defined by the last two dimensions.

    For full documentation refer to :obj:`numpy.linalg.diagonal`.

    Parameters
    ----------
    x : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Input array having shape (..., M, N) and whose innermost two
        dimensions form ``MxN`` matrices.
    offset : int, optional
        Offset specifying the off-diagonal relative to the main diagonal,
        where:

            * offset = 0: the main diagonal.
            * offset > 0: off-diagonal above the main diagonal.
            * offset < 0: off-diagonal below the main diagonal.

        Default: ``0``.

    Returns
    -------
    out : (...,min(N, M)) dpnp.ndarray
        An array containing the diagonals and whose shape is determined by
        removing the last two dimensions and appending a dimension equal to
        the size of the resulting diagonals. The returned array must have
        the same data type as `x`.

    See Also
    --------
    :obj:`dpnp.diagonal` : Similar function with support for more
                    keyword arguments.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(4).reshape(2, 2); a
    array([[0, 1],
           [2, 3]])
    >>> np.linalg.diagonal(a)
    array([0, 3])

    A 3-D example:

    >>> a = np.arange(8).reshape(2, 2, 2); a
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.linalg.diagonal(a)
    array([[0, 3],
           [4, 7]])

    Diagonals adjacent to the main diagonal can be obtained by using the
    `offset` argument:

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.linalg.diagonal(a, offset=1)  # First superdiagonal
    array([1, 5])
    >>> np.linalg.diagonal(a, offset=2)  # Second superdiagonal
    array([2])
    >>> np.linalg.diagonal(a, offset=-1)  # First subdiagonal
    array([3, 7])
    >>> np.linalg.diagonal(a, offset=-2)  # Second subdiagonal
    array([6])

    The anti-diagonal can be obtained by reversing the order of elements
    using either :obj:`dpnp.flipud` or :obj:`dpnp.fliplr`.

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.linalg.diagonal(np.fliplr(a))  # Horizontal flip
    array([2, 4, 6])
    >>> np.linalg.diagonal(np.flipud(a))  # Vertical flip
    array([6, 4, 2])

    Note that the order in which the diagonal is retrieved varies depending
    on the flip function.

    """

    return dpnp.diagonal(x, offset, axis1=-2, axis2=-1)


def eig(a):
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    For full documentation refer to :obj:`numpy.linalg.eig`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Matrices for which the eigenvalues and right eigenvectors will
        be computed.

    Returns
    -------
    A namedtuple with the following attributes:

    eigenvalues : (..., M) dpnp.ndarray
        The eigenvalues, each repeated according to its multiplicity.
        The eigenvalues are not necessarily ordered. The resulting array will
        be of complex type, unless the imaginary part is zero in which case it
        will be cast to a real type. When `a` is real the resulting eigenvalues
        will be real (zero imaginary part) or occur in conjugate pairs.
    eigenvectors : (..., M, M) dpnp.ndarray
        The normalized (unit "length") eigenvectors, such that the column
        ``eigenvectors[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``eigenvalues[i]``.

    Note
    ----
    Since there is no proper OneMKL LAPACK function, DPNP will calculate
    through a fallback on NumPy call.

    See Also
    --------
    :obj:`dpnp.linalg.eigvals` : Compute the eigenvalues of a general matrix.
    :obj:`dpnp.linalg.eigh` : Return the eigenvalues and eigenvectors of
                              a complex Hermitian (conjugate symmetric) or
                              a real symmetric matrix.
    :obj:`dpnp.linalg.eigvalsh` : Compute the eigenvalues of a complex
                                  Hermitian or real symmetric matrix.

    Examples
    --------
    >>> import dpnp as np
    >>> from dpnp import linalg as LA

    (Almost) trivial example with real eigenvalues and eigenvectors.

    >>> w, v = LA.eig(np.diag((1, 2, 3)))
    >>> w, v
    (array([1., 2., 3.]),
     array([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]))

    Real matrix possessing complex eigenvalues and eigenvectors;
    note that the eigenvalues are complex conjugates of each other.

    >>> w, v = LA.eig(np.array([[1, -1], [1, 1]]))
    >>> w, v
    (array([1.+1.j, 1.-1.j]),
     array([[0.70710678+0.j        , 0.70710678-0.j        ],
            [0.        -0.70710678j, 0.        +0.70710678j]]))

    Complex-valued matrix with real eigenvalues (but complex-valued
    eigenvectors); note that ``a.conj().T == a``, i.e., `a` is Hermitian.

    >>> a = np.array([[1, 1j], [-1j, 1]])
    >>> w, v = LA.eig(a)
    >>> w, v
    (array([2.+0.j, 0.+0.j]),
     array([[ 0.        +0.70710678j,  0.70710678+0.j        ], # may vary
            [ 0.70710678+0.j        , -0.        +0.70710678j]])

    Be careful about round-off error!

    >>> a = np.array([[1 + 1e-9, 0], [0, 1 - 1e-9]])
    >>> # Theor. eigenvalues are 1 +/- 1e-9
    >>> w, v = LA.eig(a)
    >>> w, v
    (array([1., 1.]),
     array([[1., 0.],
            [0., 1.]]))

    """

    dpnp.check_supported_arrays_type(a)
    assert_stacked_2d(a)
    assert_stacked_square(a)

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    # Since geev function from OneMKL LAPACK is not implemented yet,
    # use NumPy for this calculation.
    w_np, v_np = numpy.linalg.eig(dpnp.asnumpy(a))
    return EigResult(
        dpnp.array(w_np, sycl_queue=a_sycl_queue, usm_type=a_usm_type),
        dpnp.array(v_np, sycl_queue=a_sycl_queue, usm_type=a_usm_type),
    )


def eigh(a, UPLO="L"):
    """
    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.

    Returns two objects, a 1-D array containing the eigenvalues of `a`, and
    a 2-D square array or matrix (depending on the input type) of the
    corresponding eigenvectors (in columns).

    For full documentation refer to :obj:`numpy.linalg.eigh`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        A complex- or real-valued array whose eigenvalues and eigenvectors are
        to be computed.
    UPLO : {"L", "U"}, optional
        Specifies the calculation uses either the lower ("L") or upper ("U")
        triangular part of the matrix.
        Regardless of this choice, only the real parts of the diagonal are
        considered to preserve the Hermite matrix property.
        It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

        Default: ``"L"``.

    Returns
    -------
    A namedtuple with the following attributes:

    eigenvalues : (..., M) dpnp.ndarray
        The eigenvalues in ascending order, each repeated according to its
        multiplicity.
    eigenvectors : (..., M, M) dpnp.ndarray
        The column ``eigenvectors[:, i]`` is the normalized eigenvector
        corresponding to the eigenvalue ``eigenvalues[i]``.

    See Also
    --------
    :obj:`dpnp.linalg.eigvalsh` : Compute the eigenvalues of a complex
                                  Hermitian or real symmetric matrix.
    :obj:`dpnp.linalg.eig` : Compute the eigenvalues and right eigenvectors of
                             a square array.
    :obj:`dpnp.linalg.eigvals` : Compute the eigenvalues of a general matrix.

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
    assert_stacked_2d(a)
    assert_stacked_square(a)

    UPLO = UPLO.upper()
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    return dpnp_eigh(a, UPLO=UPLO)


def eigvals(a):
    """
    Compute the eigenvalues of a general matrix.

    For full documentation refer to :obj:`numpy.linalg.eigvals`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        A complex- or real-valued matrix whose eigenvalues will be computed.

    Returns
    -------
    w : (..., M) dpnp.ndarray
        The eigenvalues, each repeated according to its multiplicity.
        They are not necessarily ordered, nor are they necessarily
        real for real matrices.

    Note
    ----
    Since there is no proper OneMKL LAPACK function, DPNP will calculate
    through a fallback on NumPy call.

    See Also
    --------
    :obj:`dpnp.linalg.eig` : Compute the eigenvalues and right eigenvectors of
                             a square array.
    :obj:`dpnp.linalg.eigvalsh` : Compute the eigenvalues of a complex
                                  Hermitian or real symmetric matrix.
    :obj:`dpnp.linalg.eigh` : Return the eigenvalues and eigenvectors of
                              a complex Hermitian (conjugate symmetric) or
                              a real symmetric matrix.

    Examples
    --------
    Illustration, using the fact that the eigenvalues of a diagonal matrix
    are its diagonal elements, that multiplying a matrix on the left
    by an orthogonal matrix, `Q`, and on the right by `Q.T` (the transpose
    of `Q`), preserves the eigenvalues of the "middle" matrix. In other words,
    if `Q` is orthogonal, then ``Q * A * Q.T`` has the same eigenvalues as
    ``A``:

    >>> import dpnp as np
    >>> from dpnp import linalg as LA
    >>> x = np.random.random()
    >>> Q = np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])
    >>> LA.norm(Q[0, :]), LA.norm(Q[1, :]), np.dot(Q[0, :],Q[1, :])
    (array(1.), array(1.), array(0.))

    Now multiply a diagonal matrix by ``Q`` on one side and by ``Q.T`` on the
    other:

    >>> D = np.diag((-1, 1))
    >>> LA.eigvals(D)
    array([-1.,  1.])
    >>> A = np.dot(Q, D)
    >>> A = np.dot(A, Q.T)
    >>> LA.eigvals(A)
    array([-1.,  1.]) # random

    """

    dpnp.check_supported_arrays_type(a)
    assert_stacked_2d(a)
    assert_stacked_square(a)

    # Since geev function from OneMKL LAPACK is not implemented yet,
    # use NumPy for this calculation.
    w_np = numpy.linalg.eigvals(dpnp.asnumpy(a))
    return dpnp.array(w_np, sycl_queue=a.sycl_queue, usm_type=a.usm_type)


def eigvalsh(a, UPLO="L"):
    """
    Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

    Main difference from :obj:`dpnp.linalg.eigh`: the eigenvectors are not
    computed.

    For full documentation refer to :obj:`numpy.linalg.eigvalsh`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        A complex- or real-valued array whose eigenvalues are to be computed.
    UPLO : {"L", "U"}, optional
        Specifies the calculation uses either the lower ("L") or upper ("U")
        triangular part of the matrix.
        Regardless of this choice, only the real parts of the diagonal are
        considered to preserve the Hermite matrix property.
        It therefore follows that the imaginary part of the diagonal
        will always be treated as zero.

        Default: ``"L"``.

    Returns
    -------
    w : (..., M) dpnp.ndarray
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.

    See Also
    --------
    :obj:`dpnp.linalg.eigh` : Return the eigenvalues and eigenvectors of
                              a complex Hermitian (conjugate symmetric)
                              or a real symmetric matrix.
    :obj:`dpnp.linalg.eigvals` : Compute the eigenvalues of a general matrix.
    :obj:`dpnp.linalg.eig` : Compute the eigenvalues and right eigenvectors of
                             a general matrix.

    Examples
    --------
    >>> import dpnp as np
    >>> from dpnp import linalg as LA
    >>> a = np.array([[1, -2j], [2j, 5]])
    >>> LA.eigvalsh(a)
    array([0.17157288, 5.82842712])

    """

    dpnp.check_supported_arrays_type(a)
    assert_stacked_2d(a)
    assert_stacked_square(a)

    UPLO = UPLO.upper()
    if UPLO not in ("L", "U"):
        raise ValueError("UPLO argument must be 'L' or 'U'")

    return dpnp_eigh(a, UPLO=UPLO, eigen_mode="N")


def inv(a):
    """
    Compute the (multiplicative) inverse of a matrix.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.

    For full documentation refer to :obj:`numpy.linalg.inv`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Matrix to be inverted.

    Returns
    -------
    out : (..., M, M) dpnp.ndarray
        (Multiplicative) inverse of the matrix a.

    See Also
    --------
    :obj:`dpnp.linalg.cond` : Compute the condition number of a matrix.
    :obj:`dpnp.linalg.svd` : Compute the singular value decomposition.

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
    assert_stacked_2d(a)
    assert_stacked_square(a)

    return dpnp_inv(a)


def lstsq(a, b, rcond=None):
    """
    Return the least-squares solution to a linear matrix equation.

    For full documentation refer to :obj:`numpy.linalg.lstsq`.

    Parameters
    ----------
    a : (M, N) {dpnp.ndarray, usm_ndarray}
        "Coefficient" matrix.
    b : {(M,), (M, K)} {dpnp.ndarray, usm_ndarray}
        Ordinate or "dependent variable" values.
        If `b` is two-dimensional, the least-squares solution
        is calculated for each of the `K` columns of `b`.
    rcond : {None, int, float}, optional
        Cut-off ratio for small singular values of `a`.
        For the purposes of rank determination, singular values are treated as
        zero if they are smaller than `rcond` times the largest singular value
        of `a`.
        The default uses the machine precision times ``max(M, N)``. Passing
        ``-1`` will use machine precision.

        Default: ``None``.

    Returns
    -------
    x : {(N,), (N, K)} dpnp.ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.
    residuals : {(1,), (K,), (0,)} dpnp.ndarray
        Sums of squared residuals: Squared Euclidean 2-norm for each column in
        ``b - a @ x``.
        If the rank of `a` is < N or M <= N, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : int
        Rank of matrix `a`.
    s : (min(M, N),) dpnp.ndarray
        Singular values of `a`.

    Examples
    --------
    Fit a line, ``y = mx + c``, through some noisy data-points:

    >>> import dpnp as np
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])

    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 and cut the y-axis at, more or less, -1.

    We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``
    and ``p = [[m], [c]]``. Now use `lstsq` to solve for `p`:

    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    array([[0., 1.],
           [1., 1.],
           [2., 1.],
           [3., 1.]])

    >>> m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    >>> m, c
    (array(1.), array(-0.95)) # may vary

    """

    dpnp.check_supported_arrays_type(a, b)
    assert_2d(a)
    if rcond is not None and not isinstance(rcond, (int, float)):
        raise TypeError("rcond must be integer, floating type, or None")

    return dpnp_lstsq(a, b, rcond=rcond)


def matmul(x1, x2, /):
    """
    Computes the matrix product.

    This function is Array API compatible, contrary to :obj:`dpnp.matmul`.

    For full documentation refer to :obj:`numpy.linalg.matmul`.

    Parameters
    ----------
    x1 : {dpnp.ndarray, usm_ndarray}
        First input array.
    x2 : {dpnp.ndarray, usm_ndarray}
        Second input array.

    Returns
    -------
    out : dpnp.ndarray
        Returns the matrix product of the inputs.
        This is a 0-d array only when both `x1`, `x2` are 1-d vectors.

    See Also
    --------
    :obj:`dpnp.matmul` : Similar function with support for more
                    keyword arguments.

    Examples
    --------
    For 2-D arrays it is the matrix product:

    >>> import dpnp as np
    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.linalg.matmul(a, b)
    array([[4, 1],
           [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([1, 2])
    >>> np.linalg.matmul(a, b)
    array([1, 2])
    >>> np.linalg.matmul(b, a)
    array([1, 2])

    Broadcasting is conventional for stacks of arrays

    >>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
    >>> b = np.arange(2 * 2 * 4).reshape((2, 4, 2))
    >>> np.linalg.matmul(a, b).shape
    (2, 2, 2)
    >>> np.linalg.matmul(a, b)[0, 1, 1]
    array(98)
    >>> np.sum(a[0, 1, :] * b[0 , :, 1])
    array(98)

    Vector, vector returns the scalar inner product, but neither argument
    is complex-conjugated:

    >>> x1 = np.array([2j, 3j])
    >>> x2 = np.array([2j, 3j])
    >>> np.linalg.matmul(x1, x2)
    array(-13+0j)

    """

    return dpnp.matmul(x1, x2)


def matrix_norm(x, /, *, keepdims=False, ord="fro"):
    """
    Computes the matrix norm of a matrix (or a stack of matrices) `x`.

    This function is Array API compatible.

    For full documentation refer to :obj:`numpy.linalg.matrix_norm`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array having shape (..., M, N) and whose two innermost
        dimensions form ``MxN`` matrices.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are normed over are left in
        the result as dimensions with size one. With this option the result
        will broadcast correctly against the original `x`.

        Default: ``False``.
    ord : {None, 1, -1, 2, -2, dpnp.inf, -dpnp.inf, 'fro', 'nuc'}, optional
        The order of the norm. For details see the table under ``Notes``
        section in :obj:`dpnp.linalg.norm`.

        Default: ``"fro"``.

    Returns
    -------
    out : dpnp.ndarray
        Norm of the matrix.

    See Also
    --------
    :obj:`dpnp.linalg.norm` : Generic norm function.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> np.linalg.matrix_norm(b)
    array(7.74596669)
    >>> np.linalg.matrix_norm(b, ord='fro')
    array(7.74596669)
    >>> np.linalg.matrix_norm(b, ord=np.inf)
    array(9.)
    >>> np.linalg.matrix_norm(b, ord=-np.inf)
    array(2.)

    >>> np.linalg.matrix_norm(b, ord=1)
    array(7.)
    >>> np.linalg.matrix_norm(b, ord=-1)
    array(6.)
    >>> np.linalg.matrix_norm(b, ord=2)
    array(7.34846923)
    >>> np.linalg.matrix_norm(b, ord=-2)
    array(4.35106603e-18) # may vary

    """

    return dpnp.linalg.norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)


def matrix_power(a, n):
    """
    Raise a square matrix to the (integer) power `n`.

    For full documentation refer to :obj:`numpy.linalg.matrix_power`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Matrix to be "powered".
    n : int
        The exponent can be any integer or long integer, positive, negative,
        or zero.

    Returns
    -------
    a**n : (..., M, M) dpnp.ndarray
        The return value is the same shape and type as `M`;
        if the exponent is positive or zero then the type of the
        elements is the same as those of `M`. If the exponent is
        negative the elements are floating-point.

    Examples
    --------
    >>> import dpnp as np
    >>> i = np.array([[0, 1], [-1, 0]]) # matrix equiv. of the imaginary unit
    >>> np.linalg.matrix_power(i, 3) # should = -i
    array([[ 0, -1],
           [ 1,  0]])
    >>> np.linalg.matrix_power(i, 0)
    array([[1, 0],
           [0, 1]])
    >>> np.linalg.matrix_power(i, -3) # should 1/(-i) = i, but w/ f.p. elements
    array([[ 0.,  1.],
           [-1.,  0.]])

    Somewhat more sophisticated example

    >>> q = np.zeros((4, 4))
    >>> q[0:2, 0:2] = -i
    >>> q[2:4, 2:4] = i
    >>> q # one of the three quaternion units not equal to 1
    array([[ 0., -1.,  0.,  0.],
           [ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.],
           [ 0.,  0., -1.,  0.]])
    >>> np.linalg.matrix_power(q, 2) # = -np.eye(4)
    array([[-1.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0., -1.,  0.],
           [ 0.,  0.,  0., -1.]])

    """

    dpnp.check_supported_arrays_type(a)
    assert_stacked_2d(a)
    assert_stacked_square(a)

    if not isinstance(n, int):
        raise TypeError("exponent must be an integer")

    return dpnp_matrix_power(a, n)


def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
    """
    Return matrix rank of array using SVD method.

    Rank of the array is the number of singular values of the array that are
    greater than `tol`.

    Parameters
    ----------
    A : {(M,), (..., M, N)} {dpnp.ndarray, usm_ndarray}
        Input vector or stack of matrices.
    tol : (...) {None, float, dpnp.ndarray, usm_ndarray}, optional
        Threshold below which SVD values are considered zero. Only `tol` or
        `rtol` can be set at a time. If none of them are provided, defaults
        to ``S.max() * max(M, N) * eps`` where `S` is an array with singular
        values for `A`, and `eps` is the epsilon value for datatype of `S`.

        Default: ``None``.
    hermitian : bool, optional
        If ``True``, `A` is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.

        Default: ``False``.
    rtol : (...) {None, float, dpnp.ndarray, usm_ndarray}, optional
        Parameter for the relative tolerance component. Only `tol` or `rtol`
        can be set at a time. If none of them are provided, defaults to
        ``max(M, N) * eps`` where `eps` is the epsilon value for datatype
        of `S` (an array with singular values for `A`).

        Default: ``None``.

    Returns
    -------
    rank : (...) dpnp.ndarray
        Rank of `A`.

    See Also
    --------
    :obj:`dpnp.linalg.svd` : Singular Value Decomposition.

    Examples
    --------
    >>> import dpnp as np
    >>> from dpnp.linalg import matrix_rank
    >>> matrix_rank(np.eye(4)) # Full rank matrix
    array(4)
    >>> I=np.eye(4); I[-1,-1] = 0. # rank deficient matrix
    >>> matrix_rank(I)
    array(3)
    >>> matrix_rank(np.ones((4,))) # 1 dimension - rank 1 unless all 0
    array(1)
    >>> matrix_rank(np.zeros((4,)))
    array(0)

    """

    dpnp.check_supported_arrays_type(A)
    if tol is not None:
        dpnp.check_supported_arrays_type(
            tol, scalar_type=True, all_scalars=True
        )
    if rtol is not None:
        dpnp.check_supported_arrays_type(
            rtol, scalar_type=True, all_scalars=True
        )

    return dpnp_matrix_rank(A, tol=tol, hermitian=hermitian, rtol=rtol)


def matrix_transpose(x, /):
    """
    Transposes a matrix (or a stack of matrices) `x`.

    For full documentation refer to :obj:`numpy.linalg.matrix_transpose`.

    Parameters
    ----------
    x : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Input array with ``x.ndim >= 2`` and whose two innermost
        dimensions form ``MxN`` matrices.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the transpose for each matrix and having shape
        (..., N, M).

    See Also
    --------
    :obj:`dpnp.transpose` : Returns an array with axes transposed.
    :obj:`dpnp.matrix_transpose` : Equivalent function.
    :obj:`dpnp.ndarray.mT` : Equivalent method.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.linalg.matrix_transpose(a)
    array([[1, 3],
           [2, 4]])

    >>> b = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> np.linalg.matrix_transpose(b)
    array([[[1, 3],
            [2, 4]],
           [[5, 7],
            [6, 8]]])

    """

    return dpnp.matrix_transpose(x)


def multi_dot(arrays, *, out=None):
    """
    Compute the dot product of two or more arrays in a single function call.

    For full documentation refer to :obj:`numpy.linalg.multi_dot`.

    Parameters
    ----------
    arrays : sequence of dpnp.ndarray or usm_ndarray
        If the first argument is 1-D it is treated as row vector.
        If the last argument is 1-D it is treated as column vector.
        The other arguments must be 2-D.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a, b)`. If these conditions are not met, an exception is
        raised, instead of attempting to be flexible.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    :obj:`dpnp.dot` : Returns the dot product of two arrays.
    :obj:`dpnp.inner` : Returns the inner product of two arrays.

    Examples
    --------
    >>> import dpnp as np
    >>> from dpnp.linalg import multi_dot
    >>> A = np.random.random((10000, 100))
    >>> B = np.random.random((100, 1000))
    >>> C = np.random.random((1000, 5))
    >>> D = np.random.random((5, 333))

    the actual dot multiplication

    >>> multi_dot([A, B, C, D]).shape
    (10000, 333)

    instead of

    >>> np.dot(np.dot(np.dot(A, B), C), D).shape
    (10000, 333)

    or

    >>> A.dot(B).dot(C).dot(D).shape
    (10000, 333)

    """

    dpnp.check_supported_arrays_type(*arrays)
    n = len(arrays)
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    if n == 2:
        return dpnp.dot(arrays[0], arrays[1], out=out)

    return dpnp_multi_dot(n, arrays, out)


def norm(x, ord=None, axis=None, keepdims=False):
    r"""
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    For full documentation refer to :obj:`numpy.linalg.norm`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array. If `axis` is ``None``, `x` must be 1-D or 2-D, unless
        `ord` is ``None``. If both `axis` and `ord` are ``None``, the 2-norm
        of ``x.ravel`` will be returned.
    ord : {int, float, inf, -inf, "fro", "nuc"}, optional
        Norm type. inf means dpnp's `inf` object.

        Default: ``None``.
    axis : {None, int, 2-tuple of ints}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms. If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed. If `axis` is ``None`` then either a vector norm (when
        `x` is 1-D) or a matrix norm (when `x` is 2-D) is returned.

        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are normed over are left in
        the result as dimensions with size one. With this option the result
        will broadcast correctly against the original `x`.

        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        Norm of the matrix or vector(s).

    See Also
    --------
    :obj:`dpnp.linalg.matrix_norm` : Computes the matrix norm of a matrix.
    :obj:`dpnp.linalg.vector_norm` : Computes the vector norm of a vector.

    Notes
    -----
    For values of ``ord < 1``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\sum_{i, j} abs(a_{i, j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices and raise a ValueError when ``x.ndim != 2``.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, -1,  0,  1,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> np.linalg.norm(a)
    array(7.74596669)
    >>> np.linalg.norm(b)
    array(7.74596669)
    >>> np.linalg.norm(b, 'fro')
    array(7.74596669)
    >>> np.linalg.norm(a, np.inf)
    array(4.)
    >>> np.linalg.norm(b, np.inf)
    array(9.)
    >>> np.linalg.norm(a, -np.inf)
    array(0.)
    >>> np.linalg.norm(b, -np.inf)
    array(2.)

    >>> np.linalg.norm(a, 1)
    array(20.)
    >>> np.linalg.norm(b, 1)
    array(7.)
    >>> np.linalg.norm(a, -1)
    array(0.)
    >>> np.linalg.norm(b, -1)
    array(6.)
    >>> np.linalg.norm(a, 2)
    array(7.74596669)
    >>> np.linalg.norm(b, 2)
    array(7.34846923)

    >>> np.linalg.norm(a, -2)
    array(0.)
    >>> np.linalg.norm(b, -2)
    array(4.35106603e-18) # may vary
    >>> np.linalg.norm(a, 3)
    array(5.84803548) # may vary
    >>> np.linalg.norm(a, -3)
    array(0.)

    Using the `axis` argument to compute vector norms:

    >>> c = np.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> np.linalg.norm(c, axis=0)
    array([ 1.41421356,  2.23606798,  5.        ])
    >>> np.linalg.norm(c, axis=1)
    array([ 3.74165739,  4.24264069])
    >>> np.linalg.norm(c, ord=1, axis=1)
    array([ 6.,  6.])

    Using the `axis` argument to compute matrix norms:

    >>> m = np.arange(8).reshape(2, 2, 2)
    >>> np.linalg.norm(m, axis=(1, 2))
    array([  3.74165739,  11.22497216])
    >>> np.linalg.norm(m[0, :, :]), np.linalg.norm(m[1, :, :])
    (array(3.74165739), array(11.22497216))

    """

    dpnp.check_supported_arrays_type(x)
    return dpnp_norm(x, ord, axis, keepdims)


def outer(x1, x2, /):
    """
    Compute the outer product of two vectors.

    This function is Array API compatible. Compared to :obj:`dpnp.outer`,
    it accepts 1-dimensional inputs only.

    For full documentation refer to :obj:`numpy.linalg.outer`.

    Parameters
    ----------
    a : (M,) {dpnp.ndarray, usm_ndarray}
        One-dimensional input array of size ``M``.
        Must have a numeric data type.
    b : (N,) {dpnp.ndarray, usm_ndarray}
        One-dimensional input array of size ``N``.
        Must have a numeric data type.

    Returns
    -------
    out : (M, N) dpnp.ndarray
        ``out[i, j] = a[i] * b[j]``

    See Also
    --------
    :obj:`dpnp.outer` : Similar function with support for more
                    keyword arguments.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1, 1, 1])
    >>> b = np.array([1, 2, 3])
    >>> np.linalg.outer(a, b)
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])

    Make a (*very* coarse) grid for computing a Mandelbrot set:

    >>> rl = np.linalg.outer(np.ones((5,)), np.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    >>> im = np.linalg.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
    >>> im
    array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
           [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
           [0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
    >>> grid = rl + im
    >>> grid
    array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],
           [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
           [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
           [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
           [-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])

    """

    dpnp.check_supported_arrays_type(x1, x2)
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError(
            "Input arrays must be one-dimensional, but they are "
            f"{x1.ndim=} and {x2.ndim=}."
        )

    return dpnp.outer(x1, x2)


def pinv(a, rcond=None, hermitian=False, *, rtol=None):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all large singular values.

    For full documentation refer to :obj:`numpy.linalg.inv`.

    Parameters
    ----------
    a : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Matrix or stack of matrices to be pseudo-inverted.
    rcond : (...) {None, float, dpnp.ndarray, usm_ndarray}, optional
        Cutoff for small singular values.
        Singular values less than or equal to ``rcond * largest_singular_value``
        are set to zero. Broadcasts against the stack of matrices.
        Only `rcond` or `rtol` can be set at a time. If none of them are
        provided, defaults to ``max(M, N) * dpnp.finfo(a.dtype).eps``.

        Default: ``None``.
    hermitian : bool, optional
        If ``True``, a is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.

        Default: ``False``.
    rtol : (...) {None, float, dpnp.ndarray, usm_ndarray}, optional
        Same as `rcond`, but it's an Array API compatible parameter name.
        Only `rcond` or `rtol` can be set at a time. If none of them are
        provided, defaults to ``max(M, N) * dpnp.finfo(a.dtype).eps``.

        Default: ``None``.

    Returns
    -------
    out : (..., N, M) dpnp.ndarray
        The pseudo-inverse of `a`.

    Examples
    --------
    The following example checks that ``a * a+ * a == a`` and
    ``a+ * a * a+ == a+``:

    >>> import dpnp as np
    >>> a = np.random.randn(9, 6)
    >>> B = np.linalg.pinv(a)
    >>> np.allclose(a, np.dot(a, np.dot(B, a)))
    array(True)
    >>> np.allclose(B, np.dot(B, np.dot(a, B)))
    array(True)

    """

    dpnp.check_supported_arrays_type(a)
    if rcond is not None:
        dpnp.check_supported_arrays_type(
            rcond, scalar_type=True, all_scalars=True
        )
    if rtol is not None:
        dpnp.check_supported_arrays_type(
            rtol, scalar_type=True, all_scalars=True
        )
    assert_stacked_2d(a)

    return dpnp_pinv(a, rcond=rcond, hermitian=hermitian, rtol=rtol)


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

        Default: ``"reduced"``.

    Returns
    -------
    When mode is "reduced" or "complete", the result will be a namedtuple with
    the attributes `Q` and `R`:

    Q : dpnp.ndarray of float or complex, optional
        A matrix with orthonormal columns.
        When mode is ``"complete"`` the result is an orthogonal/unitary matrix
        depending on whether or not `a` is real/complex. The determinant may be
        either ``+/- 1`` in that case. In case the number of dimensions in the
        input array is greater than 2 then a stack of the matrices with above
        properties is returned.
    R : dpnp.ndarray of float or complex, optional
        The upper-triangular matrix or a stack of upper-triangular matrices if
        the number of dimensions in the input array is greater than 2.
    (h, tau) : tuple of dpnp.ndarray of float or complex, optional
        The array `h` contains the Householder reflectors that generate `Q`
        along with `R`. The `tau` array contains scaling factors for the
        reflectors.

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
    assert_stacked_2d(a)

    if mode not in ("reduced", "complete", "r", "raw"):
        raise ValueError(f"Unrecognized mode {mode}")

    return dpnp_qr(a, mode)


def solve(a, b):
    """
    Solve a linear matrix equation, or system of linear scalar equations.

    For full documentation refer to :obj:`numpy.linalg.solve`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Coefficient matrix.
    b : {(M,), (..., M, K)} {dpnp.ndarray, usm_ndarray}
        Ordinate or "dependent variable" values.

    Returns
    -------
    out : {(..., M,), (..., M, K)} dpnp.ndarray
        Solution to the system `ax = b`. Returned shape is identical to `b`.

    See Also
    --------
    :obj:`dpnp.dot` : Returns the dot product of two arrays.

    Notes
    -----
    The `b` array is only treated as a shape (M,) column vector if it is
    exactly 1-dimensional. In all other instances it is treated as a stack
    of (M, K) matrices.

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
    assert_stacked_2d(a)
    assert_stacked_square(a)

    a_shape = a.shape
    b_shape = b.shape
    b_ndim = b.ndim

    # compatible with numpy>=2.0
    if b_ndim == 0:
        raise ValueError("b must have at least one dimension")
    if b_ndim == 1:
        if a_shape[-1] != b.size:
            raise ValueError(
                "a must have (..., M, M) shape and b must have (M,) "
                "for one-dimensional b"
            )
        b = dpnp.broadcast_to(b, a_shape[:-1])
        return dpnp_solve(a, b)

    if a_shape[-1] != b_shape[-2]:
        raise ValueError(
            "a must have (..., M, M) shape and b must have (..., M, K) shape"
        )

    # Use dpnp.broadcast_shapes() to align the resulting batch shapes
    broadcasted_batch_shape = dpnp.broadcast_shapes(a_shape[:-2], b_shape[:-2])

    a_broadcasted_shape = broadcasted_batch_shape + a_shape[-2:]
    b_broadcasted_shape = broadcasted_batch_shape + b_shape[-2:]

    if a_shape != a_broadcasted_shape:
        a = dpnp.broadcast_to(a, a_broadcasted_shape)
    if b_shape != b_broadcasted_shape:
        b = dpnp.broadcast_to(b, b_broadcasted_shape)

    return dpnp_solve(a, b)


def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    """
    Singular Value Decomposition.

    For full documentation refer to :obj:`numpy.linalg.svd`.

    Parameters
    ----------
    a : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Input array with ``a.ndim >= 2``.
    full_matrices : {bool}, optional
        If ``True``, it returns `u` and `Vh` with full-sized matrices.
        If ``False``, the matrices are reduced in size.

        Default: ``True``.
    compute_uv : {bool}, optional
        If ``False``, it only returns singular values.

        Default: ``True``.
    hermitian : {bool}, optional
        If True, a is assumed to be Hermitian (symmetric if real-valued),
        enabling a more efficient method for finding singular values.

        Default: ``False``.

    Returns
    -------
    When `compute_uv` is ``True``, the result is a namedtuple with the
    following attribute names:

    U : { (…, M, M), (…, M, K) } dpnp.ndarray
        Unitary matrix, where M is the number of rows of the input array `a`.
        The shape of the matrix `U` depends on the value of `full_matrices`.
        If `full_matrices` is ``True``, `U` has the shape (…, M, M). If
        `full_matrices` is ``False``, `U` has the shape (…, M, K), where
        ``K = min(M, N)``, and N is the number of columns of the input array
        `a`. If `compute_uv` is ``False``, neither `U` or `Vh` are computed.
    S : (…, K) dpnp.ndarray
        Vector containing the singular values of `a`, sorted in descending
        order. The length of `S` is min(M, N).
    Vh : { (…, N, N), (…, K, N) } dpnp.ndarray
        Unitary matrix, where N is the number of columns of the input array `a`.
        The shape of the matrix `Vh` depends on the value of `full_matrices`.
        If `full_matrices` is ``True``, `Vh` has the shape (…, N, N).
        If `full_matrices` is ``False``, `Vh` has the shape (…, K, N).
        If `compute_uv` is ``False``, neither `U` or `Vh` are computed.

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
    assert_stacked_2d(a)

    return dpnp_svd(a, full_matrices, compute_uv, hermitian)


def svdvals(x, /):
    """
    Returns the singular values of a matrix (or a stack of matrices) `x`.

    When `x` is a stack of matrices, the function will compute
    the singular values for each matrix in the stack.

    Calling ``dpnp.linalg.svdvals(x)`` to get singular values is the same as
    ``dpnp.linalg.svd(x, compute_uv=False, hermitian=False)``.

    For full documentation refer to :obj:`numpy.linalg.svdvals`.

    Parameters
    ----------
    x : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Input array with ``x.ndim >= 2`` and whose last two dimensions
        form matrices on which to perform singular value decomposition.

    Returns
    -------
    out : (..., K) dpnp.ndarray
        Vector(s) of singular values of length K, where K = min(M, N).


    See Also
    --------
    :obj:`dpnp.linalg.svd` : Compute the singular value decomposition.


    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[3, 0], [0, 4]])
    >>> np.linalg.svdvals(a)
    array([4., 3.])

    This is equivalent to calling:

    >>> np.linalg.svd(a, compute_uv=False, hermitian=False)
    array([4., 3.])

    Stack of matrices:

    >>> b = np.array([[[6, 0], [0, 8]], [[9, 0], [0, 12]]])
    >>> np.linalg.svdvals(b)
    array([[ 8.,  6.],
           [12.,  9.]])

    """

    dpnp.check_supported_arrays_type(x)
    assert_stacked_2d(x)

    return dpnp_svd(x, full_matrices=True, compute_uv=False, hermitian=False)


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
    A namedtuple with the following attributes:

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
    The determinant of a 2-D array ``[[a, b], [c, d]]`` is ``ad - bc``:

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
    assert_stacked_2d(a)
    assert_stacked_square(a)

    return dpnp_slogdet(a)


def tensordot(a, b, /, *, axes=2):
    r"""
    Compute tensor dot product along specified axes.

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    For full documentation refer to :obj:`numpy.linalg.tensordot`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray, scalar}
        First input array. Both inputs `a` and `b` can not be scalars
        at the same time.
    b : {dpnp.ndarray, usm_ndarray, scalar}
        Second input array. Both inputs `a` and `b` can not be scalars
        at the same time.
    axes : int or (2,) array_like
        * integer_like: If an int `N`, sum over the last `N` axes of `a` and
          the first `N` axes of `b` in order. The sizes of the corresponding
          axes must match.
        * (2,) array_like: A list of axes to be summed over, first sequence
          applying to `a`, second to `b`. Both elements array_like must be of
          the same length.

          Default: ``2``.

    Returns
    -------
    out : dpnp.ndarray
        Returns the tensor dot product of `a` and `b`.

    See Also
    --------
    :obj:`dpnp.tensordot` : Equivalent function.
    :obj:`dpnp.dot` : Returns the dot product.
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention
                         on the operands.

    Notes
    -----
    Three common use cases are:
        * ``axes = 0`` : tensor product :math:`a \otimes b`
        * ``axes = 1`` : tensor dot product :math:`a \cdot b`
        * ``axes = 2`` : (default) tensor double contraction :math:`a:b`

    When `axes` is integer, the sequence for evaluation will be: first
    the -Nth axis in `a` and 0th axis in `b`, and the -1th axis in `a` and
    Nth axis in `b` last.

    When there is more than one axis to sum over - and they are not the last
    (first) axes of `a` (`b`) - the argument `axes` should consist of
    two sequences of the same length, with the first axis to sum over given
    first in both sequences, the second axis second, and so forth.

    The shape of the result consists of the non-contracted axes of the
    first tensor, followed by the non-contracted axes of the second.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b = np.array([1, 2, 3])
    >>> np.linalg.tensordot(a, b, axes=1)
    array([14, 32, 50])

    >>> a = np.arange(60.).reshape(3, 4, 5)
    >>> b = np.arange(24.).reshape(4, 3, 2)
    >>> c = np.linalg.tensordot(a, b, axes=([1, 0], [0, 1]))
    >>> c.shape
    (5, 2)
    >>> c
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    A slower but equivalent way of computing the same...

    >>> d = np.zeros((5, 2))
    >>> for i in range(5):
    ...   for j in range(2):
    ...     for k in range(3):
    ...       for n in range(4):
    ...         d[i, j] += a[k, n, i] * b[n, k, j]
    >>> c == d
    array([[ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True],
           [ True,  True]])

    """

    return dpnp.tensordot(a, b, axes=axes)


def tensorinv(a, ind=2):
    """
    Compute the 'inverse' of an N-dimensional array.

    For full documentation refer to :obj:`numpy.linalg.tensorinv`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Tensor to `invert`. Its shape must be 'square', i. e.,
        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
    ind : int, optional
        Number of first indices that are involved in the inverse sum.
        Must be a positive integer.

        Default: ``2``.

    Returns
    -------
    out : dpnp.ndarray
        The inverse of a tensor whose shape is equivalent to
        ``a.shape[ind:] + a.shape[:ind]``.

    See Also
    --------
    :obj:`dpnp.linalg.tensordot` : Compute tensor dot product along specified
                                   axes.
    :obj:`dpnp.linalg.tensorsolve` : Solve the tensor equation
                                     ``a x = b`` for x.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.eye(4*6)
    >>> a.shape = (4, 6, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=2)
    >>> ainv.shape
    (8, 3, 4, 6)

    >>> a = np.eye(4*6)
    >>> a.shape = (24, 8, 3)
    >>> ainv = np.linalg.tensorinv(a, ind=1)
    >>> ainv.shape
    (8, 3, 24)

    """

    dpnp.check_supported_arrays_type(a)

    if ind <= 0:
        raise ValueError("Invalid ind argument")

    old_shape = a.shape
    inv_shape = old_shape[ind:] + old_shape[:ind]
    prod = numpy.prod(old_shape[ind:])
    a = dpnp.reshape(a, (prod, -1))
    a_inv = inv(a)

    return a_inv.reshape(*inv_shape)


def tensorsolve(a, b, axes=None):
    """
    Solve the tensor equation ``a x = b`` for x.

    For full documentation refer to :obj:`numpy.linalg.tensorsolve`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
        'square').
    b : {dpnp.ndarray, usm_ndarray}
        Right-hand tensor, which can be of any shape.
    axes : {None, tuple of ints}, optional
        Axes in `a` to reorder to the right, before inversion.
        If ``None`` , no reordering is done.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        The tensor with shape ``Q`` such that ``b.shape + Q == a.shape``.

    See Also
    --------
    :obj:`dpnp.linalg.tensordot` : Compute tensor dot product along specified
                                   axes.
    :obj:`dpnp.linalg.tensorinv` : Compute the 'inverse' of an N-dimensional
                                   array.
    :obj:`dpnp.einsum` : Evaluates the Einstein summation convention on the
                         operands.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.eye(2*3*4)
    >>> a.shape = (2*3, 4, 2, 3, 4)
    >>> b = np.random.randn(2*3, 4)
    >>> x = np.linalg.tensorsolve(a, b)
    >>> x.shape
    (2, 3, 4)
    >>> np.allclose(np.tensordot(a, x, axes=3), b)
    array([ True])

    """

    dpnp.check_supported_arrays_type(a, b)
    a_ndim = a.ndim

    if axes is not None:
        all_axes = list(range(a_ndim))
        for k in axes:
            all_axes.remove(k)
            all_axes.insert(a_ndim, k)
        a = a.transpose(tuple(all_axes))

    old_shape = a.shape[-(a_ndim - b.ndim) :]
    prod = numpy.prod(old_shape)

    if a.size != prod**2:
        raise dpnp.linalg.LinAlgError(
            "Input arrays must satisfy the requirement "
            "prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])"
        )

    a = dpnp.reshape(a, (-1, prod))
    b = dpnp.ravel(b)
    res = solve(a, b)
    return res.reshape(old_shape)


def trace(x, /, *, offset=0, dtype=None):
    """
    Returns the sum along the specified diagonals of a matrix
    (or a stack of matrices) `x`.

    This function is Array API compatible, contrary to :obj:`dpnp.trace`.

    For full documentation refer to :obj:`numpy.linalg.trace`.

    Parameters
    ----------
    x : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Input array having shape (..., M, N) and whose innermost two
        dimensions form ``MxN`` matrices.
    offset : int, optional
        Offset specifying the off-diagonal relative to the main diagonal,
        where:

            * offset = 0: the main diagonal.
            * offset > 0: off-diagonal above the main diagonal.
            * offset < 0: off-diagonal below the main diagonal.

        Default: ``0``.
    dtype : {None, str, dtype object}, optional
        Determines the data-type of the returned array and of the accumulator
        where the elements are summed. If `dtype` has the value ``None`` and
        `a` is of integer type of precision less than the default integer
        precision, then the default integer precision is used. Otherwise, the
        precision is the same as that of `a`.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        An array containing the traces and whose shape is determined by
        removing the last two dimensions and storing the traces in the last
        array dimension. For example, if x has rank k and shape:
        (I, J, K, ..., L, M, N), then an output array has rank k-2 and shape:
        (I, J, K, ..., L) where:
        ``out[i, j, k, ..., l] = dpnp.linalg.trace(a[i, j, k, ..., l, :, :])``

        The returned array must have a data type as described by the dtype
        parameter above.

    See Also
    --------
    :obj:`dpnp.trace` : Similar function with support for more
                    keyword arguments.

    Examples
    --------
    >>> import dpnp as np
    >>> np.linalg.trace(np.eye(3))
    array(3.)
    >>> a = np.arange(8).reshape((2, 2, 2))
    >>> np.linalg.trace(a)
    array([3, 11])

    Trace is computed with the last two axes as the 2-d sub-arrays.
    This behavior differs from :obj:`dpnp.trace` which uses the first two
    axes by default.

    >>> a = np.arange(24).reshape((3, 2, 2, 2))
    >>> np.linalg.trace(a).shape
    (3, 2)

    Traces adjacent to the main diagonal can be obtained by using the
    `offset` argument:

    >>> a = np.arange(9).reshape((3, 3)); a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.linalg.trace(a, offset=1)  # First superdiagonal
    array(6)
    >>> np.linalg.trace(a, offset=2)  # Second superdiagonal
    array(2)
    >>> np.linalg.trace(a, offset=-1)  # First subdiagonal
    array(10)
    >>> np.linalg.trace(a, offset=-2)  # Second subdiagonal
    array(6)

    """

    return dpnp.trace(x, offset, axis1=-2, axis2=-1, dtype=dtype)


def vecdot(x1, x2, /, *, axis=-1):
    r"""
    Computes the vector dot product.

    This function is restricted to arguments compatible with the Array API,
    contrary to :obj:`dpnp.vecdot`.

    Let :math:`\mathbf{a}` be a vector in `x1` and :math:`\mathbf{b}` be
    a corresponding vector in `x2`. The dot product is defined as:

    .. math::
       \mathbf{a} \cdot \mathbf{b} = \sum_{i=0}^{n-1} \overline{a_i}b_i

    over the dimension specified by `axis` and where :math:`\overline{a_i}`
    denotes the complex conjugate if :math:`a_i` is complex and the identity
    otherwise.

    For full documentation refer to :obj:`numpy.linalg.vecdot`.

    Parameters
    ----------
    x1 : {dpnp.ndarray, usm_ndarray}
        First input array.
    x2 : {dpnp.ndarray, usm_ndarray}
        Second input array.
    axis : int, optional
        Axis over which to compute the dot product.

        Default: ``-1``.

    Returns
    -------
    out : dpnp.ndarray
        The vector dot product of the inputs.

    See Also
    --------
    :obj:`dpnp.vecdot` : Similar function with support for more
                    keyword arguments.
    :obj:`dpnp.vdot` : Complex-conjugating dot product.

    Examples
    --------
    Get the projected size along a given normal for an array of vectors.

    >>> import dpnp as np
    >>> v = np.array([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]])
    >>> n = np.array([0., 0.6, 0.8])
    >>> np.linalg.vecdot(v, n)
    array([ 3.,  8., 10.])

    """

    return dpnp.vecdot(x1, x2, axis=axis)


def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    """
    Computes the vector norm of a vector (or batch of vectors) `x`.

    This function is Array API compatible.

    For full documentation refer to :obj:`numpy.linalg.vector_norm`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, n-tuple of ints}, optional
        If an integer, `axis` specifies the axis (dimension) along which
        to compute vector norms. If an n-tuple, `axis` specifies the axes
        (dimensions) along which to compute batched vector norms. If ``None``,
        the vector norm must be computed over all array values (i.e.,
        equivalent to computing the vector norm of a flattened array).

        Default: ``None``.
    keepdims : {None, bool}, optional
        If this is set to ``True``, the axes which are normed over are left in
        the result as dimensions with size one. With this option the result
        will broadcast correctly against the original `x`.

        Default: ``False``.
    ord : {int, float, inf, -inf, 'fro', 'nuc'}, optional
        The order of the norm. For details see the table under ``Notes``
        section in :obj:`dpnp.linalg.norm`.

        Default: ``2``.

    Returns
    -------
    out : dpnp.ndarray
        Norm of the vector.

    See Also
    --------
    :obj:`dpnp.linalg.norm` : Generic norm function.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.arange(9) + 1
    >>> a
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> np.linalg.vector_norm(b)
    array(16.88194302)
    >>> np.linalg.vector_norm(b, ord=np.inf)
    array(9.)
    >>> np.linalg.vector_norm(b, ord=-np.inf)
    array(1.)

    >>> np.linalg.vector_norm(b, ord=1)
    array(45.)
    >>> np.linalg.vector_norm(b, ord=-1)
    array(0.35348576)
    >>> np.linalg.vector_norm(b, ord=2)
    array(16.881943016134134)
    >>> np.linalg.vector_norm(b, ord=-2)
    array(0.8058837395885292)

    """

    dpnp.check_supported_arrays_type(x)
    x_shape = list(x.shape)
    x_ndim = x.ndim
    if axis is None:
        # Note: dpnp.linalg.norm() doesn't handle 0-D arrays
        x = dpnp.ravel(x)
        _axis = 0
    elif isinstance(axis, tuple):
        # Note: The axis argument supports any number of axes, whereas
        # dpnp.linalg.norm() only supports a single axis or two axes
        # for vector norm.
        normalized_axis = normalize_axis_tuple(axis, x_ndim)
        rest = tuple(i for i in range(x_ndim) if i not in normalized_axis)
        newshape = axis + rest
        x = dpnp.transpose(x, newshape).reshape(
            (
                numpy.prod([x_shape[i] for i in axis], dtype=int),
                *[x_shape[i] for i in rest],
            )
        )
        _axis = 0
    else:
        _axis = axis

    res = dpnp.linalg.norm(x, axis=_axis, ord=ord)

    if keepdims:
        # We can't reuse dpnp.linalg.norm(keepdims) because of the reshape hacks
        # above to avoid matrix norm logic.
        _axis = normalize_axis_tuple(
            range(len(x_shape)) if axis is None else axis, len(x_shape)
        )
        for i in _axis:
            x_shape[i] = 1
        res = res.reshape(tuple(x_shape))

    return res
