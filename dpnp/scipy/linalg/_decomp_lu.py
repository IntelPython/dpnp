# *****************************************************************************
# Copyright (c) 2025, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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
Interface of the SciPy-compatible Linear Algebra subset for DPNP.

Notes
-----
This module exposes the public API for ``dpnp.scipy.linalg``.
It contains:
 - SciPy-like interface functions
 - documentation for the functions

"""

import dpnp
from dpnp.linalg.dpnp_utils_linalg import (
    assert_stacked_2d,
    assert_stacked_square,
)

from ._utils import (
    dpnp_lu,
    dpnp_lu_factor,
    dpnp_lu_solve,
)


def lu(
    a, permute_l=False, overwrite_a=False, check_finite=True, p_indices=False
):
    """
    Compute LU decomposition of a matrix with partial pivoting.

    The decomposition satisfies::

        A = P @ L @ U

    where `P` is a permutation matrix, `L` is lower triangular with unit
    diagonal elements, and `U` is upper triangular. If `permute_l` is set to
    ``True`` then `L` is returned already permuted and hence satisfying
    ``A = L @ U``.

    For full documentation refer to :obj:`scipy.linalg.lu`.

    Parameters
    ----------
    a : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Input array to decompose.
    permute_l : bool, optional
        Perform the multiplication ``P @ L`` (Default: do not permute).

        Default: ``False``.
    overwrite_a : bool, optional
        Whether to overwrite data in `a` (may increase performance).

        Default: ``False``.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

        Default: ``True``.
    p_indices : bool, optional
        If ``True`` the permutation information is returned as row indices
        instead of a permutation matrix.

        Default: ``False``.

    Returns
    -------
    The tuple ``(p, l, u)`` is returned if ``permute_l`` is ``False``
    (default), else the tuple ``(pl, u)`` is returned, where:

    p : (..., M, M) dpnp.ndarray or (..., M) dpnp.ndarray
        Permutation matrix or permutation indices.
        If `p_indices` is ``False`` (default), a permutation matrix.
        The permutation matrix always has a real-valued floating-point dtype
        even when `a` is complex, since it only contains 0s and 1s.
        If `p_indices` is ``True``, a 1-D (or batched) array of row
        permutation indices such that ``A = L[p] @ U``.
    l : (..., M, K) dpnp.ndarray
        Lower triangular or trapezoidal matrix with unit diagonal.
        ``K = min(M, N)``.
    pl : (..., M, K) dpnp.ndarray
        Permuted ``L`` matrix: ``pl = P @ L``.
        ``K = min(M, N)``.
    u : (..., K, N) dpnp.ndarray
        Upper triangular or trapezoidal matrix.

    Notes
    -----
    Permutation matrices are costly since they are nothing but row reorder of
    ``L`` and hence indices are strongly recommended to be used instead if the
    permutation is required. The relation in the 2D case then becomes simply
    ``A = L[P, :] @ U``. In higher dimensions, it is better to use `permute_l`
    to avoid complicated indexing tricks.

    In the 2D case, if one has the indices however, for some reason, the
    permutation matrix is still needed then it can be constructed by
    ``dpnp.eye(M)[P, :]``.

    Warnings
    --------
    This function synchronizes in order to validate array elements
    when ``check_finite=True``, and also synchronizes to compute the
    permutation from LAPACK pivot indices.

    See Also
    --------
    :func:`dpnp.scipy.linalg.lu_factor` : LU factorize a matrix
                                          (compact representation).
    :func:`dpnp.scipy.linalg.lu_solve` : Solve an equation system using
                                         the LU factorization of a matrix.

    Examples
    --------
    >>> import dpnp as np
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8],
    ...               [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> p, l, u = np.scipy.linalg.lu(A)
    >>> np.allclose(A, p @ l @ u)
    array(True)

    Retrieve the permutation as row indices with ``p_indices=True``:

    >>> p, l, u = np.scipy.linalg.lu(A, p_indices=True)
    >>> p
    array([1, 3, 0, 2])
    >>> np.allclose(A, l[p] @ u)
    array(True)

    Return the permuted ``L`` directly with ``permute_l=True``:

    >>> pl, u = np.scipy.linalg.lu(A, permute_l=True)
    >>> np.allclose(A, pl @ u)
    array(True)

    Non-square matrices are supported:

    >>> B = np.array([[1, 2, 3], [4, 5, 6]])
    >>> p, l, u = np.scipy.linalg.lu(B)
    >>> np.allclose(B, p @ l @ u)
    array(True)

    Batched input:

    >>> C = np.random.randn(3, 2, 4, 4)
    >>> p, l, u = np.scipy.linalg.lu(C)
    >>> np.allclose(C, p @ l @ u)
    array(True)

    """

    dpnp.check_supported_arrays_type(a)
    assert_stacked_2d(a)

    return dpnp_lu(
        a,
        overwrite_a=overwrite_a,
        check_finite=check_finite,
        p_indices=p_indices,
        permute_l=permute_l,
    )


def lu_factor(a, overwrite_a=False, check_finite=True):
    """
    Compute the pivoted LU decomposition of `a` matrix.

    The decomposition is::

        A = P @ L @ U

    where `P` is a permutation matrix, `L` is lower triangular with unit
    diagonal elements, and `U` is upper triangular.

    For full documentation refer to :obj:`scipy.linalg.lu_factor`.

    Parameters
    ----------
    a : (..., M, N) {dpnp.ndarray, usm_ndarray}
        Input array to decompose.
    overwrite_a : bool, optional
        Whether to overwrite data in `a` (may increase performance).

        Default: ``False``.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

        Default: ``True``.

    Returns
    -------
    lu : (..., M, N) dpnp.ndarray
        Matrix containing `U` in its upper triangle,
        and `L` in its lower triangle.
        The unit diagonal elements of `L` are not stored.
    piv : (..., K) dpnp.ndarray
        Pivot indices representing the permutation matrix `P`:
        row i of matrix was interchanged with row piv[i].
        Where ``K = min(M, N)``.

    Warnings
    --------
    This function synchronizes in order to validate array elements
    when ``check_finite=True``.

    See Also
    --------
    :func:`dpnp.scipy.linalg.lu_solve` : Solve an equation system using
                                         the LU factorization of `a` matrix.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([[4., 3.], [6., 3.]])
    >>> lu, piv = np.scipy.linalg.lu_factor(a)
    >>> lu
    array([[6.        , 3.        ],
           [0.66666667, 1.        ]])
    >>> piv
    array([1, 1])

    """

    dpnp.check_supported_arrays_type(a)
    assert_stacked_2d(a)

    return dpnp_lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """
    Solve a linear system, :math:`a x = b`, given the LU factorization of `a`.

    For full documentation refer to :obj:`scipy.linalg.lu_solve`.

    Parameters
    ----------
    lu, piv : {tuple of dpnp.ndarrays or usm_ndarrays}
        LU factorization of matrix `a` (..., M, M) together with pivot indices.
    b : {(M,), (..., M, K)} {dpnp.ndarray, usm_ndarray}
        Right-hand side.
    trans : {0, 1, 2} , optional
        Type of system to solve:

        =====  =================
        trans  system
        =====  =================
        0      :math:`a x = b`
        1      :math:`a^T x = b`
        2      :math:`a^H x = b`
        =====  =================

        Default: ``0``.
    overwrite_b : bool, optional
        Whether to overwrite data in `b` (may increase performance).

        Default: ``False``.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

        Default: ``True``.

    Returns
    -------
    x : {(M,), (..., M, K)} dpnp.ndarray
        Solution to the system

    Warnings
    --------
    This function synchronizes in order to validate array elements
    when ``check_finite=True``.

    See Also
    --------
    :func:`dpnp.scipy.linalg.lu_factor` : LU factorize a matrix.

    Examples
    --------
    >>> import dpnp as np
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> b = np.array([1, 1, 1, 1])
    >>> lu, piv = np.scipy.linalg.lu_factor(A)
    >>> x = np.scipy.linalg.lu_solve((lu, piv), b)
    >>> np.allclose(A @ x - b, np.zeros((4,)))
    array(True)

    """

    lu_matrix, piv = lu_and_piv
    dpnp.check_supported_arrays_type(lu_matrix, piv, b)
    assert_stacked_2d(lu_matrix)
    assert_stacked_square(lu_matrix)

    return dpnp_lu_solve(
        lu_matrix,
        piv,
        b,
        trans=trans,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )
