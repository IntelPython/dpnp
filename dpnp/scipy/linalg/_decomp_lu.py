# -*- coding: utf-8 -*-
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
from dpnp.linalg.dpnp_utils_linalg import assert_stacked_2d

from ._utils import (
    dpnp_lu_factor,
    dpnp_lu_solve,
)

__all__ = [
    "lu_factor",
    "lu_solve",
]


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
    overwrite_a : {None, bool}, optional
        Whether to overwrite data in `a` (may increase performance).

        Default: ``False``.
    check_finite : {None, bool}, optional
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

    Warning
    -------
    This function synchronizes in order to validate array elements
    when ``check_finite=True``.

    See Also
    --------
    :obj:`dpnp.scipy.linalg.lu_solve` : Solve an equation system using
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
        LU factorization of matrix `a` (M, M) together with pivot indices.
    b : {(M,), (..., M, K)} {dpnp.ndarray, usm_ndarray}
        Right-hand side
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
    overwrite_b : {None, bool}, optional
        Whether to overwrite data in `b` (may increase performance).

        Default: ``False``.
    check_finite : {None, bool}, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

        Default: ``True``.

    Returns
    -------
    x : {(M,), (M, K)} dpnp.ndarray
        Solution to the system

    Warning
    -------
    This function synchronizes in order to validate array elements
    when ``check_finite=True``.

    See Also
    --------
    :obj:`dpnp.scipy.linalg.lu_factor` : LU factorize a matrix.

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

    (lu, piv) = lu_and_piv
    dpnp.check_supported_arrays_type(lu, piv, b)
    assert_stacked_2d(lu)

    return dpnp_lu_solve(
        lu,
        piv,
        b,
        trans=trans,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )
