# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

import numpy

import dpnp
from dpnp.linalg.dpnp_utils_linalg import assert_stacked_2d

from ._utils import (
    dpnp_cho_factor,
    dpnp_cho_solve,
)


def _lower_to_bool(lower):
    """
    Convert `lower` to bool, also accepting an array or a sequence of equal
    values.

    """

    if dpnp.is_supported_array_type(lower):
        lower = dpnp.asnumpy(lower)
    elif isinstance(lower, (list, tuple)):
        # bool() of a non-empty sequence is always True
        lower = numpy.asarray(lower)
    if isinstance(lower, numpy.ndarray) and lower.size != 1:
        if lower.size == 0:
            return False
        if not (lower == lower.flat[0]).all():
            raise ValueError("All values of `lower` must be equal.")
        lower = lower.flat[0]
    return bool(lower)


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the Cholesky decomposition of a matrix, to use in
    :obj:`dpnp.scipy.linalg.cho_solve`.

    Returns a matrix containing the Cholesky decomposition,
    ``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.
    The return value can be directly used as the first parameter to
    :obj:`dpnp.scipy.linalg.cho_solve`.

    For full documentation refer to :obj:`scipy.linalg.cho_factor`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Matrix to be decomposed.
    lower : bool, optional
        Whether to compute the upper or lower triangular Cholesky
        factorization.

        Default: ``False``.
    overwrite_a : bool, optional
        Whether to overwrite data in `a` (may improve performance).

        Default: ``False``.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

        Default: ``True``.

    Returns
    -------
    c : (..., M, M) dpnp.ndarray
        Matrix whose upper or lower triangle contains the Cholesky factor
        of `a`. If `a` is not overwritten, the other triangle is zeroed,
        otherwise it contains the data of `a` not used by the decomposition.
    lower : bool
        Flag indicating whether the factor is in the lower or upper triangle.

    Raises
    ------
    dpnp.linalg.LinAlgError
        Raised if decomposition fails.

    Warnings
    --------
    This function synchronizes in order to validate array elements
    when ``check_finite=True``, and to check the factorization status.

    See Also
    --------
    :func:`dpnp.scipy.linalg.cho_solve` : Solve a linear set equations using
                                          the Cholesky factorization of
                                          a matrix.
    :func:`dpnp.linalg.cholesky` : Cholesky decomposition of a matrix.

    Notes
    -----
    During the finiteness check (if selected), the entire matrix `a` is
    checked. During decomposition, `a` is assumed to be symmetric or Hermitian
    (as applicable), and only the half selected by option `lower` is
    referenced.

    Examples
    --------
    >>> import dpnp as np
    >>> A = np.array([[9., 3., 1., 5.], [3., 7., 5., 1.],
    ...               [1., 5., 9., 2.], [5., 1., 2., 6.]])
    >>> c, low = np.scipy.linalg.cho_factor(A)
    >>> c
    array([[ 3.        ,  1.        ,  0.33333333,  1.66666667],
           [ 0.        ,  2.44948974,  1.90515869, -0.27216553],
           [ 0.        ,  0.        ,  2.29330749,  0.8559528 ],
           [ 0.        ,  0.        ,  0.        ,  1.55418563]])
    >>> np.allclose(c.T @ c, A)
    array(True)

    Batched input:

    >>> B = np.stack([A, 2 * A])
    >>> c, low = np.scipy.linalg.cho_factor(B, lower=True)
    >>> np.allclose(c @ c.mT, B)
    array(True)

    """

    dpnp.check_supported_arrays_type(a)
    assert_stacked_2d(a)

    if a.shape[-1] != a.shape[-2]:
        raise ValueError(
            "Expected a square matrix or batch thereof, "
            f"got a.shape={a.shape}"
        )

    return dpnp_cho_factor(
        a,
        lower=_lower_to_bool(lower),
        overwrite_a=overwrite_a,
        check_finite=check_finite,
    )


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """
    Solve the linear equations ``A x = b``, given the Cholesky factorization
    of `A`.

    For full documentation refer to :obj:`scipy.linalg.cho_solve`.

    Parameters
    ----------
    (c, lower) : tuple, ({dpnp.ndarray, usm_ndarray}, bool)
        Cholesky factorization of `A` (..., M, M), as given by
        :obj:`dpnp.scipy.linalg.cho_factor`. `lower` can also be an array
        or a sequence whose values are all equal.
    b : {(M,), (..., M, K)} {dpnp.ndarray, usm_ndarray}
        Right-hand side.
    overwrite_b : bool, optional
        Whether to overwrite data in `b` (may improve performance).

        Default: ``False``.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

        Default: ``True``.

    Returns
    -------
    x : {(M,), (..., M, K)} dpnp.ndarray
        The solution to the system ``A x = b``.

    Warnings
    --------
    This function synchronizes in order to validate array elements
    when ``check_finite=True``.

    See Also
    --------
    :func:`dpnp.scipy.linalg.cho_factor` : Cholesky factorization of a matrix.

    Examples
    --------
    >>> import dpnp as np
    >>> A = np.array([[9., 3., 1., 5.], [3., 7., 5., 1.],
    ...               [1., 5., 9., 2.], [5., 1., 2., 6.]])
    >>> c, low = np.scipy.linalg.cho_factor(A)
    >>> x = np.scipy.linalg.cho_solve((c, low), np.ones(4))
    >>> x
    array([-0.01749271,  0.11953353,  0.01166181,  0.1574344 ])
    >>> np.allclose(A @ x, np.ones(4))
    array(True)

    """

    c, lower = c_and_lower
    dpnp.check_supported_arrays_type(c, b)
    assert_stacked_2d(c)

    if c.shape[-1] != c.shape[-2]:
        raise ValueError("The factored matrix c is not square.")

    return dpnp_cho_solve(
        c,
        _lower_to_bool(lower),
        b,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )
