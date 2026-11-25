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

import operator

import dpnp

from ._decomp_cholesky import _lower_to_bool
from ._utils import dpnp_solve_triangular


def _trans_to_int(trans):
    """Convert `trans` of solve_triangular to 0 (N), 1 (T) or 2 (C)."""

    if isinstance(trans, str):
        res = {"N": 0, "T": 1, "C": 2}.get(trans)
    else:
        try:
            res = operator.index(trans)
        except TypeError:
            res = None

    if res not in (0, 1, 2):
        raise ValueError(
            f"`trans` must be 0, 1, 2, 'N', 'T' or 'C', got {trans!r}."
        )
    return res


def solve_triangular(
    a,
    b,
    trans=0,
    lower=False,
    unit_diagonal=False,
    overwrite_b=False,
    check_finite=True,
):
    """
    Solve the equation ``a x = b`` for `x`, assuming `a` is a triangular
    matrix.

    For full documentation refer to :obj:`scipy.linalg.solve_triangular`.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        A triangular matrix.
    b : {(M,), (..., M, N)} {dpnp.ndarray, usm_ndarray}
        Right-hand side matrix in ``a x = b``.
    trans : {0, 1, 2, 'N', 'T', 'C'}, optional
        Type of system to solve:

        ========  =========
        trans     system
        ========  =========
        0 or 'N'  a x = b
        1 or 'T'  a^T x = b
        2 or 'C'  a^H x = b
        ========  =========

        Default: ``0``.
    lower : {bool, array_like of bool}, optional
        Use only data contained in the lower triangle of `a`, instead of the
        upper one. An array or a sequence is accepted if all its values are
        equal.

        Default: ``False``.
    unit_diagonal : bool, optional
        If ``True``, diagonal elements of `a` are assumed to be 1 and will not
        be referenced.

        Default: ``False``.
    overwrite_b : bool, optional
        Allow overwriting data in `b` (may enhance performance).

        Default: ``False``.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

        Default: ``True``.

    Returns
    -------
    x : {(M,), (..., M, N)} dpnp.ndarray
        Solution to the system ``a x = b``. Its shape is the shape of `b`,
        with the batch dimensions of `a` and `b` broadcast.

    Raises
    ------
    dpnp.linalg.LinAlgError
        If `a` is singular: it has a zero on the diagonal and
        ``unit_diagonal=False``.
    ValueError
        If `a` is not square, if `b` is 0-D, if the shapes of `a` and `b` are
        incompatible, if `trans` is none of the values above, if `lower` is
        empty, not bool or its values are not all equal, or if
        ``check_finite=True`` and `a` or `b` contains infinities or NaNs.

    Warnings
    --------
    This function synchronizes in order to check the diagonal of `a` for
    zeros unless ``unit_diagonal=True``, to validate array elements when
    ``check_finite=True``, and to read `lower` or `unit_diagonal` given as
    a dpnp array.

    See Also
    --------
    :func:`dpnp.scipy.linalg.cho_solve` : Solve a linear set equations using
                                          the Cholesky factorization of
                                          a matrix.
    :func:`dpnp.scipy.linalg.lu_solve` : Solve a linear set equations using
                                         the LU factorization of a matrix.
    :func:`dpnp.linalg.solve` : Solve a linear matrix equation.

    Notes
    -----
    Unlike SciPy:

    - a zero-size batch is supported and gives an empty solution;
    - a `trans` other than the values above raises ``ValueError``, while
      SciPy may solve the transposed system for it;
    - integer and boolean inputs are treated as arrays of the default
      floating point data type of the device, as in the other linear algebra
      functions of DPNP, with no ``DeprecationWarning`` for boolean input.

    Examples
    --------
    Solve the lower triangular system ``a x = b``, where::

             [3  0  0  0]       [4]
        a =  [2  1  0  0]   b = [2]
             [1  0  1  0]       [4]
             [1  1  1  1]       [2]

    >>> import dpnp as np
    >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    >>> b = np.array([4, 2, 4, 2])
    >>> x = np.scipy.linalg.solve_triangular(a, b, lower=True)
    >>> x
    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
    >>> a @ x  # Check the result
    array([4., 2., 4., 2.])

    """

    dpnp.check_supported_arrays_type(a, b)

    if a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        raise ValueError("expected square matrix")
    if b.ndim < 1:
        raise ValueError("expected `b` to have at least one dimension")

    return dpnp_solve_triangular(
        a,
        b,
        trans=_trans_to_int(trans),
        lower=_lower_to_bool(lower),
        unit_diagonal=bool(unit_diagonal),
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )
