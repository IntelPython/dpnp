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
Utility functions for the SciPy-compatible linear algebra interface.

These include helper functions to check array properties and
functions with the main implementation part to fulfill the interface.
The main computational work is performed by enabling LAPACK functions
available as a pybind11 extension.

"""

# pylint: disable=duplicate-code
# pylint: disable=no-name-in-module
# pylint: disable=protected-access

from warnings import warn

import dpctl.tensor._tensor_impl as ti
import dpctl.utils as dpu

import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li
from dpnp.dpnp_utils import get_usm_allocations
from dpnp.linalg.dpnp_utils_linalg import _common_type, _real_type


def _align_lu_solve_broadcast(lu, b):
    """Align LU and RHS batch dimensions with SciPy-like rules."""
    lu_shape = lu.shape
    b_shape = b.shape

    if b.ndim < 2:
        if lu_shape[-2] != b_shape[0]:
            raise ValueError(
                f"Shapes of lu {lu_shape} and b {b_shape} are incompatible"
            )
        b = dpnp.broadcast_to(b, lu_shape[:-1])
        return lu, b

    if lu_shape[-2] != b_shape[-2]:
        raise ValueError(
            f"Shapes of lu {lu_shape} and b {b_shape} are incompatible"
        )

    # Use dpnp.broadcast_shapes() to align the resulting batch shapes
    batch = dpnp.broadcast_shapes(lu_shape[:-2], b_shape[:-2])
    lu_bshape = batch + lu_shape[-2:]
    b_bshape = batch + b_shape[-2:]

    if lu_shape != lu_bshape:
        lu = dpnp.broadcast_to(lu, lu_bshape)
    if b_shape != b_bshape:
        b = dpnp.broadcast_to(b, b_bshape)

    return lu, b


def _apply_permutation_to_rows(mat, perm_indices):
    """
    Apply a permutation to the rows (axis=-2) of a matrix.

    Returns ``out`` such that
    ``out[..., i, :] = mat[..., perm_indices[..., i], :]``.

    For 2-D inputs this is equivalent to ``mat[perm_indices]`` (a single
    device gather).  For batched inputs :func:`dpnp.take_along_axis` is
    used so the operation stays entirely on the device.

    Parameters
    ----------
    mat : dpnp.ndarray, shape (..., M, N)
        Matrix whose rows are to be permuted.
    perm_indices : dpnp.ndarray, shape (..., M)
        Permutation indices (dtype int64).

    Returns
    -------
    out : dpnp.ndarray, shape (..., M, N)
        Row-permuted matrix.
    """

    if perm_indices.ndim == 1:
        # 2-D case: simple fancy indexing, single kernel launch.
        return mat[perm_indices]

    # Batched case: ensure *mat* has the same batch dimensions as
    # *perm_indices*. This is needed, for example, when permuting
    # a shared identity matrix across a batch.
    target_shape = perm_indices.shape[:-1] + mat.shape[-2:]
    if mat.shape != target_shape:
        mat = dpnp.broadcast_to(mat, target_shape)

    # Expand (..., M) → (..., M, 1), then broadcast to the full shape
    # of *mat* so take_along_axis can gather along axis -2.
    idx = dpnp.expand_dims(perm_indices, axis=-1)
    idx = dpnp.broadcast_to(idx, target_shape).copy()
    return dpnp.take_along_axis(mat, idx, axis=-2)


def _batched_lu_factor_scipy(a, res_type):  # pylint: disable=too-many-locals
    """SciPy-compatible LU factorization for batched inputs."""

    # TODO: Find out at which array sizes the best performance is obtained
    # getrf_batch can be slow on large GPU arrays.
    # Use getrf_batch only on CPU.
    # On GPU fall back to calling getrf per 2D slice.
    use_batch = a.sycl_device.has_aspect_cpu

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type
    _manager = dpu.SequentialOrderManager[a_sycl_queue]

    m, n = a.shape[-2:]
    k = min(m, n)
    orig_shape = a.shape
    batch_shape = orig_shape[:-2]

    # handle empty input
    if a.size == 0:
        lu = dpnp.empty_like(a)
        piv = dpnp.empty(
            (*batch_shape, k),
            dtype=dpnp.int64,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        return lu, piv

    # get 3d input arrays by reshape
    a = dpnp.reshape(a, (-1, m, n))
    batch_size = a.shape[0]

    # Move batch axis to the end (m, n, batch) in Fortran order:
    # required by getrf_batch
    # and ensures each a[..., i] is F-contiguous for getrf
    a = dpnp.moveaxis(a, 0, -1)

    a_usm_arr = dpnp.get_usm_ndarray(a)

    # `a` must be copied because getrf/getrf_batch destroys the input matrix
    a_h = dpnp.empty_like(a, order="F", dtype=res_type)
    ht_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr,
        dst=a_h.get_array(),
        sycl_queue=a_sycl_queue,
        depends=_manager.submitted_events,
    )
    _manager.add_event_pair(ht_ev, copy_ev)

    ipiv_h = dpnp.empty(
        (batch_size, k),
        dtype=dpnp.int64,
        order="C",
        usm_type=a_usm_type,
        sycl_queue=a_sycl_queue,
    )

    if use_batch:
        dev_info_h = [0] * batch_size

        ipiv_stride = k
        a_stride = a_h.strides[-1] // a_h.itemsize

        # Call the LAPACK extension function _getrf_batch
        # to perform LU decomposition of a batch of general matrices
        ht_ev, getrf_ev = li._getrf_batch(
            a_sycl_queue,
            a_h.get_array(),
            ipiv_h.get_array(),
            dev_info_h,
            m,
            n,
            a_stride,
            ipiv_stride,
            batch_size,
            depends=[copy_ev],
        )
        _manager.add_event_pair(ht_ev, getrf_ev)

        if any(dev_info_h):
            diag_nums = ", ".join(str(v) for v in dev_info_h if v > 0)
            warn(
                f"Diagonal numbers {diag_nums} are exactly zero. "
                "Singular matrix.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        dev_info_vecs = [[0] for _ in range(batch_size)]

        # Sequential LU factorization using getrf per slice
        for i in range(batch_size):
            ht_ev, getrf_ev = li._getrf(
                a_sycl_queue,
                a_h[..., i].get_array(),
                ipiv_h[i].get_array(),
                dev_info_vecs[i],
                depends=[copy_ev],
            )
            _manager.add_event_pair(ht_ev, getrf_ev)

        diag_nums = ", ".join(
            str(v) for info in dev_info_vecs for v in info if v > 0
        )
        if diag_nums:
            warn(
                f"Diagonal number {diag_nums} are exactly zero. "
                "Singular matrix.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Restore original shape: move batch axis back and reshape
    a_h = dpnp.moveaxis(a_h, -1, 0).reshape(orig_shape)
    ipiv_h = ipiv_h.reshape((*batch_shape, k))

    # oneMKL LAPACK uses 1-origin while SciPy uses 0-origin
    ipiv_h -= 1

    # Return a tuple containing the factorized matrix 'a_h',
    # pivot indices 'ipiv_h'
    return (a_h, ipiv_h)


def _batched_lu_solve(lu, piv, b, res_type, trans=0):
    """Solve a batched equation system (SciPy-compatible behavior)."""
    res_usm_type, exec_q = get_usm_allocations([lu, piv, b])

    b_ndim_orig = b.ndim

    lu, b = _align_lu_solve_broadcast(lu, b)

    n = lu.shape[-1]
    nrhs = b.shape[-1] if b_ndim_orig > 1 else 1

    # get 3d input arrays by reshape
    if lu.ndim > 3:
        lu = dpnp.reshape(lu, (-1, n, n))
    # get 2d pivot arrays by reshape
    if piv.ndim > 2:
        piv = dpnp.reshape(piv, (-1, n))
    batch_size = lu.shape[0]

    # Move batch axis to the end (n, n, batch) in Fortran order:
    # required by getrs_batch
    # and ensures each lu[..., i] is F-contiguous for getrs_batch
    lu = dpnp.moveaxis(lu, 0, -1)

    b_orig_shape = b.shape
    if b.ndim > 3:
        b = dpnp.reshape(b, (-1, n, nrhs))

    # Move batch axis to the end (n, nrhs, batch) in Fortran order:
    # required by getrs_batch
    # and ensures each b[..., i] is F-contiguous for getrs_batch
    b = dpnp.moveaxis(b, 0, -1)

    lu_usm_arr = dpnp.get_usm_ndarray(lu)
    b_usm_arr = dpnp.get_usm_ndarray(b)

    # dpnp.linalg.lu_factor() returns 0-based pivots to match SciPy,
    # convert to 1-based for oneMKL getrs_batch
    piv_h = piv + 1

    _manager = dpu.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events

    # oneMKL LAPACK getrs_batch overwrites `lu`
    lu_h = dpnp.empty_like(lu, order="F", dtype=res_type, usm_type=res_usm_type)

    # use DPCTL tensor function to fill the сopy of the input array
    # from the input array
    ht_ev, lu_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=lu_usm_arr,
        dst=lu_h.get_array(),
        sycl_queue=lu.sycl_queue,
        depends=dep_evs,
    )
    _manager.add_event_pair(ht_ev, lu_copy_ev)

    # oneMKL LAPACK getrs_batch overwrites `b` and assumes fortran-like array
    # as input
    b_h = dpnp.empty_like(b, order="F", dtype=res_type, usm_type=res_usm_type)
    ht_ev, b_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=b_usm_arr,
        dst=b_h.get_array(),
        sycl_queue=b.sycl_queue,
        depends=dep_evs,
    )
    _manager.add_event_pair(ht_ev, b_copy_ev)
    dep_evs = [lu_copy_ev, b_copy_ev]

    lu_stride = n * n
    piv_stride = n
    b_stride = n * nrhs

    trans_mkl = _map_trans_to_mkl(trans)

    # Call the LAPACK extension function _getrs_batch
    # to solve the system of linear equations with an LU-factored
    # coefficient square matrix, with multiple right-hand sides.
    ht_ev, getrs_batch_ev = li._getrs_batch(
        exec_q,
        lu_h.get_array(),
        piv_h.get_array(),
        b_h.get_array(),
        trans_mkl,
        n,
        nrhs,
        lu_stride,
        piv_stride,
        b_stride,
        batch_size,
        depends=dep_evs,
    )
    _manager.add_event_pair(ht_ev, getrs_batch_ev)

    # Restore original shape: move batch axis back and reshape
    b_h = dpnp.moveaxis(b_h, -1, 0).reshape(b_orig_shape)

    return b_h


def _is_copy_required(a, res_type):
    """
    Determine if `a` needs to be copied before LU decomposition.
    This matches SciPy behavior: copy is needed unless input is suitable
    for in-place modification.
    """

    if a.dtype != res_type:
        return True
    if not a.flags["F_CONTIGUOUS"]:
        return True
    if not a.flags["WRITABLE"]:
        return True

    return False


def _map_trans_to_mkl(trans):
    """Map SciPy-style trans code (0,1,2) to oneMKL transpose enum."""
    if not isinstance(trans, int):
        raise TypeError("`trans` must be an integer")

    if trans == 0:
        return li.Transpose.N
    if trans == 1:
        return li.Transpose.T
    if trans == 2:
        return li.Transpose.C
    raise ValueError("`trans` must be 0 (N), 1 (T), or 2 (C)")


def _pivots_to_permutation(piv, m):
    """
    Convert 0-based LAPACK pivot indices (sequential row swaps)
    to a permutation array.

    The returned permutation ``perm`` satisfies ``A[perm] = L @ U``
    (i.e. the forward row-permutation produced by LAPACK).

    The computation is performed entirely on the device.  A host-side
    Python loop of ``K = min(M, N)`` iterations drives the sequential
    swap logic, but each iteration only launches device kernels
    (:func:`dpnp.take_along_axis` for gather,
    :func:`dpnp.put_along_axis` for scatter); **no data is transferred
    between host and device**.

    .. note::

        A future custom SYCL kernel could fuse all ``K`` swap steps
        into a single launch to eliminate per-step kernel overhead.

    Parameters
    ----------
    piv : dpnp.ndarray, shape (..., K)
        0-based pivot indices as returned by :obj:`dpnp_lu_factor`.
    m : int
        Number of rows of the original matrix.

    Returns
    -------
    perm : dpnp.ndarray, shape (..., M), dtype int64
        Permutation indices.
    """

    batch_shape = piv.shape[:-1]
    k = piv.shape[-1]

    # Initialise the identity permutation on the device.
    perm = dpnp.broadcast_to(
        dpnp.arange(
            m,
            dtype=dpnp.int64,
            usm_type=piv.usm_type,
            sycl_queue=piv.sycl_queue,
        ),
        (*batch_shape, m),
    ).copy()

    # Apply sequential row swaps entirely on the device.
    # Each iteration launches a small number of device kernels (gather +
    # slice-assign + scatter) but never transfers data to the host.
    for i in range(k):
        # Pivot target for step *i*: shape (..., 1)
        j = piv[..., i : i + 1]

        # Gather the two values to be swapped.
        val_i = perm[..., i : i + 1].copy()  # slice (free)
        val_j = dpnp.take_along_axis(perm, j, axis=-1)  # gather

        # Perform the swap.
        perm[..., i : i + 1] = val_j  # slice assign
        dpnp.put_along_axis(perm, j, val_i, axis=-1)  # scatter

    return perm


def dpnp_lu_factor(a, overwrite_a=False, check_finite=True):
    """
    dpnp_lu_factor(a, overwrite_a=False, check_finite=True)

    Compute pivoted LU decomposition (SciPy-compatible behavior).

    This function mimics the behavior of `scipy.linalg.lu_factor` including
    support for `overwrite_a`, `check_finite` and 0-based pivot indexing.

    """

    res_type = _common_type(a)
    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    if check_finite:
        if not dpnp.isfinite(a).all():
            raise ValueError("array must not contain infs or NaNs")

    if a.ndim > 2:
        # SciPy always copies each 2D slice,
        # so `overwrite_a` is ignored here
        return _batched_lu_factor_scipy(a, res_type)

    # accommodate empty arrays
    if a.size == 0:
        lu = dpnp.empty_like(a)
        piv = dpnp.arange(
            0, dtype=dpnp.int64, usm_type=a_usm_type, sycl_queue=a_sycl_queue
        )
        return lu, piv

    _manager = dpu.SequentialOrderManager[a_sycl_queue]
    a_usm_arr = dpnp.get_usm_ndarray(a)

    # SciPy-compatible behavior
    # Copy is required if:
    # - overwrite_a is False (always copy),
    # - dtype mismatch,
    # - not F-contiguous,
    # - not writeable
    if not overwrite_a or _is_copy_required(a, res_type):
        a_h = dpnp.empty_like(a, order="F", dtype=res_type)
        ht_ev, dep_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_usm_arr,
            dst=a_h.get_array(),
            sycl_queue=a_sycl_queue,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht_ev, dep_ev)
        dep_ev = [dep_ev]
    else:
        # input is suitable for in-place modification
        a_h = a
        dep_ev = _manager.submitted_events

    m, n = a.shape

    ipiv_h = dpnp.empty(
        min(m, n),
        dtype=dpnp.int64,
        order="C",
        usm_type=a_usm_type,
        sycl_queue=a_sycl_queue,
    )
    dev_info_h = [0]

    # Call the LAPACK extension function _getrf
    # to perform LU decomposition on the input matrix
    ht_ev, getrf_ev = li._getrf(
        a_sycl_queue,
        a_h.get_array(),
        ipiv_h.get_array(),
        dev_info_h,
        depends=dep_ev,
    )
    _manager.add_event_pair(ht_ev, getrf_ev)

    if any(dev_info_h):
        diag_nums = ", ".join(str(v) for v in dev_info_h if v > 0)
        warn(
            f"Diagonal number {diag_nums} is exactly zero. Singular matrix.",
            RuntimeWarning,
            stacklevel=2,
        )

    # MKL lapack uses 1-origin while SciPy uses 0-origin
    ipiv_h -= 1

    # Return a tuple containing the factorized matrix 'a_h',
    # pivot indices 'ipiv_h'
    return (a_h, ipiv_h)


def _assemble_lu_output(
    low,
    up,
    inv_perm,
    permute_l,
    p_indices,
    m,
    real_type,
    a_usm_type,
    a_sycl_queue,
):
    """Select and build the correct dpnp_lu return value."""
    if permute_l:
        return _apply_permutation_to_rows(low, inv_perm), up
    if p_indices:
        return inv_perm, low, up
    eye_m = dpnp.eye(
        m, dtype=real_type, usm_type=a_usm_type, sycl_queue=a_sycl_queue
    )
    return (
        _apply_permutation_to_rows(eye_m, inv_perm),
        low,
        up,
    )  # perm_matrix, L, U


def dpnp_lu(
    a,
    overwrite_a=False,
    check_finite=True,
    p_indices=False,
    permute_l=False,
):
    """
    dpnp_lu(a, overwrite_a=False, check_finite=True, p_indices=False,
            permute_l=False)

    Compute pivoted LU decomposition and return separate P, L, U matrices
    (SciPy-compatible behavior).

    This function mimics the behavior of `scipy.linalg.lu` including
    support for `permute_l`, `p_indices`, `overwrite_a`, and `check_finite`.

    """

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type
    m, n = a.shape[-2:]
    k = min(m, n)
    batch_shape = a.shape[:-2]

    res_type = _common_type(a)

    # The permutation matrix P uses a real dtype (SciPy convention):
    # P only contains 0s and 1s, so complex storage would be wasteful.
    real_type = _real_type(res_type)

    # ---- Fast path: scalar (1x1) matrices ----
    # For 1x1 input, P = I, L = I, U = A.  This avoids invoking LAPACK
    # entirely (matches SciPy's scalar fast path).
    if m == 1 and n == 1:
        if check_finite:
            if not dpnp.isfinite(a).all():
                raise ValueError("array must not contain infs or NaNs")

        low = dpnp.ones_like(a, dtype=res_type)
        up = dpnp.astype(a, res_type, copy=not overwrite_a)
        inv_perm = dpnp.zeros_like(a, shape=(*batch_shape, 1), dtype=dpnp.int64)

        return _assemble_lu_output(
            low,
            up,
            inv_perm,
            permute_l,
            p_indices,
            m,
            real_type,
            a_usm_type,
            a_sycl_queue,
        )

    # ---- Fast path: empty arrays ----
    if a.size == 0:
        low = dpnp.empty_like(a, shape=(*batch_shape, m, k), dtype=res_type)
        up = dpnp.empty_like(a, shape=(*batch_shape, k, n), dtype=res_type)
        inv_perm = dpnp.empty_like(a, shape=(*batch_shape, m), dtype=dpnp.int64)
        return _assemble_lu_output(
            low,
            up,
            inv_perm,
            permute_l,
            p_indices,
            m,
            real_type,
            a_usm_type,
            a_sycl_queue,
        )

    # ---- General case: LAPACK factorization ----
    lu_compact, piv = dpnp_lu_factor(
        a, overwrite_a=overwrite_a, check_finite=check_finite
    )

    # ---- Extract L: lower-triangular with unit diagonal ----
    # L has shape (..., M, K).
    low = dpnp.tril(lu_compact[..., :, :k], k=-1)
    low += dpnp.eye(
        m,
        k,
        dtype=lu_compact.dtype,
        usm_type=a_usm_type,
        sycl_queue=a_sycl_queue,
    )

    # ---- Extract U: upper-triangular ----
    # U has shape (..., K, N).
    up = dpnp.triu(lu_compact[..., :k, :])

    # ---- Convert pivot indices → row permutation ----
    # ``perm`` (forward): A[perm] = L @ U.
    # This is the only step that requires a host transfer because the
    # sequential swap semantics of LAPACK pivots cannot be parallelised.
    # Only the small pivot array (min(M, N) elements per slice) is
    # transferred; all subsequent work stays on the device.
    perm = _pivots_to_permutation(piv, m)

    # ``inv_perm`` (inverse): A = L[inv_perm] @ U.
    # This is SciPy's ``p_indices`` convention.
    # ``dpnp.argsort`` is an efficient on-device O(M log M) operation
    # that avoids a second host round-trip.
    inv_perm = dpnp.argsort(perm, axis=-1).astype(dpnp.int64)

    # ---- Assemble output (SciPy convention) ----
    return _assemble_lu_output(
        low,
        up,
        inv_perm,
        permute_l,
        p_indices,
        m,
        real_type,
        a_usm_type,
        a_sycl_queue,
    )


def dpnp_lu_solve(lu, piv, b, trans=0, overwrite_b=False, check_finite=True):
    """
    dpnp_lu_solve(lu, piv, b, trans=0, overwrite_b=False, check_finite=True)

    Solve an equation system (SciPy-compatible behavior).

    This function mimics the behavior of `scipy.linalg.lu_solve` including
    support for `trans`, `overwrite_b`, `check_finite`,
    and 0-based pivot indexing.

    """

    res_usm_type, exec_q = get_usm_allocations([lu, piv, b])

    res_type = _common_type(lu, b)

    if b.size == 0:
        return dpnp.empty_like(b, dtype=res_type, usm_type=res_usm_type)

    if check_finite:
        if not dpnp.isfinite(lu).all():
            raise ValueError(
                "LU factorization array must not contain infs or NaNs.\n"
                "Note that when a singular matrix is given, unlike "
                "dpnp.scipy.linalg.lu_factor returns an array containing NaN."
            )
        if not dpnp.isfinite(b).all():
            raise ValueError(
                "Right-hand side array must not contain infs or NaNs"
            )

    if lu.ndim > 2:
        # SciPy always copies each 2D slice,
        # so `overwrite_b` is ignored here
        return _batched_lu_solve(lu, piv, b, trans=trans, res_type=res_type)

    if lu.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shapes of lu {lu.shape} and b {b.shape} are incompatible"
        )

    lu_usm_arr = dpnp.get_usm_ndarray(lu)
    b_usm_arr = dpnp.get_usm_ndarray(b)

    # dpnp.scipy.linalg.lu_factor() returns 0-based pivots to match SciPy,
    # convert to 1-based for oneMKL getrs
    piv_h = piv + 1

    _manager = dpu.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events

    # oneMKL LAPACK getrs_batch overwrites `lu`.
    lu_h = dpnp.empty_like(lu, order="F", dtype=res_type, usm_type=res_usm_type)

    # use DPCTL tensor function to fill the сopy of the input array
    # from the input array
    ht_ev, lu_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=lu_usm_arr,
        dst=lu_h.get_array(),
        sycl_queue=lu.sycl_queue,
        depends=dep_evs,
    )
    _manager.add_event_pair(ht_ev, lu_copy_ev)

    # SciPy-compatible behavior
    # Copy is required if:
    # - overwrite_b is False (always copy),
    # - dtype mismatch,
    # - not F-contiguous,
    # - not writeable
    if not overwrite_b or _is_copy_required(b, res_type):
        b_h = dpnp.empty_like(
            b, order="F", dtype=res_type, usm_type=res_usm_type
        )
        ht_ev, b_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=b_usm_arr,
            dst=b_h.get_array(),
            sycl_queue=b.sycl_queue,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_ev, b_copy_ev)
        dep_evs = [lu_copy_ev, b_copy_ev]
    else:
        # input is suitable for in-place modification
        b_h = b
        dep_evs = [lu_copy_ev]

    trans_mkl = _map_trans_to_mkl(trans)

    # Call the LAPACK extension function _getrs
    # to solve the system of linear equations with an LU-factored
    # coefficient square matrix, with multiple right-hand sides.
    ht_ev, getrs_ev = li._getrs(
        exec_q,
        lu_h.get_array(),
        piv_h.get_array(),
        b_h.get_array(),
        trans_mkl,
        depends=dep_evs,
    )
    _manager.add_event_pair(ht_ev, getrs_ev)

    return b_h
