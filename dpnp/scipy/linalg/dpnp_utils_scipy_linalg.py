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
Utility functions for the SciPy-compatible linear algebra interface.

These include helper functions to check array properties and
functions with the main implementation part to fulfill the interface.
The main computational work is performed by enabling LAPACK functions
available as a pybind11 extension.

"""


# pylint: disable=no-name-in-module
# pylint: disable=protected-access

from warnings import warn

import dpctl.tensor._tensor_impl as ti
import dpctl.utils as dpu

import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li
from dpnp.dpnp_utils import get_usm_allocations
from dpnp.linalg.dpnp_utils_linalg import _common_type

__all__ = [
    "dpnp_lu_factor",
    "dpnp_lu_solve",
]


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
        a_stride = a_h.strides[-1]

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

    # TODO: add broadcasting
    if lu.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shapes of lu {lu.shape} and b {b.shape} are incompatible"
        )

    if b.size == 0:
        return dpnp.empty_like(b, dtype=res_type, usm_type=res_usm_type)

    if lu.ndim > 2:
        raise NotImplementedError("Batched matrices are not supported")

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

    lu_usm_arr = dpnp.get_usm_ndarray(lu)
    b_usm_arr = dpnp.get_usm_ndarray(b)

    # dpnp.scipy.linalg.lu_factor() returns 0-based pivots to match SciPy,
    # convert to 1-based for oneMKL getrs
    piv_h = piv + 1

    _manager = dpu.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events

    # oneMKL LAPACK getrs overwrites `lu`.
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

    if not isinstance(trans, int):
        raise TypeError("`trans` must be an integer")

    # Map SciPy-style trans codes (0, 1, 2) to MKL transpose enums
    if trans == 0:
        trans_mkl = li.Transpose.N
    elif trans == 1:
        trans_mkl = li.Transpose.T
    elif trans == 2:
        trans_mkl = li.Transpose.C
    else:
        raise ValueError("`trans` must be 0 (N), 1 (T), or 2 (C)")

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
