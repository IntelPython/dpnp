# *****************************************************************************
# Copyright (c) 2023-2024, Intel Corporation
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
Helping functions to implement the Linear Algebra interface.

These include assetion functions to validate input arrays and
functions with the main implementation part to fulfill the interface.
The main computational work is performed by enabling LAPACK functions
available as a pybind11 extension.

"""

# pylint: disable=invalid-name
# pylint: disable=no-name-in-module
# pylint: disable=protected-access
# pylint: disable=useless-import-alias

import dpctl
import dpctl.tensor._tensor_impl as ti
import numpy
from numpy import prod
from numpy.core.numeric import normalize_axis_index

import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li
from dpnp.dpnp_utils import get_usm_allocations
from dpnp.linalg import LinAlgError as LinAlgError

__all__ = [
    "assert_2d",
    "assert_stacked_2d",
    "assert_stacked_square",
    "dpnp_cholesky",
    "dpnp_cond",
    "dpnp_det",
    "dpnp_eigh",
    "dpnp_inv",
    "dpnp_lstsq",
    "dpnp_matrix_power",
    "dpnp_matrix_rank",
    "dpnp_multi_dot",
    "dpnp_norm",
    "dpnp_pinv",
    "dpnp_qr",
    "dpnp_slogdet",
    "dpnp_solve",
    "dpnp_svd",
]

_jobz = {"N": 0, "V": 1}
_upper_lower = {"U": 0, "L": 1}

_real_types_map = {
    "float32": "float32",  # single : single
    "float64": "float64",  # double : double
    "complex64": "float32",  # csingle : csingle
    "complex128": "float64",  # cdouble : cdouble
}


def _batched_eigh(a, UPLO, eigen_mode, w_type, v_type):
    """
    _batched_eigh(a, UPLO, eigen_mode, w_type, v_type)

    Return the eigenvalues and eigenvectors of each matrix in a batch of
    a complex Hermitian (conjugate symmetric) or a real symmetric matrix.
    Can return both eigenvalues and eigenvectors (`eigen_mode="V"`) or
    only eigenvalues (`eigen_mode="N"`).

    The main calculation is done by calling an extension function
    for LAPACK library of OneMKL. Depending on input type of `a` array,
    it will be either ``heevd`` (for complex types) or ``syevd`` (for others).

    """

    is_cpu_device = a.sycl_device.has_aspect_cpu
    orig_shape = a.shape
    # get 3d input array by reshape
    a = a.reshape(-1, orig_shape[-2], orig_shape[-1])
    a_usm_arr = dpnp.get_usm_ndarray(a)

    # allocate a memory for dpnp array of eigenvalues
    w = dpnp.empty_like(
        a,
        shape=orig_shape[:-1],
        dtype=w_type,
    )
    w_orig_shape = w.shape
    # get 2d dpnp array with eigenvalues by reshape
    w = w.reshape(-1, w_orig_shape[-1])

    # `eigen_mode` can be either "N" or "V", specifying the computation mode
    # for OneMKL LAPACK `syevd` and `heevd` routines.
    # "V" (default) means both eigenvectors and eigenvalues will be calculated
    # "N" means only eigenvalues will be calculated
    jobz = _jobz[eigen_mode]
    uplo = _upper_lower[UPLO]

    # Get LAPACK function (_syevd for real or _heevd for complex data types)
    # to compute all eigenvalues and, optionally, all eigenvectors
    lapack_func = (
        "_heevd" if dpnp.issubdtype(v_type, dpnp.complexfloating) else "_syevd"
    )

    a_sycl_queue = a.sycl_queue
    a_order = "C" if a.flags.c_contiguous else "F"

    # need to loop over the 1st dimension to get eigenvalues and
    # eigenvectors of 3d matrix A
    batch_size = a.shape[0]
    eig_vecs = [None] * batch_size
    ht_list_ev = [None] * batch_size * 2
    for i in range(batch_size):
        # oneMKL LAPACK assumes fortran-like array as input, so
        # allocate a memory with 'F' order for dpnp array of eigenvectors
        eig_vecs[i] = dpnp.empty_like(a[i], order="F", dtype=v_type)

        # use DPCTL tensor function to fill the array of eigenvectors with
        # content of input array
        ht_list_ev[2 * i], copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_usm_arr[i],
            dst=eig_vecs[i].get_array(),
            sycl_queue=a_sycl_queue,
        )

        # TODO: Remove this w/a when MKLD-17201 is solved.
        # Waiting for a host task executing an OneMKL LAPACK syevd/heevd call
        # on CPU causes deadlock due to serialization of all host tasks
        # in the queue.
        # We need to wait for each host tasks before calling _seyvd and _heevd
        # to avoid deadlock.
        if is_cpu_device:
            ht_list_ev[2 * i].wait()

        # call LAPACK extension function to get eigenvalues and
        # eigenvectors of a portion of matrix A
        ht_list_ev[2 * i + 1], _ = getattr(li, lapack_func)(
            a_sycl_queue,
            jobz,
            uplo,
            eig_vecs[i].get_array(),
            w[i].get_array(),
            depends=[copy_ev],
        )

    dpctl.SyclEvent.wait_for(ht_list_ev)

    w = w.reshape(w_orig_shape)

    if eigen_mode == "V":
        # combine the list of eigenvectors into a single array
        v = dpnp.array(eig_vecs, order=a_order).reshape(orig_shape)
        return w, v
    return w


def _batched_inv(a, res_type):
    """
    _batched_inv(a, res_type)

    Return the inverses of each matrix in a batch of matrices `a`.

    The inverse of a matrix is such that if it is multiplied by the original
    matrix, it results in the identity matrix. This function computes the
    inverses of a batch of square matrices.

    """

    orig_shape = a.shape
    # get 3d input arrays by reshape
    a = a.reshape(-1, orig_shape[-2], orig_shape[-1])
    batch_size = a.shape[0]
    a_usm_arr = dpnp.get_usm_ndarray(a)
    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type
    n = a.shape[1]

    # oneMKL LAPACK getri_batch overwrites `a`
    a_h = dpnp.empty_like(a, order="C", dtype=res_type, usm_type=a_usm_type)
    ipiv_h = dpnp.empty(
        (batch_size, n),
        dtype=dpnp.int64,
        usm_type=a_usm_type,
        sycl_queue=a_sycl_queue,
    )
    dev_info = [0] * batch_size

    # use DPCTL tensor function to fill the matrix array
    # with content from the input array `a`
    a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr, dst=a_h.get_array(), sycl_queue=a.sycl_queue
    )

    ipiv_stride = n
    a_stride = a_h.strides[0]

    # Call the LAPACK extension function _getrf_batch
    # to perform LU decomposition of a batch of general matrices
    ht_getrf_ev, getrf_ev = li._getrf_batch(
        a_sycl_queue,
        a_h.get_array(),
        ipiv_h.get_array(),
        dev_info,
        n,
        a_stride,
        ipiv_stride,
        batch_size,
        [a_copy_ev],
    )

    _check_lapack_dev_info(dev_info)

    # Call the LAPACK extension function _getri_batch
    # to compute the inverse of a batch of matrices using the results
    # from the LU decomposition performed by _getrf_batch
    ht_getri_ev, _ = li._getri_batch(
        a_sycl_queue,
        a_h.get_array(),
        ipiv_h.get_array(),
        dev_info,
        n,
        a_stride,
        ipiv_stride,
        batch_size,
        [getrf_ev],
    )

    _check_lapack_dev_info(dev_info)

    ht_getri_ev.wait()
    ht_getrf_ev.wait()
    a_ht_copy_ev.wait()

    return a_h.reshape(orig_shape)


def _batched_solve(a, b, exec_q, res_usm_type, res_type):
    """
    _batched_solve(a, b, exec_q, res_usm_type, res_type)

    Return the solution to the system of linear equations of each square
    coefficient matrix in a batch of matrices `a` and multiple dependent
    variables array `b`.

    """

    a_usm_arr = dpnp.get_usm_ndarray(a)
    b_usm_arr = dpnp.get_usm_ndarray(b)

    b_order = "C" if b.flags.c_contiguous else "F"
    a_shape = a.shape
    b_shape = b.shape

    is_cpu_device = exec_q.sycl_device.has_aspect_cpu
    reshape = False
    orig_shape_b = b_shape
    if a.ndim > 3:
        # get 3d input arrays by reshape
        if a.ndim == b.ndim:
            b = b.reshape(-1, b_shape[-2], b_shape[-1])
        else:
            b = b.reshape(-1, b_shape[-1])

        a = a.reshape(-1, a_shape[-2], a_shape[-1])

        a_usm_arr = dpnp.get_usm_ndarray(a)
        b_usm_arr = dpnp.get_usm_ndarray(b)
        reshape = True

    batch_size = a.shape[0]

    coeff_vecs = [None] * batch_size
    val_vecs = [None] * batch_size
    a_ht_copy_ev = [None] * batch_size
    b_ht_copy_ev = [None] * batch_size
    ht_lapack_ev = [None] * batch_size

    for i in range(batch_size):
        # oneMKL LAPACK assumes fortran-like array as input, so allocate
        # a memory with 'F' order for dpnp array of coefficient matrix
        coeff_vecs[i] = dpnp.empty_like(
            a[i], order="F", dtype=res_type, usm_type=res_usm_type
        )

        # use DPCTL tensor function to fill the coefficient matrix array
        # with content from the input array
        a_ht_copy_ev[i], a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_usm_arr[i],
            dst=coeff_vecs[i].get_array(),
            sycl_queue=a.sycl_queue,
        )

        # oneMKL LAPACK assumes fortran-like array as input, so
        # allocate a memory with 'F' order for dpnp array of multiple
        # dependent variables array
        val_vecs[i] = dpnp.empty_like(
            b[i], order="F", dtype=res_type, usm_type=res_usm_type
        )

        # use DPCTL tensor function to fill the array of multiple dependent
        # variables with content from the input arrays
        b_ht_copy_ev[i], b_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=b_usm_arr[i],
            dst=val_vecs[i].get_array(),
            sycl_queue=b.sycl_queue,
        )

        # Call the LAPACK extension function _gesv to solve the system of
        # linear equations using a portion of the coefficient square matrix
        # and a corresponding portion of the dependent variables array.
        ht_lapack_ev[i], _ = li._gesv(
            exec_q,
            coeff_vecs[i].get_array(),
            val_vecs[i].get_array(),
            depends=[a_copy_ev, b_copy_ev],
        )

        # TODO: Remove this w/a when MKLD-17201 is solved.
        # Waiting for a host task executing an OneMKL LAPACK gesv call
        # on CPU causes deadlock due to serialization of all host tasks
        # in the queue.
        # We need to wait for each host tasks before calling _gesv to avoid
        # deadlock.
        if is_cpu_device:
            ht_lapack_ev[i].wait()
            b_ht_copy_ev[i].wait()

    for i in range(batch_size):
        ht_lapack_ev[i].wait()
        b_ht_copy_ev[i].wait()
        a_ht_copy_ev[i].wait()

    # combine the list of solutions into a single array
    out_v = dpnp.array(
        val_vecs, order=b_order, dtype=res_type, usm_type=res_usm_type
    )
    if reshape:
        # shape of the out_v must be equal to the shape of the array of
        # dependent variables
        out_v = out_v.reshape(orig_shape_b)
    return out_v


def _batched_qr(a, mode="reduced"):
    """
    _batched_qr(a, mode="reduced")

    Return the batched qr factorization of `a` matrix.

    """

    m, n = a.shape[-2:]
    k = min(m, n)

    batch_shape = a.shape[:-2]
    batch_size = prod(batch_shape)

    res_type = _common_type(a)

    if batch_size == 0 or k == 0:
        return _zero_batched_qr(a, mode, m, n, k, res_type)

    a_sycl_queue = a.sycl_queue

    # get 3d input arrays by reshape
    a = a.reshape(-1, m, n)

    a = a.swapaxes(-2, -1)
    a_usm_arr = dpnp.get_usm_ndarray(a)

    a_t = dpnp.empty_like(a, order="C", dtype=res_type)

    # use DPCTL tensor function to fill the matrix array
    # with content from the input array `a`
    a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr, dst=a_t.get_array(), sycl_queue=a_sycl_queue
    )

    tau_h = dpnp.empty_like(
        a_t,
        shape=(batch_size, k),
        dtype=res_type,
    )

    a_stride = a_t.strides[0]
    tau_stride = tau_h.strides[0]

    # Call the LAPACK extension function _geqrf_batch to compute
    # the QR factorization of a general m x n matrix.
    ht_geqrf_batch_ev, geqrf_batch_ev = li._geqrf_batch(
        a_sycl_queue,
        a_t.get_array(),
        tau_h.get_array(),
        m,
        n,
        a_stride,
        tau_stride,
        batch_size,
        [a_copy_ev],
    )

    ht_list_ev = [ht_geqrf_batch_ev, a_ht_copy_ev]

    if mode in ["r", "raw"]:
        if mode == "r":
            r = a_t[..., :k].swapaxes(-2, -1)
            r = _triu_inplace(r, ht_list_ev, [geqrf_batch_ev])
            dpctl.SyclEvent.wait_for(ht_list_ev)
            return r.reshape(batch_shape + r.shape[-2:])

        # mode=="raw"
        dpctl.SyclEvent.wait_for(ht_list_ev)
        q = a_t.reshape(batch_shape + a_t.shape[-2:])
        r = tau_h.reshape(batch_shape + tau_h.shape[-1:])
        return (q, r)

    if mode == "complete" and m > n:
        mc = m
        q = dpnp.empty_like(
            a_t,
            shape=(batch_size, m, m),
            dtype=res_type,
        )
    else:
        mc = k
        q = dpnp.empty_like(
            a_t,
            shape=(batch_size, n, m),
            dtype=res_type,
        )

    # use DPCTL tensor function to fill the matrix array `q[..., :n, :]`
    # with content from the array `a_t` overwritten by geqrf_batch
    a_t_ht_copy_ev, a_t_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_t.get_array(),
        dst=q[..., :n, :].get_array(),
        sycl_queue=a_sycl_queue,
        depends=[geqrf_batch_ev],
    )

    ht_list_ev.append(a_t_ht_copy_ev)

    q_stride = q.strides[0]
    tau_stride = tau_h.strides[0]

    # Get LAPACK function (_orgqr_batch for real or _ungqf_batch for complex
    # data types) for QR factorization
    lapack_func = (
        "_ungqr_batch"
        if dpnp.issubdtype(res_type, dpnp.complexfloating)
        else "_orgqr_batch"
    )

    # Call the LAPACK extension function _orgqr_batch/ to generate the real
    # orthogonal/complex unitary matrices `Qi` of the QR factorization
    # for a batch of general matrices.
    ht_lapack_ev, lapack_ev = getattr(li, lapack_func)(
        a_sycl_queue,
        q.get_array(),
        tau_h.get_array(),
        m,
        mc,
        k,
        q_stride,
        tau_stride,
        batch_size,
        [a_t_copy_ev],
    )

    ht_list_ev.append(ht_lapack_ev)

    q = q[..., :mc, :].swapaxes(-2, -1)
    r = a_t[..., :mc].swapaxes(-2, -1)

    ht_list_ev.append(ht_lapack_ev)

    r = _triu_inplace(r, ht_list_ev, [lapack_ev])
    dpctl.SyclEvent.wait_for(ht_list_ev)

    return (
        q.reshape(batch_shape + q.shape[-2:]),
        r.reshape(batch_shape + r.shape[-2:]),
    )


def _batched_svd(
    a, uv_type, s_type, full_matrices=True, compute_uv=True, related_arrays=None
):
    """
    _batched_svd(
        a,
        uv_type,
        s_type,
        full_matrices=True,
        compute_uv=True,
        related_arrays=None
    )

    Return the batched singular value decomposition (SVD) of a stack
    of matrices.

    """

    # Set USM type and SYCL queue to be used based on `a`
    # and optionally provided `related_arrays`.
    # If `related_arrays` is not provided, default to USM type and SYCL queue
    # of `a`.
    # Otherwise, determine USM type and SYCL queue using
    # compute-follows-data execution model for `a` and `related arrays`.
    usm_type, exec_q = get_usm_allocations([a] + (related_arrays or []))

    reshape = False
    batch_shape_orig = a.shape[:-2]

    if a.ndim > 3:
        # get 3d input arrays by reshape
        a = a.reshape(prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        reshape = True

    batch_size = a.shape[0]
    if batch_size == 0:
        return _zero_batched_svd(
            a,
            uv_type,
            s_type,
            full_matrices,
            compute_uv,
            exec_q,
            usm_type,
            batch_shape_orig,
        )

    m, n = a.shape[-2:]
    if m == 0 or n == 0:
        return _zero_m_n_batched_svd(
            a,
            uv_type,
            s_type,
            full_matrices,
            compute_uv,
            exec_q,
            usm_type,
            batch_shape_orig,
        )

    u_matrices = [None] * batch_size
    s_matrices = [None] * batch_size
    vt_matrices = [None] * batch_size
    ht_list_ev = [None] * batch_size * 2
    for i in range(batch_size):
        if compute_uv:
            (
                u_matrices[i],
                s_matrices[i],
                vt_matrices[i],
                ht_list_ev[2 * i],
                ht_list_ev[2 * i + 1],
            ) = dpnp_svd(a[i], full_matrices, compute_uv=True, batch_call=True)
        else:
            s_matrices[i], ht_list_ev[2 * i], ht_list_ev[2 * i + 1] = dpnp_svd(
                a[i], full_matrices, compute_uv=False, batch_call=True
            )

    dpctl.SyclEvent.wait_for(ht_list_ev)

    # TODO: Need to return C-contiguous array to match the output of
    # numpy.linalg.svd
    # Allocate 'F' order memory for dpnp output arrays to be aligned with
    # dpnp_svd
    out_s = dpnp.array(s_matrices, order="F")
    if reshape:
        out_s = out_s.reshape(batch_shape_orig + out_s.shape[-1:])

    if compute_uv:
        out_u = dpnp.array(u_matrices, order="F")
        out_vt = dpnp.array(vt_matrices, order="F")
        if reshape:
            return (
                out_u.reshape(batch_shape_orig + out_u.shape[-2:]),
                out_s,
                out_vt.reshape(batch_shape_orig + out_vt.shape[-2:]),
            )
        return out_u, out_s, out_vt
    return out_s


def _calculate_determinant_sign(ipiv, diag, res_type, n):
    """
    Calculate the sign of the determinant based on row exchanges and diagonal
    values.

    Parameters
    -----------
    ipiv : {dpnp.ndarray, usm_ndarray}
        The pivot indices from LU decomposition.
    diag : {dpnp.ndarray, usm_ndarray}
        The diagonal elements of the LU decomposition matrix.
    res_type : dpnp.dtype
        The common data type for linalg operations.
    n : int
        The size of the last two dimensions of the array.

    Returns
    -------
    sign : {dpnp_array, usm_ndarray}
        The sign of the determinant.

    """

    # Checks for row exchanges in LU decomposition affecting determinant sign.
    ipiv_diff = ipiv != dpnp.arange(
        1, n + 1, usm_type=ipiv.usm_type, sycl_queue=ipiv.sycl_queue
    )

    # Counts row exchanges from 'ipiv_diff'.
    non_zero = dpnp.count_nonzero(ipiv_diff, axis=-1)

    # For floating types, adds count of negative diagonal elements
    # to determine determinant sign.
    if dpnp.issubdtype(res_type, dpnp.floating):
        non_zero += dpnp.count_nonzero(diag < 0, axis=-1)

    sign = (non_zero % 2) * -2 + 1

    # For complex types, compute sign from the phase of diagonal elements.
    if dpnp.issubdtype(res_type, dpnp.complexfloating):
        sign = sign * dpnp.prod(diag / dpnp.abs(diag), axis=-1)

    return sign.astype(res_type)


def _check_lapack_dev_info(dev_info, error_msg=None):
    """
    Check `dev_info` from OneMKL LAPACK routines, raising an error for failures.

    Parameters
    ----------
    dev_info : list of ints
        Each element of the list indicates the status of OneMKL LAPACK routine
        calls. A non-zero value signifies a failure.

    error_message : str, optional
        Custom error message for detected LAPACK errors.
        Default: `Singular matrix`

    Raises
    ------
    dpnp.linalg.LinAlgError
        On non-zero elements in dev_info, indicating LAPACK errors.

    """

    if any(dev_info):
        error_msg = error_msg or "Singular matrix"

        raise LinAlgError(error_msg)


def _common_type(*arrays):
    """
    Common type for linear algebra operations.

    This function determines the common data type for linalg operations.
    It's designed to be similar in logic to `numpy.linalg.linalg._commonType`.

    Key differences from `numpy.common_type`:
    - It accepts ``bool_`` arrays.
    - The default floating-point data type is determined by the capabilities of
      the device on which `arrays` are created, as indicated
      by `dpnp.default_float_type()`.

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
        A sequence of input arrays.

    Returns
    -------
    dtype_common : dpnp.dtype
        The common data type for linalg operations.
        This returned value is applicable both as the precision to be used
        in linalg calls and as the dtype of (possibly complex) output(s).

    """

    dtypes = [arr.dtype for arr in arrays]

    _, sycl_queue = get_usm_allocations(arrays)
    default = dpnp.default_float_type(sycl_queue=sycl_queue)
    dtype_common = _common_inexact_type(default, *dtypes)

    return dtype_common


def _common_inexact_type(default_dtype, *dtypes):
    """
    Determines the common 'inexact' data type for linear algebra operations.

    This function selects an 'inexact' data type appropriate for the device's
    capabilities. It defaults to `default_dtype` when provided types are not
    'inexact'.

    Parameters
    ----------
    default_dtype : dpnp.dtype
        The default data type. This is determined by the capabilities of
        the device and is used when none of the provided types are 'inexact'.
        *dtypes: A variable number of data types to be evaluated to find
        the common 'inexact' type.

    Returns
    -------
    dpnp.result_type : dpnp.dtype
        The resultant 'inexact' data type for linalg operations,
        ensuring computational compatibility.

    """

    inexact_dtypes = [
        dt if dpnp.issubdtype(dt, dpnp.inexact) else default_dtype
        for dt in dtypes
    ]
    return dpnp.result_type(*inexact_dtypes)


def _hermitian_svd(a, compute_uv):
    """
    _hermitian_svd(a, compute_uv)

    Return the singular value decomposition (SVD) of Hermitian matrix `a`.

    """

    assert_stacked_square(a)

    # _gesvd returns eigenvalues with s ** 2 sorted descending,
    # but dpnp.linalg.eigh returns s sorted ascending so we re-order
    # the eigenvalues and related arrays to have the correct order
    if compute_uv:
        s, u = dpnp.linalg.eigh(a)
        sgn = dpnp.sign(s)
        s = dpnp.absolute(s)
        sidx = dpnp.argsort(s)[..., ::-1]
        # Rearrange the signs according to sorted indices
        sgn = dpnp.take_along_axis(sgn, sidx, axis=-1)
        # Sort the singular values in descending order
        s = dpnp.take_along_axis(s, sidx, axis=-1)
        # Rearrange the eigenvectors according to sorted indices
        u = dpnp.take_along_axis(u, sidx[..., None, :], axis=-1)
        # Singular values are unsigned, move the sign into v
        # Compute V^T adjusting for the sign and conjugating
        vt = dpnp.transpose(u * sgn[..., None, :]).conjugate()
        return u, s, vt

    # TODO: use dpnp.linalg.eighvals when it is updated
    s, _ = dpnp.linalg.eigh(a)
    s = dpnp.abs(s)
    return dpnp.sort(s)[..., ::-1]


def _is_empty_2d(arr):
    # check size first for efficiency
    return arr.size == 0 and numpy.prod(arr.shape[-2:]) == 0


def _lu_factor(a, res_type):
    """
    Compute pivoted LU decomposition.

    Decompose a given batch of square matrices. Inputs and outputs are
    transposed.

    Parameters
    ----------
    a : (..., M, M) {dpnp.ndarray, usm_ndarray}
        Input array containing the matrices to be decomposed.
    res_type : dpnp.dtype
        Specifies the data type of the result.
        Acceptable data types are float32, float64, complex64, or complex128.

    Returns
    -------
    tuple:
        lu_t : (..., N, N) {dpnp.ndarray, usm_ndarray}
            Combined 'L' and 'U' matrices from LU decomposition
            excluding the diagonal of 'L'.
        piv : (..., N) {dpnp.ndarray, usm_ndarray}
            1-origin pivot indices indicating row permutations during
            decomposition.
        dev_info : (...) {dpnp.ndarray, usm_ndarray}
            Information on `getrf` or `getrf_batch` computation success
            (0 for success).

    """

    n = a.shape[-2]

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    # TODO: Find out at which array sizes the best performance is obtained
    # getrf_batch implementation shows slow results with large arrays on GPU.
    # Use getrf_batch only on CPU.
    # On GPU call getrf for each two-dimensional array by loop
    use_batch = a.sycl_device.has_aspect_cpu

    if a.ndim > 2:
        orig_shape = a.shape
        # get 3d input arrays by reshape
        a = a.reshape(-1, n, n)
        batch_size = a.shape[0]
        a_usm_arr = dpnp.get_usm_ndarray(a)

        if use_batch:
            # `a` must be copied because getrf_batch destroys the input matrix
            a_h = dpnp.empty_like(a, order="C", dtype=res_type)
            ipiv_h = dpnp.empty(
                (batch_size, n),
                dtype=dpnp.int64,
                order="C",
                usm_type=a_usm_type,
                sycl_queue=a_sycl_queue,
            )
            dev_info_h = [0] * batch_size

            a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_usm_arr, dst=a_h.get_array(), sycl_queue=a_sycl_queue
            )

            ipiv_stride = n
            a_stride = a_h.strides[0]

            # Call the LAPACK extension function _getrf_batch
            # to perform LU decomposition of a batch of general matrices
            ht_lapack_ev, _ = li._getrf_batch(
                a_sycl_queue,
                a_h.get_array(),
                ipiv_h.get_array(),
                dev_info_h,
                n,
                a_stride,
                ipiv_stride,
                batch_size,
                [a_copy_ev],
            )

            ht_lapack_ev.wait()
            a_ht_copy_ev.wait()

            dev_info_array = dpnp.array(
                dev_info_h, usm_type=a_usm_type, sycl_queue=a_sycl_queue
            )

            # Reshape the results back to their original shape
            a_h = a_h.reshape(orig_shape)
            ipiv_h = ipiv_h.reshape(orig_shape[:-1])
            dev_info_array = dev_info_array.reshape(orig_shape[:-2])

            return (a_h, ipiv_h, dev_info_array)

        # Initialize lists for storing arrays and events for each batch
        a_vecs = [None] * batch_size
        ipiv_vecs = [None] * batch_size
        dev_info_vecs = [None] * batch_size
        a_ht_copy_ev = [None] * batch_size
        ht_lapack_ev = [None] * batch_size

        # Process each batch
        for i in range(batch_size):
            # Copy each 2D slice to a new array because getrf will destroy
            # the input matrix
            a_vecs[i] = dpnp.empty_like(a[i], order="C", dtype=res_type)
            (
                a_ht_copy_ev[i],
                a_copy_ev,
            ) = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_usm_arr[i],
                dst=a_vecs[i].get_array(),
                sycl_queue=a_sycl_queue,
            )
            ipiv_vecs[i] = dpnp.empty(
                (n,),
                dtype=dpnp.int64,
                order="C",
                usm_type=a_usm_type,
                sycl_queue=a_sycl_queue,
            )
            dev_info_vecs[i] = [0]

            # Call the LAPACK extension function _getrf
            # to perform LU decomposition on each batch in 'a_vecs[i]'
            ht_lapack_ev[i], _ = li._getrf(
                a_sycl_queue,
                a_vecs[i].get_array(),
                ipiv_vecs[i].get_array(),
                dev_info_vecs[i],
                [a_copy_ev],
            )

        for i in range(batch_size):
            ht_lapack_ev[i].wait()
            a_ht_copy_ev[i].wait()

        # Reshape the results back to their original shape
        out_a = dpnp.array(a_vecs, order="C").reshape(orig_shape)
        out_ipiv = dpnp.array(ipiv_vecs).reshape(orig_shape[:-1])
        out_dev_info = dpnp.array(
            dev_info_vecs, usm_type=a_usm_type, sycl_queue=a_sycl_queue
        ).reshape(orig_shape[:-2])

        return (out_a, out_ipiv, out_dev_info)

    a_usm_arr = dpnp.get_usm_ndarray(a)

    # `a` must be copied because getrf destroys the input matrix
    a_h = dpnp.empty_like(a, order="C", dtype=res_type)

    # use DPCTL tensor function to fill the сopy of the input array
    # from the input array
    a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr, dst=a_h.get_array(), sycl_queue=a_sycl_queue
    )

    ipiv_h = dpnp.empty(
        n,
        dtype=dpnp.int64,
        order="C",
        usm_type=a_usm_type,
        sycl_queue=a_sycl_queue,
    )
    dev_info_h = [0]

    # Call the LAPACK extension function _getrf
    # to perform LU decomposition on the input matrix
    ht_lapack_ev, _ = li._getrf(
        a_sycl_queue,
        a_h.get_array(),
        ipiv_h.get_array(),
        dev_info_h,
        [a_copy_ev],
    )

    ht_lapack_ev.wait()
    a_ht_copy_ev.wait()

    dev_info_array = dpnp.array(
        dev_info_h, usm_type=a_usm_type, sycl_queue=a_sycl_queue
    )

    # Return a tuple containing the factorized matrix 'a_h',
    # pivot indices 'ipiv_h'
    # and the status 'dev_info_h' from the LAPACK getrf call
    return (a_h, ipiv_h, dev_info_array)


def _multi_dot(arrays, order, i, j, out=None):
    """Actually do the multiplication with the given order."""

    if i == j:
        # the initial call with non-None out should never get here
        assert out is None
        return arrays[i]

    return dpnp.dot(
        _multi_dot(arrays, order, i, order[i, j]),
        _multi_dot(arrays, order, order[i, j] + 1, j),
        out=out,
    )


def _multi_dot_matrix_chain_order(n, arrays, return_costs=False):
    """
    Return a dpnp.ndarray that encodes the optimal order of multiplications.

    The optimal order array is then used by `_multi_dot()` to do the
    multiplication.

    Also return the cost matrix if `return_costs` is ``True``.

    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.

        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])

    """

    usm_type, exec_q = get_usm_allocations(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    p = [1 if arrays[0].ndim == 1 else arrays[0].shape[0]]
    p += [a.shape[0] for a in arrays[1:-1]]
    p += (
        [arrays[-1].shape[0], 1]
        if arrays[-1].ndim == 1
        else [arrays[-1].shape[0], arrays[-1].shape[1]]
    )
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = dpnp.zeros((n, n), usm_type=usm_type, sycl_queue=exec_q)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = dpnp.zeros(
        (n, n), dtype=dpnp.intp, usm_type=usm_type, sycl_queue=exec_q
    )

    for ll in range(1, n):
        for i in range(n - ll):
            j = i + ll
            m[i, j] = dpnp.Inf
            for k in range(i, j):
                q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return (s, m) if return_costs else s


def _multi_dot_three(A, B, C, out=None):
    """Find the best order for three arrays and do the multiplication."""

    a0, a1b0 = (1, A.shape[0]) if A.ndim == 1 else A.shape
    b1c0, c1 = (C.shape[0], 1) if C.ndim == 1 else C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return dpnp.dot(dpnp.dot(A, B), C, out=out)

    return dpnp.dot(A, dpnp.dot(B, C), out=out)


def _multi_svd_norm(x, row_axis, col_axis, op):
    """
    Compute a function of the singular values of the 2-D matrices in `x`.

    This is a private utility function used by `dpnp.linalg.norm()`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
    row_axis, col_axis : int
        The axes of `x` that hold the 2-D matrices.
    op : callable
        This should be either `dpnp.min` or `dpnp.max` or `dpnp.sum`.

    Returns
    -------
    out : dpnp.ndarray
        If `x` is 2-D, the return values is a 0-d array.
        Otherwise, it is an array with ``x.ndim - 2`` dimensions.
        The return values are either the minimum or maximum or sum of the
        singular values of the matrices, depending on whether `op`
        is `dpnp.min` or `dpnp.max` or `dpnp.sum`.

    """

    y = dpnp.moveaxis(x, (row_axis, col_axis), (-2, -1))
    result = op(dpnp.linalg.svd(y, compute_uv=False), axis=-1)
    return result


def _norm_int_axis(x, ord, axis, keepdims):
    """
    _norm_int_axis(x, ord, axis, keepdims)

    Compute matrix or vector norm of `x` along integer `axis`.

    """

    if ord == dpnp.inf:
        return dpnp.abs(x).max(axis=axis, keepdims=keepdims)
    if ord == -dpnp.inf:
        return dpnp.abs(x).min(axis=axis, keepdims=keepdims)
    if ord == 0:
        # Zero norm
        # Convert to Python float in accordance with NumPy
        return (x != 0).astype(x.real.dtype).sum(axis=axis, keepdims=keepdims)
    if ord == 1:
        # special case for speedup
        return dpnp.abs(x).sum(axis=axis, keepdims=keepdims)
    if ord is None or ord == 2:
        # special case for speedup
        s = (dpnp.conj(x) * x).real
        return dpnp.sqrt(dpnp.sum(s, axis=axis, keepdims=keepdims))
    if isinstance(ord, (int, float)):
        absx = dpnp.abs(x)
        absx **= ord
        ret = absx.sum(axis=axis, keepdims=keepdims)
        ret **= numpy.reciprocal(ord, dtype=ret.dtype)
        return ret

    # including str-type keywords for ord ("fro", "nuc") which
    # are not valid for vectors
    raise ValueError(f"Invalid norm order '{ord}' for vectors")


def _norm_tuple_axis(x, ord, row_axis, col_axis, keepdims):
    """
    _norm_tuple_axis(x, ord, row_axis, col_axis, keepdims)

    Compute matrix or vector norm of `x` along 2-tuple `axis`.

    """

    axis = (row_axis, col_axis)
    if row_axis == col_axis:
        raise ValueError("Duplicate axes given.")
    if ord == 2:
        ret = _multi_svd_norm(x, row_axis, col_axis, dpnp.max)
    elif ord == -2:
        ret = _multi_svd_norm(x, row_axis, col_axis, dpnp.min)
    elif ord == 1:
        if col_axis > row_axis:
            col_axis -= 1
        ret = dpnp.abs(x).sum(axis=row_axis).max(axis=col_axis)
    elif ord == dpnp.inf:
        if row_axis > col_axis:
            row_axis -= 1
        ret = dpnp.abs(x).sum(axis=col_axis).max(axis=row_axis)
    elif ord == -1:
        if col_axis > row_axis:
            col_axis -= 1
        ret = dpnp.abs(x).sum(axis=row_axis).min(axis=col_axis)
    elif ord == -dpnp.inf:
        if row_axis > col_axis:
            row_axis -= 1
        ret = dpnp.abs(x).sum(axis=col_axis).min(axis=row_axis)
    elif ord in [None, "fro", "f"]:
        ret = dpnp.sqrt(dpnp.sum((dpnp.conj(x) * x).real, axis=axis))
    elif ord == "nuc":
        ret = _multi_svd_norm(x, row_axis, col_axis, dpnp.sum)
    else:
        raise ValueError("Invalid norm order for matrices.")

    if keepdims:
        ret_shape = list(x.shape)
        ret_shape[axis[0]] = 1
        ret_shape[axis[1]] = 1
        ret = ret.reshape(ret_shape)
    return ret


def _nrm2_last_axis(x):
    """
    Calculate the sum of squares along the last axis of an array.

    This function handles arrays containing real or complex numbers.
    For complex data types, it computes the sum of squared magnitudes;
    For real adata types, it sums the squares of the elements.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}

    Returns
    -------
    out : dpnp.ndarray
        Sum of squares calculated along the last axis.

    """

    real_dtype = _real_type(x.dtype)
    # TODO: use dpnp.sum(dpnp.square(dpnp.view(x)), axis=-1, dtype=real_dtype)
    # w/a since dpnp.view() in not implemented yet
    # Сalculate and sum the squares of both real and imaginary parts for
    # compelex array.
    if dpnp.issubdtype(x.dtype, dpnp.complexfloating):
        y = dpnp.abs(x) ** 2
    else:
        y = dpnp.square(x)
    return dpnp.sum(y, axis=-1, dtype=real_dtype)


def _real_type(dtype, device=None):
    """
    Returns the real data type corresponding to a given dpnp data type.

    Parameters
    ----------
    dtype : dpnp.dtype
        The dtype for which to find the corresponding real data type.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where an array of default floating type
        might be created.

    Returns
    -------
    out : str
        The name of the real data type.

    """

    default = dpnp.default_float_type(device)
    real_type = _real_types_map.get(dtype.name, default)
    return dpnp.dtype(real_type)


def _stacked_identity(
    batch_shape, n, dtype, usm_type="device", sycl_queue=None
):
    """
    Create stacked identity matrices of size `n x n`.

    Forms multiple identity matrices based on `batch_shape`.

    Parameters
    ----------
    batch_shape : tuple
        Shape of the batch determining the stacking of identity matrices.
    n : int
        Dimension of each identity matrix.
    dtype : dtype
        Data type of the matrix element.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of stacked `n x n` identity matrices as per `batch_shape`.

    Example
    -------
    >>> _stacked_identity((2,), 2, dtype=dpnp.int64)
    array([[[1, 0],
            [0, 1]],

           [[1, 0],
            [0, 1]]])

    """

    shape = batch_shape + (n, n)
    x = dpnp.empty(shape, dtype=dtype, usm_type=usm_type, sycl_queue=sycl_queue)
    x[...] = dpnp.eye(
        n, dtype=x.dtype, usm_type=x.usm_type, sycl_queue=x.sycl_queue
    )
    return x


def _stacked_identity_like(x):
    """
    Create stacked identity matrices based on the shape and properties of `x`.

    Parameters
    ----------
    x : dpnp.ndarray
        Input array based on whose properties (shape, data type, USM type and
        SYCL queue) the identity matrices will be created.

    Returns
    -------
    out : dpnp.ndarray
        Array of stacked `n x n` identity matrices,
        where `n` is the size of the last dimension of `x`.
        The returned array has the same shape, data type, USM type
        and uses the same SYCL queue as `x`, if applicable.

    Example
    -------
    >>> import dpnp
    >>> x = dpnp.zeros((2, 3, 3), dtype=dpnp.int64)
    >>> _stacked_identity_like(x)
    array([[[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]],

           [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]], dtype=int32)

    """

    x = dpnp.empty_like(x)
    x[...] = dpnp.eye(
        x.shape[-2], dtype=x.dtype, usm_type=x.usm_type, sycl_queue=x.sycl_queue
    )
    return x


def _triu_inplace(a, host_tasks, depends=None):
    """
    _triu_inplace(a, host_tasks, depends=None)

    Computes the upper triangular part of an array in-place,
    but currently allocates extra memory for the result.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array from which the upper triangular part is to be extracted.
    host_tasks : list
        A list to which the function appends the host event corresponding to
        the computation. This allows for dependency management and
        synchronization with other tasks.
    depends : list, optional
        A list of events that the triangular operation depends on.
        These tasks are completed before the triangular computation starts.
        If ``None``, defaults to an empty list.

    Returns
    -------
    out : dpnp.ndarray
        A new array containing the upper triangular part of the input array `a`.

    """

    # TODO: implement a dedicated kernel for in-place triu instead of
    # extra memory allocation for result
    if depends is None:
        depends = []
    out = dpnp.empty_like(a, order="C")
    ht_triu_ev, _ = ti._triu(
        src=a.get_array(),
        dst=out.get_array(),
        k=0,
        sycl_queue=a.sycl_queue,
        depends=depends,
    )
    host_tasks.append(ht_triu_ev)
    return out


def _zero_batched_qr(a, mode, m, n, k, res_type):
    """
    _zero_batched_qr(a, mode, m, n, k, res_type)

    Return the QR factorization of `a` matrix of zero batch length or
    when ``k == 0``.

    """

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    batch_shape = a.shape[:-2]

    if mode == "reduced":
        return (
            dpnp.empty_like(
                a,
                shape=batch_shape + (m, k),
                dtype=res_type,
            ),
            dpnp.empty_like(
                a,
                shape=batch_shape + (k, n),
                dtype=res_type,
            ),
        )
    if mode == "complete":
        q = _stacked_identity(
            batch_shape,
            m,
            dtype=res_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        return (
            q,
            dpnp.empty_like(
                a,
                shape=batch_shape + (m, n),
                dtype=res_type,
            ),
        )
    if mode == "r":
        return dpnp.empty_like(
            a,
            shape=batch_shape + (k, n),
            dtype=res_type,
        )

    # mode=="raw"
    return (
        dpnp.empty_like(
            a,
            shape=batch_shape + (n, m),
            dtype=res_type,
        ),
        dpnp.empty_like(
            a,
            shape=batch_shape + (k,),
            dtype=res_type,
        ),
    )


def _zero_batched_svd(
    a,
    uv_type,
    s_type,
    full_matrices,
    compute_uv,
    exec_q,
    usm_type,
    batch_shape_orig,
):
    """
    _zero_batched_svd(
        a,
        uv_type,
        s_type,
        full_matrices,
        compute_uv,
        exec_q,
        usm_type,
        batch_shape_orig,
    )

    Return the singular value decomposition (SVD) of a zero-lenth stack
    of matrices.

    """

    m, n = a.shape[-2:]
    k = min(m, n)

    s = dpnp.empty(
        batch_shape_orig + (k,),
        dtype=s_type,
        usm_type=usm_type,
        sycl_queue=exec_q,
    )

    if compute_uv:
        if full_matrices:
            u_shape = batch_shape_orig + (m, m)
            vt_shape = batch_shape_orig + (n, n)
        else:
            u_shape = batch_shape_orig + (m, k)
            vt_shape = batch_shape_orig + (k, n)

        u = dpnp.empty(
            u_shape,
            dtype=uv_type,
            usm_type=usm_type,
            sycl_queue=exec_q,
        )
        vt = dpnp.empty(
            vt_shape,
            dtype=uv_type,
            usm_type=usm_type,
            sycl_queue=exec_q,
        )
        return u, s, vt
    return s


def _zero_k_qr(a, mode, m, n, res_type):
    """
    _zero_k_qr(a, mode, m, n, res_type)

    Return the QR factorization of `a` matrix with ``k == 0``.

    """

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    m, n = a.shape

    if mode == "reduced":
        return dpnp.empty_like(
            a,
            shape=(m, 0),
            dtype=res_type,
        ), dpnp.empty_like(
            a,
            shape=(0, n),
            dtype=res_type,
        )
    if mode == "complete":
        return dpnp.identity(
            m, dtype=res_type, sycl_queue=a_sycl_queue, usm_type=a_usm_type
        ), dpnp.empty_like(
            a,
            shape=(m, n),
            dtype=res_type,
        )
    if mode == "r":
        return dpnp.empty_like(
            a,
            shape=(0, n),
            dtype=res_type,
        )

    # mode == "raw"
    return dpnp.empty_like(
        a,
        shape=(n, m),
        dtype=res_type,
    ), dpnp.empty_like(
        a,
        shape=(0,),
        dtype=res_type,
    )


def _zero_m_n_batched_svd(
    a,
    uv_type,
    s_type,
    full_matrices,
    compute_uv,
    exec_q,
    usm_type,
    batch_shape_orig,
):
    """
    _zero_m_n_batched_svd(
        a,
        uv_type,
        s_type,
        full_matrices,
        compute_uv,
        exec_q,
        usm_type,
        batch_shape_orig,
    )

    Return the singular value decomposition (SVD) of a stack
    of matrices with either ``m == 0`` or ``n == 0``.

    """

    m, n = a.shape[-2:]
    s = dpnp.empty(
        batch_shape_orig + (0,),
        dtype=s_type,
        usm_type=usm_type,
        sycl_queue=exec_q,
    )

    if compute_uv:
        if full_matrices:
            u = _stacked_identity(
                batch_shape_orig,
                m,
                dtype=uv_type,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
            vt = _stacked_identity(
                batch_shape_orig,
                n,
                dtype=uv_type,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
        else:
            u = dpnp.empty(
                batch_shape_orig + (m, 0),
                dtype=uv_type,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
            vt = dpnp.empty(
                batch_shape_orig + (0, n),
                dtype=uv_type,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
        return u, s, vt
    return s


def _zero_m_n_svd(
    a, uv_type, s_type, full_matrices, compute_uv, exec_q, usm_type
):
    """
    _zero_m_n_svd(
        a, uv_type, s_type, full_matrices, compute_uv, exec_q, usm_type
    )

    Return the singular value decomposition (SVD) of a matrix
    with either ``m == 0`` or ``n == 0``.

    """

    m, n = a.shape
    s = dpnp.empty(
        (0,),
        dtype=s_type,
        usm_type=usm_type,
        sycl_queue=exec_q,
    )
    if compute_uv:
        if full_matrices:
            u_shape = (m,)
            vt_shape = (n,)
        else:
            u_shape = (m, 0)
            vt_shape = (0, n)

        u = dpnp.eye(
            *u_shape,
            dtype=uv_type,
            usm_type=usm_type,
            sycl_queue=exec_q,
        )
        vt = dpnp.eye(
            *vt_shape,
            dtype=uv_type,
            usm_type=usm_type,
            sycl_queue=exec_q,
        )
        return u, s, vt
    return s


def assert_2d(*arrays):
    """
    Check that each array in `arrays` is exactly two-dimensional.

    If any array is not two-dimensional, `dpnp.linalg.LinAlgError` will be
    raised.

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
        A sequence of input arrays to check for dimensionality.

    Raises
    ------
    dpnp.linalg.LinAlgError
        If any array in `arrays` is not exactly two-dimensional.

    """

    for a in arrays:
        if a.ndim != 2:
            raise LinAlgError(
                f"{a.ndim}-dimensional array given. The input "
                "array must be exactly two-dimensional"
            )


def assert_stacked_2d(*arrays):
    """
    Check that each array in `arrays` has at least two dimensions.

    If any array is less than two-dimensional, `dpnp.linalg.LinAlgError` will
    be raised.

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
        A sequence of input arrays to check for dimensionality.

    Raises
    ------
    dpnp.linalg.LinAlgError
        If any array in `arrays` is less than two-dimensional.

    """

    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError(
                f"{a.ndim}-dimensional array given. The input "
                "array must be at least two-dimensional"
            )


def assert_stacked_square(*arrays):
    """
    Check that each array in `arrays` is a square matrix.

    If any array does not form a square matrix, `dpnp.linalg.LinAlgError` will
    be raised.

    Precondition: `arrays` are at least 2d. The caller should assert it
    beforehand. For example,

    >>> def solve(a):
    ...     assert_stacked_2d(a)
    ...     assert_stacked_square(a)
    ...     ...

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
        A sequence of input arrays to check for square matrix shape.

    Raises
    ------
    dpnp.linalg.LinAlgError
        If any array in `arrays` does not form a square matrix.

    """

    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise LinAlgError(
                "Last 2 dimensions of the input array must be square"
            )


def dpnp_cholesky_batch(a, upper_lower, res_type):
    """
    dpnp_cholesky_batch(a, upper_lower, res_type)

    Return the batched Cholesky decomposition of `a` array.

    """

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    n = a.shape[-2]

    orig_shape = a.shape
    # get 3d input arrays by reshape
    a = a.reshape(-1, n, n)
    batch_size = a.shape[0]
    a_usm_arr = dpnp.get_usm_ndarray(a)

    # `a` must be copied because potrf_batch destroys the input matrix
    a_h = dpnp.empty_like(a, order="C", dtype=res_type, usm_type=a_usm_type)

    # use DPCTL tensor function to fill the сopy of the input array
    # from the input array
    a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr, dst=a_h.get_array(), sycl_queue=a_sycl_queue
    )

    a_stride = a_h.strides[0]

    # Call the LAPACK extension function _potrf_batch
    # to computes the Cholesky decomposition of a batch of
    # symmetric positive-definite matrices
    ht_lapack_ev, _ = li._potrf_batch(
        a_sycl_queue,
        a_h.get_array(),
        upper_lower,
        n,
        a_stride,
        batch_size,
        [a_copy_ev],
    )

    ht_lapack_ev.wait()
    a_ht_copy_ev.wait()

    # Get upper or lower-triangular matrix part as per `upper_lower` value
    # upper_lower is 0 (lower) or 1 (upper)
    if upper_lower:
        a_h = dpnp.triu(a_h.reshape(orig_shape))
    else:
        a_h = dpnp.tril(a_h.reshape(orig_shape))

    return a_h


def dpnp_cholesky(a, upper):
    """
    dpnp_cholesky(a, upper)

    Return the Cholesky decomposition of `a` array.

    """

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    res_type = _common_type(a)

    a_shape = a.shape

    if a.size == 0:
        return dpnp.empty(
            a_shape,
            dtype=res_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )

    # Set `uplo` value for `potrf` and `potrf_batch` function based on the
    # boolean input `upper`.
    # In oneMKL, `uplo` value of 1 is equivalent to oneapi::mkl::uplo::lower
    # and `uplo` value of 0 is equivalent to oneapi::mkl::uplo::upper.
    # However, we adjust this logic based on the array's memory layout.
    # Note: lower for row-major (which is used here) is upper for column-major
    # layout.
    # Ref: comment from tbmkl/tests/lapack/unit/dpcpp/potrf_usm/potrf_usm.cpp
    # This means that if `upper` is False (lower triangular),
    # we actually use oneapi::mkl::uplo::upper (0) for the row-major layout,
    # and vice versa.
    upper_lower = int(upper)

    if a.ndim > 2:
        return dpnp_cholesky_batch(a, upper_lower, res_type)

    a_usm_arr = dpnp.get_usm_ndarray(a)

    # `a` must be copied because potrf destroys the input matrix
    a_h = dpnp.empty_like(a, order="C", dtype=res_type, usm_type=a_usm_type)

    # use DPCTL tensor function to fill the сopy of the input array
    # from the input array
    a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr, dst=a_h.get_array(), sycl_queue=a_sycl_queue
    )

    # Call the LAPACK extension function _potrf
    # to computes the Cholesky decomposition
    ht_lapack_ev, _ = li._potrf(
        a_sycl_queue,
        a_h.get_array(),
        upper_lower,
        [a_copy_ev],
    )

    ht_lapack_ev.wait()
    a_ht_copy_ev.wait()

    # Get upper or lower-triangular matrix part as per `upper` value
    if upper:
        a_h = dpnp.triu(a_h)
    else:
        a_h = dpnp.tril(a_h)

    return a_h


def dpnp_cond(x, p=None):
    """Compute the condition number of a matrix."""

    if _is_empty_2d(x):
        raise LinAlgError("cond is not defined on empty arrays")
    if p is None or p == 2 or p == -2:
        s = dpnp.linalg.svd(x, compute_uv=False)
        if p == -2:
            r = s[..., -1] / s[..., 0]
        else:
            r = s[..., 0] / s[..., -1]
    else:
        result_t = _common_type(x)
        # The result array will contain nans in the entries
        # where inversion failed
        invx = dpnp.linalg.inv(x)
        r = dpnp.linalg.norm(x, p, axis=(-2, -1)) * dpnp.linalg.norm(
            invx, p, axis=(-2, -1)
        )
        r = r.astype(result_t, copy=False)

    # Convert nans to infs unless the original array had nan entries
    nan_mask = dpnp.isnan(r)
    if nan_mask.any():
        nan_mask &= ~dpnp.isnan(x).any(axis=(-2, -1))
        r[nan_mask] = dpnp.inf

    return r


def dpnp_det(a):
    """
    dpnp_det(a)

    Returns the determinant of `a` array.

    """

    a_usm_type = a.usm_type
    a_sycl_queue = a.sycl_queue

    res_type = _common_type(a)

    a_shape = a.shape
    shape = a_shape[:-2]
    n = a_shape[-2]

    if a.size == 0:
        # empty batch (result is empty, too) or empty matrices det([[]]) == 1
        det = dpnp.ones(
            shape,
            dtype=res_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        return det

    lu, ipiv, dev_info = _lu_factor(a, res_type)

    diag = dpnp.diagonal(lu, axis1=-2, axis2=-1)

    det = dpnp.prod(dpnp.abs(diag), axis=-1)

    sign = _calculate_determinant_sign(ipiv, diag, res_type, n)

    det = sign * det
    det = det.astype(res_type, copy=False)
    singular = dev_info > 0
    det = dpnp.where(singular, res_type.type(0), det)

    return det.reshape(shape)


def dpnp_eigh(a, UPLO, eigen_mode="V"):
    """
    dpnp_eigh(a, UPLO, eigen_mode="V")

    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.
    Can return both eigenvalues and eigenvectors (`eigen_mode="V"`) or
    only eigenvalues (`eigen_mode="N"`).

    The main calculation is done by calling an extension function
    for LAPACK library of OneMKL. Depending on input type of `a` array,
    it will be either ``heevd`` (for complex types) or ``syevd`` (for others).

    """

    # get resulting type of arrays with eigenvalues and eigenvectors
    v_type = _common_type(a)
    w_type = _real_type(v_type)

    if a.size == 0:
        w = dpnp.empty_like(a, shape=a.shape[:-1], dtype=w_type)
        if eigen_mode == "V":
            v = dpnp.empty_like(a, dtype=v_type)
            return w, v
        return w

    if a.ndim > 2:
        return _batched_eigh(a, UPLO, eigen_mode, w_type, v_type)

    # `eigen_mode` can be either "N" or "V", specifying the computation mode
    # for OneMKL LAPACK `syevd` and `heevd` routines.
    # "V" (default) means both eigenvectors and eigenvalues will be calculated
    # "N" means only eigenvalues will be calculated
    jobz = _jobz[eigen_mode]
    uplo = _upper_lower[UPLO]

    # Get LAPACK function (_syevd for real or _heevd for complex data types)
    # to compute all eigenvalues and, optionally, all eigenvectors
    lapack_func = (
        "_heevd" if dpnp.issubdtype(v_type, dpnp.complexfloating) else "_syevd"
    )

    a_sycl_queue = a.sycl_queue
    a_order = "C" if a.flags.c_contiguous else "F"

    a_usm_arr = dpnp.get_usm_ndarray(a)
    ht_list_ev = []
    copy_ev = dpctl.SyclEvent()

    # When `eigen_mode == "N"` (jobz == 0), OneMKL LAPACK does not
    # overwrite the input array.
    # If the input array 'a' is already F-contiguous and matches the target
    # data type, we can avoid unnecessary memory allocation and data
    # copying.
    if eigen_mode == "N" and a_order == "F" and a.dtype == v_type:
        v = a
    else:
        # oneMKL LAPACK assumes fortran-like array as input, so
        # allocate a memory with 'F' order for dpnp array of eigenvectors
        v = dpnp.empty_like(a, order="F", dtype=v_type)

        # use DPCTL tensor function to fill the array of eigenvectors with
        # content of input array
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_usm_arr, dst=v.get_array(), sycl_queue=a_sycl_queue
        )
        ht_list_ev.append(ht_copy_ev)

    # allocate a memory for dpnp array of eigenvalues
    w = dpnp.empty_like(
        a,
        shape=a.shape[:-1],
        dtype=w_type,
    )

    # call LAPACK extension function to get eigenvalues and eigenvectors of
    # matrix A
    ht_lapack_ev, lapack_ev = getattr(li, lapack_func)(
        a_sycl_queue,
        jobz,
        uplo,
        v.get_array(),
        w.get_array(),
        depends=[copy_ev],
    )
    ht_list_ev.append(ht_lapack_ev)

    if eigen_mode == "V" and a_order != "F":
        # need to align order of eigenvectors with one of input matrix A
        out_v = dpnp.empty_like(v, order=a_order)
        ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            src=v.get_array(),
            dst=out_v.get_array(),
            sycl_queue=a_sycl_queue,
            depends=[lapack_ev],
        )
        ht_list_ev.append(ht_copy_out_ev)
    else:
        out_v = v

    dpctl.SyclEvent.wait_for(ht_list_ev)
    return (w, out_v) if eigen_mode == "V" else w


def dpnp_inv(a):
    """
    dpnp_inv(a)

    Return the inverse of `a` matrix.

    The inverse of a matrix is such that if it is multiplied by the original
    matrix, it results in the identity matrix. This function computes the
    inverse of a single square matrix.

    """

    res_type = _common_type(a)
    if a.size == 0:
        return dpnp.empty_like(a, dtype=res_type)

    if a.ndim >= 3:
        return _batched_inv(a, res_type)

    a_usm_arr = dpnp.get_usm_ndarray(a)
    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    a_order = "C" if a.flags.c_contiguous else "F"
    a_shape = a.shape

    # oneMKL LAPACK gesv overwrites `a` and assumes fortran-like array as input.
    # To use C-contiguous arrays, we transpose them before passing to gesv.
    # This transposition is effective because the input array `a` is square.
    a_f = dpnp.empty_like(a, order=a_order, dtype=res_type)

    # use DPCTL tensor function to fill the coefficient matrix array
    # with content from the input array `a`
    a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr, dst=a_f.get_array(), sycl_queue=a_sycl_queue
    )

    b_f = dpnp.eye(
        a_shape[0],
        dtype=res_type,
        order=a_order,
        sycl_queue=a_sycl_queue,
        usm_type=a_usm_type,
    )

    if a_order == "F":
        ht_lapack_ev, _ = li._gesv(
            a_sycl_queue, a_f.get_array(), b_f.get_array(), [a_copy_ev]
        )
    else:
        ht_lapack_ev, _ = li._gesv(
            a_sycl_queue, a_f.T.get_array(), b_f.T.get_array(), [a_copy_ev]
        )

    ht_lapack_ev.wait()
    a_ht_copy_ev.wait()

    return b_f


def dpnp_lstsq(a, b, rcond=None):
    """
    dpnp_lstsq(a, b, rcond=None)

    Return the least-squares solution to a linear matrix equation.

    """

    if b.ndim > 2:
        raise LinAlgError(
            f"{b.ndim}-dimensional array given. The input "
            "array must be exactly two-dimensional"
        )

    m, n = a.shape[-2:]
    m2 = b.shape[0]
    if m != m2:
        raise LinAlgError("Incompatible dimensions")

    u, s, vh = dpnp_svd(a, full_matrices=False, related_arrays=[b])

    if rcond is None:
        rcond = dpnp.finfo(s.dtype).eps * max(m, n)
    elif rcond <= 0 or rcond >= 1:
        # some doc of gelss/gelsd says "rcond < 0", but it's not true!
        rcond = dpnp.finfo(s.dtype).eps

    # number of singular values and matrix rank
    s1 = 1 / s
    rank = dpnp.array(
        s.size, dtype="int32", sycl_queue=s.sycl_queue, usm_type=s.usm_type
    )
    if s.size > 0:
        cutoff = rcond * s.max()
        sing_vals = s <= cutoff
        s1[sing_vals] = 0
        rank -= sing_vals.sum(dtype="int32")

    # Solve the least-squares solution
    # x = vh.T.conj() @ diag(s1) @ u.T.conj() @ b
    z = (dpnp.dot(b.T, u.conj()) * s1).T
    x = dpnp.dot(vh.T.conj(), z)
    # Calculate squared Euclidean 2-norm for each column in b - a*x
    if m <= n or rank != n:
        resids = dpnp.empty_like(s, shape=(0,))
    else:
        e = b - a.dot(x)
        resids = dpnp.atleast_1d(_nrm2_last_axis(e.T))

    return x, resids, rank, s


def dpnp_matrix_power(a, n):
    """
    dpnp_matrix_power(a, n)

    Raise a square matrix to the (integer) power `n`.

    """

    if n == 0:
        return _stacked_identity_like(a)

    if n < 0:
        a = dpnp.linalg.inv(a)
        n *= -1

    if n == 1:
        return a
    if n == 2:
        return dpnp.matmul(a, a)
    if n == 3:
        return dpnp.matmul(dpnp.matmul(a, a), a)

    # Use binary decomposition to reduce the number of matrix
    # multiplications for n > 3.
    # `result` will hold the final matrix power,
    # while `acc` serves as an accumulator for the intermediate matrix powers.
    result = None
    acc = a.copy()
    while n > 0:
        n, bit = divmod(n, 2)
        if bit:
            if result is None:
                result = acc.copy()
            else:
                dpnp.matmul(result, acc, out=result)
        if n > 0:
            dpnp.matmul(acc, acc, out=acc)

    return result


def dpnp_matrix_rank(A, tol=None, hermitian=False):
    """
    dpnp_matrix_rank(A, tol=None, hermitian=False)

    Return matrix rank of array using SVD method.

    """

    if A.ndim < 2:
        return (A != 0).any().astype(int)

    S = dpnp_svd(A, compute_uv=False, hermitian=hermitian)

    if tol is None:
        rtol = max(A.shape[-2:]) * dpnp.finfo(S.dtype).eps
        tol = S.max(axis=-1, keepdims=True) * rtol
    elif not dpnp.isscalar(tol):
        # Add a new axis to match NumPy's output
        tol = tol[..., None]

    return dpnp.count_nonzero(S > tol, axis=-1)


def dpnp_multi_dot(n, arrays, out=None):
    """Compute dot product of two or more arrays in a single function call."""

    if not arrays[0].ndim in [1, 2]:
        raise LinAlgError(
            f"{arrays[0].ndim}-dimensional array given. "
            "First array must be 1-D or 2-D."
        )

    if not arrays[-1].ndim in [1, 2]:
        raise LinAlgError(
            f"{arrays[-1].ndim}-dimensional array given. "
            "Last array must be 1-D or 2-D."
        )

    for arr in arrays[1:-1]:
        if arr.ndim != 2:
            raise LinAlgError(
                f"{arr.ndim}-dimensional array given. Inner arrays must be 2-D."
            )

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(n, arrays)
        result = _multi_dot(arrays, order, 0, n - 1, out=out)

    return result


def dpnp_norm(x, ord=None, axis=None, keepdims=False):
    """Compute matrix or vector norm."""

    if not dpnp.issubdtype(x.dtype, dpnp.inexact):
        x = dpnp.astype(x, dtype=dpnp.default_float_type(x.device))

    ndim = x.ndim
    # Immediately handle some default, simple, fast, and common cases.
    if axis is None:
        if (
            (ord is None)
            or (ord in ("f", "fro") and ndim == 2)
            or (ord == 2 and ndim == 1)
        ):
            # TODO: use order="K" when it is supported in dpnp.ravel
            x = dpnp.ravel(x)
            if dpnp.issubdtype(x.dtype, dpnp.complexfloating):
                x_real = x.real
                x_imag = x.imag
                sqnorm = dpnp.dot(x_real, x_real) + dpnp.dot(x_imag, x_imag)
            else:
                sqnorm = dpnp.dot(x, x)
            ret = dpnp.sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape((1,) * ndim)
            return ret

    # Normalize the `axis` argument to a tuple.
    if axis is None:
        axis = tuple(range(ndim))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception as e:
            raise TypeError(
                "'axis' must be None, an integer or a tuple of integers"
            ) from e
        axis = (axis,)

    if len(axis) == 1:
        axis = normalize_axis_index(axis[0], ndim)
        return _norm_int_axis(x, ord, axis, keepdims)

    if len(axis) == 2:
        row_axis, col_axis = axis
        row_axis = normalize_axis_index(row_axis, ndim)
        col_axis = normalize_axis_index(col_axis, ndim)
        return _norm_tuple_axis(x, ord, row_axis, col_axis, keepdims)

    raise ValueError("Improper number of dimensions to norm.")


def dpnp_pinv(a, rcond=1e-15, hermitian=False):
    """
    dpnp_pinv(a, rcond=1e-15, hermitian=False):

    Compute the Moore-Penrose pseudoinverse of `a` matrix.

    It computes a pseudoinverse of a matrix `a`, which is a generalization
    of the inverse matrix with Singular Value Decomposition (SVD).

    """

    if _is_empty_2d(a):
        m, n = a.shape[-2:]
        sh = a.shape[:-2] + (n, m)
        return dpnp.empty_like(a, shape=sh)

    if dpnp.is_supported_array_type(rcond):
        # Check that `a` and `rcond` are allocated on the same device
        # and have the same queue. Otherwise, `ValueError`` will be raised.
        get_usm_allocations([a, rcond])
    else:
        # Allocate dpnp.ndarray if rcond is a scalar
        rcond = dpnp.array(rcond, usm_type=a.usm_type, sycl_queue=a.sycl_queue)

    u, s, vt = dpnp_svd(dpnp.conj(a), full_matrices=False, hermitian=hermitian)

    # discard small singular values
    cutoff = rcond * dpnp.max(s, axis=-1)
    leq = s <= cutoff[..., None]
    dpnp.reciprocal(s, out=s)
    s[leq] = 0

    u = u.swapaxes(-2, -1)
    dpnp.multiply(s[..., None], u, out=u)
    return dpnp.matmul(vt.swapaxes(-2, -1), u)


def dpnp_qr(a, mode="reduced"):
    """
    dpnp_qr(a, mode="reduced")

    Return the qr factorization of `a` matrix.

    """

    if a.ndim > 2:
        return _batched_qr(a, mode=mode)

    a_usm_arr = dpnp.get_usm_ndarray(a)
    a_sycl_queue = a.sycl_queue

    res_type = _common_type(a)

    m, n = a.shape
    k = min(m, n)
    if k == 0:
        return _zero_k_qr(a, mode, m, n, res_type)

    # Transpose the input matrix to convert from row-major to column-major
    # order.
    # This adjustment is necessary for compatibility with OneMKL LAPACK
    # routines, which expect matrices in column-major format.
    # This allows data to be handled efficiently without the need for
    # additional conversion.
    a = a.T
    a_usm_arr = dpnp.get_usm_ndarray(a)
    a_t = dpnp.empty_like(a, order="C", dtype=res_type)

    # use DPCTL tensor function to fill the matrix array
    # with content from the input array `a`
    a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr, dst=a_t.get_array(), sycl_queue=a_sycl_queue
    )

    tau_h = dpnp.empty_like(
        a,
        shape=(k,),
        dtype=res_type,
    )

    # Call the LAPACK extension function _geqrf to compute the QR factorization
    # of a general m x n matrix.
    ht_geqrf_ev, geqrf_ev = li._geqrf(
        a_sycl_queue, a_t.get_array(), tau_h.get_array(), [a_copy_ev]
    )

    ht_list_ev = [ht_geqrf_ev, a_ht_copy_ev]

    if mode in ["r", "raw"]:
        if mode == "r":
            r = a_t[:, :k].transpose()
            r = _triu_inplace(r, ht_list_ev, [geqrf_ev])
            dpctl.SyclEvent.wait_for(ht_list_ev)
            return r

        # mode == "raw":
        dpctl.SyclEvent.wait_for(ht_list_ev)
        return (a_t, tau_h)

    # mc is the total number of columns in the q matrix.
    # In `complete` mode, mc equals the number of rows.
    # In `reduced` mode, mc is the lesser of the row count or column count.
    if mode == "complete" and m > n:
        mc = m
        q = dpnp.empty_like(
            a_t,
            shape=(m, m),
            dtype=res_type,
        )
    else:
        mc = k
        q = dpnp.empty_like(
            a_t,
            shape=(n, m),
            dtype=res_type,
        )

    # use DPCTL tensor function to fill the matrix array `q[:n]`
    # with content from the array `a_t` overwritten by geqrf
    a_t_ht_copy_ev, a_t_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_t.get_array(),
        dst=q[:n].get_array(),
        sycl_queue=a_sycl_queue,
        depends=[geqrf_ev],
    )

    ht_list_ev.append(a_t_ht_copy_ev)

    # Get LAPACK function (_orgqr for real or _ungqf for complex data types)
    # for QR factorization
    lapack_func = (
        "_ungqr"
        if dpnp.issubdtype(res_type, dpnp.complexfloating)
        else "_orgqr"
    )

    # Call the LAPACK extension function _orgqr/_ungqf to generate the real
    # orthogonal/complex unitary matrix `Q` of the QR factorization
    ht_lapack_ev, lapack_ev = getattr(li, lapack_func)(
        a_sycl_queue, m, mc, k, q.get_array(), tau_h.get_array(), [a_t_copy_ev]
    )

    q = q[:mc].transpose()
    r = a_t[:, :mc].transpose()

    ht_list_ev.append(ht_lapack_ev)

    r = _triu_inplace(r, ht_list_ev, [lapack_ev])
    dpctl.SyclEvent.wait_for(ht_list_ev)

    return (q, r)


def dpnp_solve(a, b):
    """
    dpnp_solve(a, b)

    Return the solution to the system of linear equations with
    a square coefficient matrix `a` and multiple dependent variables
    array `b`.

    """

    res_usm_type, exec_q = get_usm_allocations([a, b])

    res_type = _common_type(a, b)
    if b.size == 0:
        return dpnp.empty_like(b, dtype=res_type, usm_type=res_usm_type)

    if a.ndim > 2:
        return _batched_solve(a, b, exec_q, res_usm_type, res_type)

    a_usm_arr = dpnp.get_usm_ndarray(a)
    b_usm_arr = dpnp.get_usm_ndarray(b)

    # Due to MKLD-17226 (bug with incorrect checking ldb parameter
    # in oneapi::mkl::lapack::gesv_scratchad_size that raises an error
    # `invalid argument` when nrhs > n) we can not use _gesv directly.
    # This w/a uses _getrf and _getrs instead
    # to handle cases where nrhs > n for a.shape = (n x n)
    # and b.shape = (n x nrhs).

    # oneMKL LAPACK getrf overwrites `a`.
    a_h = dpnp.empty_like(a, order="C", dtype=res_type, usm_type=res_usm_type)

    # use DPCTL tensor function to fill the сopy of the input array
    # from the input array
    a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr, dst=a_h.get_array(), sycl_queue=a.sycl_queue
    )

    # oneMKL LAPACK getrs overwrites `b` and assumes fortran-like array as
    # input.
    # Allocate 'F' order memory for dpnp arrays to comply with
    # these requirements.
    b_h = dpnp.empty_like(b, order="F", dtype=res_type, usm_type=res_usm_type)

    # use DPCTL tensor function to fill the array of multiple dependent
    # variables with content from the input array `b`
    b_ht_copy_ev, b_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=b_usm_arr, dst=b_h.get_array(), sycl_queue=b.sycl_queue
    )

    n = a.shape[0]

    ipiv_h = dpnp.empty_like(
        a,
        shape=(n,),
        dtype=dpnp.int64,
    )
    dev_info_h = [0]

    # Call the LAPACK extension function _getrf
    # to perform LU decomposition of the input matrix
    ht_getrf_ev, getrf_ev = li._getrf(
        exec_q,
        a_h.get_array(),
        ipiv_h.get_array(),
        dev_info_h,
        [a_copy_ev],
    )

    _check_lapack_dev_info(dev_info_h)

    # Call the LAPACK extension function _getrs
    # to solve the system of linear equations with an LU-factored
    # coefficient square matrix, with multiple right-hand sides.
    ht_getrs_ev, _ = li._getrs(
        exec_q,
        a_h.get_array(),
        ipiv_h.get_array(),
        b_h.get_array(),
        [b_copy_ev, getrf_ev],
    )

    ht_list_ev = [a_ht_copy_ev, b_ht_copy_ev, ht_getrf_ev, ht_getrs_ev]
    dpctl.SyclEvent.wait_for(ht_list_ev)
    return b_h


def dpnp_slogdet(a):
    """
    dpnp_slogdet(a)

    Returns sign and logarithm of the determinant of `a` array.

    """

    a_usm_type = a.usm_type
    a_sycl_queue = a.sycl_queue

    res_type = _common_type(a)
    logdet_dtype = _real_type(res_type)

    a_shape = a.shape
    shape = a_shape[:-2]
    n = a_shape[-2]

    if a.size == 0:
        # empty batch (result is empty, too) or empty matrices det([[]]) == 1
        sign = dpnp.ones(
            shape, dtype=res_type, usm_type=a_usm_type, sycl_queue=a_sycl_queue
        )
        logdet = dpnp.zeros(
            shape,
            dtype=logdet_dtype,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        return sign, logdet

    lu, ipiv, dev_info = _lu_factor(a, res_type)

    diag = dpnp.diagonal(lu, axis1=-2, axis2=-1)

    logdet = dpnp.log(dpnp.abs(diag)).sum(axis=-1)

    sign = _calculate_determinant_sign(ipiv, diag, res_type, n)

    logdet = logdet.astype(logdet_dtype, copy=False)
    singular = dev_info > 0
    return (
        dpnp.where(singular, res_type.type(0), sign).reshape(shape),
        dpnp.where(singular, logdet_dtype.type("-inf"), logdet).reshape(shape),
    )


def dpnp_svd(
    a,
    full_matrices=True,
    compute_uv=True,
    hermitian=False,
    batch_call=False,
    related_arrays=None,
):
    """
    dpnp_svd(
        a,
        full_matrices=True,
        compute_uv=True,
        hermitian=False,
        batch_call=False,
        related_arrays=None,
    )

    Return the singular value decomposition (SVD).

    """

    if hermitian:
        return _hermitian_svd(a, compute_uv)

    uv_type = (
        _common_type(a)
        if not related_arrays
        else _common_type(a, *related_arrays)
    )
    s_type = _real_type(uv_type)

    if a.ndim > 2:
        return _batched_svd(
            a,
            uv_type,
            s_type,
            full_matrices,
            compute_uv,
            related_arrays=related_arrays,
        )

    # Set USM type and SYCL queue to be used based on `a`
    # and optionally provided `related_arrays`.
    # If `related_arrays` is not provided, default to USM type and SYCL queue
    # of `a`.
    # Otherwise, determine USM type and SYCL queue using
    # compute-follows-data execution model for `a` and `related arrays`.
    usm_type, exec_q = get_usm_allocations([a] + (related_arrays or []))

    m, n = a.shape
    if m == 0 or n == 0:
        return _zero_m_n_svd(
            a, uv_type, s_type, full_matrices, compute_uv, exec_q, usm_type
        )

    # oneMKL LAPACK gesvd destroys `a` and assumes fortran-like array as input.
    # Allocate 'F' order memory for dpnp arrays to comply with
    # these requirements.
    a_h = dpnp.empty_like(
        a, order="F", dtype=uv_type, usm_type=usm_type, sycl_queue=exec_q
    )

    a_usm_arr = dpnp.get_usm_ndarray(a)

    # use DPCTL tensor function to fill the сopy of the input array
    # from the input array
    a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
        src=a_usm_arr, dst=a_h.get_array(), sycl_queue=exec_q
    )

    k = min(m, n)
    if compute_uv:
        if full_matrices:
            u_shape = (m, m)
            vt_shape = (n, n)
            jobu = ord("A")
            jobvt = ord("A")
        else:
            u_shape = (m, k)
            vt_shape = (k, n)
            jobu = ord("S")
            jobvt = ord("S")
    else:
        u_shape = vt_shape = ()
        jobu = ord("N")
        jobvt = ord("N")

    # oneMKL LAPACK assumes fortran-like array as input.
    # Allocate 'F' order memory for dpnp output arrays to comply with
    # these requirements.
    u_h = dpnp.empty_like(
        a_h,
        shape=u_shape,
        order="F",
    )
    vt_h = dpnp.empty_like(
        a_h,
        shape=vt_shape,
        order="F",
    )
    s_h = dpnp.empty_like(a_h, shape=(k,), dtype=s_type)

    ht_lapack_ev, _ = li._gesvd(
        exec_q,
        jobu,
        jobvt,
        a_h.get_array(),
        s_h.get_array(),
        u_h.get_array(),
        vt_h.get_array(),
        [a_copy_ev],
    )

    if batch_call:
        if compute_uv:
            return u_h, s_h, vt_h, ht_lapack_ev, a_ht_copy_ev
        return s_h, ht_lapack_ev, a_ht_copy_ev

    ht_lapack_ev.wait()
    a_ht_copy_ev.wait()

    # TODO: Need to return C-contiguous array to match the output of
    # numpy.linalg.svd
    if compute_uv:
        return u_h, s_h, vt_h
    return s_h
