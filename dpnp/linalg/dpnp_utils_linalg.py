# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2023, Intel Corporation
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


import dpctl.tensor._tensor_impl as ti
from numpy import issubdtype, prod

import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li
from dpnp.dpnp_utils import get_usm_allocations

__all__ = [
    "check_stacked_2d",
    "check_stacked_square",
    "dpnp_eigh",
    "dpnp_solve",
    "dpnp_svd",
]

_jobz = {"N": 0, "V": 1}
_upper_lower = {"U": 0, "L": 1}


def _stacked_identity(batch_shape, n, dtype, usm_type=None, sycl_queue=None):
    shape = batch_shape + (n, n)
    idx = dpnp.arange(n, usm_type=usm_type, sycl_queue=sycl_queue)
    x = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type, sycl_queue=sycl_queue)
    x[..., idx, idx] = 1
    return x


def check_stacked_2d(*arrays):
    """
    Return ``True`` if each array in `arrays` has at least two dimensions.

    If any array is less than two-dimensional, `dpnp.linalg.LinAlgError` will be raised.

    Parameters
    ----------
    arrays : {dpnp_array, usm_ndarray}
        A sequence of input arrays to check for dimensionality.

    Returns
    -------
    out : bool
        ``True`` if each array in `arrays` is at least two-dimensional.

    Raises
    ------
    dpnp.linalg.LinAlgError
        If any array in `arrays` is less than two-dimensional.

    """

    for a in arrays:
        if a.ndim < 2:
            raise dpnp.linalg.LinAlgError(
                f"{a.ndim}-dimensional array given. The input "
                "array must be at least two-dimensional"
            )


def check_stacked_square(*arrays):
    """
    Return ``True`` if each array in `arrays` is a square matrix.

    If any array does not form a square matrix, `dpnp.linalg.LinAlgError` will be raised.

    Precondition: `arrays` are at least 2d. The caller should assert it
    beforehand. For example,

    >>> def solve(a):
    ...     check_stacked_2d(a)
    ...     check_stacked_square(a)
    ...     ...

    Parameters
    ----------
    arrays : {dpnp_array, usm_ndarray}
        A sequence of input arrays to check for square matrix shape.

    Returns
    -------
    out : bool
        ``True`` if each array in `arrays` forms a square matrix.

    Raises
    ------
    dpnp.linalg.LinAlgError
        If any array in `arrays` does not form a square matrix.

    """

    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise dpnp.linalg.LinAlgError(
                "Last 2 dimensions of the input array must be square"
            )


def _common_type(*arrays):
    """
    _common_type(*arrays)

    Common type for linear algebra operations.

    This function determines the common data type for linalg operations.
    It's designed to be similar in logic to `numpy.linalg.linalg._commonType`.

    Key differences from `numpy.common_type`:
    - It accepts ``bool_`` arrays.
    - The default floating-point data type is determined by the capabilities of the device
      on which `arrays` are created, as indicated by `dpnp.default_float_type()`.

    Args:
        *arrays (dpnp.ndarray): Input arrays.

    Returns:
        dtype_common (dtype): The common data type for linalg operations.

        This returned value is applicable both as the precision to be used
        in linalg calls and as the dtype of (possibly complex) output(s).

    """

    dtypes = [arr.dtype for arr in arrays]

    default = dpnp.default_float_type(device=arrays[0].device)
    dtype_common = _common_inexact_type(default, *dtypes)

    return dtype_common


def _common_inexact_type(default_dtype, *dtypes):
    """
    _common_inexact_type(default_dtype, *dtypes)

    Determines the common 'inexact' data type for linear algebra operations.

    This function selects an 'inexact' data type appropriate for the device's capabilities.
    It defaults to `default_dtype` when provided types are not 'inexact'.

    Args:
        default_dtype: The default data type. This is determined by the capabilities of
        the device and is used when none of the provided types are 'inexact'.
        *dtypes: A variable number of data types to be evaluated to find
        the common 'inexact' type.

    Returns:
        dpnp.result_type (dtype) : The resultant 'inexact' data type for linalg operations,
        ensuring computational compatibility.

    """
    inexact_dtypes = [
        dt if issubdtype(dt, dpnp.inexact) else default_dtype for dt in dtypes
    ]
    return dpnp.result_type(*inexact_dtypes)


def dpnp_eigh(a, UPLO):
    """
    dpnp_eigh(a, UPLO)

    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.

    The main calculation is done by calling an extension function
    for LAPACK library of OneMKL. Depending on input type of `a` array,
    it will be either ``heevd`` (for complex types) or ``syevd`` (for others).

    """

    a_usm_type = a.usm_type
    a_sycl_queue = a.sycl_queue
    a_order = "C" if a.flags.c_contiguous else "F"
    a_usm_arr = dpnp.get_usm_ndarray(a)

    # 'V' means both eigenvectors and eigenvalues will be calculated
    jobz = _jobz["V"]
    uplo = _upper_lower[UPLO]

    # get resulting type of arrays with eigenvalues and eigenvectors
    a_dtype = a.dtype
    lapack_func = "_syevd"
    if dpnp.issubdtype(a_dtype, dpnp.complexfloating):
        lapack_func = "_heevd"
        v_type = a_dtype
        w_type = dpnp.float64 if a_dtype == dpnp.complex128 else dpnp.float32
    elif dpnp.issubdtype(a_dtype, dpnp.floating):
        v_type = w_type = a_dtype
    elif a_sycl_queue.sycl_device.has_aspect_fp64:
        v_type = w_type = dpnp.float64
    else:
        v_type = w_type = dpnp.float32

    if a.ndim > 2:
        w = dpnp.empty(
            a.shape[:-1],
            dtype=w_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )

        # need to loop over the 1st dimension to get eigenvalues and eigenvectors of 3d matrix A
        op_count = a.shape[0]
        if op_count == 0:
            return w, dpnp.empty_like(a, dtype=v_type)

        eig_vecs = [None] * op_count
        ht_copy_ev = [None] * op_count
        ht_lapack_ev = [None] * op_count
        for i in range(op_count):
            # oneMKL LAPACK assumes fortran-like array as input, so
            # allocate a memory with 'F' order for dpnp array of eigenvectors
            eig_vecs[i] = dpnp.empty_like(a[i], order="F", dtype=v_type)

            # use DPCTL tensor function to fill the array of eigenvectors with content of input array
            ht_copy_ev[i], copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_usm_arr[i],
                dst=eig_vecs[i].get_array(),
                sycl_queue=a_sycl_queue,
            )

            # call LAPACK extension function to get eigenvalues and eigenvectors of a portion of matrix A
            ht_lapack_ev[i], _ = getattr(li, lapack_func)(
                a_sycl_queue,
                jobz,
                uplo,
                eig_vecs[i].get_array(),
                w[i].get_array(),
                depends=[copy_ev],
            )

        for i in range(op_count):
            ht_lapack_ev[i].wait()
            ht_copy_ev[i].wait()

        # combine the list of eigenvectors into a single array
        v = dpnp.array(eig_vecs, order=a_order)
        return w, v
    else:
        # oneMKL LAPACK assumes fortran-like array as input, so
        # allocate a memory with 'F' order for dpnp array of eigenvectors
        v = dpnp.empty_like(a, order="F", dtype=v_type)

        # use DPCTL tensor function to fill the array of eigenvectors with content of input array
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_usm_arr, dst=v.get_array(), sycl_queue=a_sycl_queue
        )

        # allocate a memory for dpnp array of eigenvalues
        w = dpnp.empty(
            a.shape[:-1],
            dtype=w_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )

        # call LAPACK extension function to get eigenvalues and eigenvectors of matrix A
        ht_lapack_ev, lapack_ev = getattr(li, lapack_func)(
            a_sycl_queue,
            jobz,
            uplo,
            v.get_array(),
            w.get_array(),
            depends=[copy_ev],
        )

        if a_order != "F":
            # need to align order of eigenvectors with one of input matrix A
            out_v = dpnp.empty_like(v, order=a_order)
            ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=v.get_array(),
                dst=out_v.get_array(),
                sycl_queue=a_sycl_queue,
                depends=[lapack_ev],
            )
            ht_copy_out_ev.wait()
        else:
            out_v = v

        ht_lapack_ev.wait()
        ht_copy_ev.wait()

        return w, out_v


def dpnp_solve(a, b):
    """
    dpnp_solve(a, b)

    Return the solution to the system of linear equations with
    a square coefficient matrix `a` and multiple dependent variables
    array `b`.

    """

    a_usm_arr = dpnp.get_usm_ndarray(a)
    b_usm_arr = dpnp.get_usm_ndarray(b)

    b_order = "C" if b.flags.c_contiguous else "F"
    a_shape = a.shape
    b_shape = b.shape

    res_usm_type, exec_q = get_usm_allocations([a, b])

    res_type = _common_type(a, b)
    if b.size == 0:
        return dpnp.empty_like(b, dtype=res_type, usm_type=res_usm_type)

    if a.ndim > 2:
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
            # oneMKL LAPACK assumes fortran-like array as input, so
            # allocate a memory with 'F' order for dpnp array of coefficient matrix
            # and multiple dependent variables array
            coeff_vecs[i] = dpnp.empty_like(
                a[i], order="F", dtype=res_type, usm_type=res_usm_type
            )
            val_vecs[i] = dpnp.empty_like(
                b[i], order="F", dtype=res_type, usm_type=res_usm_type
            )

            # use DPCTL tensor function to fill the coefficient matrix array
            # and the array of multiple dependent variables with content
            # from the input arrays
            a_ht_copy_ev[i], a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_usm_arr[i],
                dst=coeff_vecs[i].get_array(),
                sycl_queue=a.sycl_queue,
            )
            b_ht_copy_ev[i], b_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=b_usm_arr[i],
                dst=val_vecs[i].get_array(),
                sycl_queue=b.sycl_queue,
            )

            # Call the LAPACK extension function _gesv to solve the system of linear
            # equations using a portion of the coefficient square matrix and a
            # corresponding portion of the dependent variables array.
            ht_lapack_ev[i], _ = li._gesv(
                exec_q,
                coeff_vecs[i].get_array(),
                val_vecs[i].get_array(),
                depends=[a_copy_ev, b_copy_ev],
            )

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
    else:
        # oneMKL LAPACK gesv overwrites `a` and `b` and assumes fortran-like array as input.
        # Allocate 'F' order memory for dpnp arrays to comply with these requirements.
        a_f = dpnp.empty_like(
            a, order="F", dtype=res_type, usm_type=res_usm_type
        )

        # use DPCTL tensor function to fill the coefficient matrix array
        # with content from the input array `a`
        a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_usm_arr, dst=a_f.get_array(), sycl_queue=a.sycl_queue
        )

        b_f = dpnp.empty_like(
            b, order="F", dtype=res_type, usm_type=res_usm_type
        )

        # use DPCTL tensor function to fill the array of multiple dependent variables
        # with content from the input array `b`
        b_ht_copy_ev, b_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=b_usm_arr, dst=b_f.get_array(), sycl_queue=b.sycl_queue
        )

        # Call the LAPACK extension function _gesv to solve the system of linear
        # equations with the coefficient square matrix and the dependent variables array.
        ht_lapack_ev, _ = li._gesv(
            exec_q, a_f.get_array(), b_f.get_array(), [a_copy_ev, b_copy_ev]
        )

        ht_lapack_ev.wait()
        b_ht_copy_ev.wait()
        a_ht_copy_ev.wait()

        return b_f


def _dpnp_svd_batch(a, uv_type, s_type, full_matrices=True, compute_uv=True):
    a_usm_type = a.usm_type
    a_sycl_queue = a.sycl_queue
    reshape = False
    batch_shape_orig = a.shape[:-2]

    if a.ndim > 3:
        # get 3d input arrays by reshape
        a = a.reshape(prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        reshape = True

    batch_shape = a.shape[:-2]
    batch_size = prod(batch_shape)
    n, m = a.shape[-2:]

    if batch_size == 0:
        k = min(m, n)
        s = dpnp.empty(
            batch_shape_orig + (k,),
            dtype=s_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        if compute_uv:
            if full_matrices:
                u = dpnp.empty(
                    batch_shape_orig + (n, n),
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt = dpnp.empty(
                    batch_shape_orig + (m, m),
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
            else:
                u = dpnp.empty(
                    batch_shape_orig + (n, k),
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt = dpnp.empty(
                    batch_shape_orig + (k, m),
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
            return u, s, vt
        else:
            return s
    elif m == 0 or n == 0:
        s = dpnp.empty(
            batch_shape_orig + (0,),
            dtype=s_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        if compute_uv:
            if full_matrices:
                u = _stacked_identity(
                    batch_shape_orig,
                    n,
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt = _stacked_identity(
                    batch_shape_orig,
                    m,
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
            else:
                u = dpnp.empty(
                    batch_shape_orig + (n, 0),
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt = dpnp.empty(
                    batch_shape_orig + (0, m),
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
            return u, s, vt
        else:
            return s

    u_matrices = [None] * batch_size
    s_matrices = [None] * batch_size
    vt_matrices = [None] * batch_size
    for i in range(batch_size):
        if compute_uv:
            vt_matrices[i], s_matrices[i], u_matrices[i] = dpnp_svd(
                a[i], full_matrices, compute_uv
            )
        else:
            s_matrices[i] = dpnp_svd(a[i], full_matrices, compute_uv)

    if compute_uv:
        out_s = dpnp.array(s_matrices)
        out_vt = dpnp.array(vt_matrices)
        out_u = dpnp.array(u_matrices)
        if reshape:
            return (
                out_vt.reshape(batch_shape_orig + out_vt.shape[-2:]),
                out_s.reshape(batch_shape_orig + out_s.shape[-1:]),
                out_u.reshape(batch_shape_orig + out_u.shape[-2:]),
            )
        else:
            return out_vt, out_s, out_u
    else:
        out_s = dpnp.array(s_matrices)
        if reshape:
            return out_s.reshape(batch_shape_orig + out_s.shape[-1:])
        else:
            return out_s


def dpnp_svd(a, full_matrices=True, compute_uv=True):
    """
    dpnp_svd(a)

    Return the singular value decomposition (SVD).
    """

    a_usm_type = a.usm_type
    a_sycl_queue = a.sycl_queue

    uv_type = _common_type(a)
    s_type = uv_type.char.lower()

    if a.ndim > 2:
        return _dpnp_svd_batch(a, uv_type, s_type, full_matrices, compute_uv)

    else:
        n, m = a.shape

        if m == 0 or n == 0:
            s = dpnp.empty(
                (0,),
                dtype=s_type,
                usm_type=a_usm_type,
                sycl_queue=a_sycl_queue,
            )
            if compute_uv:
                if full_matrices:
                    u_shape = (n,)
                    vt_shape = (m,)
                else:
                    u_shape = (n, 0)
                    vt_shape = (0, m)

                u = dpnp.eye(
                    *u_shape,
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt = dpnp.eye(
                    *vt_shape,
                    dtype=uv_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                return u, s, vt
            else:
                return s

        # `a` must be copied because gesvd destroys the input matrix
        # `a` must be traspotted if m < n
        if m >= n:
            x = a
            a_h = dpnp.empty_like(a, order="C", dtype=uv_type)
            trans_flag = False
        else:
            m, n = a.shape
            x = a.transpose()
            a_h = dpnp.empty_like(x, order="C", dtype=uv_type)
            trans_flag = True

        a_usm_arr = dpnp.get_usm_ndarray(x)

        # use DPCTL tensor function to fill the сopy of the input array
        # from the input array
        a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_usm_arr, dst=a_h.get_array(), sycl_queue=a_sycl_queue
        )

        k = n  # = min(m, n) where m >= n is ensured above
        if compute_uv:
            if full_matrices:
                u_shape = (m, m)
                vt_shape = (n, n)
                jobu = ord("A")
                jobvt = ord("A")
            else:
                u_shape = x.shape
                vt_shape = (k, n)
                jobu = ord("S")
                jobvt = ord("S")
        else:
            u_shape = vt_shape = ()
            jobu = ord("N")
            jobvt = ord("N")

        u_h = dpnp.empty(
            u_shape,
            dtype=uv_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        vt_h = dpnp.empty(
            vt_shape,
            dtype=uv_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        s_h = dpnp.empty(
            k, dtype=s_type, usm_type=a_usm_type, sycl_queue=a_sycl_queue
        )

        ht_lapack_ev, _ = li._gesvd(
            a_sycl_queue,
            jobu,
            jobvt,
            m,
            n,
            a_h.get_array(),
            s_h.get_array(),
            u_h.get_array(),
            vt_h.get_array(),
            [a_copy_ev],
        )

        ht_lapack_ev.wait()
        a_ht_copy_ev.wait()

        if compute_uv:
            if trans_flag:
                return u_h.transpose(), s_h, vt_h.transpose()
            else:
                return vt_h, s_h, u_h
        else:
            return s_h
