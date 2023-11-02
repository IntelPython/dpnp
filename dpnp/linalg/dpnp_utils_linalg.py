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
from numpy import prod

import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li

__all__ = ["dpnp_eigh"]

_jobz = {"N": 0, "V": 1}
_upper_lower = {"U": 0, "L": 1}


def _stacked_identity(batch_shape, n, dtype, usm_type=None, sycl_queue=None):
    shape = batch_shape + (n, n)
    idx = dpnp.arange(n, usm_type=usm_type, sycl_queue=sycl_queue)
    x = dpnp.zeros(shape, dtype=dtype, usm_type=usm_type, sycl_queue=sycl_queue)
    x[..., idx, idx] = 1
    return x


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


def _dpnp_svd_batch(
    a, res_type, res_type_s, full_matrices=True, compute_uv=True
):
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
            dtype=res_type_s,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        if compute_uv:
            if full_matrices:
                u = dpnp.empty(
                    batch_shape_orig + (n, n),
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt = dpnp.empty(
                    batch_shape_orig + (m, m),
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
            else:
                u = dpnp.empty(
                    batch_shape_orig + (n, k),
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt = dpnp.empty(
                    batch_shape_orig + (k, m),
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
            return u, s, vt
        else:
            return s
    elif m == 0 or n == 0:
        s = dpnp.empty(
            batch_shape_orig + (0,),
            dtype=res_type_s,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        if compute_uv:
            if full_matrices:
                u = _stacked_identity(
                    batch_shape_orig,
                    n,
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt = _stacked_identity(
                    batch_shape_orig,
                    m,
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
            else:
                u = dpnp.empty(
                    batch_shape_orig + (n, 0),
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt = dpnp.empty(
                    batch_shape_orig + (0, m),
                    dtype=res_type,
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

    # TODO: Use linalg_common_type from #1598
    if dpnp.issubdtype(a.dtype, dpnp.floating):
        res_type = (
            a.dtype
            if a_sycl_queue.sycl_device.has_aspect_fp64
            else dpnp.float32
        )
    elif dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        res_type = (
            a.dtype
            if a_sycl_queue.sycl_device.has_aspect_fp64
            else dpnp.complex64
        )
    else:
        res_type = (
            dpnp.float64
            if a_sycl_queue.sycl_device.has_aspect_fp64
            else dpnp.float32
        )

    res_type_s = (
        dpnp.float64
        if a_sycl_queue.sycl_device.has_aspect_fp64
        and (res_type == dpnp.float64 or res_type == dpnp.complex128)
        else dpnp.float32
    )

    if a.ndim > 2:
        return _dpnp_svd_batch(
            a, res_type, res_type_s, full_matrices, compute_uv
        )

    else:
        n, m = a.shape

        if m == 0 or n == 0:
            s = dpnp.empty((0,), dtype=res_type_s)
            if compute_uv:
                if full_matrices:
                    u = dpnp.eye(n, dtype=res_type)
                    vt = dpnp.eye(m, dtype=res_type)
                else:
                    u = dpnp.empty((n, 0), dtype=res_type)
                    vt = dpnp.empty((0, m), dtype=res_type)
                return u, s, vt
            else:
                return s

        # `a`` must be copied because gesvd destroys the input matrix
        # `a` must be traspotted if m < n
        if m >= n:
            x = a
            a_h = dpnp.empty_like(a, order="C", dtype=res_type)
            trans_flag = False
        else:
            m, n = a.shape
            x = a.transpose()
            a_h = dpnp.empty_like(x, order="C", dtype=res_type)
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
                u_h = dpnp.empty(
                    (m, m),
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                vt_h = dpnp.empty(
                    (n, n),
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                jobu = ord("A")
                jobvt = ord("A")
            else:
                u_h = dpnp.empty_like(x, dtype=res_type)
                vt_h = dpnp.empty(
                    (k, n),
                    dtype=res_type,
                    usm_type=a_usm_type,
                    sycl_queue=a_sycl_queue,
                )
                jobu = ord("S")
                jobvt = ord("S")
        else:
            u_h = dpnp.empty(
                [],
                dtype=res_type,
                usm_type=a_usm_type,
                sycl_queue=a_sycl_queue,
            )
            vt_h = dpnp.empty(
                [],
                dtype=res_type,
                usm_type=a_usm_type,
                sycl_queue=a_sycl_queue,
            )
            jobu = ord("N")
            jobvt = ord("N")

        s_h = dpnp.empty(
            k, dtype=res_type_s, usm_type=a_usm_type, sycl_queue=a_sycl_queue
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
