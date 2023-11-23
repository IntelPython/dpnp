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

import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li

__all__ = ["dpnp_eigh", "_lu_factor"]

_jobz = {"N": 0, "V": 1}
_upper_lower = {"U": 0, "L": 1}


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


def _lu_factor(a, res_type):
    """Compute pivoted LU decomposition.

    Decompose a given batch of square matrices. Inputs and outputs are
    transposed.

    Args:
        a (dpnp.ndarray): The input matrix with dimension ``(..., N, N)``.
           The dimension condition is not checked.
        res_type (dpnp.dtype): float32, float64, complex64 or complex128.

    Returns:
        tuple:
        lu_t (dpnp.ndarray):
            ``L`` without its unit diagonal and ``U`` with
            dimension ``(..., N, N)``.
        piv (dpnp.ndarray):
            1-origin pivot indices with dimension
            ``(..., N)``.

    See Also
    --------
    :obj:`scipy.linalg.lu_factor`

    """

    # TODO: use dpnp.linalg.LinAlgError
    if a.ndim < 2:
        raise ValueError(
            f"{a.ndim}-dimensional array given. The input "
            "array must be at least two-dimensional"
        )

    n, m = a.shape[-2:]
    # TODO: use dpnp.linalg.LinAlgError
    if m != n:
        raise ValueError("Last 2 dimensions of the input array must be square")

    a_order = "C" if a.flags.c_contiguous else "F"
    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    # TODO: use getrf_batch

    if a.ndim > 2:
        orig_shape = a.shape
        a = a.reshape(-1, n, n)
        batch_size = a.shape[0]
        a_usm_arr = dpnp.get_usm_ndarray(a)

        a_vecs = [None] * batch_size
        ipiv_h = dpnp.empty(
            (batch_size, n),
            dtype=dpnp.int64,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        dev_info_h = dpnp.empty(
            (batch_size,),
            dtype=dpnp.int64,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )
        a_ht_copy_ev = [None] * batch_size
        ht_lapack_ev = [None] * batch_size

        for i in range(batch_size):
            a_vecs[i] = dpnp.empty_like(a[i], order="F", dtype=res_type)
            a_ht_copy_ev[i], a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_usm_arr[i],
                dst=a_vecs[i].get_array(),
                sycl_queue=a_sycl_queue,
            )

            ht_lapack_ev[i], _ = li._getrf(
                a_sycl_queue,
                n,
                a_vecs[i].get_array(),
                ipiv_h[i].get_array(),
                dev_info_h[i].get_array(),
                [a_copy_ev],
            )

        for i in range(batch_size):
            ht_lapack_ev[i].wait()
            a_ht_copy_ev[i].wait()

        out_v = dpnp.array(a_vecs, order=a_order).reshape(orig_shape)
        out_ipiv = ipiv_h.reshape(orig_shape[:-1])
        out_dev_info = dev_info_h.reshape(orig_shape[:-2])

        return (out_v, out_ipiv, out_dev_info)

    else:
        a_usm_arr = dpnp.get_usm_ndarray(a)

        a_h = dpnp.empty_like(a, order="F", dtype=res_type)
        ipiv_h = dpnp.empty(
            n, dtype=dpnp.int64, usm_type=a_usm_type, sycl_queue=a_sycl_queue
        )
        dev_info_h = dpnp.empty(
            1, dtype=dpnp.int64, usm_type=a_usm_type, sycl_queue=a_sycl_queue
        )

        a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_usm_arr, dst=a_h.get_array(), sycl_queue=a_sycl_queue
        )

        ht_lapack_ev, lapack_ev = li._getrf(
            a_sycl_queue,
            n,
            a_h.get_array(),
            ipiv_h.get_array(),
            dev_info_h.get_array(),
            [a_copy_ev],
        )

        if a_order != "F":
            # need to align order of the result of solutions with the
            # input array of multiple dependent variables
            a_h_f = dpnp.empty_like(a_h, order=a_order)
            ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=a_h.get_array(),
                dst=a_h_f.get_array(),
                sycl_queue=a_sycl_queue,
                depends=[lapack_ev],
            )
            ht_copy_out_ev.wait()
            out_v = (a_h_f, ipiv_h, dev_info_h)
        else:
            out_v = (a_h, ipiv_h, dev_info_h)

        ht_lapack_ev.wait()
        a_ht_copy_ev.wait()

        return out_v
