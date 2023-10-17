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


import dpctl
import dpctl.tensor._tensor_impl as ti
from numpy import prod

import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li

__all__ = ["dpnp_eigh"]

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

    if a.dtype != b.dtype:
        raise ValueError("a and b must be of the same type")

    exec_q = dpctl.utils.get_execution_queue((a.sycl_queue, b.sycl_queue))
    if exec_q is None:
        raise ValueError(
            "Execution placement can not be unambiguously inferred "
            "from input arguments."
        )

    if dpnp.issubdtype(a.dtype, dpnp.floating):
        res_type = (
            a.dtype if exec_q.sycl_device.has_aspect_fp64 else dpnp.float32
        )
    elif dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        res_type = (
            a.dtype if exec_q.sycl_device.has_aspect_fp64 else dpnp.complex64
        )
    else:
        res_type = (
            dpnp.float64 if exec_q.sycl_device.has_aspect_fp64 else dpnp.float32
        )

    if b.size == 0:
        return dpnp.empty(b.shape, dtype=res_type)

    if a.ndim > 2:
        reshape = False
        orig_shape_b = b.shape
        if a.ndim > 3:
            # get 3d input arrays by reshape
            if a.ndim == b.ndim:
                b = b.reshape(prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
            else:
                b = b.reshape(prod(b.shape[:-1]), b.shape[-1])

            a = a.reshape(prod(a.shape[:-2]), a.shape[-2], a.shape[-1])

            a_usm_arr = dpnp.get_usm_ndarray(a)
            b_usm_arr = dpnp.get_usm_ndarray(b)
            reshape = True

        op_count = a.shape[0]
        if op_count == 0:
            return dpnp.empty_like(b, dtype=res_type)

        coeff_vecs = [None] * op_count
        val_vecs = [None] * op_count
        a_ht_copy_ev = [None] * op_count
        b_ht_copy_ev = [None] * op_count
        ht_lapack_ev = [None] * op_count

        for i in range(op_count):
            # oneMKL LAPACK assumes fortran-like array as input, so
            # allocate a memory with 'F' order for dpnp array of coefficient matrix
            # and multiple dependent variables array
            coeff_vecs[i] = dpnp.empty_like(a[i], order="F", dtype=res_type)
            val_vecs[i] = dpnp.empty_like(b[i], order="F", dtype=res_type)

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
            ht_lapack_ev[i] = li._gesv(
                exec_q,
                coeff_vecs[i].get_array(),
                val_vecs[i].get_array(),
                depends=[a_copy_ev, b_copy_ev],
            )

        for i in range(op_count):
            ht_lapack_ev[i].wait()
            b_ht_copy_ev[i].wait()
            a_ht_copy_ev[i].wait()

        # combine the list of solutions into a single array
        out_v = dpnp.array(val_vecs, order=b_order)
        if reshape:
            # shape of the out_v must be equal to the shape of the array of
            # dependent variables
            out_v = out_v.reshape(orig_shape_b)
        return out_v
    else:
        # oneMKL LAPACK assumes fortran-like array as input, so
        # allocate a memory with 'F' order for dpnp array of coefficient matrix
        # and multiple dependent variables
        a_f = dpnp.empty_like(a, order="F", dtype=res_type)
        b_f = dpnp.empty_like(b, order="F", dtype=res_type)

        # use DPCTL tensor function to fill the coefficient matrix array
        # and the array of multiple dependent variables with content
        # from the input arrays
        a_ht_copy_ev, a_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=a_usm_arr, dst=a_f.get_array(), sycl_queue=a.sycl_queue
        )
        b_ht_copy_ev, b_copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=b_usm_arr, dst=b_f.get_array(), sycl_queue=b.sycl_queue
        )

        # Call the LAPACK extension function _gesv to solve the system of linear
        # equations with the coefficient square matrix and the dependent variables array.
        lapack_ev = li._gesv(
            exec_q, a_f.get_array(), b_f.get_array(), [a_copy_ev, b_copy_ev]
        )

        if b_order != "F":
            # need to align order of the result of solutions with the
            # input array of multiple dependent variables
            out_v = dpnp.empty_like(b_f, order=b_order)
            ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
                src=b_f.get_array(),
                dst=out_v.get_array(),
                sycl_queue=b.sycl_queue,
                depends=[lapack_ev],
            )
            ht_copy_out_ev.wait()
        else:
            out_v = b_f

        lapack_ev.wait()
        b_ht_copy_ev.wait()
        a_ht_copy_ev.wait()

        return out_v
