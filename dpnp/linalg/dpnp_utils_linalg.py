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


import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li

import dpctl.tensor._tensor_impl as ti

__all__ = [
    "dpnp_eigh"
]

_jobz = {'N': 0, 'V': 1}
_upper_lower = {'U': 0, 'L': 1}


def dpnp_eigh(a, UPLO):
    """
    Return the eigenvalues and eigenvectors of a complex Hermitian
    (conjugate symmetric) or a real symmetric matrix.

    The main calculation is done by calling an extention function
    for LAPACK library of OneMKL. Depending on input type of `a` array,
    it will be either ``heevd`` (for complex types) or ``syevd`` (for others).

    """

    a_usm_type = a.usm_type
    a_sycl_queue = a.sycl_queue
    a_order = 'C' if a.flags.c_contiguous else 'F'
    a_usm_arr = dpnp.get_usm_ndarray(a)

    # 'V' means both eigenvectors and eigenvalues will be calculated
    jobz = _jobz['V']
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
        w = dpnp.empty(a.shape[:-1], dtype=w_type, usm_type=a_usm_type, sycl_queue=a_sycl_queue)

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
            eig_vecs[i] = dpnp.empty_like(a[i], order='F', dtype=v_type)

            # use DPCTL tensor function to fill the array of eigenvectors with content of input array
            ht_copy_ev[i], copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(src=a_usm_arr[i], dst=eig_vecs[i].get_array(), sycl_queue=a_sycl_queue)

            # call LAPACK extension function to get eigenvalues and eigenvectors of a portion of matrix A
            ht_lapack_ev[i], _ = getattr(li, lapack_func)(a_sycl_queue, jobz, uplo, eig_vecs[i].get_array(), w[i].get_array(), depends=[copy_ev])

        for i in range(op_count):
            ht_lapack_ev[i].wait()
            ht_copy_ev[i].wait()

        # combine the list of eigenvectors into a single array
        v = dpnp.array(eig_vecs, order=a_order)
        return w, v
    else:
        # oneMKL LAPACK assumes fortran-like array as input, so
        # allocate a memory with 'F' order for dpnp array of eigenvectors
        v = dpnp.empty_like(a, order='F', dtype=v_type)

        # use DPCTL tensor function to fill the array of eigenvectors with content of input array
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(src=a_usm_arr, dst=v.get_array(), sycl_queue=a_sycl_queue)

        # allocate a memory for dpnp array of eigenvalues
        w = dpnp.empty(a.shape[:-1], dtype=w_type, usm_type=a_usm_type, sycl_queue=a_sycl_queue)

        # call LAPACK extension function to get eigenvalues and eigenvectors of matrix A
        ht_lapack_ev, lapack_ev = getattr(li, lapack_func)(a_sycl_queue, jobz, uplo, v.get_array(), w.get_array(), depends=[copy_ev])

        if a_order != 'F':
            # need to align order of eigenvectors with one of input matrix A
            out_v = dpnp.empty_like(v, order=a_order)
            ht_copy_out_ev, _ = ti._copy_usm_ndarray_into_usm_ndarray(src=v.get_array(), dst=out_v.get_array(), sycl_queue=a_sycl_queue, depends=[lapack_ev])
            ht_copy_out_ev.wait()
        else:
            out_v = v

        ht_lapack_ev.wait()
        ht_copy_ev.wait()

        return w, out_v
