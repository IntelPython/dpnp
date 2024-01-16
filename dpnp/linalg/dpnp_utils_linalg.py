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


import dpctl.tensor._tensor_impl as ti
from numpy import issubdtype

import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li
from dpnp.dpnp_utils import get_usm_allocations

__all__ = [
    "check_stacked_2d",
    "check_stacked_square",
    "dpnp_cholesky",
    "dpnp_det",
    "dpnp_eigh",
    "dpnp_slogdet",
    "dpnp_solve",
]

_jobz = {"N": 0, "V": 1}
_upper_lower = {"U": 0, "L": 1}

_real_types_map = {
    "float32": "float32",  # single : single
    "float64": "float64",  # double : double
    "complex64": "float32",  # csingle : csingle
    "complex128": "float64",  # cdouble : cdouble
}


def _calculate_determinant_sign(ipiv, diag, res_type, n):
    """
    Calculate the sign of the determinant based on row exchanges and diagonal values.

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


def _real_type(dtype, device=None):
    """
    Returns the real data type corresponding to a given dpnp data type.

    Parameters
    ----------
    dtype : dpnp.dtype
        The dtype for which to find the corresponding real data type.
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where an array of default floating type might be created.

    Returns
    -------
    out : str
        The name of the real data type.

    """

    default = dpnp.default_float_type(device)
    real_type = _real_types_map.get(dtype.name, default)
    return dpnp.dtype(real_type)


def check_stacked_2d(*arrays):
    """
    Return ``True`` if each array in `arrays` has at least two dimensions.

    If any array is less than two-dimensional, `dpnp.linalg.LinAlgError` will be raised.

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
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
    arrays : {dpnp.ndarray, usm_ndarray}
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
    Common type for linear algebra operations.

    This function determines the common data type for linalg operations.
    It's designed to be similar in logic to `numpy.linalg.linalg._commonType`.

    Key differences from `numpy.common_type`:
    - It accepts ``bool_`` arrays.
    - The default floating-point data type is determined by the capabilities of the device
      on which `arrays` are created, as indicated by `dpnp.default_float_type()`.

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

    default = dpnp.default_float_type(device=arrays[0].device)
    dtype_common = _common_inexact_type(default, *dtypes)

    return dtype_common


def _common_inexact_type(default_dtype, *dtypes):
    """
    Determines the common 'inexact' data type for linear algebra operations.

    This function selects an 'inexact' data type appropriate for the device's capabilities.
    It defaults to `default_dtype` when provided types are not 'inexact'.

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
        dt if issubdtype(dt, dpnp.inexact) else default_dtype for dt in dtypes
    ]
    return dpnp.result_type(*inexact_dtypes)


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
            1-origin pivot indices indicating row permutations during decomposition.
        dev_info : (...) {dpnp.ndarray, usm_ndarray}
            Information on `getrf` or `getrf_batch` computation success (0 for success).

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

        else:
            # Initialize lists for storing arrays and events for each batch
            a_vecs = [None] * batch_size
            ipiv_vecs = [None] * batch_size
            dev_info_vecs = [None] * batch_size
            a_ht_copy_ev = [None] * batch_size
            ht_lapack_ev = [None] * batch_size

            # Process each batch
            for i in range(batch_size):
                # Copy each 2D slice to a new array as getrf destroys the input matrix
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

    else:
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


def dpnp_cholesky_batch(a, res_type):
    """
    dpnp_cholesky_batch(a, res_type)

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
        n,
        a_stride,
        batch_size,
        [a_copy_ev],
    )

    ht_lapack_ev.wait()
    a_ht_copy_ev.wait()

    a_h = dpnp.tril(a_h.reshape(orig_shape))

    return a_h


def dpnp_cholesky(a):
    """
    dpnp_cholesky(a)

    Return the Cholesky decomposition of `a` array.

    """

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    res_type = _common_type(a)

    a_shape = a.shape
    n = a.shape[-2]

    if a.size == 0:
        return dpnp.empty(
            a_shape,
            dtype=res_type,
            usm_type=a_usm_type,
            sycl_queue=a_sycl_queue,
        )

    if a.ndim > 2:
        return dpnp_cholesky_batch(a, res_type)

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
        n,
        a_h.get_array(),
        [a_copy_ev],
    )

    ht_lapack_ev.wait()
    a_ht_copy_ev.wait()

    a_h = dpnp.tril(a_h)

    return a_h


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

    # Transposing 'lu' to swap the last two axes for compatibility
    # with 'dpnp.diagonal' as it does not support 'axis1' and 'axis2' arguments.
    # TODO: Replace with 'dpnp.diagonal(lu, axis1=-2, axis2=-1)' when supported.
    lu_transposed = lu.transpose(-2, -1, *range(lu.ndim - 2))
    diag = dpnp.diagonal(lu_transposed)

    det = dpnp.prod(dpnp.abs(diag), axis=-1)

    sign = _calculate_determinant_sign(ipiv, diag, res_type, n)

    det = sign * det
    det = det.astype(res_type, copy=False)
    singular = dev_info > 0
    det = dpnp.where(singular, res_type.type(0), det)

    return det.reshape(shape)


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

    # Transposing 'lu' to swap the last two axes for compatibility
    # with 'dpnp.diagonal' as it does not support 'axis1' and 'axis2' arguments.
    # TODO: Replace with 'dpnp.diagonal(lu, axis1=-2, axis2=-1)' when supported.
    lu_transposed = lu.transpose(-2, -1, *range(lu.ndim - 2))
    diag = dpnp.diagonal(lu_transposed)

    logdet = dpnp.log(dpnp.abs(diag)).sum(axis=-1)

    sign = _calculate_determinant_sign(ipiv, diag, res_type, n)

    logdet = logdet.astype(logdet_dtype, copy=False)
    singular = dev_info > 0
    return (
        dpnp.where(singular, res_type.type(0), sign).reshape(shape),
        dpnp.where(singular, logdet_dtype.type("-inf"), logdet).reshape(shape),
    )
