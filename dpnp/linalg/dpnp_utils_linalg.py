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
from numpy import issubdtype, prod

import dpnp
import dpnp.backend.extensions.lapack._lapack_impl as li
from dpnp.dpnp_utils import get_usm_allocations

__all__ = [
    "check_stacked_2d",
    "check_stacked_square",
    "dpnp_cholesky",
    "dpnp_det",
    "dpnp_eigh",
    "dpnp_inv",
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


def _check_lapack_dev_info(dev_info, error_msg=None):
    """
    Check `dev_info` from OneMKL LAPACK routines, raising an error for failures.

    Parameters
    ----------
    dev_info : list of ints
        Each element of the list indicates the status of OneMKL LAPACK routine calls.
        A non-zero value signifies a failure.

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

        raise dpnp.linalg.LinAlgError(error_msg)


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

    # Set `uplo` value for `potrf` and `potrf_batch` function based on the boolean input `upper`.
    # In oneMKL, `uplo` value of 1 is equivalent to oneapi::mkl::uplo::lower
    # and `uplo` value of 0 is equivalent to oneapi::mkl::uplo::upper.
    # However, we adjust this logic based on the array's memory layout.
    # Note: lower for row-major (which is used here) is upper for column-major layout.
    # Reference: comment from tbmkl/tests/lapack/unit/dpcpp/potrf_usm/potrf_usm.cpp
    # This means that if `upper` is False (lower triangular),
    # we actually use oneapi::mkl::uplo::upper (0) for the row-major layout, and vice versa.
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


def dpnp_inv_batched(a, res_type):
    """
    dpnp_inv_batched(a, res_type)

    Return the inverses of each matrix in a batch of matrices `a`.

    The inverse of a matrix is such that if it is multiplied by the original matrix,
    it results in the identity matrix. This function computes the inverses of a batch
    of square matrices.
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


def dpnp_inv(a):
    """
    dpnp_inv(a)

    Return the inverse of `a` matrix.

    The inverse of a matrix is such that if it is multiplied by the original matrix,
    it results in the identity matrix. This function computes the inverse of a single
    square matrix.

    """

    res_type = _common_type(a)
    if a.size == 0:
        return dpnp.empty_like(a, dtype=res_type)

    if a.ndim >= 3:
        return dpnp_inv_batched(a, res_type)

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


def dpnp_qr_batch(a, mode="reduced"):
    """
    dpnp_qr_batch(a, mode="reduced")

    Return the batched qr factorization of `a` matrix.

    """

    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    m, n = a.shape[-2:]
    k = min(m, n)

    batch_shape = a.shape[:-2]
    batch_size = prod(batch_shape)

    res_type = _common_type(a)

    if batch_size == 0 or k == 0:
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
        elif mode == "complete":
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
        elif mode == "r":
            return dpnp.empty_like(
                a,
                shape=batch_shape + (k, n),
                dtype=res_type,
            )
        elif mode == "raw":
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

    # Call the LAPACK extension function _geqrf_batch to compute the QR factorization
    # of a general m x n matrix.
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

    ht_geqrf_batch_ev.wait()
    a_ht_copy_ev.wait()

    if mode == "r":
        r = a_t[..., :k].swapaxes(-2, -1)
        out_r = dpnp.triu(r)
        return out_r.reshape(batch_shape + out_r.shape[-2:])

    if mode == "raw":
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
    q[..., :n, :] = a_t

    q_stride = q.strides[0]
    tau_stride = tau_h.strides[0]

    # Get LAPACK function (_orgqr_batch for real or _ungqf_batch for complex data types)
    # for QR factorization
    lapack_func = (
        "_ungqr_batch"
        if dpnp.issubdtype(res_type, dpnp.complexfloating)
        else "_orgqr_batch"
    )

    # Call the LAPACK extension function _orgqr_batch/ to generate the real orthogonal/
    # complex unitary matrices `Qi` of the QR factorization
    # for a batch of general matrices.
    ht_lapack_ev, _ = getattr(li, lapack_func)(
        a_sycl_queue,
        q.get_array(),
        tau_h.get_array(),
        m,
        mc,
        k,
        q_stride,
        tau_stride,
        batch_size,
        [geqrf_batch_ev],
    )

    ht_lapack_ev.wait()

    q = q[..., :mc, :].swapaxes(-2, -1)
    r = a_t[..., :mc].swapaxes(-2, -1)

    r_triu = dpnp.triu(r)

    return (
        q.reshape(batch_shape + q.shape[-2:]),
        r_triu.reshape(batch_shape + r_triu.shape[-2:]),
    )


def dpnp_qr(a, mode="reduced"):
    """
    dpnp_qr(a, mode="reduced")

    Return the qr factorization of `a` matrix.

    """

    if a.ndim > 2:
        return dpnp_qr_batch(a, mode=mode)

    a_usm_arr = dpnp.get_usm_ndarray(a)
    a_sycl_queue = a.sycl_queue
    a_usm_type = a.usm_type

    res_type = _common_type(a)

    m, n = a.shape
    k = min(m, n)
    if k == 0:
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
        elif mode == "complete":
            return dpnp.identity(
                m, dtype=res_type, sycl_queue=a_sycl_queue, usm_type=a_usm_type
            ), dpnp.empty_like(
                a,
                shape=(m, n),
                dtype=res_type,
            )
        elif mode == "r":
            return dpnp.empty_like(
                a,
                shape=(0, n),
                dtype=res_type,
            )
        else:  # mode == "raw"
            return dpnp.empty_like(
                a,
                shape=(n, m),
                dtype=res_type,
            ), dpnp.empty_like(
                a,
                shape=(0,),
                dtype=res_type,
            )

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

    ht_geqrf_ev.wait()
    a_ht_copy_ev.wait()

    if mode == "r":
        r = a_t[:, :k].transpose()
        return dpnp.triu(r)

    if mode == "raw":
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
    q[:n] = a_t

    # Get LAPACK function (_orgqr for real or _ungqf for complex data types)
    # for QR factorization
    lapack_func = (
        "_ungqr"
        if dpnp.issubdtype(res_type, dpnp.complexfloating)
        else "_orgqr"
    )

    # Call the LAPACK extension function _orgqr/_ungqf to generate the real orthogonal/
    # complex unitary matrix `Q` of the QR factorization
    ht_lapack_ev, _ = getattr(li, lapack_func)(
        a_sycl_queue, m, mc, k, q.get_array(), tau_h.get_array(), [geqrf_ev]
    )

    ht_lapack_ev.wait()

    q = q[:mc].transpose()
    r = a_t[:, :mc].transpose()
    return (q, dpnp.triu(r))


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
