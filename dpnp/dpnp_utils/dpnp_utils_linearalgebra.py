# *****************************************************************************
# Copyright (c) 2024, Intel Corporation
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
import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as tei
import dpctl.tensor._tensor_impl as ti
import numpy
from numpy.core.numeric import normalize_axis_tuple

import dpnp
import dpnp.backend.extensions.blas._blas_impl as bi
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations

__all__ = ["dpnp_cross", "dpnp_dot", "dpnp_kron", "dpnp_matmul"]


def _create_result_array(x1, x2, out, shape, dtype, usm_type, sycl_queue):
    """
    Create the result array.

    If `out` is not ``None`` and its features match the specified `shape`, `dtype,
    `usm_type`, and `sycl_queue` and it is C-contiguous or F-contiguous and
    does not have any memory overlap with `x1` and `x2`, `out` itself is returned.
    If these conditions are not satisfied, an empty array is returned with the
    specified `shape`, `dtype, `usm_type`, and `sycl_queue`.
    """

    if out is not None:
        x1_usm = dpnp.get_usm_ndarray(x1)
        x2_usm = dpnp.get_usm_ndarray(x2)
        out_usm = dpnp.get_usm_ndarray(out)

        if (
            out.dtype == dtype
            and out.shape == shape
            and out.usm_type == usm_type
            and out.sycl_queue == sycl_queue
            and out.flags.c_contiguous
            and not ti._array_overlap(x1_usm, out_usm)
            and not ti._array_overlap(x2_usm, out_usm)
        ):
            return out

    return dpnp.empty(
        shape,
        dtype=dtype,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def _copy_array(x, dep_events, host_events, contig_copy=False, dtype=None):
    """
    Creating a copy of input array if needed.

    If `contig_copy` is ``True``, a C-contiguous copy of input array is returned.
    In this case, the copy array has the input array data type unless `dtype` is
    determined.
    If `contig_copy` is ``False`` and input array data type is different than `dtype`,
    a C-contiguous copy of input array with specified `dtype` is returned.

    """

    if contig_copy:
        copy = contig_copy
    else:
        copy = x.dtype != dtype if dtype is not None else False

    if copy:
        x_copy = dpnp.empty_like(x, dtype=dtype, order="C")
        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=dpnp.get_usm_ndarray(x),
            dst=x_copy.get_array(),
            sycl_queue=x.sycl_queue,
        )
        dep_events.append(copy_ev)
        host_events.append(ht_copy_ev)
        return x_copy
    return x


def _gemm_batch_matmul(exec_q, x1, x2, res, x1_is_2D, x2_is_2D, dev_tasks_list):
    # If input array is F-contiguous, we need to change the order to C-contiguous.
    # because mkl::gemm_bacth needs each 2D array to be F-contiguous but
    # when the input array is F-contiguous, the data of 2D array
    # that needs to be called in mkl::gemm_batch are not contiguous.
    ht_tasks_list = []
    contig_copy = not x1.flags.c_contiguous
    x1 = _copy_array(x1, dev_tasks_list, ht_tasks_list, contig_copy=contig_copy)
    contig_copy = not x2.flags.c_contiguous
    x2 = _copy_array(x2, dev_tasks_list, ht_tasks_list, contig_copy=contig_copy)

    x1_strides = x1.strides
    x2_strides = x2.strides
    res_strides = res.strides

    # need to standardize to use in ti._contract_iter2
    x1_strides = _standardize_strides(x1_strides, x1_is_2D, x1.shape, x1.ndim)
    x2_strides = _standardize_strides(x2_strides, x2_is_2D, x2.shape, x2.ndim)

    batch_size = res.shape[:-2][0]
    stridea = x1_strides[0]
    strideb = x2_strides[0]
    stridec = res_strides[-3]

    if x1.ndim > 3:
        iter = ti._contract_iter2(
            res.shape[:-2], x1_strides[:-2], x2_strides[:-2]
        )

        if len(iter[0]) != 1:
            raise ValueError("Input arrays cannot be used in gemm_batch")
        batch_size = iter[0][0]
        stridea = iter[1][0]
        strideb = iter[3][0]

    ht_blas_ev, _ = bi._gemm_batch(
        exec_q,
        dpnp.get_usm_ndarray(x1),
        dpnp.get_usm_ndarray(x2),
        dpnp.get_usm_ndarray(res),
        batch_size,
        stridea,
        strideb,
        stridec,
        dev_tasks_list,
    )

    return ht_blas_ev, ht_tasks_list, res


def _op_res_dtype(*arrays, dtype, casting, sycl_queue):
    """
    _op_res_dtype(*arrays, dtype, casting, sycl_queue)

    Determines the output array data type and an intermediate data type
    used in performing calculations related to a specific math function.
    If dtype is ``None``, the output array data type of the operation is
    determined based on the Promotion Type Rule and device capabilities.
    Otherwise, `dtype` is used as output array dtype, if input arrays
    can cast to it according to the casting rule determined. If casting
    cannot be done, a ``TypeError`` is raised.
    The intermediate data type is the data type used for performing the math
    function calculations. If output array dtype is a floating-point data type,
    it is also used for the intermediate data type. If output array dtype is an
    integral data type, the default floating point data type of the device where
    input arrays are allocated on are used for intermediate data type.

    Parameters
    ----------
    arrays : {dpnp.ndarray, usm_ndarray}
        Input arrays.
    dtype : dtype
        If not ``None``, data type of the output array.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.
    sycl_queue : {SyclQueue}
        A SYCL queue to use for determining default floating point datat type.

    Returns
    -------
    op_dtype, res_dtype :
        `op_dtype` is the data type used in performing math function calculations.
        The input arrays of the math function are cast to `op_dtype` and then
        the calculations are performed.
        `res_dtype` is the output data type. When the result is obtained, it is cast
        to `res_dtype`.

    """

    res_dtype = dpnp.result_type(*arrays)
    default_dtype = dpnp.default_float_type(sycl_queue=sycl_queue)

    if dtype is not None:
        if dpnp.can_cast(res_dtype, dtype, casting=casting):
            res_dtype = dtype
        else:
            raise TypeError(
                f"Cannot cast from dtype({res_dtype}) to dtype({dtype}) with casting rule {casting}"
            )

    op_dtype = (
        res_dtype if dpnp.issubdtype(res_dtype, dpnp.inexact) else default_dtype
    )

    return op_dtype, res_dtype


def _shape_error(a, b, core_dim, err_msg):
    if err_msg == 0:
        raise ValueError(
            "Input arrays have a mismatch in their core dimensions. "
            "The core dimensions should follow this signature: (n?,k),(k,m?)->(n?,m?) "
            f"(size {a} is different from {b})"
        )
    elif err_msg == 1:
        raise ValueError(
            f"Output array has a mismatch in its core dimension {core_dim}. "
            "The core dimensions should follow this signature: (n?,k),(k,m?)->(n?,m?) "
            f"(size {a} is different from {b})"
        )
    elif err_msg == 2:
        raise ValueError(
            "Input arrays could not be broadcast together with remapped shapes, "
            f"{a} is different from {b}."
        )
    elif err_msg == 3:
        raise ValueError(
            "Output array could not be broadcast to input arrays with remapped shapes, "
            f"{a} is different from {b}."
        )


def _standardize_strides(strides, inherently_2D, shape, ndim):
    """
    Standardizing the strides.

    When shape of an array along any particular dimension is 1, the stride
    along that dimension is undefined. This functions standardize the strides
    in the following way:
    For N-D arrays that are inherently 2D (all dimesnsion are one except for two of them),
    we use zero as the stride for dimensions equal one.
    For other N-D arrays, the non-zero value of strides is calculated and used.

    """

    if inherently_2D:
        stndrd_strides = tuple(
            str_i if sh_i > 1 else 0 for sh_i, str_i in zip(shape, strides)
        )
    else:
        stndrd_strides = [
            numpy.prod(shape[i + 1 :]) if strides[i] == 0 else strides[i]
            for i in range(ndim - 1)
        ]
        # last dimension
        stndrd_strides.append(
            1 if strides[ndim - 1] == 0 else strides[ndim - 1]
        )
        stndrd_strides = tuple(stndrd_strides)

    return stndrd_strides


def _validate_axes(x1, x2, axes):
    """Check axes is valid for matmul function."""

    def _validate_internal(axes, i, ndim):
        if ndim == 1:
            iter = 1
            if isinstance(axes, int):
                axes = (axes,)
            elif not isinstance(axes, tuple):
                raise TypeError(
                    f"Axes item {i}: {type(axes)} object cannot be interpreted as an integer."
                )

            if len(axes) != 1:
                raise ValueError(
                    f"Axes item {i} should be a tuple with a single element, or an integer."
                )
        else:
            iter = 2
            if not isinstance(axes, tuple):
                raise TypeError(f"Axes item {i} should be a tuple.")
            if len(axes) != 2:
                raise ValueError(
                    f"Axes item {i} should be a tuple with 2 elements."
                )

        for j in range(iter):
            if not isinstance(axes[j], int):
                raise TypeError(
                    f"Axes item {i}: {type(axes[j])} object cannot be interpreted as an integer."
                )
        return axes

    if not isinstance(axes, list):
        raise TypeError("Axes should be a list.")
    else:
        if len(axes) != 3:
            raise ValueError(
                "Axes should be a list of three tuples for inputs and output."
            )

    axes[0] = _validate_internal(axes[0], 0, x1.ndim)
    axes[1] = _validate_internal(axes[1], 1, x2.ndim)

    if x1.ndim == 1 and x2.ndim == 1:
        if axes[2] != ():
            raise TypeError("Axes item 2 should be an empty tuple.")
    elif x1.ndim == 1 or x2.ndim == 1:
        axes[2] = _validate_internal(axes[2], 2, 1)
    else:
        axes[2] = _validate_internal(axes[2], 2, 2)

    return axes


def dpnp_cross(a, b, cp, exec_q):
    """Return the cross product of two (arrays of) vectors."""

    # create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]

    host_events = []
    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            cp_usm = dpnp.get_usm_ndarray(cp)
            ht_ev1, dev_ev1 = tei._multiply(
                dpnp.get_usm_ndarray(a0),
                dpnp.get_usm_ndarray(b1),
                cp_usm,
                exec_q,
            )
            host_events.append(ht_ev1)
            tmp = dpt.empty_like(cp_usm)
            ht_ev2, dev_ev2 = tei._multiply(
                dpnp.get_usm_ndarray(a1), dpnp.get_usm_ndarray(b0), tmp, exec_q
            )
            host_events.append(ht_ev2)
            ht_ev3, _ = tei._subtract_inplace(
                cp_usm, tmp, exec_q, [dev_ev1, dev_ev2]
            )
            host_events.append(ht_ev3)
        else:
            assert b.shape[-1] == 3
            # cp0 = a1 * b2 - 0  (a2 = 0)
            # cp1 = 0 - a0 * b2  (a2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            cp1_usm = dpnp.get_usm_ndarray(cp1)
            cp2_usm = dpnp.get_usm_ndarray(cp2)
            a1_usm = dpnp.get_usm_ndarray(a1)
            b2_usm = dpnp.get_usm_ndarray(b2)
            ht_ev1, _ = tei._multiply(
                a1_usm, b2_usm, dpnp.get_usm_ndarray(cp0), exec_q
            )
            host_events.append(ht_ev1)
            ht_ev2, dev_ev2 = tei._multiply(
                dpnp.get_usm_ndarray(a0), b2_usm, cp1_usm, exec_q
            )
            host_events.append(ht_ev2)
            ht_ev3, _ = tei._negative(cp1_usm, cp1_usm, exec_q, [dev_ev2])
            host_events.append(ht_ev3)
            ht_ev4, dev_ev4 = tei._multiply(
                dpnp.get_usm_ndarray(a0),
                dpnp.get_usm_ndarray(b1),
                cp2_usm,
                exec_q,
            )
            host_events.append(ht_ev4)
            tmp = dpt.empty_like(cp2_usm)
            ht_ev5, dev_ev5 = tei._multiply(
                a1_usm, dpnp.get_usm_ndarray(b0), tmp, exec_q
            )
            host_events.append(ht_ev5)
            ht_ev6, _ = tei._subtract_inplace(
                cp2_usm, tmp, exec_q, [dev_ev4, dev_ev5]
            )
            host_events.append(ht_ev6)
    else:
        assert a.shape[-1] == 3
        if b.shape[-1] == 3:
            # cp0 = a1 * b2 - a2 * b1
            # cp1 = a2 * b0 - a0 * b2
            # cp2 = a0 * b1 - a1 * b0
            cp0_usm = dpnp.get_usm_ndarray(cp0)
            cp1_usm = dpnp.get_usm_ndarray(cp1)
            cp2_usm = dpnp.get_usm_ndarray(cp2)
            a0_usm = dpnp.get_usm_ndarray(a0)
            a1_usm = dpnp.get_usm_ndarray(a1)
            a2_usm = dpnp.get_usm_ndarray(a2)
            b0_usm = dpnp.get_usm_ndarray(b0)
            b1_usm = dpnp.get_usm_ndarray(b1)
            b2_usm = dpnp.get_usm_ndarray(b2)
            ht_ev1, dev_ev1 = tei._multiply(a1_usm, b2_usm, cp0_usm, exec_q)
            host_events.append(ht_ev1)
            tmp = dpt.empty_like(cp0_usm)
            ht_ev2, dev_ev2 = tei._multiply(a2_usm, b1_usm, tmp, exec_q)
            host_events.append(ht_ev2)
            ht_ev3, dev_ev3 = tei._subtract_inplace(
                cp0_usm, tmp, exec_q, [dev_ev1, dev_ev2]
            )
            host_events.append(ht_ev3)
            ht_ev4, dev_ev4 = tei._multiply(a2_usm, b0_usm, cp1_usm, exec_q)
            host_events.append(ht_ev4)
            ht_ev5, dev_ev5 = tei._multiply(
                a0_usm, b2_usm, tmp, exec_q, [dev_ev3]
            )
            host_events.append(ht_ev5)
            ht_ev6, dev_ev6 = tei._subtract_inplace(
                cp1_usm, tmp, exec_q, [dev_ev4, dev_ev5]
            )
            host_events.append(ht_ev6)
            ht_ev7, dev_ev7 = tei._multiply(a0_usm, b1_usm, cp2_usm, exec_q)
            host_events.append(ht_ev7)
            ht_ev8, dev_ev8 = tei._multiply(
                a1_usm, b0_usm, tmp, exec_q, [dev_ev6]
            )
            host_events.append(ht_ev8)
            ht_ev9, _ = tei._subtract_inplace(
                cp2_usm, tmp, exec_q, [dev_ev7, dev_ev8]
            )
            host_events.append(ht_ev9)
        else:
            assert b.shape[-1] == 2
            # cp0 = 0 - a2 * b1  (b2 = 0)
            # cp1 = a2 * b0 - 0  (b2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            cp0_usm = dpnp.get_usm_ndarray(cp0)
            cp2_usm = dpnp.get_usm_ndarray(cp2)
            a2_usm = dpnp.get_usm_ndarray(a2)
            b1_usm = dpnp.get_usm_ndarray(b1)
            ht_ev1, dev_ev1 = tei._multiply(a2_usm, b1_usm, cp0_usm, exec_q)
            host_events.append(ht_ev1)
            ht_ev2, _ = tei._negative(cp0_usm, cp0_usm, exec_q, [dev_ev1])
            host_events.append(ht_ev2)
            ht_ev3, _ = tei._multiply(
                a2_usm,
                dpnp.get_usm_ndarray(b0),
                dpnp.get_usm_ndarray(cp1),
                exec_q,
            )
            host_events.append(ht_ev3)
            ht_ev4, dev_ev4 = tei._multiply(
                dpnp.get_usm_ndarray(a0), b1_usm, cp2_usm, exec_q
            )
            host_events.append(ht_ev4)
            tmp = dpt.empty_like(cp2_usm)
            ht_ev5, dev_ev5 = tei._multiply(
                dpnp.get_usm_ndarray(a1), dpnp.get_usm_ndarray(b0), tmp, exec_q
            )
            host_events.append(ht_ev5)
            ht_ev6, _ = tei._subtract_inplace(
                cp2_usm, tmp, exec_q, [dev_ev4, dev_ev5]
            )
            host_events.append(ht_ev6)

    dpctl.SyclEvent.wait_for(host_events)
    return cp


def dpnp_kron(a, b, a_ndim, b_ndim):
    """Returns the kronecker product of two arrays."""

    a_shape = a.shape
    b_shape = b.shape
    if not a.flags.contiguous:
        a = dpnp.reshape(a, a_shape)
    if not b.flags.contiguous:
        b = dpnp.reshape(b, b_shape)

    # Equalise the shapes by prepending smaller one with 1s
    a_shape = (1,) * max(0, b_ndim - a_ndim) + a_shape
    b_shape = (1,) * max(0, a_ndim - b_ndim) + b_shape

    # Insert empty dimensions
    a_arr = dpnp.expand_dims(a, axis=tuple(range(b_ndim - a_ndim)))
    b_arr = dpnp.expand_dims(b, axis=tuple(range(a_ndim - b_ndim)))

    # Compute the product
    ndim = max(b_ndim, a_ndim)
    a_arr = dpnp.expand_dims(a_arr, axis=tuple(range(1, 2 * ndim, 2)))
    b_arr = dpnp.expand_dims(b_arr, axis=tuple(range(0, 2 * ndim, 2)))
    result = dpnp.multiply(a_arr, b_arr)

    # Reshape back
    return result.reshape(tuple(numpy.multiply(a_shape, b_shape)))


def dpnp_dot(a, b, /, out=None, *, conjugate=False):
    """
    Return the dot product of two arrays.

    The routine that is used to perform the main calculation
    depends on input arrays data type: 1) For integer and boolean data types,
    `dpctl.tensor.vecdot` form the Data Parallel Control library is used,
    2) For real-valued floating point data types, `dot` routines from
    BLAS library of OneMKL are used, and 3) For complex data types,
    `dotu` or `dotc` routines from BLAS library of OneMKL are used.
    If `conjugate` is ``False``, `dotu` is used. Otherwise, `dotc` is used,
    for which the first array is conjugated before calculating the dot product.

    """

    if a.size != b.size:
        raise ValueError(
            "Input arrays have a mismatch in their size. "
            f"(size {a.size} is different from {b.size})"
        )

    res_usm_type, exec_q = get_usm_allocations([a, b])

    # Determine the appropriate data types
    # casting is irrelevant here since dtype is `None`
    dot_dtype, res_dtype = _op_res_dtype(
        a, b, dtype=None, casting="no", sycl_queue=exec_q
    )

    result = _create_result_array(
        a, b, out, (), dot_dtype, res_usm_type, exec_q
    )
    # input arrays should have the proper data type
    dep_events_list = []
    host_tasks_list = []
    if dpnp.issubdtype(res_dtype, dpnp.inexact):
        # copying is needed if dtypes of input arrays are different
        a = _copy_array(a, dep_events_list, host_tasks_list, dtype=dot_dtype)
        b = _copy_array(b, dep_events_list, host_tasks_list, dtype=dot_dtype)
        if dpnp.issubdtype(res_dtype, dpnp.complexfloating):
            if conjugate:
                dot_func = "_dotc"
            else:
                dot_func = "_dotu"
            ht_ev, _ = getattr(bi, dot_func)(
                exec_q,
                dpnp.get_usm_ndarray(a),
                dpnp.get_usm_ndarray(b),
                dpnp.get_usm_ndarray(result),
                dep_events_list,
            )
        else:
            ht_ev, _ = bi._dot(
                exec_q,
                dpnp.get_usm_ndarray(a),
                dpnp.get_usm_ndarray(b),
                dpnp.get_usm_ndarray(result),
                dep_events_list,
            )
        host_tasks_list.append(ht_ev)
        dpctl.SyclEvent.wait_for(host_tasks_list)
    else:
        dpt_a = dpnp.get_usm_ndarray(a)
        dpt_b = dpnp.get_usm_ndarray(b)
        result = dpnp_array._create_from_usm_ndarray(dpt.vecdot(dpt_a, dpt_b))

    if dot_dtype != res_dtype:
        result = result.astype(res_dtype, copy=False)

    # numpy.dot does not allow casting even if it is safe
    return dpnp.get_result_array(result, out, casting="no")


def dpnp_matmul(
    x1,
    x2,
    /,
    out=None,
    *,
    casting="same_kind",
    order="K",
    dtype=None,
    axes=None,
):
    """
    Return the matrix product of two arrays.

    The main calculation is done by calling an extension function
    for BLAS library of OneMKL. Depending on dimension of `x1` and `x2` arrays,
    it will be either ``gemm`` (for one- and two-dimentional arrays) or
    ``gemm_batch``(for others).

    """

    x1_ndim = x1.ndim
    x2_ndim = x2.ndim

    if x1_ndim == 0:
        raise ValueError(
            "input array 0 does not have enough dimensions (has 0, but requires at least 1)"
        )
    if x2_ndim == 0:
        raise ValueError(
            "input array 1 does not have enough dimensions (has 0, but requires at least 1)"
        )

    res_usm_type, exec_q = get_usm_allocations([x1, x2])

    if axes is not None:
        axes = _validate_axes(x1, x2, axes)

        axes_x1, axes_x2, axes_res = axes
        axes_x1 = normalize_axis_tuple(axes_x1, x1.ndim, "axis")
        axes_x2 = normalize_axis_tuple(axes_x2, x2.ndim, "axis")
        # Move the axes that are going to be used in matrix product,
        # to the end of "x1" and "x2"
        x1 = dpnp.moveaxis(x1, axes_x1, (-2, -1)) if x1.ndim != 1 else x1
        x2 = dpnp.moveaxis(x2, axes_x2, (-2, -1)) if x2.ndim != 1 else x2
        out_orig = out
        if out is not None:
            dpnp.check_supported_arrays_type(out)
            # out that is passed to the backend should have the correct shape
            if len(axes_res) == 2:
                out = dpnp.moveaxis(out, axes_res, (-2, -1))
            elif len(axes_res) == 1:
                out = dpnp.moveaxis(out, axes_res, (-1,))

    appended_axes = []
    if x1_ndim == 1:
        x1 = x1[dpnp.newaxis, :]
        x1_ndim = x1.ndim
        appended_axes.append(-2)

    if x2_ndim == 1:
        x2 = x2[:, dpnp.newaxis]
        x2_ndim = x2.ndim
        appended_axes.append(-1)

    x1_shape = x1.shape
    x2_shape = x2.shape
    if x1_shape[-1] != x2_shape[-2]:
        _shape_error(x1_shape[-1], x2_shape[-2], None, 0)

    if out is not None:
        out_shape = out.shape
        if not appended_axes:
            if out_shape[-2] != x1_shape[-2]:
                _shape_error(out_shape[-2], x1_shape[-2], 0, 1)
            if out_shape[-1] != x2_shape[-1]:
                _shape_error(out_shape[-1], x2_shape[-1], 1, 1)
        elif len(appended_axes) == 1:
            if appended_axes[0] == -1:
                if out_shape[-1] != x1_shape[-2]:
                    _shape_error(out_shape[-1], x1_shape[-2], 0, 1)
            elif appended_axes[0] == -2:
                if out_shape[-1] != x2_shape[-1]:
                    _shape_error(out_shape[-1], x2_shape[-1], 0, 1)

    # Determine the appropriate data types
    gemm_dtype, res_dtype = _op_res_dtype(
        x1, x2, dtype=dtype, casting=casting, sycl_queue=exec_q
    )

    x1_is_2D = x1_ndim == 2 or numpy.prod(x1_shape[:-2]) == 1  # inherently 2D
    x2_is_2D = x2_ndim == 2 or numpy.prod(x2_shape[:-2]) == 1

    # find the result shape
    if x1_is_2D and x2_is_2D:
        x1 = x1.reshape(x1.shape[-2], x1.shape[-1])
        x2 = x2.reshape(x2.shape[-2], x2.shape[-1])
        res_shape = (x1.shape[-2], x2.shape[-1])
    else:
        # makes the dimension of input the same by adding new axis
        if x1_ndim != x2_ndim:
            diff = abs(x1_ndim - x2_ndim)
            if x1_ndim < x2_ndim:
                x1 = x1.reshape((1,) * diff + x1.shape)
                x1_ndim = x1.ndim
                x1_shape = x1.shape
            else:
                x2 = x2.reshape((1,) * diff + x2.shape)
                x2_ndim = x2.ndim
                x2_shape = x2.shape

        # examining the option to align inputs
        # when their shapes differ but they are 1-D in some dimensions.
        tmp_shape = list(x1_shape[:-2])
        for i in range(x1_ndim - 2):
            if x1_shape[i] != x2_shape[i]:
                if x1_shape[i] == 1:
                    tmp_shape[i] = x2_shape[i]
                    # If the `x1` array is inherently 2D, there's no need to
                    # duplicate the data for the 1-D dimension;
                    # GEMM handles it automatically.
                    if not x1_is_2D:
                        x1 = dpnp.repeat(x1, x2_shape[i], axis=i)
                elif x2_shape[i] == 1:
                    tmp_shape[i] = x1_shape[i]
                    if not x2_is_2D:
                        x2 = dpnp.repeat(x2, x1_shape[i], axis=i)
                else:
                    _shape_error(x1_shape[:-2], x2_shape[:-2], None, 2)

        x1_shape = x1.shape
        x2_shape = x2.shape
        if out is not None:
            for i in range(x1_ndim - 2):
                if tmp_shape[i] != out_shape[i]:
                    if not appended_axes:
                        _shape_error(tuple(tmp_shape), out_shape[:-2], None, 3)
                    elif len(appended_axes) == 1:
                        _shape_error(tuple(tmp_shape), out_shape[:-1], None, 3)

        res_shape = tuple(tmp_shape) + (x1_shape[-2], x2_shape[-1])

    # handling a special case to provide a similar result to NumPy
    if out is not None and x1.shape == (1, 0) and x2.shape == (0, 1):
        res_shape = (0,)
        appended_axes = []

    result = _create_result_array(
        x1, x2, out, res_shape, gemm_dtype, res_usm_type, exec_q
    )

    # calculate result
    if result.size == 0:
        pass
    elif x1.size == 0 or x2.size == 0:
        result.fill(0)
    else:
        # input arrays should have the proper data type
        # and be C_CONTIGUOUS or F_CONTIGUOUS
        dep_events_list = []
        host_tasks_list = []
        contig_copy = not (x1.flags.c_contiguous or x1.flags.f_contiguous)
        x1 = _copy_array(
            x1,
            dep_events_list,
            host_tasks_list,
            contig_copy=contig_copy,
            dtype=gemm_dtype,
        )
        contig_copy = not (x2.flags.c_contiguous or x2.flags.f_contiguous)
        x2 = _copy_array(
            x2,
            dep_events_list,
            host_tasks_list,
            contig_copy=contig_copy,
            dtype=gemm_dtype,
        )

        # TODO: investigate usage of gemv (gemv_batch) function
        # from BLAS when one of the inputs is a vector to
        # gain performance.
        # TODO: investigate usage of syrk function from BLAS in
        # case of a.T @ a and a @ a.T to gain performance.
        if x1_is_2D and x2_is_2D:
            ht_blas_ev, _ = bi._gemm(
                exec_q,
                dpnp.get_usm_ndarray(x1),
                dpnp.get_usm_ndarray(x2),
                dpnp.get_usm_ndarray(result),
                dep_events_list,
            )
        else:
            (
                ht_blas_ev,
                ht_copy_ev,
                result,
            ) = _gemm_batch_matmul(
                exec_q,
                x1,
                x2,
                result,
                x1_is_2D,
                x2_is_2D,
                dep_events_list,
            )
            host_tasks_list += ht_copy_ev

        host_tasks_list.append(ht_blas_ev)
        dpctl.SyclEvent.wait_for(host_tasks_list)

    if appended_axes:
        result = dpnp.squeeze(result, tuple(appended_axes))
        if len(appended_axes) == 2 and out is not None:
            result = dpnp.tile(result, out.shape)

    if x1_is_2D and x2_is_2D:
        # add new axes only if one of the input arrays
        # was inehrently 2D
        new_size = max(x1_ndim, x2_ndim)
        for _ in range(new_size - 2):
            result = result[dpnp.newaxis, :]

    if gemm_dtype != res_dtype:
        result = dpnp.astype(result, res_dtype, copy=False)

    if out is None:
        if axes is not None:
            # Move the data to the appropriate axes of the result array
            if len(axes_res) == 2:
                result = dpnp.moveaxis(result, (-2, -1), axes_res)
            elif len(axes_res) == 1:
                result = dpnp.moveaxis(result, (-1,), axes_res)
            return result
        # If `order` was not passed as default
        # we need to update it to match the passed `order`.
        elif order not in ["k", "K"]:
            return dpnp.array(result, copy=False, order=order)
        else:
            return result
    else:
        result = dpnp.get_result_array(result, out, casting=casting)
        if axes is not None and out is result:
            # out and out_orig contain the same data but they have different shape
            return out_orig
        return result
