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
import dpctl.tensor._tensor_impl as ti
import dpctl.utils as dpu
import numpy
from dpctl.tensor._numpy_helper import normalize_axis_tuple
from dpctl.utils import ExecutionPlacementError

import dpnp
import dpnp.backend.extensions.blas._blas_impl as bi
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations

__all__ = ["dpnp_cross", "dpnp_dot", "dpnp_kron", "dpnp_matmul"]


def _compute_res_dtype(*arrays, sycl_queue, dtype=None, casting="no"):
    """
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
    casting : {"no", "equiv", "safe", "same_kind", "unsafe"}, optional
        Controls what kind of data casting may occur.
    sycl_queue : {SyclQueue}
        A SYCL queue to use for determining default floating point datat type.

    Returns
    -------
    compute_dtype, res_dtype :
        `compute_dtype` is the data type used in performing math function calculations.
        The input arrays of the math function are cast to `compute_dtype` and then
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

    compute_dtype = (
        res_dtype if dpnp.issubdtype(res_dtype, dpnp.inexact) else default_dtype
    )

    return compute_dtype, res_dtype


def _copy_array(x, copy_flag=False, dtype=None, order="C"):
    """
    Creating a copy of input array if needed.

    If `copy_flag` is ``True``, a C-contiguous copy of input array is returned.
    In this case, the copy array has the input array data type unless `dtype` is
    determined.
    If `copy_flag` is ``False`` and input array data type is different than `dtype`,
    a C-contiguous copy of input array with specified `dtype` is returned.

    """

    if copy_flag:
        copy = copy_flag
    else:
        copy = x.dtype != dtype if dtype is not None else False

    if copy:
        x_copy = dpnp.empty_like(x, dtype=dtype, order=order)

        exec_q = x_copy.sycl_queue
        _manager = dpu.SequentialOrderManager[exec_q]

        ht_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=dpnp.get_usm_ndarray(x),
            dst=x_copy.get_array(),
            sycl_queue=exec_q,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        return x_copy
    return x


def _create_result_array(
    x1, x2, out, shape, dtype, usm_type, sycl_queue, order="C"
):
    """
    Create the result array.

    If `out` is not ``None`` and its shape and dtype match the desired `shape`
    and `dtype`, and its 2-D base is contiguous and it does not have any memory
    overlap with `x1` and `x2`, `out` itself is returned.
    If these conditions are not satisfied, an empty array is returned with the
    specified `shape`, `dtype, `usm_type`, and `sycl_queue`.

    """

    if out is not None:
        x1_usm = dpnp.get_usm_ndarray(x1)
        x2_usm = dpnp.get_usm_ndarray(x2)
        out_usm = dpnp.get_usm_ndarray(out)
        contig_flag, _, _ = _define_contig_flag(out)

        if (
            out.dtype == dtype
            and out.shape == shape
            and contig_flag
            and not ti._array_overlap(x1_usm, out_usm)
            and not ti._array_overlap(x2_usm, out_usm)
        ):
            return out

    return dpnp.empty(
        shape,
        dtype=dtype,
        order=order,
        usm_type=usm_type,
        sycl_queue=sycl_queue,
    )


def _define_contig_flag(x):
    """
    Determines if the data in last two dimensions of array `x` are
    c_contiguous or f_contiguous. For 2D arrays, it is the same as using
    x.flags.c_contiguous or x.flags.f_contiguous.
    """

    flag = False
    x_strides = x.strides
    x_shape = x.shape
    if x.ndim < 2:
        return True, True, True

    x_strides = _standardize_strides_to_nonzero(x_strides, x_shape)
    x_is_c_contiguous = x_strides[-1] == 1 and x_strides[-2] == x_shape[-1]
    x_is_f_contiguous = x_strides[-2] == 1 and x_strides[-1] == x_shape[-2]
    if x_is_c_contiguous or x_is_f_contiguous:
        flag = True
    return flag, x_is_c_contiguous, x_is_f_contiguous


def _define_dim_flags(x, pos):
    """
    Define useful flags for the main calculation in dpnp_matmul.
    x_is_1D: `x` is 1D array or inherently 1D (all dimensions are equal to one
    except for one of them), for instance, if x.shape = (1, 1, 1, 2),
    then x_is_1D = True
    x_is_2D: `x` is 2D array or inherently 2D (all dimensions are equal to one
    except for the last two of them), for instance, if x.shape = (1, 1, 3, 2),
    then x_is_2D = True
    x_base_is_1D: `x` is 1D considering only its last two dimensions, for instance,
    if x.shape = (3, 4, 1, 2), then x_base_is_1D = True
    """

    index = -1 if pos == 0 else -2
    x_shape = x.shape
    x_ndim = x.ndim
    x_is_1D = x_ndim == 1
    if numpy.prod(x_shape) != 0:
        # the 2nd condition is only expected to be checked for
        # ND-arrays (N>1) since index is in [-2, -1]
        x_is_1D = x_is_1D or numpy.prod(x_shape) == x_shape[index]

    x_is_2D = False
    if not x_is_1D:
        x_is_2D = x_ndim == 2 or numpy.prod(x_shape[:-2]) == 1

    x_base_is_1D = x_is_1D
    if not x_is_1D:
        x_base_is_1D = x_shape[-1] == 1 or x_shape[-2] == 1

    return x_is_2D, x_is_1D, x_base_is_1D


def _get_result_shape(x1, x2, out, np_flag):
    """
    Three task are completed in this function:
        - Get the shape of the result array.
        - Validate the shape of output array, if provided.
        - Align the input arrays if they could be broadcast together.
    """
    x1_ndim = x1.ndim
    x2_ndim = x2.ndim

    x1_shape = x1.shape
    x2_shape = x2.shape

    if x1_ndim == 0:
        raise ValueError(
            "Input array 0 does not have enough dimensions (has 0, but requires at least 1)"
        )
    if x2_ndim == 0:
        raise ValueError(
            "Input array 1 does not have enough dimensions (has 0, but requires at least 1)"
        )

    if x1_ndim == 1 and x2_ndim == 1:
        if x1_shape[-1] != x2_shape[-1]:
            _shape_error(x1_shape[-1], x2_shape[-1], None, err_msg=0)
        result_shape = ()
    elif x1_ndim == 1:
        if x1_shape[-1] != x2_shape[-2]:
            _shape_error(x1_shape[-1], x2_shape[-2], None, err_msg=0)
        result_shape = x2_shape[:-2] + (x2_shape[-1],)
    elif x2_ndim == 1:
        if x1_shape[-1] != x2_shape[-1]:
            _shape_error(x1_shape[-1], x2_shape[-1], None, err_msg=0)
        result_shape = x1_shape[:-1]
    else:  # at least 2D
        x1_is_2D, x1_is_1D, _ = _define_dim_flags(x1, pos=0)
        x2_is_2D, x2_is_1D, _ = _define_dim_flags(x2, pos=1)

        if x1_shape[-1] != x2_shape[-2]:
            _shape_error(x1_shape[-1], x2_shape[-2], None, err_msg=0)

        if x1_ndim == 2 and x2_ndim == 2:
            result_shape = (x1_shape[-2], x2_shape[-1])
        else:
            if x1_ndim != x2_ndim:
                diff = abs(x1_ndim - x2_ndim)
                if x1_ndim < x2_ndim:
                    x1 = dpnp.reshape(x1, ((1,) * diff + x1.shape))
                    x1_shape = x1.shape
                else:
                    x2 = dpnp.reshape(x2, ((1,) * diff + x2.shape))
                    x2_shape = x2.shape

            # examining the option to align inputs when their
            # shapes differ but the shape of one of them is 1
            # in that dimension (similar to braodcasting concept)
            tmp_shape = list(x1_shape[:-2])
            for i in range(len(tmp_shape)):
                if x1_shape[i] != x2_shape[i]:
                    if x1_shape[i] == 1:
                        tmp_shape[i] = x2_shape[i]
                        # If array `x1` is inherently 1D or 2D, there's
                        # no need to duplicate the data for the dimension
                        # with shape equal to one; gemv_batch or gemm_batch
                        # can handle it by using zero as the stride between
                        # different `x1` matrices
                        if not (x1_is_2D or x1_is_1D):
                            x1 = dpnp.repeat(x1, x2_shape[i], axis=i)
                    elif x2_shape[i] == 1:
                        if not (x2_is_2D or x2_is_1D):
                            x2 = dpnp.repeat(x2, x1_shape[i], axis=i)
                    else:
                        _shape_error(
                            x1_shape[:-2], x2_shape[:-2], None, err_msg=2
                        )

            result_shape = tuple(tmp_shape) + (x1.shape[-2], x2.shape[-1])

    if out is not None:
        out_shape = out.shape
        if out_shape != result_shape and not np_flag:
            len_out = len(out_shape)
            len_res = len(result_shape)
            if len_out != len_res:
                _shape_error(len_out, len_res, None, err_msg=4)
            for i in range(len_out):
                if out_shape[i] != result_shape[i]:
                    if i == len_out - 1:
                        _shape_error(
                            out_shape[i], result_shape[i], 1, err_msg=1
                        )
                    elif i == len_out - 2:
                        _shape_error(
                            out_shape[i], result_shape[i], 0, err_msg=1
                        )
                    else:
                        _shape_error(
                            out_shape[:-2], result_shape[:-2], None, err_msg=3
                        )

    return x1, x2, result_shape


def _gemm_batch_matmul(exec_q, x1, x2, res):
    # arrays here are already at least 3D, make them 3D
    x1_shape = x1.shape
    x2_shape = x2.shape
    x1 = dpnp.reshape(x1, (-1, x1_shape[-2], x1_shape[-1]))
    x2 = dpnp.reshape(x2, (-1, x2_shape[-2], x2_shape[-1]))
    orig_shape = res.shape
    res = dpnp.reshape(res, (-1, orig_shape[-2], orig_shape[-1]))
    res_shape = res.shape

    # gemm_batch does not handle negative strides, make a copy if needed
    x1 = _copy_array(x1, copy_flag=x1.strides[0] < 0)
    x2 = _copy_array(x2, copy_flag=x2.strides[0] < 0)
    res = _copy_array(res, copy_flag=res.strides[0] < 0)

    _manager = dpu.SequentialOrderManager[exec_q]

    # onemkl::blas::gemm_bacth throws an exception (Provided range is out
    # of integer limits) if the batch_size is too large, so we need to
    # split the batch into smaller chunks, the size depnends on device
    chunk = 4096 * 4096 - 2
    batch_size = res_shape[0]
    for i in range(0, batch_size, chunk):
        if x1_shape[0] == 1:
            # x1 is repeatedly multiplied with each matrix in x2
            x1_usm = dpnp.get_usm_ndarray(x1)
            x2_usm = dpnp.get_usm_ndarray(x2[i : i + chunk, ...])
        elif x2_shape[0] == 1:
            x1_usm = dpnp.get_usm_ndarray(x1[i : i + chunk, ...])
            x2_usm = dpnp.get_usm_ndarray(x2)
        else:
            x1_usm = dpnp.get_usm_ndarray(x1[i : i + chunk, ...])
            x2_usm = dpnp.get_usm_ndarray(x2[i : i + chunk, ...])
        res_usm = dpnp.get_usm_ndarray(res[i : i + chunk, ...])

        ht_ev, blas_ev, row_major = bi._gemm_batch(
            exec_q,
            x1_usm,
            x2_usm,
            res_usm,
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht_ev, blas_ev)

    _, res_is_c_contig, res_is_f_contig = _define_contig_flag(res)
    if row_major:
        if res_is_f_contig:
            # Considering the multiplication for one of the batches,
            # we have result[0, 1] = a[0, :]*b[1, :]. In row_major mode,
            # it is assumed result array is c-contiguous, i.e. the value of
            # result[0, 1] is has the second place memory.
            # however, the result array is batches of 2D f-contiguous array,
            # i.e. the second place of memory points out to res[1, 0].
            # So, we need to read data of each 2D array in the batch in
            # "F" order and write it in "C" order
            res = (
                res.ravel(order="F")
                .reshape(res_shape[1], res_shape[2], batch_size)
                .transpose(2, 0, 1)
            )
    else:
        if res_is_c_contig:
            # read data of each 2D array in the batch in "C" order and
            # write it in "F" order
            res = (
                res.ravel(order="C")
                .reshape(batch_size, res_shape[2], res_shape[1])
                .transpose(0, 2, 1)
            )

    if res_shape != orig_shape:
        res = res.reshape(orig_shape)

    return res


def _gemm_matmul(exec_q, x1, x2, res):
    _manager = dpu.SequentialOrderManager[exec_q]

    ht_ev, gemm_ev, row_major = bi._gemm(
        exec_q,
        dpnp.get_usm_ndarray(x1),
        dpnp.get_usm_ndarray(x2),
        dpnp.get_usm_ndarray(res),
        depends=_manager.submitted_events,
    )
    _manager.add_event_pair(ht_ev, gemm_ev)

    if row_major:
        if res.flags.f_contiguous:
            # read data in "F" order and write it in "C" order
            res = dpnp.ravel(res, order="F").reshape(res.shape, order="C")
    else:
        if res.flags.c_contiguous:
            # read data in "C" order and write it in "F" order
            res = dpnp.ravel(res, order="C").reshape(res.shape, order="F")

    return res


def _shape_error(a, b, core_dim, err_msg):
    if err_msg == 0:
        raise ValueError(
            "Input arrays have a mismatch in their core dimensions. "
            "The core dimensions should follow this signature: "
            f"(n?,k),(k,m?)->(n?,m?) (size {a} is different from {b})"
        )
    elif err_msg == 1:
        raise ValueError(
            f"Output array has a mismatch in its core dimension {core_dim}. "
            "The core dimensions should follow this signature: "
            f"(n?,k),(k,m?)->(n?,m?) (size {a} is different from {b})"
        )
    elif err_msg == 2:
        raise ValueError(
            "Leading element(s) of the input arrays' shape do not match and "
            "they could not be broadcast together, the leading element(s) of "
            f"the input array 0 is {a} and it is different from leading "
            f"element(s) of the input array 1 which is {b}."
        )
    elif err_msg == 3:
        raise ValueError(
            "The shape of the output array does not have similar leading "
            "elements to the shape of input arrays, the leading element(s) of "
            f"the output array is {a} and it is different from the leading "
            f"element(s) of the input arrays which is {b}."
        )
    elif err_msg == 4:
        raise ValueError(
            "Output array does not have enough dimensions "
            f"(has {a} while requires {b})"
        )


def _standardize_strides_to_nonzero(strides, shape):
    """
    Standardizing the strides.
    When shape of an array along any particular dimension is 1, the stride
    along that dimension is undefined. This function standardize the strides
    by calculating the non-zero value of the strides.
    """

    ndim = len(strides)
    if numpy.prod(strides) == 0:
        stndrd_strides = tuple(
            numpy.prod(shape[i + 1 :]) if strides[i] == 0 else strides[i]
            for i in range(ndim - 1)
        )
        last_stride = 1 if strides[ndim - 1] == 0 else strides[ndim - 1]
        stndrd_strides += (last_stride,)
    else:
        stndrd_strides = strides

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


def dpnp_cross(a, b, cp):
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

    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            cp = dpnp.multiply(a0, b1, out=cp)
            cp -= a1 * b0
        else:
            assert b.shape[-1] == 3
            # cp0 = a1 * b2 - 0  (a2 = 0)
            cp0 = dpnp.multiply(a1, b2, out=cp0)

            # cp1 = 0 - a0 * b2  (a2 = 0)
            cp1 = dpnp.multiply(a0, b2, out=cp1)
            cp1 = dpnp.negative(cp1, out=cp1)

            # cp2 = a0 * b1 - a1 * b0
            cp2 = dpnp.multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
    else:
        assert a.shape[-1] == 3
        if b.shape[-1] == 3:
            # cp0 = a1 * b2 - a2 * b1
            cp0 = dpnp.multiply(a1, b2, out=cp0)
            cp0 -= a2 * b1

            # cp1 = a2 * b0 - a0 * b2
            cp1 = dpnp.multiply(a2, b0, out=cp1)
            cp1 -= a0 * b2

            # cp2 = a0 * b1 - a1 * b0
            cp2 = dpnp.multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
        else:
            assert b.shape[-1] == 2
            # cp0 = 0 - a2 * b1  (b2 = 0)
            cp0 = dpnp.multiply(a2, b1, out=cp0)
            cp0 = dpnp.negative(cp0, out=cp0)

            # cp1 = a2 * b0 - 0  (b2 = 0)
            cp1 = dpnp.multiply(a2, b0, out=cp1)

            # cp2 = a0 * b1 - a1 * b0
            cp2 = dpnp.multiply(a0, b1, out=cp2)
            cp2 -= a1 * b0
    return cp


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
    if (
        out is not None
        and dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None
    ):
        raise ExecutionPlacementError(
            "Input and output allocation queues are not compatible"
        )

    # Determine the appropriate data types
    dot_dtype, res_dtype = _compute_res_dtype(a, b, sycl_queue=exec_q)

    result = _create_result_array(
        a, b, out, (), dot_dtype, res_usm_type, exec_q
    )

    # input arrays should have the proper data type
    if dpnp.issubdtype(res_dtype, dpnp.inexact):
        # copying is needed if dtypes of input arrays are different
        a = _copy_array(a, dtype=dot_dtype)
        b = _copy_array(b, dtype=dot_dtype)

        _manager = dpu.SequentialOrderManager[exec_q]

        if dpnp.issubdtype(res_dtype, dpnp.complexfloating):
            dot_func = "_dotc" if conjugate else "_dotu"
        else:
            dot_func = "_dot"

        ht_ev, dot_ev = getattr(bi, dot_func)(
            exec_q,
            dpnp.get_usm_ndarray(a),
            dpnp.get_usm_ndarray(b),
            dpnp.get_usm_ndarray(result),
            depends=_manager.submitted_events,
        )
        _manager.add_event_pair(ht_ev, dot_ev)
    else:
        # oneapi::mkl::blas::dot is slow for integer data type,
        # so using dpctl.tensor.vecdot instead
        dpt_a = dpnp.get_usm_ndarray(a)
        dpt_b = dpnp.get_usm_ndarray(b)
        result = dpnp_array._create_from_usm_ndarray(dpt.vecdot(dpt_a, dpt_b))

    if dot_dtype != res_dtype:
        result = result.astype(res_dtype, copy=False)

    # numpy.dot does not allow casting even if it is safe
    return dpnp.get_result_array(result, out, casting="no")


def dpnp_kron(a, b, a_ndim, b_ndim):
    """Returns the kronecker product of two arrays."""

    a_shape = a.shape
    b_shape = b.shape

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

    dpnp.check_supported_arrays_type(x1, x2)
    res_usm_type, exec_q = get_usm_allocations([x1, x2])
    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

    if order in ["a", "A"]:
        if x1.flags.f_contiguous and x2.flags.f_contiguous:
            order = "F"
        else:
            order = "C"

    x1_ndim = x1.ndim
    x2_ndim = x2.ndim
    if axes is not None:
        axes = _validate_axes(x1, x2, axes)

        axes_x1, axes_x2, axes_res = axes
        axes_x1 = normalize_axis_tuple(axes_x1, x1_ndim, "axis")
        axes_x2 = normalize_axis_tuple(axes_x2, x2_ndim, "axis")

        # Move the axes that are going to be used in matrix product,
        # to the end of "x1" and "x2"
        x1 = dpnp.moveaxis(x1, axes_x1, (-2, -1)) if x1_ndim != 1 else x1
        x2 = dpnp.moveaxis(x2, axes_x2, (-2, -1)) if x2_ndim != 1 else x2

        out_orig = out
        if out is not None:
            # out that is passed to the backend should have the correct shape
            if len(axes_res) == 2:
                out = dpnp.moveaxis(out, axes_res, (-2, -1))
            elif len(axes_res) == 1:
                out = dpnp.moveaxis(out, axes_res, (-1,))

    # With these conditions, the result is a 0D array. However,
    # NumPy allows out to have any shape and the result is expanded to it
    NumPy_special_behavior = (
        out is not None and x1_ndim == 1 and x2_ndim == 1 and out.shape != ()
    )

    x1, x2, result_shape = _get_result_shape(
        x1, x2, out, NumPy_special_behavior
    )

    # Determine the appropriate data types
    compute_dtype, res_dtype = _compute_res_dtype(
        x1, x2, dtype=dtype, casting=casting, sycl_queue=exec_q
    )

    call_flag = None
    transpose = False
    x1_shape = x1.shape
    x2_shape = x2.shape
    x1_is_2D, x1_is_1D, x1_base_is_1D = _define_dim_flags(x1, pos=0)
    x2_is_2D, x2_is_1D, x2_base_is_1D = _define_dim_flags(x2, pos=1)

    # TODO: investigate usage of syrk function from BLAS in
    # case of a.T @ a and a @ a.T to gain performance.
    if numpy.prod(result_shape) == 0:
        res_shape = result_shape
    elif x1_shape[-1] == 1:
        call_flag = "multiply"
    elif x1_is_1D and x2_is_1D:
        call_flag = "dot"
        # arrays are inehrently 1D, make them 1D
        x1 = dpnp.ravel(x1)
        x2 = dpnp.ravel(x2)
    elif x1_base_is_1D and x2_base_is_1D:
        # TODO: implement a batch version of dot to use it here
        call_flag = "gemm_batch"
        res_shape = result_shape
    elif x1_is_1D and x2_is_2D:
        transpose = True
        call_flag = "gemv"
        x1 = dpnp.reshape(x1, x1.size)
        x2 = dpnp.reshape(x2, x2_shape[-2:])
        res_shape = (x2_shape[-1],)
    elif x1_is_2D and x2_is_1D:
        call_flag = "gemv"
        x1 = dpnp.reshape(x1, x1_shape[-2:])
        x2 = dpnp.reshape(x2, x2.size)
        res_shape = (x1_shape[-2],)
    elif x1_is_2D and x2_is_2D:
        call_flag = "gemm"
        x1 = dpnp.reshape(x1, x1_shape[-2:])
        x2 = dpnp.reshape(x2, x2_shape[-2:])
        res_shape = (x1_shape[-2], x2_shape[-1])
    elif x1_base_is_1D:
        # TODO: implement gemv_batch to use it here with transpose
        call_flag = "gemm_batch"
        if x1_ndim == 1:
            x1 = dpnp.reshape(x1, (1, 1, x1.size))
            res_shape = result_shape[:-1] + (1, result_shape[-1])
        else:
            res_shape = result_shape
    elif x2_base_is_1D:
        # TODO: implement gemv_batch to use it here without transpose
        call_flag = "gemm_batch"
        if x2_ndim == 1:
            x2 = dpnp.reshape(x2, (1, x2.size, 1))
            res_shape = result_shape + (1,)
        else:
            res_shape = result_shape
    else:
        call_flag = "gemm_batch"
        res_shape = result_shape

    # dispatch to proper function call
    if call_flag == "multiply":
        result = dpnp.multiply(x1, x2)
        res_shape = result.shape
    elif call_flag == "dot":
        if out is not None and out.shape != ():
            result = dpnp_dot(x1, x2)
        else:
            result = dpnp_dot(x1, x2, out=out)
        res_shape = result.shape
    else:
        x1_contig_flag, _, x1_f = _define_contig_flag(x1)
        x2_contig_flag, _, x2_f = _define_contig_flag(x2)

        res_order = "F" if (x1_f and x2_f and call_flag == "gemm") else "C"

        result = _create_result_array(
            x1,
            x2,
            out,
            res_shape,
            compute_dtype,
            res_usm_type,
            exec_q,
            res_order,
        )

        # calculate result
        if result.size == 0:
            pass
        elif x1.size == 0 or x2.size == 0:
            result.fill(0)
        else:
            # input arrays should have the proper data type and
            # their base (last 2-dimensions) to be c-contiguous or f-contiguous
            x1 = _copy_array(
                x1,
                copy_flag=not x1_contig_flag,
                dtype=compute_dtype,
                order=res_order,
            )
            x2 = _copy_array(
                x2,
                copy_flag=not x2_contig_flag,
                dtype=compute_dtype,
                order=res_order,
            )

            if call_flag == "gemv":
                if transpose:
                    a_usm = dpnp.get_usm_ndarray(x2)
                    x_usm = dpnp.get_usm_ndarray(x1)
                else:
                    a_usm = dpnp.get_usm_ndarray(x1)
                    x_usm = dpnp.get_usm_ndarray(x2)

                _manager = dpu.SequentialOrderManager[exec_q]

                ht_ev, gemv_ev = bi._gemv(
                    exec_q,
                    a_usm,
                    x_usm,
                    dpnp.get_usm_ndarray(result),
                    transpose,
                    depends=_manager.submitted_events,
                )
                _manager.add_event_pair(ht_ev, gemv_ev)
            elif call_flag == "gemm":
                result = _gemm_matmul(
                    exec_q,
                    x1,
                    x2,
                    result,
                )
            else:  # call_flag == "gemm_batch"
                result = _gemm_batch_matmul(
                    exec_q,
                    x1,
                    x2,
                    result,
                )

    if NumPy_special_behavior:
        result = dpnp.tile(result, out.shape)
    elif res_shape != result_shape:
        result = dpnp.reshape(result, result_shape)

    if compute_dtype != res_dtype:
        result = dpnp.astype(result, res_dtype, copy=False)

    if out is None:
        if axes is not None:
            # Move the data to the appropriate axes of the result array
            if len(axes_res) == 2:
                result = dpnp.moveaxis(result, (-2, -1), axes_res)
            elif len(axes_res) == 1:
                result = dpnp.moveaxis(result, (-1,), axes_res)
            return dpnp.ascontiguousarray(result)

        # If `order` was not passed as default
        # we need to update it to match the passed `order`.
        if order not in ["k", "K"]:
            return dpnp.asarray(result, order=order)
        # dpnp.ascontiguousarray changes 0-D array to 1-D array
        if result.ndim == 0:
            return result
        return dpnp.ascontiguousarray(result)

    result = dpnp.get_result_array(result, out, casting=casting)
    if axes is not None and out is result:
        # out and out_orig contain the same data but they have different shape
        return out_orig
    return result
