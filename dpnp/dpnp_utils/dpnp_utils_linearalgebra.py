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
import numpy

import dpnp
import dpnp.backend.extensions.blas._blas_impl as bi
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations

__all__ = ["dpnp_dot", "dpnp_matmul"]


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

    # when shape along any particular dimension is 1,
    # the stride along that dimension is not a
    # meaningful number and is undefined. Here, we
    # standardizing strides before continuing,
    # setting stride to 0 if the shape along that axis is <=1
    if x1_is_2D:
        x1_strides = tuple(
            str_i if sh_i > 1 else 0
            for sh_i, str_i in zip(x1.shape, x1_strides)
        )
    if x2_is_2D:
        x2_strides = tuple(
            str_i if sh_i > 1 else 0
            for sh_i, str_i in zip(x2.shape, x2_strides)
        )

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

    # create result array
    result = dpnp.empty(
        (),
        dtype=dot_dtype,
        usm_type=res_usm_type,
        sycl_queue=exec_q,
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
        raise ValueError(
            "Input arrays have a mismatch in their core dimensions. "
            "The core dimensions should follow this signature: (n?,k),(k,m?)->(n?,m?) "
            f"(size {x1_shape[-1]} is different from {x2_shape[-2]})"
        )

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
                    raise ValueError(
                        "arrays could not be broadcast together with remapped shapes."
                    )
        x1_shape = x1.shape
        x2_shape = x2.shape
        res_shape = tuple(tmp_shape) + (x1_shape[-2], x2_shape[-1])

    # calculate results
    result = dpnp.empty(
        res_shape,
        dtype=gemm_dtype,
        usm_type=res_usm_type,
        sycl_queue=exec_q,
    )
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

    if x1_is_2D and x2_is_2D:
        # add new axes only if one of the input arrays
        # was inehrently 2D
        new_size = max(x1_ndim, x2_ndim)
        for _ in range(new_size - 2):
            result = result[dpnp.newaxis, :]

    if gemm_dtype != res_dtype:
        result = dpnp.astype(result, res_dtype, copy=False)
    if out is None:
        # If `order` was not passed as default
        # we need to update it to match the passed `order`.
        if order not in ["k", "K"]:
            return dpnp.array(result, copy=False, order=order)
        else:
            return result
    else:
        return dpnp.get_result_array(result, out, casting=casting)
