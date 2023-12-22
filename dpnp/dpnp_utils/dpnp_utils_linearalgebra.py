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
import dpctl.tensor._tensor_impl as ti
import numpy

import dpnp
import dpnp.backend.extensions.blas._blas_impl as bi
from dpnp.dpnp_utils import get_usm_allocations

__all__ = ["dpnp_matmul"]


def _gemm_res_dtype(*arrays, sycl_queue, casting):
    """
    Determines the data types for matmul operation and the output array of matmul operation.

    The output array data type is determined based on the Promotion Type Rule
    and device capibilities. The data type used in matmul operation is an 'inexact' data type
    determined based on the output data type and device capabilities.
    Both data types are determined based on the fact that the output array data type can be cast
    to the other data type according to casting rule specified, otherwise a ``TypeError`` is raised.

    Parameters
    ----------
    arrays : {dpnp_array, usm_ndarray}
        Input arrays.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

    Returns
    -------
    gemm_dtype, res_dtype :
        `gemm_dtype` is the data type used in performing matmul calculations.
        The input arrays of matmul function are cast to `gemm_dtype` and then
        the calculations are performed.
        `res_dtype` is the output data type. When the result is obtained, it is cast
        to `res_dtype`.

    """

    res_dtype = dpnp.result_type(*arrays)
    gemm_dtype = dpnp.default_float_type(sycl_queue=sycl_queue)
    if dpnp.issubdtype(res_dtype, dpnp.complexfloating):
        gemm_dtype = (
            dpnp.complex64 if gemm_dtype == dpnp.float32 else dpnp.complex128
        )

    if dpnp.can_cast(res_dtype, gemm_dtype, casting):
        if res_dtype in [
            dpnp.float64,
            dpnp.complex128,
        ]:  # in case device does not support fp64
            return gemm_dtype, gemm_dtype
        elif res_dtype in [
            dpnp.float32,
            dpnp.complex64,
        ]:  # needed dtype is fp32 but device supports fp64
            return res_dtype, res_dtype
        else:
            return gemm_dtype, res_dtype
    else:
        raise TypeError(
            f"Cannot cast ufunc 'matmul' output from dtype({res_dtype}) to dtype({gemm_dtype}) with casting rule {casting}"
        )


def _gemm_batch_matmul(exec_q, x1, x2, res, x1_is_2D, x2_is_2D, dev_tasks_list):
    # If input array is F-contiguous, we need to change the order to C-contiguous.
    # because mkl::gemm_bacth needs each 2D array to be F-contiguous but
    # when the input array is F-contiguous, the data of 2D array
    # that needs to be called in mkl::gemm_batch are not contiguous.
    ht_tasks_list = []
    x1 = _get_gemm_contig_array(x1, dev_tasks_list, ht_tasks_list)
    x2 = _get_gemm_contig_array(x2, dev_tasks_list, ht_tasks_list)

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


def _get_gemm_contig_array(x, dep_events, host_events, dtype=None):
    if dtype is None:
        copy = not x.flags.c_contiguous
    else:
        copy = (
            not x.flags.c_contiguous
            or not x.flags.f_contiguous
            or x.dtype != dtype
        )

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
    dpnp_matmul(x1, x2, out=None, casting="same_kind", order="K", dtype=None)

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

    squeeze_flag = x1_ndim == 1 or x2_ndim == 1
    if x1_ndim == 1:
        x1 = x1[dpnp.newaxis, :]
        x1_ndim = x1.ndim

    if x2_ndim == 1:
        x2 = x2[:, dpnp.newaxis]
        x2_ndim = x2.ndim

    x1_shape = x1.shape
    x2_shape = x2.shape
    if x1_shape[-1] != x2_shape[-2]:
        raise ValueError(
            "Input arrays have a mismatch in their core dimensions. "
            "The core dimensions should follow this signature: (n?,k),(k,m?)->(n?,m?) "
            f"(size {x1_shape[-1]} is different from {x2_shape[-2]})"
        )

    # Determine the appropriate data types
    if dtype is not None:
        res_dtype = dtype
        gemm_dtype, _ = _gemm_res_dtype(
            x1, x2, sycl_queue=exec_q, casting=casting
        )
    else:
        gemm_dtype, res_dtype = _gemm_res_dtype(
            x1, x2, sycl_queue=exec_q, casting=casting
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

    # input arrays should have the proper data type
    # and be C_CONTIGUOUS or F_CONTIGUOUS
    dep_events_list = []
    host_tasks_list = []
    x1 = _get_gemm_contig_array(
        x1, dep_events_list, host_tasks_list, gemm_dtype
    )
    x2 = _get_gemm_contig_array(
        x2, dep_events_list, host_tasks_list, gemm_dtype
    )

    # calculate results
    result = dpnp.empty(
        res_shape,
        dtype=gemm_dtype,
        usm_type=res_usm_type,
        sycl_queue=exec_q,
    )
    if result.size == 0:
        pass
    else:
        if x1.size == 0 or x2.size == 0:
            result.fill(0)
        else:
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

    if squeeze_flag:
        result = dpnp.squeeze(result)

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
        # we must copy `result` to match the passed `order`.
        if order not in ["k", "K"]:
            return dpnp.array(result, copy=False, order=order)
        else:
            return result
    else:
        return dpnp.get_result_array(result, out, casting=casting)
