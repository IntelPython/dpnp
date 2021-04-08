# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
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

"""Module Backend

This module contains interface functions between C backend layer
and the rest of the library

"""

import dpnp
from dpnp.dpnp_utils cimport *
from dpnp.dpnp_algo cimport *
from dpnp.dparray cimport dparray, dparray_shape_type
import numpy
cimport numpy


__all__ = [
    "dpnp_cholesky",
    "dpnp_cond",
    "dpnp_det",
    "dpnp_eig",
    "dpnp_eigvals",
    "dpnp_inv",
    "dpnp_matrix_rank",
    "dpnp_norm",
    "dpnp_svd",
]


# C function pointer to the C library template functions
ctypedef void(*custom_linalg_1in_1out_func_ptr_t)(void *, void * , size_t * , size_t)
ctypedef void(*custom_linalg_1in_1out_func_ptr_t_)(void * , void * , size_t * )
ctypedef void(*custom_linalg_1in_1out_with_size_func_ptr_t_)(void *, void * , size_t)
ctypedef void(*custom_linalg_1in_1out_with_2size_func_ptr_t_)(void *, void * , size_t, size_t)
ctypedef void(*custom_linalg_1in_3out_shape_t)(void *, void * , void * , void * , size_t , size_t )


cpdef dparray dpnp_cholesky(dparray input):
    if input.dtype == dpnp.int32 or input.dtype == dpnp.int64:
        input_ = input.astype(dpnp.float64)
    else:
        input_ = input

    size_ = input_.shape[-1]

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input_.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CHOLESKY, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(input_.shape, dtype=result_type)

    cdef custom_linalg_1in_1out_with_2size_func_ptr_t_ func = <custom_linalg_1in_1out_with_2size_func_ptr_t_ > kernel_data.ptr

    func(input_.get_data(), result.get_data(), input.size, size_)

    return result


cpdef dparray dpnp_cond(dparray input, p):
    if p in ('f', 'fro'):
        input = input.ravel(order='K')
        sqnorm = dpnp.dot(input, input)
        res = dpnp.sqrt(sqnorm)
        ret = dpnp.array([res])
    elif p == numpy.inf:
        dpnp_sum_val = dpnp.array([dpnp.sum(dpnp.abs(input), axis=1)])
        ret = dpnp.array([dpnp_sum_val.max()])
    elif p == -numpy.inf:
        dpnp_sum_val = dpnp.array([dpnp.sum(dpnp.abs(input), axis=1)])
        ret = dpnp.array([dpnp_sum_val.min()])
    elif p == 1:
        dpnp_sum_val = dpnp.array([dpnp.sum(dpnp.abs(input), axis=0)])
        ret = dpnp.array([dpnp_sum_val.max()])
    elif p == -1:
        dpnp_sum_val = dpnp.array([dpnp.sum(dpnp.abs(input), axis=0)])
        ret = dpnp.array([dpnp_sum_val.min()])
    else:
        ret = dpnp.array([input.item(0)])
    return ret


cpdef dparray dpnp_det(dparray input):
    cdef size_t n = input.shape[-1]
    cdef size_t size_out = 1
    if input.ndim != 2:
        output_shape = tuple((list(input.shape))[:-2])
        for i in range(len(output_shape)):
            size_out *= output_shape[i]

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DET, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray(size_out, dtype=result_type)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim)

    if size_out > 1:
        dpnp_result = result.reshape(output_shape)
        return dpnp_result
    else:
        return result


cpdef tuple dpnp_eig(dparray x1):
    cdef dparray_shape_type x1_shape = x1.shape

    cdef size_t size = 0 if x1_shape.empty() else x1_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIG, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)

    cdef dparray res_val = dparray((size,), dtype=result_type)
    cdef dparray res_vec = dparray(x1_shape, dtype=result_type)

    cdef fptr_2in_1out_t func = <fptr_2in_1out_t > kernel_data.ptr
    # call FPTR function
    func(x1.get_data(), res_val.get_data(), res_vec.get_data(), size)

    return (res_val, res_vec)


cpdef dparray dpnp_eigvals(dparray input):
    cdef dparray_shape_type input_shape = input.shape

    cdef size_t size = 0 if input_shape.empty() else input_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIGVALS, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)

    cdef dparray res_val = dparray((size,), dtype=result_type)

    cdef custom_linalg_1in_1out_with_size_func_ptr_t_ func = <custom_linalg_1in_1out_with_size_func_ptr_t_ > kernel_data.ptr
    # call FPTR function
    func(input.get_data(), res_val.get_data(), size)

    return res_val


cpdef dparray dpnp_inv(dparray input_):
    cdef dparray input = input_.astype(dpnp.float64)
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_INV, param1_type, param1_type)

    cdef dparray result = dparray(input.size, dtype=dpnp.float64)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim)

    dpnp_result = result.reshape(input.shape)
    return dpnp_result


cpdef dparray dpnp_matrix_rank(dparray input):
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MATRIX_RANK, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)
    cdef dparray result = dparray((1,), dtype=result_type)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), < size_t * > input._dparray_shape.data(), input.ndim)

    return result


cpdef dparray dpnp_norm(dparray input, ord=None, axis=None):
    cdef long size_input = input.size
    cdef dparray_shape_type shape_input = input.shape

    if input.dtype == numpy.float32:
        res_type = numpy.float32
    else:
        res_type = numpy.float64

    if size_input == 0:
        return dpnp.array([numpy.nan], dtype=res_type)

    if isinstance(axis, int):
        axis_ = tuple([axis])
    else:
        axis_ = axis

    ndim = input.ndim
    if axis is None:
        if ((ord is None) or
            (ord in ('f', 'fro') and ndim == 2) or
                (ord == 2 and ndim == 1)):

            input = input.ravel(order='K')
            sqnorm = dpnp.dot(input, input)
            ret = dpnp.sqrt(sqnorm)
            return dpnp.array([ret], dtype=res_type)

    len_axis = 1 if axis is None else len(axis_)
    if len_axis == 1:
        if ord == numpy.inf:
            return dpnp.array([dpnp.abs(input).max(axis=axis)])
        elif ord == -numpy.inf:
            return dpnp.array([dpnp.abs(input).min(axis=axis)])
        elif ord == 0:
            return dpnp.array([(input != 0).astype(input.dtype).sum(axis=axis)])
        elif ord is None or ord == 2:
            s = input * input
            return dpnp.sqrt(dpnp.sum(s, axis=axis))
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = dpnp.abs(input)
            absx_size = absx.size
            absx_power = dparray(absx_size, dtype=absx.dtype)
            for i in range(absx_size):
                absx_elem = absx.item(i)
                absx_power[i] = absx_elem ** ord
            absx_ = absx_power.reshape(absx.shape)
            ret = dpnp.sum(absx_, axis=axis)
            ret_size = ret.size
            ret_power = dparray(ret_size)
            for i in range(ret_size):
                ret_elem = ret.item(i)
                ret_power[i] = ret_elem ** (1 / ord)
            ret_ = ret_power.reshape(ret.shape)
            return ret_
    elif len_axis == 2:
        row_axis, col_axis = axis_
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        # if ord == 2:
        #     ret =  _multi_svd_norm(input, row_axis, col_axis, amax)
        # elif ord == -2:
        #     ret = _multi_svd_norm(input, row_axis, col_axis, amin)
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            dpnp_sum_val_ = dpnp.sum(dpnp.abs(input), axis=row_axis)
            dpnp_sum_val = dpnp_sum_val_ if isinstance(dpnp_sum_val_, dparray) else dpnp.array([dpnp_sum_val_])
            dpnp_max_val = dpnp_sum_val.min(axis=col_axis)
            ret = dpnp_max_val if isinstance(dpnp_max_val, dparray) else dpnp.array([dpnp_max_val])
        elif ord == numpy.inf:
            if row_axis > col_axis:
                row_axis -= 1
            dpnp_sum_val_ = dpnp.sum(dpnp.abs(input), axis=col_axis)
            dpnp_sum_val = dpnp_sum_val_ if isinstance(dpnp_sum_val_, dparray) else dpnp.array([dpnp_sum_val_])
            dpnp_max_val = dpnp_sum_val.max(axis=row_axis)
            ret = dpnp_max_val if isinstance(dpnp_max_val, dparray) else dpnp.array([dpnp_max_val])
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            dpnp_sum_val_ = dpnp.sum(dpnp.abs(input), axis=row_axis)
            dpnp_sum_val = dpnp_sum_val_ if isinstance(dpnp_sum_val_, dparray) else dpnp.array([dpnp_sum_val_])
            dpnp_min_val = dpnp_sum_val.min(axis=col_axis)
            ret = dpnp_min_val if isinstance(dpnp_min_val, dparray) else dpnp.array([dpnp_min_val])
        elif ord == -numpy.inf:
            if row_axis > col_axis:
                row_axis -= 1
            dpnp_sum_val_ = dpnp.sum(dpnp.abs(input), axis=col_axis)
            dpnp_sum_val = dpnp_sum_val_ if isinstance(dpnp_sum_val_, dparray) else dpnp.array([dpnp_sum_val_])
            dpnp_min_val = dpnp_sum_val.min(axis=row_axis)
            ret = dpnp_min_val if isinstance(dpnp_min_val, dparray) else dpnp.array([dpnp_min_val])
        elif ord in [None, 'fro', 'f']:
            ret = dpnp.sqrt(dpnp.sum(input * input, axis=axis))
        # elif ord == 'nuc':
        #     ret = _multi_svd_norm(input, row_axis, col_axis, sum)
        else:
            raise ValueError("Invalid norm order for matrices.")
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")


cpdef tuple dpnp_svd(dparray x1, full_matrices, compute_uv, hermitian):
    cdef size_t size_m = x1.shape[0]
    cdef size_t size_n = x1.shape[1]

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SVD, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)

    if x1.dtype == dpnp.float32:
        type_s = dpnp.float32
    else:
        type_s = dpnp.float64

    size_s = min(size_m, size_n)

    cdef dparray res_u = dparray((size_m, size_m), dtype=result_type)
    cdef dparray res_s = dparray((size_s, ), dtype=type_s)
    cdef dparray res_vt = dparray((size_n, size_n), dtype=result_type)

    cdef custom_linalg_1in_3out_shape_t func = < custom_linalg_1in_3out_shape_t > kernel_data.ptr

    func(x1.get_data(), res_u.get_data(), res_s.get_data(), res_vt.get_data(), size_m, size_n)

    return (res_u, res_s, res_vt)
