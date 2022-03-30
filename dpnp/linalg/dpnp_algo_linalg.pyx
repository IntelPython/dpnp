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

import numpy
from dpnp.dpnp_algo cimport *
import dpnp.dpnp_utils as utils_py
import dpnp
cimport dpnp.dpnp_utils as utils

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
    "dpnp_qr",
    "dpnp_svd",
]


# C function pointer to the C library template functions
ctypedef void(*custom_linalg_1in_1out_func_ptr_t)(void *, void * , shape_elem_type * , size_t)
ctypedef void(*custom_linalg_1in_1out_func_ptr_t_)(void * , void * , size_t * )
ctypedef void(*custom_linalg_1in_1out_with_size_func_ptr_t_)(void *, void * , size_t)
ctypedef void(*custom_linalg_1in_1out_with_2size_func_ptr_t_)(void *, void * , size_t, size_t)
ctypedef void(*custom_linalg_1in_3out_shape_t)(void *, void * , void * , void * , size_t , size_t )
ctypedef void(*custom_linalg_2in_1out_func_ptr_t)(void *, void * , void * , size_t )


cpdef utils.dpnp_descriptor dpnp_cholesky(utils.dpnp_descriptor input_):
    size_ = input_.shape[-1]

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input_.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_CHOLESKY, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(input_.shape, kernel_data.return_type, None)

    cdef custom_linalg_1in_1out_with_2size_func_ptr_t_ func = <custom_linalg_1in_1out_with_2size_func_ptr_t_ > kernel_data.ptr

    func(input_.get_data(), result.get_data(), input_.size, size_)

    return result


cpdef object dpnp_cond(object input, object p):
    if p in ('f', 'fro'):
        input = dpnp.ravel(input, order='K')
        sqnorm = dpnp.dot(input, input)
        res = dpnp.sqrt(sqnorm)
        ret = dpnp.array([res])
    elif p == numpy.inf:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=1)
        ret = dpnp.max(dpnp_sum_val)
    elif p == -numpy.inf:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=1)
        ret = dpnp.min(dpnp_sum_val)
    elif p == 1:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=0)
        ret = dpnp.max(dpnp_sum_val)
    elif p == -1:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=0)
        ret = dpnp.min(dpnp_sum_val)
    else:
        ret = dpnp.array([input.item(0)])
    return ret


cpdef utils.dpnp_descriptor dpnp_det(utils.dpnp_descriptor input):
    cdef shape_type_c input_shape = input.shape
    cdef size_t n = input.shape[-1]
    cdef size_t size_out = 1
    if input.ndim != 2:
        output_shape = tuple((list(input.shape))[:-2])
        for i in range(len(output_shape)):
            size_out *= output_shape[i]

    cdef shape_type_c result_shape = (size_out,)
    if size_out > 1:
        result_shape = output_shape

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_DET, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), input_shape.data(), input.ndim)

    return result


cpdef tuple dpnp_eig(utils.dpnp_descriptor x1):
    cdef shape_type_c x1_shape = x1.shape

    cdef size_t size = 0 if x1_shape.empty() else x1_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIG, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)

    cdef utils.dpnp_descriptor res_val = utils.create_output_descriptor((size,), kernel_data.return_type, None)
    cdef utils.dpnp_descriptor res_vec = utils.create_output_descriptor(x1_shape, kernel_data.return_type, None)

    cdef custom_linalg_2in_1out_func_ptr_t func = <custom_linalg_2in_1out_func_ptr_t > kernel_data.ptr
    # call FPTR function
    func(x1.get_data(), res_val.get_data(), res_vec.get_data(), size)

    return (res_val.get_pyobj(), res_vec.get_pyobj())


cpdef utils.dpnp_descriptor dpnp_eigvals(utils.dpnp_descriptor input):
    cdef shape_type_c input_shape = input.shape

    cdef size_t size = 0 if input_shape.empty() else input_shape.front()

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EIGVALS, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor res_val = utils.create_output_descriptor((size,), kernel_data.return_type, None)

    cdef custom_linalg_1in_1out_with_size_func_ptr_t_ func = <custom_linalg_1in_1out_with_size_func_ptr_t_ > kernel_data.ptr
    # call FPTR function
    func(input.get_data(), res_val.get_data(), size)

    return res_val


cpdef utils.dpnp_descriptor dpnp_inv(utils.dpnp_descriptor input):
    cdef shape_type_c input_shape = input.shape

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_INV, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(input_shape, kernel_data.return_type, None)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), input_shape.data(), input.ndim)

    return result


cpdef utils.dpnp_descriptor dpnp_matrix_rank(utils.dpnp_descriptor input):
    cdef shape_type_c input_shape = input.shape
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MATRIX_RANK, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor((1,), kernel_data.return_type, None)

    cdef custom_linalg_1in_1out_func_ptr_t func = <custom_linalg_1in_1out_func_ptr_t > kernel_data.ptr

    func(input.get_data(), result.get_data(), input_shape.data(), input.ndim)

    return result


cpdef object dpnp_norm(object input, ord=None, axis=None):
    cdef long size_input = input.size
    cdef shape_type_c shape_input = input.shape

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

            input = dpnp.ravel(input, order='K')
            sqnorm = dpnp.dot(input, input)
            ret = dpnp.sqrt([sqnorm])
            return dpnp.array(ret.reshape(1, *ret.shape), dtype=res_type)

    len_axis = 1 if axis is None else len(axis_)
    if len_axis == 1:
        if ord == numpy.inf:
            return dpnp.array([dpnp.abs(input).max(axis=axis)])
        elif ord == -numpy.inf:
            return dpnp.array([dpnp.abs(input).min(axis=axis)])
        elif ord == 0:
            return input.dtype.type(dpnp.count_nonzero(input, axis=axis))
        elif ord is None or ord == 2:
            s = input * input
            return dpnp.sqrt(dpnp.sum(s, axis=axis))
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = dpnp.abs(input)
            absx_size = absx.size
            absx_power = utils_py.create_output_descriptor_py((absx_size,), absx.dtype, None).get_pyobj()

            absx_flatiter = absx.flat

            for i in range(absx_size):
                absx_elem = absx_flatiter[i]
                absx_power[i] = absx_elem ** ord
            absx_ = dpnp.reshape(absx_power, absx.shape)
            ret = dpnp.sum(absx_, axis=axis)
            ret_size = ret.size
            ret_power = utils_py.create_output_descriptor_py((ret_size,), None, None).get_pyobj()

            ret_flatiter = ret.flat

            for i in range(ret_size):
                ret_elem = ret_flatiter[i]
                ret_power[i] = ret_elem ** (1 / ord)
            ret_ = dpnp.reshape(ret_power, ret.shape)
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
            dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=row_axis)
            ret = dpnp_sum_val.min(axis=col_axis)
        elif ord == numpy.inf:
            if row_axis > col_axis:
                row_axis -= 1
            dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=col_axis)
            ret = dpnp_sum_val.max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=row_axis)
            ret = dpnp_sum_val.min(axis=col_axis)
        elif ord == -numpy.inf:
            if row_axis > col_axis:
                row_axis -= 1
            dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=col_axis)
            ret = dpnp_sum_val.min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            ret = dpnp.sqrt(dpnp.sum(input * input, axis=axis))
        # elif ord == 'nuc':
        #     ret = _multi_svd_norm(input, row_axis, col_axis, sum)
        else:
            raise ValueError("Invalid norm order for matrices.")

        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")


cpdef tuple dpnp_qr(utils.dpnp_descriptor x1, str mode):
    cdef size_t size_m = x1.shape[0]
    cdef size_t size_n = x1.shape[1]
    cdef size_t min_m_n = min(size_m, size_n)
    cdef size_t size_tau = min_m_n

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_QR, param1_type, param1_type)

    cdef utils.dpnp_descriptor res_q = utils.create_output_descriptor((size_m, min_m_n), kernel_data.return_type, None)
    cdef utils.dpnp_descriptor res_r = utils.create_output_descriptor((min_m_n, size_n), kernel_data.return_type, None)
    cdef utils.dpnp_descriptor tau = utils.create_output_descriptor((size_tau, ), kernel_data.return_type, None)

    cdef custom_linalg_1in_3out_shape_t func = < custom_linalg_1in_3out_shape_t > kernel_data.ptr

    func(x1.get_data(), res_q.get_data(), res_r.get_data(), tau.get_data(), size_m, size_n)

    return (res_q.get_pyobj(), res_r.get_pyobj())


cpdef tuple dpnp_svd(utils.dpnp_descriptor x1, cpp_bool full_matrices, cpp_bool compute_uv, cpp_bool hermitian):
    cdef size_t size_m = x1.shape[0]
    cdef size_t size_n = x1.shape[1]
    cdef size_t size_s = min(size_m, size_n)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SVD, param1_type, param1_type)

    cdef DPNPFuncType type_s = DPNP_FT_DOUBLE
    if x1.dtype == dpnp.float32:
        type_s = DPNP_FT_FLOAT

    cdef utils.dpnp_descriptor res_u = utils.create_output_descriptor((size_m, size_m), kernel_data.return_type, None)
    cdef utils.dpnp_descriptor res_s = utils.create_output_descriptor((size_s, ), type_s, None)
    cdef utils.dpnp_descriptor res_vt = utils.create_output_descriptor((size_n, size_n), kernel_data.return_type, None)

    cdef custom_linalg_1in_3out_shape_t func = < custom_linalg_1in_3out_shape_t > kernel_data.ptr

    func(x1.get_data(), res_u.get_data(), res_s.get_data(), res_vt.get_data(), size_m, size_n)

    return (res_u.get_pyobj(), res_s.get_pyobj(), res_vt.get_pyobj())
