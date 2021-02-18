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

"""Module Backend (Mathematical part)

This module contains interface functions between C backend layer
and the rest of the library

"""


from dpnp.dpnp_utils cimport *
import dpnp
import numpy
cimport numpy


__all__ += [
    "dpnp_absolute",
    "dpnp_add",
    "dpnp_arctan2",
    "dpnp_around",
    "dpnp_ceil",
    "dpnp_conjugate",
    "dpnp_copysign",
    "dpnp_cross",
    "dpnp_cumprod",
    "dpnp_cumsum",
    "dpnp_diff",
    "dpnp_divide",
    "dpnp_ediff1d",
    "dpnp_fabs",
    "dpnp_floor",
    "dpnp_floor_divide",
    "dpnp_fmod",
    "dpnp_gradient",
    'dpnp_hypot',
    "dpnp_maximum",
    "dpnp_minimum",
    "dpnp_modf",
    "dpnp_multiply",
    "dpnp_nancumprod",
    "dpnp_nancumsum",
    "dpnp_nanprod",
    "dpnp_nansum",
    "dpnp_negative",
    "dpnp_power",
    "dpnp_prod",
    "dpnp_sign",
    "dpnp_subtract",
    "dpnp_sum",
    "dpnp_trapz",
    "dpnp_trunc"
]


ctypedef void(*fptr_custom_elemwise_absolute_1in_1out_t)(void * , void * , size_t)
ctypedef void(*fptr_1in_2out_t)(void * , void * , void * , size_t)
ctypedef void(*ftpr_custom_trapz_2in_1out_with_2size_t)(void *, void *, void *, double, size_t, size_t)


cpdef dparray dpnp_absolute(dparray input):
    cdef dparray_shape_type input_shape = input.shape
    cdef size_t input_shape_size = input.ndim

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ABSOLUTE, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(input_shape, dtype=result_type)

    cdef fptr_custom_elemwise_absolute_1in_1out_t func = <fptr_custom_elemwise_absolute_1in_1out_t > kernel_data.ptr
    # call FPTR function
    func(input.get_data(), result.get_data(), input.size)

    return result


cpdef dparray dpnp_add(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_ADD, x1, x2, x1.shape)


cpdef dparray dpnp_arctan2(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_ARCTAN2, x1, x2, x1.shape)


cpdef dpnp_around(dparray a, decimals, out):
    cdef dparray result

    if out is None:
        result = dparray(a.shape, dtype=a.dtype)
    else:
        result = out

    for i in range(result.size):
        result[i] = round(a[i], decimals)

    return result


cpdef dparray dpnp_ceil(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_CEIL, x1, x1.shape)


cpdef dparray dpnp_conjugate(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_CONJIGUATE, x1, x1.shape)


cpdef dparray dpnp_copysign(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_COPYSIGN, x1, x2, x1.shape)


cpdef dparray dpnp_cross(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_CROSS, x1, x2, x1.shape)


cpdef dparray dpnp_cumprod(dparray x1):
    # instead of x1.shape, (x1.size, ) is passed to the function
    # due to the following:
    # >>> import numpy
    # >>> a = numpy.array([[1, 2], [2, 3]])
    # >>> res = numpy.cumprod(a)
    # >>> res.shape
    # (4,)

    return call_fptr_1in_1out(DPNP_FN_CUMPROD, x1, (x1.size,))


cpdef dparray dpnp_cumsum(dparray x1):
    # instead of x1.shape, (x1.size, ) is passed to the function
    # due to the following:
    # >>> import numpy
    # >>> a = numpy.array([[1, 2], [2, 3]])
    # >>> res = numpy.cumsum(a)
    # >>> res.shape
    # (4,)

    return call_fptr_1in_1out(DPNP_FN_CUMSUM, x1, (x1.size,))


cpdef dparray dpnp_diff(dparray input, int n):
    if n == 0:
        return input
    if n < input.shape[-1]:
        arr = input
        for _ in range(n):
            list_shape_i = list(arr.shape)
            list_shape_i[-1] = list_shape_i[-1] - 1
            output_shape = tuple(list_shape_i)
            res = []
            size_idx = output_shape[-1]
            counter = 0
            for i in range(arr.size):
                if counter < size_idx:
                    counter += 1
                    arr_elem = arr.item(i + 1) - arr.item(i)
                    res.append(arr_elem)
                else:
                    counter = 0

            dpnp_array = dpnp.array(res, dtype=input.dtype)
            arr = dpnp_array.reshape(output_shape)
        return arr
    else:
        return dpnp.array([], dtype=input.dtype)


cpdef dparray dpnp_divide(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_DIVIDE, x1, x2, x1.shape)


cpdef dparray dpnp_ediff1d(dparray x1):

    if x1.size <= 1:
        return dpnp.empty(0, dtype=x1.dtype)

    return call_fptr_1in_1out(DPNP_FN_EDIFF1D, x1, (x1.size -1,))


cpdef dparray dpnp_fabs(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_FABS, x1, x1.shape)


cpdef dparray dpnp_floor(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_FLOOR, x1, x1.shape)


cpdef dparray dpnp_floor_divide(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_FLOOR_DIVIDE, x1, x2, x1.shape)


cpdef dparray dpnp_fmod(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_FMOD, x1, x2, x1.shape)


cpdef dparray dpnp_gradient(dparray y1, int dx=1):

    size = y1.size

    cdef dparray result = dparray(size, dtype=dpnp.float64)

    cur = (y1[1] - y1[0]) / dx

    result._setitem_scalar(0, cur)

    cur = (y1[-1] - y1[-2]) / dx

    result._setitem_scalar(size - 1, cur)

    for i in range(1, size - 1):
        cur = (y1[i + 1] - y1[i - 1]) / (2 * dx)
        result._setitem_scalar(i, cur)

    return result


cpdef dparray dpnp_hypot(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_HYPOT, x1, x2, x1.shape)


cpdef dparray dpnp_maximum(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_MAXIMUM, x1, x2, x1.shape)


cpdef dparray dpnp_minimum(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_MINIMUM, x1, x2, x1.shape)


cpdef tuple dpnp_modf(dparray x1):
    """ Convert string type names (dparray.dtype) to C enum DPNPFuncType """
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    """ get the FPTR data structure """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MODF, param1_type, DPNP_FT_NONE)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    """ Create result arrays with type given by FPTR data """
    cdef dparray result1 = dparray(x1.shape, dtype=result_type)
    cdef dparray result2 = dparray(x1.shape, dtype=result_type)

    cdef fptr_1in_2out_t func = <fptr_1in_2out_t > kernel_data.ptr
    """ Call FPTR function """
    func(x1.get_data(), result1.get_data(), result2.get_data(), x1.size)

    return result1, result2


cpdef dparray dpnp_multiply(dparray x1, x2):
    cdef dparray result
    if dpnp.isscalar(x2):
        x2_ = dpnp.array([x2])

        types_map = {
            (dpnp.int32, dpnp.float64): dpnp.float64,
            (dpnp.int64, dpnp.float64): dpnp.float64,
        }

        res_type = types_map.get((x1.dtype.type, x2_.dtype.type), x1.dtype)
        result = dparray(x1.shape, dtype=res_type)
        for i in range(x1.size):
            result[i] = x1[i] * x2
        return result.reshape(x1.shape)
    else:
        return call_fptr_2in_1out(DPNP_FN_MULTIPLY, x1, x2, x1.shape)


cpdef dparray dpnp_nancumprod(dparray x1):

    cur_x1 = dpnp.copy(x1)

    for i in range(cur_x1.size):
        if dpnp.isnan(cur_x1[i]):
            cur_x1._setitem_scalar(i, 1)

    return dpnp_cumprod(cur_x1)


cpdef dparray dpnp_nancumsum(dparray x1):

    cur_x1 = dpnp.copy(x1)

    for i in range(cur_x1.size):
        if dpnp.isnan(cur_x1[i]):
            cur_x1._setitem_scalar(i, 0)

    return dpnp_cumsum(cur_x1)


cpdef dpnp_nanprod(dparray x1):
    cdef dparray result = dparray(x1.shape, dtype=x1.dtype)

    for i in range(result.size):
        input_elem = x1.item(i)

        if dpnp.isnan(input_elem):
            result._setitem_scalar(i, 1)
        else:
            result._setitem_scalar(i, input_elem)

    return dpnp_prod(result)


cpdef dpnp_nansum(dparray x1):
    cdef dparray result = dparray(x1.shape, dtype=x1.dtype)

    for i in range(result.size):
        input_elem = x1.item(i)

        if dpnp.isnan(input_elem):
            result._setitem_scalar(i, 0)
        else:
            result._setitem_scalar(i, input_elem)

    # due to bug in dpnp_sum need this workaround
    # return dpnp_sum(result)
    sum_result = dpnp_sum(result)
    return x1.dtype.type(sum_result[0])


cpdef dparray dpnp_negative(dparray array1):
    cdef dparray result = dparray(array1.shape, dtype=array1.dtype)

    for i in range(result.size):
        result[i] = -(array1[i])

    return result


cpdef dparray dpnp_power(dparray x1, x2):
    cdef dparray result
    if dpnp.isscalar(x2):
        x2_ = dpnp.array([x2])

        types_map = {
            (dpnp.int32, dpnp.float64): dpnp.float64,
            (dpnp.int64, dpnp.float64): dpnp.float64,
        }

        res_type = types_map.get((x1.dtype.type, x2_.dtype.type), x1.dtype)

        result = dparray(x1.shape, dtype=res_type)
        for i in range(x1.size):
            result[i] = x1[i] ** x2
        return result
    else:
        return call_fptr_2in_1out(DPNP_FN_POWER, x1, x2, x1.shape)


cpdef dpnp_prod(dparray x1):
    """
    input:float64   : outout:float64   : name:prod
    input:float32   : outout:float32   : name:prod
    input:int64     : outout:int64     : name:prod
    input:int32     : outout:int64     : name:prod
    input:bool      : outout:int64     : name:prod
    input:complex64 : outout:complex64 : name:prod
    input:complex128: outout:complex128: name:prod
    """

    cdef dparray result = call_fptr_1in_1out(DPNP_FN_PROD, x1, (1,))

    """ Numpy interface inconsistency """
    return_type = numpy.dtype(numpy.int64) if (x1.dtype == numpy.int32) else x1.dtype

    return return_type.type(result[0])


cpdef dparray dpnp_remainder(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_REMAINDER, x1, x2, x1.shape)


cpdef dparray dpnp_sign(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_SIGN, x1, x1.shape)


cpdef dparray dpnp_subtract(dparray x1, dparray x2):
    return call_fptr_2in_1out(DPNP_FN_SUBTRACT, x1, x2, x1.shape)


cpdef dparray dpnp_sum(dparray input, object axis=None, object dtype=None, dparray out=None, cpp_bool keepdims=False, object initial=None, object where=True):

    cdef dparray_shape_type input_shape = input.shape
    cdef DPNPFuncType input_c_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef dparray_shape_type axis_shape = _object_to_tuple(axis)

    cdef dparray_shape_type result_shape = get_reduction_output_shape(input_shape, axis, keepdims)
    cdef DPNPFuncType result_c_type = get_output_c_type(DPNP_FN_SUM, input_c_type, out, dtype)

    """ select kernel """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SUM, input_c_type, result_c_type)

    """ Create result array """
    cdef dparray result = create_output_array(result_shape, result_c_type, out)
    cdef dpnp_reduction_c_t func = <dpnp_reduction_c_t > kernel_data.ptr

    """ Call FPTR interface function """
    func(input.get_data(), input.size, result.get_data(), input_shape.data(),
         input_shape.size(), axis_shape.data(), axis_shape.size(), NULL, NULL)

    return result


cpdef dpnp_trapz(dparray y1, dparray x1, double dx):

    if y1.size <= 1:
        if y1.dtype == dpnp.float32:
            return dpnp.array([0], dtype=dpnp.float32)
        return dpnp.array([0], dtype=dpnp.float64)

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(y1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRAPZ, param1_type, param2_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    cdef dparray result = dparray((1,), dtype=result_type)

    cdef ftpr_custom_trapz_2in_1out_with_2size_t func = <ftpr_custom_trapz_2in_1out_with_2size_t > kernel_data.ptr
    func(y1.get_data(), x1.get_data(), result.get_data(), dx, y1.size, x1.size)

    return result[0]


cpdef dparray dpnp_trunc(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_TRUNC, x1, x1.shape)
