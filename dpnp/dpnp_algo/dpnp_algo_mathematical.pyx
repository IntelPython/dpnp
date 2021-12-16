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

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

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
    "dpnp_remainder",
    "dpnp_sign",
    "dpnp_subtract",
    "dpnp_sum",
    "dpnp_trapz",
    "dpnp_trunc"
]


ctypedef void(*fptr_custom_elemwise_absolute_1in_1out_t)(void * , void * , size_t)
ctypedef void(*fptr_1in_2out_t)(void * , void * , void * , size_t)
ctypedef void(*ftpr_custom_trapz_2in_1out_with_2size_t)(void *, void * , void * , double, size_t, size_t)
ctypedef void(*ftpr_custom_around_1in_1out_t)(const void * , void * , const size_t, const int)


cpdef utils.dpnp_descriptor dpnp_absolute(utils.dpnp_descriptor input):
    cdef shape_type_c input_shape = input.shape
    cdef size_t input_shape_size = input.ndim

    # convert string type names (array.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_ABSOLUTE, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(input_shape, kernel_data.return_type, None)

    cdef fptr_custom_elemwise_absolute_1in_1out_t func = <fptr_custom_elemwise_absolute_1in_1out_t > kernel_data.ptr
    # call FPTR function
    func(input.get_data(), result.get_data(), input.size)

    return result


cpdef utils.dpnp_descriptor dpnp_add(utils.dpnp_descriptor x1_obj,
                                     utils.dpnp_descriptor x2_obj,
                                     object dtype=None,
                                     utils.dpnp_descriptor out=None,
                                     object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_ADD, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_arctan2(utils.dpnp_descriptor x1_obj,
                                         utils.dpnp_descriptor x2_obj,
                                         object dtype=None,
                                         utils.dpnp_descriptor out=None,
                                         object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_ARCTAN2, x1_obj, x2_obj, dtype, out, where, func_name="arctan2")


cpdef utils.dpnp_descriptor dpnp_around(utils.dpnp_descriptor x1, int decimals):

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_AROUND, param1_type, param1_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = x1.shape
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef ftpr_custom_around_1in_1out_t func = <ftpr_custom_around_1in_1out_t > kernel_data.ptr

    func(x1.get_data(), result.get_data(), x1.size, decimals)

    return result


cpdef utils.dpnp_descriptor dpnp_ceil(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_CEIL, x1, dtype=None, out=out, where=True, func_name='ceil')


cpdef utils.dpnp_descriptor dpnp_conjugate(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_CONJIGUATE, x1)


cpdef utils.dpnp_descriptor dpnp_copysign(utils.dpnp_descriptor x1_obj,
                                          utils.dpnp_descriptor x2_obj,
                                          object dtype=None,
                                          utils.dpnp_descriptor out=None,
                                          object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_COPYSIGN, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_cross(utils.dpnp_descriptor x1_obj,
                                       utils.dpnp_descriptor x2_obj,
                                       object dtype=None,
                                       utils.dpnp_descriptor out=None,
                                       object where=True):
    return call_fptr_2in_1out(DPNP_FN_CROSS, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_cumprod(utils.dpnp_descriptor x1):
    # instead of x1.shape, (x1.size, ) is passed to the function
    # due to the following:
    # >>> import numpy
    # >>> a = numpy.array([[1, 2], [2, 3]])
    # >>> res = numpy.cumprod(a)
    # >>> res.shape
    # (4,)

    return call_fptr_1in_1out(DPNP_FN_CUMPROD, x1, (x1.size,))


cpdef utils.dpnp_descriptor dpnp_cumsum(utils.dpnp_descriptor x1):
    # instead of x1.shape, (x1.size, ) is passed to the function
    # due to the following:
    # >>> import numpy
    # >>> a = numpy.array([[1, 2], [2, 3]])
    # >>> res = numpy.cumsum(a)
    # >>> res.shape
    # (4,)

    return call_fptr_1in_1out(DPNP_FN_CUMSUM, x1, (x1.size,))


cpdef utils.dpnp_descriptor dpnp_diff(utils.dpnp_descriptor x1, int n):
    cdef utils.dpnp_descriptor res

    if x1.size - n < 1:
        res = utils.dpnp_descriptor(dpnp.empty(0, dtype=x1.dtype))
        return res

    res = utils.dpnp_descriptor(dpnp.empty(x1.size - 1, dtype=x1.dtype))
    for i in range(res.size):
        res.get_pyobj()[i] = x1.get_pyobj()[i+1] - x1.get_pyobj()[i]

    if n == 1:
        return res

    return dpnp_diff(res, n-1)


cpdef utils.dpnp_descriptor dpnp_divide(utils.dpnp_descriptor x1_obj,
                                        utils.dpnp_descriptor x2_obj,
                                        object dtype=None,
                                        utils.dpnp_descriptor out=None,
                                        object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_DIVIDE, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_ediff1d(utils.dpnp_descriptor x1):

    if x1.size <= 1:
        return utils.dpnp_descriptor(dpnp.empty(0, dtype=x1.dtype))  # TODO need to call dpnp_empty instead

    # Convert type (x1.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_EDIFF1D, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)

    # Currently shape and strides of the input array are not took into account for the function ediff1d
    cdef shape_type_c x1_shape = (x1.size,)
    cdef shape_type_c x1_strides = utils.strides_to_vector(None, x1_shape)

    cdef shape_type_c result_shape = (x1.size - 1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)
    cdef shape_type_c result_strides = utils.strides_to_vector(result.strides, result_shape)

    # Call FPTR function
    cdef fptr_1in_1out_strides_t func = <fptr_1in_1out_strides_t > kernel_data.ptr
    func(result.get_data(),
         result.size,
         result.ndim,
         result_shape.data(),
         result_strides.data(),
         x1.get_data(),
         x1.size,
         x1.ndim,
         x1_shape.data(),
         x1_strides.data(),
         NULL)

    return result


cpdef utils.dpnp_descriptor dpnp_fabs(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_FABS, x1)


cpdef utils.dpnp_descriptor dpnp_floor(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_FLOOR, x1, dtype=None, out=out, where=True, func_name='floor')


cpdef utils.dpnp_descriptor dpnp_floor_divide(utils.dpnp_descriptor x1_obj,
                                              utils.dpnp_descriptor x2_obj,
                                              object dtype=None,
                                              utils.dpnp_descriptor out=None,
                                              object where=True):
    return call_fptr_2in_1out(DPNP_FN_FLOOR_DIVIDE, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_fmod(utils.dpnp_descriptor x1_obj,
                                      utils.dpnp_descriptor x2_obj,
                                      object dtype=None,
                                      utils.dpnp_descriptor out=None,
                                      object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_FMOD, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_gradient(utils.dpnp_descriptor y1, int dx=1):

    cdef size_t size = y1.size

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = utils._object_to_tuple(size)
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(result_shape, dpnp.float64, None)

    cdef double cur = (y1.get_pyobj()[1] - y1.get_pyobj()[0]) / dx

    result.get_pyobj().flat[0] = cur

    cur = (y1.get_pyobj()[-1] - y1.get_pyobj()[-2]) / dx

    result.get_pyobj().flat[size - 1] = cur

    for i in range(1, size - 1):
        cur = (y1.get_pyobj()[i + 1] - y1.get_pyobj()[i - 1]) / (2 * dx)
        result.get_pyobj().flat[i] = cur

    return result


cpdef utils.dpnp_descriptor dpnp_hypot(utils.dpnp_descriptor x1_obj,
                                       utils.dpnp_descriptor x2_obj,
                                       object dtype=None,
                                       utils.dpnp_descriptor out=None,
                                       object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_HYPOT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_maximum(utils.dpnp_descriptor x1_obj,
                                         utils.dpnp_descriptor x2_obj,
                                         object dtype=None,
                                         utils.dpnp_descriptor out=None,
                                         object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_MAXIMUM, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_minimum(utils.dpnp_descriptor x1_obj,
                                         utils.dpnp_descriptor x2_obj,
                                         object dtype=None,
                                         utils.dpnp_descriptor out=None,
                                         object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_MINIMUM, x1_obj, x2_obj, dtype, out, where)


cpdef tuple dpnp_modf(utils.dpnp_descriptor x1):
    """ Convert string type names (array.dtype) to C enum DPNPFuncType """
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    """ get the FPTR data structure """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_MODF, param1_type, DPNP_FT_NONE)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = x1.shape
    cdef utils.dpnp_descriptor result1 = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)
    cdef utils.dpnp_descriptor result2 = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef fptr_1in_2out_t func = <fptr_1in_2out_t > kernel_data.ptr
    """ Call FPTR function """
    func(x1.get_data(), result1.get_data(), result2.get_data(), x1.size)

    return (result1.get_pyobj(), result2.get_pyobj())


cpdef utils.dpnp_descriptor dpnp_multiply(utils.dpnp_descriptor x1_obj,
                                          utils.dpnp_descriptor x2_obj,
                                          object dtype=None,
                                          utils.dpnp_descriptor out=None,
                                          object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_MULTIPLY, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_nancumprod(utils.dpnp_descriptor x1):
    cur_x1 = dpnp_copy(x1).get_pyobj()

    for i in range(cur_x1.size):
        if dpnp.isnan(cur_x1[numpy.unravel_index(i, cur_x1.shape)]):
            cur_x1[numpy.unravel_index(i, cur_x1.shape)] = 1

    x1_desc = dpnp.get_dpnp_descriptor(cur_x1)
    return dpnp_cumprod(x1_desc)


cpdef utils.dpnp_descriptor dpnp_nancumsum(utils.dpnp_descriptor x1):
    cur_x1 = dpnp_copy(x1).get_pyobj()

    for i in range(cur_x1.size):
        if dpnp.isnan(cur_x1[numpy.unravel_index(i, cur_x1.shape)]):
            cur_x1[numpy.unravel_index(i, cur_x1.shape)] = 0

    x1_desc = dpnp.get_dpnp_descriptor(cur_x1)
    return dpnp_cumsum(x1_desc)


cpdef utils.dpnp_descriptor dpnp_nanprod(utils.dpnp_descriptor x1):
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(x1.shape, x1.dtype, None)

    for i in range(result.size):
        input_elem = x1.get_pyobj().flat[i]

        if dpnp.isnan(input_elem):
            result.get_pyobj().flat[i] = 1
        else:
            result.get_pyobj().flat[i] = input_elem

    return dpnp_prod(result)


cpdef utils.dpnp_descriptor dpnp_nansum(utils.dpnp_descriptor x1):
    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(x1.shape, x1.dtype, None)

    for i in range(result.size):
        input_elem = x1.get_pyobj().flat[i]

        if dpnp.isnan(input_elem):
            result.get_pyobj().flat[i] = 0
        else:
            result.get_pyobj().flat[i] = input_elem

    return dpnp_sum(result)


cpdef utils.dpnp_descriptor dpnp_negative(dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_NEGATIVE, x1)


cpdef utils.dpnp_descriptor dpnp_power(utils.dpnp_descriptor x1_obj,
                                       utils.dpnp_descriptor x2_obj,
                                       object dtype=None,
                                       utils.dpnp_descriptor out=None,
                                       object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_POWER, x1_obj, x2_obj, dtype, out, where, func_name="power")


cpdef utils.dpnp_descriptor dpnp_prod(utils.dpnp_descriptor input,
                                      object axis=None,
                                      object dtype=None,
                                      utils.dpnp_descriptor out=None,
                                      cpp_bool keepdims=False,
                                      object initial=None,
                                      object where=True):
    """
    input:float64   : outout:float64   : name:prod
    input:float32   : outout:float32   : name:prod
    input:int64     : outout:int64     : name:prod
    input:int32     : outout:int64     : name:prod
    input:bool      : outout:int64     : name:prod
    input:complex64 : outout:complex64 : name:prod
    input:complex128: outout:complex128: name:prod
    """

    cdef shape_type_c input_shape = input.shape
    cdef DPNPFuncType input_c_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef shape_type_c axis_shape = utils._object_to_tuple(axis)

    cdef shape_type_c result_shape = utils.get_reduction_output_shape(input_shape, axis, keepdims)
    cdef DPNPFuncType result_c_type = utils.get_output_c_type(DPNP_FN_PROD, input_c_type, out, dtype)

    """ select kernel """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_PROD, input_c_type, result_c_type)

    """ Create result array """
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, result_c_type, out)
    cdef dpnp_reduction_c_t func = <dpnp_reduction_c_t > kernel_data.ptr

    """ Call FPTR interface function """
    func(result.get_data(), input.get_data(), input_shape.data(), input_shape.size(), axis_shape.data(), axis_shape.size(), NULL, NULL)

    return result


cpdef utils.dpnp_descriptor dpnp_remainder(utils.dpnp_descriptor x1_obj,
                                           utils.dpnp_descriptor x2_obj,
                                           object dtype=None,
                                           utils.dpnp_descriptor out=None,
                                           object where=True):
    return call_fptr_2in_1out(DPNP_FN_REMAINDER, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_sign(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_SIGN, x1)


cpdef utils.dpnp_descriptor dpnp_subtract(utils.dpnp_descriptor x1_obj,
                                          utils.dpnp_descriptor x2_obj,
                                          object dtype=None,
                                          utils.dpnp_descriptor out=None,
                                          object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_SUBTRACT, x1_obj, x2_obj, dtype, out, where)


cpdef utils.dpnp_descriptor dpnp_sum(utils.dpnp_descriptor input,
                                     object axis=None,
                                     object dtype=None,
                                     utils.dpnp_descriptor out=None,
                                     cpp_bool keepdims=False,
                                     object initial=None,
                                     object where=True):

    cdef shape_type_c input_shape = input.shape
    cdef DPNPFuncType input_c_type = dpnp_dtype_to_DPNPFuncType(input.dtype)

    cdef shape_type_c axis_shape = utils._object_to_tuple(axis)

    cdef shape_type_c result_shape = utils.get_reduction_output_shape(input_shape, axis, keepdims)
    cdef DPNPFuncType result_c_type = utils.get_output_c_type(DPNP_FN_SUM, input_c_type, out, dtype)

    """ select kernel """
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SUM, input_c_type, result_c_type)

    """ Create result array """
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, result_c_type, out)

    """ Call FPTR interface function """
    cdef dpnp_reduction_c_t func = <dpnp_reduction_c_t > kernel_data.ptr
    func(result.get_data(), input.get_data(), input_shape.data(), input_shape.size(), axis_shape.data(), axis_shape.size(), NULL, NULL)

    return result


cpdef utils.dpnp_descriptor dpnp_trapz(utils.dpnp_descriptor y1, utils.dpnp_descriptor x1, double dx):

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(y1.dtype)
    cdef DPNPFuncType param2_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_TRAPZ, param1_type, param2_type)

    # ceate result array with type given by FPTR data
    cdef shape_type_c result_shape = (1,)
    cdef utils.dpnp_descriptor result = utils.create_output_descriptor(result_shape, kernel_data.return_type, None)

    cdef ftpr_custom_trapz_2in_1out_with_2size_t func = <ftpr_custom_trapz_2in_1out_with_2size_t > kernel_data.ptr
    func(y1.get_data(), x1.get_data(), result.get_data(), dx, y1.size, x1.size)

    return result


cpdef utils.dpnp_descriptor dpnp_trunc(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_TRUNC, x1, dtype=None, out=out, where=True, func_name='trunc')
