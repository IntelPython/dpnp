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

"""Module Backend (Trigonometric part)

This module contains interface functions between C backend layer
and the rest of the library

"""


from dpnp.dpnp_utils cimport checker_throw_type_error


__all__ += [
    'dpnp_arccos',
    'dpnp_arccosh',
    'dpnp_arcsin',
    'dpnp_arcsinh',
    'dpnp_arctan',
    'dpnp_arctanh',
    'dpnp_cbrt',
    'dpnp_cos',
    'dpnp_cosh',
    'dpnp_degrees',
    'dpnp_exp',
    'dpnp_exp2',
    'dpnp_expm1',
    'dpnp_log',
    'dpnp_log10',
    'dpnp_log1p',
    'dpnp_log2',
    'dpnp_radians',
    'dpnp_recip',
    'dpnp_sin',
    'dpnp_sinh',
    'dpnp_sqrt',
    'dpnp_square',
    'dpnp_tan',
    'dpnp_tanh',
    'dpnp_unwrap'
]


cpdef dparray dpnp_arccos(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_acos_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_acos_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_acos_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_acos_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_arccos", call_type)

    return result


cpdef dparray dpnp_arccosh(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_acosh_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_acosh_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_acosh_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_acosh_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_arccosh", call_type)

    return result


cpdef dparray dpnp_arcsin(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_asin_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_asin_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_asin_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_asin_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_arcsin", call_type)

    return result


cpdef dparray dpnp_arcsinh(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_asinh_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_asinh_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_asinh_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_asinh_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_arcsinh", call_type)

    return result


cpdef dparray dpnp_arctan(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_atan_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_atan_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_atan_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_atan_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_arctan", call_type)

    return result


cpdef dparray dpnp_arctanh(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_atanh_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_atanh_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_atanh_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_atanh_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_arctanh", call_type)

    return result


cpdef dparray dpnp_cbrt(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_cbrt_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_cbrt_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_cbrt_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_cbrt_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_cbrt", call_type)

    return result


cpdef dparray dpnp_cos(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_cos_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_cos_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_cos_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_cos_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_cos", call_type)

    return result


cpdef dparray dpnp_cosh(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_cosh_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_cosh_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_cosh_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_cosh_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_cosh", call_type)

    return result


cpdef dparray dpnp_degrees(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_degrees_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_degrees_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_degrees_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_degrees_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_degrees", call_type)

    return result


cpdef dparray dpnp_exp(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_exp_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_exp_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_exp_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_exp_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_exp", call_type)

    return result


cpdef dparray dpnp_exp2(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_exp2_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_exp2_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_exp2_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_exp2_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_exp2", call_type)

    return result


cpdef dparray dpnp_expm1(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_expm1_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_expm1_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_expm1_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_expm1_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_expm1", call_type)

    return result


cpdef dparray dpnp_log(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_log_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_log_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_log_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_log_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_log", call_type)

    return result


cpdef dparray dpnp_log10(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_log10_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_log10_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_log10_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_log10_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_log10", call_type)

    return result


cpdef dparray dpnp_log1p(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_log1p_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_log1p_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_log1p_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_log1p_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_log1p", call_type)

    return result


cpdef dparray dpnp_log2(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_log2_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_log2_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_log2_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_log2_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_log2", call_type)

    return result


cpdef dparray dpnp_recip(dparray array1):
    call_type = array1.dtype
    cdef size_t size = array1.size
    cdef dparray result = dparray(array1.shape, dtype=call_type)

    if call_type == numpy.float64:
        custom_elemwise_recip_c[double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_recip_c[float](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_recip", call_type)

    return result


cpdef dparray dpnp_radians(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_radians_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_radians_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_radians_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_radians_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_radians", call_type)

    return result


cpdef dparray dpnp_sin(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_sin_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_sin_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_sin_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_sin_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_sin", call_type)

    return result


cpdef dparray dpnp_sinh(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_sinh_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_sinh_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_sinh_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_sinh_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_sinh", call_type)

    return result


cpdef dparray dpnp_sqrt(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_sqrt_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_sqrt_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_sqrt_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_sqrt_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_sqrt", call_type)

    return result


cpdef dparray dpnp_square(dparray array1):
    call_type = array1.dtype
    cdef dparray result = dparray(array1.shape, dtype=call_type)
    cdef size_t size = result.size

    if call_type == numpy.float64:
        custom_elemwise_square_c[double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_square_c[float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_square_c[long](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_square_c[int](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_square", call_type)

    return result


cpdef dparray dpnp_tan(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_tan_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_tan_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_tan_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_tan_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_tan", call_type)

    return result


cpdef dparray dpnp_tanh(dparray array1):
    cdef dparray result
    cdef size_t size = array1.size
    call_type = array1.dtype

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    if call_type == numpy.float64:
        custom_elemwise_tanh_c[double, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.float32:
        custom_elemwise_tanh_c[float, float](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int64:
        custom_elemwise_tanh_c[long, double](array1.get_data(), result.get_data(), size)
    elif call_type == numpy.int32:
        custom_elemwise_tanh_c[int, double](array1.get_data(), result.get_data(), size)
    else:
        checker_throw_type_error("dpnp_tanh", call_type)

    return result


cpdef dparray dpnp_unwrap(dparray array1):

    call_type = array1.dtype
    cdef dparray result

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    for i in range(result.size):
        result[i] = numpy.unwrap(array1[i])

    return result
