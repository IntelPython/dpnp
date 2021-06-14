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


from dpnp.dpnp_utils cimport *


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


cpdef dparray dpnp_arccos(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_ARCCOS, x1, x1.shape)


cpdef dparray dpnp_arccosh(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_ARCCOSH, x1, x1.shape)


cpdef dparray dpnp_arcsin(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_ARCSIN, x1, x1.shape)


cpdef dparray dpnp_arcsinh(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_ARCSINH, x1, x1.shape)


cpdef dparray dpnp_arctan(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_ARCTAN, x1, x1.shape)


cpdef dparray dpnp_arctanh(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_ARCTANH, x1, x1.shape)


cpdef dparray dpnp_cbrt(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_CBRT, x1, x1.shape)


cpdef dparray dpnp_cos(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_COS, x1, x1.shape)


cpdef dparray dpnp_cosh(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_COSH, x1, x1.shape)


cpdef dparray dpnp_degrees(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_DEGREES, x1, x1.shape)


cpdef dparray dpnp_exp(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_EXP, x1, x1.shape)


cpdef dparray dpnp_exp2(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_EXP2, x1, x1.shape)


cpdef dparray dpnp_expm1(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_EXPM1, x1, x1.shape)


cpdef dparray dpnp_log(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_LOG, x1, x1.shape)


cpdef dparray dpnp_log10(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_LOG10, x1, x1.shape)


cpdef dparray dpnp_log1p(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_LOG1P, x1, x1.shape)


cpdef dparray dpnp_log2(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_LOG2, x1, x1.shape)


cpdef dparray dpnp_recip(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_RECIP, x1, x1.shape)


cpdef dparray dpnp_radians(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_RADIANS, x1, x1.shape)


cpdef dparray dpnp_sin(dparray x1, dparray out=None):

    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(x1.dtype)

    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_SIN, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype(< size_t > kernel_data.return_type)

    shape_result = x1.shape

    cdef dparray result

    if out is not None:
        if out.dtype != result_type:
            checker_throw_value_error('sin', 'out.dtype', out.dtype, result_type)
        if out.shape != shape_result:
            checker_throw_value_error('sin', 'out.shape', out.shape, shape_result)
        result = out
    else:
        result = dparray(shape_result, dtype=result_type)

    cdef fptr_1in_1out_t func = <fptr_1in_1out_t > kernel_data.ptr

    func(x1.get_data(), result.get_data(), x1.size)

    return result


cpdef dparray dpnp_sinh(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_SINH, x1, x1.shape)


cpdef dparray dpnp_sqrt(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_SQRT, x1, x1.shape)


cpdef dparray dpnp_square(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_SQUARE, x1, x1.shape)


cpdef dparray dpnp_tan(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_TAN, x1, x1.shape)


cpdef dparray dpnp_tanh(dparray x1):
    return call_fptr_1in_1out(DPNP_FN_TANH, x1, x1.shape)


cpdef dparray dpnp_unwrap(dparray array1):

    call_type = array1.dtype
    cdef dparray result

    if call_type == numpy.float32:
        result = dparray(array1.shape, dtype=call_type)
    else:
        result = dparray(array1.shape)

    for i in range(result.size):
        val, = numpy.unwrap([array1[i]])
        result[i] = val

    return result
