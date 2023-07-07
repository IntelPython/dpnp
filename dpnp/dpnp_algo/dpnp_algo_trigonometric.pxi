# cython: language_level=3
# cython: linetrace=True
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    'dpnp_arccos',
    'dpnp_arccosh',
    'dpnp_arcsin',
    'dpnp_arcsinh',
    'dpnp_arctan',
    'dpnp_arctanh',
    'dpnp_cbrt',
    'dpnp_cosh',
    'dpnp_degrees',
    'dpnp_exp',
    'dpnp_exp2',
    'dpnp_expm1',
    'dpnp_log10',
    'dpnp_log1p',
    'dpnp_log2',
    'dpnp_radians',
    'dpnp_recip',
    'dpnp_sinh',
    'dpnp_sqrt',
    'dpnp_square',
    'dpnp_tan',
    'dpnp_tanh',
    'dpnp_unwrap'
]


cpdef utils.dpnp_descriptor dpnp_arccos(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_ARCCOS_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_arccosh(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_ARCCOSH_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_arcsin(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_ARCSIN_EXT, x1, dtype=None, out=out, where=True, func_name='arcsin')


cpdef utils.dpnp_descriptor dpnp_arcsinh(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_ARCSINH_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_arctan(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_ARCTAN_EXT, x1, dtype=None, out=out, where=True, func_name='arctan')


cpdef utils.dpnp_descriptor dpnp_arctanh(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_ARCTANH_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_cbrt(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_CBRT_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_cosh(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_COSH_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_degrees(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_DEGREES_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_exp(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_EXP_EXT, x1, dtype=None, out=out, where=True, func_name='exp')


cpdef utils.dpnp_descriptor dpnp_exp2(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_EXP2_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_expm1(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_EXPM1_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_log10(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_LOG10_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_log1p(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_LOG1P_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_log2(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_LOG2_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_recip(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_RECIP_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_radians(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_RADIANS_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_sinh(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_SINH_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_sqrt(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_SQRT_EXT, x1, dtype=None, out=out, where=True, func_name='sqrt')


cpdef utils.dpnp_descriptor dpnp_square(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_SQUARE_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_tan(utils.dpnp_descriptor x1, utils.dpnp_descriptor out):
    return call_fptr_1in_1out_strides(DPNP_FN_TAN_EXT, x1, dtype=None, out=out, where=True, func_name='tan')


cpdef utils.dpnp_descriptor dpnp_tanh(utils.dpnp_descriptor x1):
    return call_fptr_1in_1out_strides(DPNP_FN_TANH_EXT, x1)


cpdef utils.dpnp_descriptor dpnp_unwrap(utils.dpnp_descriptor array1):

    result_type = dpnp.float64

    if array1.dtype == dpnp.float32:
        result_type = dpnp.float32

    array1_obj = array1.get_array()

    cdef utils.dpnp_descriptor result = utils_py.create_output_descriptor_py(array1.shape,
                                                                             result_type,
                                                                             None,
                                                                             device=array1_obj.sycl_device,
                                                                             usm_type=array1_obj.usm_type,
                                                                             sycl_queue=array1_obj.sycl_queue)

    for i in range(result.size):
        val, = numpy.unwrap([array1.get_pyobj()[i]])
        result.get_pyobj()[i] = val

    return result
