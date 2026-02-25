# cython: language_level=3
# cython: linetrace=True
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

cimport dpctl as c_dpctl

cimport dpnp.dpnp_utils as utils
from dpnp.dpnp_algo cimport shape_elem_type, shape_type_c

__all__ = [
]


include "dpnp_algo_sorting.pxi"


"""
Internal functions
"""
cdef DPNPFuncType dpnp_dtype_to_DPNPFuncType(dtype):
    dt_c = dpnp.dtype(dtype).char
    kind = dpnp.dtype(dtype).kind
    if isinstance(kind, int):
        kind = chr(kind)
    itemsize = dpnp.dtype(dtype).itemsize

    if dt_c == 'd':
        return DPNP_FT_DOUBLE
    elif dt_c == 'f':
        return DPNP_FT_FLOAT
    elif kind == 'i':
        if itemsize == 8:
            return DPNP_FT_LONG
        elif itemsize == 4:
            return DPNP_FT_INT
        else:
            utils.checker_throw_type_error("dpnp_dtype_to_DPNPFuncType", dtype)
    elif dt_c == 'F':
        return DPNP_FT_CMPLX64
    elif dt_c == 'D':
        return DPNP_FT_CMPLX128
    elif dt_c == '?':
        return DPNP_FT_BOOL
    else:
        utils.checker_throw_type_error("dpnp_dtype_to_DPNPFuncType", dtype)


cdef dpnp_DPNPFuncType_to_dtype(size_t type):
    """
    Type 'size_t' used instead 'DPNPFuncType' because Cython has lack of Enum support (0.29)
    TODO needs to use DPNPFuncType here
    """
    if type == <size_t > DPNP_FT_DOUBLE:
        return dpnp.float64
    elif type == <size_t > DPNP_FT_FLOAT:
        return dpnp.float32
    elif type == <size_t > DPNP_FT_LONG:
        return dpnp.int64
    elif type == <size_t > DPNP_FT_INT:
        return dpnp.int32
    elif type == <size_t > DPNP_FT_CMPLX64:
        return dpnp.complex64
    elif type == <size_t > DPNP_FT_CMPLX128:
        return dpnp.complex128
    elif type == <size_t > DPNP_FT_BOOL:
        return dpnp.bool
    else:
        utils.checker_throw_type_error("dpnp_DPNPFuncType_to_dtype", type)
