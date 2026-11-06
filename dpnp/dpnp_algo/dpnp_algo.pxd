# cython: language_level=3
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


# need this namespace for Enum import
cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncName":
    cdef enum DPNPFuncName "DPNPFuncName":
        DPNP_FN_PARTITION_EXT

# need this namespace for Enum import
cdef extern from "dpnp_iface_fptr.hpp" namespace "DPNPFuncType":
    cdef enum DPNPFuncType "DPNPFuncType":
        DPNP_FT_NONE
        DPNP_FT_INT
        DPNP_FT_LONG
        DPNP_FT_FLOAT
        DPNP_FT_DOUBLE
        DPNP_FT_CMPLX64
        DPNP_FT_CMPLX128
        DPNP_FT_BOOL

cdef extern from "dpnp_iface_fptr.hpp":
    struct DPNPFuncData:
        DPNPFuncType return_type
        void * ptr
        DPNPFuncType return_type_no_fp64
        void *ptr_no_fp64

    DPNPFuncData get_dpnp_function_ptr(
        DPNPFuncName name,
        DPNPFuncType first_type,
        DPNPFuncType second_type,
    ) except +


# Internal functions
cdef DPNPFuncType dpnp_dtype_to_DPNPFuncType(dtype)
cdef dpnp_DPNPFuncType_to_dtype(size_t type)
