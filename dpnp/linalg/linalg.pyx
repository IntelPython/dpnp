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

from dpnp.dpnp_utils cimport checker_throw_type_error
from libcpp.vector cimport vector
from dpnp.backend cimport *
from dpnp.dparray cimport dparray, dparray_shape_type
import numpy
cimport numpy


__all__ = [
    "dpnp_eig",
]


cpdef dparray dpnp_eig(dparray in_array1):
    cdef vector[Py_ssize_t] shape1 = in_array1.shape

    call_type = in_array1.dtype

    cdef size_t size1 = 0
    if not shape1.empty():
        size1 = shape1.front()

    cdef dparray res_val = dparray((size1,), dtype=call_type)

    # this array is used as input for MKL and will be overwritten with eigen vectors
    cdef dparray res_vec = dparray(shape1, dtype=call_type)
    for i in range(in_array1.size):
        res_vec[i] = in_array1[i]

    if call_type == numpy.float64:
        mkl_lapack_syevd_c[double](res_vec.get_data(), res_val.get_data(), size1)
    elif call_type == numpy.float32:
        mkl_lapack_syevd_c[float](res_vec.get_data(), res_val.get_data(), size1)
    else:
        checker_throw_type_error("dpnp_eig", call_type)

    return res_val, res_vec
