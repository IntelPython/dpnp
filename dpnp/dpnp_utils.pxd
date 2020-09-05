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

from dpnp.dparray cimport dparray, dparray_shape_type
from libcpp cimport bool as cpp_bool
from libcpp.vector cimport vector


cpdef checker_throw_runtime_error(function_name, message)
""" Throw exception RuntimeError with 'message'

"""

cpdef checker_throw_value_error(function_name, param_name, param, expected)
""" Throw exception ValueError if 'param' is not 'expected'

"""


cpdef checker_throw_axis_error(function_name, param_name, param, expected)
""" Throw exception AxisError if 'param' is not 'expected'

"""


cpdef checker_throw_type_error(function_name, given_type)
""" Throw exception TypeError if 'given_type' type is not supported

"""


cpdef checker_throw_index_error(function_name, index, size)
""" Throw exception IndexError if 'index' is out of bounds

"""


cpdef cpp_bool use_origin_backend(input1=*, size_t compute_size=*)
"""
This function needs to redirect particular computation cases to original backend

Parameters:
    input1: One of the input parameter of the API function
    compute_size: Some amount of total compute size of the task
Return:
    True - computations are better to be executed on original backend
    False - it is better to use this SW to compute
"""


cpdef tuple _object_to_tuple(object obj)
cdef int _normalize_order(order, cpp_bool allow_k=*) except? 0

cpdef dparray_shape_type normalize_axis(dparray_shape_type axis, size_t shape_size)
"""
Conversion of the transformation shape axis [-1, 0, 1] into [2, 0, 1] where numbers are `id`s of array shape axis
"""

cdef tuple get_shape_dtype(object input_obj)
"""
input_obj: Complex object with lists, scalars and dparrays

Returns a tuple of:
1. concatenated shape, empty `dparray_shape_type` if unsuccessful.
2. dtype
"""

cdef long copy_values_to_dparray(dparray dst, input_obj, size_t dst_idx=*) except -1
"""
Copy values to `dst` by iterating element by element in `input_obj`
"""

cpdef long _get_linear_index(key, tuple shape, int ndim)
"""
Compute linear index of an element in memory from array indices
"""
