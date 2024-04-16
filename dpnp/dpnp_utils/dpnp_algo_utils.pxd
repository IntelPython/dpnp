# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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

from libcpp cimport bool as cpp_bool

from dpnp.dpnp_algo cimport shape_type_c
from dpnp.dpnp_algo.dpnp_algo cimport DPNPFuncData, DPNPFuncName, DPNPFuncType


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

cpdef shape_type_c normalize_axis(object axis, size_t shape_size)
"""
Conversion of the transformation shape axis [-1, 0, 1] into [2, 0, 1] where numbers are `id`s of array shape axis
"""

cpdef long _get_linear_index(key, tuple shape, int ndim)
"""
Compute linear index of an element in memory from array indices
"""

cpdef tuple get_axis_offsets(shape)
"""
Compute axis offsets in the linear array memory
"""

cdef class dpnp_descriptor:
    """array DPNP descriptor"""

    cdef public:  # TODO remove "public" as python accessible attribute
        object origin_pyobj
        dpnp_descriptor origin_desc
        dict descriptor
        Py_ssize_t dpnp_descriptor_data_size
        cpp_bool dpnp_descriptor_is_scalar

    cdef void * get_data(self)
    cdef cpp_bool match_ctype(self, DPNPFuncType ctype)


cdef shape_type_c get_common_shape(shape_type_c input1_shape, shape_type_c input2_shape) except *
"""
Calculate common shape from input shapes
"""

cdef dpnp_descriptor create_output_descriptor(shape_type_c output_shape,
                                              DPNPFuncType c_type,
                                              dpnp_descriptor requested_out,
                                              device=*,
                                              usm_type=*,
                                              sycl_queue=*)
"""
Create output dpnp_descriptor based on shape, type and 'out' parameters
"""

cdef shape_type_c strides_to_vector(object strides, object shape) except *
"""
Get or calculate srtides based on shape.
"""

cdef tuple get_common_usm_allocation(dpnp_descriptor x1, dpnp_descriptor x2)
"""
Get common USM allocation in the form of (sycl_device, usm_type, sycl_queue)
"""

cdef (DPNPFuncType, void *) get_ret_type_and_func(DPNPFuncData kernel_data,
                                                  cpp_bool has_aspect_fp64)
"""
Get the corresponding return type and function pointer based on the
capability of the allocated result array device.
"""
