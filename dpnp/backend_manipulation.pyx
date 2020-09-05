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

"""Module Backend (Array manipulation routines)

This module contains interface functions between C backend layer
and the rest of the library

"""


from dpnp.dpnp_utils cimport checker_throw_type_error, normalize_axis


__all__ += [
    "dpnp_repeat",
    "dpnp_transpose"
]


cpdef dparray dpnp_repeat(dparray array1, repeats, axes=None):
    cdef long new_size = array1.size * repeats
    cdef dparray result = dparray((new_size, ), dtype=array1.dtype)

    for idx2 in range(array1.size):
        for idx1 in range(repeats):
            result[(idx2 * repeats) + idx1] = array1[idx2]

    return result


cpdef dparray dpnp_transpose(dparray array1, axes=None):
    call_type = array1.dtype
    cdef size_t data_size = array1.size
    cdef dparray_shape_type input_shape = array1.shape
    cdef size_t input_shape_size = array1.ndim
    cdef dparray_shape_type result_shape = dparray_shape_type(input_shape_size, 1)

    cdef dparray_shape_type permute_axes
    if axes is None:
        """
        template to do transpose a tensor
        input_shape=[2, 3, 4]
        permute_axes=[2, 1, 0]
        after application `permute_axes` to `input_shape` result:
        result_shape=[4, 3, 2]

        'do nothing' axes variable is `permute_axes=[0, 1, 2]`

        test: pytest tests/third_party/cupy/manipulation_tests/test_transpose.py::TestTranspose::test_external_transpose_all
        """
        permute_axes = list(reversed([i for i in range(input_shape_size)]))
    else:
        permute_axes = utils.normalize_axis(axes, input_shape_size)

    for i in range(input_shape_size):
        """ construct output shape """
        result_shape[i] = input_shape[permute_axes[i]]

    cdef dparray result = dparray(result_shape, dtype=call_type)

    if call_type == numpy.float64:
        custom_elemwise_transpose_c[double](
            array1.get_data(),
            input_shape,
            result_shape,
            permute_axes,
            result.get_data(),
            data_size)
    elif call_type == numpy.float32:
        custom_elemwise_transpose_c[float](
            array1.get_data(),
            input_shape,
            result_shape,
            permute_axes,
            result.get_data(),
            data_size)
    elif call_type == numpy.int64:
        custom_elemwise_transpose_c[long](
            array1.get_data(),
            input_shape,
            result_shape,
            permute_axes,
            result.get_data(),
            data_size)
    elif call_type == numpy.int32:
        custom_elemwise_transpose_c[int](
            array1.get_data(),
            input_shape,
            result_shape,
            permute_axes,
            result.get_data(),
            data_size)
    else:
        checker_throw_type_error("dpnp_transpose", call_type)

    return result
