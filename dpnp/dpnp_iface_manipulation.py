# cython: language_level=3
# distutils: language = c++
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

"""
Interface of the Array manipulation routines part of the Intel NumPy

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import collections.abc
import numpy

import dpnp
from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import checker_throw_value_error, use_origin_backend, normalize_axis, checker_throw_axis_error


__all__ = [
    "moveaxis",
    "repeat",
    "swapaxes",
    "transpose"
]


def moveaxis(x1, source, destination):
    """
    Move axes of an array to new positions. Other axes remain in their original order.
    """

    if (use_origin_backend(x1)):
        return numpy.swapaxes(x1, source, destination)

    if (not isinstance(x1, dparray)):
        return numpy.swapaxes(x1, source, destination)

    if not isinstance(source, collections.abc.Sequence):  # assume scalar
        source = (source,)

    if not isinstance(destination, collections.abc.Sequence):  # assume scalar
        destination = (destination,)

    source_norm = normalize_axis(source, x1.ndim)
    destination_norm = normalize_axis(destination, x1.ndim)

    if len(source_norm) != len(destination_norm):
        checker_throw_axis_error(
            "swapaxes",
            "source_norm.size() != destination_norm.size()",
            source_norm,
            destination_norm)

    # 'do nothing' pattern for transpose() with no elements in 'source'
    input_permute = []
    for i in range(x1.ndim):
        if i not in source_norm:
            input_permute.append(i)

    # insert moving axes into proper positions
    for destination_id, source_id in sorted(zip(destination_norm, source_norm)):
        # if destination_id in input_permute:
        # pytest tests/third_party/cupy/manipulation_tests/test_transpose.py::TestTranspose::test_moveaxis_invalid5_3
        #checker_throw_value_error("swapaxes", "source_id exists", source_id, input_permute)
        input_permute.insert(destination_id, source_id)

    return transpose(x1, axes=input_permute)


def repeat(x1, repeats, axis=None):
    """
    Repeat elements of an array.
    """

    is_x1_dparray = isinstance(x1, dparray)

    if (not use_origin_backend(x1)
        and is_x1_dparray
        and (axis is None or axis == 0)
        and (x1.ndim < 2)
        ):

        repeat_val = repeats
        if isinstance(repeats, (tuple, list)):
            if (len(repeats) > 1):
                checker_throw_value_error("repeat", "len(repeats)", len(repeats), 1)

            repeat_val = repeats[0]

        return dpnp_repeat(x1, repeat_val, axis)

    input1 = dpnp.asnumpy(x1) if is_x1_dparray else x1

    # TODO need to put dparray memory into NumPy call
    result_numpy = numpy.repeat(input1, repeats, axis=axis)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result


def swapaxes(x1, axis1, axis2):
    """
    Interchange two axes of an array.
    """

    if (use_origin_backend(x1)):
        return numpy.swapaxes(x1, axis1, axis2)

    if (not isinstance(x1, dparray)):
        return numpy.swapaxes(x1, axis1, axis2)

    if not (axis1 < x1.ndim):
        checker_throw_value_error("swapaxes", "axis1", axis1, x1.ndim - 1)

    if not (axis2 < x1.ndim):
        checker_throw_value_error("swapaxes", "axis2", axis2, x1.ndim - 1)

    # 'do nothing' pattern for transpose()
    input_permute = [i for i in range(x1.ndim)]
    # swap axes
    input_permute[axis1], input_permute[axis2] = input_permute[axis2], input_permute[axis1]

    return transpose(x1, axes=input_permute)


def transpose(x1, axes=None):
    """
    Reverse or permute the axes of an array; returns the modified array.
    """

    if (use_origin_backend(x1)):
        return numpy.transpose(x1, axes=axes)

    if (not isinstance(x1, dparray)):
        return numpy.transpose(x1, axes=axes)

    if (axes is not None):
        if (not any(axes)):
            """
            pytest tests/third_party/cupy/manipulation_tests/test_transpose.py
            """
            axes = None

    return dpnp_transpose(x1, axes=axes)
