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
Interface of the statistics function of the Intel NumPy

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""


import numpy

from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import checker_throw_value_error, use_origin_backend

__all__ = [
    'mean'
]


def mean(a, axis=None):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
        .. versionadded:: 1.7.0
        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    """

    if use_origin_backend():
        return numpy.mean(a, axis=axis)

    # TODO This block needs to be redesigned
    # Test pytest tests/third_party/cupy/statistics_tests/test_mean.py::TestMeanVar::test_mean_all
    # failed if uncomment the code
    # if numpy.shape(a) == ():
    #     if axis is None:
    #         return a
    #     else:
    #         raise AttributeError("Object has no attribute 'mean'")

    if not isinstance(axis, int) and axis is not None:
        raise TypeError("Parameter axis can be only int!")

    if isinstance(a, dparray):
        result = dpnp_mean(a, axis=axis)

        # scalar returned
        if result.shape == (1,):
            return result.dtype.type(result[0])

        return result

    result_numpy = numpy.mean(a, axis=axis)
    result = result_numpy
    if isinstance(result, numpy.ndarray):
        result = dparray(result_numpy.shape, dtype=result_numpy.dtype)
        for i in range(result.size):
            result._setitem_scalar(i, result_numpy.item(i))

    return result
