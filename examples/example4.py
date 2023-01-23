# cython: language_level=3
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

"""Example 1.

This example shows input and output types of specified function
This is usefull for development

"""


import numpy

"""
Unary functions
"""
for function in [numpy.sqrt, numpy.fabs, numpy.reciprocal, numpy.square, numpy.cbrt, numpy.degrees, numpy.radians]:
    print()
    for test_type in [numpy.float64, numpy.float32, numpy.int64, numpy.int32, numpy.bool_]:
        data = numpy.array([1, 2, 3, 4], dtype=test_type)
        result = function(data)
        print(f"input:{data.dtype.name:10}: outout:{result.dtype.name:10}: name:{function.__name__}")

"""
Two arguments functions
"""
for function in [numpy.equal, numpy.arctan2]:
    print()
    for input1_type in [numpy.float64, numpy.float32, numpy.int64, numpy.int32, numpy.bool_]:
        for input2_type in [numpy.float64, numpy.float32, numpy.int64, numpy.int32, numpy.bool_]:
            data1 = numpy.array([1, 2, 3, 4], dtype=input1_type)
            data2 = numpy.array([11, 21, 31, 41], dtype=input2_type)
            result = function(data1, data2)

            msg = f"input1:{data1.dtype.name:10}: input2:{data2.dtype.name:10}"
            msg += f": output:{result.dtype.name:10}: name:{function}"
            print(msg)
