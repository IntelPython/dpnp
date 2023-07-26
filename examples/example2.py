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

"""Example 2.

This example shows usage of diffrent input data types
with same third party library call

Also, it produces performance comparison between a same third party function call
over diffrent types of input data

"""


import time

import numpy

import dpnp


def run_third_party_function(xp, input, repetition):
    times = []
    for _ in range(repetition):
        start_time = time.time()
        result = xp.sin(input)
        end_time = time.time()
        times.append(end_time - start_time)

    execution_time = numpy.median(times)
    return execution_time, result.item(5)


if __name__ == "__main__":
    test_repetition = 5
    for xp in [numpy, dpnp]:
        type_name = xp.__name__
        print(
            f"...Test data type is {type_name}, each test repetitions {test_repetition}"
        )

        for size in range(20, 25):
            size = 2**size
            input_data = xp.arange(size)
            result_time, result = run_third_party_function(
                xp, input_data, test_repetition
            )

            print(
                f"type:{type_name}:N:{size:6}:Time:{result_time:.3e}:result:{result}"
            )
