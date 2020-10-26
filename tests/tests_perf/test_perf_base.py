#!/usr/bin/env python
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

import os
import statistics

import dpnp
import numpy

from tests.tests_perf.test_perf_utils import get_exec_times, is_true, TestResults


class TestBase:
    csv_result_path = "perf_results.csv"
    float_format = "%.3e"
    seed = 777

    @classmethod
    def setup_class(cls):
        cls.test_results = TestResults()

    @classmethod
    def teardown_class(cls):
        cls.test_results.print(float_format=cls.float_format)
        cls.test_results.to_csv(cls.csv_result_path, float_format=cls.float_format)

    def _test_func(self, name, lib, dtype, size, *args, repeat=5, number=1000000):
        """
        Test performance of specified function.

        Parameters
        ----------
        name : str
            name of the function
        lib : type
            library
        dtype : dtype
            data type of the input array
        size : int
            size of the input array
        args : tuple
            parameters of the fucntion
        repeat : int
            number of measurements
        number : int
            number of the function calls within a single measurement
        """
        exec_times = get_exec_times(getattr(lib, name), *args, repeat=repeat, number=number)
        result = {
            'min': [min(exec_times)],
            'max': [max(exec_times)],
            'median': [statistics.median(exec_times)],
        }
        self.test_results.add(name, lib, dtype, size, **result)
