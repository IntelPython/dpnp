
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

import pytest

import dpnp
import numpy as np

ROUNDS = 30
ITERATIONS = 4

NNUMBERS = 2**26


@pytest.mark.parametrize('function', [dpnp.random.beta, np.random.beta],
                         ids=['dpnp', 'numpy'])
def test_beta(benchmark, function):
    result = benchmark.pedantic(target=function, args=(4.0, 5.0, NNUMBERS,),
                                rounds=ROUNDS, iterations=ITERATIONS)


@pytest.mark.parametrize('function', [dpnp.random.exponential, np.random.exponential],
                         ids=['dpnp', 'numpy'])
def test_exponential(benchmark, function):
    result = benchmark.pedantic(target=function, args=(4.0, NNUMBERS,),
                                rounds=ROUNDS, iterations=ITERATIONS)


@pytest.mark.parametrize('function', [dpnp.random.gamma, np.random.gamma],
                         ids=['dpnp', 'numpy'])
def test_gamma(benchmark, function):
    result = benchmark.pedantic(target=function, args=(2.0, 4.0, NNUMBERS,),
                                rounds=ROUNDS, iterations=ITERATIONS)


@pytest.mark.parametrize('function', [dpnp.random.normal, np.random.normal],
                         ids=['dpnp', 'numpy'])
def test_normal(benchmark, function):
    result = benchmark.pedantic(target=function, args=(0.0, 1.0, NNUMBERS,),
                                rounds=ROUNDS, iterations=ITERATIONS)


@pytest.mark.parametrize('function', [dpnp.random.uniform, np.random.uniform],
                         ids=['dpnp', 'numpy'])
def test_uniform(benchmark, function):
    result = benchmark.pedantic(target=function, args=(0.0, 1.0, NNUMBERS,),
                                rounds=ROUNDS, iterations=ITERATIONS)
