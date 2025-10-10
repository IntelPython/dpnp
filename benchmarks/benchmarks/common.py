# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

import random

import numpy

# Various pre-crafted datasets/variables for testing
# !!! Must not be changed -- only appended !!!
# while testing numpy we better not rely on numpy to produce random
# sequences
random.seed(1)
# but will seed it nevertheless
numpy.random.seed(1)

nx, ny = 1000, 1000
# reduced squares based on indexes_rand, primarily for testing more
# time-consuming functions (ufunc, linalg, etc)
nxs, nys = 100, 100

# a set of interesting types to test
TYPES1 = [
    "int16",
    "float16",
    "int32",
    "float32",
    "int64",
    "float64",
    "complex64",
    "longfloat",
    "complex128",
]
if "complex256" in numpy.typeDict:
    TYPES1.append("complex256")


def memoize(func):
    result = []

    def wrapper():
        if not result:
            result.append(func())
        return result[0]

    return wrapper


# values which will be used to construct our sample data matrices
# replicate 10 times to speed up initial imports of this helper
# and generate some redundancy


@memoize
def get_values():
    rnd = numpy.random.RandomState(1)
    values = numpy.tile(rnd.uniform(0, 100, size=nx * ny // 10), 10)
    return values


@memoize
def get_squares():
    values = get_values()
    squares = {
        t: numpy.array(values, dtype=getattr(numpy, t)).reshape((nx, ny))
        for t in TYPES1
    }

    # adjust complex ones to have non-degenerated imagery part -- use
    # original data transposed for that
    for t, v in squares.items():
        if t.startswith("complex"):
            v += v.T * 1j
    return squares


@memoize
def get_squares_():
    # smaller squares
    squares_ = {t: s[:nxs, :nys] for t, s in get_squares().items()}
    return squares_


@memoize
def get_vectors():
    # vectors
    vectors = {t: s[0] for t, s in get_squares().items()}
    return vectors


@memoize
def get_indexes():
    indexes = list(range(nx))
    # so we do not have all items
    indexes.pop(5)
    indexes.pop(95)

    indexes = numpy.array(indexes)
    return indexes


@memoize
def get_indexes_rand():
    rnd = random.Random(1)

    indexes_rand = get_indexes().tolist()  # copy
    rnd.shuffle(indexes_rand)  # in-place shuffle
    indexes_rand = numpy.array(indexes_rand)
    return indexes_rand


@memoize
def get_indexes_():
    # smaller versions
    indexes = get_indexes()
    indexes_ = indexes[indexes < nxs]
    return indexes_


@memoize
def get_indexes_rand_():
    indexes_rand = get_indexes_rand()
    indexes_rand_ = indexes_rand[indexes_rand < nxs]
    return indexes_rand_


class Benchmark:
    # warmup_time = 0
    # number = 3 # test repeats for one setup
    # repeat = 1
    # rounds = 1
    pass
