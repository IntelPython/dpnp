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
import dpnp
from dpnp.dparray import dparray

import numpy


__all__ = [
    'gen_array_1d',
    'gen_array_2d'
]


def gen_ndarray(size, dtype=numpy.float64, low=None, high=None, seed=None):
    """
    Generate ndarray of random numbers of specified size and type.

    Parameters
    ----------
    size : int
        size of output array
    dtype : dtype
        data type of output array
    low : int
        lowest integers to be generated
    high : int
        highest integers to be generated
    seed : int
        random seed

    Returns
    -------
    ndarray
        generated ndarray
    """
    if seed is not None:
        numpy.random.seed(seed)

    if dtype in (numpy.float32, numpy.float64):
        return numpy.random.ranf(size)

    if dtype in (numpy.int32, numpy.int64):
        low = low or numpy.iinfo(dtype).min
        high = high or numpy.iinfo(dtype).max

        return numpy.random.randint(low, high, size=size, dtype=dtype)

    raise NotImplementedError(f'Generator of ndarray of type {dtype.__name__} not found.')


def gen_dparray(size, dtype=numpy.float64, low=None, high=None, seed=None):
    """
    Generate dparray of random numbers of specified size and type.

    Parameters
    ----------
    size : int
        size of output array
    dtype : dtype
        data type of output array
    low : int
        lowest integers to be generated
    high : int
        highest integers to be generated
    seed : int
        random seed

    Returns
    -------
    dparray
        generated dparray
    """
    ndarr = gen_ndarray(size, dtype=dtype, low=low, high=high, seed=seed)

    dparr = dparray(ndarr.shape, dtype=dtype)

    for i in range(dparr.size):
        dparr._setitem_scalar(i, ndarr.item(i))

    return dparr


def gen_array_1d(lib, size, dtype=numpy.float64, low=None, high=None, seed=None):
    """
    Generate array of random numbers bases on library.

    Parameters
    ----------
    lib : type
        library
    size : int
        size of output array
    dtype : dtype
        data type of output array
    low : int
        lowest integers to be generated
    high : int
        highest integers to be generated
    seed : int
        random seed

    Returns
    -------
    dparray
        generated dparray
    """
    if lib is numpy:
        return gen_ndarray(size, dtype=dtype, low=low, high=high, seed=seed)
    if lib is dpnp:
        return gen_dparray(size, dtype=dtype, low=low, high=high, seed=seed)

    raise NotImplementedError(f'{lib.__name__} array generator not found.')


def gen_array_2d(lib, size_x, size_y, dtype=numpy.float64, low=None, high=None, seed=None):
    return gen_array_1d(lib, size_x * size_y, dtype=dtype, low=low, high=high, seed=seed).reshape((size_x, size_y))
