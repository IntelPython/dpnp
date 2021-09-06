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

import numpy
import numpy.lib.stride_tricks as np_st

import dpnp.config as config


assert_allclose_orig = numpy.testing.assert_allclose
assert_array_equal_orig = numpy.testing.assert_array_equal
assert_equal_orig = numpy.testing.assert_equal


if config.__DPNP_OUTPUT_DPCTL__:
    try:
        """
        Detect DPCtl availability to use data container
        """
        import dpctl.tensor as dpt

    except ImportError:
        """
        No DPCtl data container available
        """
        config.__DPNP_OUTPUT_DPCTL__ = 0


def _asnumpy(ary):
    if config.__DPNP_OUTPUT_DPCTL__ and isinstance(ary, dpt.usm_ndarray):
        return np_st.as_strided(ary.usm_data.copy_to_host().view(ary.dtype), shape=ary.shape)

    return numpy.asnumpy(ary)


def _to_numpy(item):
    if config.__DPNP_OUTPUT_DPCTL__ and isinstance(item, dpt.usm_ndarray):
        item = _asnumpy(item)
    elif isinstance(item, tuple):
        item = tuple(_to_numpy(i) for i in item)
    elif isinstance(item, list):
        item = [_to_numpy(i) for i in item]

    return item

def _assert(assert_func, result, expected, *args, **kwargs):
    if config.__DPNP_OUTPUT_DPCTL__:
        result = _to_numpy(result)
        expected = _to_numpy(expected)

    assert_func(result, expected, *args, **kwargs)


def assert_allclose(result, expected, *args, **kwargs):
    _assert(assert_allclose_orig, result, expected, *args, **kwargs)


def assert_array_equal(result, expected, *args, **kwargs):
    _assert(assert_array_equal_orig, result, expected, *args, **kwargs)


def assert_equal(result, expected, *args, **kwargs):
    _assert(assert_equal_orig, result, expected, *args, **kwargs)
