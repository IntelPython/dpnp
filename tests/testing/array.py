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


# https://github.com/IntelPython/dpctl/blob/3fe25706995e76255a931d8ed87786da69db685c/dpctl/tests/test_usm_ndarray_ctor.py#L157  # noqa
def _to_numpy(usm_ary):
    if type(usm_ary) is dpt.usm_ndarray:
        usm_buf = usm_ary.usm_data
        s = usm_buf.nbytes
        host_buf = usm_buf.copy_to_host().view(usm_ary.dtype)
        usm_ary_itemsize = usm_ary.itemsize
        R_offset = (
            usm_ary.__sycl_usm_array_interface__["offset"] * usm_ary_itemsize
        )
        R = numpy.ndarray((s,), dtype="u1", buffer=host_buf)
        R = R[R_offset:].view(usm_ary.dtype)
        R_strides = (usm_ary_itemsize * si for si in usm_ary.strides)
        return np_st.as_strided(R, shape=usm_ary.shape, strides=R_strides)
    else:
        raise ValueError(
            "Expected dpctl.tensor.usm_ndarray, got {}".format(type(usm_ary))
        )


def assert_allclose(result, expected, *args, **kwargs):
    if config.__DPNP_OUTPUT_DPCTL__:
        if isinstance(result, dpt.usm_ndarray):
            result = _to_numpy(result)
        if isinstance(expected, dpt.usm_ndarray):
            expected = _to_numpy(expected)

    assert_allclose_orig(result, expected, *args, **kwargs)


def assert_array_equal(result, expected, *args, **kwargs):
    if config.__DPNP_OUTPUT_DPCTL__:
        if isinstance(result, dpt.usm_ndarray):
            result = _to_numpy(result)
        if isinstance(expected, dpt.usm_ndarray):
            expected = _to_numpy(expected)

    assert_array_equal_orig(result, expected, *args, **kwargs)
