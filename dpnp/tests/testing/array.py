# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2025, Intel Corporation
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

import dpctl
import numpy

from dpnp.dpnp_utils import convert_item

assert_allclose_orig = numpy.testing.assert_allclose
assert_almost_equal_orig = numpy.testing.assert_almost_equal
assert_array_almost_equal_orig = numpy.testing.assert_array_almost_equal
assert_array_equal_orig = numpy.testing.assert_array_equal
assert_equal_orig = numpy.testing.assert_equal


def _assert(assert_func, result, expected, *args, **kwargs):
    result = convert_item(result)
    expected = convert_item(expected)

    # original versions of assert_equal, assert_array_equal, and assert_allclose
    # (since NumPy 2.0) have `strict` parameter. Added here for
    # assert_almost_equal, assert_array_almost_equal
    flag = assert_func in [
        assert_almost_equal_orig,
        assert_array_almost_equal_orig,
    ]
    # For numpy < 2.0, some tests will fail for dtype mismatch
    dev = dpctl.select_default_device()
    if (
        numpy.lib.NumpyVersion(numpy.__version__) >= "2.0.0"
        and dev.has_aspect_fp64
    ):
        strict = kwargs.setdefault("strict", True)
        if flag:
            if strict:
                if hasattr(expected, "dtype"):
                    assert (
                        result.dtype == expected.dtype
                    ), f"{result.dtype} != {expected.dtype}"
                    assert (
                        result.shape == expected.shape
                    ), f"{result.shape} != {expected.shape}"
                else:
                    # numpy output is scalar, then dpnp is 0-D array
                    assert result.shape == (), f"{result.shape} != ()"
            kwargs.pop("strict")
    else:
        kwargs.pop("strict", None)

    assert_func(result, expected, *args, **kwargs)


def assert_allclose(result, expected, *args, **kwargs):
    _assert(assert_allclose_orig, result, expected, *args, **kwargs)


def assert_almost_equal(result, expected, *args, **kwargs):
    _assert(assert_almost_equal_orig, result, expected, *args, **kwargs)


def assert_array_almost_equal(result, expected, *args, **kwargs):
    _assert(assert_array_almost_equal_orig, result, expected, *args, **kwargs)


def assert_array_equal(result, expected, *args, **kwargs):
    _assert(assert_array_equal_orig, result, expected, *args, **kwargs)


def assert_equal(result, expected, *args, **kwargs):
    _assert(assert_equal_orig, result, expected, *args, **kwargs)
