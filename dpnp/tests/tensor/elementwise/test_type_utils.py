# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

import dpctl

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
import dpctl_ext.tensor._type_utils as tu
import numpy as np
import pytest

from .utils import (
    _all_dtypes,
    _map_to_device_dtype,
)


class MockDevice:
    def __init__(self, fp16: bool, fp64: bool):
        self.has_aspect_fp16 = fp16
        self.has_aspect_fp64 = fp64


@pytest.mark.parametrize("dtype", _all_dtypes)
def test_type_utils_map_to_device_type(dtype):
    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            dev = MockDevice(fp16, fp64)
            dt_in = dpt.dtype(dtype)
            dt_out = _map_to_device_dtype(dt_in, dev)
            assert isinstance(dt_out, dpt.dtype)


def test_type_util_all_data_types():
    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            r = tu._all_data_types(fp16, fp64)
            assert isinstance(r, list)
            # 11: bool + 4 signed + 4 unsigned inegral + float32 + complex64
            assert len(r) == 11 + int(fp16) + 2 * int(fp64)


def test_type_util_can_cast():
    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            for from_ in _all_dtypes:
                for to_ in _all_dtypes:
                    r = tu._can_cast(
                        dpt.dtype(from_), dpt.dtype(to_), fp16, fp64
                    )
                    assert isinstance(r, bool)


def test_type_utils_find_buf_dtype():
    def _denier_fn(dt):
        return False

    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            dev = MockDevice(fp16, fp64)
            arg_dt = dpt.float64
            r = tu._find_buf_dtype(
                arg_dt, _denier_fn, dev, tu._acceptance_fn_default_unary
            )
            assert r == (
                None,
                None,
            )


def test_type_utils_get_device_default_type():
    with pytest.raises(RuntimeError):
        tu._get_device_default_dtype("-", MockDevice(True, True))
    try:
        dev = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    for k in ["b", "i", "u", "f", "c"]:
        dt = tu._get_device_default_dtype(k, dev)
        assert isinstance(dt, dpt.dtype)
        assert dt.kind == k


def test_type_utils_find_buf_dtype2():
    def _denier_fn(dt1, dt2):
        return False

    for fp64 in [
        True,
        False,
    ]:
        for fp16 in [True, False]:
            dev = MockDevice(fp16, fp64)
            arg1_dt = dpt.float64
            arg2_dt = dpt.complex64
            r = tu._find_buf_dtype2(
                arg1_dt,
                arg2_dt,
                _denier_fn,
                dev,
                tu._acceptance_fn_default_binary,
            )
            assert r == (
                None,
                None,
                None,
            )


def test_unary_func_arg_validation():
    with pytest.raises(TypeError):
        dpt.abs([1, 2, 3])
    try:
        a = dpt.arange(8)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    dpt.abs(a, order="invalid")


def test_binary_func_arg_validation():
    with pytest.raises(dpctl.utils.ExecutionPlacementError):
        dpt.add([1, 2, 3], 1)
    try:
        a = dpt.arange(8)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    with pytest.raises(ValueError):
        dpt.add(a, Ellipsis)
    dpt.add(a, a, order="invalid")


def test_all_data_types():
    fp16_fp64_types = {dpt.float16, dpt.float64, dpt.complex128}
    fp64_types = {dpt.float64, dpt.complex128}

    all_dts = tu._all_data_types(True, True)
    assert fp16_fp64_types.issubset(all_dts)

    all_dts = tu._all_data_types(True, False)
    assert dpt.float16 in all_dts
    assert not fp64_types.issubset(all_dts)

    all_dts = tu._all_data_types(False, True)
    assert dpt.float16 not in all_dts
    assert fp64_types.issubset(all_dts)

    all_dts = tu._all_data_types(False, False)
    assert not fp16_fp64_types.issubset(all_dts)


@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("fp64", [True, False])
def test_maximal_inexact_types(fp16, fp64):
    assert not tu._is_maximal_inexact_type(dpt.int32, fp16, fp64)
    assert fp64 == tu._is_maximal_inexact_type(dpt.float64, fp16, fp64)
    assert fp64 == tu._is_maximal_inexact_type(dpt.complex128, fp16, fp64)
    assert fp64 != tu._is_maximal_inexact_type(dpt.float32, fp16, fp64)
    assert fp64 != tu._is_maximal_inexact_type(dpt.complex64, fp16, fp64)


def test_can_cast_device():
    assert tu._can_cast(dpt.int64, dpt.float64, True, True)
    # if f8 is available, can't cast i8 to f4
    assert not tu._can_cast(dpt.int64, dpt.float32, True, True)
    assert not tu._can_cast(dpt.int64, dpt.float32, False, True)
    # should be able to cast to f8 when f2 unavailable
    assert tu._can_cast(dpt.int64, dpt.float64, False, True)
    # casting to f4 acceptable when f8 unavailable
    assert tu._can_cast(dpt.int64, dpt.float32, True, False)
    assert tu._can_cast(dpt.int64, dpt.float32, False, False)
    # can't safely cast inexact type to inexact type of lesser precision
    assert not tu._can_cast(dpt.float32, dpt.float16, True, False)
    assert not tu._can_cast(dpt.float64, dpt.float32, False, True)


def test_acceptance_fns():
    """Check type promotion acceptance functions"""
    try:
        dev = dpctl.SyclDevice()
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default device is not available")
    assert tu._acceptance_fn_reciprocal(
        dpt.float32, dpt.float32, dpt.float32, dev
    )
    assert tu._acceptance_fn_negative(dpt.int8, dpt.int16, dpt.int16, dev)


def test_weak_types():
    wbt = tu.WeakBooleanType(True)
    assert wbt.get()
    assert tu._weak_type_num_kind(wbt) == 0

    wit = tu.WeakIntegralType(7)
    assert wit.get() == 7
    assert tu._weak_type_num_kind(wit) == 1

    wft = tu.WeakFloatingType(3.1415926)
    assert wft.get() == 3.1415926
    assert tu._weak_type_num_kind(wft) == 2

    wct = tu.WeakComplexType(2.0 + 3.0j)
    assert wct.get() == 2 + 3j
    assert tu._weak_type_num_kind(wct) == 3


def test_arg_validation():
    with pytest.raises(TypeError):
        tu._weak_type_num_kind(dict())

    with pytest.raises(TypeError):
        tu._strong_dtype_num_kind(Ellipsis)

    with pytest.raises(ValueError):
        tu._strong_dtype_num_kind(np.dtype("O"))

    wt = tu.WeakFloatingType(2.0)
    with pytest.raises(ValueError):
        tu._resolve_weak_types(wt, wt, None)
