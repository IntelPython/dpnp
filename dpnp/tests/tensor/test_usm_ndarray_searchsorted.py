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
import numpy as np
import pytest

import dpnp.tensor as dpt

from .helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)


def _check(hay_stack, needles, needles_np):
    assert hay_stack.dtype == needles.dtype
    assert hay_stack.ndim == 1

    info_ = dpt.__array_namespace_info__()
    default_dts_dev = info_.default_dtypes(device=hay_stack.device)
    index_dt = default_dts_dev["indexing"]

    p_left = dpt.searchsorted(hay_stack, needles, side="left")
    assert p_left.dtype == index_dt

    hs_np = dpt.asnumpy(hay_stack)
    ref_left = np.searchsorted(hs_np, needles_np, side="left")
    assert dpt.all(p_left == dpt.asarray(ref_left))

    p_right = dpt.searchsorted(hay_stack, needles, side="right")
    assert p_right.dtype == index_dt

    ref_right = np.searchsorted(hs_np, needles_np, side="right")
    assert dpt.all(p_right == dpt.asarray(ref_right))

    sorter = dpt.arange(hay_stack.size)
    ps_left = dpt.searchsorted(hay_stack, needles, side="left", sorter=sorter)
    assert ps_left.dtype == index_dt
    assert dpt.all(ps_left == p_left)
    ps_right = dpt.searchsorted(hay_stack, needles, side="right", sorter=sorter)
    assert ps_right.dtype == index_dt
    assert dpt.all(ps_right == p_right)


def test_searchsorted_contig_bool():
    get_queue_or_skip()

    dt = dpt.bool

    hay_stack = dpt.arange(0, 1, dtype=dt)
    needles_np = np.random.choice([True, False], size=1024)
    needles = dpt.asarray(needles_np)

    _check(hay_stack, needles, needles_np)
    _check(
        hay_stack,
        dpt.reshape(needles, (32, 32)),
        np.reshape(needles_np, (32, 32)),
    )


def test_searchsorted_strided_bool():
    get_queue_or_skip()

    dt = dpt.bool

    hay_stack = dpt.repeat(dpt.arange(0, 1, dtype=dt), 4)[::4]
    needles_np = np.random.choice([True, False], size=2 * 1024)
    needles = dpt.asarray(needles_np)
    sl = slice(None, None, -2)

    _check(hay_stack, needles[sl], needles_np[sl])
    _check(
        hay_stack,
        dpt.reshape(needles[sl], (32, 32)),
        np.reshape(needles_np[sl], (32, 32)),
    )


@pytest.mark.parametrize(
    "idt",
    [
        dpt.int8,
        dpt.uint8,
        dpt.int16,
        dpt.uint16,
        dpt.int32,
        dpt.uint32,
        dpt.int64,
        dpt.uint64,
    ],
)
def test_searchsorted_contig_int(idt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(idt, q)

    dt = dpt.dtype(idt)
    max_v = dpt.iinfo(dt).max

    hay_stack = dpt.arange(0, min(max_v, 255), dtype=dt)
    needles_np = np.random.randint(0, max_v, dtype=dt, size=1024)
    needles = dpt.asarray(needles_np)

    _check(hay_stack, needles, needles_np)
    _check(
        hay_stack,
        dpt.reshape(needles, (32, 32)),
        np.reshape(needles_np, (32, 32)),
    )


@pytest.mark.parametrize(
    "idt",
    [
        dpt.int8,
        dpt.uint8,
        dpt.int16,
        dpt.uint16,
        dpt.int32,
        dpt.uint32,
        dpt.int64,
        dpt.uint64,
    ],
)
def test_searchsorted_strided_int(idt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(idt, q)

    dt = dpt.dtype(idt)
    max_v = dpt.iinfo(dt).max

    hay_stack = dpt.repeat(dpt.arange(0, min(max_v, 255), dtype=dt), 4)[1::4]
    needles_np = np.random.randint(0, max_v, dtype=dt, size=2 * 1024)
    needles = dpt.asarray(needles_np)
    sl = slice(None, None, -2)

    _check(hay_stack, needles[sl], needles_np[sl])
    _check(
        hay_stack,
        dpt.reshape(needles[sl], (32, 32)),
        np.reshape(needles_np[sl], (32, 32)),
    )


def _add_extended_fp(array):
    array[0] = -dpt.inf
    array[-2] = dpt.inf
    array[-1] = dpt.nan


@pytest.mark.parametrize("idt", [dpt.float16, dpt.float32, dpt.float64])
def test_searchsorted_contig_fp(idt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(idt, q)

    dt = dpt.dtype(idt)

    hay_stack = dpt.linspace(0, 1, num=255, dtype=dt, endpoint=True)
    _add_extended_fp(hay_stack)

    needles_np = np.random.uniform(-0.1, 1.1, size=1024).astype(dt)
    needles = dpt.asarray(needles_np)

    _check(hay_stack, needles, needles_np)
    _check(
        hay_stack,
        dpt.reshape(needles, (32, 32)),
        np.reshape(needles_np, (32, 32)),
    )


@pytest.mark.parametrize("idt", [dpt.float16, dpt.float32, dpt.float64])
def test_searchsorted_strided_fp(idt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(idt, q)

    dt = dpt.dtype(idt)

    hay_stack = dpt.repeat(
        dpt.linspace(0, 1, num=255, dtype=dt, endpoint=True), 4
    )[1::4]
    _add_extended_fp(hay_stack)

    needles_np = np.random.uniform(-0.1, 1.1, size=3 * 1024).astype(dt)
    needles = dpt.asarray(needles_np)
    sl = slice(1, None, 3)

    _check(hay_stack, needles[sl], needles_np[sl])
    _check(
        hay_stack,
        dpt.reshape(needles[sl], (32, 32)),
        np.reshape(needles_np[sl], (32, 32)),
    )


def _add_extended_cfp(array):
    dt = array.dtype
    ev_li = [
        complex(-dpt.inf, -1),
        complex(-dpt.inf, -dpt.inf),
        complex(-dpt.inf, dpt.inf),
        complex(-dpt.inf, dpt.nan),
        complex(0, -dpt.inf),
        complex(0, -1),
        complex(0, dpt.inf),
        complex(0, dpt.nan),
        complex(dpt.inf, -dpt.inf),
        complex(dpt.inf, -1),
        complex(dpt.inf, dpt.inf),
        complex(dpt.inf, dpt.nan),
        complex(dpt.nan, -dpt.inf),
        complex(dpt.nan, -1),
        complex(dpt.nan, dpt.inf),
        complex(dpt.nan, dpt.nan),
    ]
    ev = dpt.asarray(ev_li, dtype=dt, device=array.device)
    return dpt.sort(dpt.concat((ev, array)))


@pytest.mark.parametrize("idt", [dpt.complex64, dpt.complex128])
def test_searchsorted_contig_cfp(idt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(idt, q)

    dt = dpt.dtype(idt)

    hay_stack = dpt.linspace(0, 1, num=255, dtype=dt, endpoint=True)
    hay_stack = _add_extended_cfp(hay_stack)
    needles_np = np.random.uniform(-0.1, 1.1, size=1024).astype(dt)
    needles = dpt.asarray(needles_np)

    _check(hay_stack, needles, needles_np)
    _check(
        hay_stack,
        dpt.reshape(needles, (32, 32)),
        np.reshape(needles_np, (32, 32)),
    )


@pytest.mark.parametrize("idt", [dpt.complex64, dpt.complex128])
def test_searchsorted_strided_cfp(idt):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(idt, q)

    dt = dpt.dtype(idt)

    hay_stack = dpt.repeat(
        dpt.linspace(0, 1, num=255, dtype=dt, endpoint=True), 4
    )[1::4]
    needles_np = np.random.uniform(-0.1, 1.1, size=3 * 1024).astype(dt)
    needles = dpt.asarray(needles_np)
    sl = slice(1, None, 3)

    _check(hay_stack, needles[sl], needles_np[sl])
    _check(
        hay_stack,
        dpt.reshape(needles[sl], (32, 32)),
        np.reshape(needles_np[sl], (32, 32)),
    )

    hay_stack = _add_extended_cfp(hay_stack)
    _check(hay_stack, needles[sl], needles_np[sl])
    _check(
        hay_stack,
        dpt.reshape(needles[sl], (32, 32)),
        np.reshape(needles_np[sl], (32, 32)),
    )


def test_searchsorted_coerce():
    get_queue_or_skip()

    x1_i4 = dpt.arange(5, dtype="i4")
    x1_i8 = dpt.arange(5, dtype="i8")
    x2_i4 = dpt.arange(5, dtype="i4")
    x2_i8 = dpt.arange(5, dtype="i8")

    p1 = dpt.searchsorted(x1_i4, x2_i8)
    p2 = dpt.searchsorted(x1_i8, x2_i8)
    p3 = dpt.searchsorted(x1_i8, x2_i4)
    assert dpt.all(p1 == p2)
    assert dpt.all(p2 == p3)


def test_searchsorted_validation():
    with pytest.raises(TypeError):
        dpt.searchsorted(None, None)
    try:
        x1 = dpt.arange(10, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default device could not be created")
    with pytest.raises(TypeError):
        dpt.searchsorted(x1, None)
    with pytest.raises(TypeError):
        dpt.searchsorted(x1, x1, sorter=dict())
    with pytest.raises(ValueError):
        dpt.searchsorted(x1, x1, side="unknown")


def test_searchsorted_validation2():
    try:
        x1 = dpt.arange(10, dtype="i4")
        sorter = dpt.arange(10, dtype="i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("Default device could not be created")
    d = x1.sycl_device
    q2 = dpctl.SyclQueue(d, property="in_order")
    x2 = dpt.ones(5, dtype=x1.dtype, sycl_queue=q2)

    with pytest.raises(dpt.ExecutionPlacementError):
        dpt.searchsorted(x1, x2)

    with pytest.raises(dpt.ExecutionPlacementError):
        dpt.searchsorted(x1, x2, sorter=sorter)

    sorter = dpt.ones(x1.shape, dtype=dpt.bool)
    # non-integral sorter.dtype raises
    with pytest.raises(ValueError):
        dpt.searchsorted(x1, x1, sorter=sorter)

    # non-matching x1.shape and sorter.shape raises
    with pytest.raises(ValueError):
        dpt.searchsorted(x1, x1, sorter=sorter[:-1])

    # x1 must be 1d, or ValueError is raised
    with pytest.raises(ValueError):
        dpt.searchsorted(x1[dpt.newaxis, :], x1)


def test_pw_linear_interpolation_example():
    get_queue_or_skip()

    bins = dpt.asarray([0.0, 0.05, 0.2, 0.25, 0.5, 0.8, 0.95, 1])
    vals = dpt.asarray([0.1, 0.15, 0.3, 0.5, 0.7, 0.53, 0.37, 0.1])
    assert vals.shape == bins.shape
    data_np = np.random.uniform(0, 1, size=10000)
    data = dpt.asarray(data_np)

    p = dpt.searchsorted(bins, data)
    w = (data - bins[p]) / (bins[p - 1] - bins[p])
    assert dpt.min(w) >= 0
    assert dpt.max(w) <= 1
    interp_vals = vals[p - 1] * w + (1 - w) * vals[p]

    assert interp_vals.shape == data.shape
    assert dpt.min(interp_vals) >= dpt.zeros(tuple())
    av = dpt.sum(interp_vals) / data.size
    exp = dpt.vecdot(vals[1:] + vals[:-1], bins[1:] - bins[:-1]) / 2

    assert dpt.abs(av - exp) < 0.1


def test_out_of_bound_sorter_values():
    get_queue_or_skip()

    x = dpt.asarray([1, 2, 0], dtype="i4")
    n = x.shape[0]

    # use out-of-bounds indices in sorter
    sorter = dpt.asarray([2, 0 - n, 1 - n], dtype="i8")

    x2 = dpt.arange(3, dtype=x.dtype)
    p = dpt.searchsorted(x, x2, sorter=sorter)
    # verify that they were applied with mode="wrap"
    assert dpt.all(p == dpt.arange(3, dtype=p.dtype))


def test_searchsorted_strided_scalar_needle():
    get_queue_or_skip()

    a_max = 255

    hay_stack = dpt.flip(
        dpt.repeat(dpt.arange(a_max - 1, -1, -1, dtype=dpt.int32), 4)
    )
    needles_np = np.squeeze(
        np.random.randint(0, a_max, dtype=dpt.int32, size=1), axis=0
    )
    needles = dpt.asarray(needles_np)

    _check(hay_stack, needles, needles_np)
