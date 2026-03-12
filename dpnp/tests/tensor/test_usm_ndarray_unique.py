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
import pytest

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
from dpnp.tests.tensor.helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)


@pytest.mark.parametrize(
    "dtype",
    [
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
def test_unique_values(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, roll = 10000, 734
    inp = dpt.roll(
        dpt.concat((dpt.ones(n, dtype=dtype), dpt.zeros(n, dtype=dtype))),
        roll,
    )

    uv = dpt.unique_values(inp)
    assert dpt.all(uv == dpt.arange(2, dtype=dtype))


def test_unique_values_strided():
    get_queue_or_skip()

    n, m = 1000, 20
    inp = dpt.ones((n, m), dtype="i4", order="F")
    inp[:, ::2] = 0

    uv = dpt.unique_values(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))

    inp = dpt.reshape(inp, -1)
    inp = dpt.flip(dpt.reshape(inp, -1))

    uv = dpt.unique_values(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))


@pytest.mark.parametrize(
    "dtype",
    [
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
def test_unique_counts(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, roll = 10000, 734
    inp = dpt.roll(
        dpt.concat((dpt.ones(n, dtype=dtype), dpt.zeros(n, dtype=dtype))),
        roll,
    )

    uv, uv_counts = dpt.unique_counts(inp)
    assert dpt.all(uv == dpt.arange(2, dtype=dtype))
    assert dpt.all(uv_counts == dpt.full(2, n, dtype=uv_counts.dtype))


def test_unique_counts_strided():
    get_queue_or_skip()

    n, m = 1000, 20
    inp = dpt.ones((n, m), dtype="i4", order="F")
    inp[:, ::2] = 0

    uv, uv_counts = dpt.unique_counts(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(uv_counts == dpt.full(2, n / 2 * m, dtype=uv_counts.dtype))

    inp = dpt.flip(dpt.reshape(inp, -1))

    uv, uv_counts = dpt.unique_counts(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(uv_counts == dpt.full(2, n / 2 * m, dtype=uv_counts.dtype))


@pytest.mark.parametrize(
    "dtype",
    [
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
def test_unique_inverse(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, roll = 10000, 734
    inp = dpt.roll(
        dpt.concat((dpt.ones(n, dtype=dtype), dpt.zeros(n, dtype=dtype))),
        roll,
    )

    uv, inv = dpt.unique_inverse(inp)
    assert dpt.all(uv == dpt.arange(2, dtype=dtype))
    assert dpt.all(inp == uv[inv])
    assert inp.shape == inv.shape


def test_unique_inverse_strided():
    get_queue_or_skip()

    n, m = 1000, 20
    inp = dpt.ones((n, m), dtype="i4", order="F")
    inp[:, ::2] = 0

    uv, inv = dpt.unique_inverse(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(inp == uv[inv])
    assert inp.shape == inv.shape

    inp = dpt.flip(dpt.reshape(inp, -1))

    uv, inv = dpt.unique_inverse(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(inp == uv[inv])
    assert inp.shape == inv.shape


@pytest.mark.parametrize(
    "dtype",
    [
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
    ],
)
def test_unique_all(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    n, roll = 10000, 734
    inp = dpt.roll(
        dpt.concat((dpt.ones(n, dtype=dtype), dpt.zeros(n, dtype=dtype))),
        roll,
    )

    uv, ind, inv, uv_counts = dpt.unique_all(inp)
    assert dpt.all(uv == dpt.arange(2, dtype=dtype))
    assert dpt.all(uv == inp[ind])
    assert dpt.all(inp == uv[inv])
    assert inp.shape == inv.shape
    assert dpt.all(uv_counts == dpt.full(2, n, dtype=uv_counts.dtype))


def test_unique_all_strided():
    get_queue_or_skip()

    n, m = 1000, 20
    inp = dpt.ones((n, m), dtype="i4", order="F")
    inp[:, ::2] = 0

    uv, ind, inv, uv_counts = dpt.unique_all(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(uv == dpt.reshape(inp, -1)[ind])
    assert dpt.all(inp == uv[inv])
    assert inp.shape == inv.shape
    assert dpt.all(uv_counts == dpt.full(2, n / 2 * m, dtype=uv_counts.dtype))

    inp = dpt.flip(dpt.reshape(inp, -1))

    uv, ind, inv, uv_counts = dpt.unique_all(inp)
    assert dpt.all(uv == dpt.arange(2, dtype="i4"))
    assert dpt.all(uv == inp[ind])
    assert dpt.all(inp == uv[inv])
    assert inp.shape == inv.shape
    assert dpt.all(uv_counts == dpt.full(2, n / 2 * m, dtype=uv_counts.dtype))


def test_set_functions_empty_input():
    get_queue_or_skip()
    x = dpt.ones((10, 0, 1), dtype="i4")

    res = dpt.unique_values(x)
    assert isinstance(res, dpt.usm_ndarray)
    assert res.size == 0
    assert res.dtype == x.dtype

    res = dpt.unique_inverse(x)
    assert type(res).__name__ == "UniqueInverseResult"
    uv, inv = res
    assert isinstance(uv, dpt.usm_ndarray)
    assert uv.size == 0
    assert isinstance(inv, dpt.usm_ndarray)
    assert inv.size == 0

    res = dpt.unique_counts(x)
    assert type(res).__name__ == "UniqueCountsResult"
    uv, uv_counts = res
    assert isinstance(uv, dpt.usm_ndarray)
    assert uv.size == 0
    assert isinstance(uv_counts, dpt.usm_ndarray)
    assert uv_counts.size == 0

    res = dpt.unique_all(x)
    assert type(res).__name__ == "UniqueAllResult"
    uv, ind, inv, uv_counts = res
    assert isinstance(uv, dpt.usm_ndarray)
    assert uv.size == 0
    assert isinstance(ind, dpt.usm_ndarray)
    assert ind.size == 0
    assert isinstance(inv, dpt.usm_ndarray)
    assert inv.size == 0
    assert isinstance(uv_counts, dpt.usm_ndarray)
    assert uv_counts.size == 0


def test_set_function_outputs():
    get_queue_or_skip()
    # check standard and early exit paths
    x1 = dpt.arange(10, dtype="i4")
    x2 = dpt.ones((10, 10), dtype="i4")

    assert isinstance(dpt.unique_values(x1), dpt.usm_ndarray)
    assert isinstance(dpt.unique_values(x2), dpt.usm_ndarray)

    assert type(dpt.unique_inverse(x1)).__name__ == "UniqueInverseResult"
    assert type(dpt.unique_inverse(x2)).__name__ == "UniqueInverseResult"

    assert type(dpt.unique_counts(x1)).__name__ == "UniqueCountsResult"
    assert type(dpt.unique_counts(x2)).__name__ == "UniqueCountsResult"

    assert type(dpt.unique_all(x1)).__name__ == "UniqueAllResult"
    assert type(dpt.unique_all(x2)).__name__ == "UniqueAllResult"


def test_set_functions_compute_follows_data():
    # tests that all intermediate calls and allocations
    # are compatible with an input with an arbitrary queue
    get_queue_or_skip()
    q = dpctl.SyclQueue()
    x = dpt.arange(10, dtype="i4", sycl_queue=q)

    uv = dpt.unique_values(x)
    assert isinstance(uv, dpt.usm_ndarray)
    assert uv.sycl_queue == q
    uv, uc = dpt.unique_counts(x)
    assert isinstance(uv, dpt.usm_ndarray)
    assert isinstance(uc, dpt.usm_ndarray)
    assert uv.sycl_queue == q
    assert uc.sycl_queue == q
    uv, inv_ind = dpt.unique_inverse(x)
    assert isinstance(uv, dpt.usm_ndarray)
    assert isinstance(inv_ind, dpt.usm_ndarray)
    assert uv.sycl_queue == q
    assert inv_ind.sycl_queue == q
    uv, ind, inv_ind, uc = dpt.unique_all(x)
    assert isinstance(uv, dpt.usm_ndarray)
    assert isinstance(ind, dpt.usm_ndarray)
    assert isinstance(inv_ind, dpt.usm_ndarray)
    assert isinstance(uc, dpt.usm_ndarray)
    assert uv.sycl_queue == q
    assert ind.sycl_queue == q
    assert inv_ind.sycl_queue == q
    assert uc.sycl_queue == q


def test_gh_1738():
    get_queue_or_skip()

    ones = dpt.ones(10, dtype="i8")
    iota = dpt.arange(10, dtype="i8")

    assert ones.device == iota.device

    dpt_info = dpt.__array_namespace_info__()
    ind_dt = dpt_info.default_dtypes(device=ones.device)["indexing"]

    dt = dpt.unique_inverse(ones).inverse_indices.dtype
    assert dt == ind_dt
    dt = dpt.unique_all(ones).inverse_indices.dtype
    assert dt == ind_dt

    dt = dpt.unique_inverse(iota).inverse_indices.dtype
    assert dt == ind_dt
    dt = dpt.unique_all(iota).inverse_indices.dtype
    assert dt == ind_dt
