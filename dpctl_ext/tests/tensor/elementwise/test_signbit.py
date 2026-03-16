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

import numpy as np
import pytest

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt

from ..helper import (
    get_queue_or_skip,
    skip_if_dtype_not_supported,
)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_signbit_out_type_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    x = dpt.linspace(1, 10, num=256, dtype=arg_dt)
    sb = dpt.signbit(x)
    assert sb.dtype == dpt.bool

    assert not dpt.any(sb)

    x2 = dpt.linspace(-10, -1, num=256, dtype=arg_dt)
    sb2 = dpt.signbit(x2)
    assert dpt.all(sb2)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_signbit_out_type_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    x = dpt.linspace(1, 10, num=256, dtype=arg_dt)
    sb = dpt.signbit(x[::-3])
    assert sb.dtype == dpt.bool

    assert not dpt.any(sb)

    x2 = dpt.linspace(-10, -1, num=256, dtype=arg_dt)
    sb2 = dpt.signbit(x2[::-3])
    assert dpt.all(sb2)


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_signbit_special_cases_contig(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    n = 63
    x1 = dpt.full(n, -dpt.inf, dtype=arg_dt)
    x2 = dpt.full(n, -0.0, dtype=arg_dt)
    x3 = dpt.full(n, 0.0, dtype=arg_dt)
    x4 = dpt.full(n, dpt.inf, dtype=arg_dt)

    x = dpt.concat((x1, x2, x3, x4))
    actual = dpt.signbit(x)

    expected = dpt.concat(
        (
            dpt.full(x1.size, True),
            dpt.full(x2.size, True),
            dpt.full(x3.size, False),
            dpt.full(x4.size, False),
        )
    )

    assert dpt.all(dpt.equal(actual, expected))


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8"])
def test_signbit_special_cases_strided(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    arg_dt = np.dtype(dtype)
    x1 = dpt.full(63, -dpt.inf, dtype=arg_dt)
    x2 = dpt.full(63, -0.0, dtype=arg_dt)
    x3 = dpt.full(63, 0.0, dtype=arg_dt)
    x4 = dpt.full(63, dpt.inf, dtype=arg_dt)

    x = dpt.concat((x1, x2, x3, x4))
    actual = dpt.signbit(x[::-1])

    expected = dpt.concat(
        (
            dpt.full(x4.size, False),
            dpt.full(x3.size, False),
            dpt.full(x2.size, True),
            dpt.full(x1.size, True),
        )
    )

    assert dpt.all(dpt.equal(actual, expected))
