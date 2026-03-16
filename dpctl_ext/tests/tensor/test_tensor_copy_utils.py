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
import dpctl_ext.tensor._copy_utils as cu

from .helper import get_queue_or_skip


def test_copy_utils_empty_like_orderK():
    get_queue_or_skip()
    a = dpt.empty((10, 10), dtype=dpt.int32, order="F")
    X = cu._empty_like_orderK(a, dpt.int32, a.usm_type, a.device)
    assert X.flags["F"]


def test_copy_utils_empty_like_orderK_invalid_args():
    get_queue_or_skip()
    with pytest.raises(TypeError):
        cu._empty_like_orderK([1, 2, 3], dpt.int32, "device", None)
    with pytest.raises(TypeError):
        cu._empty_like_pair_orderK(
            [1, 2, 3],
            (
                1,
                2,
                3,
            ),
            dpt.int32,
            (3,),
            "device",
            None,
        )

    a = dpt.empty(10, dtype=dpt.int32)
    with pytest.raises(TypeError):
        cu._empty_like_pair_orderK(
            a,
            (
                1,
                2,
                3,
            ),
            dpt.int32,
            (10,),
            "device",
            None,
        )


def test_copy_utils_from_numpy_empty_like_orderK():
    q = get_queue_or_skip()

    a = np.empty((10, 10), dtype=np.int32, order="C")
    r0 = cu._from_numpy_empty_like_orderK(a, dpt.int32, "device", q)
    assert r0.flags["C"]

    b = np.empty((10, 10), dtype=np.int32, order="F")
    r1 = cu._from_numpy_empty_like_orderK(b, dpt.int32, "device", q)
    assert r1.flags["F"]

    c = np.empty((2, 3, 4), dtype=np.int32, order="C")
    c = np.transpose(c, (1, 0, 2))
    r2 = cu._from_numpy_empty_like_orderK(c, dpt.int32, "device", q)
    assert not r2.flags["C"] and not r2.flags["F"]


def test_copy_utils_from_numpy_empty_like_orderK_invalid_args():
    with pytest.raises(TypeError):
        cu._from_numpy_empty_like_orderK([1, 2, 3], dpt.int32, "device", None)


def test_gh_2055():
    """
    Test that `dpt.asarray` works on contiguous NumPy arrays with `order="K"`
    when dimensions are permuted.

    See: https://github.com/IntelPython/dpctl/issues/2055
    """
    get_queue_or_skip()

    a = np.ones((2, 3, 4), dtype=dpt.int32)
    a_t = np.transpose(a, (2, 0, 1))
    r = dpt.asarray(a_t)
    assert not r.flags["C"] and not r.flags["F"]
