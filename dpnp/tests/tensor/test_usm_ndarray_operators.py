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


class Dummy:
    @staticmethod
    def abs(a):
        return a

    @staticmethod
    def add(a, b):
        if isinstance(a, dpt.usm_ndarray):
            return a
        else:
            return b

    @staticmethod
    def subtract(a, b):
        if isinstance(a, dpt.usm_ndarray):
            return a
        else:
            return b

    @staticmethod
    def multiply(a, b):
        if isinstance(a, dpt.usm_ndarray):
            return a
        else:
            return b


@pytest.mark.parametrize("namespace", [dpt, Dummy()])
def test_fp_ops(namespace):
    try:
        X = dpt.ones(1)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X._set_namespace(namespace)
    assert X.__array_namespace__() is namespace
    X[0] = -2.5
    X.__abs__()
    X.__add__(1.0)
    X.__radd__(1.0)
    X.__sub__(1.0)
    X.__rsub__(1.0)
    X.__mul__(1.0)
    X.__rmul__(1.0)
    X.__truediv__(1.0)
    X.__rtruediv__(1.0)
    X.__floordiv__(1.0)
    X.__rfloordiv__(1.0)
    X.__pos__()
    X.__neg__()
    X.__eq__(-2.5)
    X.__ne__(-2.5)
    X.__le__(-2.5)
    X.__ge__(-2.5)
    X.__gt__(-2.0)
    X.__iadd__(X)
    X.__isub__(X)
    X.__imul__(X)
    X.__itruediv__(1.0)
    X.__ifloordiv__(1.0)


@pytest.mark.parametrize("namespace", [dpt, Dummy()])
def test_int_ops(namespace):
    try:
        X = dpt.usm_ndarray(1, "i4")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X._set_namespace(namespace)
    assert X.__array_namespace__() is namespace
    X.__lshift__(2)
    X.__rshift__(2)
    X.__rlshift__(2)
    X.__rrshift__(2)
    X.__ilshift__(2)
    X.__irshift__(2)
    X.__and__(X)
    X.__rand__(X)
    X.__iand__(X)
    X.__or__(X)
    X.__ror__(X)
    X.__ior__(X)
    X.__xor__(X)
    X.__rxor__(X)
    X.__ixor__(X)
    X.__invert__()
    X.__mod__(5)
    X.__rmod__(5)
    X.__imod__(5)
    X.__pow__(2)
    X.__rpow__(2)
    X.__ipow__(2)


@pytest.mark.parametrize("namespace", [dpt, Dummy()])
def test_mat_ops(namespace):
    try:
        M = dpt.eye(3, 3)
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    M._set_namespace(namespace)
    assert M.__array_namespace__() is namespace
    M.__matmul__(M)
    M.__imatmul__(M)
    M.__rmatmul__(M)


@pytest.mark.parametrize("namespace", [dpt, Dummy()])
def test_comp_ops(namespace):
    try:
        X = dpt.asarray(1, dtype="u8")
    except dpctl.SyclDeviceCreationError:
        pytest.skip("No SYCL devices available")
    X._set_namespace(namespace)
    assert X.__array_namespace__() is namespace
    assert X.__gt__(-1)
    assert X.__ge__(-1)
    assert not X.__lt__(-1)
    assert not X.__le__(-1)
    assert not X.__eq__(-1)
    assert X.__ne__(-1)
