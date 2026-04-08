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

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
import pytest

from ..helper import get_queue_or_skip

unary_fn = dpt.negative
binary_fn = dpt.divide


def test_unary_class_getters():
    fn = unary_fn.get_implementation_function()
    assert callable(fn)

    fn = unary_fn.get_type_result_resolver_function()
    assert callable(fn)


def test_unary_class_types_property():
    get_queue_or_skip()
    loop_types = unary_fn.types
    assert isinstance(loop_types, list)
    assert len(loop_types) > 0
    assert all(isinstance(sig, str) for sig in loop_types)
    assert all("->" in sig for sig in loop_types)


def test_unary_class_str_repr():
    s = str(unary_fn)
    r = repr(unary_fn)

    assert isinstance(s, str)
    assert isinstance(r, str)
    kl_n = unary_fn.__name__
    assert kl_n in s
    assert kl_n in r


def test_unary_read_only_out():
    get_queue_or_skip()
    x = dpt.arange(32, dtype=dpt.int32)
    r = dpt.empty_like(x)
    r.flags["W"] = False
    with pytest.raises(ValueError):
        unary_fn(x, out=r)


def test_binary_class_getters():
    fn = binary_fn.get_implementation_function()
    assert callable(fn)

    fn = binary_fn.get_implementation_inplace_function()
    assert callable(fn)

    fn = binary_fn.get_type_result_resolver_function()
    assert callable(fn)

    fn = binary_fn.get_type_promotion_path_acceptance_function()
    assert callable(fn)


def test_binary_class_types_property():
    get_queue_or_skip()
    loop_types = binary_fn.types
    assert isinstance(loop_types, list)
    assert len(loop_types) > 0
    assert all(isinstance(sig, str) for sig in loop_types)
    assert all("->" in sig for sig in loop_types)


def test_binary_class_str_repr():
    s = str(binary_fn)
    r = repr(binary_fn)

    assert isinstance(s, str)
    assert isinstance(r, str)
    kl_n = binary_fn.__name__
    assert kl_n in s
    assert kl_n in r


def test_unary_class_nin():
    nin = unary_fn.nin
    assert isinstance(nin, int)
    assert nin == 1


def test_binary_class_nin():
    nin = binary_fn.nin
    assert isinstance(nin, int)
    assert nin == 2


def test_unary_class_nout():
    nout = unary_fn.nout
    assert isinstance(nout, int)
    assert nout == 1


def test_binary_class_nout():
    nout = binary_fn.nout
    assert isinstance(nout, int)
    assert nout == 1


def test_binary_read_only_out():
    get_queue_or_skip()
    x1 = dpt.ones(32, dtype=dpt.float32)
    x2 = dpt.ones_like(x1)
    r = dpt.empty_like(x1)
    r.flags["W"] = False
    with pytest.raises(ValueError):
        binary_fn(x1, x2, out=r)


def test_binary_no_inplace_op():
    get_queue_or_skip()
    x1 = dpt.ones(10, dtype="i4")
    x2 = dpt.ones_like(x1)

    with pytest.raises(ValueError):
        dpt.logaddexp._inplace_op(x1, x2)
