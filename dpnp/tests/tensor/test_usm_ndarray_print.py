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


class TestPrint:
    def setup_method(self):
        self._retain_options = dpt.get_print_options()

    def teardown_method(self):
        dpt.set_print_options(**self._retain_options)


class TestArgValidation(TestPrint):
    @pytest.mark.parametrize(
        "arg,err",
        [
            ({"linewidth": "I"}, TypeError),
            ({"edgeitems": "I"}, TypeError),
            ({"threshold": "I"}, TypeError),
            ({"precision": "I"}, TypeError),
            ({"floatmode": "I"}, ValueError),
            ({"edgeitems": "I"}, TypeError),
            ({"sign": "I"}, ValueError),
            ({"nanstr": np.nan}, TypeError),
            ({"infstr": np.nan}, TypeError),
        ],
    )
    def test_print_option_arg_validation(self, arg, err):
        with pytest.raises(err):
            dpt.set_print_options(**arg)

    def test_usm_ndarray_repr_arg_validation(self):
        X = {}
        with pytest.raises(TypeError):
            dpt.usm_ndarray_repr(X)

        try:
            X = dpt.arange(4)
        except dpctl.SyclDeviceCreationError:
            pytest.skip("No SYCL devices available")
        with pytest.raises(TypeError):
            dpt.usm_ndarray_repr(X, line_width="I")

        with pytest.raises(TypeError):
            dpt.usm_ndarray_repr(X, precision="I")

        with pytest.raises(TypeError):
            dpt.usm_ndarray_repr(X, prefix=4)

    def test_usm_ndarray_str_arg_validation(self):
        X = {}
        with pytest.raises(TypeError):
            dpt.usm_ndarray_str(X)

        try:
            X = dpt.arange(4)
        except dpctl.SyclDeviceCreationError:
            pytest.skip("No SYCL devices available")

        with pytest.raises(TypeError):
            dpt.usm_ndarray_str(X, line_width="I")

        with pytest.raises(TypeError):
            dpt.usm_ndarray_str(X, edge_items="I")

        with pytest.raises(TypeError):
            dpt.usm_ndarray_str(X, threshold="I")

        with pytest.raises(TypeError):
            dpt.usm_ndarray_str(X, precision="I")

        with pytest.raises(ValueError):
            dpt.usm_ndarray_str(X, floatmode="I")

        with pytest.raises(TypeError):
            dpt.usm_ndarray_str(X, edge_items="I")

        with pytest.raises(ValueError):
            dpt.usm_ndarray_str(X, sign="I")

        with pytest.raises(TypeError):
            dpt.usm_ndarray_str(X, prefix=4)

        with pytest.raises(TypeError):
            dpt.usm_ndarray_str(X, prefix=4)

        with pytest.raises(TypeError):
            dpt.usm_ndarray_str(X, suffix=4)


class TestSetPrintOptions(TestPrint):
    def test_set_linewidth(self):
        q = get_queue_or_skip()

        dpt.set_print_options(linewidth=1)
        x = dpt.asarray([0, 1], sycl_queue=q)
        assert str(x) == "[0\n 1]"

    def test_set_precision(self):
        q = get_queue_or_skip()

        dpt.set_print_options(precision=4)
        x = dpt.asarray([1.23450], sycl_queue=q)
        assert str(x) == "[1.2345]"

    def test_threshold_edgeitems(self):
        q = get_queue_or_skip()

        dpt.set_print_options(threshold=1, edgeitems=1)
        x = dpt.arange(9, sycl_queue=q)
        assert str(x) == "[0 ... 8]"
        dpt.set_print_options(edgeitems=9)
        assert str(x) == "[0 1 2 3 4 5 6 7 8]"

    def test_floatmodes(self):
        q = get_queue_or_skip()

        x = dpt.asarray([0.1234, 0.1234678], sycl_queue=q)
        dpt.set_print_options(floatmode="fixed", precision=4)
        assert str(x) == "[0.1234 0.1235]"

        dpt.set_print_options(floatmode="unique")
        assert str(x) == "[0.1234    0.1234678]"

        dpt.set_print_options(floatmode="maxprec")
        assert str(x) == "[0.1234 0.1235]"

        dpt.set_print_options(floatmode="maxprec", precision=8)
        assert str(x) == "[0.1234    0.1234678]"

        dpt.set_print_options(floatmode="maxprec_equal", precision=4)
        assert str(x) == "[0.1234 0.1235]"

        dpt.set_print_options(floatmode="maxprec_equal", precision=8)
        assert str(x) == "[0.1234000 0.1234678]"

    def test_nan_inf_suppress(self):
        q = get_queue_or_skip()

        dpt.set_print_options(nanstr="nan1", infstr="inf1")
        x = dpt.asarray([np.nan, np.inf], sycl_queue=q)
        assert str(x) == "[nan1 inf1]"

    def test_suppress_small(self):
        q = get_queue_or_skip()

        dpt.set_print_options(suppress=True)
        x = dpt.asarray(5e-10, sycl_queue=q)
        assert str(x) == "0."

    def test_sign(self):
        q = get_queue_or_skip()

        x = dpt.asarray([0.0, 1.0, 2.0], sycl_queue=q)
        y = dpt.asarray(1.0, sycl_queue=q)
        z = dpt.asarray([1.0 + 1.0j], sycl_queue=q)
        assert str(x) == "[0. 1. 2.]"
        assert str(y) == "1."
        assert str(z) == "[1.+1.j]"

        dpt.set_print_options(sign="+")
        assert str(x) == "[+0. +1. +2.]"
        assert str(y) == "+1."
        assert str(z) == "[+1.+1.j]"

        dpt.set_print_options(sign=" ")
        assert str(x) == "[ 0.  1.  2.]"
        assert str(y) == " 1."
        assert str(z) == "[ 1.+1.j]"

    def test_numpy(self):
        dpt.set_print_options(numpy=True)
        options = dpt.get_print_options()
        np_options = np.get_printoptions()
        assert all(np_options[k] == options[k] for k in options.keys())


class TestPrintFns(TestPrint):
    @pytest.mark.parametrize(
        "dtype,x_str",
        [
            ("b1", "[False  True  True  True]"),
            ("i1", "[0 1 2 3]"),
            ("u1", "[0 1 2 3]"),
            ("i2", "[0 1 2 3]"),
            ("u2", "[0 1 2 3]"),
            ("i4", "[0 1 2 3]"),
            ("u4", "[0 1 2 3]"),
            ("i8", "[0 1 2 3]"),
            ("u8", "[0 1 2 3]"),
            ("f2", "[0. 1. 2. 3.]"),
            ("f4", "[0. 1. 2. 3.]"),
            ("f8", "[0. 1. 2. 3.]"),
            ("c8", "[0.+0.j 1.+0.j 2.+0.j 3.+0.j]"),
            ("c16", "[0.+0.j 1.+0.j 2.+0.j 3.+0.j]"),
        ],
    )
    def test_print_types(self, dtype, x_str):
        q = get_queue_or_skip()
        skip_if_dtype_not_supported(dtype, q)

        x = dpt.asarray([0, 1, 2, 3], dtype=dtype, sycl_queue=q)
        assert str(x) == x_str

    def test_print_str(self):
        q = get_queue_or_skip()

        x = dpt.asarray(0, sycl_queue=q)
        assert str(x) == "0"

        x = dpt.asarray([np.nan, np.inf], sycl_queue=q)
        assert str(x) == "[nan inf]"

        x = dpt.arange(9, sycl_queue=q)
        assert str(x) == "[0 1 2 3 4 5 6 7 8]"

        y = dpt.reshape(x, (3, 3), copy=True)
        assert str(y) == "[[0 1 2]\n [3 4 5]\n [6 7 8]]"

    def test_print_str_abbreviated(self):
        q = get_queue_or_skip()

        dpt.set_print_options(threshold=0, edgeitems=1)
        x = dpt.arange(9, sycl_queue=q)
        assert str(x) == "[0 ... 8]"

        x = dpt.reshape(x, (3, 3))
        assert str(x) == "[[0 ... 2]\n ...\n [6 ... 8]]"

    def test_usm_ndarray_str_separator(self):
        q = get_queue_or_skip()

        x = dpt.reshape(dpt.arange(4, sycl_queue=q), (2, 2))

        np.testing.assert_equal(
            dpt.usm_ndarray_str(x, prefix="test", separator="   "),
            "[[0   1]\n     [2   3]]",
        )

    def test_print_repr(self):
        q = get_queue_or_skip()

        x = dpt.asarray(3, dtype="int64", sycl_queue=q)
        assert repr(x) == "usm_ndarray(3)"

        x = dpt.asarray([np.nan, np.inf], sycl_queue=q)
        if x.sycl_device.has_aspect_fp64:
            assert repr(x) == "usm_ndarray([nan, inf])"
        else:
            assert repr(x) == "usm_ndarray([nan, inf], dtype=float32)"

        x = dpt.arange(9, sycl_queue=q, dtype="int64")
        assert repr(x) == "usm_ndarray([0, 1, 2, 3, 4, 5, 6, 7, 8])"

        x = dpt.reshape(x, (3, 3))
        np.testing.assert_equal(
            repr(x),
            "usm_ndarray([[0, 1, 2],"
            "\n             [3, 4, 5],"
            "\n             [6, 7, 8]])",
        )

        x = dpt.arange(4, dtype="i4", sycl_queue=q)
        assert repr(x) == "usm_ndarray([0, 1, 2, 3], dtype=int32)"

        dpt.set_print_options(linewidth=1)
        np.testing.assert_equal(
            repr(x),
            "usm_ndarray([0,"
            "\n             1,"
            "\n             2,"
            "\n             3],"
            "\n            dtype=int32)",
        )

        # zero-size array
        dpt.set_print_options(linewidth=75)
        x = dpt.ones((9, 0), dtype="i4", sycl_queue=q)
        assert repr(x) == "usm_ndarray([], shape=(9, 0), dtype=int32)"

    def test_print_repr_abbreviated(self):
        q = get_queue_or_skip()

        dpt.set_print_options(threshold=0, edgeitems=1)
        x = dpt.arange(9, dtype="int64", sycl_queue=q)
        assert repr(x) == "usm_ndarray([0, ..., 8], shape=(9,))"

        y = dpt.asarray(x, dtype="i4", copy=True)
        assert repr(y) == "usm_ndarray([0, ..., 8], shape=(9,), dtype=int32)"

        x = dpt.reshape(x, (3, 3))
        np.testing.assert_equal(
            repr(x),
            "usm_ndarray([[0, ..., 2],"
            "\n             ...,"
            "\n             [6, ..., 8]], shape=(3, 3))",
        )

        y = dpt.reshape(y, (3, 3))
        np.testing.assert_equal(
            repr(y),
            "usm_ndarray([[0, ..., 2],"
            "\n             ...,"
            "\n             [6, ..., 8]], shape=(3, 3), dtype=int32)",
        )

        dpt.set_print_options(linewidth=1)
        np.testing.assert_equal(
            repr(y),
            "usm_ndarray([[0,"
            "\n              ...,"
            "\n              2],"
            "\n             ...,"
            "\n             [6,"
            "\n              ...,"
            "\n              8]],"
            "\n            shape=(3, 3),"
            "\n            dtype=int32)",
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
            "u8",
            "f2",
            "f4",
            "c8",
        ],
    )
    def test_repr_appended_dtype(self, dtype):
        q = get_queue_or_skip()
        skip_if_dtype_not_supported(dtype, q)

        x = dpt.empty(4, dtype=dtype)
        assert repr(x).split("=")[-1][:-1] == x.dtype.name

    def test_usm_ndarray_repr_prefix(self):
        q = get_queue_or_skip()

        x = dpt.arange(4, dtype=np.intp, sycl_queue=q)
        np.testing.assert_equal(
            dpt.usm_ndarray_repr(x, prefix="test"), "test([0, 1, 2, 3])"
        )
        x = dpt.reshape(x, (2, 2))
        np.testing.assert_equal(
            dpt.usm_ndarray_repr(x, prefix="test"),
            "test([[0, 1]," "\n      [2, 3]])",
        )


class TestContextManager:
    def test_context_manager_basic(self):
        options = dpt.get_print_options()
        try:
            X = dpt.asarray(1.234567)
        except dpctl.SyclDeviceCreationError:
            pytest.skip("No SYCL devices available")
        with dpt.print_options(precision=4):
            s = str(X)
        assert s == "1.2346"
        assert options == dpt.get_print_options()

    def test_context_manager_as(self):
        with dpt.print_options(precision=4) as x:
            options = x.copy()
        assert options["precision"] == 4
