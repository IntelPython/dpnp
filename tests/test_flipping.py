from math import prod

import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from numpy.testing import (
    assert_equal,
)

import dpnp

from .helper import (
    get_all_dtypes,
)


class TestFlip:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_arange_2d_default_axis(self, dtype):
        sh = (2, 3) if dtype != dpnp.bool else (1, 1)
        dp_a = dpnp.arange(prod(sh), dtype=dtype).reshape(sh)
        np_a = numpy.arange(prod(sh), dtype=dtype).reshape(sh)

        assert_equal(dpnp.flip(dp_a), numpy.flip(np_a))

    @pytest.mark.parametrize("axis", list(range(3)))
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_arange_3d(self, axis, dtype):
        sh = (2, 2, 2)
        dp_a = dpnp.arange(prod(sh), dtype=dtype).reshape(sh)
        np_a = numpy.arange(prod(sh), dtype=dtype).reshape(sh)

        assert_equal(dpnp.flip(dp_a, axis=axis), numpy.flip(np_a, axis=axis))

    @pytest.mark.parametrize("axis", [(), (0, 2), (1, 2)])
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_arange_3d_multiple_axes(self, axis, dtype):
        sh = (2, 2, 2)
        dp_a = dpnp.arange(prod(sh), dtype=dtype).reshape(sh)
        np_a = numpy.arange(prod(sh), dtype=dtype).reshape(sh)

        assert_equal(dpnp.flip(dp_a, axis=axis), numpy.flip(np_a, axis=axis))

    @pytest.mark.parametrize("axis", list(range(4)))
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_arange_4d(self, axis, dtype):
        sh = (2, 3, 4, 5)
        dp_a = dpnp.arange(prod(sh), dtype=dtype).reshape(sh)
        np_a = numpy.arange(prod(sh), dtype=dtype).reshape(sh)

        assert_equal(dpnp.flip(dp_a, axis=axis), numpy.flip(np_a, axis=axis))

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_lr_equivalent(self, dtype):
        dp_a = dpnp.arange(4, dtype=dtype)
        dp_a = dpnp.add.outer(dp_a, dp_a)
        assert_equal(dpnp.flip(dp_a, 1), dpnp.fliplr(dp_a))

        np_a = numpy.arange(4, dtype=dtype)
        np_a = numpy.add.outer(np_a, np_a)
        assert_equal(dpnp.flip(dp_a, 1), numpy.flip(np_a, 1))

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_ud_equivalent(self, dtype):
        dp_a = dpnp.arange(4, dtype=dtype)
        dp_a = dpnp.add.outer(dp_a, dp_a)
        assert_equal(dpnp.flip(dp_a, 0), dpnp.flipud(dp_a))

        np_a = numpy.arange(4, dtype=dtype)
        np_a = numpy.add.outer(np_a, np_a)
        assert_equal(dpnp.flip(dp_a, 0), numpy.flip(np_a, 0))

    @pytest.mark.parametrize(
        "x, axis",
        [
            pytest.param(dpnp.ones(4), 1, id="1-d, axis=1"),
            pytest.param(dpnp.ones((4, 4)), 2, id="2-d, axis=2"),
            pytest.param(dpnp.ones((4, 4)), -3, id="2-d, axis=-3"),
            pytest.param(dpnp.ones((4, 4)), (0, 3), id="2-d, axis=(0, 3)"),
        ],
    )
    def test_axes(self, x, axis):
        with pytest.raises(AxisError):
            dpnp.flip(x, axis=axis)


class TestFliplr:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_arange(self, dtype):
        sh = (2, 3) if dtype != dpnp.bool else (1, 1)
        dp_a = dpnp.arange(prod(sh), dtype=dtype).reshape(sh)
        np_a = numpy.arange(prod(sh), dtype=dtype).reshape(sh)

        assert_equal(dpnp.fliplr(dp_a), numpy.fliplr(np_a))

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_equivalent(self, dtype):
        dp_a = dpnp.arange(4, dtype=dtype)
        dp_a = dp_a[:, dpnp.newaxis] + dp_a[dpnp.newaxis, :]
        assert_equal(dpnp.fliplr(dp_a), dp_a[:, ::-1])

        np_a = numpy.arange(4, dtype=dtype)
        np_a = numpy.add.outer(np_a, np_a)
        assert_equal(dpnp.fliplr(dp_a), numpy.fliplr(np_a))

    @pytest.mark.parametrize(
        "val",
        [-1.2, numpy.arange(7), [2, 7, 3.6], (-3, 4), range(4)],
        ids=["scalar", "numpy.array", "list", "tuple", "range"],
    )
    def test_raises_array_type(self, val):
        with pytest.raises(
            TypeError, match="An array must be any of supported type, but got"
        ):
            dpnp.fliplr(val)

    def test_raises_1d(self):
        a = dpnp.ones(4)
        with pytest.raises(ValueError, match="Input must be >= 2-d, but got"):
            dpnp.fliplr(a)


class TestFlipud:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_arange(self, dtype):
        sh = (2, 3) if dtype != dpnp.bool else (1, 1)
        dp_a = dpnp.arange(prod(sh), dtype=dtype).reshape(sh)
        np_a = numpy.arange(prod(sh), dtype=dtype).reshape(sh)

        assert_equal(dpnp.flipud(dp_a), numpy.flipud(np_a))

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_equivalent(self, dtype):
        dp_a = dpnp.arange(4, dtype=dtype)
        dp_a = dp_a[:, dpnp.newaxis] + dp_a[dpnp.newaxis, :]
        assert_equal(dpnp.flipud(dp_a), dp_a[::-1, :])

        np_a = numpy.arange(4, dtype=dtype)
        np_a = numpy.add.outer(np_a, np_a)
        assert_equal(dpnp.flipud(dp_a), numpy.flipud(np_a))

    @pytest.mark.parametrize(
        "val",
        [3.4, numpy.arange(6), [2, -1.7, 6], (-2, 4), range(5)],
        ids=["scalar", "numpy.array", "list", "tuple", "range"],
    )
    def test_raises_array_type(self, val):
        with pytest.raises(
            TypeError, match="An array must be any of supported type, but got"
        ):
            dpnp.flipud(val)

    def test_raises_0d(self):
        a = dpnp.array(3)
        with pytest.raises(ValueError, match="Input must be >= 1-d, but got"):
            dpnp.flipud(a)
