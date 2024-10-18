import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from numpy.testing import (
    assert_allclose,
    assert_equal,
    assert_raises,
)

import dpnp

from .helper import (
    get_all_dtypes,
    get_float_dtypes,
)


class TestCountNonZero:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("size", [2, 4, 8, 16, 3, 9, 27, 81])
    def test_basic(self, dtype, size):
        if dtype != dpnp.bool:
            a = numpy.arange(size, dtype=dtype)
        else:
            a = numpy.resize(numpy.arange(2, dtype=dtype), size)

        for i in range(int(size / 2)):
            a[(i * (int(size / 3) - 1)) % size] = 0

        ia = dpnp.array(a)

        result = dpnp.count_nonzero(ia)
        expected = numpy.count_nonzero(a)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("data", [[], [0], [1]])
    def test_trivial(self, data):
        a = numpy.array(data)
        ia = dpnp.array(a)

        result = dpnp.count_nonzero(ia)
        expected = numpy.count_nonzero(a)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("data", [[], [0], [1]])
    def test_trivial_boolean_dtype(self, data):
        a = numpy.array(data, dtype="?")
        ia = dpnp.array(a)

        result = dpnp.count_nonzero(ia)
        expected = numpy.count_nonzero(a)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_axis_basic(self, axis):
        a = numpy.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])
        ia = dpnp.array(a)

        result = dpnp.count_nonzero(ia, axis=axis)
        expected = numpy.count_nonzero(a, axis=axis)
        assert_equal(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_axis_raises(self, xp):
        a = xp.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        assert_raises(ValueError, xp.count_nonzero, a, axis=(1, 1))
        assert_raises(TypeError, xp.count_nonzero, a, axis="foo")
        assert_raises(AxisError, xp.count_nonzero, a, axis=3)

        # different exception type in numpy and dpnp
        with pytest.raises((ValueError, TypeError)):
            xp.count_nonzero(a, axis=xp.array([[1], [2]]))

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [0, 1, (0, 1), None])
    def test_axis_all_dtypes(self, dt, axis):
        a = numpy.zeros((3, 3), dtype=dt)
        a[0, 0] = a[1, 0] = 1
        ia = dpnp.array(a)

        result = dpnp.count_nonzero(ia, axis=axis)
        expected = numpy.count_nonzero(a, axis=axis)
        assert_equal(result, expected)

    def test_axis_empty(self):
        axis = ()
        a = numpy.array([[0, 0, 1], [1, 0, 1]])
        ia = dpnp.array(a)

        result = dpnp.count_nonzero(ia, axis=axis)
        expected = numpy.count_nonzero(a, axis=axis)
        assert_equal(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_keepdims(self, axis):
        a = numpy.array([[0, 0, 1, 0], [0, 3, 5, 0], [7, 9, 2, 0]])
        ia = dpnp.array(a)

        result = dpnp.count_nonzero(ia, axis=axis, keepdims=True)
        expected = numpy.count_nonzero(a, axis=axis, keepdims=True)
        assert_equal(result, expected)

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_out(self, dt):
        a = numpy.array([[0, 1, 0], [2, 0, 3]], dtype=dt)
        ia = dpnp.array(a)
        iout = dpnp.empty_like(ia, shape=ia.shape[1], dtype=dpnp.intp)

        result = dpnp.count_nonzero(ia, axis=0, out=iout)
        expected = numpy.count_nonzero(a, axis=0)  # no out keyword
        assert_equal(result, expected)
        assert result is iout

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_out_floating_dtype(self, dt):
        a = dpnp.array([[0, 1, 0], [2, 0, 3]])
        out = dpnp.empty_like(a, shape=a.shape[1], dtype=dt)
        assert_raises(ValueError, dpnp.count_nonzero, a, axis=0, out=out)

    def test_array_method(self):
        a = numpy.array([[1, 0, 0], [4, 0, 6]])
        ia = dpnp.array(a)
        assert_equal(ia.nonzero(), a.nonzero())
