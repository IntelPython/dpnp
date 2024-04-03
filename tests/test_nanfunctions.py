import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
)

import dpnp
from dpnp.dpnp_array import dpnp_array
from tests.third_party.cupy import testing

from .helper import (
    assert_dtype_allclose,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
)


@testing.parameterize(
    *testing.product(
        {
            "func": ("nancumsum", "nancumprod"),
        }
    )
)
class TestNanCumSumProd:
    @pytest.fixture(autouse=True)
    def setUp(self):
        if self.func == "nancumprod":
            pytest.skip("nancumprod() is not implemented yet")
        pass

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize(
        "array",
        [numpy.array(numpy.nan), numpy.full((3, 3), numpy.nan)],
        ids=["0d", "2d"],
    )
    def test_allnans(self, dtype, array):
        a = numpy.array(array, dtype=dtype)
        ia = dpnp.array(a, dtype=dtype)

        result = getattr(dpnp, self.func)(ia)
        expected = getattr(numpy, self.func)(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_empty(self, axis):
        a = numpy.zeros((0, 3))
        ia = dpnp.array(a)

        result = getattr(dpnp, self.func)(ia, axis=axis)
        expected = getattr(numpy, self.func)(a, axis=axis)
        assert_equal(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_keepdims(self, axis):
        a = numpy.eye(3)
        ia = dpnp.array(a)

        result = getattr(dpnp, self.func)(ia, axis=axis, out=None)
        expected = getattr(numpy, self.func)(a, axis=axis, out=None)
        assert_equal(result, expected)
        assert result.ndim == expected.ndim

    @pytest.mark.parametrize("axis", [None] + list(range(4)))
    def test_keepdims_random(self, axis):
        a = numpy.ones((3, 5, 7, 11))
        # Randomly set some elements to NaN:
        rs = numpy.random.RandomState(0)
        a[rs.rand(*a.shape) < 0.5] = numpy.nan
        ia = dpnp.array(a)

        result = getattr(dpnp, self.func)(ia, axis=axis)
        expected = getattr(numpy, self.func)(a, axis=axis)
        assert_equal(result, expected)

    @pytest.mark.parametrize("axis", [-2, -1, 0, 1, None])
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_ndat_ones(self, axis, dtype):
        a = numpy.array(
            [
                [0.6244, 1.0, 0.2692, 0.0116, 1.0, 0.1170],
                [0.5351, -0.9403, 1.0, 0.2100, 0.4759, 0.2833],
                [1.0, 1.0, 1.0, 0.1042, 1.0, -0.5954],
                [0.1610, 1.0, 1.0, 0.1859, 0.3146, 1.0],
            ]
        )
        a = a.astype(dtype=dtype)
        ia = dpnp.array(a)

        result = getattr(dpnp, self.func)(ia, axis=axis)
        expected = getattr(numpy, self.func)(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [-2, -1, 0, 1, None])
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_ndat_zeros(self, axis, dtype):
        a = numpy.array(
            [
                [0.6244, 0.0, 0.2692, 0.0116, 0.0, 0.1170],
                [0.5351, -0.9403, 0.0, 0.2100, 0.4759, 0.2833],
                [0.0, 0.0, 0.0, 0.1042, 0.0, -0.5954],
                [0.1610, 0.0, 0.0, 0.1859, 0.3146, 0.0],
            ]
        )
        a = a.astype(dtype=dtype)
        ia = dpnp.array(a)

        result = getattr(dpnp, self.func)(ia, axis=axis)
        expected = getattr(numpy, self.func)(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [-2, -1, 0, 1])
    def test_out(self, axis):
        a = numpy.eye(3)
        out = numpy.eye(3)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = getattr(dpnp, self.func)(ia, axis=axis, out=iout)
        expected = getattr(numpy, self.func)(a, axis=axis, out=out)
        assert_almost_equal(result, expected)
        assert result is iout


class TestNanSum:
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_nansum(self, dtype, axis, keepdims):
        dp_array = dpnp.array([[dpnp.nan, 1, 2], [3, dpnp.nan, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.nansum(np_array, axis=axis, keepdims=keepdims)
        result = dpnp.nansum(dp_array, axis=axis, keepdims=keepdims)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_nansum_complex(self, dtype):
        x1 = numpy.random.rand(10)
        x2 = numpy.random.rand(10)
        a = numpy.array(x1 + 1j * x2, dtype=dtype)
        a[::3] = numpy.nan
        ia = dpnp.array(a)

        expected = numpy.nansum(a)
        result = dpnp.nansum(ia)

        # use only type kinds check when dpnp handles complex64 arrays
        # since `dpnp.sum()` and `numpy.sum()` return different dtypes
        assert_dtype_allclose(
            result, expected, check_only_type_kind=(dtype == dpnp.complex64)
        )

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("axis", [0, 1])
    def test_nansum_out(self, dtype, axis):
        dp_array = dpnp.array([[dpnp.nan, 1, 2], [3, dpnp.nan, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.nansum(np_array, axis=axis)
        out = dpnp.empty_like(dpnp.asarray(expected))
        result = dpnp.nansum(dp_array, axis=axis, out=out)
        assert out is result
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_nansum_dtype(self, dtype):
        dp_array = dpnp.array([[dpnp.nan, 1, 2], [3, dpnp.nan, 0]])
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.nansum(np_array, dtype=dtype)
        result = dpnp.nansum(dp_array, dtype=dtype)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_nansum_strided(self, dtype):
        dp_array = dpnp.arange(20, dtype=dtype)
        dp_array[::3] = dpnp.nan
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.nansum(dp_array[::-1])
        expected = numpy.nansum(np_array[::-1])
        assert_allclose(result, expected)

        result = dpnp.nansum(dp_array[::2])
        expected = numpy.nansum(np_array[::2])
        assert_allclose(result, expected)
