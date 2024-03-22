import numpy
import pytest
from numpy.testing import assert_array_equal, assert_equal, assert_raises

import dpnp

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_dtypes,
)


class TestArgsort:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_argsort_dtype(self, dtype):
        a = numpy.random.uniform(-5, 5, 10)
        np_array = numpy.array(a, dtype=dtype)
        dp_array = dpnp.array(np_array)

        result = dpnp.argsort(dp_array, kind="stable")
        expected = numpy.argsort(np_array, kind="stable")
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_argsort_complex(self, dtype):
        a = numpy.random.uniform(-5, 5, 10)
        b = numpy.random.uniform(-5, 5, 10)
        np_array = numpy.array(a + b * 1j, dtype=dtype)
        dp_array = dpnp.array(np_array)

        result = dpnp.argsort(dp_array)
        expected = numpy.argsort(np_array)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, -2, -1, 0, 1, 2])
    def test_argsort_axis(self, axis):
        a = numpy.random.uniform(-10, 10, 36)
        np_array = numpy.array(a).reshape(3, 4, 3)
        dp_array = dpnp.array(np_array)

        result = dpnp.argsort(dp_array, axis=axis)
        expected = numpy.argsort(np_array, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("axis", [None, -2, -1, 0, 1])
    def test_argsort_ndarray(self, dtype, axis):
        a = numpy.random.uniform(-10, 10, 12)
        np_array = numpy.array(a, dtype=dtype).reshape(6, 2)
        dp_array = dpnp.array(np_array)

        result = dp_array.argsort(axis=axis)
        expected = np_array.argsort(axis=axis)
        assert_dtype_allclose(result, expected)

    def test_argsort_stable(self):
        np_array = numpy.repeat(numpy.arange(10), 10)
        dp_array = dpnp.array(np_array)

        result = dpnp.argsort(dp_array, kind="stable")
        expected = numpy.argsort(np_array, kind="stable")
        assert_dtype_allclose(result, expected)

    def test_argsort_zero_dim(self):
        np_array = numpy.array(2.5)
        dp_array = dpnp.array(np_array)

        # with default axis=-1
        with pytest.raises(numpy.AxisError):
            dpnp.argsort(dp_array)

        # with axis = None
        result = dpnp.argsort(dp_array, axis=None)
        expected = numpy.argsort(np_array, axis=None)
        assert_dtype_allclose(result, expected)

    def test_sort_notimplemented(self):
        dp_array = dpnp.arange(10)

        with pytest.raises(NotImplementedError):
            dpnp.argsort(dp_array, kind="quicksort")

        with pytest.raises(NotImplementedError):
            dpnp.argsort(dp_array, order=["age"])


class TestSearchSorted:
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("dtype", get_float_dtypes(no_float16=False))
    def test_nans_float(self, side, dtype):
        a = numpy.array([0, 1, numpy.nan], dtype=dtype)
        dp_a = dpnp.array(a)

        result = dp_a.searchsorted(dp_a, side=side)
        expected = a.searchsorted(a, side=side)
        assert_equal(result, expected)

        result = dpnp.searchsorted(dp_a, dp_a[-1], side=side)
        expected = numpy.searchsorted(a, a[-1], side=side)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_nans_complex(self, side, dtype):
        a = numpy.zeros(9, dtype=dtype)
        a.real += [0, 0, 1, 1, 0, 1, numpy.nan, numpy.nan, numpy.nan]
        a.imag += [0, 1, 0, 1, numpy.nan, numpy.nan, 0, 1, numpy.nan]
        dp_a = dpnp.array(a)

        result = dp_a.searchsorted(dp_a, side=side)
        expected = a.searchsorted(a, side=side)
        assert_equal(result, expected)

    @pytest.mark.parametrize("n", range(3))
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_n_elements(self, n, side):
        a = numpy.ones(n)
        dp_a = dpnp.array(a)

        v = numpy.array([0, 1, 2])
        dp_v = dpnp.array(v)

        result = dp_a.searchsorted(dp_v, side=side)
        expected = a.searchsorted(v, side=side)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_smart_resetting(self, side):
        a = numpy.arange(5)
        dp_a = dpnp.array(a)

        v = numpy.array([6, 5, 4])
        dp_v = dpnp.array(v)

        result = dp_a.searchsorted(dp_v, side=side)
        expected = a.searchsorted(v, side=side)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
    def test_type_specific(self, side, dtype):
        if dtype == numpy.bool_:
            a = numpy.arange(2, dtype=dtype)
        else:
            a = numpy.arange(0, 5, dtype=dtype)
        dp_a = dpnp.array(a)

        result = dp_a.searchsorted(dp_a, side=side)
        expected = a.searchsorted(a, side=side)
        assert_equal(result, expected)

        e = numpy.ndarray(shape=0, buffer=b"", dtype=dtype)
        dp_e = dpnp.array(e)

        result = dp_e.searchsorted(dp_a, side=side)
        expected = e.searchsorted(a, side=side)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_sorter(self, dtype):
        a = numpy.random.rand(300).astype(dtype)
        s = a.argsort()
        k = numpy.linspace(0, 1, 20, dtype=dtype)

        dp_a = dpnp.array(a)
        dp_s = dpnp.array(s)
        dp_k = dpnp.array(k)

        result = dp_a.searchsorted(dp_k, sorter=dp_s)
        expected = a.searchsorted(k, sorter=s)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_sorter_with_side(self, side):
        a = numpy.array([0, 1, 2, 3, 5] * 20)
        s = a.argsort()
        k = [0, 1, 2, 3, 5]

        dp_a = dpnp.array(a)
        dp_s = dpnp.array(s)
        dp_k = dpnp.array(k)

        result = dp_a.searchsorted(dp_k, side=side, sorter=dp_s)
        expected = a.searchsorted(k, side=side, sorter=s)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
    def test_sorter_type_specific(self, side, dtype):
        if dtype == numpy.bool_:
            a = numpy.array([1, 0], dtype=dtype)
            # a sorter array to be of a type that is different
            # from np.intp in all platforms
            s = numpy.array([1, 0], dtype=numpy.int16)
        else:
            a = numpy.arange(0, 5, dtype=dtype)
            # a sorter array to be of a type that is different
            # from np.intp in all platforms
            s = numpy.array([4, 2, 3, 0, 1], dtype=numpy.int16)

        dp_a = dpnp.array(a)
        dp_s = dpnp.array(s)

        result = dp_a.searchsorted(dp_a, side, dp_s)
        expected = a.searchsorted(a, side, s)
        assert_equal(result, expected)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_sorter_non_contiguous(self, side):
        a = numpy.array([3, 4, 1, 2, 0])
        srt = numpy.empty((10,), dtype=numpy.intp)
        srt[1::2] = -1
        srt[::2] = [4, 2, 3, 0, 1]
        s = srt[::2]

        dp_a = dpnp.array(a)
        dp_s = dpnp.array(s)

        result = dp_a.searchsorted(dp_a, side=side, sorter=dp_s)
        expected = a.searchsorted(a, side=side, sorter=s)
        assert_equal(result, expected)

    def test_invalid_sorter(self):
        for xp in [dpnp, numpy]:
            a = xp.array([5, 2, 1, 3, 4])

            assert_raises(
                TypeError,
                ValueError,
                xp.searchsorted,
                a,
                0,
                sorter=xp.array([1.1]),
            )
            assert_raises(
                ValueError, xp.searchsorted, a, 0, sorter=xp.array([1, 2, 3, 4])
            )
            assert_raises(
                ValueError,
                xp.searchsorted,
                a,
                0,
                sorter=xp.array([1, 2, 3, 4, 5, 6]),
            )

    def test_v_scalar(self):
        v = 0
        a = numpy.array([-8, -5, -1, 3, 6, 10])
        dp_a = dpnp.array(a)

        result = dpnp.searchsorted(dp_a, v)
        expected = numpy.searchsorted(a, v)
        assert_equal(result, expected)


class TestSort:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_sort_dtype(self, dtype):
        a = numpy.random.uniform(-5, 5, 10)
        np_array = numpy.array(a, dtype=dtype)
        dp_array = dpnp.array(np_array)

        result = dpnp.sort(dp_array)
        expected = numpy.sort(np_array)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_sort_complex(self, dtype):
        a = numpy.random.uniform(-5, 5, 10)
        b = numpy.random.uniform(-5, 5, 10)
        np_array = numpy.array(a + b * 1j, dtype=dtype)
        dp_array = dpnp.array(np_array)

        result = dpnp.sort(dp_array)
        expected = numpy.sort(np_array)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, -2, -1, 0, 1, 2])
    def test_sort_axis(self, axis):
        a = numpy.random.uniform(-10, 10, 36)
        np_array = numpy.array(a).reshape(3, 4, 3)
        dp_array = dpnp.array(np_array)

        result = dpnp.sort(dp_array, axis=axis)
        expected = numpy.sort(np_array, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("axis", [-2, -1, 0, 1])
    def test_sort_ndarray(self, dtype, axis):
        a = numpy.random.uniform(-10, 10, 12)
        np_array = numpy.array(a, dtype=dtype).reshape(6, 2)
        dp_array = dpnp.array(np_array)

        dp_array.sort(axis=axis)
        np_array.sort(axis=axis)
        assert_dtype_allclose(dp_array, np_array)

    def test_sort_stable(self):
        np_array = numpy.repeat(numpy.arange(10), 10)
        dp_array = dpnp.array(np_array)

        result = dpnp.sort(dp_array, kind="stable")
        expected = numpy.sort(np_array, kind="stable")
        assert_dtype_allclose(result, expected)

    def test_sort_ndarray_axis_none(self):
        a = numpy.random.uniform(-10, 10, 12)
        dp_array = dpnp.array(a).reshape(6, 2)
        with pytest.raises(TypeError):
            dp_array.sort(axis=None)

    def test_sort_zero_dim(self):
        np_array = numpy.array(2.5)
        dp_array = dpnp.array(np_array)

        # with default axis=-1
        with pytest.raises(numpy.AxisError):
            dpnp.sort(dp_array)

        # with axis = None
        result = dpnp.sort(dp_array, axis=None)
        expected = numpy.sort(np_array, axis=None)
        assert_dtype_allclose(result, expected)

    def test_sort_notimplemented(self):
        dp_array = dpnp.arange(10)

        with pytest.raises(NotImplementedError):
            dpnp.sort(dp_array, kind="quicksort")

        with pytest.raises(NotImplementedError):
            dpnp.sort(dp_array, order=["age"])


@pytest.mark.parametrize("kth", [0, 1], ids=["0", "1"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
@pytest.mark.parametrize(
    "array",
    [
        [3, 4, 2, 1],
        [[1, 0], [3, 0]],
        [[3, 2], [1, 6]],
        [[4, 2, 3], [3, 4, 1]],
        [[[1, -3], [3, 0]], [[5, 2], [0, 1]], [[1, 0], [0, 1]]],
        [
            [[[8, 2], [3, 0]], [[5, 2], [0, 1]]],
            [[[1, 3], [3, 1]], [[5, 2], [0, 1]]],
        ],
    ],
    ids=[
        "[3, 4, 2, 1]",
        "[[1, 0], [3, 0]]",
        "[[3, 2], [1, 6]]",
        "[[4, 2, 3], [3, 4, 1]]",
        "[[[1, -3], [3, 0]], [[5, 2], [0, 1]], [[1, 0], [0, 1]]]",
        "[[[[8, 2], [3, 0]], [[5, 2], [0, 1]]], [[[1, 3], [3, 1]], [[5, 2], [0, 1]]]]",
    ],
)
def test_partition(array, dtype, kth):
    a = dpnp.array(array, dtype)
    p = dpnp.partition(a, kth)

    assert (p[..., 0:kth] <= p[..., kth : kth + 1]).all()
    assert (p[..., kth : kth + 1] <= p[..., kth + 1 :]).all()
