import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp

from .helper import assert_dtype_allclose, get_all_dtypes, get_complex_dtypes


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


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("side", ["left", "right"], ids=['"left"', '"right"'])
@pytest.mark.parametrize(
    "v_",
    [
        [[3, 4], [2, 1]],
        [[1, 0], [3, 0]],
        [[3, 2, 1, 6]],
        [[4, 2], [3, 3], [4, 1]],
        [[1, -3, 3], [0, 5, 2], [0, 1, 1], [0, 0, 1]],
        [
            [[[8, 2], [3, 0]], [[5, 2], [0, 1]]],
            [[[1, 3], [3, 1]], [[5, 2], [0, 1]]],
        ],
    ],
    ids=[
        "[[3, 4], [2, 1]]",
        "[[1, 0], [3, 0]]",
        "[[3, 2, 1, 6]]",
        "[[4, 2], [3, 3], [4, 1]]",
        "[[1, -3, 3], [0, 5, 2], [0, 1, 1], [0, 0, 1]]",
        "[[[[8, 2], [3, 0]], [[5, 2], [0, 1]]], [[[1, 3], [3, 1]], [[5, 2], [0, 1]]]]",
    ],
)
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_none=True, no_bool=True, no_complex=True)
)
@pytest.mark.parametrize(
    "array",
    [
        [1, 2, 3, 4],
        [-5, -1, 0, 3, 17, 100],
        [1, 0, 3, 0],
        [3, 2, 1, 6],
        [4, 2, 3, 3, 4, 1],
        [1, -3, 3, 0, 5, 2, 0, 1, 1, 0, 0, 1],
        [8, 2, 3, 0, 5, 2, 0, 1, 1, 3, 3, 1, 5, 2, 0, 1],
    ],
    ids=[
        "[1, 2, 3, 4]",
        "[-5, -1, 0, 3, 17, 100]",
        "[1, 0, 3, 0]",
        "[3, 2, 1, 6]",
        "[4, 2, 3, 3, 4, 1]",
        "[1, -3, 3, 0, 5, 2, 0, 1, 1, 0, 0, 1]",
        "[8, 2, 3, 0, 5, 2, 0, 1, 1, 3, 3, 1, 5, 2, 0, 1]",
    ],
)
def test_searchsorted(array, dtype, v_, side):
    a = numpy.array(array, dtype)
    ia = dpnp.array(array, dtype)
    v = numpy.array(v_, dtype)
    iv = dpnp.array(v_, dtype)
    expected = numpy.searchsorted(a, v, side=side)
    result = dpnp.searchsorted(ia, iv, side=side)
    assert_array_equal(expected, result)
