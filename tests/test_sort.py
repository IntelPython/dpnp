import pytest
from .helper import get_all_dtypes

import dpnp

import numpy


@pytest.mark.parametrize("kth", [0, 1], ids=["0", "1"])
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
)
@pytest.mark.parametrize(
    "array",
    [
        [3, 4, 2, 1],
        [[1, 0], [3, 0]],
        [[3, 2], [1, 6]],
        [[4, 2, 3], [3, 4, 1]],
        [[[1, -3], [3, 0]], [[5, 2], [0, 1]], [[1, 0], [0, 1]]],
        [[[[8, 2], [3, 0]], [[5, 2], [0, 1]]], [[[1, 3], [3, 1]], [[5, 2], [0, 1]]]],
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
    a = numpy.array(array, dtype)
    ia = dpnp.array(array, dtype)
    expected = numpy.partition(a, kth)
    result = dpnp.partition(ia, kth)
    numpy.testing.assert_array_equal(expected, result)


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
        [[[[8, 2], [3, 0]], [[5, 2], [0, 1]]], [[[1, 3], [3, 1]], [[5, 2], [0, 1]]]],
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
    "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
)
@pytest.mark.parametrize(
    "array",
    [[1, 2, 3, 4], [-5, -1, 0, 3, 17, 100]],
    ids=[
        "[1, 2, 3, 4]",
        "[-5, -1, 0, 3, 17, 100]"
        # '[1, 0, 3, 0]',
        # '[3, 2, 1, 6]',
        # '[4, 2, 3, 3, 4, 1]',
        # '[1, -3, 3, 0, 5, 2, 0, 1, 1, 0, 0, 1]',
        # '[8, 2, 3, 0, 5, 2, 0, 1, 1, 3, 3, 1, 5, 2, 0, 1]'
    ],
)
def test_searchsorted(array, dtype, v_, side):
    a = numpy.array(array, dtype)
    ia = dpnp.array(array, dtype)
    v = numpy.array(v_, dtype)
    iv = dpnp.array(v_, dtype)
    expected = numpy.searchsorted(a, v, side=side)
    result = dpnp.searchsorted(ia, iv, side=side)
    numpy.testing.assert_array_equal(expected, result)
