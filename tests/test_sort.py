import pytest
from .helper import get_all_dtypes

import dpnp

import numpy
from numpy.testing import assert_array_equal


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

    # TODO: remove once dpnp.less_equal() support complex types
    p = p.asnumpy()

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
    "dtype",
    [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
    ids=["float64", "float32", "int64", "int32"],
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
    assert_array_equal(expected, result)
