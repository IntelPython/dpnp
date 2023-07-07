import numpy
import pytest
from numpy.testing import assert_, assert_array_equal, assert_equal

import dpnp

from .helper import get_all_dtypes, has_support_aspect64


class TestIndexing:
    def test_ellipsis_index(self):
        a = dpnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_(a[...] is not a)
        assert_equal(a[...], a)

        # test that slicing with ellipsis doesn't skip an arbitrary number of dimensions
        assert_equal(a[0, ...], a[0])
        assert_equal(a[0, ...], a[0, :])
        assert_equal(a[..., 0], a[:, 0])

        # test that slicing with ellipsis always results in an array
        assert_equal(a[0, ..., 1], dpnp.array(2))

        # assignment with `(Ellipsis,)` on 0-d arrays
        b = dpnp.array(1)
        b[(Ellipsis,)] = 2
        assert_equal(b, 2)

    def test_boolean_indexing_list(self):
        a = dpnp.array([1, 2, 3])
        b = dpnp.array([True, False, True])

        assert_equal(a[b], [1, 3])
        assert_equal(a[None, b], [[1, 3]])

    def test_indexing_array_weird_strides(self):
        np_x = numpy.ones(10)
        dp_x = dpnp.ones(10)

        np_ind = numpy.arange(10)[:, None, None, None]
        np_ind = numpy.broadcast_to(np_ind, (10, 55, 4, 4))

        dp_ind = dpnp.arange(10)[:, None, None, None]
        dp_ind = dpnp.broadcast_to(dp_ind, (10, 55, 4, 4))

        # single advanced index case
        assert_array_equal(dp_x[dp_ind], np_x[np_ind])

        np_x2 = numpy.ones((10, 2))
        dp_x2 = dpnp.ones((10, 2))

        np_zind = numpy.zeros(4, dtype=np_ind.dtype)
        dp_zind = dpnp.zeros(4, dtype=dp_ind.dtype)

        # higher dimensional advanced index
        assert_array_equal(dp_x2[dp_ind, dp_zind], np_x2[np_ind, np_zind])

    def test_indexing_array_negative_strides(self):
        arr = dpnp.zeros((4, 4))[::-1, ::-1]

        slices = (slice(None), dpnp.array([0, 1, 2, 3]))
        arr[slices] = 10
        assert_array_equal(arr, 10.0)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_choose():
    a = numpy.r_[:4]
    ia = dpnp.array(a)
    b = numpy.r_[-4:0]
    ib = dpnp.array(b)
    c = numpy.r_[100:500:100]
    ic = dpnp.array(c)

    expected = numpy.choose([0, 0, 0, 0], [a, b, c])
    result = dpnp.choose([0, 0, 0, 0], [ia, ib, ic])
    assert_array_equal(expected, result)


@pytest.mark.parametrize("offset", [0, 1], ids=["0", "1"])
@pytest.mark.parametrize(
    "array",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
        [
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
            [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
        ],
        [
            [[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]],
            [[[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]],
        ],
        [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
        ],
    ],
    ids=[
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
        "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]",
        "[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]",
        "[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]",
        "[[[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]], [[[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]]]",
        "[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]]",
    ],
)
def test_diagonal(array, offset):
    a = numpy.array(array)
    ia = dpnp.array(a)
    expected = numpy.diagonal(a, offset)
    result = dpnp.diagonal(ia, offset)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("arr_dtype", get_all_dtypes())
@pytest.mark.parametrize("cond_dtype", get_all_dtypes())
def test_extract_1d(arr_dtype, cond_dtype):
    a = numpy.array([-2, -1, 0, 1, 2, 3], dtype=arr_dtype)
    ia = dpnp.array(a)
    cond = numpy.array([1, -1, 2, 0, -2, 3], dtype=cond_dtype)
    icond = dpnp.array(cond)
    expected = numpy.extract(cond, a)
    result = dpnp.extract(icond, ia)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("val", [-1, 0, 1], ids=["-1", "0", "1"])
@pytest.mark.parametrize(
    "array",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        [
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
            [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
        ],
    ],
    ids=[
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
        "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]",
        "[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]",
    ],
)
def test_fill_diagonal(array, val):
    a = numpy.array(array)
    ia = dpnp.array(a)
    expected = numpy.fill_diagonal(a, val)
    result = dpnp.fill_diagonal(ia, val)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "dimension",
    [(1,), (2,), (1, 2), (2, 3), (3, 2), [1], [2], [1, 2], [2, 3], [3, 2]],
    ids=[
        "(1, )",
        "(2, )",
        "(1, 2)",
        "(2, 3)",
        "(3, 2)",
        "[1]",
        "[2]",
        "[1, 2]",
        "[2, 3]",
        "[3, 2]",
    ],
)
def test_indices(dimension):
    expected = numpy.indices(dimension)
    result = dpnp.indices(dimension)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "array",
    [
        [],
        [[0, 0], [0, 0]],
        [[1, 0], [1, 0]],
        [[1, 2], [3, 4]],
        [[0, 1, 2], [3, 0, 5], [6, 7, 0]],
        [[0, 1, 0, 3, 0], [5, 0, 7, 0, 9]],
        [[[1, 2], [0, 4]], [[0, 2], [0, 1]], [[0, 0], [3, 1]]],
        [
            [[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]],
            [[[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]],
        ],
    ],
    ids=[
        "[]",
        "[[0, 0], [0, 0]]",
        "[[1, 0], [1, 0]]",
        "[[1, 2], [3, 4]]",
        "[[0, 1, 2], [3, 0, 5], [6, 7, 0]]",
        "[[0, 1, 0, 3, 0], [5, 0, 7, 0, 9]]",
        "[[[1, 2], [0, 4]], [[0, 2], [0, 1]], [[0, 0], [3, 1]]]",
        "[[[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]], [[[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]]]",
    ],
)
def test_nonzero(array):
    a = numpy.array(array)
    ia = dpnp.array(array)
    expected = numpy.nonzero(a)
    result = dpnp.nonzero(ia)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "vals", [[100, 200], (100, 200)], ids=["[100, 200]", "(100, 200)"]
)
@pytest.mark.parametrize(
    "mask",
    [
        [[True, False], [False, True]],
        [[False, True], [True, False]],
        [[False, False], [True, True]],
    ],
    ids=[
        "[[True, False], [False, True]]",
        "[[False, True], [True, False]]",
        "[[False, False], [True, True]]",
    ],
)
@pytest.mark.parametrize(
    "arr",
    [[[0, 0], [0, 0]], [[1, 2], [1, 2]], [[1, 2], [3, 4]]],
    ids=["[[0, 0], [0, 0]]", "[[1, 2], [1, 2]]", "[[1, 2], [3, 4]]"],
)
def test_place1(arr, mask, vals):
    a = numpy.array(arr)
    ia = dpnp.array(a)
    m = numpy.array(mask)
    im = dpnp.array(m)
    iv = dpnp.array(vals)
    numpy.place(a, m, vals)
    dpnp.place(ia, im, iv)
    assert_array_equal(a, ia)


@pytest.mark.parametrize(
    "vals",
    [
        [100, 200],
        [100, 200, 300, 400, 500, 600],
        [100, 200, 300, 400, 500, 600, 800, 900],
    ],
    ids=[
        "[100, 200]",
        "[100, 200, 300, 400, 500, 600]",
        "[100, 200, 300, 400, 500, 600, 800, 900]",
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
            [[False, False], [True, True]],
        ]
    ],
    ids=[
        "[[[True, False], [False, True]], [[False, True], [True, False]], [[False, False], [True, True]]]"
    ],
)
@pytest.mark.parametrize(
    "arr",
    [[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]],
    ids=["[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]"],
)
def test_place2(arr, mask, vals):
    a = numpy.array(arr)
    ia = dpnp.array(a)
    m = numpy.array(mask)
    im = dpnp.array(m)
    iv = dpnp.array(vals)
    numpy.place(a, m, vals)
    dpnp.place(ia, im, iv)
    assert_array_equal(a, ia)


@pytest.mark.parametrize(
    "vals",
    [
        [100, 200],
        [100, 200, 300, 400, 500, 600],
        [100, 200, 300, 400, 500, 600, 800, 900],
    ],
    ids=[
        "[100, 200]",
        "[100, 200, 300, 400, 500, 600]",
        "[100, 200, 300, 400, 500, 600, 800, 900]",
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        [
            [[[False, False], [True, True]], [[True, True], [True, True]]],
            [[[False, False], [True, True]], [[False, False], [False, False]]],
        ]
    ],
    ids=[
        "[[[[False, False], [True, True]], [[True, True], [True, True]]], [[[False, False], [True, True]], [[False, False], [False, False]]]]"
    ],
)
@pytest.mark.parametrize(
    "arr",
    [
        [
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
            [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
        ]
    ],
    ids=[
        "[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]"
    ],
)
def test_place3(arr, mask, vals):
    a = numpy.array(arr)
    ia = dpnp.array(a)
    m = numpy.array(mask)
    im = dpnp.array(m)
    iv = dpnp.array(vals)
    numpy.place(a, m, vals)
    dpnp.place(ia, im, iv)
    assert_array_equal(a, ia)


@pytest.mark.parametrize("v", [0, 1, 2, 3, 4], ids=["0", "1", "2", "3", "4"])
@pytest.mark.parametrize("ind", [0, 1, 2, 3], ids=["0", "1", "2", "3"])
@pytest.mark.parametrize(
    "array",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
        [
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
            [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
        ],
    ],
    ids=[
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
        "[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]",
        "[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]",
    ],
)
def test_put(array, ind, v):
    a = numpy.array(array)
    ia = dpnp.array(a)
    numpy.put(a, ind, v)
    dpnp.put(ia, ind, v)
    assert_array_equal(a, ia)


@pytest.mark.parametrize(
    "v", [[10, 20], [30, 40]], ids=["[10, 20]", "[30, 40]"]
)
@pytest.mark.parametrize("ind", [[0, 1], [2, 3]], ids=["[0, 1]", "[2, 3]"])
@pytest.mark.parametrize(
    "array",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
        [
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
            [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
        ],
    ],
    ids=[
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
        "[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]",
        "[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]",
    ],
)
def test_put2(array, ind, v):
    a = numpy.array(array)
    ia = dpnp.array(a)
    numpy.put(a, ind, v)
    dpnp.put(ia, ind, v)
    assert_array_equal(a, ia)


def test_put3():
    a = numpy.arange(5)
    ia = dpnp.array(a)
    dpnp.put(ia, [0, 2], [-44, -55])
    numpy.put(a, [0, 2], [-44, -55])
    assert_array_equal(a, ia)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_put_along_axis_val_int():
    a = numpy.arange(16).reshape(4, 4)
    ai = dpnp.array(a)
    ind_r = numpy.array([[3, 0, 2, 1]])
    ind_r_i = dpnp.array(ind_r)
    for axis in range(2):
        numpy.put_along_axis(a, ind_r, 777, axis)
        dpnp.put_along_axis(ai, ind_r_i, 777, axis)
        assert_array_equal(a, ai)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_put_along_axis1():
    a = numpy.arange(64).reshape(4, 4, 4)
    ai = dpnp.array(a)
    ind_r = numpy.array([[[3, 0, 2, 1]]])
    ind_r_i = dpnp.array(ind_r)
    for axis in range(3):
        numpy.put_along_axis(a, ind_r, 777, axis)
        dpnp.put_along_axis(ai, ind_r_i, 777, axis)
        assert_array_equal(a, ai)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_put_along_axis2():
    a = numpy.arange(64).reshape(4, 4, 4)
    ai = dpnp.array(a)
    ind_r = numpy.array([[[3, 0, 2, 1]]])
    ind_r_i = dpnp.array(ind_r)
    for axis in range(3):
        numpy.put_along_axis(a, ind_r, [100, 200, 300, 400], axis)
        dpnp.put_along_axis(ai, ind_r_i, [100, 200, 300, 400], axis)
        assert_array_equal(a, ai)


@pytest.mark.parametrize("vals", [[100, 200]], ids=["[100, 200]"])
@pytest.mark.parametrize(
    "mask",
    [
        [[True, False], [False, True]],
        [[False, True], [True, False]],
        [[False, False], [True, True]],
    ],
    ids=[
        "[[True, False], [False, True]]",
        "[[False, True], [True, False]]",
        "[[False, False], [True, True]]",
    ],
)
@pytest.mark.parametrize(
    "arr",
    [[[0, 0], [0, 0]], [[1, 2], [1, 2]], [[1, 2], [3, 4]]],
    ids=["[[0, 0], [0, 0]]", "[[1, 2], [1, 2]]", "[[1, 2], [3, 4]]"],
)
def test_putmask1(arr, mask, vals):
    a = numpy.array(arr)
    ia = dpnp.array(a)
    m = numpy.array(mask)
    im = dpnp.array(m)
    v = numpy.array(vals)
    iv = dpnp.array(v)
    numpy.putmask(a, m, v)
    dpnp.putmask(ia, im, iv)
    assert_array_equal(a, ia)


@pytest.mark.parametrize(
    "vals",
    [
        [100, 200],
        [100, 200, 300, 400, 500, 600],
        [100, 200, 300, 400, 500, 600, 800, 900],
    ],
    ids=[
        "[100, 200]",
        "[100, 200, 300, 400, 500, 600]",
        "[100, 200, 300, 400, 500, 600, 800, 900]",
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
            [[False, False], [True, True]],
        ]
    ],
    ids=[
        "[[[True, False], [False, True]], [[False, True], [True, False]], [[False, False], [True, True]]]"
    ],
)
@pytest.mark.parametrize(
    "arr",
    [[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]],
    ids=["[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]"],
)
def test_putmask2(arr, mask, vals):
    a = numpy.array(arr)
    ia = dpnp.array(a)
    m = numpy.array(mask)
    im = dpnp.array(m)
    v = numpy.array(vals)
    iv = dpnp.array(v)
    numpy.putmask(a, m, v)
    dpnp.putmask(ia, im, iv)
    assert_array_equal(a, ia)


@pytest.mark.parametrize(
    "vals",
    [
        [100, 200],
        [100, 200, 300, 400, 500, 600],
        [100, 200, 300, 400, 500, 600, 800, 900],
    ],
    ids=[
        "[100, 200]",
        "[100, 200, 300, 400, 500, 600]",
        "[100, 200, 300, 400, 500, 600, 800, 900]",
    ],
)
@pytest.mark.parametrize(
    "mask",
    [
        [
            [[[False, False], [True, True]], [[True, True], [True, True]]],
            [[[False, False], [True, True]], [[False, False], [False, False]]],
        ]
    ],
    ids=[
        "[[[[False, False], [True, True]], [[True, True], [True, True]]], [[[False, False], [True, True]], [[False, False], [False, False]]]]"
    ],
)
@pytest.mark.parametrize(
    "arr",
    [
        [
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
            [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
        ]
    ],
    ids=[
        "[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]"
    ],
)
def test_putmask3(arr, mask, vals):
    a = numpy.array(arr)
    ia = dpnp.array(a)
    m = numpy.array(mask)
    im = dpnp.array(m)
    v = numpy.array(vals)
    iv = dpnp.array(v)
    numpy.putmask(a, m, v)
    dpnp.putmask(ia, im, iv)
    assert_array_equal(a, ia)


def test_select():
    cond_val1 = numpy.array(
        [True, True, True, False, False, False, False, False, False, False]
    )
    cond_val2 = numpy.array(
        [False, False, False, False, False, True, True, True, True, True]
    )
    icond_val1 = dpnp.array(cond_val1)
    icond_val2 = dpnp.array(cond_val2)
    condlist = [cond_val1, cond_val2]
    icondlist = [icond_val1, icond_val2]
    choice_val1 = numpy.full(10, -2)
    choice_val2 = numpy.full(10, -1)
    ichoice_val1 = dpnp.array(choice_val1)
    ichoice_val2 = dpnp.array(choice_val2)
    choicelist = [choice_val1, choice_val2]
    ichoicelist = [ichoice_val1, ichoice_val2]
    expected = numpy.select(condlist, choicelist)
    result = dpnp.select(icondlist, ichoicelist)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("array_type", get_all_dtypes(no_none=True))
@pytest.mark.parametrize(
    "indices_type", [numpy.int32, numpy.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "indices",
    [[[0, 0], [0, 0]], [[1, 2], [1, 2]], [[1, 2], [3, 4]]],
    ids=["[[0, 0], [0, 0]]", "[[1, 2], [1, 2]]", "[[1, 2], [3, 4]]"],
)
@pytest.mark.parametrize(
    "array",
    [
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
        [
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]]],
            [[[1, 3], [3, 1]], [[0, 1], [1, 3]]],
        ],
        [
            [[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]],
            [[[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]],
        ],
        [
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
        ],
    ],
    ids=[
        "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]",
        "[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]",
        "[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]",
        "[[[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]], [[[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]]]",
        "[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]]",
    ],
)
def test_take(array, indices, array_type, indices_type):
    if not has_support_aspect64() and array_type is numpy.complex64:
        pytest.skip("The device does not support result of take function")
    a = numpy.array(array, dtype=array_type)
    ind = numpy.array(indices, dtype=indices_type)
    ia = dpnp.array(a)
    iind = dpnp.array(ind)
    expected = numpy.take(a, ind)
    result = dpnp.take(ia, iind)
    assert_array_equal(expected, result)


def test_take_along_axis():
    a = numpy.arange(16).reshape(4, 4)
    ai = dpnp.array(a)
    ind_r = numpy.array([[3, 0, 2, 1]])
    ind_r_i = dpnp.array(ind_r)
    for axis in range(2):
        expected = numpy.take_along_axis(a, ind_r, axis)
        result = dpnp.take_along_axis(ai, ind_r_i, axis)
        assert_array_equal(expected, result)


def test_take_along_axis1():
    a = numpy.arange(64).reshape(4, 4, 4)
    ai = dpnp.array(a)
    ind_r = numpy.array([[[3, 0, 2, 1]]])
    ind_r_i = dpnp.array(ind_r)
    for axis in range(3):
        expected = numpy.take_along_axis(a, ind_r, axis)
        result = dpnp.take_along_axis(ai, ind_r_i, axis)
        assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "m", [None, 0, 1, 2, 3, 4], ids=["None", "0", "1", "2", "3", "4"]
)
@pytest.mark.parametrize(
    "k", [0, 1, 2, 3, 4, 5], ids=["0", "1", "2", "3", "4", "5"]
)
@pytest.mark.parametrize(
    "n", [1, 2, 3, 4, 5, 6], ids=["1", "2", "3", "4", "5", "6"]
)
def test_tril_indices(n, k, m):
    result = dpnp.tril_indices(n, k, m)
    expected = numpy.tril_indices(n, k, m)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "k", [0, 1, 2, 3, 4, 5], ids=["0", "1", "2", "3", "4", "5"]
)
@pytest.mark.parametrize(
    "array",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
    ],
    ids=["[[0, 0], [0, 0]]", "[[1, 2], [1, 2]]", "[[1, 2], [3, 4]]"],
)
def test_tril_indices_from(array, k):
    a = numpy.array(array)
    ia = dpnp.array(a)
    result = dpnp.tril_indices_from(ia, k)
    expected = numpy.tril_indices_from(a, k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "m", [None, 0, 1, 2, 3, 4], ids=["None", "0", "1", "2", "3", "4"]
)
@pytest.mark.parametrize(
    "k", [0, 1, 2, 3, 4, 5], ids=["0", "1", "2", "3", "4", "5"]
)
@pytest.mark.parametrize(
    "n", [1, 2, 3, 4, 5, 6], ids=["1", "2", "3", "4", "5", "6"]
)
def test_triu_indices(n, k, m):
    result = dpnp.triu_indices(n, k, m)
    expected = numpy.triu_indices(n, k, m)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "k", [0, 1, 2, 3, 4, 5], ids=["0", "1", "2", "3", "4", "5"]
)
@pytest.mark.parametrize(
    "array",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
    ],
    ids=["[[0, 0], [0, 0]]", "[[1, 2], [1, 2]]", "[[1, 2], [3, 4]]"],
)
def test_triu_indices_from(array, k):
    a = numpy.array(array)
    ia = dpnp.array(a)
    result = dpnp.triu_indices_from(ia, k)
    expected = numpy.triu_indices_from(a, k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("cond_dtype", get_all_dtypes())
@pytest.mark.parametrize("scalar_dtype", get_all_dtypes(no_none=True))
def test_where_with_scalars(cond_dtype, scalar_dtype):
    a = numpy.array([-1, 0, 1, 0], dtype=cond_dtype)
    ia = dpnp.array(a)

    result = dpnp.where(ia, scalar_dtype(1), scalar_dtype(0))
    expected = numpy.where(a, scalar_dtype(1), scalar_dtype(0))
    assert_array_equal(expected, result)

    result = dpnp.where(ia, ia * 2, scalar_dtype(0))
    expected = numpy.where(a, a * 2, scalar_dtype(0))
    assert_array_equal(expected, result)

    result = dpnp.where(ia, scalar_dtype(1), dpnp.array(0))
    expected = numpy.where(a, scalar_dtype(1), numpy.array(0))
    assert_array_equal(expected, result)
