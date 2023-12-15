import functools

import numpy
import pytest
from numpy.testing import (
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

import dpnp

from .helper import get_all_dtypes, get_integer_dtypes


def _add_keepdims(func):
    """
    Hack in keepdims behavior into a function taking an axis.
    """

    @functools.wraps(func)
    def wrapped(a, axis, **kwargs):
        res = func(a, axis=axis, **kwargs)
        if axis is None:
            axis = 0  # res is now 0d and we can insert this anywhere
        return dpnp.expand_dims(res, axis=axis)

    return wrapped


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


class TestPutAlongAxis:
    @pytest.mark.parametrize(
        "arr_dt", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.parametrize("axis", list(range(2)) + [None])
    def test_replace_max(self, arr_dt, axis):
        a = dpnp.array([[10, 30, 20], [60, 40, 50]], dtype=arr_dt)

        # replace the max with a small value
        i_max = _add_keepdims(dpnp.argmax)(a, axis=axis)
        dpnp.put_along_axis(a, i_max, -99, axis=axis)

        # find the new minimum, which should max
        i_min = _add_keepdims(dpnp.argmin)(a, axis=axis)
        assert_array_equal(i_min, i_max)

    @pytest.mark.parametrize(
        "arr_dt", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.parametrize("idx_dt", get_integer_dtypes())
    @pytest.mark.parametrize("ndim", list(range(1, 4)))
    @pytest.mark.parametrize(
        "values",
        [
            777,
            [100, 200, 300, 400],
            (42,),
            range(4),
            numpy.arange(4),
            dpnp.ones(4),
        ],
        ids=[
            "scalar",
            "list",
            "tuple",
            "range",
            "numpy.ndarray",
            "dpnp.ndarray",
        ],
    )
    def test_values(self, arr_dt, idx_dt, ndim, values):
        np_a = numpy.arange(4**ndim, dtype=arr_dt).reshape((4,) * ndim)
        np_ai = numpy.array([3, 0, 2, 1], dtype=idx_dt).reshape(
            (1,) * (ndim - 1) + (4,)
        )

        dp_a = dpnp.array(np_a, dtype=arr_dt)
        dp_ai = dpnp.array(np_ai, dtype=idx_dt)

        for axis in range(ndim):
            numpy.put_along_axis(np_a, np_ai, values, axis)
            dpnp.put_along_axis(dp_a, dp_ai, values, axis)
            assert_array_equal(np_a, dp_a)

    @pytest.mark.parametrize("arr_dt", get_all_dtypes())
    @pytest.mark.parametrize("idx_dt", get_integer_dtypes())
    def test_broadcast(self, arr_dt, idx_dt):
        np_a = numpy.ones((3, 4, 1), dtype=arr_dt)
        np_ai = numpy.arange(10, dtype=idx_dt).reshape((1, 2, 5)) % 4

        dp_a = dpnp.array(np_a, dtype=arr_dt)
        dp_ai = dpnp.array(np_ai, dtype=idx_dt)

        numpy.put_along_axis(np_a, np_ai, 20, axis=1)
        dpnp.put_along_axis(dp_a, dp_ai, 20, axis=1)
        assert_array_equal(np_a, dp_a)


class TestTakeAlongAxis:
    # TODO: remove fixture once `dpnp.sort` is fully implemented
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "func, argfunc, kwargs",
        [
            pytest.param(dpnp.sort, dpnp.argsort, {}),
            pytest.param(
                _add_keepdims(dpnp.min), _add_keepdims(dpnp.argmin), {}
            ),
            pytest.param(
                _add_keepdims(dpnp.max), _add_keepdims(dpnp.argmax), {}
            ),
            # TODO: unmute, once `dpnp.argpartition` is implemented
            # pytest.param(dpnp.partition, dpnp.argpartition, {"kth": 2}),
        ],
    )
    def test_argequivalent(self, func, argfunc, kwargs):
        a = dpnp.random.random(size=(3, 4, 5))

        for axis in list(range(a.ndim)) + [None]:
            a_func = func(a, axis=axis, **kwargs)
            ai_func = argfunc(a, axis=axis, **kwargs)
            assert_array_equal(
                a_func, dpnp.take_along_axis(a, ai_func, axis=axis)
            )

    @pytest.mark.parametrize(
        "arr_dt", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.parametrize("idx_dt", get_integer_dtypes())
    @pytest.mark.parametrize("ndim", list(range(1, 4)))
    def test_multi_dimensions(self, arr_dt, idx_dt, ndim):
        np_a = numpy.arange(4**ndim, dtype=arr_dt).reshape((4,) * ndim)
        np_ai = numpy.array([3, 0, 2, 1], dtype=idx_dt).reshape(
            (1,) * (ndim - 1) + (4,)
        )

        dp_a = dpnp.array(np_a, dtype=arr_dt)
        dp_ai = dpnp.array(np_ai, dtype=idx_dt)

        for axis in range(ndim):
            expected = numpy.take_along_axis(np_a, np_ai, axis)
            result = dpnp.take_along_axis(dp_a, dp_ai, axis)
            assert_array_equal(expected, result)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid(self, xp):
        a = xp.ones((10, 10))
        ai = xp.ones((10, 2), dtype=xp.intp)

        # not enough indices
        assert_raises(ValueError, xp.take_along_axis, a, xp.array(1), axis=1)

        # bool arrays not allowed
        assert_raises(
            IndexError, xp.take_along_axis, a, ai.astype(bool), axis=1
        )

        # float arrays not allowed
        assert_raises(
            IndexError, xp.take_along_axis, a, ai.astype(numpy.float32), axis=1
        )

        # invalid axis
        assert_raises(numpy.AxisError, xp.take_along_axis, a, ai, axis=10)

    @pytest.mark.parametrize("arr_dt", get_all_dtypes())
    @pytest.mark.parametrize("idx_dt", get_integer_dtypes())
    def test_empty(self, arr_dt, idx_dt):
        np_a = numpy.ones((3, 4, 5), dtype=arr_dt)
        np_ai = numpy.ones((3, 0, 5), dtype=idx_dt)

        dp_a = dpnp.array(np_a, dtype=arr_dt)
        dp_ai = dpnp.array(np_ai, dtype=idx_dt)

        expected = numpy.take_along_axis(np_a, np_ai, axis=1)
        result = dpnp.take_along_axis(dp_a, dp_ai, axis=1)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("arr_dt", get_all_dtypes())
    @pytest.mark.parametrize("idx_dt", get_integer_dtypes())
    def test_broadcast(self, arr_dt, idx_dt):
        np_a = numpy.ones((3, 4, 1), dtype=arr_dt)
        np_ai = numpy.ones((1, 2, 5), dtype=idx_dt)

        dp_a = dpnp.array(np_a, dtype=arr_dt)
        dp_ai = dpnp.array(np_ai, dtype=idx_dt)

        expected = numpy.take_along_axis(np_a, np_ai, axis=1)
        result = dpnp.take_along_axis(dp_a, dp_ai, axis=1)
        assert_array_equal(expected, result)


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


@pytest.mark.parametrize("arr_dtype", get_all_dtypes(no_bool=True))
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
def test_diagonal(array, arr_dtype, offset):
    a = numpy.array(array, dtype=arr_dtype)
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


@pytest.mark.parametrize("array_dtype", get_all_dtypes())
@pytest.mark.parametrize(
    "indices_dtype", [dpnp.int32, dpnp.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "indices", [[-2, 2], [-5, 4]], ids=["[-2, 2]", "[-5, 4]"]
)
@pytest.mark.parametrize(
    "vals",
    [0, [1, 2], (2, 2), dpnp.array([1, 2])],
    ids=["0", "[1, 2]", "(2, 2)", "dpnp.array([1,2])"],
)
@pytest.mark.parametrize("mode", ["clip", "wrap"], ids=["clip", "wrap"])
def test_put_1d(indices, vals, array_dtype, indices_dtype, mode):
    a = numpy.array([-2, -1, 0, 1, 2], dtype=array_dtype)
    b = numpy.copy(a)
    ia = dpnp.array(a)
    ib = dpnp.array(b)
    ind = numpy.array(indices, dtype=indices_dtype)
    iind = dpnp.array(ind)

    # TODO: remove when #1382(dpctl) is solved
    if dpnp.is_supported_array_type(vals):
        vals = dpnp.astype(vals, ia.dtype)

    numpy.put(a, ind, vals, mode=mode)
    dpnp.put(ia, iind, vals, mode=mode)
    assert_array_equal(a, ia)

    b.put(ind, vals, mode=mode)
    ib.put(iind, vals, mode=mode)
    assert_array_equal(b, ib)


@pytest.mark.parametrize("array_dtype", get_all_dtypes())
@pytest.mark.parametrize(
    "indices_dtype", [dpnp.int32, dpnp.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize("vals", [[10, 20]], ids=["[10, 20]"])
@pytest.mark.parametrize(
    "indices",
    [
        [0, 7],
        [3, 4],
        [-9, 8],
    ],
    ids=[
        "[0, 7]",
        "[3, 4]",
        "[-9, 8]",
    ],
)
@pytest.mark.parametrize("mode", ["clip", "wrap"], ids=["clip", "wrap"])
def test_put_2d(array_dtype, indices_dtype, indices, vals, mode):
    a = numpy.array([[-1, 0, 1], [-2, -3, -4], [2, 3, 4]], dtype=array_dtype)
    ia = dpnp.array(a)
    ind = numpy.array(indices, dtype=indices_dtype)
    iind = dpnp.array(ind)
    numpy.put(a, ind, vals, mode=mode)
    dpnp.put(ia, iind, vals, mode=mode)
    assert_array_equal(a, ia)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_put_2d_ind():
    a = numpy.arange(5)
    ia = dpnp.array(a)
    ind = numpy.array([[3, 0, 2, 1]])
    iind = dpnp.array(ind)
    numpy.put(a, ind, 10)
    dpnp.put(ia, iind, 10)
    assert_array_equal(a, ia)


@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (3,),
        (4,),
    ],
    ids=[
        "(0,)",
        "(3,)",
        "(4,)",
    ],
)
@pytest.mark.parametrize("mode", ["clip", "wrap"], ids=["clip", "wrap"])
def test_put_invalid_shape(shape, mode):
    a = dpnp.arange(7)
    ind = dpnp.array([2])
    vals = dpnp.ones(shape, dtype=a.dtype)
    # vals must be broadcastable to the shape of ind`
    with pytest.raises(ValueError):
        dpnp.put(a, ind, vals, mode=mode)


@pytest.mark.parametrize(
    "axis",
    [
        1.0,
        (0,),
        [0, 1],
    ],
    ids=[
        "1.0",
        "(0,)",
        "[0, 1]",
    ],
)
def test_put_invalid_axis(axis):
    a = dpnp.arange(6).reshape(2, 3)
    ind = dpnp.array([1])
    vals = [1]
    with pytest.raises(TypeError):
        dpnp.put(a, ind, vals, axis=axis)


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


@pytest.mark.parametrize("array_type", get_all_dtypes())
@pytest.mark.parametrize(
    "indices_type", [numpy.int32, numpy.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "indices", [[-2, 2], [-5, 4]], ids=["[-2, 2]", "[-5, 4]"]
)
@pytest.mark.parametrize("mode", ["clip", "wrap"], ids=["clip", "wrap"])
def test_take_1d(indices, array_type, indices_type, mode):
    a = numpy.array([-2, -1, 0, 1, 2], dtype=array_type)
    ind = numpy.array(indices, dtype=indices_type)
    ia = dpnp.array(a)
    iind = dpnp.array(ind)
    expected = numpy.take(a, ind, mode=mode)
    result = dpnp.take(ia, iind, mode=mode)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("array_type", get_all_dtypes())
@pytest.mark.parametrize(
    "indices_type", [numpy.int32, numpy.int64], ids=["int32", "int64"]
)
@pytest.mark.parametrize(
    "indices", [[-1, 0], [-3, 2]], ids=["[-1, 0]", "[-3, 2]"]
)
@pytest.mark.parametrize("mode", ["clip", "wrap"], ids=["clip", "wrap"])
@pytest.mark.parametrize("axis", [0, 1], ids=["0", "1"])
def test_take_2d(indices, array_type, indices_type, axis, mode):
    a = numpy.array([[-1, 0, 1], [-2, -3, -4], [2, 3, 4]], dtype=array_type)
    ind = numpy.array(indices, dtype=indices_type)
    ia = dpnp.array(a)
    iind = dpnp.array(ind)
    expected = numpy.take(a, ind, axis=axis, mode=mode)
    result = dpnp.take(ia, iind, axis=axis, mode=mode)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("array_type", get_all_dtypes())
@pytest.mark.parametrize("indices", [[-5, 5]], ids=["[-5, 5]"])
@pytest.mark.parametrize("mode", ["clip", "wrap"], ids=["clip", "wrap"])
def test_take_over_index(indices, array_type, mode):
    a = dpnp.array([-2, -1, 0, 1, 2], dtype=array_type)
    ind = dpnp.array(indices, dtype=dpnp.int64)
    expected = dpnp.array([-2, 2], dtype=a.dtype)
    result = dpnp.take(a, ind, mode=mode)
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
