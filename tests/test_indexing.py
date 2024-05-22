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


class TestNonzero:
    @pytest.mark.parametrize("list_val", [[], [0], [1]])
    def test_trivial(self, list_val):
        np_res = numpy.nonzero(numpy.array(list_val))
        dpnp_res = dpnp.nonzero(dpnp.array(list_val))
        assert_array_equal(np_res, dpnp_res)

    @pytest.mark.parametrize("val", [0, 1])
    def test_0d(self, val):
        assert_raises(ValueError, dpnp.nonzero, dpnp.array(val))
        assert_raises(ValueError, dpnp.nonzero, dpnp.array(val))

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_1d(self, dtype):
        a = numpy.array([1, 0, 2, -1, 0, 0, 8], dtype=dtype)
        ia = dpnp.array(a)

        np_res = numpy.nonzero(a)
        dpnp_res = dpnp.nonzero(ia)
        assert_array_equal(np_res, dpnp_res)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_2d(self, dtype):
        a = numpy.array([[0, 1, 0], [2, 0, 3]], dtype=dtype)
        ia = dpnp.array(a)

        np_res = numpy.nonzero(a)
        dpnp_res = dpnp.nonzero(ia)
        assert_array_equal(np_res, dpnp_res)

        a = numpy.eye(3, dtype=dtype)
        ia = dpnp.eye(3, dtype=dtype)

        np_res = numpy.nonzero(a)
        dpnp_res = dpnp.nonzero(ia)
        assert_array_equal(np_res, dpnp_res)

    def test_sparse(self):
        for i in range(20):
            a = numpy.zeros(200, dtype=bool)
            a[i::20] = True
            ia = dpnp.array(a)

            np_res = numpy.nonzero(a)
            dpnp_res = dpnp.nonzero(ia)
            assert_array_equal(np_res, dpnp_res)

            a = numpy.zeros(400, dtype=bool)
            a[10 + i : 20 + i] = True
            a[20 + i * 2] = True
            ia = dpnp.array(a)

            np_res = numpy.nonzero(a)
            dpnp_res = dpnp.nonzero(ia)
            assert_array_equal(np_res, dpnp_res)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_array_method(self, dtype):
        a = numpy.array([[1, 0, 0], [4, 0, 6]], dtype=dtype)
        ia = dpnp.array(a)
        assert_array_equal(a.nonzero(), ia.nonzero())


class TestPut:
    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "indices", [[0, 2], [-5, 4]], ids=["[0, 2]", "[-5, 4]"]
    )
    @pytest.mark.parametrize("ind_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "vals",
        [0, [1, 2], (2, 2), dpnp.array([1, 2])],
        ids=["0", "[1, 2]", "(2, 2)", "dpnp.array([1,2])"],
    )
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_input_1d(self, a_dt, indices, ind_dt, vals, mode):
        a = numpy.array([-2, -1, 0, 1, 2], dtype=a_dt)
        b = numpy.copy(a)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        ind = numpy.array(indices, dtype=ind_dt)
        if ind_dt == dpnp.bool and ind.all():
            ind[0] = False  # to get rid of duplicate indices
        iind = dpnp.array(ind)

        if numpy.can_cast(ind_dt, numpy.intp, casting="safe"):
            numpy.put(a, ind, vals, mode=mode)
            dpnp.put(ia, iind, vals, mode=mode)
            assert_array_equal(ia, a)

            b.put(ind, vals, mode=mode)
            ib.put(iind, vals, mode=mode)
            assert_array_equal(ib, b)
        else:
            assert_raises(TypeError, numpy.put, a, ind, vals, mode=mode)
            assert_raises(TypeError, dpnp.put, ia, iind, vals, mode=mode)

            assert_raises(TypeError, b.put, ind, vals, mode=mode)
            assert_raises(TypeError, ib.put, iind, vals, mode=mode)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
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
    @pytest.mark.parametrize("ind_dt", get_integer_dtypes())
    @pytest.mark.parametrize("vals", [[10, 20]], ids=["[10, 20]"])
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_input_2d(self, a_dt, indices, ind_dt, vals, mode):
        a = numpy.array([[-1, 0, 1], [-2, -3, -4], [2, 3, 4]], dtype=a_dt)
        ia = dpnp.array(a)

        ind = numpy.array(indices, dtype=ind_dt)
        iind = dpnp.array(ind)

        numpy.put(a, ind, vals, mode=mode)
        dpnp.put(ia, iind, vals, mode=mode)
        assert_array_equal(ia, a)

    def test_indices_2d(self):
        a = numpy.arange(5)
        ia = dpnp.array(a)
        ind = numpy.array([[3, 0, 2, 1]])
        iind = dpnp.array(ind)

        numpy.put(a, ind, 10)
        dpnp.put(ia, iind, 10)
        assert_array_equal(ia, a)

    def test_non_contiguous(self):
        # force non C-contiguous array
        a = numpy.arange(6).reshape(2, 3).T
        ia = dpnp.arange(6).reshape(2, 3).T

        a.put([0, 2], [44, 55])
        ia.put([0, 2], [44, 55])
        assert_equal(ia, a)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_empty(self, dtype, mode):
        a = numpy.zeros(1000, dtype=dtype)
        ia = dpnp.array(a)

        numpy.put(a, [1, 2, 3], [], mode=mode)
        dpnp.put(ia, [1, 2, 3], [], mode=mode)
        assert_array_equal(ia, a)

    # TODO: enable test for numpy also since 2.0
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_empty_input(self, mode):
        empty = dpnp.asarray(list())
        with pytest.raises(IndexError):
            empty.put(1, 1, mode=mode)

    @pytest.mark.parametrize(
        "shape",
        [
            (3,),
            (4,),
        ],
        ids=[
            "(3,)",
            "(4,)",
        ],
    )
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_invalid_shape(self, shape, mode):
        a = dpnp.arange(7)
        ind = dpnp.array([2])
        vals = dpnp.ones(shape, dtype=a.dtype)
        # vals must be broadcastable to the shape of ind`
        with pytest.raises(ValueError):
            dpnp.put(a, ind, vals, mode=mode)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
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
    def test_invalid_axis(self, xp, axis):
        a = xp.arange(6).reshape(2, 3)
        ind = xp.array([1])
        with pytest.raises(TypeError):
            a.put(ind, [1], axis=axis)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_unsupported_input_array_type(self, xp):
        with pytest.raises(TypeError):
            xp.put([1, 2, 3], [0, 2], 5)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_non_writeable_input_array(self, xp):
        a = xp.zeros(6)
        a.flags["W"] = False
        with pytest.raises(ValueError):
            a.put([1, 3, 5], [1, 3, 5])


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


class TestDiagonal:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("offset", [-3, -1, 0, 1, 3])
    @pytest.mark.parametrize(
        "shape",
        [(2, 2), (3, 3), (2, 5), (3, 2, 2), (2, 2, 2, 2), (2, 2, 2, 3)],
        ids=[
            "(2,2)",
            "(3,3)",
            "(2,5)",
            "(3,2,2)",
            "(2,2,2,2)",
            "(2,2,2,3)",
        ],
    )
    def test_diagonal_offset(self, shape, dtype, offset):
        a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
        a_dp = dpnp.array(a)
        expected = numpy.diagonal(a, offset)
        result = dpnp.diagonal(a_dp, offset)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape, axis_pairs",
        [
            ((3, 4), [(0, 1), (1, 0)]),
            ((3, 4, 5), [(0, 1), (1, 2), (0, 2)]),
            ((4, 3, 5, 2), [(0, 1), (1, 2), (2, 3), (0, 3)]),
        ],
    )
    def test_diagonal_axes(self, shape, axis_pairs, dtype):
        a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
        a_dp = dpnp.array(a)
        for axis1, axis2 in axis_pairs:
            expected = numpy.diagonal(a, axis1=axis1, axis2=axis2)
            result = dpnp.diagonal(a_dp, axis1=axis1, axis2=axis2)
            assert_array_equal(expected, result)

    def test_diagonal_errors(self):
        a = dpnp.arange(12).reshape(3, 4)

        # unsupported type
        a_np = dpnp.asnumpy(a)
        assert_raises(TypeError, dpnp.diagonal, a_np)

        # a.ndim < 2
        a_ndim_1 = a.flatten()
        assert_raises(ValueError, dpnp.diagonal, a_ndim_1)

        # unsupported type `offset`
        assert_raises(TypeError, dpnp.diagonal, a, offset=1.0)
        assert_raises(TypeError, dpnp.diagonal, a, offset=[0])

        # axes are out of bounds
        assert_raises(numpy.AxisError, a.diagonal, axis1=0, axis2=5)
        assert_raises(numpy.AxisError, a.diagonal, axis1=5, axis2=0)
        assert_raises(numpy.AxisError, a.diagonal, axis1=5, axis2=5)

        # same axes
        assert_raises(ValueError, a.diagonal, axis1=1, axis2=1)
        assert_raises(ValueError, a.diagonal, axis1=1, axis2=-1)


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
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize("sparse", [True, False], ids=["True", "False"])
def test_indices(dimension, dtype, sparse):
    expected = numpy.indices(dimension, dtype=dtype, sparse=sparse)
    result = dpnp.indices(dimension, dtype=dtype, sparse=sparse)
    for Xnp, X in zip(expected, result):
        assert_array_equal(Xnp, X)


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
