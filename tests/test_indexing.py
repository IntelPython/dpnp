import functools

import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from numpy.testing import (
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)

import dpnp
from dpnp.dpnp_array import dpnp_array

from .helper import get_all_dtypes, get_integer_dtypes, has_support_aspect64


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
        assert_raises(AxisError, a.diagonal, axis1=0, axis2=5)
        assert_raises(AxisError, a.diagonal, axis1=5, axis2=0)
        assert_raises(AxisError, a.diagonal, axis1=5, axis2=5)

        # same axes
        assert_raises(ValueError, a.diagonal, axis1=1, axis2=1)
        assert_raises(ValueError, a.diagonal, axis1=1, axis2=-1)


class TestExtins:
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_extract(self, dt):
        a = numpy.array([1, 3, 2, 1, 2, 3, 3], dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.extract(ia > 1, ia)
        expected = numpy.extract(a > 1, a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("cond_dt", get_all_dtypes(no_none=True))
    def test_extract_diff_dtypes(self, a_dt, cond_dt):
        a = numpy.array([-2, -1, 0, 1, 2, 3], dtype=a_dt)
        cond = numpy.array([1, -1, 2, 0, -2, 3], dtype=cond_dt)
        ia, icond = dpnp.array(a), dpnp.array(cond)

        result = dpnp.extract(icond, ia)
        expected = numpy.extract(cond, a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    def test_extract_list_cond(self, a_dt):
        a = numpy.array([-2, -1, 0, 1, 2, 3], dtype=a_dt)
        cond = [1, -1, 2, 0, -2, 3]
        ia = dpnp.array(a)

        result = dpnp.extract(cond, ia)
        expected = numpy.extract(cond, a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_place(self, dt):
        a = numpy.array([1, 4, 3, 2, 5, 8, 7], dtype=dt)
        ia = dpnp.array(a)

        dpnp.place(ia, [0, 1, 0, 1, 0, 1, 0], [2, 4, 6])
        numpy.place(a, [0, 1, 0, 1, 0, 1, 0], [2, 4, 6])
        assert_array_equal(ia, a)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("mask_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("vals_dt", get_all_dtypes(no_none=True))
    def test_place_diff_dtypes(self, a_dt, mask_dt, vals_dt):
        a = numpy.array(
            [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]], dtype=a_dt
        )
        mask = numpy.array(
            [
                [[True, False], [False, True]],
                [[False, True], [True, False]],
                [[False, False], [True, True]],
            ],
            dtype=mask_dt,
        )
        vals = numpy.array(
            [100, 200, 300, 400, 500, 600, 800, 900], dtype=vals_dt
        )
        ia, imask, ivals = dpnp.array(a), dpnp.array(mask), dpnp.array(vals)

        if numpy.can_cast(vals_dt, a_dt, casting="safe"):
            dpnp.place(ia, imask, ivals)
            numpy.place(a, mask, vals)
            assert_array_equal(ia, a)
        else:
            assert_raises(TypeError, dpnp.place, ia, imask, ivals)
            assert_raises(TypeError, numpy.place, a, mask, vals)

    def test_place_broadcast_vals(self):
        a = numpy.array([1, 4, 3, 2, 5, 8, 7])
        ia = dpnp.array(a)

        dpnp.place(ia, [1, 0, 1, 0, 1, 0, 1], [8, 9])
        numpy.place(a, [1, 0, 1, 0, 1, 0, 1], [8, 9])
        assert_array_equal(ia, a)

    def test_place_empty_vals(self):
        a = numpy.array([1, 4, 3, 2, 5, 8, 7])
        mask = numpy.zeros(7)
        ia, imask = dpnp.array(a), dpnp.array(mask)
        vals = []

        dpnp.place(ia, imask, vals)
        numpy.place(a, mask, vals)
        assert_array_equal(ia, a)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_place_insert_from_empty_vals(self, xp):
        a = xp.array([1, 4, 3, 2, 5, 8, 7])
        assert_raises_regex(
            ValueError,
            "Cannot insert from an empty array",
            lambda: xp.place(a, [0, 0, 0, 0, 0, 1, 0], []),
        )

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_place_wrong_array_type(self, xp):
        assert_raises(TypeError, xp.place, [1, 2, 3], [True, False], [0, 1])

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_both(self, dt):
        a = numpy.random.rand(10).astype(dt)
        mask = a > 0.5
        ia, imask = dpnp.array(a), dpnp.array(mask)

        result = dpnp.extract(imask, ia)
        expected = numpy.extract(mask, a)
        assert_array_equal(result, expected)

        ic = dpnp.extract(imask, ia)
        c = numpy.extract(mask, a)
        assert_array_equal(ic, c)

        dpnp.place(ia, imask, 0)
        dpnp.place(ia, imask, ic)

        numpy.place(a, mask, 0)
        numpy.place(a, mask, c)
        assert_array_equal(ia, a)


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


class TestIx:
    @pytest.mark.parametrize(
        "x0", [[0, 1], [True, True]], ids=["[0, 1]", "[True, True]"]
    )
    @pytest.mark.parametrize(
        "x1",
        [[2, 4], [False, False, True, False, True]],
        ids=["[2, 4]", "[False, False, True, False, True]"],
    )
    def test_ix(self, x0, x1):
        expected = dpnp.ix_(dpnp.array(x0), dpnp.array(x1))
        result = numpy.ix_(numpy.array(x0), numpy.array(x1))

        assert_array_equal(result[0], expected[0])
        assert_array_equal(result[1], expected[1])

    @pytest.mark.parametrize("dt", [dpnp.intp, dpnp.float32])
    def test_ix_empty_out(self, dt):
        a = numpy.array([], dtype=dt)
        ia = dpnp.array(a)

        (result,) = dpnp.ix_(ia)
        (expected,) = numpy.ix_(a)
        assert_array_equal(result, expected)
        assert a.dtype == dt

    def test_repeated_input(self):
        a = numpy.arange(5)
        ia = dpnp.array(a)

        result = dpnp.ix_(ia, ia)
        expected = numpy.ix_(a, a)
        assert_array_equal(result[0], expected[0])
        assert_array_equal(result[1], expected[1])

    @pytest.mark.parametrize("arr", [[2, 4, 0, 1], [True, False, True, True]])
    def test_usm_ndarray_input(self, arr):
        a = numpy.array(arr)
        ia = dpt.asarray(a)

        (result,) = dpnp.ix_(ia)
        (expected,) = numpy.ix_(a)
        assert_array_equal(result, expected)
        assert isinstance(result, dpnp_array)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize("shape", [(), (2, 2)])
    def test_ix_error(self, xp, shape):
        assert_raises(ValueError, xp.ix_, xp.ones(shape))


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

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", [bool, numpy.float32])
    def test_invalid_indices_dtype(self, xp, dt):
        a = xp.ones((10, 10))
        ind = xp.ones(10, dtype=dt)
        assert_raises(IndexError, xp.put_along_axis, a, ind, 7, axis=1)

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


class TestTake:
    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("ind_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "indices", [[-2, 2], [-5, 4]], ids=["[-2, 2]", "[-5, 4]"]
    )
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_1d(self, a_dt, ind_dt, indices, mode):
        a = numpy.array([-2, -1, 0, 1, 2], dtype=a_dt)
        ind = numpy.array(indices, dtype=ind_dt)
        ia, iind = dpnp.array(a), dpnp.array(ind)

        if numpy.can_cast(ind_dt, numpy.intp, casting="safe"):
            result = dpnp.take(ia, iind, mode=mode)
            expected = numpy.take(a, ind, mode=mode)
            assert_array_equal(result, expected)
        else:
            assert_raises(TypeError, ia.take, iind, mode=mode)
            assert_raises(TypeError, a.take, ind, mode=mode)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("ind_dt", get_integer_dtypes())
    @pytest.mark.parametrize(
        "indices", [[-1, 0], [-3, 2]], ids=["[-1, 0]", "[-3, 2]"]
    )
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    @pytest.mark.parametrize("axis", [0, 1], ids=["0", "1"])
    def test_2d(self, a_dt, ind_dt, indices, mode, axis):
        a = numpy.array([[-1, 0, 1], [-2, -3, -4], [2, 3, 4]], dtype=a_dt)
        ind = numpy.array(indices, dtype=ind_dt)
        ia, iind = dpnp.array(a), dpnp.array(ind)

        result = ia.take(iind, axis=axis, mode=mode)
        expected = a.take(ind, axis=axis, mode=mode)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("indices", [[-5, 5]], ids=["[-5, 5]"])
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_over_index(self, a_dt, indices, mode):
        a = dpnp.array([-2, -1, 0, 1, 2], dtype=a_dt)
        ind = dpnp.array(indices, dtype=numpy.intp)

        result = dpnp.take(a, ind, mode=mode)
        expected = dpnp.array([-2, 2], dtype=a.dtype)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("indices", [[0], [1]], ids=["[0]", "[1]"])
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_index_error(self, xp, indices, mode):
        # take from a 0-length dimension
        a = xp.empty((2, 3, 0, 4))
        assert_raises(IndexError, a.take, indices, axis=2, mode=mode)

    def test_bool_axis(self):
        a = numpy.array([[[1]]])
        ia = dpnp.array(a)

        result = ia.take([0], axis=False)
        expected = a.take([0], axis=0)  # numpy raises an error for bool axis
        assert_array_equal(result, expected)

    def test_axis_as_array(self):
        a = numpy.array([[[1]]])
        ia = dpnp.array(a)

        result = ia.take([0], axis=ia)
        expected = a.take(
            [0], axis=1
        )  # numpy raises an error for axis as array
        assert_array_equal(result, expected)

    def test_mode_raise(self):
        a = dpnp.array([[1, 2], [3, 4]])
        assert_raises(ValueError, a.take, [-1, 4], mode="raise")

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_unicode_mode(self, xp):
        a = xp.arange(10)
        k = b"\xc3\xa4".decode("UTF8")
        assert_raises(ValueError, a.take, 5, mode=k)


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
        a = numpy.arange(4**ndim, dtype=arr_dt).reshape((4,) * ndim)
        ind = numpy.array([3, 0, 2, 1], dtype=idx_dt).reshape(
            (1,) * (ndim - 1) + (4,)
        )
        ia, iind = dpnp.array(a), dpnp.array(ind)

        for axis in range(ndim):
            result = dpnp.take_along_axis(ia, iind, axis)
            expected = numpy.take_along_axis(a, ind, axis)
            assert_array_equal(expected, result)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_not_enough_indices(self, xp):
        a = xp.ones((10, 10))
        assert_raises(ValueError, xp.take_along_axis, a, xp.array(1), axis=1)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", [bool, numpy.float32])
    def test_invalid_indices_dtype(self, xp, dt):
        a = xp.ones((10, 10))
        ind = xp.ones((10, 2), dtype=dt)
        assert_raises(IndexError, xp.take_along_axis, a, ind, axis=1)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_axis(self, xp):
        a = xp.ones((10, 10))
        ind = xp.ones((10, 2), dtype=xp.intp)
        assert_raises(AxisError, xp.take_along_axis, a, ind, axis=10)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_indices_ndim_axis_none(self, xp):
        a = xp.ones((10, 10))
        ind = xp.ones((10, 2), dtype=xp.intp)
        assert_raises(ValueError, xp.take_along_axis, a, ind, axis=None)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("idx_dt", get_integer_dtypes())
    def test_empty(self, a_dt, idx_dt):
        a = numpy.ones((3, 4, 5), dtype=a_dt)
        ind = numpy.ones((3, 0, 5), dtype=idx_dt)
        ia, iind = dpnp.array(a), dpnp.array(ind)

        result = dpnp.take_along_axis(ia, iind, axis=1)
        expected = numpy.take_along_axis(a, ind, axis=1)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("idx_dt", get_integer_dtypes())
    def test_broadcast(self, a_dt, idx_dt):
        a = numpy.ones((3, 4, 1), dtype=a_dt)
        ind = numpy.ones((1, 2, 5), dtype=idx_dt)
        ia, iind = dpnp.array(a), dpnp.array(ind)

        result = dpnp.take_along_axis(ia, iind, axis=1)
        expected = numpy.take_along_axis(a, ind, axis=1)
        assert_array_equal(expected, result)

    def test_mode_wrap(self):
        a = numpy.array([-2, -1, 0, 1, 2])
        ind = numpy.array([-2, 2, -5, 4])
        ia, iind = dpnp.array(a), dpnp.array(ind)

        result = dpnp.take_along_axis(ia, iind, axis=0, mode="wrap")
        expected = numpy.take_along_axis(a, ind, axis=0)
        assert_array_equal(result, expected)

    def test_mode_clip(self):
        a = dpnp.array([-2, -1, 0, 1, 2])
        ind = dpnp.array([-2, 2, -5, 4])

        # numpy does not support keyword `mode`
        result = dpnp.take_along_axis(a, ind, axis=0, mode="clip")
        assert (result == dpnp.array([-2, 0, -2, 2])).all()


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
    [(), (1,), (2,), (1, 2), (2, 3), (3, 2), [1], [2], [1, 2], [2, 3], [3, 2]],
    ids=[
        "()",
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


@pytest.mark.parametrize(
    "m", [None, 0, 1, 2, 3, 4], ids=["None", "0", "1", "2", "3", "4"]
)
@pytest.mark.parametrize(
    "k", [-3, -2, -1, 0, 1, 2, 3], ids=["-3", "-2", "-1", "0", "1", "2", "3"]
)
@pytest.mark.parametrize(
    "n", [1, 2, 3, 4, 5, 6], ids=["1", "2", "3", "4", "5", "6"]
)
def test_tril_indices(n, k, m):
    result = dpnp.tril_indices(n, k, m)
    expected = numpy.tril_indices(n, k, m)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "k", [-3, -2, -1, 0, 1, 2, 3], ids=["-3", "-2", "-1", "0", "1", "2", "3"]
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
    "k", [-3, -2, -1, 0, 1, 2, 3], ids=["-3", "-2", "-1", "0", "1", "2", "3"]
)
@pytest.mark.parametrize(
    "n", [1, 2, 3, 4, 5, 6], ids=["1", "2", "3", "4", "5", "6"]
)
def test_triu_indices(n, k, m):
    result = dpnp.triu_indices(n, k, m)
    expected = numpy.triu_indices(n, k, m)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "k", [-3, -2, -1, 0, 1, 2, 3], ids=["-3", "-2", "-1", "0", "1", "2", "3"]
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


def test_indices_from_err():
    arr = dpnp.array([1, 2, 3])
    with pytest.raises(ValueError):
        dpnp.tril_indices_from(arr)
    with pytest.raises(ValueError):
        dpnp.triu_indices_from(arr)
    with pytest.raises(ValueError):
        dpnp.diag_indices_from(arr)
    with pytest.raises(ValueError):
        dpnp.diag_indices_from(dpnp.ones((2, 3)))


def test_fill_diagonal_error():
    arr = dpnp.ones((1, 2, 3))
    with pytest.raises(ValueError):
        dpnp.fill_diagonal(arr, 5)


class TestRavelIndex:
    def test_basic(self):
        expected = numpy.ravel_multi_index(numpy.array([1, 0]), (2, 2))
        result = dpnp.ravel_multi_index(dpnp.array([1, 0]), (2, 2))
        assert_equal(expected, result)

        x_np = numpy.array([[3, 6, 6], [4, 5, 1]])
        x_dp = dpnp.array([[3, 6, 6], [4, 5, 1]])

        expected = numpy.ravel_multi_index(x_np, (7, 6))
        result = dpnp.ravel_multi_index(x_dp, (7, 6))
        assert_equal(expected, result)

    def test_mode(self):
        x_np = numpy.array([[3, 6, 6], [4, 5, 1]])
        x_dp = dpnp.array([[3, 6, 6], [4, 5, 1]])

        expected = numpy.ravel_multi_index(x_np, (4, 6), mode="clip")
        result = dpnp.ravel_multi_index(x_dp, (4, 6), mode="clip")
        assert_equal(expected, result)

        expected = numpy.ravel_multi_index(x_np, (4, 4), mode=("clip", "wrap"))
        result = dpnp.ravel_multi_index(x_dp, (4, 4), mode=("clip", "wrap"))
        assert_equal(expected, result)

    def test_order_f(self):
        x_np = numpy.array([[3, 6, 6], [4, 5, 1]])
        x_dp = dpnp.array([[3, 6, 6], [4, 5, 1]])
        expected = numpy.ravel_multi_index(x_np, (7, 6), order="F")
        result = dpnp.ravel_multi_index(x_dp, (7, 6), order="F")
        assert_equal(expected, result)

    def test_error(self):
        assert_raises(
            ValueError, dpnp.ravel_multi_index, dpnp.array([2, 1]), (2, 2)
        )
        assert_raises(
            ValueError, dpnp.ravel_multi_index, dpnp.array([0, -3]), (2, 2)
        )
        assert_raises(
            ValueError, dpnp.ravel_multi_index, dpnp.array([0, 2]), (2, 2)
        )
        assert_raises(
            TypeError, dpnp.ravel_multi_index, dpnp.array([0.1, 0.0]), (2, 2)
        )

    def test_empty_indices_error(self):
        assert_raises(TypeError, dpnp.ravel_multi_index, ([], []), (10, 3))
        assert_raises(TypeError, dpnp.ravel_multi_index, ([], ["abc"]), (10, 3))
        assert_raises(
            TypeError,
            dpnp.ravel_multi_index,
            (dpnp.array([]), dpnp.array([])),
            (5, 3),
        )

    def test_empty_indices(self):
        assert_equal(
            dpnp.ravel_multi_index(
                (dpnp.array([], dtype=int), dpnp.array([], dtype=int)), (5, 3)
            ),
            [],
        )
        assert_equal(
            dpnp.ravel_multi_index(dpnp.array([[], []], dtype=int), (5, 3)), []
        )


class TestUnravelIndex:
    def test_basic(self):
        expected = numpy.unravel_index(numpy.array(2), (2, 2))
        result = dpnp.unravel_index(dpnp.array(2), (2, 2))
        assert_equal(expected, result)

        x_np = numpy.array([22, 41, 37])
        x_dp = dpnp.array([22, 41, 37])

        expected = numpy.unravel_index(x_np, (7, 6))
        result = dpnp.unravel_index(x_dp, (7, 6))
        assert_equal(expected, result)

    def test_order_f(self):
        x_np = numpy.array([31, 41, 13])
        x_dp = dpnp.array([31, 41, 13])
        expected = numpy.unravel_index(x_np, (7, 6), order="F")
        result = dpnp.unravel_index(x_dp, (7, 6), order="F")
        assert_equal(expected, result)

    def test_new_shape(self):
        expected = numpy.unravel_index(numpy.array(2), shape=(2, 2))
        result = dpnp.unravel_index(dpnp.array(2), shape=(2, 2))
        assert_equal(expected, result)

    def test_error(self):
        assert_raises(ValueError, dpnp.unravel_index, dpnp.array(-1), (2, 2))
        assert_raises(TypeError, dpnp.unravel_index, dpnp.array(0.5), (2, 2))
        assert_raises(ValueError, dpnp.unravel_index, dpnp.array(4), (2, 2))
        assert_raises(TypeError, dpnp.unravel_index, dpnp.array([]), (10, 3, 5))

    def test_empty_indices(self):
        assert_equal(
            dpnp.unravel_index(dpnp.array([], dtype=int), (10, 3, 5)),
            [[], [], []],
        )


class TestSelect:
    choices_np = [
        numpy.array([1, 2, 3]),
        numpy.array([4, 5, 6]),
        numpy.array([7, 8, 9]),
    ]
    choices_dp = [
        dpnp.array([1, 2, 3]),
        dpnp.array([4, 5, 6]),
        dpnp.array([7, 8, 9]),
    ]
    conditions_np = [
        numpy.array([False, False, False]),
        numpy.array([False, True, False]),
        numpy.array([False, False, True]),
    ]
    conditions_dp = [
        dpnp.array([False, False, False]),
        dpnp.array([False, True, False]),
        dpnp.array([False, False, True]),
    ]

    def test_basic(self):
        expected = numpy.select(self.conditions_np, self.choices_np, default=15)
        result = dpnp.select(self.conditions_dp, self.choices_dp, default=15)
        assert_array_equal(expected, result)

    def test_broadcasting(self):
        conditions_np = [numpy.array(True), numpy.array([False, True, False])]
        conditions_dp = [dpnp.array(True), dpnp.array([False, True, False])]
        choices_np = [numpy.array(1), numpy.arange(12).reshape(4, 3)]
        choices_dp = [dpnp.array(1), dpnp.arange(12).reshape(4, 3)]
        expected = numpy.select(conditions_np, choices_np)
        result = dpnp.select(conditions_dp, choices_dp)
        assert_array_equal(expected, result)

    def test_return_dtype(self):
        dtype = dpnp.complex128 if has_support_aspect64() else dpnp.complex64
        assert_equal(
            dpnp.select(self.conditions_dp, self.choices_dp, 1j).dtype, dtype
        )

        choices = [choice.astype(dpnp.int32) for choice in self.choices_dp]
        assert_equal(dpnp.select(self.conditions_dp, choices).dtype, dpnp.int32)

    def test_nan(self):
        choice_np = numpy.array([1, 2, 3, numpy.nan, 5, 7])
        choice_dp = dpnp.array([1, 2, 3, dpnp.nan, 5, 7])
        condition_np = numpy.isnan(choice_np)
        condition_dp = dpnp.isnan(choice_dp)
        expected = numpy.select([condition_np], [choice_np])
        result = dpnp.select([condition_dp], [choice_dp])
        assert_array_equal(expected, result)

    def test_many_arguments(self):
        condition_np = [numpy.array([False])] * 100
        condition_dp = [dpnp.array([False])] * 100
        choice_np = [numpy.array([1])] * 100
        choice_dp = [dpnp.array([1])] * 100
        expected = numpy.select(condition_np, choice_np)
        result = dpnp.select(condition_dp, choice_dp)
        assert_array_equal(expected, result)

    def test_deprecated_empty(self):
        assert_raises(ValueError, dpnp.select, [], [], 3j)
        assert_raises(ValueError, dpnp.select, [], [])

    def test_non_bool_deprecation(self):
        choices = self.choices_dp
        conditions = self.conditions_dp[:]
        conditions[0] = conditions[0].astype(dpnp.int64)
        assert_raises(TypeError, dpnp.select, conditions, choices)

    def test_error(self):
        x0 = dpnp.array([True, False])
        x1 = dpnp.array([1, 2])
        with pytest.raises(ValueError):
            dpnp.select([x0], [x1, x1])
        with pytest.raises(TypeError):
            dpnp.select([x0], [x1], default=x1)
        with pytest.raises(TypeError):
            dpnp.select([x1], [x1])
