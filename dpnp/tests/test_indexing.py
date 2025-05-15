import functools

import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from dpctl.tensor._type_utils import _to_device_supported_dtype
from dpctl.utils import ExecutionPlacementError
from numpy.testing import (
    assert_,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)

import dpnp
from dpnp.dpnp_array import dpnp_array

from .helper import (
    get_abs_array,
    get_all_dtypes,
    get_array,
    get_integer_dtypes,
    has_support_aspect64,
    is_win_platform,
    numpy_version,
)
from .third_party.cupy import testing


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
            "(2, 2)",
            "(3, 3)",
            "(2, 5)",
            "(3, 2, 2)",
            "(2, 2, 2, 2)",
            "(2, 2, 2, 3)",
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

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize("offset", [-3, -1, 0, 1, 3])
    def test_linalg_diagonal(self, offset):
        a = numpy.arange(24).reshape(2, 2, 2, 3)
        a_dp = dpnp.array(a)
        expected = numpy.linalg.diagonal(a, offset=offset)
        result = dpnp.linalg.diagonal(a_dp, offset=offset)
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
        a = get_abs_array([-2, -1, 0, 1, 2, 3], a_dt)
        cond = get_abs_array([1, -1, 2, 0, -2, 3], cond_dt)
        ia, icond = dpnp.array(a), dpnp.array(cond)

        result = dpnp.extract(icond, ia)
        expected = numpy.extract(cond, a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    def test_extract_list_cond(self, a_dt):
        a = get_abs_array([-2, -1, 0, 1, 2, 3], a_dt)
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
            [101, 102, 103, 104, 105, 106, 108, 109], dtype=vals_dt
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
        assert_equal(arr, 10.0, strict=False)


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


class TestIterable:
    @pytest.mark.parametrize("data", [[1.0], [2, 3]])
    def test_basic(self, data):
        a = numpy.array(data)
        ia = dpnp.array(a)
        assert dpnp.iterable(ia) == numpy.iterable(a)


@pytest.mark.parametrize(
    "shape", [[1, 2, 3], [(1, 2, 3)], [(3,)], [3], [], [()], [0]]
)
class TestNdindex:
    def test_basic(self, shape):
        result = dpnp.ndindex(*shape)
        expected = numpy.ndindex(*shape)

        for x, y in zip(result, expected):
            assert x == y

    def test_next(self, shape):
        dind = dpnp.ndindex(*shape)
        nind = numpy.ndindex(*shape)

        while True:
            try:
                ditem = next(dind)
            except StopIteration:
                assert_raises(StopIteration, next, nind)
                break  # both reach ends
            else:
                nitem = next(nind)
                assert ditem == nitem


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
        a = get_abs_array([1, 0, 2, -1, 0, 0, 8], dtype)
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
        "indices", [[0, 2], [-3, 4]], ids=["[0, 2]", "[-3, 4]"]
    )
    @pytest.mark.parametrize("ind_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "ivals",
        [0, [1, 2], (2, 2), dpnp.array([1, 2])],
        ids=["0", "[1, 2]", "(2, 2)", "dpnp.array([1, 2])"],
    )
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_input_1d(self, a_dt, indices, ind_dt, ivals, mode):
        a = get_abs_array([-2, -1, 0, 1, 2], a_dt)
        b, vals = numpy.copy(a), get_array(numpy, ivals)
        ia, ib = dpnp.array(a), dpnp.array(b)

        ind = get_abs_array(indices, ind_dt)
        if ind_dt == dpnp.bool and ind.all():
            ind[0] = False  # to get rid of duplicate indices
        iind = dpnp.array(ind)

        if numpy.can_cast(ind_dt, numpy.intp, casting="safe"):
            numpy.put(a, ind, vals, mode=mode)
            dpnp.put(ia, iind, ivals, mode=mode)
            assert_array_equal(ia, a)

            b.put(ind, vals, mode=mode)
            ib.put(iind, ivals, mode=mode)
            assert_array_equal(ib, b)
        elif ind_dt == numpy.uint64:
            # For this special case, NumPy raises an error but dpnp works
            assert_raises(TypeError, numpy.put, a, ind, vals, mode=mode)
            assert_raises(TypeError, b.put, ind, vals, mode=mode)

            numpy.put(a, ind.astype(numpy.int64), vals, mode=mode)
            dpnp.put(ia, iind, vals, mode=mode)
            assert_array_equal(ia, a)

            b.put(ind.astype(numpy.int64), vals, mode=mode)
            ib.put(iind, vals, mode=mode)
            assert_array_equal(ib, b)
        else:
            assert_raises(TypeError, numpy.put, a, ind, vals, mode=mode)
            assert_raises(TypeError, dpnp.put, ia, iind, ivals, mode=mode)

            assert_raises(TypeError, b.put, ind, vals, mode=mode)
            assert_raises(TypeError, ib.put, iind, ivals, mode=mode)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "indices",
        [[0, 7], [3, 4], [-7, 8]],
        ids=["[0, 7]", "[3, 4]", "[-7, 8]"],
    )
    @pytest.mark.parametrize("ind_dt", get_integer_dtypes())
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_input_2d(self, a_dt, indices, ind_dt, mode):
        a = get_abs_array([[-1, 0, 1], [-2, -3, -4], [2, 3, 4]], a_dt)
        ia = dpnp.array(a)
        vals = [10, 20]

        ind = get_abs_array(indices, ind_dt)
        iind = dpnp.array(ind)

        if ind_dt == numpy.uint64:
            # For this special case, NumPy raises an error but dpnp works
            assert_raises(TypeError, numpy.put, a, ind, vals, mode=mode)

            numpy.put(a, ind.astype(numpy.int64), vals, mode=mode)
            dpnp.put(ia, iind, vals, mode=mode)
            assert_array_equal(ia, a)
        else:
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

    @pytest.mark.parametrize("shape", [(3,), (4,)], ids=["(3,)", "(4,)"])
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
        "axis", [1.0, (0,), [0, 1]], ids=["1.0", "(0,)", "[0, 1]"]
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
        val = 0 if numpy.issubdtype(arr_dt, numpy.unsignedinteger) else -99
        dpnp.put_along_axis(a, i_max, val, axis=axis)

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
            77,
            [101, 102, 103, 104],
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
        a = numpy.arange(4**ndim, dtype=arr_dt).reshape((4,) * ndim)
        ind = numpy.array([3, 0, 2, 1], dtype=idx_dt).reshape(
            (1,) * (ndim - 1) + (4,)
        )
        ia, iind = dpnp.array(a), dpnp.array(ind)

        for axis in range(ndim):
            numpy.put_along_axis(a, ind, get_array(numpy, values), axis)
            dpnp.put_along_axis(ia, iind, values, axis)
            assert_array_equal(ia, a)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", [bool, numpy.float32])
    def test_invalid_indices_dtype(self, xp, dt):
        a = xp.ones((10, 10))
        ind = xp.ones_like(a, dtype=dt)
        assert_raises(IndexError, xp.put_along_axis, a, ind, 7, axis=1)

    @pytest.mark.parametrize("arr_dt", get_all_dtypes())
    @pytest.mark.parametrize("idx_dt", get_integer_dtypes())
    def test_broadcast(self, arr_dt, idx_dt):
        a = numpy.ones((3, 4, 1), dtype=arr_dt)
        ind = numpy.arange(10, dtype=idx_dt).reshape((1, 2, 5)) % 4
        ia, iind = dpnp.array(a), dpnp.array(ind)

        if idx_dt == numpy.uint64:
            numpy.put_along_axis(a, ind, 20, axis=1)
            dpnp.put_along_axis(ia, iind, 20, axis=1)
            assert_array_equal(ia, a)

    def test_mode_wrap(self):
        a = numpy.array([-2, -1, 0, 1, 2])
        ind = numpy.array([-2, 2, -5, 4])
        ia, iind = dpnp.array(a), dpnp.array(ind)

        dpnp.put_along_axis(ia, iind, 3, axis=0, mode="wrap")
        numpy.put_along_axis(a, ind, 3, axis=0)
        assert_array_equal(ia, a)

    def test_mode_clip(self):
        a = dpnp.array([-2, -1, 0, 1, 2])
        ind = dpnp.array([-2, 2, -5, 4])

        # numpy does not support keyword `mode`
        dpnp.put_along_axis(a, ind, 4, axis=0, mode="clip")
        assert (a == dpnp.array([4, -1, 4, 1, 4])).all()

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_indices_ndim_axis_none(self, xp):
        a = xp.ones((10, 10))
        ind = xp.ones((10, 2), dtype=xp.intp)
        assert_raises(ValueError, xp.put_along_axis, a, ind, -1, axis=None)


class TestTake:
    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("ind_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "indices", [[-2, 2], [-4, 4]], ids=["[-2, 2]", "[-4, 4]"]
    )
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_1d(self, a_dt, ind_dt, indices, mode):
        a = get_abs_array([-2, -1, 0, 1, 2], a_dt)
        ind = get_abs_array(indices, ind_dt)
        ia, iind = dpnp.array(a), dpnp.array(ind)

        if numpy.can_cast(ind_dt, numpy.intp, casting="safe"):
            result = dpnp.take(ia, iind, mode=mode)
            expected = numpy.take(a, ind, mode=mode)
            assert_array_equal(result, expected)
        elif ind_dt == numpy.uint64:
            # For this special case, although casting `ind_dt` to numpy.intp
            # is not safe, both NumPy and dpnp work properly
            # NumPy < "2.2.0" raises an error
            if numpy_version() < "2.2.0":
                ind = ind.astype(numpy.int64)
            result = dpnp.take(ia, iind, mode=mode)
            expected = numpy.take(a, ind, mode=mode)
            assert_array_equal(result, expected)
        else:
            assert_raises(TypeError, ia.take, iind, mode=mode)
            assert_raises(TypeError, a.take, ind, mode=mode)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("ind_dt", get_integer_dtypes())
    @pytest.mark.parametrize(
        "indices", [[-1, 0], [-2, 2]], ids=["[-1, 0]", "[-2, 2]"]
    )
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_2d(self, a_dt, ind_dt, indices, mode, axis):
        a = get_abs_array([[-1, 0, 1], [-2, -3, -4], [2, 3, 4]], a_dt)
        ind = get_abs_array(indices, ind_dt)
        ia, iind = dpnp.array(a), dpnp.array(ind)

        if ind_dt == numpy.uint64:
            # For this special case, NumPy raises an error on Windows
            result = ia.take(iind, axis=axis, mode=mode)
            expected = a.take(ind.astype(numpy.int64), axis=axis, mode=mode)
            assert_array_equal(result, expected)
        else:
            result = ia.take(iind, axis=axis, mode=mode)
            expected = a.take(ind, axis=axis, mode=mode)
            assert_array_equal(result, expected)

    @pytest.mark.parametrize("a_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_over_index(self, a_dt, mode):
        a = get_abs_array([-2, -1, 0, 1, 2], a_dt)
        a = dpnp.array(a)
        ind = dpnp.array([-5, 5], dtype=numpy.intp)

        result = dpnp.take(a, ind, mode=mode)
        expected = get_abs_array([-2, 2], a_dt)
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
        # TODO: to roll back the change once the issue with CUDA support is resolved for random
        # a = dpnp.random.random(size=(3, 4, 5))
        a = dpnp.asarray(numpy.random.random(size=(3, 4, 5)))

        for axis in list(range(a.ndim)) + [None, -1]:
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


@pytest.mark.parametrize("val", [-1, 0, 1])
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
@pytest.mark.parametrize("sparse", [True, False])
def test_indices(dimension, dtype, sparse):
    expected = numpy.indices(dimension, dtype=dtype, sparse=sparse)
    result = dpnp.indices(dimension, dtype=dtype, sparse=sparse)
    for Xnp, X in zip(expected, result):
        assert_array_equal(Xnp, X)


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
def test_putmask1(arr, mask):
    a = numpy.array(arr)
    ia = dpnp.array(a)
    m = numpy.array(mask)
    im = dpnp.array(m)
    v = numpy.array([100, 200])
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


@pytest.mark.parametrize("m", [None, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("k", [-3, -2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_tril_indices(n, k, m):
    result = dpnp.tril_indices(n, k, m)
    expected = numpy.tril_indices(n, k, m)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("k", [-3, -2, -1, 0, 1, 2, 3])
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


@pytest.mark.parametrize("m", [None, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("k", [-3, -2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
def test_triu_indices(n, k, m):
    result = dpnp.triu_indices(n, k, m)
    expected = numpy.triu_indices(n, k, m)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("k", [-3, -2, -1, 0, 1, 2, 3])
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
        a = numpy.array([], dtype=int)
        ia = dpnp.array(a)
        result = dpnp.ravel_multi_index((ia, ia), (5, 3))
        expected = numpy.ravel_multi_index((a, a), (5, 3))
        assert_equal(result, expected)

        a = numpy.array([[], []], dtype=int)
        ia = dpnp.array(a)
        result = dpnp.ravel_multi_index(ia, (5, 3))
        expected = numpy.ravel_multi_index(a, (5, 3))
        assert_equal(result, expected)


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


class TestCompress:
    def test_compress_basic(self):
        conditions = [True, False, True]
        a_np = numpy.arange(16).reshape(4, 4)
        a = dpnp.arange(16).reshape(4, 4)
        cond_np = numpy.array(conditions)
        cond = dpnp.array(conditions)
        expected = numpy.compress(cond_np, a_np, axis=0)
        result = dpnp.compress(cond, a, axis=0)
        assert_array_equal(expected, result)

    def test_compress_method_basic(self):
        conditions = [True, True, False, True]
        a_np = numpy.arange(3 * 4).reshape(3, 4)
        a = dpnp.arange(3 * 4).reshape(3, 4)
        cond_np = numpy.array(conditions)
        cond = dpnp.array(conditions)
        expected = a_np.compress(cond_np, axis=1)
        result = a.compress(cond, axis=1)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_compress_condition_all_dtypes(self, dtype):
        a_np = numpy.arange(10, dtype="i4")
        a = dpnp.arange(10, dtype="i4")
        cond_np = numpy.tile(numpy.asarray([0, 1], dtype=dtype), 5)
        cond = dpnp.tile(dpnp.asarray([0, 1], dtype=dtype), 5)
        expected = numpy.compress(cond_np, a_np)
        result = dpnp.compress(cond, a)
        assert_array_equal(expected, result)

    def test_compress_invalid_out_errors(self):
        q1 = dpctl.SyclQueue()
        q2 = dpctl.SyclQueue()
        a = dpnp.ones(10, dtype="i4", sycl_queue=q1)
        condition = dpnp.asarray([True], sycl_queue=q1)
        out_bad_shape = dpnp.empty_like(a)
        with pytest.raises(ValueError):
            dpnp.compress(condition, a, out=out_bad_shape)
        out_bad_queue = dpnp.empty(1, dtype="i4", sycl_queue=q2)
        with pytest.raises(ExecutionPlacementError):
            dpnp.compress(condition, a, out=out_bad_queue)
        out_bad_dt = dpnp.empty(1, dtype="i8", sycl_queue=q1)
        with pytest.raises(TypeError):
            dpnp.compress(condition, a, out=out_bad_dt)
        out_read_only = dpnp.empty(1, dtype="i4", sycl_queue=q1)
        out_read_only.flags.writable = False
        with pytest.raises(ValueError):
            dpnp.compress(condition, a, out=out_read_only)

    def test_compress_empty_axis(self):
        a = dpnp.ones((10, 0, 5), dtype="i4")
        condition = [True, False, True]
        r = dpnp.compress(condition, a, axis=0)
        assert r.shape == (2, 0, 5)
        # empty take from empty axis is permitted
        assert dpnp.compress([False], a, axis=1).shape == (10, 0, 5)
        # non-empty take from empty axis raises IndexError
        with pytest.raises(IndexError):
            dpnp.compress(condition, a, axis=1)

    def test_compress_in_overlaps_out(self):
        conditions = [False, True, True]
        a_np = numpy.arange(6)
        a = dpnp.arange(6)
        cond_np = numpy.array(conditions)
        cond = dpnp.array(conditions)
        out = a[2:4]
        expected = numpy.compress(cond_np, a_np, axis=None)
        result = dpnp.compress(cond, a, axis=None, out=out)
        assert_array_equal(expected, result)
        assert result is out
        assert (a[2:4] == out).all()

    def test_compress_condition_not_1d(self):
        a = dpnp.arange(4)
        cond = dpnp.ones((1, 4), dtype="?")
        with pytest.raises(ValueError):
            dpnp.compress(cond, a, axis=None)

    def test_compress_strided(self):
        a = dpnp.arange(20)
        a_np = dpnp.asnumpy(a)
        cond = dpnp.tile(dpnp.array([True, False, False, True]), 5)
        cond_np = dpnp.asnumpy(cond)
        result = dpnp.compress(cond, a)
        expected = numpy.compress(cond_np, a_np)
        assert_array_equal(result, expected)
        # use axis keyword
        a = dpnp.arange(50).reshape(10, 5)
        a_np = dpnp.asnumpy(a)
        cond = dpnp.array(dpnp.array([True, False, False, True, False]))
        cond_np = dpnp.asnumpy(cond)
        result = dpnp.compress(cond, a)
        expected = numpy.compress(cond_np, a_np)
        assert_array_equal(result, expected)


class TestChoose:
    def test_choose_basic(self):
        indices = [0, 1, 0]
        # use a single array for choices
        chcs_np = numpy.arange(2 * len(indices))
        chcs = dpnp.arange(2 * len(indices))
        inds_np = numpy.array(indices)
        inds = dpnp.array(indices)
        expected = numpy.choose(inds_np, chcs_np)
        result = dpnp.choose(inds, chcs)
        assert_array_equal(expected, result)

    def test_choose_method_basic(self):
        indices = [0, 1, 2]
        # use a single array for choices
        chcs_np = numpy.arange(3 * len(indices))
        chcs = dpnp.arange(3 * len(indices))
        inds_np = numpy.array(indices)
        inds = dpnp.array(indices)
        expected = inds_np.choose(chcs_np)
        result = inds.choose(chcs)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_choose_inds_all_dtypes(self, dtype):
        if not dpnp.issubdtype(dtype, dpnp.integer) and dtype != dpnp.bool:
            inds = dpnp.zeros(1, dtype=dtype)
            chcs = dpnp.ones(1, dtype=dtype)
            with pytest.raises(TypeError):
                dpnp.choose(inds, chcs)
        elif dtype == numpy.uint64:
            # For this special case, NumPy raises an error but dpnp works
            inds_np = numpy.array([1, 0, 1], dtype=dtype)
            inds = dpnp.array(inds_np)
            chcs_np = numpy.array([1, 2, 3], dtype=dtype)
            chcs = dpnp.array(chcs_np)
            assert_raises(TypeError, numpy.choose, inds_np, chcs_np)
            expected = numpy.choose(inds_np.astype(numpy.int64), chcs_np)
            result = dpnp.choose(inds, chcs)
            assert_array_equal(expected, result)
        else:
            inds_np = numpy.array([1, 0, 1], dtype=dtype)
            inds = dpnp.array(inds_np)
            chcs_np = numpy.array([1, 2, 3], dtype=dtype)
            chcs = dpnp.array(chcs_np)
            expected = numpy.choose(inds_np, chcs_np)
            result = dpnp.choose(inds, chcs)
            assert_array_equal(expected, result)

    def test_choose_invalid_out_errors(self):
        q1 = dpctl.SyclQueue()
        q2 = dpctl.SyclQueue()
        chcs = dpnp.ones(10, dtype="i4", sycl_queue=q1)
        inds = dpnp.zeros(10, dtype="i4", sycl_queue=q1)
        out_bad_shape = dpnp.empty(11, dtype=chcs.dtype, sycl_queue=q1)
        with pytest.raises(ValueError):
            dpnp.choose(inds, [chcs], out=out_bad_shape)
        out_bad_queue = dpnp.empty(chcs.shape, dtype=chcs.dtype, sycl_queue=q2)
        with pytest.raises(ExecutionPlacementError):
            dpnp.choose(inds, [chcs], out=out_bad_queue)
        out_bad_dt = dpnp.empty(chcs.shape, dtype="i8", sycl_queue=q1)
        with pytest.raises(TypeError):
            dpnp.choose(inds, [chcs], out=out_bad_dt)
        out_read_only = dpnp.empty(chcs.shape, dtype=chcs.dtype, sycl_queue=q1)
        out_read_only.flags.writable = False
        with pytest.raises(ValueError):
            dpnp.choose(inds, [chcs], out=out_read_only)

    def test_choose_empty(self):
        sh = (10, 0, 5)
        inds = dpnp.ones(sh, dtype="i4")
        chcs = dpnp.ones(sh)
        r = dpnp.choose(inds, chcs)
        assert r.shape == sh
        r = dpnp.choose(inds, (chcs,) * 2)
        assert r.shape == sh
        inds = dpnp.unstack(inds)[0]
        r = dpnp.choose(inds, chcs)
        assert r.shape == sh[1:]
        r = dpnp.choose(inds, [chcs])
        assert r.shape == sh

    def test_choose_0d_inputs(self):
        sh = ()
        inds = dpnp.zeros(sh, dtype="i4")
        chc = dpnp.ones(sh, dtype="i4")
        r = dpnp.choose(inds, [chc])
        assert r == chc

    def test_choose_out_keyword(self):
        inds = dpnp.tile(dpnp.array([0, 1, 2], dtype="i4"), (5, 3))
        inds_np = dpnp.asnumpy(inds)
        chc1 = dpnp.zeros(9, dtype="f4")
        chc2 = dpnp.ones(9, dtype="f4")
        chc3 = dpnp.full(9, 2, dtype="f4")
        chcs = [chc1, chc2, chc3]
        chcs_np = [dpnp.asnumpy(chc) for chc in chcs]
        out = dpnp.empty_like(inds, dtype="f4")
        dpnp.choose(inds, chcs, out=out)
        expected = numpy.choose(inds_np, chcs_np)
        assert_array_equal(out, expected)

    def test_choose_in_overlaps_out(self):
        # overlap with inds
        inds = dpnp.zeros(6, dtype="i4")
        inds_np = dpnp.asnumpy(inds)
        chc_np = numpy.arange(6, dtype="i4")
        chc = dpnp.arange(6, dtype="i4")
        out = inds
        expected = numpy.choose(inds_np, chc_np)
        result = dpnp.choose(inds, chc, out=out)
        assert_array_equal(expected, result)
        assert result is out
        assert (inds == out).all()
        # overlap with chc
        inds = dpnp.zeros(6, dtype="i4")
        out = chc
        expected = numpy.choose(inds_np, chc_np)
        result = dpnp.choose(inds, chc, out=out)
        assert_array_equal(expected, result)
        assert result is out
        assert (inds == out).all()

    def test_choose_strided(self):
        # inds strided
        inds = dpnp.tile(dpnp.array([0, 1], dtype="i4"), 5)
        inds_np = dpnp.asnumpy(inds)
        c1 = dpnp.arange(5, dtype="i4")
        c2 = dpnp.full(5, -1, dtype="i4")
        chcs = [c1, c2]
        chcs_np = [dpnp.asnumpy(chc) for chc in chcs]
        result = dpnp.choose(inds[::-2], chcs)
        expected = numpy.choose(inds_np[::-2], chcs_np)
        assert_array_equal(result, expected)
        # choices strided
        c3 = dpnp.arange(20, dtype="i4")
        c4 = dpnp.full(20, -1, dtype="i4")
        chcs = [c3[::-2], c4[::-2]]
        chcs_np = [dpnp.asnumpy(c3)[::-2], dpnp.asnumpy(c4)[::-2]]
        result = dpnp.choose(inds, chcs)
        expected = numpy.choose(inds_np, chcs_np)
        assert_array_equal(result, expected)
        # all strided
        result = dpnp.choose(inds[::-1], chcs)
        expected = numpy.choose(inds_np[::-1], chcs_np)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "indices", [[0, 2], [-5, 4]], ids=["[0, 2]", "[-5, 4]"]
    )
    @pytest.mark.parametrize("mode", ["clip", "wrap"])
    def test_choose_modes(self, indices, mode):
        chc = dpnp.array([-2, -1, 0, 1, 2], dtype="i4")
        chc_np = dpnp.asnumpy(chc)
        inds = dpnp.array(indices, dtype="i4")
        inds_np = dpnp.asnumpy(inds)
        expected = numpy.choose(inds_np, chc_np, mode=mode)
        result = dpnp.choose(inds, chc, mode=mode)
        assert_array_equal(expected, result)

    def test_choose_arg_validation(self):
        # invalid choices
        with pytest.raises(TypeError):
            dpnp.choose(dpnp.zeros((), dtype="i4"), 1)
        # invalid mode keyword
        with pytest.raises(ValueError):
            dpnp.choose(dpnp.zeros(()), dpnp.ones(()), mode="err")

    # based on examples from NumPy
    def test_choose_broadcasting(self):
        inds = dpnp.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype="i4")
        inds_np = dpnp.asnumpy(inds)
        chcs = dpnp.array([-10, 10])
        chcs_np = dpnp.asnumpy(chcs)
        result = dpnp.choose(inds, chcs)
        expected = numpy.choose(inds_np, chcs_np)
        assert_array_equal(result, expected)

        inds = dpnp.array([0, 1]).reshape((2, 1, 1))
        inds_np = dpnp.asnumpy(inds)
        chc1 = dpnp.array([1, 2, 3]).reshape((1, 3, 1))
        chc2 = dpnp.array([-1, -2, -3, -4, -5]).reshape(1, 1, 5)
        chcs = [chc1, chc2]
        chcs_np = [dpnp.asnumpy(chc) for chc in chcs]
        result = dpnp.choose(inds, chcs)
        expected = numpy.choose(inds_np, chcs_np)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("chc1_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("chc2_dt", get_all_dtypes(no_none=True))
    def test_choose_promote_choices(self, chc1_dt, chc2_dt):
        inds = dpnp.array([0, 1], dtype="i4")
        inds_np = dpnp.asnumpy(inds)
        chc1 = dpnp.zeros(1, dtype=chc1_dt)
        chc2 = dpnp.ones(1, dtype=chc2_dt)
        chcs = [chc1, chc2]
        chcs_np = [dpnp.asnumpy(chc) for chc in chcs]
        result = dpnp.choose(inds, chcs)
        expected = numpy.choose(inds_np, chcs_np)
        assert (
            _to_device_supported_dtype(expected.dtype, inds.sycl_device)
            == result.dtype
        )
