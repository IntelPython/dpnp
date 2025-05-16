import itertools

import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from numpy.testing import (
    assert_array_equal,
    assert_equal,
    assert_raises,
)

import dpnp

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_array,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    get_integer_dtypes,
    get_integer_float_dtypes,
    has_support_aspect64,
    numpy_version,
)
from .third_party.cupy import testing


def _compare_results(result, expected):
    """Compare lists of arrays."""
    if len(result) != len(expected):
        raise ValueError("Iterables have different lengths")

    for x, y in zip(result, expected):
        assert_array_equal(x, y)


def test_result_type():
    X = [dpnp.ones((2), dtype=dpnp.int64), dpnp.int32, "float32"]
    X_np = [numpy.ones((2), dtype=numpy.int64), numpy.int32, "float32"]

    if has_support_aspect64():
        assert dpnp.result_type(*X) == numpy.result_type(*X_np)
    else:
        assert dpnp.result_type(*X) == dpnp.default_float_type(X[0].device)


def test_result_type_only_dtypes():
    X = [dpnp.int64, dpnp.int32, dpnp.bool, dpnp.float32]
    X_np = [numpy.int64, numpy.int32, numpy.bool_, numpy.float32]

    assert dpnp.result_type(*X) == numpy.result_type(*X_np)


def test_result_type_only_arrays():
    X = [dpnp.ones((2), dtype=dpnp.int64), dpnp.ones((7, 4), dtype=dpnp.int32)]
    X_np = [
        numpy.ones((2), dtype=numpy.int64),
        numpy.ones((7, 4), dtype=numpy.int32),
    ]

    assert dpnp.result_type(*X) == numpy.result_type(*X_np)


def test_ndim():
    a = [[1, 2, 3], [4, 5, 6]]
    ia = dpnp.array(a)

    exp = numpy.ndim(a)
    assert ia.ndim == exp
    assert dpnp.ndim(a) == exp
    assert dpnp.ndim(ia) == exp


def test_size():
    a = [[1, 2, 3], [4, 5, 6]]
    ia = dpnp.array(a)

    exp = numpy.size(a)
    assert ia.size == exp
    assert dpnp.size(a) == exp
    assert dpnp.size(ia) == exp

    exp = numpy.size(a, 0)
    assert dpnp.size(a, 0) == exp
    assert dpnp.size(ia, 0) == exp


class TestAppend:
    @pytest.mark.parametrize(
        "arr",
        [[], [1, 2, 3], [[1, 2, 3], [4, 5, 6]]],
        ids=["empty", "1D", "2D"],
    )
    @pytest.mark.parametrize(
        "value",
        [[], [1, 2, 3], [[1, 2, 3], [4, 5, 6]]],
        ids=["empty", "1D", "2D"],
    )
    def test_basic(self, arr, value):
        a = numpy.array(arr)
        b = numpy.array(value)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        expected = numpy.append(a, b)
        result = dpnp.append(ia, ib)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "arr",
        [[], [1, 2, 3], [[1, 2, 3], [4, 5, 6]]],
        ids=["empty", "1D", "2D"],
    )
    @pytest.mark.parametrize(
        "value",
        [5, [1, 2, 3], [[1, 2, 3], [4, 5, 6]]],
        ids=["scalar", "1D", "2D"],
    )
    def test_array_like_value(self, arr, value):
        a = numpy.array(arr)
        ia = dpnp.array(a)

        expected = numpy.append(a, value)
        result = dpnp.append(ia, value)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "arr",
        [[1, 2, 3], [[1, 2, 3], [4, 5, 6]]],
        ids=["1D", "2D"],
    )
    @pytest.mark.parametrize(
        "value",
        [[1, 2, 3], [[1, 2, 3], [4, 5, 6]]],
        ids=["1D", "2D"],
    )
    def test_usm_ndarray(self, arr, value):
        a = numpy.array(arr)
        b = numpy.array(value)
        ia = dpt.asarray(a)
        ib = dpt.asarray(b)

        expected = numpy.append(a, b)
        result = dpnp.append(ia, ib)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dtype2", get_all_dtypes(no_none=True))
    def test_axis(self, dtype1, dtype2):
        a = numpy.ones((2, 3), dtype=dtype1)
        b = numpy.zeros((2, 4), dtype=dtype1)
        ia = dpnp.asarray(a)
        ib = dpnp.asarray(b)

        expected = numpy.append(a, b, axis=1)
        result = dpnp.append(ia, ib, axis=1)
        assert_array_equal(result, expected)


class TestArraySplit:
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        # 0 split is not allowed
        a = xp.arange(10)
        assert_raises(ValueError, xp.array_split, a, 0)

        # invalid indices_or_sections
        a = xp.arange(10)
        assert_raises(TypeError, xp.array_split, a, "wrong")

        # non-integer sequence
        a = xp.arange(10)
        assert_raises(TypeError, xp.array_split, a, [3, 5.0])

        # not 1D array
        a = xp.arange(10)
        indices = dpnp.array([[1, 5], [7, 9]])
        assert_raises(ValueError, xp.array_split, a, indices)

    @pytest.mark.parametrize(
        "indices",
        [
            1,
            2,
            3.0,
            dpnp.int64(5),
            dpnp.int32(5),
            dpnp.array(6),
            numpy.array(7),
            numpy.int32(5),
            9,
            10,
            11,
        ],
    )
    def test_integer_split(self, indices):
        a = numpy.arange(10)
        a_dp = dpnp.array(a)

        expected = numpy.array_split(a, indices)
        result = dpnp.array_split(a_dp, indices)
        _compare_results(result, expected)

    def test_integer_split_2D_rows(self):
        a = numpy.array([numpy.arange(10), numpy.arange(10)])
        a_dp = dpnp.array(a)
        expected = numpy.array_split(a, 3, axis=0)
        result = dpnp.array_split(a_dp, 3, axis=0)
        _compare_results(result, expected)
        assert a.dtype.type is result[-1].dtype.type

        # Same thing for manual splits:
        expected = numpy.array_split(a, [0, 1], axis=0)
        result = dpnp.array_split(a_dp, [0, 1], axis=0)
        _compare_results(result, expected)
        assert a.dtype.type is result[-1].dtype.type

    def test_integer_split_2D_cols(self):
        a = numpy.array([numpy.arange(10), numpy.arange(10)])
        a_dp = dpnp.array(a)
        expected = numpy.array_split(a, 3, axis=-1)
        result = dpnp.array_split(a_dp, 3, axis=-1)
        _compare_results(result, expected)

    @testing.slow
    def test_integer_split_2D_rows_greater_max_int32(self):
        a = numpy.broadcast_to([0], (1 << 32, 2))
        a_dp = dpnp.broadcast_to(dpnp.array([0]), (1 << 32, 2))
        expected = numpy.array_split(a, 4)
        result = dpnp.array_split(a_dp, 4)
        _compare_results(result, expected)

    @pytest.mark.parametrize(
        "indices",
        [[1, 5, 7], (1, 5, 7), dpnp.array([1, 5, 7]), numpy.array([1, 5, 7])],
    )
    def test_index_split_simple(self, indices):
        a = numpy.arange(10)
        a_dp = dpnp.array(a)
        expected = numpy.array_split(a, indices, axis=-1)
        result = dpnp.array_split(a_dp, indices, axis=-1)
        _compare_results(result, expected)

    def test_index_split_low_bound(self):
        a = numpy.arange(10)
        a_dp = dpnp.array(a)
        indices = [0, 5, 7]
        expected = numpy.array_split(a, indices, axis=-1)
        result = dpnp.array_split(a_dp, indices, axis=-1)
        _compare_results(result, expected)

    def test_index_split_high_bound(self):
        a = numpy.arange(10)
        a_dp = dpnp.array(a)
        indices = [0, 5, 7, 10, 12]
        expected = numpy.array_split(a, indices, axis=-1)
        result = dpnp.array_split(a_dp, indices, axis=-1)
        _compare_results(result, expected)


class TestAsarrayCheckFinite:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_basic(self, dtype):
        a = [1, 2, 3]
        expected = numpy.asarray_chkfinite(a, dtype=dtype)
        result = dpnp.asarray_chkfinite(a, dtype=dtype)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        b = [1, 2, numpy.inf]
        c = [1, 2, numpy.nan]
        assert_raises(ValueError, xp.asarray_chkfinite, b)
        assert_raises(ValueError, xp.asarray_chkfinite, c)

    @pytest.mark.parametrize("order", ["C", "F", "A", "K"])
    def test_dtype_order(self, order):
        a = [1, 2, 3]
        expected = numpy.asarray_chkfinite(a, order=order)
        result = dpnp.asarray_chkfinite(a, order=order)
        assert_array_equal(result, expected)

    def test_no_copy(self):
        a = dpnp.ones(10)

        # No copy is performed if the input is already an ndarray
        b = dpnp.asarray_chkfinite(a)

        # b is a view of a, changing b, modifies a
        b[0::2] = 0
        assert_array_equal(b, a)


class TestBroadcast:
    @pytest.mark.parametrize(
        "shape",
        [
            [(1,), (3,)],
            [(1, 3), (3, 3)],
            [(3, 1), (3, 3)],
            [(1, 3), (3, 1)],
            [(1, 1), (3, 3)],
            [(1, 1), (1, 3)],
            [(1, 1), (3, 1)],
            [(1, 0), (0, 0)],
            [(0, 1), (0, 0)],
            [(1, 0), (0, 1)],
            [(1, 1), (0, 0)],
            [(1, 1), (1, 0)],
            [(1, 1), (0, 1)],
        ],
    )
    def test_broadcast_shapes(self, shape):
        expected = numpy.broadcast_shapes(*shape)
        result = dpnp.broadcast_shapes(*shape)
        assert_equal(result, expected)


class TestCopyTo:
    testdata = []
    testdata += [
        ([True, False, True], dtype)
        for dtype in get_all_dtypes(no_none=True, no_complex=True)
    ]
    testdata += [
        ([1, -1, 0], dtype)
        for dtype in get_integer_float_dtypes(no_unsigned=True)
    ]
    testdata += [([0.1, 0.0, -0.1], dtype) for dtype in get_float_dtypes()]
    testdata += [([1j, -1j, 1 - 2j], dtype) for dtype in get_complex_dtypes()]

    @pytest.mark.parametrize("data, dt_out", testdata)
    def test_dtype(self, data, dt_out):
        a = numpy.array(data)
        ia = dpnp.array(a)

        expected = numpy.empty(a.size, dtype=dt_out)
        result = dpnp.empty(ia.size, dtype=dt_out)
        numpy.copyto(expected, a)
        dpnp.copyto(result, ia)

        assert_array_equal(result, expected)

    @pytest.mark.parametrize("data, dt_out", testdata)
    def test_dtype_input_list(self, data, dt_out):
        expected = numpy.empty(3, dtype=dt_out)
        result = dpnp.empty(3, dtype=dt_out)
        assert isinstance(data, list)
        numpy.copyto(expected, data)
        dpnp.copyto(result, data)

        assert_array_equal(result, expected)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "data", [(1, 2, -3), [1, 2, -3]], ids=["tuple", "list"]
    )
    @pytest.mark.parametrize(
        "dst_dt", [dpnp.uint8, dpnp.uint16, dpnp.uint32, dpnp.uint64]
    )
    def test_casting_error(self, xp, data, dst_dt):
        # cannot cast to unsigned integer
        dst = xp.empty(3, dtype=dst_dt)
        assert_raises(TypeError, xp.copyto, dst, data)

    @pytest.mark.parametrize(
        "dt_out", [dpnp.uint8, dpnp.uint16, dpnp.uint32, dpnp.uint64]
    )
    def test_positive_python_scalar(self, dt_out):
        # src is python scalar and positive
        expected = numpy.empty(1, dtype=dt_out)
        result = dpnp.array(expected)
        numpy.copyto(expected, 5)
        dpnp.copyto(result, 5)

        assert_array_equal(result, expected)

    @testing.with_requires("numpy>=2.1")
    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "dst_dt", [dpnp.uint8, dpnp.uint16, dpnp.uint32, dpnp.uint64]
    )
    def test_numpy_scalar(self, xp, dst_dt):
        dst = xp.empty(1, dtype=dst_dt)
        # cannot cast from signed int to unsigned int, src is numpy scalar
        assert_raises(TypeError, xp.copyto, dst, numpy.int32(5))
        assert_raises(TypeError, xp.copyto, dst, numpy.int32(-5))

        # Python integer -5 out of bounds, src is python scalar and negative
        assert_raises(OverflowError, xp.copyto, dst, -5)

    @pytest.mark.parametrize("dst", [7, numpy.ones(10), (2, 7), [5], range(3)])
    def test_dst_raises(self, dst):
        a = dpnp.array(4)
        with pytest.raises(
            TypeError,
            match="Destination array must be any of supported type, but got",
        ):
            dpnp.copyto(dst, a)

    @pytest.mark.parametrize("where", [numpy.ones(10), (2, 7), [5], range(3)])
    def test_where_raises(self, where):
        a = dpnp.empty((2, 3))
        b = dpnp.arange(6).reshape((2, 3))

        with pytest.raises(
            TypeError,
            match="`where` array must be any of supported type, but got",
        ):
            dpnp.copyto(a, b, where=where)


class TestDelete:
    @pytest.mark.parametrize(
        "obj", [slice(0, 4, 2), 3, [2, 3]], ids=["slice", "int", "list"]
    )
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_dtype(self, dt, obj):
        a = numpy.array([0, 1, 2, 3, 4, 5], dtype=dt)
        a_dp = dpnp.array(a)

        expected = numpy.delete(a, obj)
        result = dpnp.delete(a_dp, obj)
        assert result.dtype == dt
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("start", [-6, -2, 0, 1, 2, 4, 5])
    @pytest.mark.parametrize("stop", [-6, -2, 0, 1, 2, 4, 5])
    @pytest.mark.parametrize("step", [-3, -1, 1, 3])
    def test_slice_1D(self, start, stop, step):
        indices = slice(start, stop, step)
        # 1D array
        a = numpy.arange(5)
        a_dp = dpnp.array(a)
        expected = numpy.delete(a, indices)
        result = dpnp.delete(a_dp, indices)
        assert_array_equal(result, expected)

        # N-D array
        a = numpy.arange(10).reshape(1, 5, 2)
        a_dp = dpnp.array(a)
        for axis in [None, 1, -1]:
            expected = numpy.delete(a, indices, axis=axis)
            result = dpnp.delete(a_dp, indices, axis=axis)
            assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "indices", [0, -4, [], [0, -1, 2, 2], [True, False, False, True, False]]
    )
    def test_indices_1D(self, indices):
        # 1D array
        a = numpy.arange(5)
        a_dp = dpnp.array(a)
        expected = numpy.delete(a, indices)
        result = dpnp.delete(a_dp, indices)
        assert_array_equal(result, expected)

        # N-D array
        a = numpy.arange(10).reshape(1, 5, 2)
        a_dp = dpnp.array(a)
        expected = numpy.delete(a, indices, axis=1)
        result = dpnp.delete(a_dp, indices, axis=1)
        assert_array_equal(result, expected)

    def test_obj_ndarray(self):
        # 1D array
        a = numpy.arange(5)
        ind = numpy.array([[0, 1], [2, 1]])
        a_dp = dpnp.array(a)
        ind_dp = dpnp.array(ind)

        expected = numpy.delete(a, ind)
        # both numpy.ndarray and dpnp.ndarray are supported for obj in dpnp
        for indices in [ind, ind_dp]:
            result = dpnp.delete(a_dp, indices)
            assert_array_equal(result, expected)

        # N-D array
        b = numpy.arange(10).reshape(1, 5, 2)
        b_dp = dpnp.array(b)
        expected = numpy.delete(b, ind, axis=1)
        for indices in [ind, ind_dp]:
            result = dpnp.delete(b_dp, indices, axis=1)
            assert_array_equal(result, expected)

    def test_error(self):
        a = dpnp.arange(5)
        # out of bounds index
        with pytest.raises(IndexError):
            dpnp.delete(a, [100])
        with pytest.raises(IndexError):
            dpnp.delete(a, [-100])

        # boolean array argument obj must be one dimensional
        with pytest.raises(ValueError):
            dpnp.delete(a, True)

        # not enough items
        with pytest.raises(ValueError):
            dpnp.delete(a, [False] * 4)

        # 0-D array
        a = dpnp.array(1)
        with pytest.raises(AxisError):
            dpnp.delete(a, [], axis=0)
        with pytest.raises(TypeError):
            dpnp.delete(a, [], axis="nonsense")

        # index float
        a = dpnp.array([1, 2, 3])
        with pytest.raises(IndexError):
            dpnp.delete(a, dpnp.array([1.0, 2.0]))
        with pytest.raises(IndexError):
            dpnp.delete(a, dpnp.array([], dtype=dpnp.float32))

    @pytest.mark.parametrize("order", ["C", "F"])
    def test_order(self, order):
        a = numpy.arange(10).reshape(2, 5, order=order)
        a_dp = dpnp.array(a)

        expected = numpy.delete(a, slice(3, None), axis=1)
        result = dpnp.delete(a_dp, slice(3, None), axis=1)

        assert_equal(result.flags.c_contiguous, expected.flags.c_contiguous)
        assert_equal(result.flags.f_contiguous, expected.flags.f_contiguous)

    @pytest.mark.parametrize("indexer", [1, dpnp.array([1]), [1]])
    def test_single_item_array(self, indexer):
        a = numpy.arange(5)
        a_dp = dpnp.array(a)
        expected = numpy.delete(a, 1)
        result = dpnp.delete(a_dp, indexer)
        assert_equal(result, expected)

        b = numpy.arange(10).reshape(1, 5, 2)
        b_dp = dpnp.array(b)
        expected = numpy.delete(b, 1, axis=1)
        result = dpnp.delete(b_dp, indexer, axis=1)
        assert_equal(result, expected)

    @pytest.mark.parametrize("flag", [True, False])
    def test_boolean_obj(self, flag):
        expected = numpy.delete(numpy.ones(1), numpy.array([flag]))
        result = dpnp.delete(dpnp.ones(1), dpnp.array([flag]))
        assert_array_equal(result, expected)

        expected = numpy.delete(
            numpy.ones((3, 1)), numpy.array([flag]), axis=-1
        )
        result = dpnp.delete(dpnp.ones((3, 1)), dpnp.array([flag]), axis=-1)
        assert_array_equal(result, expected)


class TestDsplit:
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        # 0D array
        a = xp.array(1)
        assert_raises(ValueError, xp.dsplit, a, 2)

        # 1D array
        a = xp.array([1, 2, 3, 4])
        assert_raises(ValueError, xp.dsplit, a, 2)

        # 2D array
        a = xp.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        assert_raises(ValueError, xp.dsplit, a, 2)

    def test_3D_array(self):
        a = numpy.array(
            [[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]]
        )
        a_dp = dpnp.array(a)

        expected = numpy.dsplit(a, 2)
        result = dpnp.dsplit(a_dp, 2)
        _compare_results(result, expected)


class TestInsert:
    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize(
        "obj", [slice(0, 4, 2), 3, [2, 3]], ids=["slice", "scalar", "list"]
    )
    @pytest.mark.parametrize("dt1", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dt2", get_all_dtypes(no_none=True))
    def test_dtype(self, obj, dt1, dt2):
        a = numpy.array([0, 1, 2, 3, 4, 5], dtype=dt1)
        a_dp = dpnp.array(a)

        values = numpy.array(3, dtype=dt2)

        expected = numpy.insert(a, obj, values)
        result = dpnp.insert(a_dp, obj, values)
        assert result.dtype == dt1
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("order", ["C", "F"])
    def test_order(self, order):
        a = numpy.arange(10).reshape(2, 5, order=order)
        a_dp = dpnp.array(a)

        expected = numpy.insert(a, slice(3, None), -1, axis=1)
        result = dpnp.insert(a_dp, slice(3, None), -1, axis=1)

        assert_equal(result.flags.c_contiguous, expected.flags.c_contiguous)
        assert_equal(result.flags.f_contiguous, expected.flags.f_contiguous)

    @pytest.mark.parametrize(
        "obj",
        [numpy.array([2]), dpnp.array([0, 2]), dpnp.asarray([1])],
    )
    @pytest.mark.parametrize(
        "values",
        [numpy.array([-1]), dpnp.array([-2, -3])],
    )
    def test_ndarray_obj_values(self, obj, values):
        a = numpy.array([1, 2, 3])
        ia = dpnp.array(a)

        values_np = (
            values if isinstance(values, numpy.ndarray) else values.asnumpy()
        )
        obj_np = obj if isinstance(obj, numpy.ndarray) else obj.asnumpy()
        expected = numpy.insert(a, obj_np, values_np)
        result = dpnp.insert(ia, obj, values)
        assert_equal(result, expected)

    @testing.with_requires("numpy>=2.2")
    @pytest.mark.parametrize(
        "obj",
        [[False], numpy.array([True] * 4), [True, False, True, False]],
    )
    def test_boolean_obj(self, obj):
        if not isinstance(obj, numpy.ndarray):
            # numpy.insert raises exception
            # TODO: remove once NumPy resolves that
            obj = numpy.array(obj)

        a = numpy.array([1, 2, 3])
        ia = dpnp.array(a)
        assert_equal(dpnp.insert(ia, obj, 9), numpy.insert(a, obj, 9))

    @testing.with_requires("numpy>=2.2")
    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "obj_data",
        [True, [[True, False], [True, False]]],
        ids=["0d", "2d"],
    )
    def test_boolean_obj_error(self, xp, obj_data):
        a = xp.array([1, 2, 3])
        obj = xp.array(obj_data)
        with pytest.raises(ValueError):
            xp.insert(a, obj, 9)

    def test_1D_array(self):
        a = numpy.array([1, 2, 3])
        ia = dpnp.array(a)

        assert_equal(dpnp.insert(ia, 0, 1), numpy.insert(a, 0, 1))
        assert_equal(dpnp.insert(ia, 3, 1), numpy.insert(a, 3, 1))
        assert_equal(
            dpnp.insert(ia, -1, [1, 2, 3]), numpy.insert(a, -1, [1, 2, 3])
        )
        assert_equal(
            dpnp.insert(ia, [1, -1, 3], 9), numpy.insert(a, [1, -1, 3], 9)
        )

        expected = numpy.insert(a, [1, 1, 1], [1, 2, 3])
        result = dpnp.insert(ia, [1, 1, 1], [1, 2, 3])
        assert_equal(result, expected)

        expected = numpy.insert(a, slice(-1, None, -1), 9)
        result = dpnp.insert(ia, slice(-1, None, -1), 9)
        assert_equal(result, expected)

        expected = numpy.insert(a, slice(None, 1, None), 9)
        result = dpnp.insert(ia, slice(None, 1, None), 9)
        assert_equal(result, expected)

        expected = numpy.insert(a, [-1, 1, 3], [7, 8, 9])
        result = dpnp.insert(ia, [-1, 1, 3], [7, 8, 9])
        assert_equal(result, expected)

        b = numpy.array([0, 1], dtype=numpy.float32)
        ib = dpnp.array(b)
        assert_equal(dpnp.insert(ib, 0, ib[0]), numpy.insert(b, 0, b[0]))
        assert_equal(dpnp.insert(ib, [], []), numpy.insert(b, [], []))

    def test_ND_array(self):
        a = numpy.array([[1, 1, 1]])
        ia = dpnp.array(a)

        assert_equal(dpnp.insert(ia, 0, [1]), numpy.insert(a, 0, [1]))
        assert_equal(
            dpnp.insert(ia, 0, 2, axis=0), numpy.insert(a, 0, 2, axis=0)
        )
        assert_equal(
            dpnp.insert(ia, 2, 2, axis=1), numpy.insert(a, 2, 2, axis=1)
        )
        expected = numpy.insert(a, 0, [2, 2, 2], axis=0)
        result = dpnp.insert(ia, 0, [2, 2, 2], axis=0)
        assert_equal(result, expected)

        a = numpy.array([[1, 1], [2, 2], [3, 3]])
        ia = dpnp.array(a)
        expected = numpy.insert(a, [1], [[1], [2], [3]], axis=1)
        result = dpnp.insert(ia, [1], [[1], [2], [3]], axis=1)
        assert_equal(result, expected)

        expected = numpy.insert(a, [1], [1, 2, 3], axis=1)
        result = dpnp.insert(ia, [1], [1, 2, 3], axis=1)
        assert_equal(result, expected)
        # scalars behave differently
        expected = numpy.insert(a, 1, [1, 2, 3], axis=1)
        result = dpnp.insert(ia, 1, [1, 2, 3], axis=1)
        assert_equal(result, expected)

        expected = numpy.insert(a, 1, [[1], [2], [3]], axis=1)
        result = dpnp.insert(ia, 1, [[1], [2], [3]], axis=1)
        assert_equal(result, expected)

        a = numpy.arange(4).reshape(2, 2)
        ia = dpnp.array(a)
        expected = numpy.insert(a[:, :1], 1, a[:, 1], axis=1)
        result = dpnp.insert(ia[:, :1], 1, ia[:, 1], axis=1)
        assert_equal(result, expected)

        expected = numpy.insert(a[:1, :], 1, a[1, :], axis=0)
        result = dpnp.insert(ia[:1, :], 1, ia[1, :], axis=0)
        assert_equal(result, expected)

        # negative axis value
        a = numpy.arange(24).reshape((2, 3, 4))
        ia = dpnp.array(a)
        expected = numpy.insert(a, 1, a[:, :, 3], axis=-1)
        result = dpnp.insert(ia, 1, ia[:, :, 3], axis=-1)
        assert_equal(result, expected)

        expected = numpy.insert(a, 1, a[:, 2, :], axis=-2)
        result = dpnp.insert(ia, 1, ia[:, 2, :], axis=-2)
        assert_equal(result, expected)

        # invalid axis value
        assert_raises(AxisError, dpnp.insert, ia, 1, ia[:, 2, :], axis=3)
        assert_raises(AxisError, dpnp.insert, ia, 1, ia[:, 2, :], axis=-4)

    def test_index_array_copied(self):
        a = dpnp.array([0, 1, 2])
        x = dpnp.array([1, 1, 1])
        dpnp.insert(a, x, [3, 4, 5])
        assert_equal(x, dpnp.array([1, 1, 1]))

    def test_error(self):
        a = dpnp.array([0, 1, 2])

        # index float
        with pytest.raises(IndexError):
            dpnp.insert(a, dpnp.array([1.0, 2.0]), [10, 20])
        with pytest.raises(IndexError):
            dpnp.insert(a, dpnp.array([], dtype=dpnp.float32), [])

        # index 2d
        with pytest.raises(ValueError):
            dpnp.insert(a, dpnp.array([[1.0], [2.0]]), [10, 20])

        # incorrect axis
        a = dpnp.array(1)
        with pytest.raises(AxisError):
            dpnp.insert(a, [], 2, axis=0)
        with pytest.raises(TypeError):
            dpnp.insert(a, [], 2, axis="nonsense")

    @pytest.mark.parametrize("idx", [4, -4])
    def test_index_out_of_bounds(self, idx):
        a = dpnp.array([0, 1, 2])
        with pytest.raises(IndexError, match="out of bounds"):
            dpnp.insert(a, [idx], [3, 4])


# array_split has more comprehensive test of splitting.
# only do simple test on hsplit, vsplit, and dsplit
class TestHsplit:
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        # 0D array
        a = xp.array(1)
        assert_raises(ValueError, xp.hsplit, a, 2)

    def test_1D_array(self):
        a = numpy.array([1, 2, 3, 4])
        a_dp = dpnp.array(a)

        expected = numpy.hsplit(a, 2)
        result = dpnp.hsplit(a_dp, 2)
        _compare_results(result, expected)

    def test_2D_array(self):
        a = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        a_dp = dpnp.array(a)

        expected = numpy.hsplit(a, 2)
        result = dpnp.hsplit(a_dp, 2)
        _compare_results(result, expected)


class TestRavel:
    def test_error(self):
        ia = dpnp.arange(10).reshape(2, 5)
        assert_raises(NotImplementedError, dpnp.ravel, ia, order="K")

    @pytest.mark.parametrize("order", ["C", "F", "A"])
    def test_non_contiguous(self, order):
        a = numpy.arange(10)[::2]
        ia = dpnp.arange(10)[::2]
        expected = numpy.ravel(a, order=order)
        result = dpnp.ravel(ia, order=order)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_array_equal(result, expected)


class TestRepeat:
    @pytest.mark.parametrize(
        "data",
        [[], [1, 2, 3, 4], [[1, 2], [3, 4]], [[[1], [2]], [[3], [4]]]],
        ids=[
            "[]",
            "[1, 2, 3, 4]",
            "[[1, 2], [3, 4]]",
            "[[[1], [2]], [[3], [4]]]",
        ],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_data(self, data, dtype):
        a = numpy.array(data, dtype=dtype)
        ia = dpnp.array(a)

        expected = numpy.repeat(a, 2)
        result = dpnp.repeat(ia, 2)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "repeats", [2, (2, 2, 2, 2, 2)], ids=["scalar", "tuple"]
    )
    def test_scalar_sequence_agreement(self, repeats):
        a = numpy.arange(5, dtype="i4")
        ia = dpnp.array(a)

        expected = numpy.repeat(a, repeats)
        result = dpnp.repeat(ia, repeats)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_broadcasting(self, axis):
        reps = 5
        a = numpy.arange(reps, dtype="i4")
        if axis == 0:
            sh = (reps, 1)
        else:
            sh = (1, reps)
        a = a.reshape(sh)
        ia = dpnp.array(a)

        expected = numpy.repeat(a, reps)
        result = dpnp.repeat(ia, reps)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_axes(self, axis):
        reps = 2
        a = numpy.arange(5 * 10, dtype="i4").reshape((5, 10))
        ia = dpnp.array(a)

        expected = numpy.repeat(a, reps, axis=axis)
        result = dpnp.repeat(ia, reps, axis=axis)
        assert_array_equal(result, expected)

    def test_size_0_outputs(self):
        reps = 10
        a = dpnp.ones((3, 0, 5), dtype="i4")
        ia = dpnp.array(a)

        expected = numpy.repeat(a, reps, axis=0)
        result = dpnp.repeat(ia, reps, axis=0)
        assert_array_equal(result, expected)

        expected = numpy.repeat(a, reps, axis=1)
        result = dpnp.repeat(ia, reps, axis=1)
        assert_array_equal(result, expected)

        reps = (2, 2, 2)
        expected = numpy.repeat(a, reps, axis=0)
        result = dpnp.repeat(ia, reps, axis=0)
        assert_array_equal(result, expected)

        a = numpy.ones((3, 2, 5))
        ia = dpnp.array(a)

        reps = 0
        expected = numpy.repeat(a, reps, axis=1)
        result = dpnp.repeat(ia, reps, axis=1)
        assert_array_equal(result, expected)

        reps = (0, 0)
        expected = numpy.repeat(a, reps, axis=1)
        result = dpnp.repeat(ia, reps, axis=1)
        assert_array_equal(result, expected)

    def test_strides_0(self):
        reps = 2
        a = numpy.arange(10 * 10, dtype="i4").reshape((10, 10))
        ia = dpnp.array(a)

        a = a[::-2, :]
        ia = ia[::-2, :]

        expected = numpy.repeat(a, reps, axis=0)
        result = dpnp.repeat(ia, reps, axis=0)
        assert_array_equal(result, expected)

        expected = numpy.repeat(a, (reps,) * a.shape[0], axis=0)
        result = dpnp.repeat(ia, (reps,) * ia.shape[0], axis=0)
        assert_array_equal(result, expected)

    def test_strides_1(self):
        reps = 2
        a = numpy.arange(10 * 10, dtype="i4").reshape((10, 10))
        ia = dpnp.array(a)

        a = a[:, ::-2]
        ia = ia[:, ::-2]

        expected = numpy.repeat(a, reps, axis=1)
        result = dpnp.repeat(ia, reps, axis=1)
        assert_array_equal(result, expected)

        expected = numpy.repeat(a, (reps,) * a.shape[1], axis=1)
        result = dpnp.repeat(ia, (reps,) * ia.shape[1], axis=1)
        assert_array_equal(result, expected)

    def test_casting(self):
        a = numpy.arange(5, dtype="i4")
        ia = dpnp.array(a)

        # i4 is cast to i8
        reps = numpy.ones(5, dtype="i4")
        ireps = dpnp.array(reps)

        expected = numpy.repeat(a, reps)
        result = dpnp.repeat(ia, ireps)
        assert_array_equal(result, expected)

    def test_strided_repeats(self):
        a = numpy.arange(5, dtype="i4")
        ia = dpnp.array(a)

        reps = numpy.ones(10, dtype="i8")
        reps[::2] = 0
        ireps = dpnp.array(reps)

        reps = reps[::-2]
        ireps = ireps[::-2]

        expected = numpy.repeat(a, reps)
        result = dpnp.repeat(ia, ireps)
        assert_array_equal(result, expected)

    def test_usm_ndarray_as_input_array(self):
        reps = [1, 3, 2, 1, 1, 2]
        a = numpy.array([[1, 2, 3, 4, 5, 6]])
        ia = dpt.asarray(a)

        expected = numpy.repeat(a, reps)
        result = dpnp.repeat(ia, reps)
        assert_array_equal(result, expected)
        assert isinstance(result, dpnp.ndarray)

    def test_scalar_as_input_array(self):
        assert_raises(TypeError, dpnp.repeat, 3, 2)

    def test_usm_ndarray_as_repeats(self):
        a = numpy.array([1, 2, 3, 4, 5, 6]).reshape((2, 3))
        ia = dpnp.asarray(a)

        reps = numpy.array([1, 3, 2])
        ireps = dpt.asarray(reps)

        expected = a.repeat(reps, axis=1)
        result = ia.repeat(ireps, axis=1)
        assert_array_equal(result, expected)
        assert isinstance(result, dpnp.ndarray)

    def test_unsupported_array_as_repeats(self):
        assert_raises(TypeError, dpnp.arange(5, dtype="i4"), numpy.array(3))

    @pytest.mark.parametrize(
        "data, dtype",
        [
            pytest.param([1, 2**7 - 1, -(2**7)], numpy.int8, id="int8"),
            pytest.param([1, 2**15 - 1, -(2**15)], numpy.int16, id="int16"),
            pytest.param([1, 2**31 - 1, -(2**31)], numpy.int32, id="int32"),
            pytest.param([1, 2**63 - 1, -(2**63)], numpy.int64, id="int64"),
        ],
    )
    def test_maximum_signed_integers(self, data, dtype):
        reps = 129
        a = numpy.array(data, dtype=dtype)
        ia = dpnp.asarray(a)

        expected = a.repeat(reps)
        result = ia.repeat(reps)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "data, dtype",
        [
            pytest.param(
                [1, -(2**7), -(2**7) + 1, 2**7 - 1], numpy.int8, id="int8"
            ),
            pytest.param(
                [1, -(2**15), -(2**15) + 1, 2**15 - 1], numpy.int16, id="int16"
            ),
            pytest.param(
                [1, -(2**31), -(2**31) + 1, 2**31 - 1], numpy.int32, id="int32"
            ),
            pytest.param(
                [1, -(2**63), -(2**63) + 1, 2**63 - 1], numpy.int64, id="int64"
            ),
        ],
    )
    def test_minimum_signed_integers(self, data, dtype):
        reps = 129
        a = numpy.array(data, dtype=dtype)
        ia = dpnp.asarray(a)

        expected = a.repeat(reps)
        result = ia.repeat(reps)
        assert_array_equal(result, expected)


class TestRequire:
    flag_names = ["C", "C_CONTIGUOUS", "F", "F_CONTIGUOUS", "W"]

    def generate_all_false(self, dtype):
        a_np = numpy.zeros((10, 10), dtype=dtype)
        a_dp = dpnp.zeros((10, 10), dtype=dtype)
        a_np = a_np[::2, ::2]
        a_dp = a_dp[::2, ::2]
        a_np.flags["W"] = False
        a_dp.flags["W"] = False
        assert not a_dp.flags["C"]
        assert not a_dp.flags["F"]
        assert not a_dp.flags["W"]
        return a_np, a_dp

    def set_and_check_flag(self, flag, dtype, arr):
        if dtype is None:
            dtype = arr[1].dtype
        result = numpy.require(arr[0], dtype, [flag])
        expected = dpnp.require(arr[1], dtype, [flag])
        assert result.flags[flag] == expected.flags[flag]
        assert result.dtype == expected.dtype

        # a further call to dpnp.require ought to return the same array
        c = dpnp.require(expected, None, [flag])
        assert c is expected

    def test_require_each(self):
        id = ["f4", "i4"]
        fd = [None, "f4", "c8"]
        for idtype, fdtype, flag in itertools.product(id, fd, self.flag_names):
            a = self.generate_all_false(idtype)
            self.set_and_check_flag(flag, fdtype, a)

    def test_unknown_requirement(self):
        a = self.generate_all_false("f4")
        assert_raises(KeyError, numpy.require, a[0], None, "Q")
        assert_raises(ValueError, dpnp.require, a[1], None, "Q")

    def test_non_array_input(self):
        a_np = numpy.array([1, 2, 3, 4])
        a_dp = dpnp.array(a_np)
        expected = numpy.require(a_np, "i4", ["C", "W"])
        result = dpnp.require(a_dp, "i4", ["C", "W"])
        assert expected.flags["C"] == result.flags["C"]
        assert expected.flags["F"] == result.flags["F"]
        assert expected.flags["W"] == result.flags["W"]
        assert_array_equal(result, expected)

    def test_C_and_F_simul(self):
        a = self.generate_all_false("f4")
        assert_raises(ValueError, numpy.require, a[0], None, ["C", "F"])
        assert_raises(ValueError, dpnp.require, a[1], None, ["C", "F"])

    def test_copy(self):
        a_np = numpy.arange(6).reshape(2, 3)
        a_dp = dpnp.arange(6).reshape(2, 3)
        a_np.flags["W"] = False
        a_dp.flags["W"] = False
        expected = numpy.require(a_np, requirements=["W", "C"])
        result = dpnp.require(a_dp, requirements=["W", "C"])
        # copy is done
        assert result is not a_dp
        assert_array_equal(result, expected)


class TestReshape:
    def test_error(self):
        ia = dpnp.arange(10)
        assert_raises(TypeError, dpnp.reshape, ia)
        assert_raises(
            TypeError, dpnp.reshape, ia, shape=(2, 5), newshape=(2, 5)
        )

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_newshape(self):
        a = numpy.arange(10)
        ia = dpnp.array(a)
        expected = numpy.reshape(a, (2, 5))
        result = dpnp.reshape(ia, newshape=(2, 5))
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("order", [None, "C", "F", "A"])
    def test_order(self, order):
        a = numpy.arange(10)
        ia = dpnp.array(a)
        expected = numpy.reshape(a, (2, 5), order)
        result = dpnp.reshape(ia, (2, 5), order)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_array_equal(result, expected)

        # ndarray
        result = ia.reshape(2, 5, order=order)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_array_equal(result, expected)

    def test_ndarray(self):
        a = numpy.arange(10)
        ia = dpnp.array(a)
        expected = a.reshape(2, 5)
        result = ia.reshape(2, 5)
        assert_array_equal(result, expected)

        # packed
        result = ia.reshape((2, 5))
        assert_array_equal(result, expected)

    @testing.with_requires("numpy>=2.1")
    def test_copy(self):
        a = numpy.arange(10).reshape(2, 5)
        ia = dpnp.array(a)
        expected = numpy.reshape(a, 10, copy=None)
        expected[0] = -1
        result = dpnp.reshape(ia, 10, copy=None)
        result[0] = -1
        assert a[0, 0] == expected[0]  # a is also modified, no copy
        assert ia[0, 0] == result[0]  # ia is also modified, no copy
        assert_array_equal(result, expected)

        a = numpy.arange(10).reshape(2, 5)
        ia = dpnp.array(a)
        expected = numpy.reshape(a, 10, copy=True)
        expected[0] = -1
        result = dpnp.reshape(ia, 10, copy=True)
        result[0] = -1
        assert a[0, 0] != expected[0]  # a is not modified, copy is done
        assert ia[0, 0] != result[0]  # ia is not modified, copy is done
        assert_array_equal(result, expected)

        a = numpy.arange(10).reshape(2, 5)
        ia = dpnp.array(a)
        assert_raises(
            ValueError, dpnp.reshape, ia, (5, 2), order="F", copy=False
        )
        assert_raises(
            ValueError, dpnp.reshape, ia, (5, 2), order="F", copy=False
        )


class TestResize:
    @pytest.mark.parametrize(
        "data, shape",
        [
            pytest.param([[1, 2], [3, 4]], (2, 4)),
            pytest.param([[1, 2], [3, 4], [1, 2], [3, 4]], (4, 2)),
            pytest.param([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]], (4, 3)),
        ],
    )
    def test_copies(self, data, shape):
        a = numpy.array(data)
        ia = dpnp.array(a)
        assert_equal(dpnp.resize(ia, shape), numpy.resize(a, shape))

    @pytest.mark.parametrize("newshape", [(2, 4), [2, 4], (10,), 10])
    def test_newshape_type(self, newshape):
        a = numpy.array([[1, 2], [3, 4]])
        ia = dpnp.array(a)
        assert_equal(dpnp.resize(ia, newshape), numpy.resize(a, newshape))

    @pytest.mark.parametrize(
        "data, shape",
        [
            pytest.param([1, 2, 3], (2, 4)),
            pytest.param([[1, 2], [3, 1], [2, 3], [1, 2]], (4, 2)),
            pytest.param([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], (4, 3)),
        ],
    )
    def test_repeats(self, data, shape):
        a = numpy.array(data)
        ia = dpnp.array(a)
        assert_equal(dpnp.resize(ia, shape), numpy.resize(a, shape))

    def test_zeroresize(self):
        a = numpy.array([[1, 2], [3, 4]])
        ia = dpnp.array(a)
        assert_array_equal(dpnp.resize(ia, (0,)), numpy.resize(a, (0,)))
        assert_equal(a.dtype, ia.dtype)

        assert_equal(dpnp.resize(ia, (0, 2)), numpy.resize(a, (0, 2)))
        assert_equal(dpnp.resize(ia, (2, 0)), numpy.resize(a, (2, 0)))

    def test_reshape_from_zero(self):
        a = numpy.zeros(0, dtype=numpy.float32)
        ia = dpnp.array(a)
        assert_array_equal(dpnp.resize(ia, (2, 1)), numpy.resize(a, (2, 1)))
        assert_equal(a.dtype, ia.dtype)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_negative_resize(self, xp):
        a = xp.arange(0, 10, dtype=xp.float32)
        new_shape = (-10, -1)
        with pytest.raises(ValueError, match=r"negative"):
            xp.resize(a, new_shape=new_shape)


class TestRot90:
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        assert_raises(ValueError, xp.rot90, xp.ones(4))
        assert_raises(ValueError, xp.rot90, xp.ones((2, 2, 2)), axes=(0, 1, 2))
        assert_raises(ValueError, xp.rot90, xp.ones((2, 2)), axes=(0, 2))
        assert_raises(ValueError, xp.rot90, xp.ones((2, 2)), axes=(1, 1))
        assert_raises(ValueError, xp.rot90, xp.ones((2, 2, 2)), axes=(-2, 1))

    def test_error_float_k(self):
        assert_raises(TypeError, dpnp.rot90, dpnp.ones((2, 2)), k=2.5)

    def test_basic(self):
        a = numpy.array([[0, 1, 2], [3, 4, 5]])
        ia = dpnp.array(a)

        for k in range(-3, 13, 4):
            assert_equal(dpnp.rot90(ia, k=k), numpy.rot90(a, k=k))
        for k in range(-2, 13, 4):
            assert_equal(dpnp.rot90(ia, k=k), numpy.rot90(a, k=k))
        for k in range(-1, 13, 4):
            assert_equal(dpnp.rot90(ia, k=k), numpy.rot90(a, k=k))
        for k in range(0, 13, 4):
            assert_equal(dpnp.rot90(ia, k=k), numpy.rot90(a, k=k))

        assert_equal(dpnp.rot90(dpnp.rot90(ia, axes=(0, 1)), axes=(1, 0)), ia)
        assert_equal(
            dpnp.rot90(ia, k=1, axes=(1, 0)), dpnp.rot90(ia, k=-1, axes=(0, 1))
        )

    def test_axes(self):
        a = numpy.ones((50, 40, 3))
        ia = dpnp.array(a)
        assert_equal(dpnp.rot90(ia), numpy.rot90(a))
        assert_equal(dpnp.rot90(ia, axes=(0, 2)), dpnp.rot90(ia, axes=(0, -1)))
        assert_equal(dpnp.rot90(ia, axes=(1, 2)), dpnp.rot90(ia, axes=(-2, -1)))

    @pytest.mark.parametrize(
        "axes", [(1, 2), [1, 2], numpy.array([1, 2]), dpnp.array([1, 2])]
    )
    def test_axes_type(self, axes):
        a = numpy.ones((50, 40, 3))
        ia = dpnp.array(a)
        assert_equal(
            dpnp.rot90(ia, axes=axes),
            numpy.rot90(a, axes=get_array(numpy, axes)),
        )

    def test_rotation_axes(self):
        a = numpy.arange(8).reshape((2, 2, 2))
        ia = dpnp.array(a)

        assert_equal(dpnp.rot90(ia, axes=(0, 1)), numpy.rot90(a, axes=(0, 1)))
        assert_equal(dpnp.rot90(ia, axes=(1, 0)), numpy.rot90(a, axes=(1, 0)))
        assert_equal(dpnp.rot90(ia, axes=(1, 2)), numpy.rot90(a, axes=(1, 2)))

        for k in range(1, 5):
            assert_equal(
                dpnp.rot90(ia, k=k, axes=(2, 0)),
                numpy.rot90(a, k=k, axes=(2, 0)),
            )


class TestSplit:
    # The split function is essentially the same as array_split,
    # except that it test if splitting will result in an
    # equal split. Only test for this case.
    def test_equal_split(self):
        a = numpy.arange(10)
        a_dp = dpnp.array(a)

        expected = numpy.split(a, 2)
        result = dpnp.split(a_dp, 2)
        _compare_results(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_unequal_split(self, xp):
        a = xp.arange(10)
        assert_raises(ValueError, xp.split, a, 3)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        # axis out of range
        a = xp.arange(9)
        assert_raises(IndexError, xp.split, a, 3, axis=1)

    @pytest.mark.parametrize(
        "indices",
        [
            2,
            3.0,
            dpnp.int64(5),
            dpnp.int32(5),
            dpnp.array(6),
            numpy.array(7),
            numpy.int32(5),
        ],
    )
    def test_integer_split(self, indices):
        a = numpy.arange(10)
        a_dp = dpnp.array(a)

        expected = numpy.array_split(a, indices)
        result = dpnp.array_split(a_dp, indices)
        _compare_results(result, expected)


class TestTranspose:
    @pytest.mark.parametrize("axes", [(0, 1), (1, 0), [0, 1]])
    def test_2d_with_axes(self, axes):
        na = numpy.array([[1, 2], [3, 4]])
        da = dpnp.array(na)

        expected = numpy.transpose(na, axes)
        result = dpnp.transpose(da, axes)
        assert_array_equal(result, expected)

        # ndarray
        expected = na.transpose(axes)
        result = da.transpose(axes)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            (1, 0, 2),
            [1, 0, 2],
            ((1, 0, 2),),
            ([1, 0, 2],),
            [(1, 0, 2)],
            [[1, 0, 2]],
        ],
    )
    def test_3d_with_packed_axes(self, axes):
        na = numpy.ones((1, 2, 3))
        da = dpnp.array(na)

        expected = na.transpose(*axes)
        result = da.transpose(*axes)
        assert_array_equal(result, expected)

        # ndarray
        expected = na.transpose(*axes)
        result = da.transpose(*axes)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("shape", [(10,), (2, 4), (5, 3, 7), (3, 8, 4, 1)])
    def test_none_axes(self, shape):
        na = numpy.ones(shape)
        da = dpnp.ones(shape)

        assert_array_equal(numpy.transpose(na), dpnp.transpose(da))
        assert_array_equal(numpy.transpose(na, None), dpnp.transpose(da, None))

        # ndarray
        assert_array_equal(na.transpose(), da.transpose())
        assert_array_equal(na.transpose(None), da.transpose(None))

    def test_ndarray_axes_n_int(self):
        na = numpy.ones((1, 2, 3))
        da = dpnp.array(na)

        expected = na.transpose(1, 0, 2)
        result = da.transpose(1, 0, 2)
        assert_array_equal(result, expected)

    def test_alias(self):
        a = dpnp.arange(15).reshape(5, 3)

        res1 = dpnp.transpose(a)
        res2 = dpnp.permute_dims(a)
        assert_array_equal(res1, res2)

    def test_usm_array(self):
        a = numpy.arange(9).reshape(3, 3)
        ia = dpt.asarray(a)

        expected = numpy.transpose(a)
        result = dpnp.transpose(ia)
        assert_array_equal(result, expected)


class TestTrimZeros:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, dtype):
        a = numpy.array([0, 0, 1, 0, 2, 3, 4, 0], dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.trim_zeros(ia)
        expected = numpy.trim_zeros(a)
        assert_array_equal(result, expected)

    @testing.with_requires("numpy>=2.2")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("trim", ["F", "B", "fb"])
    @pytest.mark.parametrize("ndim", [0, 1, 2, 3])
    def test_basic_nd(self, dtype, trim, ndim):
        a = numpy.ones((2,) * ndim, dtype=dtype)
        a = numpy.pad(a, (2, 1), mode="constant", constant_values=0)
        ia = dpnp.array(a)

        for axis in list(range(ndim)) + [None]:
            result = dpnp.trim_zeros(ia, trim=trim, axis=axis)
            expected = numpy.trim_zeros(a, trim=trim, axis=axis)
            assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("trim", ["F", "B"])
    def test_trim(self, dtype, trim):
        a = numpy.array([0, 0, 1, 0, 2, 3, 4, 0], dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.trim_zeros(ia, trim)
        expected = numpy.trim_zeros(a, trim)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("trim", ["F", "B"])
    def test_all_zero(self, dtype, trim):
        a = numpy.zeros((8,), dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.trim_zeros(ia, trim)
        expected = numpy.trim_zeros(a, trim)
        assert_array_equal(result, expected)

    @testing.with_requires("numpy>=2.2")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("trim", ["F", "B", "fb"])
    @pytest.mark.parametrize("ndim", [0, 1, 2, 3])
    def test_all_zero_nd(self, dtype, trim, ndim):
        a = numpy.zeros((3,) * ndim, dtype=dtype)
        ia = dpnp.array(a)

        for axis in list(range(ndim)) + [None]:
            result = dpnp.trim_zeros(ia, trim=trim, axis=axis)
            expected = numpy.trim_zeros(a, trim=trim, axis=axis)
            assert_array_equal(result, expected)

    def test_size_zero(self):
        a = numpy.zeros(0)
        ia = dpnp.array(a)

        result = dpnp.trim_zeros(ia)
        expected = numpy.trim_zeros(a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "a", [numpy.array([0, 2**62, 0]), numpy.array([0, 2**63, 0])]
    )
    def test_overflow(self, a):
        ia = dpnp.array(a)

        result = dpnp.trim_zeros(ia)
        expected = numpy.trim_zeros(a)
        assert_array_equal(result, expected)

    @testing.with_requires("numpy>=2.2")
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_trim_no_fb_in_rule(self, xp):
        a = xp.array([0, 0, 1, 0, 2, 3, 4, 0])
        assert_raises(ValueError, xp.trim_zeros, a, "ADE")

    def test_list_array(self):
        assert_raises(TypeError, dpnp.trim_zeros, [0, 0, 1, 0, 2, 3, 4, 0])

    @pytest.mark.parametrize(
        "trim", [1, ["F"], numpy.array("B")], ids=["int", "list", "array"]
    )
    def test_unsupported_trim(self, trim):
        a = numpy.array([0, 0, 1, 0, 2, 3, 4, 0])
        ia = dpnp.array(a)

        assert_raises(TypeError, dpnp.trim_zeros, ia, trim)
        assert_raises(AttributeError, numpy.trim_zeros, a, trim)


class TestUnique:
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_1d(self, dt):
        a = numpy.array([5, 7, 1, 2, 1, 5, 7] * 10, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.unique(ia)
        expected = numpy.unique(a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "return_index, return_inverse, return_counts",
        [
            pytest.param(True, False, False),
            pytest.param(False, True, False),
            pytest.param(False, False, True),
            pytest.param(True, True, False),
            pytest.param(True, False, True),
            pytest.param(False, True, True),
            pytest.param(True, True, True),
        ],
    )
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_1d_return_flags(
        self, return_index, return_inverse, return_counts, dt
    ):
        a = numpy.array([5, 7, 1, 2, 1, 5, 7] * 10, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.unique(ia, return_index, return_inverse, return_counts)
        expected = numpy.unique(a, return_index, return_inverse, return_counts)
        for iv, v in zip(result, expected):
            assert_array_equal(iv, v)

    def test_1d_complex(self):
        a = numpy.array([1.0 + 0.0j, 1 - 1.0j, 1])
        ia = dpnp.array(a)

        result = dpnp.unique(ia)
        expected = numpy.unique(a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "return_kwds",
        [
            {"return_index": True},
            {"return_inverse": True},
            {"return_index": True, "return_inverse": True},
            {
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
            },
        ],
    )
    def test_1d_empty(self, return_kwds):
        a = numpy.array([])
        ia = dpnp.array(a)

        result = dpnp.unique(ia, **return_kwds)
        expected = numpy.unique(a, **return_kwds)
        for idx, (iv, v) in enumerate(zip(result, expected)):
            assert_array_equal(iv, v)
            if idx > 0:  # skip values and check only indices
                assert iv.dtype == v.dtype

    @pytest.mark.parametrize(
        "return_kwds",
        [
            {"return_index": True},
            {"return_inverse": True},
            {"return_counts": True},
            {"return_index": True, "return_inverse": True},
            {
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
            },
        ],
    )
    def test_1d_nans(self, return_kwds):
        a = numpy.array([2.0, numpy.nan, 1.0, numpy.nan])
        ia = dpnp.array(a)

        result = dpnp.unique(ia, **return_kwds)
        expected = numpy.unique(a, **return_kwds)
        for iv, v in zip(result, expected):
            assert_array_equal(iv, v)

    @pytest.mark.parametrize(
        "return_kwds",
        [
            {"return_index": True},
            {"return_inverse": True},
            {"return_counts": True},
            {"return_index": True, "return_inverse": True},
            {
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
            },
        ],
    )
    def test_1d_complex_nans(self, return_kwds):
        a = numpy.array(
            [
                2.0 - 1j,
                numpy.nan,
                1.0 + 1j,
                complex(0.0, numpy.nan),
                complex(1.0, numpy.nan),
            ]
        )
        ia = dpnp.array(a)

        result = dpnp.unique(ia, **return_kwds)
        expected = numpy.unique(a, **return_kwds)
        for iv, v in zip(result, expected):
            assert_array_equal(iv, v)

    @pytest.mark.parametrize(
        "return_kwds",
        [
            {"return_index": True},
            {"return_inverse": True},
            {"return_counts": True},
            {"return_index": True, "return_inverse": True},
            {
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
            },
        ],
    )
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_1d_all_nans(self, return_kwds, dt):
        a = numpy.array([numpy.nan] * 4, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.unique(ia, **return_kwds)
        expected = numpy.unique(a, **return_kwds)
        for iv, v in zip(result, expected):
            assert_array_equal(iv, v)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("axis", [2, -2])
    def test_axis_errors(self, xp, axis):
        assert_raises(AxisError, xp.unique, xp.arange(10), axis=axis)
        assert_raises(AxisError, xp.unique, xp.arange(10), axis=axis)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_axis_list(self, axis):
        a = numpy.array([[0, 1, 0], [0, 1, 0]])
        ia = dpnp.array(a)

        result = dpnp.unique(ia, axis=axis)
        expected = numpy.unique(a, axis=axis)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "axis_kwd",
        [
            {},
            {"axis": 0},
            {"axis": 1},
        ],
    )
    @pytest.mark.parametrize(
        "return_kwds",
        [
            {},
            {
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
            },
        ],
    )
    def test_2d_axis(self, dt, axis_kwd, return_kwds):
        a = numpy.array(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        ).astype(dt)
        ia = dpnp.array(a)

        result = dpnp.unique(ia, **axis_kwd, **return_kwds)
        expected = numpy.unique(a, **axis_kwd, **return_kwds)
        if len(return_kwds) == 0:
            assert_array_equal(result, expected)
        else:
            if (
                len(axis_kwd) == 0
                and numpy.lib.NumpyVersion(numpy.__version__) < "2.0.1"
            ):
                # gh-26961: numpy.unique(..., return_inverse=True, axis=None)
                # returned flatten unique_inverse till 2.0.1 version
                expected = (
                    expected[:2]
                    + (expected[2].reshape(a.shape),)
                    + expected[3:]
                )
            for iv, v in zip(result, expected):
                assert_array_equal(iv, v)

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_3d_axis(self, dt):
        a = numpy.array([[[1, 1], [1, 0]], [[0, 1], [0, 0]]]).astype(dt)
        ia = dpnp.array(a)

        result = dpnp.unique(ia, axis=2)
        expected = numpy.unique(a, axis=2)
        assert_array_equal(result, expected)

    def test_2d_axis_negative_zero_equality(self):
        a = numpy.array([[-0.0, 0.0], [0.0, -0.0], [-0.0, 0.0], [0.0, -0.0]])
        ia = dpnp.array(a)

        result = dpnp.unique(ia, axis=0)
        expected = numpy.unique(a, axis=0)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("axis", [0, -1])
    def test_1d_axis(self, axis):
        a = numpy.array([4, 3, 2, 3, 2, 1, 2, 2])
        ia = dpnp.array(a)

        result = dpnp.unique(ia, axis=axis)
        expected = numpy.unique(a, axis=axis)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, -1])
    def test_2d_axis_inverse(self, axis):
        a = numpy.array([[4, 4, 3], [2, 2, 1], [2, 2, 1], [4, 4, 3]])
        ia = dpnp.array(a)

        result = dpnp.unique(ia, return_inverse=True, axis=axis)
        expected = numpy.unique(a, return_inverse=True, axis=axis)
        if axis is None and numpy.lib.NumpyVersion(numpy.__version__) < "2.0.1":
            # gh-26961: numpy.unique(..., return_inverse=True, axis=None)
            # returned flatten unique_inverse till 2.0.1 version
            expected = expected[:1] + (expected[1].reshape(a.shape),)

        for iv, v in zip(result, expected):
            assert_array_equal(iv, v)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_2d_axis_zeros(self, axis):
        a = numpy.empty(shape=(2, 0), dtype=numpy.int8)
        ia = dpnp.array(a)

        result = dpnp.unique(
            ia,
            axis=axis,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        expected = numpy.unique(
            a,
            axis=axis,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )
        for iv, v in zip(result, expected):
            assert_array_equal(iv, v)

    @pytest.mark.parametrize("axis", range(7))  # len(shape) = 7
    def test_7d_axis_zeros(self, axis):
        shape = (0, 2, 0, 3, 0, 4, 0)
        a = numpy.empty(shape=shape, dtype=numpy.int8)
        ia = dpnp.array(a)

        result = dpnp.unique(ia, axis=axis)
        expected = numpy.unique(a, axis=axis)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", get_integer_dtypes(no_unsigned=True))
    def test_2d_axis_signed_inetger(self, dt):
        a = numpy.array([[-1], [0]], dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.unique(ia, axis=0)
        expected = numpy.unique(a, axis=0)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("dt", "bBhHiIlLqQ")
    def test_1d_axis_all_inetger(self, axis, dt):
        a = numpy.array([5, 7, 1, 2, 1, 5, 7], dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.unique(ia, True, True, True, axis=axis)
        expected = numpy.unique(a, True, True, True, axis=axis)
        for iv, v in zip(result, expected):
            assert_array_equal(iv, v)

    @pytest.mark.parametrize(
        "eq_nan_kwd",
        [
            {},
            {"equal_nan": False},
        ],
    )
    def test_equal_nan(self, eq_nan_kwd):
        a = numpy.array([1, 1, numpy.nan, numpy.nan, numpy.nan])
        ia = dpnp.array(a)

        result = dpnp.unique(ia, **eq_nan_kwd)
        expected = numpy.unique(a, **eq_nan_kwd)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_complex_dtypes())
    @pytest.mark.parametrize(
        "axis_kwd",
        [
            {},
            {"axis": 0},
            {"axis": 1},
        ],
    )
    @pytest.mark.parametrize(
        "return_kwds",
        [
            {},
            {
                "return_index": True,
                "return_inverse": True,
                "return_counts": True,
            },
        ],
    )
    @pytest.mark.parametrize(
        "row", [[2, 3, 4], [2, numpy.nan, 4], [numpy.nan, 3, 4]]
    )
    def test_2d_axis_nans(self, dt, axis_kwd, return_kwds, row):
        a = numpy.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [numpy.nan, numpy.nan, numpy.nan],
                row,
                [1, 0, 1],
                [numpy.nan, numpy.nan, numpy.nan],
            ]
        ).astype(dt)
        ia = dpnp.array(a)

        result = dpnp.unique(ia, **axis_kwd, **return_kwds)
        expected = numpy.unique(a, **axis_kwd, **return_kwds)
        if len(return_kwds) == 0:
            assert_array_equal(result, expected)
        else:
            if len(axis_kwd) == 0 and numpy_version() < "2.0.1":
                # gh-26961: numpy.unique(..., return_inverse=True, axis=None)
                # returned flatten unique_inverse till 2.0.1 version
                expected = (
                    expected[:2]
                    + (expected[2].reshape(a.shape),)
                    + expected[3:]
                )
            for iv, v in zip(result, expected):
                assert_array_equal(iv, v)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize(
        "func",
        ["unique_all", "unique_counts", "unique_inverse", "unique_values"],
    )
    def test_array_api_functions(self, func):
        a = numpy.array([numpy.nan, 1, 4, 1, 3, 4, 5, 5, 1])
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia)
        expected = getattr(numpy, func)(a)
        for iv, v in zip(result, expected):
            assert_array_equal(iv, v)


class TestVsplit:
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        # 0D array
        a = xp.array(1)
        assert_raises(ValueError, xp.vsplit, a, 2)

        # 1D array
        a = xp.array([1, 2, 3, 4])
        assert_raises(ValueError, xp.vsplit, a, 2)

    def test_2D_array(self):
        a = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        a_dp = dpnp.array(a)

        expected = numpy.vsplit(a, 2)
        result = dpnp.vsplit(a_dp, 2)
        _compare_results(result, expected)
