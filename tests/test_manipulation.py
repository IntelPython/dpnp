import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from numpy.testing import assert_array_equal, assert_raises

import dpnp

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    get_integer_dtypes,
    has_support_aspect64,
)

testdata = []
testdata += [
    ([True, False, True], dtype)
    for dtype in get_all_dtypes(no_none=True, no_complex=True)
]
testdata += [
    ([1, -1, 0], dtype)
    for dtype in get_all_dtypes(no_none=True, no_bool=True, no_complex=True)
]
testdata += [([0.1, 0.0, -0.1], dtype) for dtype in get_float_dtypes()]
testdata += [([1j, -1j, 1 - 2j], dtype) for dtype in get_complex_dtypes()]


@pytest.mark.parametrize("in_obj, out_dtype", testdata)
def test_copyto_dtype(in_obj, out_dtype):
    ndarr = numpy.array(in_obj)
    expected = numpy.empty(ndarr.size, dtype=out_dtype)
    numpy.copyto(expected, ndarr)

    dparr = dpnp.array(in_obj)
    result = dpnp.empty(dparr.size, dtype=out_dtype)
    dpnp.copyto(result, dparr)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("dst", [7, numpy.ones(10), (2, 7), [5], range(3)])
def test_copyto_dst_raises(dst):
    a = dpnp.array(4)
    with pytest.raises(
        TypeError,
        match="Destination array must be any of supported type, but got",
    ):
        dpnp.copyto(dst, a)


@pytest.mark.parametrize("where", [numpy.ones(10), (2, 7), [5], range(3)])
def test_copyto_where_raises(where):
    a = dpnp.empty((2, 3))
    b = dpnp.arange(6).reshape((2, 3))

    with pytest.raises(
        TypeError, match="`where` array must be any of supported type, but got"
    ):
        dpnp.copyto(a, b, where=where)


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


class TestTrimZeros:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, dtype):
        a = numpy.array([0, 0, 1, 0, 2, 3, 4, 0], dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.trim_zeros(ia)
        expected = numpy.trim_zeros(a)
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

    def test_trim_no_rule(self):
        a = numpy.array([0, 0, 1, 0, 2, 3, 4, 0])
        ia = dpnp.array(a)
        trim = "ADE"  # no "F" or "B" in trim string

        result = dpnp.trim_zeros(ia, trim)
        expected = numpy.trim_zeros(a, trim)
        assert_array_equal(result, expected)

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

    @pytest.mark.parametrize("dt", get_integer_dtypes())
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
