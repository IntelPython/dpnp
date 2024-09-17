import itertools

import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from numpy.lib._arraypad_impl import _as_pairs as numpy_as_pairs
from numpy.testing import assert_array_equal, assert_equal, assert_raises

import dpnp
from dpnp.dpnp_utils.dpnp_utils_pad import _as_pairs as dpnp_as_pairs
from tests.third_party.cupy import testing

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


def _compare_results(result, expected):
    """Compare lists of arrays."""
    if len(result) != len(expected):
        raise ValueError("Iterables have different lengths")

    for x, y in zip(result, expected):
        assert_array_equal(x, y)


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


class TestPad:
    _all_modes = {
        "constant": {"constant_values": 0},
        "edge": {},
        "linear_ramp": {"end_values": 0},
        "maximum": {"stat_length": None},
        "mean": {"stat_length": None},
        "minimum": {"stat_length": None},
        "reflect": {"reflect_type": "even"},
        "symmetric": {"reflect_type": "even"},
        "wrap": {},
        "empty": {},
    }

    @pytest.mark.parametrize("mode", _all_modes.keys() - {"empty"})
    def test_basic(self, mode):
        a_np = numpy.arange(100)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, (25, 20), mode=mode)
        result = dpnp.pad(a_dp, (25, 20), mode=mode)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_memory_layout_persistence(self, mode):
        """Test if C and F order is preserved for all pad modes."""
        x = dpnp.ones((5, 10), order="C")
        assert dpnp.pad(x, 5, mode).flags.c_contiguous
        x = dpnp.ones((5, 10), order="F")
        assert dpnp.pad(x, 5, mode).flags.f_contiguous

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_dtype_persistence(self, dtype, mode):
        arr = dpnp.zeros((3, 2, 1), dtype=dtype)
        result = dpnp.pad(arr, 1, mode=mode)
        assert result.dtype == dtype

    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_non_contiguous_array(self, mode):
        a_np = numpy.arange(24).reshape(4, 6)[::2, ::2]
        a_dp = dpnp.arange(24).reshape(4, 6)[::2, ::2]
        expected = numpy.pad(a_np, (2, 3), mode=mode)
        result = dpnp.pad(a_dp, (2, 3), mode=mode)
        assert_array_equal(result, expected)

    # TODO: include "linear_ramp" when dpnp issue gh-2084 is resolved
    @pytest.mark.parametrize("pad_width", [0, (0, 0), ((0, 0), (0, 0))])
    @pytest.mark.parametrize("mode", _all_modes.keys() - {"linear_ramp"})
    def test_zero_pad_width(self, pad_width, mode):
        arr = dpnp.arange(30).reshape(6, 5)
        assert_array_equal(arr, dpnp.pad(arr, pad_width, mode=mode))

    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_pad_non_empty_dimension(self, mode):
        a_np = numpy.ones((2, 0, 2))
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, ((3,), (0,), (1,)), mode=mode)
        result = dpnp.pad(a_dp, ((3,), (0,), (1,)), mode=mode)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "pad_width",
        [
            (4, 5, 6, 7),
            ((1,), (2,), (3,)),
            ((1, 2), (3, 4), (5, 6)),
            ((3, 4, 5), (0, 1, 2)),
        ],
    )
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_misshaped_pad_width1(self, pad_width, mode):
        arr = dpnp.arange(30).reshape((6, 5))
        match = "operands could not be broadcast together"
        with pytest.raises(ValueError, match=match):
            dpnp.pad(arr, pad_width, mode)

    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_misshaped_pad_width2(self, mode):
        arr = dpnp.arange(30).reshape((6, 5))
        match = (
            "input operand has more dimensions than allowed by the axis "
            "remapping"
        )
        with pytest.raises(ValueError, match=match):
            dpnp.pad(arr, (((3,), (4,), (5,)), ((0,), (1,), (2,))), mode)

    @pytest.mark.parametrize(
        "pad_width", [-2, (-2,), (3, -1), ((5, 2), (-2, 3)), ((-4,), (2,))]
    )
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_negative_pad_width(self, pad_width, mode):
        arr = dpnp.arange(30).reshape((6, 5))
        match = "index can't contain negative values"
        with pytest.raises(ValueError, match=match):
            dpnp.pad(arr, pad_width, mode)

    @pytest.mark.parametrize(
        "pad_width",
        ["3", "word", None, 3.4, complex(1, -1), ((-2.1, 3), (3, 2))],
    )
    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_bad_type(self, pad_width, mode):
        arr = dpnp.arange(30).reshape((6, 5))
        match = "`pad_width` must be of integral type."
        with pytest.raises(TypeError, match=match):
            dpnp.pad(arr, pad_width, mode)

    @pytest.mark.parametrize("mode", _all_modes.keys())
    def test_kwargs(self, mode):
        """Test behavior of pad's kwargs for the given mode."""
        allowed = self._all_modes[mode]
        not_allowed = {}
        for kwargs in self._all_modes.values():
            if kwargs != allowed:
                not_allowed.update(kwargs)
        # Test if allowed keyword arguments pass
        dpnp.pad(dpnp.array([1, 2, 3]), 1, mode, **allowed)
        # Test if prohibited keyword arguments of other modes raise an error
        for key, value in not_allowed.items():
            match = f"unsupported keyword arguments for mode '{mode}'"
            with pytest.raises(ValueError, match=match):
                dpnp.pad(dpnp.array([1, 2, 3]), 1, mode, **{key: value})

    @pytest.mark.parametrize("mode", [1, "const", object(), None, True, False])
    def test_unsupported_mode(self, mode):
        match = f"mode '{mode}' is not supported"
        with pytest.raises(ValueError, match=match):
            dpnp.pad(dpnp.array([1, 2, 3]), 4, mode=mode)

    def test_pad_default(self):
        a_np = numpy.array([1, 1])
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, 2)
        result = dpnp.pad(a_dp, 2)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "pad_width",
        [numpy.array(((2, 3), (3, 2))), dpnp.array(((2, 3), (3, 2)))],
    )
    def test_pad_width_as_ndarray(self, pad_width):
        a_np = numpy.arange(12).reshape(4, 3)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, ((2, 3), (3, 2)))
        result = dpnp.pad(a_dp, pad_width)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("pad_width", [5, (25, 20)])
    @pytest.mark.parametrize("constant_values", [3, (10, 20)])
    def test_constant_1d(self, pad_width, constant_values):
        a_np = numpy.arange(100)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(
            a_np, pad_width, "constant", constant_values=constant_values
        )
        result = dpnp.pad(
            a_dp, pad_width, "constant", constant_values=constant_values
        )
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("pad_width", [((1,), (2,)), ((1, 2), (1, 3))])
    @pytest.mark.parametrize("constant_values", [3, ((1, 2), (3, 4))])
    def test_constant_2d(self, pad_width, constant_values):
        a_np = numpy.arange(30).reshape(5, 6)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(
            a_np, pad_width, "constant", constant_values=constant_values
        )
        result = dpnp.pad(
            a_dp, pad_width, "constant", constant_values=constant_values
        )
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [numpy.int32, numpy.float32])
    def test_constant_float(self, dtype):
        # If input array is int, but constant_values are float, the dtype of
        # the array to be padded is kept
        # If input array is float, and constant_values are float, the dtype of
        # the array to be padded is kept - here retaining the float constants
        a_np = numpy.arange(30, dtype=dtype).reshape(5, 6)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, (25, 20), "constant", constant_values=1.1)
        result = dpnp.pad(a_dp, (25, 20), "constant", constant_values=1.1)
        assert_array_equal(result, expected)

    def test_constant_large_integers(self):
        uint64_max = 2**64 - 1
        a_np = numpy.full(5, uint64_max, dtype=numpy.uint64)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, 1, "constant", constant_values=a_np.min())
        result = dpnp.pad(a_dp, 1, "constant", constant_values=a_dp.min())
        assert_array_equal(result, expected)

        int64_max = 2**63 - 1
        a_np = numpy.full(5, int64_max, dtype=numpy.int64)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, 1, "constant", constant_values=a_np.min())
        result = dpnp.pad(a_dp, 1, "constant", constant_values=a_dp.min())
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("pad_width", [((1,), (2,)), ((2, 3), (3, 2))])
    def test_edge(self, pad_width):
        a_np = numpy.arange(12).reshape(4, 3)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, pad_width, "edge")
        result = dpnp.pad(a_dp, pad_width, "edge")
        assert_array_equal(result, expected)

    def test_edge_nd(self):
        a_np = numpy.array([1, 2, 3])
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, ((1, 2),), "edge")
        result = dpnp.pad(a_dp, ((1, 2),), "edge")
        assert_array_equal(result, expected)

        a_np = numpy.array([[1, 2, 3], [4, 5, 6]])
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, ((1, 2),), "edge")
        result = dpnp.pad(a_dp, ((1, 2), (1, 2)), "edge")
        assert_array_equal(result, expected)

        a_np = numpy.arange(24).reshape(2, 3, 4)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, ((1, 2),), "edge")
        result = dpnp.pad(a_dp, ((1, 2), (1, 2), (1, 2)), "edge")
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("pad_width", [5, (25, 20)])
    @pytest.mark.parametrize("end_values", [3, (10, 20)])
    def test_linear_ramp_1d(self, pad_width, end_values):
        a_np = numpy.arange(100, dtype=numpy.float32)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(
            a_np, pad_width, "linear_ramp", end_values=end_values
        )
        result = dpnp.pad(a_dp, pad_width, "linear_ramp", end_values=end_values)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "pad_width", [(2, 2), ((1,), (2,)), ((1, 2), (1, 3))]
    )
    @pytest.mark.parametrize("end_values", [3, (0, 0), ((1, 2), (3, 4))])
    def test_linear_ramp_2d(self, pad_width, end_values):
        a_np = numpy.arange(20, dtype=numpy.float32).reshape(4, 5)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(
            a_np, pad_width, "linear_ramp", end_values=end_values
        )
        result = dpnp.pad(a_dp, pad_width, "linear_ramp", end_values=end_values)
        assert_dtype_allclose(result, expected)

    def test_linear_ramp_end_values(self):
        """Ensure that end values are exact."""
        a_dp = dpnp.ones(10).reshape(2, 5)
        a = dpnp.pad(a_dp, (223, 123), mode="linear_ramp")
        assert_equal(a[:, 0], 0.0)
        assert_equal(a[:, -1], 0.0)
        assert_equal(a[0, :], 0.0)
        assert_equal(a[-1, :], 0.0)

    @pytest.mark.parametrize(
        "dtype", [numpy.uint32, numpy.uint64] + get_all_dtypes(no_none=True)
    )
    @pytest.mark.parametrize("data, end_values", [([3], 0), ([0], 3)])
    def test_linear_ramp_negative_diff(self, dtype, data, end_values):
        """
        Check correct behavior of unsigned dtypes if there is a negative
        difference between the edge to pad and `end_values`. Check both cases
        to be independent of implementation. Test behavior for all other dtypes
        in case dtype casting interferes with complex dtypes.
        """
        a_np = numpy.array(data, dtype=dtype)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, 3, mode="linear_ramp", end_values=end_values)
        result = dpnp.pad(a_dp, 3, mode="linear_ramp", end_values=end_values)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("pad_width", [5, (25, 20)])
    @pytest.mark.parametrize("mode", ["maximum", "minimum", "mean"])
    @pytest.mark.parametrize("stat_length", [10, (2, 3)])
    def test_stat_func_1d(self, pad_width, mode, stat_length):
        a_np = numpy.arange(100)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(
            a_np, pad_width, mode=mode, stat_length=stat_length
        )
        result = dpnp.pad(a_dp, pad_width, mode=mode, stat_length=stat_length)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("pad_width", [((1,), (2,)), ((2, 3), (3, 2))])
    @pytest.mark.parametrize("mode", ["maximum", "minimum", "mean"])
    @pytest.mark.parametrize("stat_length", [(3,), (2, 3)])
    def test_stat_func_2d(self, pad_width, mode, stat_length):
        a_np = numpy.arange(30).reshape(6, 5)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(
            a_np, pad_width, mode=mode, stat_length=stat_length
        )
        result = dpnp.pad(a_dp, pad_width, mode=mode, stat_length=stat_length)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("mode", ["mean", "minimum", "maximum"])
    def test_same_prepend_append(self, mode):
        """Test that appended and prepended values are equal"""
        a = dpnp.array([-1, 2, -1]) + dpnp.array(
            [0, 1e-12, 0], dtype=dpnp.float32
        )
        result = dpnp.pad(a, (1, 1), mode)
        assert_equal(result[0], result[-1])

    def test_mean_with_zero_stat_length(self):
        a_np = numpy.array([1.0, 2.0])
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, (1, 2), "mean")
        result = dpnp.pad(a_dp, (1, 2), "mean")
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("mode", ["mean", "minimum", "maximum"])
    @pytest.mark.parametrize(
        "stat_length", [-2, (-2,), (3, -1), ((5, 2), (-2, 3)), ((-4,), (2,))]
    )
    def test_negative_stat_length(self, mode, stat_length):
        a_dp = dpnp.arange(30).reshape((6, 5))
        match = "index can't contain negative values"
        with pytest.raises(ValueError, match=match):
            dpnp.pad(a_dp, 2, mode, stat_length=stat_length)

    @pytest.mark.parametrize("mode", ["minimum", "maximum"])
    def test_zero_stat_length_invalid(self, mode):
        a = dpnp.array([1.0, 2.0])
        match = "stat_length of 0 yields no value for padding"
        with pytest.raises(ValueError, match=match):
            dpnp.pad(a, 0, mode, stat_length=0)
        with pytest.raises(ValueError, match=match):
            dpnp.pad(a, 0, mode, stat_length=(1, 0))
        with pytest.raises(ValueError, match=match):
            dpnp.pad(a, 1, mode, stat_length=0)
        with pytest.raises(ValueError, match=match):
            dpnp.pad(a, 1, mode, stat_length=(1, 0))

    @pytest.mark.parametrize("pad_width", [2, 3, 4, [1, 10], [15, 2], [45, 10]])
    @pytest.mark.parametrize("mode", ["reflect", "symmetric"])
    @pytest.mark.parametrize("reflect_type", ["even", "odd"])
    def test_reflect_symmetric_1d(self, pad_width, mode, reflect_type):
        a_np = numpy.array([1, 2, 3, 4])
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(
            a_np, pad_width, mode=mode, reflect_type=reflect_type
        )
        result = dpnp.pad(a_dp, pad_width, mode=mode, reflect_type=reflect_type)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("data", [[[4, 5, 6], [6, 7, 8]], [[4, 5, 6]]])
    @pytest.mark.parametrize("pad_width", [10, (5, 7)])
    @pytest.mark.parametrize("mode", ["reflect", "symmetric"])
    @pytest.mark.parametrize("reflect_type", ["even", "odd"])
    def test_reflect_symmetric_2d(self, data, pad_width, mode, reflect_type):
        a_np = numpy.array(data)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(
            a_np, pad_width, mode=mode, reflect_type=reflect_type
        )
        result = dpnp.pad(a_dp, pad_width, mode=mode, reflect_type=reflect_type)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("pad_width", [2, 3, 4, [1, 10], [15, 2], [45, 10]])
    def test_wrap_1d(self, pad_width):
        a_np = numpy.array([1, 2, 3, 4])
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, pad_width, "wrap")
        result = dpnp.pad(a_dp, pad_width, "wrap")
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("data", [[[4, 5, 6], [6, 7, 8]], [[4, 5, 6]]])
    @pytest.mark.parametrize("pad_width", [10, (5, 7), (1, 3), (3, 1)])
    def test_wrap_2d(self, data, pad_width):
        a_np = numpy.array(data)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, pad_width, "wrap")
        result = dpnp.pad(a_dp, pad_width, "wrap")
        assert_array_equal(result, expected)

    def test_empty(self):
        a_np = numpy.arange(24).reshape(4, 6)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, [(2, 3), (3, 1)], "empty")
        result = dpnp.pad(a_dp, [(2, 3), (3, 1)], "empty")
        # omit uninitialized "empty" boundary from the comparison
        assert result.shape == expected.shape
        assert_equal(result[2:-3, 3:-1], expected[2:-3, 3:-1])

    # Check how padding behaves on arrays with an empty dimension.
    # empty axis can only be padded using modes 'constant' or 'empty'
    @pytest.mark.parametrize("mode", ["constant", "empty"])
    def test_pad_empty_dim_valid(self, mode):
        """empty axis can only be padded using modes 'constant' or 'empty'"""
        a_np = numpy.zeros((3, 0, 2))
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, [(0,), (2,), (1,)], mode)
        result = dpnp.pad(a_dp, [(0,), (2,), (1,)], mode)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "mode",
        _all_modes.keys() - {"constant", "empty"},
    )
    def test_pad_empty_dim_invalid(self, mode):
        match = (
            "can't extend empty axis 0 using modes other than 'constant' "
            "or 'empty'"
        )
        with pytest.raises(ValueError, match=match):
            dpnp.pad(dpnp.array([]), 4, mode=mode)
        with pytest.raises(ValueError, match=match):
            dpnp.pad(dpnp.ndarray(0), 4, mode=mode)
        with pytest.raises(ValueError, match=match):
            dpnp.pad(dpnp.zeros((0, 3)), ((1,), (0,)), mode=mode)

    def test_vector_functionality(self):
        def _padwithtens(vector, pad_width, iaxis, kwargs):
            vector[: pad_width[0]] = 10
            vector[-pad_width[1] :] = 10

        a_np = numpy.arange(6).reshape(2, 3)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, 2, _padwithtens)
        result = dpnp.pad(a_dp, 2, _padwithtens)
        assert_array_equal(result, expected)

    # test _as_pairs an internal function used by dpnp.pad
    def test_as_pairs_single_value(self):
        """Test casting for a single value."""
        for x in (3, [3], [[3]]):
            result = dpnp_as_pairs(x, 10)
            expected = numpy_as_pairs(x, 10)
            assert_equal(result, expected)

    def test_as_pairs_two_values(self):
        """Test proper casting for two different values."""
        # Broadcasting in the first dimension with numbers
        for x in ([3, 4], [[3, 4]]):
            result = dpnp_as_pairs(x, 10)
            expected = numpy_as_pairs(x, 10)
            assert_equal(result, expected)

        # Broadcasting in the second / last dimension with numbers
        assert_equal(
            dpnp_as_pairs([[3], [4]], 2),
            numpy_as_pairs([[3], [4]], 2),
        )

    def test_as_pairs_with_none(self):
        assert_equal(
            dpnp_as_pairs(None, 3, as_index=False),
            numpy_as_pairs(None, 3, as_index=False),
        )
        assert_equal(
            dpnp_as_pairs(None, 3, as_index=True),
            numpy_as_pairs(None, 3, as_index=True),
        )

    def test_as_pairs_pass_through(self):
        """Test if `x` already matching desired output are passed through."""
        a_np = numpy.arange(12).reshape((6, 2))
        a_dp = dpnp.arange(12).reshape((6, 2))
        assert_equal(
            dpnp_as_pairs(a_dp, 6),
            numpy_as_pairs(a_np, 6),
        )

    def test_as_pairs_as_index(self):
        """Test results if `as_index=True`."""
        assert_equal(
            dpnp_as_pairs([2.6, 3.3], 10, as_index=True),
            numpy_as_pairs([2.6, 3.3], 10, as_index=True),
        )
        assert_equal(
            dpnp_as_pairs([2.6, 4.49], 10, as_index=True),
            numpy_as_pairs([2.6, 4.49], 10, as_index=True),
        )
        for x in (
            -3,
            [-3],
            [[-3]],
            [-3, 4],
            [3, -4],
            [[-3, 4]],
            [[4, -3]],
            [[1, 2]] * 9 + [[1, -2]],
        ):
            with pytest.raises(ValueError, match="negative values"):
                dpnp_as_pairs(x, 10, as_index=True)

    def test_as_pairs_exceptions(self):
        """Ensure faulty usage is discovered."""
        with pytest.raises(ValueError, match="more dimensions than allowed"):
            dpnp_as_pairs([[[3]]], 10)
        with pytest.raises(ValueError, match="could not be broadcast"):
            dpnp_as_pairs([[1, 2], [3, 4]], 3)
        with pytest.raises(ValueError, match="could not be broadcast"):
            dpnp_as_pairs(dpnp.ones((2, 3)), 3)


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
        assert expected.dtype == result.dtype
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
        assert_equal(dpnp.rot90(ia, axes=axes), numpy.rot90(a, axes=axes))

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
        a = dpnp.ones((5, 3))

        res1 = dpnp.transpose((a))
        res2 = dpnp.permute_dims((a))

        assert_array_equal(res1, res2)


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
