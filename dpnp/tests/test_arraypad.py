import numpy
import pytest

if numpy.lib.NumpyVersion(numpy.__version__) >= "2.0.0":
    from numpy.lib._arraypad_impl import _as_pairs as numpy_as_pairs
else:
    from numpy.lib.arraypad import _as_pairs as numpy_as_pairs

from numpy.testing import assert_array_equal, assert_equal

import dpnp
from dpnp.dpnp_utils.dpnp_utils_pad import _as_pairs as dpnp_as_pairs

from .helper import assert_dtype_allclose, get_all_dtypes
from .third_party.cupy import testing


class TestPad:
    _all_modes = {
        "constant": {"constant_values": 0},
        "edge": {},
        "linear_ramp": {"end_values": 0},
        "maximum": {"stat_length": None},
        "mean": {"stat_length": None},
        "median": {"stat_length": None},
        "minimum": {"stat_length": None},
        "reflect": {"reflect_type": "even"},
        "symmetric": {"reflect_type": "even"},
        "wrap": {},
        "empty": {},
    }

    # .keys() returns set which is not ordered
    # consistent order is required by xdist plugin
    _modes = sorted(_all_modes.keys())

    @pytest.mark.parametrize("mode", _modes)
    def test_basic(self, mode):
        a_np = numpy.arange(100)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, (25, 20), mode=mode)
        result = dpnp.pad(a_dp, (25, 20), mode=mode)
        if mode == "empty":
            # omit uninitialized "empty" boundary from the comparison
            assert_equal(result[25:-20], expected[25:-20])
        else:
            assert_array_equal(result, expected)

    @pytest.mark.parametrize("mode", _modes)
    def test_memory_layout_persistence(self, mode):
        """Test if C and F order is preserved for all pad modes."""
        x = dpnp.ones((5, 10), order="C")
        assert dpnp.pad(x, 5, mode).flags.c_contiguous
        x = dpnp.ones((5, 10), order="F")
        assert dpnp.pad(x, 5, mode).flags.f_contiguous

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("mode", _modes)
    def test_dtype_persistence(self, dtype, mode):
        arr = dpnp.zeros((3, 2, 1), dtype=dtype)
        result = dpnp.pad(arr, 1, mode=mode)
        assert result.dtype == dtype

    @pytest.mark.parametrize("mode", _modes)
    def test_non_contiguous_array(self, mode):
        a_np = numpy.arange(24).reshape(4, 6)[::2, ::2]
        a_dp = dpnp.arange(24).reshape(4, 6)[::2, ::2]
        expected = numpy.pad(a_np, (2, 3), mode=mode)
        result = dpnp.pad(a_dp, (2, 3), mode=mode)
        if mode == "empty":
            # omit uninitialized "empty" boundary from the comparison
            assert_equal(result[2:-3, 2:-3], expected[2:-3, 2:-3])
        else:
            assert_array_equal(result, expected)

    # TODO: include "linear_ramp" when dpnp issue gh-2084 is resolved
    @pytest.mark.parametrize("pad_width", [0, (0, 0), ((0, 0), (0, 0))])
    @pytest.mark.parametrize(
        "mode", [m for m in _modes if m not in {"linear_ramp"}]
    )
    def test_zero_pad_width(self, pad_width, mode):
        arr = dpnp.arange(30).reshape(6, 5)
        assert_array_equal(arr, dpnp.pad(arr, pad_width, mode=mode))

    @pytest.mark.parametrize("mode", _modes)
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
    @pytest.mark.parametrize("mode", _modes)
    def test_misshaped_pad_width1(self, pad_width, mode):
        arr = dpnp.arange(30).reshape((6, 5))
        match = "operands could not be broadcast together"
        with pytest.raises(ValueError, match=match):
            dpnp.pad(arr, pad_width, mode)

    @pytest.mark.parametrize("mode", _modes)
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
    @pytest.mark.parametrize("mode", _modes)
    def test_negative_pad_width(self, pad_width, mode):
        arr = dpnp.arange(30).reshape((6, 5))
        match = "index can't contain negative values"
        with pytest.raises(ValueError, match=match):
            dpnp.pad(arr, pad_width, mode)

    @pytest.mark.parametrize(
        "pad_width",
        ["3", "word", None, 3.4, complex(1, -1), ((-2.1, 3), (3, 2))],
    )
    @pytest.mark.parametrize("mode", _modes)
    def test_bad_type(self, pad_width, mode):
        arr = dpnp.arange(30).reshape((6, 5))
        match = "`pad_width` must be of integral type."
        with pytest.raises(TypeError, match=match):
            dpnp.pad(arr, pad_width, mode)

    @pytest.mark.parametrize("mode", _modes)
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
        assert_equal(a[:, 0], 0.0, strict=False)
        assert_equal(a[:, -1], 0.0, strict=False)
        assert_equal(a[0, :], 0.0, strict=False)
        assert_equal(a[-1, :], 0.0, strict=False)

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
    @pytest.mark.parametrize("mode", ["maximum", "minimum", "mean", "median"])
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
    @pytest.mark.parametrize("mode", ["maximum", "minimum", "mean", "median"])
    @pytest.mark.parametrize("stat_length", [(3,), (2, 3)])
    def test_stat_func_2d(self, pad_width, mode, stat_length):
        a_np = numpy.arange(30).reshape(6, 5)
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(
            a_np, pad_width, mode=mode, stat_length=stat_length
        )
        result = dpnp.pad(a_dp, pad_width, mode=mode, stat_length=stat_length)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("mode", ["mean", "minimum", "maximum", "median"])
    def test_same_prepend_append(self, mode):
        """Test that appended and prepended values are equal"""
        a = dpnp.array([-1, 2, -1]) + dpnp.array(
            [0, 1e-12, 0], dtype=dpnp.float32
        )
        result = dpnp.pad(a, (1, 1), mode)
        assert_equal(result[0], result[-1])

    @pytest.mark.usefixtures(
        "suppress_invalid_numpy_warnings",
        "suppress_mean_empty_slice_numpy_warnings",
    )
    @pytest.mark.parametrize("mode", ["mean", "median"])
    def test_zero_stat_length_valid(self, mode):
        a_np = numpy.array([1.0, 2.0])
        a_dp = dpnp.array(a_np)
        expected = numpy.pad(a_np, (1, 2), mode, stat_length=0)
        result = dpnp.pad(a_dp, (1, 2), mode, stat_length=0)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("mode", ["mean", "minimum", "maximum", "median"])
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

    @testing.with_requires("numpy>=2.0")  # numpy<2 has a bug, numpy-gh-25963
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
        assert result.shape == expected.shape
        if mode == "constant":
            # In "empty" mode, arrays are uninitialized and comparing may fail
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "mode",
        [m for m in _modes if m not in {"constant", "empty"}],
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
