import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

import dpnp

from .helper import (
    get_all_dtypes,
    is_cpu_device,
    is_win_platform,
)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestConvolve:
    def test_object(self):
        d = [1.0] * 100
        k = [1.0] * 3
        assert_array_almost_equal(dpnp.convolve(d, k)[2:-2], dpnp.full(98, 3))

    def test_no_overwrite(self):
        d = dpnp.ones(100)
        k = dpnp.ones(3)
        dpnp.convolve(d, k)
        assert_array_equal(d, dpnp.ones(100))
        assert_array_equal(k, dpnp.ones(3))

    def test_mode(self):
        d = dpnp.ones(100)
        k = dpnp.ones(3)
        default_mode = dpnp.convolve(d, k, mode="full")
        full_mode = dpnp.convolve(d, k, mode="f")
        assert_array_equal(full_mode, default_mode)
        # integer mode
        with assert_raises(ValueError):
            dpnp.convolve(d, k, mode=-1)
        assert_array_equal(dpnp.convolve(d, k, mode=2), full_mode)
        # illegal arguments
        with assert_raises(TypeError):
            dpnp.convolve(d, k, mode=None)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
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
def test_diff(array):
    np_a = numpy.array(array)
    dpnp_a = dpnp.array(array)
    expected = numpy.diff(np_a)
    result = dpnp.diff(dpnp_a)
    assert_allclose(expected, result)


@pytest.mark.parametrize("dtype1", get_all_dtypes())
@pytest.mark.parametrize("dtype2", get_all_dtypes())
@pytest.mark.parametrize(
    "func", ["add", "divide", "multiply", "power", "subtract"]
)
@pytest.mark.parametrize("data", [[[1, 2], [3, 4]]], ids=["[[1, 2], [3, 4]]"])
def test_op_multiple_dtypes(dtype1, func, dtype2, data):
    np_a = numpy.array(data, dtype=dtype1)
    dpnp_a = dpnp.array(data, dtype=dtype1)

    np_b = numpy.array(data, dtype=dtype2)
    dpnp_b = dpnp.array(data, dtype=dtype2)

    if func == "subtract" and (dtype1 == dtype2 == dpnp.bool):
        with pytest.raises(TypeError):
            result = getattr(dpnp, func)(dpnp_a, dpnp_b)
            expected = getattr(numpy, func)(np_a, np_b)
    else:
        result = getattr(dpnp, func)(dpnp_a, dpnp_b)
        expected = getattr(numpy, func)(np_a, np_b)
        assert_allclose(result, expected)


@pytest.mark.parametrize(
    "rhs", [[[1, 2, 3], [4, 5, 6]], [2.0, 1.5, 1.0], 3, 0.3]
)
@pytest.mark.parametrize("lhs", [[[6, 5, 4], [3, 2, 1]], [1.3, 2.6, 3.9]])
# TODO: achieve the same level of dtype support for all mathematical operations, like
# @pytest.mark.parametrize("dtype", get_all_dtypes())
# and to get rid of fallbacks on numpy allowed by below fixture
# @pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestMathematical:
    @staticmethod
    def array_or_scalar(xp, data, dtype=None):
        if numpy.isscalar(data):
            return data

        return xp.array(data, dtype=dtype)

    def _test_mathematical(self, name, dtype, lhs, rhs):
        a_dpnp = self.array_or_scalar(dpnp, lhs, dtype=dtype)
        b_dpnp = self.array_or_scalar(dpnp, rhs, dtype=dtype)

        a_np = self.array_or_scalar(numpy, lhs, dtype=dtype)
        b_np = self.array_or_scalar(numpy, rhs, dtype=dtype)

        if (
            name == "subtract"
            and not numpy.isscalar(rhs)
            and dtype == dpnp.bool
        ):
            with pytest.raises(TypeError):
                result = getattr(dpnp, name)(a_dpnp, b_dpnp)
                expected = getattr(numpy, name)(a_np, b_np)
        else:
            result = getattr(dpnp, name)(a_dpnp, b_dpnp)
            expected = getattr(numpy, name)(a_np, b_np)
            assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_add(self, dtype, lhs, rhs):
        self._test_mathematical("add", dtype, lhs, rhs)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_arctan2(self, dtype, lhs, rhs):
        self._test_mathematical("arctan2", dtype, lhs, rhs)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_copysign(self, dtype, lhs, rhs):
        self._test_mathematical("copysign", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_divide(self, dtype, lhs, rhs):
        self._test_mathematical("divide", dtype, lhs, rhs)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_fmod(self, dtype, lhs, rhs):
        self._test_mathematical("fmod", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_floor_divide(self, dtype, lhs, rhs):
        if dtype == dpnp.float32 and rhs == 0.3:
            pytest.skip(
                "In this case, a different result, but similar to xp.floor(xp.divide(lhs, rhs)."
            )
        self._test_mathematical("floor_divide", dtype, lhs, rhs)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_hypot(self, dtype, lhs, rhs):
        self._test_mathematical("hypot", dtype, lhs, rhs)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_maximum(self, dtype, lhs, rhs):
        self._test_mathematical("maximum", dtype, lhs, rhs)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_minimum(self, dtype, lhs, rhs):
        self._test_mathematical("minimum", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_multiply(self, dtype, lhs, rhs):
        self._test_mathematical("multiply", dtype, lhs, rhs)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_remainder(self, dtype, lhs, rhs):
        self._test_mathematical("remainder", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_power(self, dtype, lhs, rhs):
        self._test_mathematical("power", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_subtract(self, dtype, lhs, rhs):
        self._test_mathematical("subtract", dtype, lhs, rhs)


@pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
@pytest.mark.parametrize(
    "val_type", [bool, int, float], ids=["bool", "int", "float"]
)
@pytest.mark.parametrize("data_type", get_all_dtypes())
@pytest.mark.parametrize(
    "func", ["add", "divide", "multiply", "power", "subtract"]
)
@pytest.mark.parametrize("val", [0, 1, 5], ids=["0", "1", "5"])
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
def test_op_with_scalar(array, val, func, data_type, val_type):
    np_a = numpy.array(array, dtype=data_type)
    dpnp_a = dpnp.array(array, dtype=data_type)
    val_ = val_type(val)

    if func == "power":
        if val_ == 0 and numpy.issubdtype(data_type, numpy.complexfloating):
            pytest.skip(
                "(0j ** 0) is different: (NaN + NaNj) in dpnp and (1 + 0j) in numpy"
            )
        elif is_cpu_device() and data_type == dpnp.complex128:
            # TODO: discuss the bahavior with OneMKL team
            pytest.skip(
                "(0j ** 5) is different: (NaN + NaNj) in dpnp and (0j) in numpy"
            )

    if func == "subtract" and val_type == bool and data_type == dpnp.bool:
        with pytest.raises(TypeError):
            result = getattr(dpnp, func)(dpnp_a, val_)
            expected = getattr(numpy, func)(np_a, val_)

            result = getattr(dpnp, func)(val_, dpnp_a)
            expected = getattr(numpy, func)(val_, np_a)
    else:
        result = getattr(dpnp, func)(dpnp_a, val_)
        expected = getattr(numpy, func)(np_a, val_)
        assert_allclose(result, expected, rtol=1e-6)

        result = getattr(dpnp, func)(val_, dpnp_a)
        expected = getattr(numpy, func)(val_, np_a)
        assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("shape", [(), (3, 2)], ids=["()", "(3, 2)"])
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_multiply_scalar(shape, dtype):
    np_a = numpy.ones(shape, dtype=dtype)
    dpnp_a = dpnp.ones(shape, dtype=dtype)

    result = 0.5 * dpnp_a * 1.7
    expected = 0.5 * np_a * 1.7
    assert_allclose(result, expected)


@pytest.mark.parametrize("shape", [(), (3, 2)], ids=["()", "(3, 2)"])
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_add_scalar(shape, dtype):
    np_a = numpy.ones(shape, dtype=dtype)
    dpnp_a = dpnp.ones(shape, dtype=dtype)

    result = 0.5 + dpnp_a + 1.7
    expected = 0.5 + np_a + 1.7
    assert_allclose(result, expected)


@pytest.mark.parametrize("shape", [(), (3, 2)], ids=["()", "(3, 2)"])
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_subtract_scalar(shape, dtype):
    np_a = numpy.ones(shape, dtype=dtype)
    dpnp_a = dpnp.ones(shape, dtype=dtype)

    result = 0.5 - dpnp_a - 1.7
    expected = 0.5 - np_a - 1.7
    assert_allclose(result, expected)


@pytest.mark.parametrize("shape", [(), (3, 2)], ids=["()", "(3, 2)"])
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_divide_scalar(shape, dtype):
    np_a = numpy.ones(shape, dtype=dtype)
    dpnp_a = dpnp.ones(shape, dtype=dtype)

    result = 0.5 / dpnp_a / 1.7
    expected = 0.5 / np_a / 1.7
    assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize("shape", [(), (3, 2)], ids=["()", "(3, 2)"])
@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.skip("mute until in-place support in dpctl is done")
def test_power_scalar(shape, dtype):
    np_a = numpy.ones(shape, dtype=dtype)
    dpnp_a = dpnp.ones(shape, dtype=dtype)

    result = 4.2**dpnp_a**-1.3
    expected = 4.2**np_a**-1.3
    assert_allclose(result, expected, rtol=1e-6)

    result **= dpnp_a
    expected **= np_a
    assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize(
    "array",
    [
        [1, 2, 3, 4, 5],
        [1, 2, numpy.nan, 4, 5],
        [[1, 2, numpy.nan], [3, -4, -5]],
    ],
)
def test_nancumprod(array):
    np_a = numpy.array(array)
    dpnp_a = dpnp.array(np_a)

    result = dpnp.nancumprod(dpnp_a)
    expected = numpy.nancumprod(np_a)
    assert_array_equal(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize(
    "array",
    [
        [1, 2, 3, 4, 5],
        [1, 2, numpy.nan, 4, 5],
        [[1, 2, numpy.nan], [3, -4, -5]],
    ],
)
def test_nancumsum(array):
    np_a = numpy.array(array)
    dpnp_a = dpnp.array(np_a)

    result = dpnp.nancumsum(dpnp_a)
    expected = numpy.nancumsum(np_a)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "data",
    [[[1.0, -1.0], [0.1, -0.1]], [-2, -1, 0, 1, 2]],
    ids=["[[1., -1.], [0.1, -0.1]]", "[-2, -1, 0, 1, 2]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
def test_negative(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

    result = dpnp.negative(dpnp_a)
    expected = numpy.negative(np_a)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("val_type", get_all_dtypes(no_none=True))
@pytest.mark.parametrize("data_type", get_all_dtypes())
@pytest.mark.parametrize("val", [1.5, 1, 5], ids=["1.5", "1", "5"])
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
def test_power(array, val, data_type, val_type):
    np_a = numpy.array(array, dtype=data_type)
    dpnp_a = dpnp.array(array, dtype=data_type)
    val_ = val_type(val)

    if is_cpu_device() and (
        dpnp.complex128 in (data_type, val_type)
        or dpnp.complex64 in (data_type, val_type)
    ):
        # TODO: discuss the behavior with OneMKL team
        pytest.skip(
            "(0j ** 5) is different: (NaN + NaNj) in dpnp and (0j) in numpy"
        )

    result = dpnp.power(dpnp_a, val_)
    expected = numpy.power(np_a, val_)
    assert_allclose(expected, result, rtol=1e-6)


class TestEdiff1d:
    @pytest.mark.parametrize(
        "data_type", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "array",
        [
            [1, 2, 4, 7, 0],
            [],
            [1],
            [[1, 2, 3], [5, 2, 8], [7, 3, 4]],
        ],
    )
    def test_ediff1d_int(self, array, data_type):
        np_a = numpy.array(array, dtype=data_type)
        dpnp_a = dpnp.array(array, dtype=data_type)

        result = dpnp.ediff1d(dpnp_a)
        expected = numpy.ediff1d(np_a)
        assert_array_equal(expected, result)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_ediff1d_args(self):
        np_a = numpy.array([1, 2, 4, 7, 0])

        to_begin = numpy.array([-20, -30])
        to_end = numpy.array([20, 15])

        result = dpnp.ediff1d(np_a, to_end=to_end, to_begin=to_begin)
        expected = numpy.ediff1d(np_a, to_end=to_end, to_begin=to_begin)
        assert_array_equal(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestTrapz:
    @pytest.mark.parametrize(
        "data_type", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "array",
        [[1, 2, 3], [[1, 2, 3], [4, 5, 6]], [1, 4, 6, 9, 10, 12], [], [1]],
    )
    def test_trapz_default(self, array, data_type):
        np_a = numpy.array(array, dtype=data_type)
        dpnp_a = dpnp.array(array, dtype=data_type)

        result = dpnp.trapz(dpnp_a)
        expected = numpy.trapz(np_a)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "data_type_y", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "data_type_x", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize("y_array", [[1, 2, 4, 5], [1.0, 2.5, 6.0, 7.0]])
    @pytest.mark.parametrize("x_array", [[2, 5, 6, 9]])
    def test_trapz_with_x_params(
        self, y_array, x_array, data_type_y, data_type_x
    ):
        np_y = numpy.array(y_array, dtype=data_type_y)
        dpnp_y = dpnp.array(y_array, dtype=data_type_y)

        np_x = numpy.array(x_array, dtype=data_type_x)
        dpnp_x = dpnp.array(x_array, dtype=data_type_x)

        result = dpnp.trapz(dpnp_y, dpnp_x)
        expected = numpy.trapz(np_y, np_x)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("array", [[1, 2, 3], [4, 5, 6]])
    def test_trapz_with_x_param_2ndim(self, array):
        np_a = numpy.array(array)
        dpnp_a = dpnp.array(array)

        result = dpnp.trapz(dpnp_a, dpnp_a)
        expected = numpy.trapz(np_a, np_a)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "y_array",
        [
            [1, 2, 4, 5],
            [
                1.0,
                2.5,
                6.0,
                7.0,
            ],
        ],
    )
    @pytest.mark.parametrize("dx", [2, 3, 4])
    def test_trapz_with_dx_params(self, y_array, dx):
        np_y = numpy.array(y_array)
        dpnp_y = dpnp.array(y_array)

        result = dpnp.trapz(dpnp_y, dx=dx)
        expected = numpy.trapz(np_y, dx=dx)
        assert_array_equal(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestCross:
    @pytest.mark.parametrize("axis", [None, 0], ids=["None", "0"])
    @pytest.mark.parametrize("axisc", [-1, 0], ids=["-1", "0"])
    @pytest.mark.parametrize("axisb", [-1, 0], ids=["-1", "0"])
    @pytest.mark.parametrize("axisa", [-1, 0], ids=["-1", "0"])
    @pytest.mark.parametrize(
        "x1",
        [[1, 2, 3], [1.0, 2.5, 6.0], [2, 4, 6]],
        ids=["[1, 2, 3]", "[1., 2.5, 6.]", "[2, 4, 6]"],
    )
    @pytest.mark.parametrize(
        "x2",
        [[4, 5, 6], [1.0, 5.0, 2.0], [6, 4, 3]],
        ids=["[4, 5, 6]", "[1., 5., 2.]", "[6, 4, 3]"],
    )
    def test_cross_3x3(self, x1, x2, axisa, axisb, axisc, axis):
        np_x1 = numpy.array(x1)
        dpnp_x1 = dpnp.array(x1)

        np_x2 = numpy.array(x2)
        dpnp_x2 = dpnp.array(x2)

        result = dpnp.cross(dpnp_x1, dpnp_x2, axisa, axisb, axisc, axis)
        expected = numpy.cross(np_x1, np_x2, axisa, axisb, axisc, axis)
        assert_array_equal(expected, result)


class TestGradient:
    @pytest.mark.parametrize(
        "array", [[2, 3, 6, 8, 4, 9], [3.0, 4.0, 7.5, 9.0], [2, 6, 8, 10]]
    )
    def test_gradient_y1(self, array):
        np_y = numpy.array(array)
        dpnp_y = dpnp.array(array)

        result = dpnp.gradient(dpnp_y)
        expected = numpy.gradient(np_y)
        assert_array_equal(expected, result)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "array", [[2, 3, 6, 8, 4, 9], [3.0, 4.0, 7.5, 9.0], [2, 6, 8, 10]]
    )
    @pytest.mark.parametrize("dx", [2, 3.5])
    def test_gradient_y1_dx(self, array, dx):
        np_y = numpy.array(array)
        dpnp_y = dpnp.array(array)

        result = dpnp.gradient(dpnp_y, dx)
        expected = numpy.gradient(np_y, dx)
        assert_array_equal(expected, result)


class TestCeil:
    def test_ceil(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.ceil(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.ceil(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.ceil(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.ceil(dp_array, out=dp_out)


class TestFloor:
    def test_floor(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.floor(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.floor(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.floor(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.floor(dp_array, out=dp_out)


class TestTrunc:
    def test_trunc(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.trunc(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.trunc(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.trunc(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.trunc(dp_array, out=dp_out)


class TestAdd:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_add(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.add(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.add(np_array1, np_array2, out=out)

        assert_allclose(expected, result)
        assert_allclose(out, dp_out)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_out_dtypes(self, dtype):
        size = 2 if dtype == dpnp.bool else 10

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.complex64)
        expected = numpy.add(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)

        dp_out = dpnp.empty(size, dtype=dpnp.complex64)
        if dtype != dpnp.complex64:
            # dtype of out mismatches types of input arrays
            with pytest.raises(TypeError):
                dpnp.add(dp_array1, dp_array2, out=dp_out)

            # allocate new out with expected type
            dp_out = dpnp.empty(size, dtype=dtype)

        result = dpnp.add(dp_array1, dp_array2, out=dp_out)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        with pytest.raises(TypeError):
            dpnp.add(dp_a[size::], dp_a[::2], out=dp_a[:size:])

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.skip("mute unttil in-place support in dpctl is done")
    def test_inplace_strided_out(self, dtype):
        size = 21

        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] += 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] += 4

        assert_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10, dtype=dpnp.float64)
        dp_array2 = dpnp.arange(5, 15, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(TypeError):
            dpnp.add(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.add, a, 2, out)
        assert_raises(TypeError, numpy.add, a.asnumpy(), 2, out)


class TestMultiply:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_multiply(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.multiply(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.multiply(np_array1, np_array2, out=out)

        assert_allclose(expected, result)
        assert_allclose(out, dp_out)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_out_dtypes(self, dtype):
        size = 2 if dtype == dpnp.bool else 10

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.complex64)
        expected = numpy.multiply(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)

        dp_out = dpnp.empty(size, dtype=dpnp.complex64)
        if dtype != dpnp.complex64:
            # dtype of out mismatches types of input arrays
            with pytest.raises(TypeError):
                dpnp.multiply(dp_array1, dp_array2, out=dp_out)

            # allocate new out with expected type
            dp_out = dpnp.empty(size, dtype=dtype)

        result = dpnp.multiply(dp_array1, dp_array2, out=dp_out)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        with pytest.raises(TypeError):
            dpnp.multiply(dp_a[size::], dp_a[::2], out=dp_a[:size:])

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.skip("mute unttil in-place support in dpctl is done")
    def test_inplace_strided_out(self, dtype):
        size = 21

        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] *= 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] *= 4

        assert_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10, dtype=dpnp.float64)
        dp_array2 = dpnp.arange(5, 15, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(TypeError):
            dpnp.multiply(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.multiply, a, 2, out)
        assert_raises(TypeError, numpy.multiply, a.asnumpy(), 2, out)


class TestPower:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_power(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.power(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.power(np_array1, np_array2, out=out)

        if dtype is dpnp.complex128:
            # ((0 + 0j) ** 2) == (Nan + NaNj) in dpnp and == (0 + 0j) in numpy
            assert_allclose(expected[1:], result[1:], rtol=1e-06)
        else:
            assert_allclose(expected, result, rtol=1e-06)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_out_dtypes(self, dtype):
        size = 2 if dtype == dpnp.bool else 5

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.complex64)
        expected = numpy.power(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)
        dp_out = dpnp.empty(size, dtype=dpnp.complex64)
        if dtype != dpnp.complex64:
            # dtype of out mismatches types of input arrays
            with pytest.raises(TypeError):
                dpnp.power(dp_array1, dp_array2, out=dp_out)

            # allocate new out with expected type
            out_dtype = dtype if dtype != dpnp.bool else dpnp.int64
            dp_out = dpnp.empty(size, dtype=out_dtype)

        result = dpnp.power(dp_array1, dp_array2, out=dp_out)
        assert_allclose(expected, result, rtol=1e-06)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_out_overlap(self, dtype):
        size = 5
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        with pytest.raises(TypeError):
            dpnp.power(dp_a[size::], dp_a[::2], out=dp_a[:size:])

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.skip("mute until in-place support in dpctl is done")
    def test_inplace_strided_out(self, dtype):
        size = 5

        np_a = numpy.arange(2 * size, dtype=dtype)
        np_a[::3] **= 3

        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dp_a[::3] **= 3

        assert_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10, dtype=dpnp.float32)
        dp_array2 = dpnp.arange(5, 15, dtype=dpnp.float32)
        dp_out = dpnp.empty(shape, dtype=dpnp.float32)

        with pytest.raises(TypeError):
            dpnp.power(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.power, a, 2, out)
        assert_raises(TypeError, numpy.power, a.asnumpy(), 2, out)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    def test_complex_values(self):
        np_arr = numpy.array([0j, 1 + 1j, 0 + 2j, 1 + 2j, numpy.nan, numpy.inf])
        dp_arr = dpnp.array(np_arr)
        func = lambda x: x**2

        # Linux: ((inf + 0j) ** 2) == (Inf + NaNj) in dpnp and == (NaN + NaNj) in numpy
        # Win:   ((inf + 0j) ** 2) == (Inf + 0j)   in dpnp and == (Inf + NaNj) in numpy
        if is_win_platform():
            assert_equal(func(dp_arr)[5], numpy.inf)
        else:
            assert_equal(func(dp_arr)[5], (numpy.inf + 0j) * 1)
        assert_allclose(func(np_arr)[:5], func(dp_arr).asnumpy()[:5], rtol=1e-6)

    @pytest.mark.parametrize("val", [0, 1], ids=["0", "1"])
    @pytest.mark.parametrize("dtype", [dpnp.int32, dpnp.int64])
    def test_integer_power_of_0_or_1(self, val, dtype):
        np_arr = numpy.arange(10, dtype=dtype)
        dp_arr = dpnp.array(np_arr)
        func = lambda x: 1**x

        assert_equal(func(np_arr), func(dp_arr))

    @pytest.mark.parametrize("dtype", [dpnp.int32, dpnp.int64])
    def test_integer_to_negative_power(self, dtype):
        a = dpnp.arange(2, 10, dtype=dtype)
        b = dpnp.full(8, -2, dtype=dtype)
        zeros = dpnp.zeros(8, dtype=dtype)
        ones = dpnp.ones(8, dtype=dtype)

        assert_array_equal(ones ** (-2), zeros)
        assert_equal(
            a ** (-3), zeros
        )  # positive integer to negative integer power
        assert_equal(
            b ** (-4), zeros
        )  # negative integer to negative integer power

    def test_float_to_inf(self):
        a = numpy.array(
            [1, 1, 2, 2, -2, -2, numpy.inf, -numpy.inf], dtype=numpy.float32
        )
        b = numpy.array(
            [
                numpy.inf,
                -numpy.inf,
                numpy.inf,
                -numpy.inf,
                numpy.inf,
                -numpy.inf,
                numpy.inf,
                -numpy.inf,
            ],
            dtype=numpy.float32,
        )
        numpy_res = a**b
        dpnp_res = dpnp.array(a) ** dpnp.array(b)

        assert_allclose(numpy_res, dpnp_res.asnumpy())


@pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True, no_bool=True))
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3])
def test_sum_empty(dtype, axis):
    a = numpy.empty((1, 2, 0, 4), dtype=dtype)
    numpy_res = a.sum(axis=axis)
    dpnp_res = dpnp.array(a).sum(axis=axis)
    assert_array_equal(numpy_res, dpnp_res.asnumpy())


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True, no_bool=True))
def test_sum_empty_out(dtype):
    a = dpnp.empty((1, 2, 0, 4), dtype=dtype)
    out = dpnp.ones(())
    res = a.sum(out=out)
    assert_array_equal(out.asnumpy(), res.asnumpy())
    assert_array_equal(out.asnumpy(), numpy.array(0, dtype=dtype))


@pytest.mark.parametrize(
    "shape", [(), (1, 2, 3), (1, 0, 2), (10), (3, 3, 3), (5, 5), (0, 6)]
)
@pytest.mark.parametrize(
    "dtype_in", get_all_dtypes(no_complex=True, no_bool=True)
)
@pytest.mark.parametrize(
    "dtype_out", get_all_dtypes(no_complex=True, no_bool=True)
)
def test_sum(shape, dtype_in, dtype_out):
    a_np = numpy.ones(shape, dtype=dtype_in)
    a = dpnp.ones(shape, dtype=dtype_in)
    axes = [None, 0, 1, 2]
    for axis in axes:
        if axis is None or axis < a.ndim:
            numpy_res = a_np.sum(axis=axis, dtype=dtype_out)
            dpnp_res = a.sum(axis=axis, dtype=dtype_out)
            assert_array_equal(numpy_res, dpnp_res.asnumpy())


class TestMean:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_mean_axis_tuple(self, dtype):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.mean(dp_array, axis=(0, 1))
        expected = numpy.mean(np_array, axis=(0, 1))
        assert_allclose(expected, result)

    def test_mean_axis_zero_size(self):
        dp_array = dpnp.array([], dtype="int64")
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.mean(dp_array)
        expected = numpy.mean(np_array)
        assert_allclose(expected, result)

    def test_mean_strided(self):
        dp_array = dpnp.array([-2, -1, 0, 1, 0, 2], dtype="f4")
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.mean(dp_array[::-1])
        expected = numpy.mean(np_array[::-1])
        assert_allclose(expected, result)

        result = dpnp.mean(dp_array[::2])
        expected = numpy.mean(np_array[::2])
        assert_allclose(expected, result)

    def test_mean_scalar(self):
        dp_array = dpnp.array(5)
        np_array = dpnp.asnumpy(dp_array)

        result = dp_array.mean()
        expected = np_array.mean()
        assert_allclose(expected, result)
