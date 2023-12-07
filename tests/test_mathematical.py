from itertools import permutations

import dpctl
import dpctl.tensor as dpt
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
    assert_dtype_allclose,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    has_support_aspect64,
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

    def _test_mathematical(self, name, dtype, lhs, rhs, check_type=True):
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
            assert_dtype_allclose(result, expected, check_type)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_add(self, dtype, lhs, rhs):
        self._test_mathematical("add", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
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

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_fmax(self, dtype, lhs, rhs):
        self._test_mathematical("fmax", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_fmin(self, dtype, lhs, rhs):
        self._test_mathematical("fmin", dtype, lhs, rhs, check_type=False)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_fmod(self, dtype, lhs, rhs):
        if rhs == 0.3:
            """
            Due to accuracy reason, the results are different for `float32` and `float64`
                >>> numpy.fmod(numpy.array([3.9], dtype=numpy.float32), 0.3)
                array([0.29999995], dtype=float32)

                >>> numpy.fmod(numpy.array([3.9], dtype=numpy.float64), 0.3)
                array([9.53674318e-08])
            On a gpu without support for `float64`, dpnp produces results similar to the second one.
            """
            pytest.skip("Due to accuracy reason, the results are different.")
        self._test_mathematical("fmod", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_floor_divide(self, dtype, lhs, rhs):
        if dtype == dpnp.float32 and rhs == 0.3:
            pytest.skip(
                "In this case, a different result, but similar to xp.floor(xp.divide(lhs, rhs)."
            )
        self._test_mathematical(
            "floor_divide", dtype, lhs, rhs, check_type=False
        )

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_hypot(self, dtype, lhs, rhs):
        self._test_mathematical("hypot", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_maximum(self, dtype, lhs, rhs):
        self._test_mathematical("maximum", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_minimum(self, dtype, lhs, rhs):
        self._test_mathematical("minimum", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_multiply(self, dtype, lhs, rhs):
        self._test_mathematical("multiply", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_remainder(self, dtype, lhs, rhs):
        if (
            dtype in [dpnp.int32, dpnp.int64, None]
            and rhs == 0.3
            and not has_support_aspect64()
        ):
            """
            Due to accuracy reason, the results are different for `float32` and `float64`
                >>> numpy.remainder(numpy.array([6, 3], dtype='i4'), 0.3, dtype='f8')
                array([2.22044605e-16, 1.11022302e-16])

                >>> numpy.remainder(numpy.array([6, 3], dtype='i4'), 0.3, dtype='f4')
                usm_ndarray([0.29999977, 0.2999999 ], dtype=float32)
            On a gpu without support for `float64`, dpnp produces results similar to the second one.
            """
            pytest.skip("Due to accuracy reason, the results are different.")
        self._test_mathematical("remainder", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_power(self, dtype, lhs, rhs):
        self._test_mathematical("power", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_subtract(self, dtype, lhs, rhs):
        self._test_mathematical("subtract", dtype, lhs, rhs, check_type=False)


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
        if (
            val_ == 0
            and numpy.issubdtype(data_type, numpy.complexfloating)
            and not dpnp.all(dpnp_a)
        ):
            pytest.skip(
                "(0j ** 0) is different: (NaN + NaNj) in dpnp and (1 + 0j) in numpy"
            )
        # TODO: Remove when #1378 (dpctl) is solved
        elif (
            is_cpu_device()
            and dpnp_a.dtype == dpnp.complex128
            and dpnp_a.size >= 8
            and not dpnp.all(dpnp_a)
        ):
            pytest.skip(
                "[..., 0j ** val] is different for x.size >= 8: [..., NaN + NaNj] in dpnp and [..., 0 + 0j] in numpy"
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
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_negative(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

    result = dpnp.negative(dpnp_a)
    expected = numpy.negative(np_a)
    assert_allclose(result, expected)


def test_negative_boolean():
    dpnp_a = dpnp.array([True, False])

    with pytest.raises(TypeError):
        dpnp.negative(dpnp_a)


@pytest.mark.parametrize(
    "data",
    [[[1.0, -1.0], [0.1, -0.1]], [-2, -1, 0, 1, 2]],
    ids=["[[1., -1.], [0.1, -0.1]]", "[-2, -1, 0, 1, 2]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_positive(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

    result = dpnp.positive(dpnp_a)
    expected = numpy.positive(np_a)
    assert_allclose(result, expected)

    result = +dpnp_a
    expected = +np_a
    assert_allclose(result, expected)


def test_positive_boolean():
    dpnp_a = dpnp.array([True, False])

    with pytest.raises(TypeError):
        dpnp.positive(dpnp_a)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("func", ["prod", "nanprod"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_prod_nanprod(func, axis, keepdims, dtype):
    a = numpy.arange(1, 13, dtype=dtype).reshape((2, 2, 3))
    if func == "nanprod" and issubclass(a.dtype.type, dpnp.inexact):
        a[1:2:] = numpy.nan
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
    dpnp_res = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
def test_prod_zero_size(axis):
    a = numpy.empty((2, 3, 0))
    ia = dpnp.array(a)

    np_res = numpy.prod(a, axis=axis)
    dpnp_res = dpnp.prod(ia, axis=axis)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["prod", "nanprod"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_prod_nanprod_bool(func, axis, keepdims):
    a = numpy.arange(2, dtype=dpnp.bool)
    a = numpy.tile(a, (2, 2))
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
    dpnp_res = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.usefixtures("suppress_complex_warning")
@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
@pytest.mark.parametrize("func", ["prod", "nanprod"])
@pytest.mark.parametrize("in_dtype", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize("out_dtype", get_all_dtypes(no_bool=True))
def test_prod_nanprod_dtype(func, in_dtype, out_dtype):
    a = numpy.arange(1, 13, dtype=in_dtype).reshape((2, 2, 3))
    if func == "nanprod" and issubclass(a.dtype.type, dpnp.inexact):
        a[1:2:] = numpy.nan
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, dtype=out_dtype)
    dpnp_res = getattr(dpnp, func)(ia, dtype=out_dtype)

    if out_dtype is not None:
        assert dpnp_res.dtype == out_dtype
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["prod", "nanprod"])
def test_prod_nanprod_out(func):
    a = numpy.arange(1, 7).reshape((2, 3))
    if func == "nanprod" and issubclass(a.dtype.type, dpnp.inexact):
        a[1:2:] = numpy.nan
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=0)
    dpnp_res = dpnp.array(numpy.empty_like(np_res))
    getattr(dpnp, func)(ia, axis=0, out=dpnp_res)
    assert_allclose(dpnp_res, np_res)

    dpnp_res = dpt.asarray(numpy.empty_like(np_res))
    getattr(dpnp, func)(ia, axis=0, out=dpnp_res)
    assert_allclose(dpnp_res, np_res)

    dpnp_res = numpy.empty_like(np_res)
    with pytest.raises(TypeError):
        getattr(dpnp, func)(ia, axis=0, out=dpnp_res)

    dpnp_res = dpnp.array(numpy.empty((2, 3)))
    with pytest.raises(ValueError):
        getattr(dpnp, func)(ia, axis=0, out=dpnp_res)


def test_prod_nanprod_Error():
    ia = dpnp.arange(5)

    with pytest.raises(TypeError):
        dpnp.prod(dpnp.asnumpy(ia))
    with pytest.raises(TypeError):
        dpnp.nanprod(dpnp.asnumpy(ia))
    with pytest.raises(NotImplementedError):
        dpnp.prod(ia, where=False)
    with pytest.raises(NotImplementedError):
        dpnp.prod(ia, initial=6)


@pytest.mark.parametrize(
    "data",
    [[2, 0, -2], [1.1, -1.1]],
    ids=["[2, 0, -2]", "[1.1, -1.1]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_sign(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

    result = dpnp.sign(dpnp_a)
    expected = numpy.sign(np_a)
    assert_allclose(result, expected)


def test_sign_boolean():
    dpnp_a = dpnp.array([True, False])

    with pytest.raises(TypeError):
        dpnp.sign(dpnp_a)


@pytest.mark.parametrize(
    "data",
    [[2, 0, -2], [1.1, -1.1]],
    ids=["[2, 0, -2]", "[1.1, -1.1]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
def test_signbit(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

    result = dpnp.signbit(dpnp_a)
    expected = numpy.signbit(np_a)
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "data",
    [complex(-1, -4), complex(-1, 2), complex(3, -7), complex(4, 12)],
    ids=[
        "complex(-1, -4)",
        "complex(-1, 2)",
        "complex(3, -7)",
        "complex(4, 12)",
    ],
)
@pytest.mark.parametrize("dtype", get_complex_dtypes())
def test_real_imag(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

    result = dpnp.real(dpnp_a)
    expected = numpy.real(np_a)
    assert_allclose(result, expected)

    result = dpnp.imag(dpnp_a)
    expected = numpy.imag(np_a)
    assert_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_complex_dtypes())
def test_projection_infinity(dtype):
    X = [
        complex(1, 2),
        complex(dpnp.inf, -1),
        complex(0, -dpnp.inf),
        complex(-dpnp.inf, dpnp.nan),
    ]
    Y = [
        complex(1, 2),
        complex(dpnp.inf, -0.0),
        complex(dpnp.inf, -0.0),
        complex(dpnp.inf, 0.0),
    ]

    result = dpnp.proj(dpnp.array(X, dtype=dtype))
    expected = dpnp.array(Y, dtype=dtype)
    assert_allclose(result, expected)


@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_projection(dtype):
    result = dpnp.proj(dpnp.array(1, dtype=dtype))
    expected = dpnp.array(complex(1, 0))
    assert_allclose(result, expected)


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

    # TODO: Remove when #1378 (dpctl) is solved
    if (
        is_cpu_device()
        and (
            dpnp.complex128 in (data_type, val_type)
            or dpnp.complex64 in (data_type, val_type)
        )
        and dpnp_a.size >= 8
    ):
        pytest.skip(
            "[..., 0j ** val] is different for x.size >= 8: [..., NaN + NaNj] in dpnp and [..., 0 + 0j] in numpy"
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
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_ceil(self, dtype):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.ceil(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.ceil(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.ceil(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.ceil(dp_array, out=dp_out)


class TestFloor:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_floor(self, dtype):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.floor(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.floor(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.floor(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.floor(dp_array, out=dp_out)


class TestTrunc:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_trunc(self, dtype):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.trunc(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.trunc(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.trunc(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

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
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.add(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.add(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_allclose(np_a, dp_a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
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
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
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


class TestDivide:
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_divide(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.divide(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.divide(np_array1, np_array2, out=out)

        assert_dtype_allclose(result, expected)
        assert_dtype_allclose(dp_out, out)

    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_out_dtypes(self, dtype):
        size = 10

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.complex64)
        expected = numpy.divide(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)

        dp_out = dpnp.empty(size, dtype=dpnp.complex64)
        if dtype != dpnp.complex64:
            # dtype of out mismatches types of input arrays
            with pytest.raises(TypeError):
                dpnp.divide(dp_array1, dp_array2, out=dp_out)

            # allocate new out with expected type
            dp_out = dpnp.empty(size, dtype=dtype)

        result = dpnp.divide(dp_array1, dp_array2, out=dp_out)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_out_overlap(self, dtype):
        size = 15
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.divide(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.divide(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_inplace_strided_out(self, dtype):
        size = 21

        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] /= 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] /= 4

        assert_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.divide(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.divide, a, 2, out)
        assert_raises(TypeError, numpy.divide, a.asnumpy(), 2, out)


class TestFloorDivide:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_floor_divide(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.floor_divide(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.floor_divide(np_array1, np_array2, out=out)

        assert_allclose(result, expected)
        assert_allclose(dp_out, out)

    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_out_dtypes(self, dtype):
        size = 10

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.complex64)
        expected = numpy.floor_divide(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)

        dp_out = dpnp.empty(size, dtype=dpnp.complex64)
        if dtype != dpnp.complex64:
            # dtype of out mismatches types of input arrays
            with pytest.raises(TypeError):
                dpnp.floor_divide(dp_array1, dp_array2, out=dp_out)

            # allocate new out with expected type
            dp_out = dpnp.empty(size, dtype=dtype)

        result = dpnp.floor_divide(dp_array1, dp_array2, out=dp_out)
        assert_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_out_overlap(self, dtype):
        size = 15
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.floor_divide(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.floor_divide(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_inplace_strided_out(self, dtype):
        size = 21

        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] //= 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] //= 4

        assert_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.floor_divide(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.floor_divide, a, 2, out)
        assert_raises(TypeError, numpy.floor_divide, a.asnumpy(), 2, out)


class TestFmax:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True)
    )
    def test_fmax(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.fmax(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.fmax(np_array1, np_array2, out=out)

        assert_allclose(expected, result)
        assert_allclose(out, dp_out)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True)
    )
    def test_out_dtypes(self, dtype):
        size = 10

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.float32)
        expected = numpy.fmax(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)
        with pytest.raises(TypeError):
            dpnp.fmax(dp_array1, dp_array2, out=np_out)

        dp_out = dpnp.empty(size, dtype=dpnp.float32)
        result = dpnp.fmax(dp_array1, dp_array2, out=dp_out)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True)
    )
    def test_out_overlap(self, dtype):
        size = 15
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.fmax(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.fmax(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_allclose(np_a, dp_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.fmax(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.fmax, a, 2, out)
        assert_raises(TypeError, numpy.fmax, a.asnumpy(), 2, out)


class TestFmin:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True)
    )
    def test_fmin(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.fmin(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.fmin(np_array1, np_array2, out=out)

        assert_allclose(expected, result)
        assert_allclose(out, dp_out)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True)
    )
    def test_out_dtypes(self, dtype):
        size = 10

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.float32)
        expected = numpy.fmin(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)
        with pytest.raises(TypeError):
            dpnp.fmin(dp_array1, dp_array2, out=np_out)

        dp_out = dpnp.empty(size, dtype=dpnp.float32)
        result = dpnp.fmin(dp_array1, dp_array2, out=dp_out)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True)
    )
    def test_out_overlap(self, dtype):
        size = 15
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.fmin(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.fmin(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_allclose(np_a, dp_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.fmin(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.fmin, a, 2, out)
        assert_raises(TypeError, numpy.fmin, a.asnumpy(), 2, out)


class TestHypot:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_hypot(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.hypot(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.hypot(np_array1, np_array2, out=out)

        assert_allclose(expected, result)
        assert_allclose(out, dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_out_dtypes(self, dtype):
        size = 10

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.float32)
        expected = numpy.hypot(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)

        dp_out = dpnp.empty(size, dtype=dpnp.float32)
        if dtype != dpnp.float32:
            # dtype of out mismatches types of input arrays
            with pytest.raises(TypeError):
                dpnp.hypot(dp_array1, dp_array2, out=dp_out)

            # allocate new out with expected type
            dp_out = dpnp.empty(size, dtype=dtype)

        result = dpnp.hypot(dp_array1, dp_array2, out=dp_out)

        tol = numpy.finfo(numpy.float32).resolution
        assert_allclose(expected, result, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_out_overlap(self, dtype):
        size = 15
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.hypot(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.hypot(np_a[size::], np_a[::2], out=np_a[:size:])

        tol = numpy.finfo(numpy.float32).resolution
        assert_allclose(np_a, dp_a, rtol=tol, atol=tol)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.hypot(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.hypot, a, 2, out)
        assert_raises(TypeError, numpy.hypot, a.asnumpy(), 2, out)


class TestMaximum:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_maximum(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.maximum(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.maximum(np_array1, np_array2, out=out)

        assert_allclose(expected, result)
        assert_allclose(out, dp_out)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_out_dtypes(self, dtype):
        size = 2 if dtype == dpnp.bool else 10

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.complex64)
        expected = numpy.maximum(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)

        dp_out = dpnp.empty(size, dtype=dpnp.complex64)
        if dtype != dpnp.complex64:
            # dtype of out mismatches types of input arrays
            with pytest.raises(TypeError):
                dpnp.maximum(dp_array1, dp_array2, out=dp_out)

            # allocate new out with expected type
            dp_out = dpnp.empty(size, dtype=dtype)

        result = dpnp.maximum(dp_array1, dp_array2, out=dp_out)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.maximum(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.maximum(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_allclose(np_a, dp_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.maximum(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.maximum, a, 2, out)
        assert_raises(TypeError, numpy.maximum, a.asnumpy(), 2, out)


class TestMinimum:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_minimum(self, dtype):
        array1_data = numpy.arange(10)
        array2_data = numpy.arange(5, 15)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array1 = dpnp.array(array1_data, dtype=dtype)
        dp_array2 = dpnp.array(array2_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.minimum(dp_array1, dp_array2, out=dp_out)

        # original
        np_array1 = numpy.array(array1_data, dtype=dtype)
        np_array2 = numpy.array(array2_data, dtype=dtype)
        expected = numpy.minimum(np_array1, np_array2, out=out)

        assert_allclose(expected, result)
        assert_allclose(out, dp_out)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_out_dtypes(self, dtype):
        size = 2 if dtype == dpnp.bool else 10

        np_array1 = numpy.arange(size, 2 * size, dtype=dtype)
        np_array2 = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.complex64)
        expected = numpy.minimum(np_array1, np_array2, out=np_out)

        dp_array1 = dpnp.arange(size, 2 * size, dtype=dtype)
        dp_array2 = dpnp.arange(size, dtype=dtype)

        dp_out = dpnp.empty(size, dtype=dpnp.complex64)
        if dtype != dpnp.complex64:
            # dtype of out mismatches types of input arrays
            with pytest.raises(TypeError):
                dpnp.minimum(dp_array1, dp_array2, out=dp_out)

            # allocate new out with expected type
            dp_out = dpnp.empty(size, dtype=dtype)

        result = dpnp.minimum(dp_array1, dp_array2, out=dp_out)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.minimum(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.minimum(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_allclose(np_a, dp_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            dpnp.minimum(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        assert_raises(TypeError, dpnp.minimum, a, 2, out)
        assert_raises(TypeError, numpy.minimum, a.asnumpy(), 2, out)


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
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.multiply(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.multiply(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_allclose(np_a, dp_a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
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
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(5, 15)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
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
            if dtype == dpnp.bool:
                out_dtype = numpy.int8
            else:
                out_dtype = dtype
            dp_out = dpnp.empty(size, dtype=out_dtype)

        result = dpnp.power(dp_array1, dp_array2, out=dp_out)
        assert_allclose(expected, result, rtol=1e-06)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_out_overlap(self, dtype):
        size = 10
        # DPNP
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.power(dp_a[size::], dp_a[::2], out=dp_a[:size:]),

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.power(np_a[size::], np_a[::2], out=np_a[:size:])

        rtol = 1e-05 if dtype is dpnp.complex64 else 1e-07
        assert_allclose(np_a, dp_a, rtol=rtol)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
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

        with pytest.raises(ValueError):
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

        # TODO: unmute the test once it's available
        if is_win_platform():
            pytest.skip(
                "Until the latest dpctl is available on internal channel"
            )

        assert_dtype_allclose(func(dp_arr), func(np_arr))

    @pytest.mark.parametrize("val", [0, 1], ids=["0", "1"])
    @pytest.mark.parametrize("dtype", [dpnp.int32, dpnp.int64])
    def test_integer_power_of_0_or_1(self, val, dtype):
        np_arr = numpy.arange(10, dtype=dtype)
        dp_arr = dpnp.array(np_arr)
        func = lambda x: val**x

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


@pytest.mark.usefixtures("suppress_complex_warning")
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1, 2, 3),
        (1, 0, 2),
        (10,),
        (3, 3, 3),
        (5, 5),
        (0, 6),
        (10, 1),
        (1, 10),
        (35, 40),
        (40, 35),
    ],
)
@pytest.mark.parametrize("dtype_in", get_all_dtypes())
@pytest.mark.parametrize("dtype_out", get_all_dtypes())
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("order", ["C", "F"])
def test_sum(shape, dtype_in, dtype_out, transpose, keepdims, order):
    size = numpy.prod(shape)
    a_np = numpy.arange(size).astype(dtype_in).reshape(shape, order=order)
    a = dpnp.asarray(a_np)

    if transpose:
        a_np = a_np.T
        a = a.T

    axes_range = list(numpy.arange(len(shape)))
    axes = [None]
    axes += axes_range
    axes += permutations(axes_range, 2)
    axes.append(tuple(axes_range))

    for axis in axes:
        numpy_res = a_np.sum(axis=axis, dtype=dtype_out, keepdims=keepdims)
        dpnp_res = a.sum(axis=axis, dtype=dtype_out, keepdims=keepdims)
        assert_array_equal(numpy_res, dpnp_res.asnumpy())


@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
)
def test_inplace_remainder(dtype):
    size = 21
    np_a = numpy.arange(size, dtype=dtype)
    dp_a = dpnp.arange(size, dtype=dtype)

    np_a %= 4
    dp_a %= 4

    assert_allclose(dp_a, np_a)


@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
)
def test_inplace_floor_divide(dtype):
    size = 21
    np_a = numpy.arange(size, dtype=dtype)
    dp_a = dpnp.arange(size, dtype=dtype)

    np_a //= 4
    dp_a //= 4

    assert_allclose(dp_a, np_a)


@pytest.mark.parametrize(
    "order_pair", [("C", "C"), ("C", "F"), ("F", "C"), ("F", "F")]
)
@pytest.mark.parametrize(
    "shape_pair",
    [
        ((4,), (4,)),
        ((4,), (4, 2)),
        ((2, 4), (4,)),
        ((2, 4), (4, 3)),
        ((1, 2, 3), (1, 3, 5)),
        ((4, 2, 3), (4, 3, 5)),
        ((1, 2, 3), (4, 3, 5)),
        ((2, 3), (4, 3, 5)),
        ((4, 2, 3), (1, 3, 5)),
        ((4, 2, 3), (3, 5)),
        ((1, 1, 4, 3), (1, 1, 3, 5)),
        ((6, 7, 4, 3), (6, 7, 3, 5)),
        ((6, 7, 4, 3), (1, 1, 3, 5)),
        ((6, 7, 4, 3), (1, 3, 5)),
        ((6, 7, 4, 3), (3, 5)),
        ((6, 7, 4, 3), (1, 7, 3, 5)),
        ((6, 7, 4, 3), (7, 3, 5)),
        ((6, 7, 4, 3), (6, 1, 3, 5)),
        ((1, 1, 4, 3), (6, 7, 3, 5)),
        ((1, 4, 3), (6, 7, 3, 5)),
        ((4, 3), (6, 7, 3, 5)),
        ((6, 1, 4, 3), (6, 7, 3, 5)),
        ((1, 7, 4, 3), (6, 7, 3, 5)),
        ((7, 4, 3), (6, 7, 3, 5)),
        ((1, 5, 3, 2), (6, 5, 2, 4)),
        ((5, 3, 2), (6, 5, 2, 4)),
        ((1, 3, 3), (10, 1, 3, 1)),
    ],
)
def test_matmul(order_pair, shape_pair):
    order1, order2 = order_pair
    shape1, shape2 = shape_pair
    a1 = numpy.arange(numpy.prod(shape1)).reshape(shape1)
    a2 = numpy.arange(numpy.prod(shape2)).reshape(shape2)
    a1 = numpy.array(a1, order=order1)
    a2 = numpy.array(a2, order=order2)

    b1 = dpnp.asarray(a1)
    b2 = dpnp.asarray(a2)

    result = dpnp.matmul(b1, b2)
    expected = numpy.matmul(a1, a2)
    assert_allclose(expected, result)


@pytest.mark.parametrize("dtype1", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize("dtype2", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize(
    "shape_pair",
    [
        ((2, 4), (4, 3)),
        ((4, 2, 3), (4, 3, 5)),
        ((6, 7, 4, 3), (6, 7, 3, 5)),
    ],
    ids=[
        "((2, 4), (4, 3))",
        "((4, 2, 3), (4, 3, 5))",
        "((6, 7, 4, 3), (6, 7, 3, 5))",
    ],
)
def test_matmul_dtype(dtype1, dtype2, shape_pair):
    shape1, shape2 = shape_pair
    a1 = numpy.arange(numpy.prod(shape1), dtype=dtype1).reshape(shape1)
    a2 = numpy.arange(numpy.prod(shape2), dtype=dtype2).reshape(shape2)

    b1 = dpnp.asarray(a1)
    b2 = dpnp.asarray(a2)

    result = dpnp.matmul(b1, b2)
    expected = numpy.matmul(a1, a2)
    assert_dtype_allclose(result, expected)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("order", ["C", "F", "K", "A"])
@pytest.mark.parametrize(
    "shape_pair",
    [
        ((2, 4), (4, 3)),
        ((4, 2, 3), (4, 3, 5)),
        ((6, 7, 4, 3), (6, 7, 3, 5)),
    ],
    ids=[
        "((2, 4), (4, 3))",
        "((4, 2, 3), (4, 3, 5))",
        "((6, 7, 4, 3), (6, 7, 3, 5))",
    ],
)
def test_matmul_order(order, shape_pair):
    shape1, shape2 = shape_pair
    a1 = numpy.arange(numpy.prod(shape1)).reshape(shape1)
    a2 = numpy.arange(numpy.prod(shape2)).reshape(shape2)

    b1 = dpnp.asarray(a1)
    b2 = dpnp.asarray(a2)

    result = dpnp.matmul(b1, b2, order=order)
    expected = numpy.matmul(a1, a2, order=order)
    assert result.flags.c_contiguous == expected.flags.c_contiguous
    assert result.flags.f_contiguous == expected.flags.f_contiguous
    assert_allclose(expected, result)


def test_matmul_strided():
    for dim in [1, 2, 3, 4]:
        A = numpy.random.rand(*([20] * dim))
        B = dpnp.asarray(A)
        # positive strides
        slices = tuple(slice(None, None, 2) for _ in range(dim))
        a = A[slices]
        b = B[slices]

        result = dpnp.matmul(b, b)
        expected = numpy.matmul(a, a)
        assert_allclose(expected, result, rtol=1e-06)

        # negative strides
        slices = tuple(slice(None, None, -2) for _ in range(dim))
        a = A[slices]
        b = B[slices]

        result = dpnp.matmul(b, b)
        expected = numpy.matmul(a, a)
        assert_allclose(expected, result, rtol=1e-06)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
def test_matmul_out(dtype):
    a1 = numpy.arange(5 * 4, dtype=dtype).reshape(5, 4)
    a2 = numpy.arange(7 * 4, dtype=dtype).reshape(4, 7)

    b1 = dpnp.asarray(a1)
    b2 = dpnp.asarray(a2)

    result = dpnp.empty((5, 7), dtype=dtype)
    dpnp.matmul(b1, b2, out=result)
    expected = numpy.matmul(a1, a2)
    assert_allclose(expected, result)


class TestMatmulInvalidCases:
    @pytest.mark.parametrize(
        "shape_pair",
        [
            ((3, 2), ()),
            ((), (3, 2)),
            ((), ()),
        ],
    )
    def test_zero_dim(self, shape_pair):
        for xp in (numpy, dpnp):
            shape1, shape2 = shape_pair
            x1 = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
            x2 = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)
            with pytest.raises(ValueError):
                xp.matmul(x1, x2)

    @pytest.mark.parametrize(
        "shape_pair",
        [
            ((5, 3, 1), (3, 1, 4)),
            ((3, 2, 3), (3, 2, 4)),
            ((3, 2), (1,)),
            ((1, 2), (3, 1)),
            ((4, 3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (3, 2, 4)),
        ],
    )
    def test_invalid_shape(self, shape_pair):
        for xp in (numpy, dpnp):
            shape1, shape2 = shape_pair
            x1 = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
            x2 = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)
            with pytest.raises(ValueError):
                xp.matmul(x1, x2)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True)[:-2])
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_none=True)[-1]
        a1 = dpnp.arange(5 * 4, dtype=dpnp_dtype).reshape(5, 4)
        a2 = dpnp.arange(7 * 4, dtype=dpnp_dtype).reshape(4, 7)
        dp_out = dpnp.empty((5, 7), dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.matmul(a1, a2, out=dp_out)

    def test_exe_q(self):
        try:
            x1 = dpnp.ones((5, 4), device="cpu")
        except dpctl.SyclDeviceCreationError:
            pytest.skip("No SYCL devices available")
        try:
            x2 = dpnp.ones((4, 7), device="gpu")
        except dpctl.SyclDeviceCreationError:
            pytest.skip("No SYCL devices available")
        with pytest.raises(ValueError):
            dpnp.matmul(x1, x2)

    def test_matmul_casting(self):
        a1 = dpnp.arange(2 * 4, dtype=dpnp.float32).reshape(2, 4)
        a2 = dpnp.arange(4 * 3).reshape(4, 3)

        res = dpnp.empty((2, 3), dtype=dpnp.int64)
        with pytest.raises(TypeError):
            dpnp.matmul(a1, a2, out=res, casting="safe")

    def test_matmul_subok(self):
        a1 = dpnp.arange(2 * 4).reshape(2, 4)
        a2 = dpnp.arange(4 * 3).reshape(4, 3)

        with pytest.raises(NotImplementedError):
            dpnp.matmul(a1, a2, subok=False)
