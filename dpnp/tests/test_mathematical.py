import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.tensor._numpy_helper import (
    AxisError,
    normalize_axis_index,
)
from dpctl.utils import ExecutionPlacementError
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)

import dpnp
from dpnp.dpnp_array import dpnp_array

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_abs_array,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    get_integer_dtypes,
    has_support_aspect16,
    has_support_aspect64,
    numpy_version,
)
from .test_umath import (
    _get_numpy_arrays_1in_1out,
    _get_numpy_arrays_2in_1out,
    _get_output_data_type,
)
from .third_party.cupy import testing


@pytest.mark.parametrize("deg", [True, False])
class TestAngle:
    def test_angle_bool(self, deg):
        dp_a = dpnp.array([True, False])
        np_a = dp_a.asnumpy()

        expected = numpy.angle(np_a, deg=deg)
        result = dpnp.angle(dp_a, deg=deg)

        # In NumPy, for boolean arguments the output data type is always default floating data type.
        # while data type of output in DPNP is determined by Type Promotion Rules.
        # data type should not be compared
        assert_allclose(result.asnumpy(), expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_angle(self, dtype, deg):
        dp_a = dpnp.arange(10, dtype=dtype)
        np_a = dp_a.asnumpy()

        expected = numpy.angle(np_a, deg=deg)
        result = dpnp.angle(dp_a, deg=deg)

        # For dtype=int8, uint8, NumPy returns float16, but dpnp returns float32
        dt_int8 = dtype in [dpnp.int8, dpnp.uint8]
        assert_dtype_allclose(result, expected, check_only_type_kind=dt_int8)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_angle_complex(self, dtype, deg):
        a = numpy.random.rand(10)
        b = numpy.random.rand(10)
        np_a = numpy.array(a + 1j * b, dtype=dtype)
        dp_a = dpnp.array(np_a)

        expected = numpy.angle(np_a, deg=deg)
        result = dpnp.angle(dp_a, deg=deg)

        assert_dtype_allclose(result, expected)


class TestConj:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_conj(self, dtype):
        a = generate_random_numpy_array(20, dtype)
        ia = dpnp.array(a)

        result = dpnp.conj(ia)
        expected = numpy.conj(a)
        assert_dtype_allclose(result, expected)

        # ndarray
        result = ia.conj()
        if not dpnp.issubdtype(dtype, dpnp.complexfloating):
            assert result is ia
        assert_dtype_allclose(result, a.conj())

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_conj_out(self, dtype):
        a = generate_random_numpy_array(20, dtype)
        ia = dpnp.array(a)

        expected = numpy.conj(a)
        dp_out = dpnp.empty(ia.shape, dtype=dtype)
        result = dpnp.conj(ia, out=dp_out)
        assert dp_out is result
        assert_dtype_allclose(result, expected)


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
        full_mode = dpnp.convolve(d, k, mode="full")
        assert_array_equal(full_mode, default_mode)
        # integer mode
        with assert_raises(ValueError):
            dpnp.convolve(d, k, mode=-1)
        assert_array_equal(dpnp.convolve(d, k, mode=2), full_mode)
        # illegal arguments
        with assert_raises(TypeError):
            dpnp.convolve(d, k, mode=None)


class TestClip:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("order", ["C", "F", "A", "K", None])
    def test_clip(self, dtype, order):
        dp_a = dpnp.asarray([[1, 2, 8], [1, 6, 4], [9, 5, 1]], dtype=dtype)
        np_a = dpnp.asnumpy(dp_a)

        result = dpnp.clip(dp_a, 2, 6, order=order)
        expected = numpy.clip(np_a, 2, 6, order=order)
        assert_allclose(expected, result)
        assert expected.flags.c_contiguous == result.flags.c_contiguous
        assert expected.flags.f_contiguous == result.flags.f_contiguous

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_clip_arrays(self, dtype):
        dp_a = dpnp.asarray([1, 2, 8, 1, 6, 4, 1], dtype=dtype)
        np_a = dpnp.asnumpy(dp_a)

        min_v = dpnp.asarray(2, dtype=dtype)
        max_v = dpnp.asarray(6, dtype=dtype)

        result = dpnp.clip(dp_a, min_v, max_v)
        expected = numpy.clip(np_a, min_v.asnumpy(), max_v.asnumpy())
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("in_dp", [dpnp, dpt])
    @pytest.mark.parametrize("out_dp", [dpnp, dpt])
    def test_clip_out(self, dtype, in_dp, out_dp):
        np_a = numpy.array([[1, 2, 8], [1, 6, 4], [9, 5, 1]], dtype=dtype)
        dp_a = in_dp.asarray(np_a)

        dp_out = out_dp.ones(dp_a.shape, dtype=dtype)
        np_out = numpy.ones(np_a.shape, dtype=dtype)

        result = dpnp.clip(dp_a, 2, 6, out=dp_out)
        expected = numpy.clip(np_a, 2, 6, out=np_out)
        assert_allclose(expected, result)
        assert_allclose(np_out, dp_out)
        assert isinstance(result, dpnp_array)

    def test_input_nan(self):
        np_a = numpy.array([-2.0, numpy.nan, 0.5, 3.0, 0.25, numpy.nan])
        dp_a = dpnp.array(np_a)

        result = dpnp.clip(dp_a, -1, 1)
        expected = numpy.clip(np_a, -1, 1)
        assert_array_equal(result, expected)

    @testing.with_requires("numpy>=1.25.0")
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"min": numpy.nan},
            {"max": numpy.nan},
            {"min": numpy.nan, "max": numpy.nan},
            {"min": -2, "max": numpy.nan},
            {"min": numpy.nan, "max": 10},
        ],
    )
    def test_nan_edges(self, kwargs):
        np_a = numpy.arange(7.0)
        dp_a = dpnp.asarray(np_a)

        result = dp_a.clip(**kwargs)
        expected = np_a.clip(**kwargs)
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"casting": "same_kind"},
            {"dtype": "i8"},
            {"subok": True},
            {"where": True},
        ],
    )
    def test_not_implemented_kwargs(self, kwargs):
        a = dpnp.arange(8, dtype="i4")

        numpy.clip(a.asnumpy(), 1, 5, **kwargs)
        with pytest.raises(NotImplementedError):
            dpnp.clip(a, 1, 5, **kwargs)


class TestCumLogSumExp:
    def _assert_arrays(self, res, exp, axis, include_initial):
        if include_initial:
            if axis != None:
                res_initial = dpnp.take(res, dpnp.array([0]), axis=axis)
                res_no_initial = dpnp.take(
                    res, dpnp.array(range(1, res.shape[axis])), axis=axis
                )
            else:
                res_initial = res[0]
                res_no_initial = res[1:]
            assert_dtype_allclose(res_no_initial, exp)
            assert (res_initial == -dpnp.inf).all()
        else:
            assert_dtype_allclose(res, exp)

    def _get_exp_array(self, a, axis, dtype):
        np_a = dpnp.asnumpy(a)
        if axis != None:
            return numpy.logaddexp.accumulate(np_a, axis=axis, dtype=dtype)
        return numpy.logaddexp.accumulate(np_a.ravel(), dtype=dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axis", [None, 2, -1])
    @pytest.mark.parametrize("include_initial", [True, False])
    def test_basic(self, dtype, axis, include_initial):
        a = dpnp.ones((3, 4, 5, 6, 7), dtype=dtype)
        res = dpnp.cumlogsumexp(a, axis=axis, include_initial=include_initial)

        exp_dt = None
        dtype_list = [dpnp.bool, dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        if dtype in dtype_list:
            exp_dt = dpnp.default_float_type(a.device)

        exp = self._get_exp_array(a, axis, exp_dt)
        self._assert_arrays(res, exp, axis, include_initial)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axis", [None, 2, -1])
    @pytest.mark.parametrize("include_initial", [True, False])
    def test_out(self, dtype, axis, include_initial):
        a = dpnp.ones((3, 4, 5, 6, 7), dtype=dtype)

        if dpnp.issubdtype(a, dpnp.float32):
            exp_dt = dpnp.float32
        else:
            exp_dt = dpnp.default_float_type(a.device)

        if axis != None:
            if include_initial:
                norm_axis = normalize_axis_index(axis, a.ndim, "axis")
                out_sh = (
                    a.shape[:norm_axis]
                    + (a.shape[norm_axis] + 1,)
                    + a.shape[norm_axis + 1 :]
                )
            else:
                out_sh = a.shape
        else:
            out_sh = (a.size + int(include_initial),)
        out = dpnp.empty_like(a, shape=out_sh, dtype=exp_dt)
        res = dpnp.cumlogsumexp(
            a, axis=axis, include_initial=include_initial, out=out
        )

        exp = self._get_exp_array(a, axis, exp_dt)

        assert res is out
        self._assert_arrays(res, exp, axis, include_initial)

    def test_axis_tuple(self):
        a = dpnp.ones((3, 4))
        assert_raises(TypeError, dpnp.cumlogsumexp, a, axis=(0, 1))

    @pytest.mark.parametrize(
        "in_dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize("out_dtype", get_all_dtypes(no_bool=True))
    def test_dtype(self, in_dtype, out_dtype):
        a = dpnp.ones(100, dtype=in_dtype)
        res = dpnp.cumlogsumexp(a, dtype=out_dtype)
        exp = numpy.logaddexp.accumulate(dpnp.asnumpy(a))
        exp = exp.astype(out_dtype)

        dtype_list = [dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        tol = 1e-2 if in_dtype in dtype_list else 1e-6
        assert_allclose(res, exp, rtol=tol)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "arr_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "out_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_out_dtype(self, arr_dt, out_dt, dtype):
        a = numpy.arange(10, 20).reshape((2, 5)).astype(dtype=arr_dt)
        out = numpy.zeros_like(a, dtype=out_dt)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = dpnp.cumlogsumexp(ia, out=iout, dtype=dtype, axis=1)
        exp = numpy.logaddexp.accumulate(a, out=out, axis=1)
        assert_allclose(result, exp.astype(dtype), rtol=1e-06)
        assert result is iout


class TestCumProd:
    @pytest.mark.parametrize(
        "arr, axis",
        [
            pytest.param([1, 2, 10, 11, 6, 5, 4], -1),
            pytest.param([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], 0),
            pytest.param([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], -1),
        ],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_axis(self, arr, axis, dtype):
        a = numpy.array(arr, dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.cumprod(ia, axis=axis)
        expected = numpy.cumprod(a, axis=axis)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_ndarray_method(self, dtype):
        a = numpy.arange(1, 10).astype(dtype=dtype)
        ia = dpnp.array(a)

        result = ia.cumprod()
        expected = a.cumprod()
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("sh", [(10,), (2, 5)])
    @pytest.mark.parametrize(
        "xp_in, xp_out, check",
        [
            pytest.param(dpt, dpt, False),
            pytest.param(dpt, dpnp, True),
            pytest.param(dpnp, dpt, False),
        ],
    )
    def test_usm_ndarray(self, sh, xp_in, xp_out, check):
        a = numpy.arange(-12, -2).reshape(sh)
        ia = xp_in.asarray(a)

        result = dpnp.cumprod(ia)
        expected = numpy.cumprod(a)
        assert_array_equal(expected, result)

        out = numpy.empty((10,))
        iout = xp_out.asarray(out)

        result = dpnp.cumprod(ia, out=iout)
        expected = numpy.cumprod(a, out=out)
        assert_array_equal(expected, result)
        assert (result is iout) is check

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("arr_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_out_dtype(self, arr_dt, out_dt, dtype):
        a = numpy.arange(1, 6).astype(dtype=arr_dt)
        out = numpy.zeros_like(a, dtype=out_dt)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = ia.cumprod(out=iout, dtype=dtype)
        expected = a.cumprod(out=out, dtype=dtype)
        assert_array_equal(expected, result)
        assert result is iout

    @testing.with_requires("numpy>=2.1.0")
    def test_include_initial(self):
        a = numpy.arange(8).reshape(2, 2, 2)
        ia = dpnp.array(a)

        expected = numpy.cumulative_prod(a, axis=1, include_initial=True)
        result = dpnp.cumulative_prod(ia, axis=1, include_initial=True)
        assert_array_equal(result, expected)

        expected = numpy.cumulative_prod(a, axis=0, include_initial=True)
        result = dpnp.cumulative_prod(ia, axis=0, include_initial=True)
        assert_array_equal(result, expected)

        a = numpy.arange(1, 5).reshape(2, 2)
        ia = dpnp.array(a)
        out = numpy.zeros((3, 2), dtype=numpy.float32)
        out_dp = dpnp.array(out)

        expected = numpy.cumulative_prod(
            a, axis=0, out=out, include_initial=True
        )
        result = dpnp.cumulative_prod(
            ia, axis=0, out=out_dp, include_initial=True
        )
        assert result is out_dp
        assert_array_equal(result, expected)

        a = numpy.array([2, 2])
        ia = dpnp.array(a)
        expected = numpy.cumulative_prod(a, include_initial=True)
        result = dpnp.cumulative_prod(ia, include_initial=True)
        assert_array_equal(result, expected)


class TestCumSum:
    @pytest.mark.parametrize(
        "arr, axis",
        [
            pytest.param([1, 2, 10, 11, 6, 5, 4], 0),
            pytest.param([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], 0),
            pytest.param([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], 1),
            pytest.param([[0, 1, 2], [3, 4, 5]], 0),
            pytest.param([[0, 1, 2], [3, 4, 5]], -1),
        ],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_axis(self, arr, axis, dtype):
        a = numpy.array(arr, dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.cumsum(ia, axis=axis)
        expected = numpy.cumsum(a, axis=axis)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_ndarray_method(self, dtype):
        a = numpy.arange(10).astype(dtype=dtype)
        ia = dpnp.array(a)

        result = ia.cumsum()
        expected = a.cumsum()
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("sh", [(10,), (2, 5)])
    @pytest.mark.parametrize(
        "xp_in, xp_out, check",
        [
            pytest.param(dpt, dpt, False),
            pytest.param(dpt, dpnp, True),
            pytest.param(dpnp, dpt, False),
        ],
    )
    def test_usm_ndarray(self, sh, xp_in, xp_out, check):
        a = numpy.arange(10).reshape(sh)
        ia = xp_in.asarray(a)

        result = dpnp.cumsum(ia)
        expected = numpy.cumsum(a)
        assert_array_equal(expected, result)

        out = numpy.empty((10,))
        iout = xp_out.asarray(out)

        result = dpnp.cumsum(ia, out=iout)
        expected = numpy.cumsum(a, out=out)
        assert_array_equal(expected, result)
        assert (result is iout) is check

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("arr_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_out_dtype(self, arr_dt, out_dt, dtype):
        a = numpy.arange(5, 15).astype(dtype=arr_dt)
        out = numpy.zeros_like(a, dtype=out_dt)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = ia.cumsum(out=iout, dtype=dtype)
        expected = a.cumsum(out=out, dtype=dtype)
        assert_array_equal(expected, result)
        assert result is iout

    @testing.with_requires("numpy>=2.1.0")
    def test_include_initial(self):
        a = numpy.arange(8).reshape(2, 2, 2)
        ia = dpnp.array(a)

        expected = numpy.cumulative_sum(a, axis=1, include_initial=True)
        result = dpnp.cumulative_sum(ia, axis=1, include_initial=True)
        assert_array_equal(result, expected)

        expected = numpy.cumulative_sum(a, axis=0, include_initial=True)
        result = dpnp.cumulative_sum(ia, axis=0, include_initial=True)
        assert_array_equal(result, expected)

        a = numpy.arange(1, 5).reshape(2, 2)
        ia = dpnp.array(a)
        out = numpy.zeros((3, 2), dtype=numpy.float32)
        out_dp = dpnp.array(out)

        expected = numpy.cumulative_sum(
            a, axis=0, out=out, include_initial=True
        )
        result = dpnp.cumulative_sum(
            ia, axis=0, out=out_dp, include_initial=True
        )
        assert result is out_dp
        assert_array_equal(result, expected)

        a = numpy.array([2, 2])
        ia = dpnp.array(a)
        expected = numpy.cumulative_sum(a, include_initial=True)
        result = dpnp.cumulative_sum(ia, include_initial=True)
        assert_array_equal(result, expected)


class TestDiff:
    @pytest.mark.parametrize("n", list(range(0, 3)))
    @pytest.mark.parametrize("dt", get_integer_dtypes())
    def test_basic_integer(self, n, dt):
        x = [1, 4, 6, 7, 12]
        np_a = numpy.array(x, dtype=dt)
        dpnp_a = dpnp.array(x, dtype=dt)

        expected = numpy.diff(np_a, n=n)
        result = dpnp.diff(dpnp_a, n=n)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_basic_floating(self, dt):
        x = [1.1, 2.2, 3.0, -0.2, -0.1]
        np_a = numpy.array(x, dtype=dt)
        dpnp_a = dpnp.array(x, dtype=dt)

        expected = numpy.diff(np_a)
        result = dpnp.diff(dpnp_a)
        assert_almost_equal(expected, result)

    @pytest.mark.parametrize("n", [1, 2])
    def test_basic_boolean(self, n):
        x = [True, True, False, False]
        np_a = numpy.array(x)
        dpnp_a = dpnp.array(x)

        expected = numpy.diff(np_a, n=n)
        result = dpnp.diff(dpnp_a, n=n)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_basic_complex(self, dt):
        x = [1.1 + 1j, 2.2 + 4j, 3.0 + 6j, -0.2 + 7j, -0.1 + 12j]
        np_a = numpy.array(x, dtype=dt)
        dpnp_a = dpnp.array(x, dtype=dt)

        expected = numpy.diff(np_a)
        result = dpnp.diff(dpnp_a)
        assert_allclose(expected, result)

    @pytest.mark.parametrize("axis", [None] + list(range(-3, 2)))
    def test_axis(self, axis):
        np_a = numpy.zeros((10, 20, 30))
        np_a[:, 1::2, :] = 1
        dpnp_a = dpnp.array(np_a)

        kwargs = {} if axis is None else {"axis": axis}
        expected = numpy.diff(np_a, **kwargs)
        result = dpnp.diff(dpnp_a, **kwargs)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("axis", [-4, 3])
    def test_axis_error(self, xp, axis):
        a = xp.ones((10, 20, 30))
        assert_raises(AxisError, xp.diff, a, axis=axis)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_ndim_error(self, xp):
        a = xp.array(1.1111111, xp.float32)
        assert_raises(ValueError, xp.diff, a)

    @pytest.mark.parametrize("n", [None, 2])
    @pytest.mark.parametrize("axis", [None, 0])
    def test_nd(self, n, axis):
        np_a = 20 * numpy.random.rand(10, 20, 30)
        dpnp_a = dpnp.array(np_a)

        kwargs = {} if n is None else {"n": n}
        if axis is not None:
            kwargs.update({"axis": axis})

        expected = numpy.diff(np_a, **kwargs)
        result = dpnp.diff(dpnp_a, **kwargs)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("n", list(range(0, 5)))
    def test_n(self, n):
        np_a = numpy.array(list(range(3)))
        dpnp_a = dpnp.array(np_a)

        expected = numpy.diff(np_a, n=n)
        result = dpnp.diff(dpnp_a, n=n)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_n_error(self, xp):
        a = xp.array(list(range(3)))
        assert_raises(ValueError, xp.diff, a, n=-1)

    @pytest.mark.parametrize("prepend", [0, [0], [-1, 0]])
    def test_prepend(self, prepend):
        np_a = numpy.arange(5) + 1
        dpnp_a = dpnp.array(np_a)

        np_p = prepend if numpy.isscalar(prepend) else numpy.array(prepend)
        dpnp_p = prepend if dpnp.isscalar(prepend) else dpnp.array(prepend)

        expected = numpy.diff(np_a, prepend=np_p)
        result = dpnp.diff(dpnp_a, prepend=dpnp_p)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "axis, prepend",
        [
            pytest.param(0, 0),
            pytest.param(0, [[0, 0]]),
            pytest.param(1, 0),
            pytest.param(1, [[0], [0]]),
        ],
    )
    def test_prepend_axis(self, axis, prepend):
        np_a = numpy.arange(4).reshape(2, 2)
        dpnp_a = dpnp.array(np_a)

        np_p = prepend if numpy.isscalar(prepend) else numpy.array(prepend)
        dpnp_p = prepend if dpnp.isscalar(prepend) else dpnp.array(prepend)

        expected = numpy.diff(np_a, axis=axis, prepend=np_p)
        result = dpnp.diff(dpnp_a, axis=axis, prepend=dpnp_p)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("append", [0, [0], [0, 2]])
    def test_append(self, append):
        np_a = numpy.arange(5)
        dpnp_a = dpnp.array(np_a)

        np_ap = append if numpy.isscalar(append) else numpy.array(append)
        dpnp_ap = append if dpnp.isscalar(append) else dpnp.array(append)

        expected = numpy.diff(np_a, append=np_ap)
        result = dpnp.diff(dpnp_a, append=dpnp_ap)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "axis, append",
        [
            pytest.param(0, 0),
            pytest.param(0, [[0, 0]]),
            pytest.param(1, 0),
            pytest.param(1, [[0], [0]]),
        ],
    )
    def test_append_axis(self, axis, append):
        np_a = numpy.arange(4).reshape(2, 2)
        dpnp_a = dpnp.array(np_a)

        np_ap = append if numpy.isscalar(append) else numpy.array(append)
        dpnp_ap = append if dpnp.isscalar(append) else dpnp.array(append)

        expected = numpy.diff(np_a, axis=axis, append=np_ap)
        result = dpnp.diff(dpnp_a, axis=axis, append=dpnp_ap)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_prepend_append_error(self, xp):
        a = xp.arange(4).reshape(2, 2)
        p = xp.zeros((3, 3))
        assert_raises(ValueError, xp.diff, a, prepend=p)
        assert_raises(ValueError, xp.diff, a, append=p)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_prepend_append_axis_error(self, xp):
        a = xp.arange(4).reshape(2, 2)
        assert_raises(AxisError, xp.diff, a, axis=3, prepend=0)
        assert_raises(AxisError, xp.diff, a, axis=3, append=0)


class TestEdiff1d:
    @pytest.mark.parametrize("data_type", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "array",
        [
            [1, 2, 4, 7, 0],
            [],
            [1],
            [[1, 2, 3], [5, 2, 8], [7, 3, 4]],
        ],
    )
    def test_ediff1d(self, array, data_type):
        np_a = numpy.array(array, dtype=data_type)
        dpnp_a = dpnp.array(array, dtype=data_type)

        result = dpnp.ediff1d(dpnp_a)
        expected = numpy.ediff1d(np_a)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "to_begin",
        [
            -20,
            numpy.array([-20, -30]),
            dpnp.array([-20, -30]),
            dpnp.array([[-20], [-30]]),
            [1, 2],
            (1, 2),
        ],
    )
    def test_ediff1d_to_begin(self, to_begin):
        np_a = numpy.array([1, 2, 4, 7, 0])
        dpnp_a = dpnp.array([1, 2, 4, 7, 0])

        if isinstance(to_begin, dpnp.ndarray):
            np_to_begin = dpnp.asnumpy(to_begin)
        else:
            np_to_begin = to_begin

        result = dpnp.ediff1d(dpnp_a, to_begin=to_begin)
        expected = numpy.ediff1d(np_a, to_begin=np_to_begin)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "to_end",
        [
            20,
            numpy.array([20, 15]),
            dpnp.array([20, 15]),
            dpnp.array([[-20], [-30]]),
            [3, 4],
            (3, 4),
        ],
    )
    def test_ediff1d_to_end(self, to_end):
        np_a = numpy.array([1, 2, 4, 7, 0])
        dpnp_a = dpnp.array([1, 2, 4, 7, 0])

        if isinstance(to_end, dpnp.ndarray):
            np_to_end = dpnp.asnumpy(to_end)
        else:
            np_to_end = to_end

        result = dpnp.ediff1d(dpnp_a, to_end=to_end)
        expected = numpy.ediff1d(np_a, to_end=np_to_end)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "to_begin, to_end",
        [
            (-20, 20),
            (numpy.array([-20, -30]), numpy.array([20, 15])),
            (dpnp.array([-20, -30]), dpnp.array([20, 15])),
            (dpnp.array([[-20], [-30]]), dpnp.array([[20], [15]])),
            ([1, 2], [3, 4]),
            ((1, 2), (3, 4)),
        ],
    )
    def test_ediff1d_to_begin_to_end(self, to_begin, to_end):
        np_a = numpy.array([1, 2, 4, 7, 0])
        dpnp_a = dpnp.array([1, 2, 4, 7, 0])

        if isinstance(to_begin, dpnp.ndarray):
            np_to_begin = dpnp.asnumpy(to_begin)
        else:
            np_to_begin = to_begin

        if isinstance(to_end, dpnp.ndarray):
            np_to_end = dpnp.asnumpy(to_end)
        else:
            np_to_end = to_end

        result = dpnp.ediff1d(dpnp_a, to_end=to_end, to_begin=to_begin)
        expected = numpy.ediff1d(np_a, to_end=np_to_end, to_begin=np_to_begin)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "to_begin, to_end",
        [
            (-20, 20),
            (dpt.asarray([-20, -30]), dpt.asarray([20, 15])),
            (dpt.asarray([[-20, -30]]), dpt.asarray([[20, 15]])),
            ([1, 2], [3, 4]),
            ((1, 2), (3, 4)),
        ],
    )
    def test_ediff1d_usm_ndarray(self, to_begin, to_end):
        np_a = numpy.array([[1, 2, 0]])
        dpt_a = dpt.asarray(np_a)

        if isinstance(to_begin, dpt.usm_ndarray):
            np_to_begin = dpt.asnumpy(to_begin)
        else:
            np_to_begin = to_begin

        if isinstance(to_end, dpt.usm_ndarray):
            np_to_end = dpt.asnumpy(to_end)
        else:
            np_to_end = to_end

        result = dpnp.ediff1d(dpt_a, to_end=to_end, to_begin=to_begin)
        expected = numpy.ediff1d(np_a, to_end=np_to_end, to_begin=np_to_begin)

        assert_array_equal(expected, result)
        assert isinstance(result, dpnp.ndarray)

    def test_ediff1d_errors(self):
        a_dp = dpnp.array([[1, 2], [2, 5]])

        # unsupported type
        a_np = dpnp.asnumpy(a_dp)
        assert_raises(TypeError, dpnp.ediff1d, a_np)

        # unsupported `to_begin` type according to the `same_kind` rules
        to_begin = dpnp.array([-5], dtype="f4")
        assert_raises(TypeError, dpnp.ediff1d, a_dp, to_begin=to_begin)

        # unsupported `to_end` type according to the `same_kind` rules
        to_end = dpnp.array([5], dtype="f4")
        assert_raises(TypeError, dpnp.ediff1d, a_dp, to_end=to_end)

        # another `to_begin` sycl queue
        to_begin = dpnp.array([-20, -15], sycl_queue=dpctl.SyclQueue())
        assert_raises(
            ExecutionPlacementError, dpnp.ediff1d, a_dp, to_begin=to_begin
        )

        # another `to_end` sycl queue
        to_end = dpnp.array([15, 20], sycl_queue=dpctl.SyclQueue())
        assert_raises(
            ExecutionPlacementError, dpnp.ediff1d, a_dp, to_end=to_end
        )


class TestFix:
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_basic(self, dt):
        a = get_abs_array([[1.0, 1.1, 1.5, 1.8], [-1.0, -1.1, -1.5, -1.8]], dt)
        ia = dpnp.array(a)

        result = dpnp.fix(ia)
        expected = numpy.fix(a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_complex(self, xp, dt):
        a = xp.array([1.1, -1.1], dtype=dt)
        with pytest.raises((ValueError, TypeError)):
            xp.fix(a)

    @pytest.mark.parametrize(
        "a_dt", get_all_dtypes(no_none=True, no_bool=True, no_complex=True)
    )
    def test_out(self, a_dt):
        a = get_abs_array(
            [[1.0, 1.1, 1.5, 1.8], [-1.0, -1.1, -1.5, -1.8]], a_dt
        )
        ia = dpnp.array(a)

        out_dt = _get_output_data_type(a.dtype)
        out = numpy.zeros_like(a, dtype=out_dt)
        iout = dpnp.array(out)

        result = dpnp.fix(ia, out=iout)
        expected = numpy.fix(a, out=out)
        assert_array_equal(result, expected)

    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
    @pytest.mark.parametrize("dt", [bool, numpy.float16])
    def test_out_float16(self, dt):
        a = numpy.array(
            [[1.0, 1.1], [1.5, 1.8], [-1.0, -1.1], [-1.5, -1.8]], dtype=dt
        )
        out = numpy.zeros_like(a, dtype=numpy.float16)
        ia, iout = dpnp.array(a), dpnp.array(out)

        result = dpnp.fix(ia, out=iout)
        expected = numpy.fix(a, out=out)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", [bool] + get_integer_dtypes())
    def test_out_invalid_dtype(self, xp, dt):
        a = xp.array([[1.5, 1.8], [-1.0, -1.1]])
        out = xp.zeros_like(a, dtype=dt)

        with pytest.raises((ValueError, TypeError)):
            xp.fix(a, out=out)

    def test_scalar(self):
        assert_raises(TypeError, dpnp.fix, -3.4)


class TestGradient:
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True, no_bool=True))
    def test_basic(self, dt):
        x = numpy.array([[1, 1], [3, 4]], dtype=dt)
        ix = dpnp.array(x)

        expected = numpy.gradient(x)
        result = dpnp.gradient(ix)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "args",
        [3.0, numpy.array(3.0), numpy.cumsum(numpy.ones(5))],
        ids=["scalar", "array", "cumsum"],
    )
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True, no_bool=True))
    def test_args_1d(self, args, dt):
        x = numpy.arange(5, dtype=dt)
        ix = dpnp.array(x)

        if numpy.isscalar(args):
            iargs = args
        else:
            iargs = dpnp.array(args)

        expected = numpy.gradient(x, args)
        result = dpnp.gradient(ix, iargs)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "args", [1.5, numpy.array(1.5)], ids=["scalar", "array"]
    )
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True, no_bool=True))
    def test_args_2d(self, args, dt):
        x = numpy.arange(25, dtype=dt).reshape(5, 5)
        ix = dpnp.array(x)

        if numpy.isscalar(args):
            iargs = args
        else:
            iargs = dpnp.array(args)

        expected = numpy.gradient(x, args)
        result = dpnp.gradient(ix, iargs)
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True, no_bool=True))
    def test_args_2d_uneven(self, dt):
        x = numpy.arange(25, dtype=dt).reshape(5, 5)
        ix = dpnp.array(x)

        dx = numpy.array([1.0, 2.0, 5.0, 9.0, 11.0])
        idx = dpnp.array(dx)

        expected = numpy.gradient(x, dx, dx)
        result = dpnp.gradient(ix, idx, idx)
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True, no_bool=True))
    def test_args_2d_mix_with_scalar(self, dt):
        x = numpy.arange(25, dtype=dt).reshape(5, 5)
        ix = dpnp.array(x)

        dx = numpy.cumsum(numpy.ones(5))
        idx = dpnp.array(dx)

        expected = numpy.gradient(x, dx, 2)
        result = dpnp.gradient(ix, idx, 2)
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True, no_bool=True))
    def test_axis_args_2d(self, dt):
        x = numpy.arange(25, dtype=dt).reshape(5, 5)
        ix = dpnp.array(x)

        dx = numpy.cumsum(numpy.ones(5))
        idx = dpnp.array(dx)

        expected = numpy.gradient(x, dx, axis=1)
        result = dpnp.gradient(ix, idx, axis=1)
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_args_2d_error(self, xp):
        x = xp.arange(25).reshape(5, 5)
        dx = xp.cumsum(xp.ones(5))
        assert_raises_regex(
            ValueError,
            ".*scalars or 1d",
            xp.gradient,
            x,
            xp.stack([dx] * 2, axis=-1),
            1,
        )

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_badargs(self, xp):
        x = xp.arange(25).reshape(5, 5)
        dx = xp.cumsum(xp.ones(5))

        # wrong sizes
        assert_raises(ValueError, xp.gradient, x, x, xp.ones(2))
        assert_raises(ValueError, xp.gradient, x, 1, xp.ones(2))
        assert_raises(ValueError, xp.gradient, x, xp.ones(2), xp.ones(2))
        # wrong number of arguments
        assert_raises(TypeError, xp.gradient, x, x)
        assert_raises(TypeError, xp.gradient, x, dx, axis=(0, 1))
        assert_raises(TypeError, xp.gradient, x, dx, dx, dx)
        assert_raises(TypeError, xp.gradient, x, 1, 1, 1)
        assert_raises(TypeError, xp.gradient, x, dx, dx, axis=1)
        assert_raises(TypeError, xp.gradient, x, 1, 1, axis=1)

    @pytest.mark.parametrize(
        "x",
        [
            numpy.linspace(0, 1, 10),
            numpy.sort(numpy.random.RandomState(0).random(10)),
        ],
        ids=["linspace", "random_sorted"],
    )
    @pytest.mark.parametrize("dt", get_float_dtypes())
    # testing that the relative numerical error is close to numpy
    def test_second_order_accurate(self, x, dt):
        x = x.astype(dt)
        dx = x[1] - x[0]
        y = 2 * x**3 + 4 * x**2 + 2 * x

        iy = dpnp.array(y)
        idx = dpnp.array(dx)

        expected = numpy.gradient(y, dx, edge_order=2)
        result = dpnp.gradient(iy, idx, edge_order=2)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("edge_order", [1, 2])
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_spacing_axis_scalar(self, edge_order, axis, dt):
        x = numpy.array([0, 2.0, 3.0, 4.0, 5.0, 5.0], dtype=dt)
        x = numpy.tile(x, (6, 1)) + x.reshape(-1, 1)
        ix = dpnp.array(x)

        expected = numpy.gradient(x, 1.0, axis=axis, edge_order=edge_order)
        result = dpnp.gradient(ix, 1.0, axis=axis, edge_order=edge_order)
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    @pytest.mark.parametrize("edge_order", [1, 2])
    @pytest.mark.parametrize("axis", [(0, 1), None])
    @pytest.mark.parametrize("dt", get_float_dtypes())
    @pytest.mark.parametrize(
        "dx",
        [numpy.arange(6.0), numpy.array([0.0, 0.5, 1.0, 3.0, 5.0, 7.0])],
        ids=["even", "uneven"],
    )
    def test_spacing_axis_two_args(self, edge_order, axis, dt, dx):
        x = numpy.array([0, 2.0, 3.0, 4.0, 5.0, 5.0], dtype=dt)
        x = numpy.tile(x, (6, 1)) + x.reshape(-1, 1)

        ix = dpnp.array(x)
        idx = dpnp.array(dx)

        expected = numpy.gradient(x, dx, dx, axis=axis, edge_order=edge_order)
        result = dpnp.gradient(ix, idx, idx, axis=axis, edge_order=edge_order)
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    @pytest.mark.parametrize("edge_order", [1, 2])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("dt", get_float_dtypes())
    @pytest.mark.parametrize(
        "dx",
        [numpy.arange(6.0), numpy.array([0.0, 0.5, 1.0, 3.0, 5.0, 7.0])],
        ids=["even", "uneven"],
    )
    def test_spacing_axis_args(self, edge_order, axis, dt, dx):
        x = numpy.array([0, 2.0, 3.0, 4.0, 5.0, 5.0], dtype=dt)
        x = numpy.tile(x, (6, 1)) + x.reshape(-1, 1)

        ix = dpnp.array(x)
        idx = dpnp.array(dx)

        expected = numpy.gradient(x, dx, axis=axis, edge_order=edge_order)
        result = dpnp.gradient(ix, idx, axis=axis, edge_order=edge_order)
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    @pytest.mark.parametrize("edge_order", [1, 2])
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_spacing_mix_args(self, edge_order, dt):
        x = numpy.array([0, 2.0, 3.0, 4.0, 5.0, 5.0], dtype=dt)
        x = numpy.tile(x, (6, 1)) + x.reshape(-1, 1)
        x_uneven = numpy.array([0.0, 0.5, 1.0, 3.0, 5.0, 7.0])
        x_even = numpy.arange(6.0)

        ix = dpnp.array(x)
        ix_uneven = dpnp.array(x_uneven)
        ix_even = dpnp.array(x_even)

        expected = numpy.gradient(
            x, x_even, x_uneven, axis=(0, 1), edge_order=edge_order
        )
        result = dpnp.gradient(
            ix, ix_even, ix_uneven, axis=(0, 1), edge_order=edge_order
        )
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

        expected = numpy.gradient(
            x, x_uneven, x_even, axis=(1, 0), edge_order=edge_order
        )
        result = dpnp.gradient(
            ix, ix_uneven, ix_even, axis=(1, 0), edge_order=edge_order
        )
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    @pytest.mark.parametrize("axis", [0, 1, -1, (1, 0), None])
    def test_specific_axes(self, axis):
        x = numpy.array([[1, 1], [3, 4]])
        ix = dpnp.array(x)

        expected = numpy.gradient(x, axis=axis)
        result = dpnp.gradient(ix, axis=axis)
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    def test_axis_scalar_args(self):
        x = numpy.array([[1, 1], [3, 4]])
        ix = dpnp.array(x)

        expected = numpy.gradient(x, 2, 3, axis=(1, 0))
        result = dpnp.gradient(ix, 2, 3, axis=(1, 0))
        for gr, igr in zip(expected, result):
            assert_dtype_allclose(igr, gr)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_wrong_number_of_args(self, xp):
        x = xp.array([[1, 1], [3, 4]])
        assert_raises(TypeError, xp.gradient, x, 1, 2, axis=1)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_wrong_axis(self, xp):
        x = xp.array([[1, 1], [3, 4]])
        assert_raises(AxisError, xp.gradient, x, axis=3)

    @pytest.mark.parametrize(
        "size, edge_order",
        [
            pytest.param(2, 1),
            pytest.param(3, 2),
        ],
    )
    def test_min_size_with_edge_order(self, size, edge_order):
        x = numpy.arange(size)
        ix = dpnp.array(x)

        expected = numpy.gradient(x, edge_order=edge_order)
        result = dpnp.gradient(ix, edge_order=edge_order)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "size, edge_order",
        [
            pytest.param(0, 1),
            pytest.param(0, 2),
            pytest.param(1, 1),
            pytest.param(1, 2),
            pytest.param(2, 2),
        ],
    )
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_wrong_size_with_edge_order(self, size, edge_order, xp):
        assert_raises(
            ValueError, xp.gradient, xp.arange(size), edge_order=edge_order
        )

    @pytest.mark.parametrize(
        "dt", [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]
    )
    def test_f_decreasing_unsigned_int(self, dt):
        x = numpy.array([5, 4, 3, 2, 1], dtype=dt)
        ix = dpnp.array(x)

        expected = numpy.gradient(x)
        result = dpnp.gradient(ix)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dt", [numpy.int8, numpy.int16, numpy.int32, numpy.int64]
    )
    def test_f_signed_int_big_jump(self, dt):
        maxint = numpy.iinfo(dt).max
        x = numpy.array([-1, maxint], dtype=dt)
        dx = numpy.array([1, 3])

        ix = dpnp.array(x)
        idx = dpnp.array(dx)

        expected = numpy.gradient(x, dx)
        result = dpnp.gradient(ix, idx)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dt", [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]
    )
    def test_x_decreasing_unsigned(self, dt):
        x = numpy.array([3, 2, 1], dtype=dt)
        f = numpy.array([0, 2, 4])

        dp_x = dpnp.array(x)
        dp_f = dpnp.array(f)

        expected = numpy.gradient(f, x)
        result = dpnp.gradient(dp_f, dp_x)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dt", [numpy.int8, numpy.int16, numpy.int32, numpy.int64]
    )
    def test_x_signed_int_big_jump(self, dt):
        minint = numpy.iinfo(dt).min
        maxint = numpy.iinfo(dt).max
        x = numpy.array([-1, maxint], dtype=dt)
        f = numpy.array([minint // 2, 0])

        dp_x = dpnp.array(x)
        dp_f = dpnp.array(f)

        expected = numpy.gradient(f, x)
        result = dpnp.gradient(dp_f, dp_x)
        assert_array_equal(result, expected)

    def test_return_type(self):
        x = dpnp.array([[1, 2], [2, 3]])
        res = dpnp.gradient(x)
        assert type(res) is tuple


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


class TestI0:
    def test_0d(self):
        a = dpnp.array(0.5)
        na = a.asnumpy()
        assert_dtype_allclose(dpnp.i0(a), numpy.i0(na))

    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_1d(self, dt):
        a = numpy.array(
            [0.49842636, 0.6969809, 0.22011976, 0.0155549, 10.0], dtype=dt
        )
        ia = dpnp.array(a)

        result = dpnp.i0(ia)
        expected = numpy.i0(a)
        # NumPy promotes result of integer inputs to float64, but dpnp
        # follows Type Promotion Rules
        flag = dt in [numpy.int8, numpy.int16, numpy.uint8, numpy.uint16]
        assert_dtype_allclose(result, expected, check_only_type_kind=flag)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_2d(self, dt):
        a = numpy.array(
            [
                [0.827002, 0.99959078],
                [0.89694769, 0.39298162],
                [0.37954418, 0.05206293],
                [0.36465447, 0.72446427],
                [0.48164949, 0.50324519],
            ],
            dtype=dt,
        )
        ia = dpnp.array(a)

        result = dpnp.i0(ia)
        expected = numpy.i0(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_nan(self):
        a = numpy.array(numpy.nan)
        ia = dpnp.array(a)

        result = dpnp.i0(ia)
        expected = numpy.i0(a)
        assert_equal(result, expected)

    # numpy.i0(numpy.inf) returns NaN, but expected Inf
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_infs(self, dt):
        a = dpnp.array([dpnp.inf, -dpnp.inf], dtype=dt)
        assert (dpnp.i0(a) == dpnp.inf).all()

    # dpnp.i0 returns float16, but numpy.i0 returns float64
    def test_bool(self):
        a = numpy.array([False, True, False])
        ia = dpnp.array(a)

        result = dpnp.i0(ia)
        expected = numpy.i0(a)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_complex(self, xp):
        a = xp.array([0, 1 + 2j])
        assert_raises((ValueError, TypeError), xp.i0, a)


@pytest.mark.parametrize(
    "rhs", [[[1, 2, 3], [4, 5, 6]], [2.0, 1.5, 1.0], 3, 0.3]
)
@pytest.mark.parametrize("lhs", [[[6, 5, 4], [3, 2, 1]], [1.3, 2.6, 3.9]])
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

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_fmod(self, dtype, lhs, rhs):
        if rhs == 0.3 and not has_support_aspect64():
            """
            Due to accuracy reason, the results are different for `float32` and `float64`
                >>> numpy.fmod(numpy.array([3.9], dtype=numpy.float32), 0.3)
                array([0.29999995], dtype=float32)

                >>> numpy.fmod(numpy.array([3.9], dtype=numpy.float64), 0.3)
                array([9.53674318e-08])
            On a gpu without fp64 support, dpnp produces results similar to the second one.
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

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_subtract(self, dtype, lhs, rhs):
        self._test_mathematical("subtract", dtype, lhs, rhs, check_type=False)


class TestNanToNum:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("shape", [(3,), (2, 3), (3, 2, 2)])
    def test_nan_to_num(self, dtype, shape):
        a = numpy.random.randn(*shape).astype(dtype)
        if not dpnp.issubdtype(dtype, dpnp.integer):
            a.flat[1] = numpy.nan
        a_dp = dpnp.array(a)

        result = dpnp.nan_to_num(a_dp)
        expected = numpy.nan_to_num(a)
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "data", [[], [numpy.nan], [numpy.inf], [-numpy.inf]]
    )
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_empty_and_single_value_arrays(self, data, dtype):
        a = numpy.array(data, dtype)
        ia = dpnp.array(a)

        result = dpnp.nan_to_num(ia)
        expected = numpy.nan_to_num(a)
        assert_allclose(result, expected)

    def test_boolean_array(self):
        a = numpy.array([True, False, numpy.nan], dtype=bool)
        ia = dpnp.array(a)

        result = dpnp.nan_to_num(ia)
        expected = numpy.nan_to_num(a)
        assert_allclose(result, expected)

    def test_errors(self):
        ia = dpnp.array([0, 1, dpnp.nan, dpnp.inf, -dpnp.inf])

        # unsupported type `a`
        a_np = dpnp.asnumpy(ia)
        assert_raises(TypeError, dpnp.nan_to_num, a_np)

        # unsupported type `nan`
        i_nan = dpnp.array(1)
        assert_raises(TypeError, dpnp.nan_to_num, ia, nan=i_nan)

        # unsupported type `posinf`
        i_posinf = dpnp.array(1)
        assert_raises(TypeError, dpnp.nan_to_num, ia, posinf=i_posinf)

        # unsupported type `neginf`
        i_neginf = dpnp.array(1)
        assert_raises(TypeError, dpnp.nan_to_num, ia, neginf=i_neginf)

    @pytest.mark.parametrize("kwarg", ["nan", "posinf", "neginf"])
    @pytest.mark.parametrize("value", [1 - 0j, [1, 2], (1,)])
    def test_errors_diff_types(self, kwarg, value):
        ia = dpnp.array([0, 1, dpnp.nan, dpnp.inf, -dpnp.inf])
        with pytest.raises(TypeError):
            dpnp.nan_to_num(ia, **{kwarg: value})

    def test_error_readonly(self):
        a = dpnp.array([0, 1, dpnp.nan, dpnp.inf, -dpnp.inf])
        a.flags.writable = False
        with pytest.raises(ValueError):
            dpnp.nan_to_num(a, copy=False)

    @pytest.mark.parametrize("copy", [True, False])
    @pytest.mark.parametrize("dt", get_all_dtypes(no_bool=True, no_none=True))
    def test_nan_to_num_strided(self, copy, dt):
        n = 10
        dt = numpy.dtype(dt)
        np_a = numpy.arange(2 * n, dtype=dt)
        dp_a = dpnp.arange(2 * n, dtype=dt)
        if dt.kind in "fc":
            np_a[::4] = numpy.nan
            dp_a[::4] = dpnp.nan
        dp_r = dpnp.nan_to_num(dp_a[::-2], copy=copy, nan=57.0)
        np_r = numpy.nan_to_num(np_a[::-2], copy=copy, nan=57.0)

        assert_dtype_allclose(dp_r, np_r)


class TestProd:
    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
    @pytest.mark.parametrize("keepdims", [False, True])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_prod(self, axis, keepdims, dtype):
        a = numpy.arange(1, 13, dtype=dtype).reshape((2, 2, 3))
        ia = dpnp.array(a)

        np_res = numpy.prod(a, axis=axis, keepdims=keepdims)
        dpnp_res = dpnp.prod(ia, axis=axis, keepdims=keepdims)

        assert dpnp_res.shape == np_res.shape
        assert_allclose(dpnp_res, np_res)

    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
    def test_prod_zero_size(self, axis):
        a = numpy.empty((2, 3, 0))
        ia = dpnp.array(a)

        np_res = numpy.prod(a, axis=axis)
        dpnp_res = dpnp.prod(ia, axis=axis)
        assert_dtype_allclose(dpnp_res, np_res)

    @pytest.mark.parametrize("axis", [None, 0, 1, -1])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_prod_bool(self, axis, keepdims):
        a = numpy.arange(2, dtype=numpy.bool_)
        a = numpy.tile(a, (2, 2))
        ia = dpnp.array(a)

        np_res = numpy.prod(a, axis=axis, keepdims=keepdims)
        dpnp_res = dpnp.prod(ia, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(dpnp_res, np_res)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize("in_dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "out_dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_prod_dtype(self, in_dtype, out_dtype):
        a = numpy.arange(1, 13, dtype=in_dtype).reshape((2, 2, 3))
        ia = dpnp.array(a)

        np_res = numpy.prod(a, dtype=out_dtype)
        dpnp_res = dpnp.prod(ia, dtype=out_dtype)
        assert_dtype_allclose(dpnp_res, np_res)

    @pytest.mark.usefixtures(
        "suppress_overflow_encountered_in_cast_numpy_warnings"
    )
    def test_prod_out(self):
        ia = dpnp.arange(1, 7).reshape((2, 3))
        ia = ia.astype(dpnp.default_float_type(ia.device))
        a = dpnp.asnumpy(ia)

        # output is dpnp_array
        np_res = numpy.prod(a, axis=0)
        dpnp_out = dpnp.empty(np_res.shape, dtype=np_res.dtype)
        dpnp_res = dpnp.prod(ia, axis=0, out=dpnp_out)
        assert dpnp_out is dpnp_res
        assert_allclose(dpnp_res, np_res)

        # output is usm_ndarray
        dpt_out = dpt.empty(np_res.shape, dtype=np_res.dtype)
        dpnp_res = dpnp.prod(ia, axis=0, out=dpt_out)
        assert dpt_out is dpnp_res.get_array()
        assert_allclose(dpnp_res, np_res)

        # out is a numpy array -> TypeError
        dpnp_res = numpy.empty_like(np_res)
        with pytest.raises(TypeError):
            dpnp.prod(ia, axis=0, out=dpnp_res)

        # incorrect shape for out
        dpnp_res = dpnp.array(numpy.empty((2, 3)))
        with pytest.raises(ValueError):
            dpnp.prod(ia, axis=0, out=dpnp_res)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("arr_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_prod_out_dtype(self, arr_dt, out_dt, dtype):
        a = numpy.arange(1, 7).reshape((2, 3)).astype(dtype=arr_dt)
        out = numpy.zeros_like(a, shape=(2,), dtype=out_dt)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = dpnp.prod(ia, out=iout, dtype=dtype, axis=1)
        expected = numpy.prod(a, out=out, dtype=dtype, axis=1)
        assert_array_equal(expected, result)
        assert result is iout

    def test_prod_Error(self):
        ia = dpnp.arange(5)

        with pytest.raises(TypeError):
            dpnp.prod(dpnp.asnumpy(ia))
        with pytest.raises(NotImplementedError):
            dpnp.prod(ia, where=False)
        with pytest.raises(NotImplementedError):
            dpnp.prod(ia, initial=6)


class TestRealIfClose:
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_basic(self, dt):
        a = numpy.random.rand(10).astype(dt)
        ia = dpnp.array(a)

        result = dpnp.real_if_close(ia + 1e-15j)
        expected = numpy.real_if_close(a + 1e-15j)
        assert_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_singlecomplex(self, dt):
        a = numpy.random.rand(10).astype(dt)
        ia = dpnp.array(a)

        result = dpnp.real_if_close(ia + 1e-7j)
        expected = numpy.real_if_close(a + 1e-7j)
        assert_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_tol(self, dt):
        a = numpy.random.rand(10).astype(dt)
        ia = dpnp.array(a)

        result = dpnp.real_if_close(ia + 1e-7j, tol=1e-6)
        expected = numpy.real_if_close(a + 1e-7j, tol=1e-6)
        assert_equal(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("tol_val", [[10], (1, 2)], ids=["list", "tuple"])
    def test_wrong_tol_type(self, xp, tol_val):
        a = xp.array([2.1 + 4e-14j, 5.2 + 3e-15j])
        assert_raises(TypeError, xp.real_if_close, a, tol=tol_val)


class TestSinc:
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_bool=True, no_float16=False)
    )
    def test_basic(self, dt):
        low = 0 if dpnp.issubdtype(dt, dpnp.integer) else -1
        a = numpy.linspace(-1, 1, 100, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.sinc(ia)
        expected = numpy.sinc(a)
        # numpy promotes result for integer inputs to float64 dtype, but dpnp
        # follows Type Promotion Rules similar to other trigonometric functions
        flag = dt in [numpy.int8, numpy.int16, numpy.uint8, numpy.uint16]
        assert_dtype_allclose(result, expected, check_only_type_kind=flag)

    def test_bool(self):
        a = numpy.array([True, False, True])
        ia = dpnp.array(a)

        result = dpnp.sinc(ia)
        expected = numpy.sinc(a)
        # numpy promotes result to float64 dtype, but expected float16
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True, no_bool=True))
    def test_zero(self, dt):
        a = numpy.array([0.0], dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.sinc(ia)
        expected = numpy.sinc(a)
        # numpy promotes result for integer inputs to float64 dtype, but dpnp
        # follows Type Promotion Rules similar to other trigonometric functions
        flag = dt in [numpy.int8, numpy.int16, numpy.uint8, numpy.uint16]
        assert_dtype_allclose(result, expected, check_only_type_kind=flag)

    # TODO: add a proper NumPy version once resolved
    @testing.with_requires("numpy>=2.0.0")
    def test_zero_fp16(self):
        a = numpy.array([0.0], dtype=numpy.float16)
        ia = dpnp.array(a)

        result = dpnp.sinc(ia)
        # expected = numpy.sinc(a) # numpy returns NaN, but expected 1.0
        expected = numpy.ones_like(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    def test_nan_infs(self):
        a = numpy.array([numpy.inf, -numpy.inf, numpy.nan])
        ia = dpnp.array(a)

        result = dpnp.sinc(ia)
        expected = numpy.sinc(a)
        assert_equal(result, expected)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    def test_nan_infs_complex(self):
        a = numpy.array(
            [
                numpy.inf,
                -numpy.inf,
                numpy.nan,
                complex(numpy.nan),
                complex(numpy.nan, numpy.nan),
                complex(0, numpy.nan),
                complex(numpy.inf, numpy.nan),
                complex(numpy.nan, numpy.inf),
                complex(-numpy.inf, numpy.nan),
                complex(numpy.nan, -numpy.inf),
                complex(numpy.inf, numpy.inf),
                complex(numpy.inf, -numpy.inf),
                complex(-numpy.inf, numpy.inf),
                complex(-numpy.inf, -numpy.inf),
            ]
        )
        ia = dpnp.array(a)

        result = dpnp.sinc(ia)
        expected = numpy.sinc(a)
        assert_equal(result, expected)


class TestSpacing:
    @pytest.mark.parametrize("sign", [1, -1])
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_basic(self, sign, dt):
        a = numpy.array(
            [1, numpy.nan, numpy.inf, 1e10, 1e-5, 1000, 10500], dtype=dt
        )
        a *= sign
        ia = dpnp.array(a)

        result = dpnp.spacing(ia)
        expected = numpy.spacing(a)
        assert_equal(result, expected)

        # switch to negatives
        result = dpnp.spacing(-ia)
        expected = numpy.spacing(-a)
        assert_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_zeros(self, dt):
        a = numpy.array([0.0, -0.0], dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.spacing(ia)
        expected = numpy.spacing(a)
        tol = numpy.finfo(expected.dtype).resolution
        if numpy_version() < "2.0.0":
            assert_allclose(result, expected, rtol=tol, atol=tol)
        else:
            # numpy.spacing(-0.0) == numpy.spacing(0.0), i.e. NumPy returns
            # positive value (looks as a bug in NumPy), because for any other
            # negative input the NumPy result will be also a negative value.
            expected[1] *= -1
            assert_allclose(result, expected, rtol=tol, atol=tol)

    @pytest.mark.parametrize("dt", get_float_dtypes(no_float16=False))
    @pytest.mark.parametrize("val", [1, 1e-5, 1000])
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_vs_nextafter(self, val, dt, xp):
        a = xp.array(val, dtype=dt)
        a1 = xp.array(val + 1, dtype=dt)
        assert (xp.nextafter(a, a1) - a) == xp.spacing(a)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
    def test_fp16(self):
        a = numpy.arange(0x7C00, dtype=numpy.uint16)

        # all values are non-negative finites
        b = a.view(dtype=numpy.float16)
        ib = dpnp.array(b)

        result = dpnp.spacing(ib)
        expected = numpy.spacing(b)
        assert_equal(result, expected)

        # switch to negatives
        a |= 0x8000
        ib = dpnp.array(b)

        result = dpnp.spacing(ib)
        expected = numpy.spacing(b)
        assert_equal(result, expected)

    @pytest.mark.parametrize("dt", get_integer_dtypes())
    def test_integer(self, dt):
        a = get_abs_array([1, 0, -3], dt)
        ia = dpnp.array(a)

        result = dpnp.spacing(ia)
        expected = numpy.spacing(a)
        assert_dtype_allclose(result, expected)

    def test_bool(self):
        a = numpy.array([True, False])
        ia = dpnp.array(a)

        result = dpnp.spacing(ia)
        expected = numpy.spacing(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_complex(self, xp):
        a = xp.array([2.1 + 4e-14j, 5.2 + 3e-15j])
        assert_raises((TypeError, ValueError), xp.spacing, a)


class TestTrapezoid:
    def get_numpy_func(self):
        if numpy_version() < "2.0.0":
            # `trapz` is deprecated in NumPy 2.0
            return numpy.trapz
        return numpy.trapezoid

    @pytest.mark.parametrize("dt", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "data",
        [[1, 2, 3], [[1, 2, 3], [4, 5, 6]], [1, 4, 6, 9, 10, 12], [], [1]],
    )
    def test_basic(self, data, dt):
        a = numpy.array(data, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.trapezoid(ia)
        expected = self.get_numpy_func()(a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_simple(self, dt):
        x = numpy.arange(-10, 10, 0.1, dtype=dt)
        ix = dpnp.array(x)

        # integral of normal equals 1
        sqrt_2pi = numpy.sqrt(2 * numpy.pi)
        result = dpnp.trapezoid(dpnp.exp(-0.5 * ix**2) / sqrt_2pi, dx=0.1)
        expected = self.get_numpy_func()(
            numpy.exp(-0.5 * x**2) / sqrt_2pi, dx=0.1
        )
        assert_almost_equal(result, expected, 7)

    @pytest.mark.parametrize("y_dt", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("x_dt", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("y_arr", [[1, 2, 4, 5], [1.0, 2.5, 6.0, 7.0]])
    @pytest.mark.parametrize("x_arr", [[2, 5, 6, 9]])
    def test_x_samples(self, y_arr, x_arr, y_dt, x_dt):
        y = numpy.array(y_arr, dtype=y_dt)
        x = numpy.array(x_arr, dtype=x_dt)
        iy, ix = dpnp.array(y), dpnp.array(x)

        result = dpnp.trapezoid(iy, ix)
        expected = self.get_numpy_func()(y, x)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("data", [[1, 2, 3], [4, 5, 6]])
    def test_2d_with_x_samples(self, data):
        a = numpy.array(data)
        ia = dpnp.array(a)

        result = dpnp.trapezoid(ia, ia)
        expected = self.get_numpy_func()(a, a)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "data",
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
    def test_dx_samples(self, data, dx):
        y = numpy.array(data)
        iy = dpnp.array(y)

        result = dpnp.trapezoid(iy, dx=dx)
        expected = self.get_numpy_func()(y, dx=dx)
        assert_array_equal(expected, result)

    def test_ndim(self):
        x = numpy.linspace(0, 1, 3)
        y = numpy.linspace(0, 2, 8)
        z = numpy.linspace(0, 3, 13)
        ix, iy, iz = dpnp.array(x), dpnp.array(y), dpnp.array(z)

        q = x[:, None, None] + y[None, :, None] + z[None, None, :]
        iq = ix[:, None, None] + iy[None, :, None] + iz[None, None, :]

        # n-d `x`
        result = dpnp.trapezoid(iq, x=ix[:, None, None], axis=0)
        expected = self.get_numpy_func()(q, x=x[:, None, None], axis=0)
        assert_dtype_allclose(result, expected)

        result = dpnp.trapezoid(iq, x=iy[None, :, None], axis=1)
        expected = self.get_numpy_func()(q, x=y[None, :, None], axis=1)
        assert_dtype_allclose(result, expected)

        result = dpnp.trapezoid(iq, x=iz[None, None, :], axis=2)
        expected = self.get_numpy_func()(q, x=z[None, None, :], axis=2)
        assert_dtype_allclose(result, expected)

        # 1-d `x`
        result = dpnp.trapezoid(iq, x=ix, axis=0)
        expected = self.get_numpy_func()(q, x=x, axis=0)
        assert_dtype_allclose(result, expected)

        result = dpnp.trapezoid(iq, x=iy, axis=1)
        expected = self.get_numpy_func()(q, x=y, axis=1)
        assert_dtype_allclose(result, expected)

        result = dpnp.trapezoid(iq, x=iz, axis=2)
        expected = self.get_numpy_func()(q, x=z, axis=2)
        assert_dtype_allclose(result, expected)


class TestUnwrap:
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_basic(self, dt):
        a = numpy.array([1, 1 + 2 * numpy.pi], dtype=dt)
        ia = dpnp.array(a)

        # unwrap removes jumps greater than 2*pi
        result = dpnp.unwrap(ia)
        expected = numpy.unwrap(a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_rand(self, dt):
        a = generate_random_numpy_array(10) * 100
        a = a.astype(dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.unwrap(ia)
        expected = numpy.unwrap(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dt",
        get_all_dtypes(
            no_none=True, no_bool=True, no_complex=True, no_unsigned=True
        ),
    )
    def test_period(self, dt):
        a = numpy.array([1, 1 + 108], dtype=dt)
        ia = dpnp.array(a)

        # unwrap removes jumps greater than 107
        result = dpnp.unwrap(ia, period=107)
        expected = numpy.unwrap(a, period=107)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dt",
        get_all_dtypes(
            no_none=True, no_bool=True, no_complex=True, no_unsigned=True
        ),
    )
    def test_rand_period(self, dt):
        a = generate_random_numpy_array(10, dt, low=-100, high=100)
        ia = dpnp.array(a)

        result = dpnp.unwrap(ia, period=25)
        expected = numpy.unwrap(a, period=25)
        assert_dtype_allclose(result, expected)

    def test_simple_seq(self):
        simple_seq = numpy.array([0, 75, 150, 225, 300])
        wrap_seq = numpy.mod(simple_seq, 255)
        isimple_seq, iwrap_seq = dpnp.array(simple_seq), dpnp.array(wrap_seq)

        result = dpnp.unwrap(iwrap_seq, period=255)
        expected = numpy.unwrap(wrap_seq, period=255)
        assert_array_equal(result, expected)
        assert_array_equal(result, isimple_seq)

    @pytest.mark.parametrize(
        "dt",
        get_all_dtypes(
            no_bool=True, no_none=True, no_complex=True, no_unsigned=True
        ),
    )
    def test_discont(self, dt):
        a = numpy.array([0, 8, 20, 25, 35, 50], dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.unwrap(ia, period=20)
        expected = numpy.unwrap(a, period=20)
        assert_array_equal(result, expected)

        result = dpnp.unwrap(ia, period=20, discont=14)
        expected = numpy.unwrap(a, period=20, discont=14)
        assert_array_equal(result, expected)
        assert result.dtype == ia.dtype == a.dtype


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


@pytest.mark.parametrize(
    "data",
    [[[1.0, -1.0], [0.1, -0.1]], [-2, -1, 0, 1, 2]],
    ids=["[[1., -1.], [0.1, -0.1]]", "[-2, -1, 0, 1, 2]"],
)
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_unsigned=True)
)
def test_negative(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(np_a)

    result = dpnp.negative(dpnp_a)
    expected = numpy.negative(np_a)
    assert_allclose(result, expected)

    result = -dpnp_a
    expected = -np_a
    assert_allclose(result, expected)

    # out keyword
    if dtype is not None:
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.negative(dpnp_a, out=dp_out)
        assert result is dp_out
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
    np_a = get_abs_array(data, dtype=dtype)
    dpnp_a = dpnp.array(np_a)

    result = dpnp.positive(dpnp_a)
    expected = numpy.positive(np_a)
    assert_allclose(result, expected)

    result = +dpnp_a
    expected = +np_a
    assert_allclose(result, expected)

    # out keyword
    if dtype is not None:
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.positive(dpnp_a, out=dp_out)
        assert result is dp_out
        assert_allclose(result, expected)


def test_positive_boolean():
    dpnp_a = dpnp.array([True, False])

    with pytest.raises(TypeError):
        dpnp.positive(dpnp_a)


@pytest.mark.parametrize("dtype", get_float_dtypes(no_float16=False))
def test_float_remainder_magnitude(dtype):
    b = numpy.array(1.0, dtype=dtype)
    a = numpy.nextafter(numpy.array(0.0, dtype=dtype), -b)

    ia = dpnp.array(a)
    ib = dpnp.array(b)

    result = dpnp.remainder(ia, ib)
    expected = numpy.remainder(a, b)
    assert_equal(result, expected)

    result = dpnp.remainder(-ia, -ib)
    expected = numpy.remainder(-a, -b)
    assert_equal(result, expected)


@pytest.mark.usefixtures("suppress_divide_numpy_warnings")
@pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
@pytest.mark.parametrize("func", ["remainder", "fmod"])
@pytest.mark.parametrize("dtype", get_float_dtypes(no_float16=False))
@pytest.mark.parametrize(
    "lhs, rhs",
    [
        pytest.param(1.0, 0.0, id="one-zero"),
        pytest.param(1.0, numpy.inf, id="one-inf"),
        pytest.param(numpy.inf, 1.0, id="inf-one"),
        pytest.param(numpy.inf, numpy.inf, id="inf-inf"),
        pytest.param(numpy.inf, 0.0, id="inf-zero"),
        pytest.param(1.0, numpy.nan, id="one-nan"),
        pytest.param(numpy.nan, 0.0, id="nan-zero"),
        pytest.param(numpy.nan, 1.0, id="nan-one"),
    ],
)
def test_float_remainder_fmod_nans_inf(func, dtype, lhs, rhs):
    a = numpy.array(lhs, dtype=dtype)
    b = numpy.array(rhs, dtype=dtype)

    ia = dpnp.array(a)
    ib = dpnp.array(b)

    result = getattr(dpnp, func)(ia, ib)
    expected = getattr(numpy, func)(a, b)
    assert_equal(result, expected)


@testing.with_requires("numpy>=2.0.0")
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_none=True, no_unsigned=True)
)
def test_sign(dtype):
    a = generate_random_numpy_array((2, 3), dtype=dtype)
    ia = dpnp.array(a, dtype=dtype)

    if dtype == dpnp.bool:
        assert_raises(TypeError, dpnp.sign, ia)
        assert_raises(TypeError, numpy.sign, a)
    else:
        result = dpnp.sign(ia)
        expected = numpy.sign(a)
        assert_dtype_allclose(result, expected)

        # out keyword
        iout = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = dpnp.sign(ia, out=iout)
        assert iout is result
        assert_dtype_allclose(result, expected)


@pytest.mark.parametrize(
    "data",
    [[2, 0, -2], [1.1, -1.1]],
    ids=["[2, 0, -2]", "[1.1, -1.1]"],
)
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_complex=True, no_unsigned=True)
)
def test_signbit(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

    result = dpnp.signbit(dpnp_a)
    expected = numpy.signbit(np_a)
    assert_dtype_allclose(result, expected)

    # out keyword
    dp_out = dpnp.empty(expected.shape, dtype=expected.dtype)
    result = dpnp.signbit(dpnp_a, out=dp_out)
    assert dp_out is result
    assert_dtype_allclose(result, expected)


class TestRealImag:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_real_imag(self, dtype):
        a = generate_random_numpy_array(20, dtype)
        ia = dpnp.array(a)

        result = dpnp.real(ia)
        expected = numpy.real(a)
        if not dpnp.issubdtype(dtype, dpnp.complexfloating):
            assert result is ia
            assert expected is a
        assert_dtype_allclose(result, expected)

        result = dpnp.imag(ia)
        expected = numpy.imag(a)
        assert_dtype_allclose(result, expected)

        # ndarray
        result = ia.real
        if not dpnp.issubdtype(dtype, dpnp.complexfloating):
            assert result is ia
        assert_dtype_allclose(result, a.real)
        assert_dtype_allclose(ia.imag, a.imag)


class TestProjection:
    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_projection_infinity(self, dtype):
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

        a = dpnp.array(X, dtype=dtype)
        result = dpnp.proj(a)
        expected = dpnp.array(Y, dtype=dtype)
        assert_dtype_allclose(result, expected)

        # out keyword
        dp_out = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = dpnp.proj(a, out=dp_out)
        assert dp_out is result
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_projection(self, dtype):
        result = dpnp.proj(dpnp.array(1, dtype=dtype))
        expected = dpnp.array(complex(1, 0))
        assert_allclose(result, expected)


class TestRoundingFuncs:
    @pytest.fixture(
        params=[
            {"func_name": "ceil", "input_values": [-5, 5, 10]},
            {"func_name": "floor", "input_values": [-5, 5, 10]},
            {"func_name": "trunc", "input_values": [-5, 5, 10]},
        ],
        ids=[
            "ceil",
            "floor",
            "trunc",
        ],
    )
    def func_params(self, request):
        return request.param

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_out(self, func_params, dtype):
        func_name = func_params["func_name"]
        input_values = func_params["input_values"]
        np_array, expected = _get_numpy_arrays_1in_1out(
            func_name, dtype, input_values
        )

        dp_array = dpnp.array(np_array)
        out_dtype = numpy.int8 if dtype == numpy.bool_ else dtype
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = getattr(dpnp, func_name)(dp_array, out=dp_out)

        assert result is dp_out
        # numpy.ceil, numpy.floor, numpy.trunc always return float dtype for
        # NumPy < 2.1.0 while output has the dtype of input for NumPy >= 2.1.0
        # (dpnp follows the latter behavior except for boolean dtype where it
        # returns int8)
        if numpy_version() < "2.1.0" or dtype == numpy.bool:
            check_type = False
        else:
            check_type = True
        assert_dtype_allclose(result, expected, check_type=check_type)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, func_params, dtype):
        func_name = func_params["func_name"]
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(ValueError):
            getattr(dpnp, func_name)(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, func_params, shape):
        func_name = func_params["func_name"]
        dp_array = dpnp.arange(10, dtype=dpnp.float32)
        dp_out = dpnp.empty(shape, dtype=dpnp.float32)

        with pytest.raises(ValueError):
            getattr(dpnp, func_name)(dp_array, out=dp_out)


class TestHypot:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_hypot(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "hypot", dtype, [0, 10, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.hypot(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_out_overlap(self, dtype):
        size = 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.hypot(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.hypot(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(10)
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


class TestLogSumExp:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axis", [None, 2, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_logsumexp(self, dtype, axis, keepdims):
        a = dpnp.ones((3, 4, 5, 6, 7), dtype=dtype)
        res = dpnp.logsumexp(a, axis=axis, keepdims=keepdims)

        exp_dt = None
        dtype_list = [dpnp.bool, dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        if dtype in dtype_list:
            exp_dt = dpnp.default_float_type(a.device)

        exp = numpy.logaddexp.reduce(
            dpnp.asnumpy(a), axis=axis, keepdims=keepdims, dtype=exp_dt
        )

        assert_dtype_allclose(res, exp)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axis", [None, 2, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_logsumexp_out(self, dtype, axis, keepdims):
        a = dpnp.ones((3, 4, 5, 6, 7), dtype=dtype)
        exp_dt = None
        dtype_list = [dpnp.bool, dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        if dtype in dtype_list:
            exp_dt = dpnp.default_float_type(a.device)
        exp = numpy.logaddexp.reduce(
            dpnp.asnumpy(a), axis=axis, keepdims=keepdims, dtype=exp_dt
        )

        exp_dt = exp.dtype
        if exp_dt == numpy.float64 and not has_support_aspect64():
            exp_dt = numpy.float32
        dpnp_out = dpnp.empty_like(a, shape=exp.shape, dtype=exp_dt)
        res = dpnp.logsumexp(a, axis=axis, out=dpnp_out, keepdims=keepdims)

        assert res is dpnp_out
        assert_dtype_allclose(res, exp)

    @pytest.mark.parametrize(
        "in_dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize("out_dtype", get_all_dtypes(no_bool=True))
    def test_logsumexp_dtype(self, in_dtype, out_dtype):
        a = dpnp.ones(100, dtype=in_dtype)
        res = dpnp.logsumexp(a, dtype=out_dtype)
        exp = numpy.logaddexp.reduce(dpnp.asnumpy(a))
        exp = exp.astype(out_dtype)

        dtype_list = [dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        tol = 1e-2 if in_dtype in dtype_list else 1e-6
        assert_allclose(res, exp, rtol=tol)

    @testing.with_requires("numpy>=1.26.4")
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "arr_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "out_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_logsumexp_out_dtype(self, arr_dt, out_dt, dtype):
        a = numpy.arange(1, 11).reshape((2, 5)).astype(dtype=arr_dt)
        out = numpy.zeros_like(a, shape=(2,), dtype=out_dt)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = dpnp.logsumexp(ia, out=iout, dtype=dtype, axis=1)
        if numpy.issubdtype(out_dt, numpy.uint64):
            # NumPy returns incorrect results for this case if out kwarg is used
            exp = numpy.logaddexp.reduce(a, axis=1)
            exp = exp.astype(out_dt)
        else:
            exp = numpy.logaddexp.reduce(a, out=out, axis=1)

        dtype_list = [dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        tol = 1e-2 if arr_dt in dtype_list else 1e-6
        assert_allclose(result, exp.astype(dtype), rtol=tol)
        assert result is iout


class TestReduceHypot:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axis", [None, 2, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_reduce_hypot(self, dtype, axis, keepdims):
        a = dpnp.ones((3, 4, 5, 6, 7), dtype=dtype)
        res = dpnp.reduce_hypot(a, axis=axis, keepdims=keepdims)

        exp_dt = None
        dtype_list = [dpnp.bool, dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        if dtype in dtype_list:
            exp_dt = dpnp.default_float_type(a.device)

        exp = numpy.hypot.reduce(
            dpnp.asnumpy(a), axis=axis, keepdims=keepdims, dtype=exp_dt
        )

        assert_dtype_allclose(res, exp)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axis", [None, 2, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_reduce_hypot_out(self, dtype, axis, keepdims):
        a = dpnp.ones((3, 4, 5, 6, 7), dtype=dtype)
        exp_dt = None
        dtype_list = [dpnp.bool, dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        if dtype in dtype_list:
            exp_dt = dpnp.default_float_type(a.device)
        exp = numpy.hypot.reduce(
            dpnp.asnumpy(a), axis=axis, keepdims=keepdims, dtype=exp_dt
        )

        exp_dt = exp.dtype
        if exp_dt == numpy.float64 and not has_support_aspect64():
            exp_dt = numpy.float32
        dpnp_out = dpnp.empty_like(a, shape=exp.shape, dtype=exp_dt)
        res = dpnp.reduce_hypot(a, axis=axis, out=dpnp_out, keepdims=keepdims)

        assert res is dpnp_out
        assert_dtype_allclose(res, exp)

    @pytest.mark.parametrize(
        "in_dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize("out_dtype", get_all_dtypes(no_bool=True))
    def test_reduce_hypot_dtype(self, in_dtype, out_dtype):
        a = dpnp.ones(99, dtype=in_dtype)
        res = dpnp.reduce_hypot(a, dtype=out_dtype)
        exp = numpy.hypot.reduce(dpnp.asnumpy(a))
        exp = exp.astype(out_dtype)

        dtype_list = [dpnp.int8, dpnp.uint8]
        tol = 1e-2 if in_dtype in dtype_list else 1e-6
        assert_allclose(res, exp, rtol=tol)

    @pytest.mark.parametrize(
        "arr_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "out_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_reduce_hypot_out_dtype(self, arr_dt, out_dt, dtype):
        a = numpy.arange(10, 20).reshape((2, 5)).astype(dtype=arr_dt)
        out = numpy.zeros_like(a, shape=(2,), dtype=out_dt)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = dpnp.reduce_hypot(ia, out=iout, dtype=dtype, axis=1)
        exp = numpy.hypot.reduce(a, out=out, axis=1)
        assert_allclose(result, exp.astype(dtype), rtol=1e-06)
        assert result is iout


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


def test_elemenwise_nin_nout():
    assert dpnp.abs.nin == 1
    assert dpnp.add.nin == 2

    assert dpnp.abs.nout == 1
    assert dpnp.add.nout == 1


def test_elemenwise_error():
    x = dpnp.array([1, 2, 3])
    out = dpnp.array([1, 2, 3])

    with pytest.raises(NotImplementedError):
        dpnp.abs(x, unknown_kwarg=1)
    with pytest.raises(NotImplementedError):
        dpnp.abs(x, where=False)
    with pytest.raises(NotImplementedError):
        dpnp.abs(x, subok=False)
    with pytest.raises(TypeError):
        dpnp.abs(1)
    with pytest.raises(TypeError):
        dpnp.abs([1, 2])
    with pytest.raises(TypeError):
        dpnp.abs(x, out=out, dtype="f4")
    with pytest.raises(ValueError):
        dpnp.abs(x, order="H")

    with pytest.raises(NotImplementedError):
        dpnp.add(x, x, unknown_kwarg=1)
    with pytest.raises(NotImplementedError):
        dpnp.add(x, x, where=False)
    with pytest.raises(NotImplementedError):
        dpnp.add(x, x, subok=False)
    with pytest.raises(TypeError):
        dpnp.add(1, 2)
    with pytest.raises(TypeError):
        dpnp.add([1, 2], [1, 2])
    with pytest.raises(TypeError):
        dpnp.add(x, [1, 2])
    with pytest.raises(TypeError):
        dpnp.add([1, 2], x)
    with pytest.raises(TypeError):
        dpnp.add(x, x, out=out, dtype="f4")
    with pytest.raises(ValueError):
        dpnp.add(x, x, order="H")


def test_elemenwise_order_none():
    x_np = numpy.array([1, 2, 3])
    x = dpnp.array([1, 2, 3])

    result = dpnp.abs(x, order=None)
    expected = numpy.abs(x_np, order=None)
    assert_dtype_allclose(result, expected)

    result = dpnp.add(x, x, order=None)
    expected = numpy.add(x_np, x_np, order=None)
    assert_dtype_allclose(result, expected)


def test_bitwise_1array_input():
    x = dpnp.array([1, 2, 3])
    x_np = numpy.array([1, 2, 3])

    result = dpnp.add(x, 1, dtype="f4")
    expected = numpy.add(x_np, 1, dtype="f4")
    assert_dtype_allclose(result, expected)

    result = dpnp.add(1, x, dtype="f4")
    expected = numpy.add(1, x_np, dtype="f4")
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize(
    "x_shape",
    [
        (),
        (2),
        (3, 4),
        (3, 4, 5),
    ],
)
@pytest.mark.parametrize(
    "y_shape",
    [
        (),
        (2),
        (3, 4),
        (3, 4, 5),
    ],
)
def test_elemenwise_outer(x_shape, y_shape):
    x_np = numpy.random.random(x_shape)
    y_np = numpy.random.random(y_shape)
    expected = numpy.multiply.outer(x_np, y_np)

    x = dpnp.asarray(x_np)
    y = dpnp.asarray(y_np)
    result = dpnp.multiply.outer(x, y)

    assert_dtype_allclose(result, expected)

    result_outer = dpnp.outer(x, y)
    assert dpnp.allclose(result.flatten(), result_outer.flatten())


def test_elemenwise_outer_scalar():
    s = 5
    x = dpnp.asarray([1, 2, 3])
    y = dpnp.asarray(s)
    expected = dpnp.add.outer(x, y)
    result = dpnp.add.outer(x, s)
    assert_dtype_allclose(result, expected)
