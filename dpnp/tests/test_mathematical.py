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
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    get_integer_dtypes,
    has_support_aspect16,
    has_support_aspect64,
    is_cuda_device,
)
from .test_umath import (
    _get_numpy_arrays_1in_1out,
    _get_numpy_arrays_2in_1out,
    _get_output_data_type,
)
from .third_party.cupy import testing


class TestAngle:
    @pytest.mark.parametrize("deg", [True, False])
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
    @pytest.mark.parametrize("deg", [True, False])
    def test_angle(self, dtype, deg):
        dp_a = dpnp.arange(10, dtype=dtype)
        np_a = dp_a.asnumpy()

        expected = numpy.angle(np_a, deg=deg)
        result = dpnp.angle(dp_a, deg=deg)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("deg", [True, False])
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
        if dtype == dpnp.bool:
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

        assert_allclose(res, exp, rtol=1e-06)

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
        a = numpy.arange(5, 10).astype(dtype=arr_dt)
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
        a = numpy.arange(10, 20).astype(dtype=arr_dt)
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
        a = numpy.array(
            [[1.0, 1.1, 1.5, 1.8], [-1.0, -1.1, -1.5, -1.8]], dtype=dt
        )
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
        a = numpy.array(
            [[1.0, 1.1, 1.5, 1.8], [-1.0, -1.1, -1.5, -1.8]], dtype=a_dt
        )
        ia = dpnp.array(a)

        if a.dtype != numpy.float32 and has_support_aspect64():
            out_dt = numpy.float64
        else:
            out_dt = numpy.float32
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


class TestHeavside:
    @pytest.mark.parametrize("val", [0.5, 1.0])
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_basic(self, val, dt):
        a = numpy.array(
            [[-30.0, -0.1, 0.0, 0.2], [7.5, numpy.nan, numpy.inf, -numpy.inf]],
            dtype=dt,
        )
        ia = dpnp.array(a)

        result = dpnp.heaviside(ia, val)
        expected = numpy.heaviside(a, val)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "a_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "b_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_both_input_as_arrays(self, a_dt, b_dt):
        a = numpy.array([-1.5, 0, 2.0], dtype=a_dt)
        b = numpy.array([-0, 0.5, 1.0], dtype=b_dt)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.heaviside(ia, ib)
        expected = numpy.heaviside(a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_complex_dtype(self, xp, dt):
        a = xp.array([-1.5, 0, 2.0], dtype=dt)
        assert_raises((TypeError, ValueError), xp.heaviside, a, 0.5)


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
        assert_dtype_allclose(result, expected)

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


class TestLdexp:
    @pytest.mark.parametrize("mant_dt", get_float_dtypes())
    @pytest.mark.parametrize("exp_dt", get_integer_dtypes())
    def test_basic(self, mant_dt, exp_dt):
        if (
            numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0"
            and exp_dt == numpy.int64
            and numpy.dtype("l") != numpy.int64
        ):
            pytest.skip("numpy.ldexp doesn't have a loop for the input types")

        mant = numpy.array(2.0, dtype=mant_dt)
        exp = numpy.array(3, dtype=exp_dt)
        imant, iexp = dpnp.array(mant), dpnp.array(exp)

        result = dpnp.ldexp(imant, iexp)
        expected = numpy.ldexp(mant, exp)
        assert_almost_equal(result, expected)

    def test_float_scalar(self):
        a = numpy.array(3)
        ia = dpnp.array(a)

        result = dpnp.ldexp(2.0, ia)
        expected = numpy.ldexp(2.0, a)
        assert_almost_equal(result, expected)

    @pytest.mark.parametrize("max_min", ["max", "min"])
    def test_overflow(self, max_min):
        exp_val = getattr(numpy.iinfo(numpy.dtype("l")), max_min)

        result = dpnp.ldexp(dpnp.array(2.0), exp_val)
        with numpy.errstate(over="ignore"):
            # we can't use here numpy.array(2.0), because NumPy 2.0 will cast
            # `exp_val` to int32 dtype then and `OverflowError` will be raised
            expected = numpy.ldexp(2.0, exp_val)
        assert_equal(result, expected)

    @pytest.mark.parametrize("val", [numpy.nan, numpy.inf, -numpy.inf])
    def test_nan_int_mant(self, val):
        mant = numpy.array(val)
        imant = dpnp.array(mant)

        result = dpnp.ldexp(imant, 5)
        expected = numpy.ldexp(mant, 5)
        assert_equal(result, expected)

    def test_zero_exp(self):
        exp = numpy.array(0)
        iexp = dpnp.array(exp)

        result = dpnp.ldexp(-2.5, iexp)
        expected = numpy.ldexp(-2.5, exp)
        assert_equal(result, expected)

    @pytest.mark.parametrize("stride", [-4, -2, -1, 1, 2, 4])
    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_strides(self, stride, dt):
        mant = numpy.array(
            [0.125, 0.25, 0.5, 1.0, 1.0, 2.0, 4.0, 8.0], dtype=dt
        )
        exp = numpy.array([3, 2, 1, 0, 0, -1, -2, -3], dtype="i")
        out = numpy.zeros(8, dtype=dt)
        imant, iexp, iout = dpnp.array(mant), dpnp.array(exp), dpnp.array(out)

        result = dpnp.ldexp(imant[::stride], iexp[::stride], out=iout[::stride])
        expected = numpy.ldexp(mant[::stride], exp[::stride], out=out[::stride])
        assert_equal(result, expected)

    def test_bool_exp(self):
        result = dpnp.ldexp(3.7, dpnp.array(True))
        expected = numpy.ldexp(3.7, numpy.array(True))
        assert_almost_equal(result, expected)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_uint64_exp(self, xp):
        x = xp.array(4, dtype=numpy.uint64)
        assert_raises((ValueError, TypeError), xp.ldexp, 7.3, x)


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


class TestNextafter:
    @pytest.mark.parametrize("dt", get_float_dtypes())
    @pytest.mark.parametrize(
        "val1, val2",
        [
            pytest.param(1, 2),
            pytest.param(1, 0),
            pytest.param(1, 1),
        ],
    )
    def test_float(self, val1, val2, dt):
        v1 = numpy.array(val1, dtype=dt)
        v2 = numpy.array(val2, dtype=dt)
        iv1, iv2 = dpnp.array(v1), dpnp.array(v2)

        result = dpnp.nextafter(iv1, iv2)
        expected = numpy.nextafter(v1, v2)
        assert_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_float_nan(self, dt):
        a = numpy.array(1, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.nextafter(ia, dpnp.nan)
        expected = numpy.nextafter(a, numpy.nan)
        assert_equal(result, expected)

        result = dpnp.nextafter(dpnp.nan, ia)
        expected = numpy.nextafter(numpy.nan, a)
        assert_equal(result, expected)

    @pytest.mark.parametrize("val", [0x7C00, 0x8000], ids=["val1", "val2"])
    def test_f16_strides(self, val):
        a = numpy.arange(val, dtype=numpy.uint16).astype(numpy.float16)
        hinf = numpy.array((numpy.inf,), dtype=numpy.float16)
        ia, ihinf = dpnp.array(a), dpnp.array(hinf)

        result = dpnp.nextafter(ia[:-1], ihinf)
        expected = numpy.nextafter(a[:-1], hinf)
        assert_equal(result, expected)

        result = dpnp.nextafter(ia[0], -ihinf)
        expected = numpy.nextafter(a[0], -hinf)
        assert_equal(result, expected)

        result = dpnp.nextafter(ia[1:], -ihinf)
        expected = numpy.nextafter(a[1:], -hinf)
        assert_equal(result, expected)

    @pytest.mark.parametrize("val", [0x7C00, 0x8000], ids=["val1", "val2"])
    def test_f16_array_inf(self, val):
        a = numpy.arange(val, dtype=numpy.uint16).astype(numpy.float16)
        hinf = numpy.array((numpy.inf,), dtype=numpy.float16)
        ia, ihinf = dpnp.array(a), dpnp.array(hinf)

        result = dpnp.nextafter(ihinf, ia)
        expected = numpy.nextafter(hinf, a)
        assert_equal(result, expected)

        result = dpnp.nextafter(-ihinf, ia)
        expected = numpy.nextafter(-hinf, a)
        assert_equal(result, expected)

    @pytest.mark.parametrize(
        "sign1, sign2",
        [
            pytest.param(1, 1),
            pytest.param(1, -1),
            pytest.param(-1, 1),
            pytest.param(-1, -1),
        ],
    )
    def test_f16_inf(self, sign1, sign2):
        hinf1 = numpy.array((sign1 * numpy.inf,), dtype=numpy.float16)
        hinf2 = numpy.array((sign2 * numpy.inf,), dtype=numpy.float16)
        ihinf1, ihinf2 = dpnp.array(hinf1), dpnp.array(hinf2)

        result = dpnp.nextafter(ihinf1, ihinf2)
        expected = numpy.nextafter(hinf1, hinf2)
        assert_equal(result, expected)

    @pytest.mark.parametrize("val", [0x7C00, 0x8000], ids=["val1", "val2"])
    def test_f16_array_nan(self, val):
        a = numpy.arange(val, dtype=numpy.uint16).astype(numpy.float16)
        nan = numpy.array((numpy.nan,), dtype=numpy.float16)
        ia, inan = dpnp.array(a), dpnp.array(nan)

        result = dpnp.nextafter(ia, inan)
        expected = numpy.nextafter(a, nan)
        assert_equal(result, expected)

        result = dpnp.nextafter(inan, ia)
        expected = numpy.nextafter(nan, a)
        assert_equal(result, expected)

    @pytest.mark.parametrize(
        "val1, val2",
        [
            pytest.param(numpy.nan, numpy.nan),
            pytest.param(numpy.inf, numpy.nan),
            pytest.param(numpy.nan, numpy.inf),
        ],
    )
    def test_f16_inf_nan(self, val1, val2):
        v1 = numpy.array((val1,), dtype=numpy.float16)
        v2 = numpy.array((val2,), dtype=numpy.float16)
        iv1, iv2 = dpnp.array(v1), dpnp.array(v2)

        result = dpnp.nextafter(iv1, iv2)
        expected = numpy.nextafter(v1, v2)
        assert_equal(result, expected)

    @pytest.mark.parametrize(
        "val, scalar",
        [
            pytest.param(65504, -numpy.inf),
            pytest.param(-65504, numpy.inf),
            pytest.param(numpy.inf, 0),
            pytest.param(-numpy.inf, 0),
            pytest.param(0, numpy.nan),
            pytest.param(numpy.nan, 0),
        ],
    )
    def test_f16_corner_values_with_scalar(self, val, scalar):
        a = numpy.array(val, dtype=numpy.float16)
        ia = dpnp.array(a)
        scalar = numpy.float16(scalar)

        result = dpnp.nextafter(ia, scalar)
        expected = numpy.nextafter(a, scalar)
        assert_equal(result, expected)


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
        a = numpy.arange(10, 20).reshape((2, 5)).astype(dtype=arr_dt)
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


class TestRationalFunctions:
    @pytest.mark.parametrize("func", ["gcd", "lcm"])
    @pytest.mark.parametrize("dt1", get_integer_dtypes())
    @pytest.mark.parametrize("dt2", get_integer_dtypes())
    def test_basic(self, func, dt1, dt2):
        a = numpy.array([12, 120], dtype=dt1)
        b = numpy.array([20, 120], dtype=dt2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        expected = getattr(numpy, func)(a, b)
        result = getattr(dpnp, func)(ia, ib)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("func", ["gcd", "lcm"])
    @pytest.mark.parametrize("dt", get_integer_dtypes())
    def test_broadcasting(self, func, dt):
        a = numpy.arange(6, dtype=dt)
        ia = dpnp.array(a)
        b = 20

        expected = getattr(numpy, func)(a, b)
        result = getattr(dpnp, func)(ia, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", [numpy.int32, numpy.int64])
    def test_gcd_overflow(self, dt):
        a = dt(numpy.iinfo(dt).min)  # negative power of two
        ia = dpnp.array(a)
        q = -(a // 4)

        # verify that we don't overflow when taking abs(x)
        # not relevant for lcm, where the result is unrepresentable anyway
        expected = numpy.gcd(a, q)
        result = dpnp.gcd(ia, q)
        assert_array_equal(result, expected)

    def test_lcm_overflow(self):
        big = numpy.int32(numpy.iinfo(numpy.int32).max // 11)
        a, b = 2 * big, 5 * big
        ia, ib = dpnp.array(a), dpnp.array(b)

        # verify that we don't overflow when a*b does overflow
        expected = numpy.lcm(a, b)
        result = dpnp.lcm(ia, ib)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("func", ["gcd", "lcm"])
    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_inf_and_nan(self, func, xp):
        inf = xp.array([xp.inf])
        assert_raises((TypeError, ValueError), getattr(xp, func), inf, 1)
        assert_raises((TypeError, ValueError), getattr(xp, func), 1, inf)
        assert_raises((TypeError, ValueError), getattr(xp, func), xp.nan, inf)
        assert_raises(
            (TypeError, ValueError), getattr(xp, func), 4, float(xp.inf)
        )


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
        a = numpy.linspace(-1, 1, 100, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.sinc(ia)
        expected = numpy.sinc(a)
        assert_dtype_allclose(result, expected)

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
        assert_dtype_allclose(result, expected)

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
        if is_cuda_device():
            if dt is dpnp.float32:
                pytest.skip("SAT-7588")
        a = numpy.array([0.0, -0.0], dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.spacing(ia)
        expected = numpy.spacing(a)
        if numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0":
            assert_equal(result, expected)
        else:
            # numpy.spacing(-0.0) == numpy.spacing(0.0), i.e. NumPy returns
            # positive value (looks as a bug in NumPy), because for any other
            # negative input the NumPy result will be also a negative value.
            expected[1] *= -1
            assert_equal(result, expected)

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
        a = numpy.array([1, 0, -3], dtype=dt)
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
        if numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0":
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
        "dt", get_all_dtypes(no_none=True, no_bool=True, no_complex=True)
    )
    def test_period(self, dt):
        a = numpy.array([1, 1 + 256], dtype=dt)
        ia = dpnp.array(a)

        # unwrap removes jumps greater than 255
        result = dpnp.unwrap(ia, period=255)
        expected = numpy.unwrap(a, period=255)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_bool=True, no_complex=True)
    )
    def test_rand_period(self, dt):
        a = generate_random_numpy_array(10) * 1000
        a = a.astype(dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.unwrap(ia, period=255)
        expected = numpy.unwrap(a, period=255)
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
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_discont(self, dt):
        a = numpy.array([0, 75, 150, 225, 300, 430], dtype=dt)
        a = numpy.mod(a, 250)
        ia = dpnp.array(a)

        result = dpnp.unwrap(ia, period=250)
        expected = numpy.unwrap(a, period=250)
        assert_array_equal(result, expected)

        result = dpnp.unwrap(ia, period=250, discont=140)
        expected = numpy.unwrap(a, period=250, discont=140)
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
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_negative(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

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
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

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


@pytest.mark.parametrize(
    "data",
    [[2, 0, -2], [1.1, -1.1]],
    ids=["[2, 0, -2]", "[1.1, -1.1]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_sign(data, dtype):
    np_a = numpy.array(data, dtype=dtype)
    dpnp_a = dpnp.array(data, dtype=dtype)

    result = dpnp.sign(dpnp_a)
    expected = numpy.sign(np_a)
    assert_dtype_allclose(result, expected)

    # out keyword
    if dtype is not None:
        dp_out = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = dpnp.sign(dpnp_a, out=dp_out)
        assert dp_out is result
        assert_dtype_allclose(result, expected)


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
        # NumPy < 2.0.0 while output has the dtype of input for NumPy >= 2.0.0
        # (dpnp follows the latter behavior except for boolean dtype where it
        # returns int8)
        if (
            numpy.lib.NumpyVersion(numpy.__version__) < "2.0.0"
            or dtype == numpy.bool
        ):
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


class TestAdd:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_add(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "add", dtype, [-5, 5, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.add(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.add(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.add(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_inplace_strides(self, dtype):
        size = 21
        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] += 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] += 4

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(10)
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
    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_divide(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "divide", dtype, [-5, 5, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = _get_output_data_type(dtype)
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.divide(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_out_overlap(self, dtype):
        size = 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.divide(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.divide(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_inplace_strides(self, dtype):
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


class TestFmaxFmin:
    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
    @pytest.mark.parametrize("func", ["fmax", "fmin"])
    def test_half(self, func):
        a = numpy.array([0, 1, 2, 4, 2], dtype=numpy.float16)
        b = numpy.array([-2, 5, 1, 4, 3], dtype=numpy.float16)
        c = numpy.array([0, -1, -numpy.inf, numpy.nan, 6], dtype=numpy.float16)
        ia, ib, ic = dpnp.array(a), dpnp.array(b), dpnp.array(c)

        result = getattr(dpnp, func)(ia, ib)
        expected = getattr(numpy, func)(a, b)
        assert_equal(result, expected)

        result = getattr(dpnp, func)(ib, ic)
        expected = getattr(numpy, func)(b, c)
        assert_equal(result, expected)

    @pytest.mark.parametrize("func", ["fmax", "fmin"])
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_float_nans(self, func, dtype):
        a = numpy.array([0, numpy.nan, numpy.nan], dtype=dtype)
        b = numpy.array([numpy.nan, 0, numpy.nan], dtype=dtype)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = getattr(dpnp, func)(ia, ib)
        expected = getattr(numpy, func)(a, b)
        assert_equal(result, expected)

    @pytest.mark.parametrize("func", ["fmax", "fmin"])
    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize(
        "nan_val",
        [
            complex(numpy.nan, 0),
            complex(0, numpy.nan),
            complex(numpy.nan, numpy.nan),
        ],
        ids=["nan+0j", "nanj", "nan+nanj"],
    )
    def test_complex_nans(self, func, dtype, nan_val):
        a = numpy.array([0, nan_val, nan_val], dtype=dtype)
        b = numpy.array([nan_val, 0, nan_val], dtype=dtype)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = getattr(dpnp, func)(ia, ib)
        expected = getattr(numpy, func)(a, b)
        assert_equal(result, expected)

    @pytest.mark.parametrize("func", ["fmax", "fmin"])
    @pytest.mark.parametrize("dtype", get_float_dtypes(no_float16=False))
    def test_precision(self, func, dtype):
        dtmin = numpy.finfo(dtype).min
        dtmax = numpy.finfo(dtype).max
        d1 = dtype(0.1)
        d1_next = numpy.nextafter(d1, numpy.inf)

        test_cases = [
            # v1     v2
            (dtmin, -numpy.inf),
            (dtmax, -numpy.inf),
            (d1, d1_next),
            (dtmax, numpy.nan),
        ]

        for v1, v2 in test_cases:
            a = numpy.array([v1])
            b = numpy.array([v2])
            ia, ib = dpnp.array(a), dpnp.array(b)

            result = getattr(dpnp, func)(ia, ib)
            expected = getattr(numpy, func)(a, b)
            assert_allclose(result, expected)


class TestFloorDivide:
    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_floor_divide(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "floor_divide", dtype, [-5, 5, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.floor_divide(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_out_overlap(self, dtype):
        size = 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.floor_divide(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        # original
        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.floor_divide(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_inplace_strides(self, dtype):
        size = 21

        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] //= 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] //= 4

        assert_dtype_allclose(dp_a, np_a)

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


class TestBoundFuncs:
    @pytest.fixture(
        params=[
            {"func_name": "fmax", "input_values": [-5, 5, 10]},
            {"func_name": "fmin", "input_values": [-5, 5, 10]},
            {"func_name": "maximum", "input_values": [-5, 5, 10]},
            {"func_name": "minimum", "input_values": [-5, 5, 10]},
        ],
        ids=[
            "fmax",
            "fmin",
            "maximum",
            "minimum",
        ],
    )
    def func_params(self, request):
        return request.param

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_out(self, func_params, dtype):
        func_name = func_params["func_name"]
        input_values = func_params["input_values"]
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            func_name, dtype, input_values
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = getattr(dpnp, func_name)(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_out_overlap(self, func_params, dtype):
        func_name = func_params["func_name"]
        size = 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        getattr(dpnp, func_name)(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        np_a = numpy.arange(2 * size, dtype=dtype)
        getattr(numpy, func_name)(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, func_params, shape):
        func_name = func_params["func_name"]
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(10)
        dp_out = dpnp.empty(shape)

        with pytest.raises(ValueError):
            getattr(dpnp, func_name)(dp_array1, dp_array2, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, func_params, out):
        func_name = func_params["func_name"]
        a = dpnp.arange(10)

        assert_raises(TypeError, getattr(dpnp, func_name), a, 2, out)
        assert_raises(TypeError, getattr(numpy, func_name), a.asnumpy(), 2, out)


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
        exp_dtype = (
            dpnp.default_float_type(a.device) if dtype == dpnp.bool else None
        )
        exp = numpy.logaddexp.reduce(
            dpnp.asnumpy(a), axis=axis, keepdims=keepdims, dtype=exp_dtype
        )

        assert_dtype_allclose(res, exp)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axis", [None, 2, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_logsumexp_out(self, dtype, axis, keepdims):
        a = dpnp.ones((3, 4, 5, 6, 7), dtype=dtype)
        exp_dtype = (
            dpnp.default_float_type(a.device) if dtype == dpnp.bool else None
        )
        exp = numpy.logaddexp.reduce(
            dpnp.asnumpy(a), axis=axis, keepdims=keepdims, dtype=exp_dtype
        )

        exp_dtype = exp.dtype
        if exp_dtype == numpy.float64 and not has_support_aspect64():
            exp_dtype = numpy.float32
        dpnp_out = dpnp.empty_like(a, shape=exp.shape, dtype=exp_dtype)
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

        assert_allclose(res, exp, rtol=1e-06)

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
        a = numpy.arange(10, 20).reshape((2, 5)).astype(dtype=arr_dt)
        out = numpy.zeros_like(a, shape=(2,), dtype=out_dt)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = dpnp.logsumexp(ia, out=iout, dtype=dtype, axis=1)
        exp = numpy.logaddexp.reduce(a, out=out, axis=1)
        assert_allclose(result, exp.astype(dtype), rtol=1e-06)
        assert result is iout


class TestReduceHypot:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axis", [None, 2, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_reduce_hypot(self, dtype, axis, keepdims):
        a = dpnp.ones((3, 4, 5, 6, 7), dtype=dtype)
        res = dpnp.reduce_hypot(a, axis=axis, keepdims=keepdims)
        exp_dtype = (
            dpnp.default_float_type(a.device) if dtype == dpnp.bool else None
        )
        exp = numpy.hypot.reduce(
            dpnp.asnumpy(a), axis=axis, keepdims=keepdims, dtype=exp_dtype
        )

        assert_dtype_allclose(res, exp)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axis", [None, 2, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_reduce_hypot_out(self, dtype, axis, keepdims):
        a = dpnp.ones((3, 4, 5, 6, 7), dtype=dtype)
        exp_dtype = (
            dpnp.default_float_type(a.device) if dtype == dpnp.bool else None
        )
        exp = numpy.hypot.reduce(
            dpnp.asnumpy(a), axis=axis, keepdims=keepdims, dtype=exp_dtype
        )

        exp_dtype = exp.dtype
        if exp_dtype == numpy.float64 and not has_support_aspect64():
            exp_dtype = numpy.float32
        dpnp_out = dpnp.empty_like(a, shape=exp.shape, dtype=exp_dtype)
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

        assert_allclose(res, exp, rtol=1e-06)

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


class TestMultiply:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_multiply(self, dtype):
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "multiply", dtype, [0, 10, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        dp_out = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.multiply(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 15
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.multiply(dp_a[size::], dp_a[::2], out=dp_a[:size:])

        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.multiply(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_inplace_strides(self, dtype):
        size = 21
        np_a = numpy.arange(size, dtype=dtype)
        np_a[::3] *= 4

        dp_a = dpnp.arange(size, dtype=dtype)
        dp_a[::3] *= 4

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(10)
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
    def test_basic(self, array, val, data_type, val_type):
        np_a = numpy.array(array, dtype=data_type)
        dpnp_a = dpnp.array(array, dtype=data_type)
        val_ = val_type(val)

        result = dpnp.power(dpnp_a, val_)
        expected = numpy.power(np_a, val_)
        assert_allclose(expected, result, rtol=1e-6)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_power(self, dtype):
        numpy.random.seed(42)
        np_array1, np_array2, expected = _get_numpy_arrays_2in_1out(
            "power", dtype, [0, 10, 10]
        )

        dp_array1 = dpnp.array(np_array1)
        dp_array2 = dpnp.array(np_array2)
        out_dtype = numpy.int8 if dtype == numpy.bool_ else dtype
        dp_out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = dpnp.power(dp_array1, dp_array2, out=dp_out)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_out_overlap(self, dtype):
        size = 1 if dtype == dpnp.bool else 10
        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dpnp.power(dp_a[size::], dp_a[::2], out=dp_a[:size:]),

        np_a = numpy.arange(2 * size, dtype=dtype)
        numpy.power(np_a[size::], np_a[::2], out=np_a[:size:])

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_inplace_strided_out(self, dtype):
        size = 5
        np_a = numpy.arange(2 * size, dtype=dtype)
        np_a[::3] **= 3

        dp_a = dpnp.arange(2 * size, dtype=dtype)
        dp_a[::3] **= 3

        assert_dtype_allclose(dp_a, np_a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array1 = dpnp.arange(10)
        dp_array2 = dpnp.arange(10)
        dp_out = dpnp.empty(shape)

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

    @pytest.mark.parametrize("shape", [(), (3, 2)], ids=["()", "(3, 2)"])
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_power_scalar(self, shape, dtype):
        np_a = numpy.ones(shape, dtype=dtype)
        dpnp_a = dpnp.ones(shape, dtype=dtype)

        result = 4.2**dpnp_a**-1.3
        expected = 4.2**np_a**-1.3
        assert_allclose(result, expected, rtol=1e-6)

        result **= dpnp_a
        expected **= np_a
        assert_allclose(result, expected, rtol=1e-6)

    def test_alias(self):
        a = dpnp.arange(10)
        res1 = dpnp.power(a, 3)
        res2 = dpnp.pow(a, 3)

        assert_array_equal(res1, res2)


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


class TestMatmul:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize(
        "order1, order2", [("C", "C"), ("C", "F"), ("F", "C"), ("F", "F")]
    )
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((4,), (4,)),
            ((1, 4), (4, 1)),
            ((4,), (4, 2)),
            ((1, 4), (4, 2)),
            ((2, 4), (4,)),
            ((2, 4), (4, 1)),
            ((1, 4), (4,)),  # output should be 1-d not 0-d
            ((4,), (4, 1)),
            ((1, 4), (4, 1)),
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
            ((2, 3, 3), (10, 1, 3, 1)),
            ((10, 2, 3, 3), (10, 1, 3, 1)),
            ((10, 2, 3, 3), (10, 2, 3, 1)),
            ((10, 1, 1, 3), (1, 3, 3)),
            ((10, 1, 1, 3), (2, 3, 3)),
            ((10, 1, 1, 3), (10, 2, 3, 3)),
            ((10, 2, 1, 3), (10, 2, 3, 3)),
            ((3, 3, 1), (3, 1, 2)),
            ((3, 3, 1), (1, 1, 2)),
            ((1, 3, 1), (3, 1, 2)),
            ((4, 1, 3, 1), (1, 3, 1, 2)),
            ((1, 3, 3, 1), (4, 1, 1, 2)),
        ],
    )
    def test_matmul(self, order1, order2, shape1, shape2):
        # input should be float type otherwise they are copied to c-contigous array
        # so testing order becomes meaningless
        dtype = dpnp.default_float_type()
        a1 = numpy.arange(numpy.prod(shape1), dtype=dtype).reshape(shape1)
        a2 = numpy.arange(numpy.prod(shape2), dtype=dtype).reshape(shape2)
        a1 = numpy.array(a1, order=order1)
        a2 = numpy.array(a2, order=order2)

        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)
        result = dpnp.matmul(b1, b2)
        expected = numpy.matmul(a1, a2)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "order1, order2", [("C", "C"), ("C", "F"), ("F", "C"), ("F", "F")]
    )
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((2, 0), (0, 3)),
            ((0, 4), (4, 3)),
            ((2, 4), (4, 0)),
            ((1, 2, 3), (0, 3, 5)),
            ((0, 2, 3), (1, 3, 5)),
            ((2, 3), (0, 3, 5)),
            ((0, 2, 3), (3, 5)),
            ((0, 0, 4, 3), (1, 1, 3, 5)),
            ((6, 0, 4, 3), (1, 3, 5)),
            ((0, 7, 4, 3), (3, 5)),
            ((0, 7, 4, 3), (1, 7, 3, 5)),
            ((0, 7, 4, 3), (7, 3, 5)),
            ((6, 0, 4, 3), (6, 1, 3, 5)),
            ((1, 1, 4, 3), (0, 0, 3, 5)),
            ((1, 4, 3), (6, 0, 3, 5)),
            ((4, 3), (0, 0, 3, 5)),
            ((6, 1, 4, 3), (6, 0, 3, 5)),
            ((1, 7, 4, 3), (0, 7, 3, 5)),
            ((7, 4, 3), (0, 7, 3, 5)),
        ],
    )
    def test_matmul_empty(self, order1, order2, shape1, shape2):
        dtype = dpnp.default_float_type()
        a1 = numpy.arange(numpy.prod(shape1), dtype=dtype).reshape(shape1)
        a2 = numpy.arange(numpy.prod(shape2), dtype=dtype).reshape(shape2)
        a1 = numpy.array(a1, order=order1)
        a2 = numpy.array(a2, order=order2)
        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        result = dpnp.matmul(b1, b2)
        expected = numpy.matmul(a1, a2)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "shape1, shape2",
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
    def test_matmul_bool(self, shape1, shape2):
        x = numpy.arange(2, dtype=numpy.bool_)
        a1 = numpy.resize(x, numpy.prod(shape1)).reshape(shape1)
        a2 = numpy.resize(x, numpy.prod(shape2)).reshape(shape2)
        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        result = dpnp.matmul(b1, b2)
        expected = numpy.matmul(a1, a2)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "axes",
        [
            [(-3, -1), (0, 2), (-2, -3)],
            [(3, 1), (2, 0), (3, 1)],
            [(3, 1), (2, 0), (0, 1)],
        ],
    )
    def test_matmul_axes_ND_ND(self, dtype, axes):
        a = generate_random_numpy_array((2, 5, 3, 4), dtype)
        b = generate_random_numpy_array((4, 2, 5, 3), dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.matmul(ia, ib, axes=axes)
        expected = numpy.matmul(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            [(1, 0), (0), (0)],
            [(1, 0), 0, 0],
            [(1, 0), (0,), (0,)],
        ],
    )
    def test_matmul_axes_ND_1D(self, axes):
        a = numpy.arange(3 * 4 * 5).reshape(3, 4, 5)
        b = numpy.arange(3)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.matmul(ia, ib, axes=axes)
        expected = numpy.matmul(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            [(0,), (0, 1), (0)],
            [(0), (0, 1), 0],
            [0, (0, 1), (0,)],
        ],
    )
    def test_matmul_axes_1D_ND(self, axes):
        a = numpy.arange(3 * 4 * 5).reshape(3, 4, 5)
        b = numpy.arange(3)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.matmul(ib, ia, axes=axes)
        expected = numpy.matmul(b, a, axes=axes)
        assert_dtype_allclose(result, expected)

    def test_matmul_axes_1D_1D(self):
        a = numpy.arange(3)
        ia = dpnp.array(a)

        axes = [0, 0, ()]
        result = dpnp.matmul(ia, ia, axes=axes)
        expected = numpy.matmul(a, a, axes=axes)
        assert_dtype_allclose(result, expected)

        out = dpnp.empty((), dtype=ia.dtype)
        result = dpnp.matmul(ia, ia, axes=axes, out=out)
        assert out is result
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "axes, out_shape",
        [
            ([(-3, -1), (0, 2), (-2, -3)], (2, 5, 5, 3)),
            ([(3, 1), (2, 0), (3, 1)], (2, 4, 3, 4)),
            ([(3, 1), (2, 0), (1, 2)], (2, 4, 4, 3)),
        ],
    )
    def test_matmul_axes_out(self, dtype, axes, out_shape):
        a = generate_random_numpy_array((2, 5, 3, 4), dtype)
        b = generate_random_numpy_array((4, 2, 5, 3), dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        out_dp = dpnp.empty(out_shape, dtype=dtype)
        result = dpnp.matmul(ia, ib, axes=axes, out=out_dp)
        assert result is out_dp
        expected = numpy.matmul(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes, b_shape, out_shape",
        [
            ([(1, 0), 0, 0], (3,), (4, 5)),
            ([(1, 0), 0, 1], (3,), (5, 4)),
            ([(1, 0), (0, 1), (1, 2)], (3, 1), (5, 4, 1)),
            ([(1, 0), (0, 1), (0, 2)], (3, 1), (4, 5, 1)),
            ([(1, 0), (0, 1), (1, 0)], (3, 1), (1, 4, 5)),
        ],
    )
    def test_matmul_axes_out_1D(self, axes, b_shape, out_shape):
        a = numpy.arange(3 * 4 * 5).reshape(3, 4, 5)
        b = numpy.arange(3).reshape(b_shape)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        out_dp = dpnp.empty(out_shape)
        out_np = numpy.empty(out_shape)
        result = dpnp.matmul(ia, ib, axes=axes, out=out_dp)
        assert result is out_dp
        expected = numpy.matmul(a, b, axes=axes, out=out_np)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("in_dt", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "out_dt", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.parametrize(
        "shape1, shape2",
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
    def test_matmul_dtype_matrix_inout(self, in_dt, out_dt, shape1, shape2):
        a1 = generate_random_numpy_array(shape1, in_dt)
        a2 = generate_random_numpy_array(shape2, in_dt)
        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        if dpnp.can_cast(dpnp.result_type(b1, b2), out_dt, casting="same_kind"):
            result = dpnp.matmul(b1, b2, dtype=out_dt)
            expected = numpy.matmul(a1, a2, dtype=out_dt)
            assert_dtype_allclose(result, expected)
        else:
            with pytest.raises(TypeError):
                dpnp.matmul(b1, b2, dtype=out_dt)

    @pytest.mark.parametrize("dtype1", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("dtype2", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
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
    def test_matmul_dtype_matrix_inputs(self, dtype1, dtype2, shape1, shape2):
        a1 = generate_random_numpy_array(shape1, dtype1)
        a2 = generate_random_numpy_array(shape2, dtype2)
        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        result = dpnp.matmul(b1, b2)
        expected = numpy.matmul(a1, a2)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("order1", ["C", "F", "A"])
    @pytest.mark.parametrize("order2", ["C", "F", "A"])
    @pytest.mark.parametrize("order", ["C", "F", "K", "A"])
    @pytest.mark.parametrize(
        "shape1, shape2",
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
    def test_matmul_order(self, order1, order2, order, shape1, shape2):
        a1 = numpy.arange(numpy.prod(shape1)).reshape(shape1, order=order1)
        a2 = numpy.arange(numpy.prod(shape2)).reshape(shape2, order=order2)
        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        result = dpnp.matmul(b1, b2, order=order)
        expected = numpy.matmul(a1, a2, order=order)
        # For the special case of shape1 = (6, 7, 4, 3), shape2 = (6, 7, 3, 5)
        # and order1 = "F" and order2 = "F", NumPy result is not c-contiguous
        # nor f-contiguous, while dpnp (and cupy) results are c-contiguous
        if not (
            shape1 == (6, 7, 4, 3)
            and shape2 == (6, 7, 3, 5)
            and order1 == "F"
            and order2 == "F"
            and order == "K"
        ):
            assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "stride",
        [(-2, -2, -2, -2), (2, 2, 2, 2), (-2, 2, -2, 2), (2, -2, 2, -2)],
        ids=["-2", "2", "(-2, 2)", "(2, -2)"],
    )
    def test_matmul_strided1(self, stride):
        for dim in [1, 2, 3, 4]:
            shape = tuple(20 for _ in range(dim))
            A = numpy.random.rand(*shape)
            A_dp = dpnp.asarray(A)
            slices = tuple(slice(None, None, stride[i]) for i in range(dim))
            a = A[slices]
            a_dp = A_dp[slices]
            # input arrays will be copied into c-contiguous arrays
            # the 2D base is not c-contiguous nor f-contigous
            result = dpnp.matmul(a_dp, a_dp)
            expected = numpy.matmul(a, a)
            assert_dtype_allclose(result, expected)

            OUT = dpnp.empty(shape, dtype=result.dtype)
            out = OUT[slices]
            result = dpnp.matmul(a_dp, a_dp, out=out)
            assert result is out
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "shape", [(10, 3, 3), (12, 10, 3, 3)], ids=["3D", "4D"]
    )
    @pytest.mark.parametrize("stride", [-1, -2, 2], ids=["-1", "-2", "2"])
    @pytest.mark.parametrize("transpose", [False, True], ids=["False", "True"])
    def test_matmul_strided2(self, shape, stride, transpose):
        # one dimension (-3) is strided
        # if negative stride, copy is needed and the base becomes c-contiguous
        # otherwise the base remains the same as input in gemm_batch
        A = numpy.random.rand(*shape)
        A_dp = dpnp.asarray(A)
        if transpose:
            A = numpy.moveaxis(A, (-2, -1), (-1, -2))
            A_dp = dpnp.moveaxis(A_dp, (-2, -1), (-1, -2))
        index = [slice(None)] * len(shape)
        index[-3] = slice(None, None, stride)
        index = tuple(index)
        a = A[index]
        a_dp = A_dp[index]
        result = dpnp.matmul(a_dp, a_dp)
        expected = numpy.matmul(a, a)
        assert_dtype_allclose(result, expected)

        OUT = dpnp.empty(shape, dtype=result.dtype)
        out = OUT[index]
        result = dpnp.matmul(a_dp, a_dp, out=out)
        assert result is out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "stride",
        [(-2, -2), (2, 2), (-2, 2), (2, -2)],
        ids=["(-2, -2)", "(2, 2)", "(-2, 2)", "(2, -2)"],
    )
    @pytest.mark.parametrize("transpose", [False, True], ids=["False", "True"])
    def test_matmul_strided3(self, stride, transpose):
        # 4D case, the 1st and 2nd dimensions are strided
        # For negative stride, copy is needed and the base becomes c-contiguous.
        # For positive stride, no copy but reshape makes the base c-contiguous.
        stride0, stride1 = stride
        shape = (12, 10, 3, 3)  # 4D array
        A = numpy.random.rand(*shape)
        A_dp = dpnp.asarray(A)
        if transpose:
            A = numpy.moveaxis(A, (-2, -1), (-1, -2))
            A_dp = dpnp.moveaxis(A_dp, (-2, -1), (-1, -2))
        a = A[::stride0, ::stride1]
        a_dp = A_dp[::stride0, ::stride1]
        result = dpnp.matmul(a_dp, a_dp)
        expected = numpy.matmul(a, a)
        assert_dtype_allclose(result, expected)

        OUT = dpnp.empty(shape, dtype=result.dtype)
        out = OUT[::stride0, ::stride1]
        result = dpnp.matmul(a_dp, a_dp, out=out)
        assert result is out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("shape", [(8, 10)], ids=["2D"])
    @pytest.mark.parametrize("incx", [-2, 2], ids=["-2", "2"])
    @pytest.mark.parametrize("incy", [-2, 2], ids=["-2", "2"])
    @pytest.mark.parametrize("transpose", [False, True], ids=["False", "True"])
    def test_matmul_strided_mat_vec(self, shape, incx, incy, transpose):
        # vector is strided
        if transpose:
            s1 = shape[-2]
            s2 = shape[-1]
        else:
            s1 = shape[-1]
            s2 = shape[-2]
        a = numpy.random.rand(*shape)
        B = numpy.random.rand(2 * s1)
        a_dp = dpnp.asarray(a)
        if transpose:
            a = numpy.moveaxis(a, (-2, -1), (-1, -2))
            a_dp = dpnp.moveaxis(a_dp, (-2, -1), (-1, -2))
        B_dp = dpnp.asarray(B)
        b = B[::incx]
        b_dp = B_dp[::incx]

        result = dpnp.matmul(a_dp, b_dp)
        expected = numpy.matmul(a, b)
        assert_dtype_allclose(result, expected)

        out_shape = shape[:-2] + (2 * s2,)
        OUT = dpnp.empty(out_shape, dtype=result.dtype)
        out = OUT[..., ::incy]
        result = dpnp.matmul(a_dp, b_dp, out=out)
        assert result is out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("shape", [(8, 10)], ids=["2D"])
    @pytest.mark.parametrize("incx", [-2, 2], ids=["-2", "2"])
    @pytest.mark.parametrize("incy", [-2, 2], ids=["-2", "2"])
    @pytest.mark.parametrize("transpose", [False, True], ids=["False", "True"])
    def test_matmul_strided_vec_mat(self, shape, incx, incy, transpose):
        # vector is strided
        if transpose:
            s1 = shape[-2]
            s2 = shape[-1]
        else:
            s1 = shape[-1]
            s2 = shape[-2]
        a = numpy.random.rand(*shape)
        B = numpy.random.rand(2 * s2)
        a_dp = dpnp.asarray(a)
        if transpose:
            a = numpy.moveaxis(a, (-2, -1), (-1, -2))
            a_dp = dpnp.moveaxis(a_dp, (-2, -1), (-1, -2))
        B_dp = dpnp.asarray(B)
        b = B[::incx]
        b_dp = B_dp[::incx]

        result = dpnp.matmul(b_dp, a_dp)
        expected = numpy.matmul(b, a)
        assert_dtype_allclose(result, expected)

        out_shape = shape[:-2] + (2 * s1,)
        OUT = dpnp.empty(out_shape, dtype=result.dtype)
        out = OUT[..., ::incy]
        result = dpnp.matmul(b_dp, a_dp, out=out)
        assert result is out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "order1, order2, out_order",
        [
            ("C", "C", "C"),
            ("C", "C", "F"),
            ("C", "F", "C"),
            ("C", "F", "F"),
            ("F", "C", "C"),
            ("F", "C", "F"),
            ("F", "F", "F"),
            ("F", "F", "C"),
        ],
    )
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_matmul_out1(self, order1, order2, out_order, dtype):
        # test gemm with out keyword
        a1 = numpy.arange(20, dtype=dtype).reshape(5, 4, order=order1)
        a2 = numpy.arange(28, dtype=dtype).reshape(4, 7, order=order2)

        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        dpnp_out = dpnp.empty((5, 7), dtype=dtype, order=out_order)
        result = dpnp.matmul(b1, b2, out=dpnp_out)
        assert result is dpnp_out

        out = numpy.empty((5, 7), dtype=dtype, order=out_order)
        expected = numpy.matmul(a1, a2, out=out)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("trans", [True, False])
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_matmul_out2(self, trans, dtype):
        # test gemm_batch with out keyword
        # the base of input arrays is c-contiguous
        # the base of output array is c-contiguous or f-contiguous
        a1 = numpy.arange(24, dtype=dtype).reshape(2, 3, 4)
        a2 = numpy.arange(40, dtype=dtype).reshape(2, 4, 5)
        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        if trans:
            dpnp_out = dpnp.empty((2, 5, 3), dtype=dtype).transpose(0, 2, 1)
            out = numpy.empty((2, 5, 3), dtype=dtype).transpose(0, 2, 1)
        else:
            dpnp_out = dpnp.empty((2, 3, 5), dtype=dtype)
            out = numpy.empty((2, 3, 5), dtype=dtype)

        result = dpnp.matmul(b1, b2, out=dpnp_out)
        assert result is dpnp_out

        expected = numpy.matmul(a1, a2, out=out)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("trans", [True, False])
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_matmul_out3(self, trans, dtype):
        # test gemm_batch with out keyword
        # the base of input arrays is f-contiguous
        # the base of output array is c-contiguous or f-contiguous
        a1 = numpy.arange(24, dtype=dtype).reshape(2, 4, 3)
        a2 = numpy.arange(40, dtype=dtype).reshape(2, 5, 4)
        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        a1 = numpy.asarray(a1).transpose(0, 2, 1)
        a2 = numpy.asarray(a2).transpose(0, 2, 1)
        b1 = b1.transpose(0, 2, 1)
        b2 = b2.transpose(0, 2, 1)

        if trans:
            dpnp_out = dpnp.empty((2, 5, 3), dtype=dtype).transpose(0, 2, 1)
            out = numpy.empty((2, 5, 3), dtype=dtype).transpose(0, 2, 1)
        else:
            dpnp_out = dpnp.empty((2, 3, 5), dtype=dtype)
            out = numpy.empty((2, 3, 5), dtype=dtype)

        result = dpnp.matmul(b1, b2, out=dpnp_out)
        assert result is dpnp_out

        expected = numpy.matmul(a1, a2, out=out)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "out_shape",
        [
            ((4, 5)),
            ((6,)),
            ((4, 7, 2)),
        ],
    )
    def test_matmul_out_0D(self, out_shape):
        # for matmul of 1-D arrays, output is 0-D and if out keyword is given
        # NumPy repeats the data to match the shape of output array
        a = numpy.arange(3)
        b = dpnp.asarray(a)

        numpy_out = numpy.empty(out_shape)
        dpnp_out = dpnp.empty(out_shape)
        result = dpnp.matmul(b, b, out=dpnp_out)
        expected = numpy.matmul(a, a, out=numpy_out)
        assert result is dpnp_out
        assert_dtype_allclose(result, expected)

    @testing.slow
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((5000, 5000, 2, 2), (5000, 5000, 2, 2)),
            ((2, 2), (5000, 5000, 2, 2)),
            ((5000, 5000, 2, 2), (2, 2)),
        ],
    )
    def test_matmul_large(self, shape1, shape2):
        a = generate_random_numpy_array(shape1)
        b = generate_random_numpy_array(shape2)
        a_dp = dpnp.asarray(a)
        b_dp = dpnp.asarray(b)

        result = dpnp.matmul(a_dp, b_dp)
        expected = numpy.matmul(a, b)
        assert_dtype_allclose(result, expected, factor=24)

    @testing.with_requires("numpy>=2.0")
    def test_linalg_matmul(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.linalg.matmul(ia, ib)
        expected = numpy.linalg.matmul(a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "sh1, sh2",
        [
            ((2, 3, 3), (2, 3, 3)),
            ((3, 3, 3, 3), (3, 3, 3, 3)),
        ],
        ids=["gemm", "gemm_batch"],
    )
    def test_matmul_with_offsets(self, sh1, sh2):
        a = generate_random_numpy_array(sh1)
        b = generate_random_numpy_array(sh2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = ia[1] @ ib[1]
        expected = a[1] @ b[1]
        assert_dtype_allclose(result, expected)


class TestMatmulInplace:
    ALL_DTYPES = get_all_dtypes(no_none=True)
    DTYPES = {}
    for i in ALL_DTYPES:
        for j in ALL_DTYPES:
            if numpy.can_cast(j, i):
                DTYPES[f"{i}-{j}"] = (i, j)

    @pytest.mark.parametrize("dtype1, dtype2", DTYPES.values())
    def test_basic(self, dtype1, dtype2):
        a = numpy.arange(10).reshape(5, 2).astype(dtype1)
        b = numpy.ones((2, 2), dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)
        ia_id = id(ia)

        a @= b
        ia @= ib
        assert id(ia) == ia_id
        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize(
        "a_sh, b_sh",
        [
            pytest.param((10**5, 10), (10, 10), id="2d_large"),
            pytest.param((10**4, 10, 10), (1, 10, 10), id="3d_large"),
            pytest.param((3,), (3,), id="1d"),
            pytest.param((3, 3), (3,), id="2d_1d"),
            pytest.param((3,), (3, 3), id="1d_2d"),
            pytest.param((3, 3), (3, 1), id="2d_broadcast"),
            pytest.param((1, 3), (3, 3), id="2d_broadcast_reverse"),
            pytest.param((3, 3, 3), (1, 3, 1), id="3d_broadcast1"),
            pytest.param((3, 3, 3), (1, 3, 3), id="3d_broadcast2"),
            pytest.param((3, 3, 3), (3, 3, 1), id="3d_broadcast3"),
            pytest.param((1, 3, 3), (3, 3, 3), id="3d_broadcast_reverse1"),
            pytest.param((3, 1, 3), (3, 3, 3), id="3d_broadcast_reverse2"),
            pytest.param((1, 1, 3), (3, 3, 3), id="3d_broadcast_reverse3"),
        ],
    )
    def test_shapes(self, a_sh, b_sh):
        a_sz, b_sz = numpy.prod(a_sh), numpy.prod(b_sh)
        a = numpy.arange(a_sz).reshape(a_sh).astype(numpy.float64)
        b = numpy.arange(b_sz).reshape(b_sh)

        ia, ib = dpnp.array(a), dpnp.array(b)
        ia_id = id(ia)

        expected = a @ b
        if expected.shape != a_sh:
            if len(b_sh) == 1:
                # check the exception matches NumPy
                match = "inplace matrix multiplication requires"
            else:
                match = None

            with pytest.raises(ValueError, match=match):
                a @= b

            with pytest.raises(ValueError, match=match):
                ia @= ib
        else:
            ia @= ib
            assert id(ia) == ia_id
            assert_dtype_allclose(ia, expected)


class TestMatmulInvalidCases:
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((3, 2), ()),
            ((), (3, 2)),
            ((), ()),
        ],
    )
    def test_zero_dim(self, xp, shape1, shape2):
        x1 = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
        x2 = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)

        with pytest.raises(ValueError):
            xp.matmul(x1, x2)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((3,), (4,)),
            ((2, 3), (4, 5)),
            ((2, 4), (3, 5)),
            ((2, 3), (4,)),
            ((3,), (4, 5)),
            ((2, 2, 3), (2, 4, 5)),
            ((3, 2, 3), (2, 4, 5)),
            ((4, 3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (3, 2, 4)),
        ],
    )
    def test_invalid_shape(self, xp, shape1, shape2):
        x1 = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
        x2 = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)

        with pytest.raises(ValueError):
            xp.matmul(x1, x2)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize(
        "shape1, shape2, out_shape",
        [
            ((5, 4, 3), (3, 1), (3, 4, 1)),
            ((5, 4, 3), (3, 1), (5, 6, 1)),
            ((5, 4, 3), (3, 1), (5, 4, 2)),
            ((5, 4, 3), (3, 1), (4, 1)),
            ((5, 4, 3), (3,), (5, 3)),
            ((5, 4, 3), (3,), (6, 4)),
            ((4,), (3, 4, 5), (4, 5)),
            ((4,), (3, 4, 5), (3, 6)),
        ],
    )
    def test_invalid_shape_out(self, xp, shape1, shape2, out_shape):
        x1 = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
        x2 = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)
        res = xp.empty(out_shape)

        with pytest.raises(ValueError):
            xp.matmul(x1, x2, out=res)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True)[:-2])
    def test_invalid_dtype(self, xp, dtype):
        dpnp_dtype = get_all_dtypes(no_none=True)[-1]
        a1 = xp.arange(5 * 4, dtype=dpnp_dtype).reshape(5, 4)
        a2 = xp.arange(7 * 4, dtype=dpnp_dtype).reshape(4, 7)
        dp_out = xp.empty((5, 7), dtype=dtype)

        with pytest.raises(TypeError):
            xp.matmul(a1, a2, out=dp_out)

    def test_exe_q(self):
        x1 = dpnp.ones((5, 4), sycl_queue=dpctl.SyclQueue())
        x2 = dpnp.ones((4, 7), sycl_queue=dpctl.SyclQueue())
        with pytest.raises(ValueError):
            dpnp.matmul(x1, x2)

        x1 = dpnp.ones((5, 4))
        x2 = dpnp.ones((4, 7))
        out = dpnp.empty((5, 7), sycl_queue=dpctl.SyclQueue())
        with pytest.raises(ExecutionPlacementError):
            dpnp.matmul(x1, x2, out=out)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_matmul_casting(self, xp):
        a1 = xp.arange(2 * 4, dtype=xp.float32).reshape(2, 4)
        a2 = xp.arange(4 * 3).reshape(4, 3)

        res = xp.empty((2, 3), dtype=xp.int64)
        with pytest.raises(TypeError):
            xp.matmul(a1, a2, out=res, casting="safe")

    def test_matmul_not_implemented(self):
        a1 = dpnp.arange(2 * 4).reshape(2, 4)
        a2 = dpnp.arange(4 * 3).reshape(4, 3)

        with pytest.raises(NotImplementedError):
            dpnp.matmul(a1, a2, subok=False)

        with pytest.raises(NotImplementedError):
            signature = (dpnp.float32, dpnp.float32, dpnp.float32)
            dpnp.matmul(a1, a2, signature=signature)

        with pytest.raises(NotImplementedError):
            dpnp.matmul(a1, a2, axis=2)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_matmul_axes(self, xp):
        a1 = xp.arange(120).reshape(2, 5, 3, 4)
        a2 = xp.arange(120).reshape(4, 2, 5, 3)

        # axes must be a list
        axes = ((3, 1), (2, 0), (0, 1))
        with pytest.raises(TypeError):
            xp.matmul(a1, a2, axes=axes)

        # axes must be be a list of three tuples
        axes = [(3, 1), (2, 0)]
        with pytest.raises(ValueError):
            xp.matmul(a1, a2, axes=axes)

        # axes item should be a tuple
        axes = [(3, 1), (2, 0), [0, 1]]
        with pytest.raises(TypeError):
            xp.matmul(a1, a2, axes=axes)

        # axes item should be a tuple with 2 elements
        axes = [(3, 1), (2, 0), (0, 1, 2)]
        with pytest.raises(AxisError):
            xp.matmul(a1, a2, axes=axes)

        # axes must be an integer
        axes = [(3, 1), (2, 0), (0.0, 1)]
        with pytest.raises(TypeError):
            xp.matmul(a1, a2, axes=axes)

        # axes item 2 should be an empty tuple
        a = xp.arange(3)
        axes = [0, 0, 0]
        with pytest.raises(AxisError):
            xp.matmul(a, a, axes=axes)

        a = xp.arange(3 * 4 * 5).reshape(3, 4, 5)
        b = xp.arange(3)
        # list object cannot be interpreted as an integer
        axes = [(1, 0), (0), [0]]
        with pytest.raises(TypeError):
            xp.matmul(a, b, axes=axes)

        # axes item should be a tuple with a single element, or an integer
        axes = [(1, 0), (0), (0, 1)]
        with pytest.raises(AxisError):
            xp.matmul(a, b, axes=axes)


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
