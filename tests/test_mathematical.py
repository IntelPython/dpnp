import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
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
from tests.third_party.cupy import testing

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    get_integer_dtypes,
    has_support_aspect16,
    has_support_aspect64,
)
from .test_umath import (
    _get_numpy_arrays_1in_1out,
    _get_numpy_arrays_2in_1out,
    _get_output_data_type,
)


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
                norm_axis = numpy.core.numeric.normalize_axis_index(
                    axis, a.ndim, "axis"
                )
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
        assert_raises(numpy.AxisError, xp.diff, a, axis=axis)

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
        assert_raises(numpy.AxisError, xp.diff, a, axis=3, prepend=0)
        assert_raises(numpy.AxisError, xp.diff, a, axis=3, append=0)


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
        assert_raises(numpy.AxisError, xp.gradient, x, axis=3)

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


class TestConj:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_conj(self, dtype):
        a = numpy.array(numpy.random.uniform(-5, 5, 20), dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.conj(ia)
        expected = numpy.conj(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_conj_complex(self, dtype):
        x1 = numpy.random.uniform(-5, 5, 20)
        x2 = numpy.random.uniform(-5, 5, 20)
        a = numpy.array(x1 + 1j * x2, dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.conj(ia)
        expected = numpy.conj(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_conj_ndarray(self, dtype):
        a = numpy.array(numpy.random.uniform(-5, 5, 20), dtype=dtype)
        ia = dpnp.array(a)

        result = ia.conj()
        assert result is ia
        assert_dtype_allclose(result, a.conj())

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_conj_complex_ndarray(self, dtype):
        x1 = numpy.random.uniform(-5, 5, 20)
        x2 = numpy.random.uniform(-5, 5, 20)
        a = numpy.array(x1 + 1j * x2, dtype=dtype)
        ia = dpnp.array(a)

        assert_dtype_allclose(ia.conj(), a.conj())

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_conj_out(self, dtype):
        a = numpy.array(numpy.random.uniform(-5, 5, 20), dtype=dtype)
        ia = dpnp.array(a)

        expected = numpy.conj(a)
        dp_out = dpnp.empty(ia.shape, dtype=dtype)
        result = dpnp.conj(ia, out=dp_out)
        assert dp_out is result
        assert_dtype_allclose(result, expected)


class TestRealImag:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_real_imag(self, dtype):
        a = numpy.array(numpy.random.uniform(-5, 5, 20), dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.real(ia)
        assert result is ia
        expected = numpy.real(a)
        assert expected is a
        assert_dtype_allclose(result, expected)

        result = dpnp.imag(ia)
        expected = numpy.imag(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_real_imag_complex(self, dtype):
        x1 = numpy.random.uniform(-5, 5, 20)
        x2 = numpy.random.uniform(-5, 5, 20)
        a = numpy.array(x1 + 1j * x2, dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.real(ia)
        expected = numpy.real(a)
        assert_dtype_allclose(result, expected)

        result = dpnp.imag(ia)
        expected = numpy.imag(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    def test_real_imag_ndarray(self, dtype):
        a = numpy.array(numpy.random.uniform(-5, 5, 20), dtype=dtype)
        ia = dpnp.array(a)

        result = ia.real
        assert result is ia
        assert_dtype_allclose(result, a.real)
        assert_dtype_allclose(ia.imag, a.imag)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_real_imag_complex_ndarray(self, dtype):
        x1 = numpy.random.uniform(-5, 5, 20)
        x2 = numpy.random.uniform(-5, 5, 20)
        a = numpy.array(x1 + 1j * x2, dtype=dtype)
        ia = dpnp.array(a)

        assert_dtype_allclose(ia.real, a.real)
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
        check_type = True if dpnp.issubdtype(dtype, dpnp.floating) else False
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
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_power(self, dtype):
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
        "order_pair", [("C", "C"), ("C", "F"), ("F", "C"), ("F", "F")]
    )
    @pytest.mark.parametrize(
        "shape_pair",
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
    def test_matmul(self, order_pair, shape_pair):
        order1, order2 = order_pair
        shape1, shape2 = shape_pair
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
        "order_pair", [("C", "C"), ("C", "F"), ("F", "C"), ("F", "F")]
    )
    @pytest.mark.parametrize(
        "shape_pair",
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
    def test_matmul_empty(self, order_pair, shape_pair):
        order1, order2 = order_pair
        shape1, shape2 = shape_pair
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
    def test_matmul_bool(self, shape_pair):
        shape1, shape2 = shape_pair
        a1 = numpy.resize(
            numpy.arange(2, dtype=numpy.bool_), numpy.prod(shape1)
        ).reshape(shape1)
        a2 = numpy.resize(
            numpy.arange(2, dtype=numpy.bool_), numpy.prod(shape2)
        ).reshape(shape2)

        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        result = dpnp.matmul(b1, b2)
        expected = numpy.matmul(a1, a2)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
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
    def test_matmul_dtype(self, dtype, shape_pair):
        shape1, shape2 = shape_pair
        a1 = numpy.arange(numpy.prod(shape1)).reshape(shape1)
        a2 = numpy.arange(numpy.prod(shape2)).reshape(shape2)

        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        result = dpnp.matmul(b1, b2, dtype=dtype)
        expected = numpy.matmul(a1, a2, dtype=dtype)
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
        a = numpy.array(
            numpy.random.uniform(-10, 10, 120), dtype=dtype
        ).reshape(2, 5, 3, 4)
        b = numpy.array(
            numpy.random.uniform(-10, 10, 120), dtype=dtype
        ).reshape(4, 2, 5, 3)
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
        a = numpy.array(
            numpy.random.uniform(-10, 10, 120), dtype=dtype
        ).reshape(2, 5, 3, 4)
        b = numpy.array(
            numpy.random.uniform(-10, 10, 120), dtype=dtype
        ).reshape(4, 2, 5, 3)
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

    @pytest.mark.parametrize("dtype1", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "dtype2", get_all_dtypes(no_bool=True, no_none=True)
    )
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
    def test_matmul_dtype_matrix_inout(self, dtype1, dtype2, shape_pair):
        shape1, shape2 = shape_pair
        a1 = numpy.arange(numpy.prod(shape1), dtype=dtype1).reshape(shape1)
        a2 = numpy.arange(numpy.prod(shape2), dtype=dtype1).reshape(shape2)

        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        if dpnp.can_cast(dpnp.result_type(b1, b2), dtype2, casting="same_kind"):
            result = dpnp.matmul(b1, b2, dtype=dtype2)
            expected = numpy.matmul(a1, a2, dtype=dtype2)
            assert_dtype_allclose(result, expected)
        else:
            with pytest.raises(TypeError):
                dpnp.matmul(b1, b2, dtype=dtype2)

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
    def test_matmul_dtype_matrix_inputs(self, dtype1, dtype2, shape_pair):
        shape1, shape2 = shape_pair
        a1 = numpy.arange(numpy.prod(shape1), dtype=dtype1).reshape(shape1)
        a2 = numpy.arange(numpy.prod(shape2), dtype=dtype2).reshape(shape2)

        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        result = dpnp.matmul(b1, b2)
        expected = numpy.matmul(a1, a2)
        assert_dtype_allclose(result, expected)

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
    def test_matmul_order(self, order, shape_pair):
        shape1, shape2 = shape_pair
        a1 = numpy.arange(numpy.prod(shape1)).reshape(shape1)
        a2 = numpy.arange(numpy.prod(shape2)).reshape(shape2)

        b1 = dpnp.asarray(a1)
        b2 = dpnp.asarray(a2)

        result = dpnp.matmul(b1, b2, order=order)
        expected = numpy.matmul(a1, a2, order=order)
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
        # for matmul of 0-D arrays with out keyword,
        # NumPy repeats the data to match the shape
        # of output array
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
        "shape_pair",
        [
            ((5000, 5000, 2, 2), (5000, 5000, 2, 2)),
            ((2, 2), (5000, 5000, 2, 2)),
            ((5000, 5000, 2, 2), (2, 2)),
        ],
    )
    def test_matmul_large(self, shape_pair):
        shape1, shape2 = shape_pair
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        a = numpy.array(numpy.random.uniform(-5, 5, size1)).reshape(shape1)
        b = numpy.array(numpy.random.uniform(-5, 5, size2)).reshape(shape2)
        a_dp = dpnp.asarray(a)
        b_dp = dpnp.asarray(b)

        result = dpnp.matmul(a_dp, b_dp)
        expected = numpy.matmul(a, b)
        assert_dtype_allclose(result, expected, factor=24)


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
    def test_invalid_shape(self, shape_pair):
        for xp in (numpy, dpnp):
            shape1, shape2 = shape_pair
            x1 = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
            x2 = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)
            with pytest.raises(ValueError):
                xp.matmul(x1, x2)

    @pytest.mark.parametrize(
        "shape_pair",
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
    def test_invalid_shape_out(self, shape_pair):
        for xp in (numpy, dpnp):
            shape1, shape2, out_shape = shape_pair
            x1 = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
            x2 = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)
            res = xp.empty(out_shape)
            with pytest.raises(ValueError):
                xp.matmul(x1, x2, out=res)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True)[:-2])
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_none=True)[-1]
        a1 = dpnp.arange(5 * 4, dtype=dpnp_dtype).reshape(5, 4)
        a2 = dpnp.arange(7 * 4, dtype=dpnp_dtype).reshape(4, 7)
        dp_out = dpnp.empty((5, 7), dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.matmul(a1, a2, out=dp_out)

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

    def test_matmul_casting(self):
        a1 = dpnp.arange(2 * 4, dtype=dpnp.float32).reshape(2, 4)
        a2 = dpnp.arange(4 * 3).reshape(4, 3)

        res = dpnp.empty((2, 3), dtype=dpnp.int64)
        with pytest.raises(TypeError):
            dpnp.matmul(a1, a2, out=res, casting="safe")

    def test_matmul_not_implemented(self):
        a1 = dpnp.arange(2 * 4).reshape(2, 4)
        a2 = dpnp.arange(4 * 3).reshape(4, 3)

        with pytest.raises(NotImplementedError):
            dpnp.matmul(a1, a2, subok=False)

        with pytest.raises(NotImplementedError):
            dpnp.matmul(
                a1, a2, signature=(dpnp.float32, dpnp.float32, dpnp.float32)
            )

        def custom_error_callback(err):
            print("Custom error callback triggered with error:", err)

        with pytest.raises(NotImplementedError):
            dpnp.matmul(a1, a2, extobj=[32, 1, custom_error_callback])

        with pytest.raises(NotImplementedError):
            dpnp.matmul(a1, a2, axis=2)

    def test_matmul_axes(self):
        a1 = dpnp.arange(120).reshape(2, 5, 3, 4)
        a2 = dpnp.arange(120).reshape(4, 2, 5, 3)

        # axes must be a list
        axes = ((3, 1), (2, 0), (0, 1))
        with pytest.raises(TypeError):
            dpnp.matmul(a1, a2, axes=axes)

        # axes must be be a list of three tuples
        axes = [(3, 1), (2, 0)]
        with pytest.raises(ValueError):
            dpnp.matmul(a1, a2, axes=axes)

        # axes item should be a tuple
        axes = [(3, 1), (2, 0), [0, 1]]
        with pytest.raises(TypeError):
            dpnp.matmul(a1, a2, axes=axes)

        # axes item should be a tuple with 2 elements
        axes = [(3, 1), (2, 0), (0, 1, 2)]
        with pytest.raises(ValueError):
            dpnp.matmul(a1, a2, axes=axes)

        # axes must be an integer
        axes = [(3, 1), (2, 0), (0.0, 1)]
        with pytest.raises(TypeError):
            dpnp.matmul(a1, a2, axes=axes)

        # axes item 2 should be an empty tuple
        a = dpnp.arange(3)
        axes = [0, 0, 0]
        with pytest.raises(TypeError):
            dpnp.matmul(a, a, axes=axes)

        a = dpnp.arange(3 * 4 * 5).reshape(3, 4, 5)
        b = dpnp.arange(3)
        # list object cannot be interpreted as an integer
        axes = [(1, 0), (0), [0]]
        with pytest.raises(TypeError):
            dpnp.matmul(a, b, axes=axes)

        # axes item should be a tuple with a single element, or an integer
        axes = [(1, 0), (0), (0, 1)]
        with pytest.raises(ValueError):
            dpnp.matmul(a, b, axes=axes)


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
