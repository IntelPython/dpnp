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
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)

import dpnp
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import map_dtype_to_device

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_abs_array,
    get_all_dtypes,
    get_array,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    get_integer_dtypes,
    get_integer_float_dtypes,
    has_support_aspect16,
    has_support_aspect64,
    is_intel_numpy,
    numpy_version,
)
from .third_party.cupy import testing


@pytest.mark.parametrize("deg", [True, False])
class TestAngle:
    def test_angle_bool(self, deg):
        ia = dpnp.array([True, False])
        a = ia.asnumpy()

        expected = numpy.angle(a, deg=deg)
        result = dpnp.angle(ia, deg=deg)

        # In NumPy, for boolean arguments the output data type is always
        # default floating data type. while data type of output in DPNP is
        # determined by Type Promotion Rules.
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    def test_angle(self, dtype, deg):
        ia = dpnp.arange(10, dtype=dtype)
        a = ia.asnumpy()

        expected = numpy.angle(a, deg=deg)
        result = dpnp.angle(ia, deg=deg)

        # For dtype=int8, uint8, NumPy returns float16, but dpnp returns float32
        dt_int8 = dtype in [dpnp.int8, dpnp.uint8]
        assert_dtype_allclose(result, expected, check_only_type_kind=dt_int8)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_angle_complex(self, dtype, deg):
        a = generate_random_numpy_array(10, dtype=dtype)
        ia = dpnp.array(a)

        expected = numpy.angle(a, deg=deg)
        result = dpnp.angle(ia, deg=deg)
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

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_conj_out(self, dtype):
        a = generate_random_numpy_array(20, dtype)
        ia = dpnp.array(a)

        expected = numpy.conj(a)
        iout = dpnp.empty(ia.shape, dtype=dtype)
        result = dpnp.conj(ia, out=iout)
        assert iout is result
        assert_dtype_allclose(result, expected)


class TestClip:
    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    @pytest.mark.parametrize("order", ["C", "F", "A", "K", None])
    def test_clip(self, dtype, order):
        ia = dpnp.asarray([[1, 2, 8], [1, 6, 4], [9, 5, 1]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        result = dpnp.clip(ia, 2, 6, order=order)
        expected = numpy.clip(a, 2, 6, order=order)
        assert_allclose(result, expected)
        assert expected.flags.c_contiguous == result.flags.c_contiguous
        assert expected.flags.f_contiguous == result.flags.f_contiguous

    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    def test_clip_arrays(self, dtype):
        ia = dpnp.asarray([1, 2, 8, 1, 6, 4, 1], dtype=dtype)
        a = dpnp.asnumpy(ia)

        min_v = dpnp.array(2, dtype=dtype)
        max_v = dpnp.array(6, dtype=dtype)

        result = dpnp.clip(ia, min_v, max_v)
        expected = numpy.clip(a, min_v.asnumpy(), max_v.asnumpy())
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    @pytest.mark.parametrize("in_dp", [dpnp, dpt])
    @pytest.mark.parametrize("out_dp", [dpnp, dpt])
    def test_clip_out(self, dtype, in_dp, out_dp):
        a = numpy.array([[1, 2, 8], [1, 6, 4], [9, 5, 1]], dtype=dtype)
        ia = in_dp.asarray(a)

        iout = out_dp.ones(ia.shape, dtype=dtype)
        out = numpy.ones(a.shape, dtype=dtype)

        result = dpnp.clip(ia, 2, 6, out=iout)
        expected = numpy.clip(a, 2, 6, out=out)
        assert_allclose(result, expected)
        assert_allclose(iout, out)
        assert isinstance(result, dpnp_array)

    def test_input_nan(self):
        a = numpy.array([-2.0, numpy.nan, 0.5, 3.0, 0.25, numpy.nan])
        ia = dpnp.array(a)

        result = dpnp.clip(ia, -1, 1)
        expected = numpy.clip(a, -1, 1)
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
        a = numpy.arange(7.0)
        ia = dpnp.asarray(a)

        result = ia.clip(**kwargs)
        expected = a.clip(**kwargs)
        assert_allclose(result, expected)

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
        a = dpnp.asnumpy(a)
        if axis != None:
            return numpy.logaddexp.accumulate(a, axis=axis, dtype=dtype)
        return numpy.logaddexp.accumulate(a.ravel(), dtype=dtype)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
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

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("axis", [None, 2, -1])
    @pytest.mark.parametrize("include_initial", [True, False])
    def test_include_initial(self, dtype, axis, include_initial):
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

    @pytest.mark.parametrize("in_dt", get_integer_float_dtypes())
    @pytest.mark.parametrize("dt", get_all_dtypes(no_bool=True))
    def test_dtype(self, in_dt, dt):
        a = dpnp.ones(100, dtype=in_dt)
        res = dpnp.cumlogsumexp(a, dtype=dt)
        exp = numpy.logaddexp.accumulate(dpnp.asnumpy(a)).astype(res.dtype)

        dtype_list = [dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        tol = 1e-2 if in_dt in dtype_list else 1e-6
        assert_allclose(res, exp, rtol=tol)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "in_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "out_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_out(self, in_dt, out_dt):
        a = numpy.arange(10, 20).reshape(2, 5).astype(dtype=in_dt)
        out = numpy.zeros_like(a, dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        result = dpnp.cumlogsumexp(ia, out=iout, axis=1)
        exp = numpy.logaddexp.accumulate(a, out=out, axis=1)

        assert result is iout
        assert_allclose(result, exp)


class TestCumProdCumSum:
    @pytest.mark.parametrize("func", ["cumprod", "cumsum"])
    @pytest.mark.parametrize(
        "arr, axis",
        [
            pytest.param([1, 2, 10, 11, 6, 5, 4], -1),
            pytest.param([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], 0),
            pytest.param([[1, 2, 3, 4], [5, 6, 7, 9], [10, 3, 4, 5]], -1),
        ],
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_axis(self, func, arr, axis, dtype):
        a = numpy.array(arr, dtype=dtype)
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia, axis=axis)
        expected = getattr(numpy, func)(a, axis=axis)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("func", ["cumprod", "cumsum"])
    @pytest.mark.parametrize("sh", [(10,), (2, 5)])
    @pytest.mark.parametrize(
        "xp_in, xp_out, check",
        [
            pytest.param(dpt, dpt, False),
            pytest.param(dpt, dpnp, True),
            pytest.param(dpnp, dpt, False),
        ],
    )
    def test_usm_ndarray(self, func, sh, xp_in, xp_out, check):
        a = generate_random_numpy_array(sh, low=-5, high=5)
        ia = xp_in.asarray(a)

        result = getattr(dpnp, func)(ia)
        expected = getattr(numpy, func)(a)
        assert_dtype_allclose(result, expected)

        out = numpy.empty(10)
        iout = xp_out.asarray(out)

        result = getattr(dpnp, func)(ia, out=iout)
        expected = getattr(numpy, func)(a, out=out)
        assert_dtype_allclose(result, expected)
        assert (result is iout) is check

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("func", ["cumprod", "cumsum"])
    @pytest.mark.parametrize("in_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dt", get_all_dtypes())
    def test_dtype(self, func, in_dt, dt):
        a = generate_random_numpy_array(5, dtype=in_dt, low=-5, high=5)
        ia = dpnp.array(a)

        expected = getattr(a, func)(dtype=dt)
        result = getattr(ia, func)(dtype=dt)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("func", ["cumprod", "cumsum"])
    @pytest.mark.parametrize("in_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    def test_out(self, func, in_dt, out_dt):
        a = generate_random_numpy_array(5, dtype=in_dt, low=-5, high=5)
        out = numpy.zeros_like(a, dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        expected = getattr(a, func)(out=out)
        result = getattr(ia, func)(out=iout)
        assert result is iout
        assert_allclose(result, expected, rtol=1e-06)

    @testing.with_requires("numpy>=2.1.0")
    @pytest.mark.parametrize("func", ["cumulative_prod", "cumulative_sum"])
    def test_include_initial(self, func):
        a = numpy.arange(8).reshape(2, 2, 2)
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=1, include_initial=True)
        result = getattr(dpnp, func)(ia, axis=1, include_initial=True)
        assert_array_equal(result, expected)

        expected = getattr(numpy, func)(a, axis=0, include_initial=True)
        result = getattr(dpnp, func)(ia, axis=0, include_initial=True)
        assert_array_equal(result, expected)

        a = numpy.arange(1, 5).reshape(2, 2)
        ia = dpnp.array(a)
        out = numpy.zeros((3, 2), dtype=numpy.float32)
        iout = dpnp.array(out)

        expected = getattr(numpy, func)(
            a, axis=0, out=out, include_initial=True
        )
        result = getattr(dpnp, func)(ia, axis=0, out=iout, include_initial=True)
        assert result is iout
        assert_array_equal(result, expected)

        a = numpy.array([2, 2])
        ia = dpnp.array(a)
        expected = getattr(numpy, func)(a, include_initial=True)
        result = getattr(dpnp, func)(ia, include_initial=True)
        assert_array_equal(result, expected)


class TestDiff:
    @pytest.mark.parametrize("n", list(range(0, 3)))
    @pytest.mark.parametrize("dt", get_integer_dtypes())
    def test_basic_integer(self, n, dt):
        x = [1, 4, 6, 7, 12]
        a = numpy.array(x, dtype=dt)
        ia = dpnp.array(x, dtype=dt)

        expected = numpy.diff(a, n=n)
        result = dpnp.diff(ia, n=n)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_basic_floating(self, dt):
        x = [1.1, 2.2, 3.0, -0.2, -0.1]
        a = numpy.array(x, dtype=dt)
        ia = dpnp.array(x, dtype=dt)

        expected = numpy.diff(a)
        result = dpnp.diff(ia)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("n", [1, 2])
    def test_basic_boolean(self, n):
        x = [True, True, False, False]
        a = numpy.array(x)
        ia = dpnp.array(x)

        expected = numpy.diff(a, n=n)
        result = dpnp.diff(ia, n=n)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_basic_complex(self, dt):
        x = [1.1 + 1j, 2.2 + 4j, 3.0 + 6j, -0.2 + 7j, -0.1 + 12j]
        a = numpy.array(x, dtype=dt)
        ia = dpnp.array(x, dtype=dt)

        expected = numpy.diff(a)
        result = dpnp.diff(ia)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None] + list(range(-3, 2)))
    def test_axis(self, axis):
        a = numpy.zeros((10, 20, 30))
        a[:, 1::2, :] = 1
        ia = dpnp.array(a)

        kwargs = {} if axis is None else {"axis": axis}
        expected = numpy.diff(a, **kwargs)
        result = dpnp.diff(ia, **kwargs)
        assert_array_equal(result, expected)

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
        a = 20 * numpy.random.rand(10, 20, 30)
        ia = dpnp.array(a)

        kwargs = {} if n is None else {"n": n}
        if axis is not None:
            kwargs.update({"axis": axis})

        expected = numpy.diff(a, **kwargs)
        result = dpnp.diff(ia, **kwargs)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("n", list(range(0, 5)))
    def test_n(self, n):
        a = numpy.array(list(range(3)))
        ia = dpnp.array(a)

        expected = numpy.diff(a, n=n)
        result = dpnp.diff(ia, n=n)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_n_error(self, xp):
        a = xp.array(list(range(3)))
        assert_raises(ValueError, xp.diff, a, n=-1)

    @pytest.mark.parametrize("prepend", [0, [0], [-1, 0]])
    def test_prepend(self, prepend):
        a = numpy.arange(5) + 1
        ia = dpnp.array(a)

        np_p = prepend if numpy.isscalar(prepend) else numpy.array(prepend)
        dpnp_p = prepend if dpnp.isscalar(prepend) else dpnp.array(prepend)

        expected = numpy.diff(a, prepend=np_p)
        result = dpnp.diff(ia, prepend=dpnp_p)
        assert_array_equal(result, expected)

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
        a = numpy.arange(4).reshape(2, 2)
        ia = dpnp.array(a)

        np_p = prepend if numpy.isscalar(prepend) else numpy.array(prepend)
        dpnp_p = prepend if dpnp.isscalar(prepend) else dpnp.array(prepend)

        expected = numpy.diff(a, axis=axis, prepend=np_p)
        result = dpnp.diff(ia, axis=axis, prepend=dpnp_p)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("append", [0, [0], [0, 2]])
    def test_append(self, append):
        a = numpy.arange(5)
        ia = dpnp.array(a)

        ap = append if numpy.isscalar(append) else numpy.array(append)
        iap = append if dpnp.isscalar(append) else dpnp.array(append)

        expected = numpy.diff(a, append=ap)
        result = dpnp.diff(ia, append=iap)
        assert_array_equal(result, expected)

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
        a = numpy.arange(4).reshape(2, 2)
        ia = dpnp.array(a)

        ap = append if numpy.isscalar(append) else numpy.array(append)
        iap = append if dpnp.isscalar(append) else dpnp.array(append)

        expected = numpy.diff(a, axis=axis, append=ap)
        result = dpnp.diff(ia, axis=axis, append=iap)
        assert_array_equal(result, expected)

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
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
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
    def test_basic(self, array, dtype):
        a = numpy.array(array, dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.ediff1d(ia)
        expected = numpy.ediff1d(a)
        assert_array_equal(result, expected)

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
    def test_to_begin(self, to_begin):
        a = numpy.array([1, 2, 4, 7, 0])
        ia = dpnp.array(a)
        np_to_begin = get_array(numpy, to_begin)

        result = dpnp.ediff1d(ia, to_begin=to_begin)
        expected = numpy.ediff1d(a, to_begin=np_to_begin)
        assert_array_equal(result, expected)

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
    def test_to_end(self, to_end):
        a = numpy.array([1, 2, 4, 7, 0])
        ia = dpnp.array(a)
        np_to_end = get_array(numpy, to_end)

        result = dpnp.ediff1d(ia, to_end=to_end)
        expected = numpy.ediff1d(a, to_end=np_to_end)
        assert_array_equal(result, expected)

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
    def test_to_begin_to_end(self, to_begin, to_end):
        a = numpy.array([1, 2, 4, 7, 0])
        ia = dpnp.array(a)

        np_to_begin = get_array(numpy, to_begin)
        np_to_end = get_array(numpy, to_end)

        result = dpnp.ediff1d(ia, to_end=to_end, to_begin=to_begin)
        expected = numpy.ediff1d(a, to_end=np_to_end, to_begin=np_to_begin)
        assert_array_equal(result, expected)

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
    def test_usm_ndarray(self, to_begin, to_end):
        a = numpy.array([[1, 2, 0]])
        dpt_a = dpt.asarray(a)

        if isinstance(to_begin, dpt.usm_ndarray):
            np_to_begin = dpt.asnumpy(to_begin)
        else:
            np_to_begin = to_begin

        if isinstance(to_end, dpt.usm_ndarray):
            np_to_end = dpt.asnumpy(to_end)
        else:
            np_to_end = to_end

        result = dpnp.ediff1d(dpt_a, to_end=to_end, to_begin=to_begin)
        expected = numpy.ediff1d(a, to_end=np_to_end, to_begin=np_to_begin)

        assert_array_equal(result, expected)
        assert isinstance(result, dpnp.ndarray)

    def test_errors(self):
        ia = dpnp.array([[1, 2], [2, 5]])

        # unsupported type
        a = dpnp.asnumpy(ia)
        assert_raises(TypeError, dpnp.ediff1d, a)

        # unsupported `to_begin` type according to the `same_kind` rules
        to_begin = dpnp.array([-5], dtype="f4")
        assert_raises(TypeError, dpnp.ediff1d, ia, to_begin=to_begin)

        # unsupported `to_end` type according to the `same_kind` rules
        to_end = dpnp.array([5], dtype="f4")
        assert_raises(TypeError, dpnp.ediff1d, ia, to_end=to_end)

        # another `to_begin` sycl queue
        to_begin = dpnp.array([-20, -15], sycl_queue=dpctl.SyclQueue())
        assert_raises(
            ExecutionPlacementError, dpnp.ediff1d, ia, to_begin=to_begin
        )

        # another `to_end` sycl queue
        to_end = dpnp.array([15, 20], sycl_queue=dpctl.SyclQueue())
        assert_raises(ExecutionPlacementError, dpnp.ediff1d, ia, to_end=to_end)


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


@pytest.mark.parametrize("dtype1", get_all_dtypes(no_none=True))
@pytest.mark.parametrize("dtype2", get_all_dtypes(no_none=True))
@pytest.mark.parametrize(
    "func", ["add", "divide", "multiply", "power", "subtract"]
)
def test_op_multiple_dtypes(dtype1, func, dtype2):
    a = numpy.array([[1, 2], [3, 4]], dtype=dtype1)
    b = numpy.array([[1, 2], [3, 4]], dtype=dtype2)
    ia, ib = dpnp.array(a), dpnp.array(b)

    if func == "subtract" and (dtype1 == dtype2 == dpnp.bool):
        with pytest.raises(TypeError):
            result = getattr(dpnp, func)(ia, ib)
            expected = getattr(numpy, func)(a, b)
    else:
        result = getattr(dpnp, func)(ia, ib)
        expected = getattr(numpy, func)(a, b)
        assert_allclose(result, expected)


class TestI0:
    def test_0d(self):
        a = dpnp.array(0.5)
        na = a.asnumpy()
        assert_dtype_allclose(dpnp.i0(a), numpy.i0(na))

    @pytest.mark.parametrize("dt", get_integer_float_dtypes())
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


class TestInterp:
    @pytest.mark.parametrize(
        "dtype_x", get_all_dtypes(no_complex=True, no_none=True)
    )
    @pytest.mark.parametrize(
        "dtype_xp", get_all_dtypes(no_complex=True, no_none=True)
    )
    @pytest.mark.parametrize("dtype_y", get_all_dtypes(no_none=True))
    def test_all_dtypes(self, dtype_x, dtype_xp, dtype_y):
        x = numpy.linspace(0.1, 9.9, 20).astype(dtype_x)
        xp = numpy.linspace(0.0, 10.0, 5).astype(dtype_xp)
        fp = (xp * 1.5 + 1).astype(dtype_y)

        ix = dpnp.array(x)
        ixp = dpnp.array(xp)
        ifp = dpnp.array(fp)

        expected = numpy.interp(x, xp, fp)
        result = dpnp.interp(ix, ixp, ifp)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype_x", get_all_dtypes(no_complex=True, no_none=True)
    )
    @pytest.mark.parametrize("dtype_y", get_complex_dtypes())
    def test_complex_fp(self, dtype_x, dtype_y):
        x = numpy.array([0.25, 0.75], dtype=dtype_x)
        xp = numpy.array([0.0, 1.0], dtype=dtype_x)
        fp = numpy.array([1 + 1j, 3 + 3j], dtype=dtype_y)

        ix = dpnp.array(x)
        ixp = dpnp.array(xp)
        ifp = dpnp.array(fp)

        expected = numpy.interp(x, xp, fp)
        result = dpnp.interp(ix, ixp, ifp)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)
    )
    @pytest.mark.parametrize(
        "left, right", [[-40, 40], [dpnp.array(-40), dpnp.array(40)]]
    )
    def test_left_right_args(self, dtype, left, right):
        x = numpy.array([0, 1, 2, 3, 4, 5, 6], dtype=dtype)
        xp = numpy.array([0, 3, 6], dtype=dtype)
        fp = numpy.array([0, 9, 18], dtype=dtype)

        ix = dpnp.array(x)
        ixp = dpnp.array(xp)
        ifp = dpnp.array(fp)

        expected = numpy.interp(
            x,
            xp,
            fp,
            left=get_array(numpy, left),
            right=get_array(numpy, right),
        )
        result = dpnp.interp(ix, ixp, ifp, left=left, right=right)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("val", [numpy.nan, numpy.inf, -numpy.inf])
    def test_naninf(self, val):
        x = numpy.array([0, 1, 2, val])
        xp = numpy.array([0, 1, 2])
        fp = numpy.array([10, 20, 30])

        ix = dpnp.array(x)
        ixp = dpnp.array(xp)
        ifp = dpnp.array(fp)

        expected = numpy.interp(x, xp, fp)
        result = dpnp.interp(ix, ixp, ifp)
        assert_dtype_allclose(result, expected)

    def test_empty_x(self):
        x = numpy.array([])
        xp = numpy.array([0, 1])
        fp = numpy.array([10, 20])

        ix = dpnp.array(x)
        ixp = dpnp.array(xp)
        ifp = dpnp.array(fp)

        expected = numpy.interp(x, xp, fp)
        result = dpnp.interp(ix, ixp, ifp)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_period(self, dtype):
        x = numpy.array([-180, 0, 180], dtype=dtype)
        xp = numpy.array([-90, 0, 90], dtype=dtype)
        fp = numpy.array([0, 1, 0], dtype=dtype)

        ix = dpnp.array(x)
        ixp = dpnp.array(xp)
        ifp = dpnp.array(fp)

        expected = numpy.interp(x, xp, fp, period=180)
        result = dpnp.interp(ix, ixp, ifp, period=180)
        assert_dtype_allclose(result, expected)

    def test_errors(self):
        x = dpnp.array([0.5])

        # xp and fp have different lengths
        xp = dpnp.array([0])
        fp = dpnp.array([1, 2])
        assert_raises(ValueError, dpnp.interp, x, xp, fp)

        # xp is not 1D
        xp = dpnp.array([[0, 1]])
        fp = dpnp.array([1, 2])
        assert_raises(ValueError, dpnp.interp, x, xp, fp)

        # fp is not 1D
        xp = dpnp.array([0, 1])
        fp = dpnp.array([[1, 2]])
        assert_raises(ValueError, dpnp.interp, x, xp, fp)

        # xp and fp are empty
        xp = dpnp.array([])
        fp = dpnp.array([])
        assert_raises(ValueError, dpnp.interp, x, xp, fp)

        # x complex
        x_complex = dpnp.array([1 + 2j])
        xp = dpnp.array([0.0, 2.0])
        fp = dpnp.array([0.0, 1.0])
        assert_raises(TypeError, dpnp.interp, x_complex, xp, fp)

        # xp complex
        xp_complex = dpnp.array([0 + 1j, 2 + 1j])
        assert_raises(TypeError, dpnp.interp, x, xp_complex, fp)

        # period is zero
        x = dpnp.array([1.0])
        xp = dpnp.array([0.0, 2.0])
        fp = dpnp.array([0.0, 1.0])
        assert_raises(ValueError, dpnp.interp, x, xp, fp, period=0)

        # period is not scalar or 0-dim
        assert_raises(TypeError, dpnp.interp, x, xp, fp, period=[180])

        # left is not scalar or 0-dim
        left = [1]
        assert_raises(TypeError, dpnp.interp, x, xp, fp, left=left)

        # left is 1-d array
        left = dpnp.array([1.0])
        assert_raises(ValueError, dpnp.interp, x, xp, fp, left=left)

        # left has a different SYCL queue
        q1 = dpctl.SyclQueue()
        q2 = dpctl.SyclQueue()
        left = dpnp.array(1.0, sycl_queue=q2)
        if q1 != q2:
            assert_raises(ValueError, dpnp.interp, x, xp, fp, left=left)


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
        ia = self.array_or_scalar(dpnp, lhs, dtype=dtype)
        ib = self.array_or_scalar(dpnp, rhs, dtype=dtype)

        a = self.array_or_scalar(numpy, lhs, dtype=dtype)
        b = self.array_or_scalar(numpy, rhs, dtype=dtype)

        if (
            name == "subtract"
            and not numpy.isscalar(rhs)
            and dtype == dpnp.bool
        ):
            with pytest.raises(TypeError):
                result = getattr(dpnp, name)(ia, ib)
                expected = getattr(numpy, name)(a, b)
        else:
            result = getattr(dpnp, name)(ia, ib)
            expected = getattr(numpy, name)(a, b)
            assert_dtype_allclose(result, expected, check_type)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_add(self, dtype, lhs, rhs):
        self._test_mathematical("add", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_arctan2(self, dtype, lhs, rhs):
        self._test_mathematical("arctan2", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    def test_copysign(self, dtype, lhs, rhs):
        self._test_mathematical("copysign", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_divide(self, dtype, lhs, rhs):
        self._test_mathematical("divide", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_fmax(self, dtype, lhs, rhs):
        self._test_mathematical("fmax", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_fmin(self, dtype, lhs, rhs):
        self._test_mathematical("fmin", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
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

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_floor_divide(self, dtype, lhs, rhs):
        if dtype == dpnp.float32 and rhs == 0.3:
            pytest.skip(
                "In this case, a different result, but similar to xp.floor(xp.divide(lhs, rhs)."
            )
        self._test_mathematical(
            "floor_divide", dtype, lhs, rhs, check_type=False
        )

    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    def test_hypot(self, dtype, lhs, rhs):
        self._test_mathematical("hypot", dtype, lhs, rhs)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_maximum(self, dtype, lhs, rhs):
        self._test_mathematical("maximum", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_minimum(self, dtype, lhs, rhs):
        self._test_mathematical("minimum", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_multiply(self, dtype, lhs, rhs):
        self._test_mathematical("multiply", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_remainder(self, dtype, lhs, rhs):
        if (
            dpnp.issubdtype(dtype, dpnp.integer)
            or dtype is None
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

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_power(self, dtype, lhs, rhs):
        self._test_mathematical("power", dtype, lhs, rhs, check_type=False)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_subtract(self, dtype, lhs, rhs):
        self._test_mathematical("subtract", dtype, lhs, rhs, check_type=False)


class TestNanToNum:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("shape", [(3,), (2, 3), (3, 2, 2)])
    def test_basic(self, dtype, shape):
        a = generate_random_numpy_array(shape, dtype=dtype)
        if not dpnp.issubdtype(dtype, dpnp.integer):
            a.flat[1] = numpy.nan
        ia = dpnp.array(a)

        result = dpnp.nan_to_num(ia)
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
        a = dpnp.asnumpy(ia)
        assert_raises(TypeError, dpnp.nan_to_num, a)

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
    def test_strided(self, copy, dt):
        n = 10
        dt = numpy.dtype(dt)
        a = numpy.arange(2 * n, dtype=dt)
        ia = dpnp.arange(2 * n, dtype=dt)
        if dt.kind in "fc":
            a[::4] = numpy.nan
            ia[::4] = dpnp.nan
        result = dpnp.nan_to_num(ia[::-2], copy=copy, nan=57.0)
        expected = numpy.nan_to_num(a[::-2], copy=copy, nan=57.0)

        assert_dtype_allclose(result, expected)


class TestProd:
    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
    @pytest.mark.parametrize("keepdims", [False, True])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, axis, keepdims, dtype):
        a = generate_random_numpy_array((2, 2, 3), dtype=dtype, low=-5, high=5)
        ia = dpnp.array(a)

        expected = numpy.prod(a, axis=axis, keepdims=keepdims)
        result = dpnp.prod(ia, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
    def test_zero_size(self, axis):
        a = numpy.empty((2, 3, 0))
        ia = dpnp.array(a)

        expected = numpy.prod(a, axis=axis)
        result = dpnp.prod(ia, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize("in_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_dtype(self, in_dt, dt):
        a = generate_random_numpy_array((2, 2, 3), dtype=in_dt, low=-5, high=5)
        ia = dpnp.array(a)

        expected = numpy.prod(a, dtype=dt)
        result = dpnp.prod(ia, dtype=dt)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_overflow_encountered_in_cast_numpy_warnings"
    )
    def test_out(self):
        ia = dpnp.arange(1, 7).reshape((2, 3))
        ia = ia.astype(dpnp.default_float_type(ia.device))
        a = dpnp.asnumpy(ia)

        # output is dpnp_array
        expected = numpy.prod(a, axis=0)
        iout = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = dpnp.prod(ia, axis=0, out=iout)
        assert iout is result
        assert_allclose(result, expected)

        # output is usm_ndarray
        dpt_out = dpt.empty(expected.shape, dtype=expected.dtype)
        result = dpnp.prod(ia, axis=0, out=dpt_out)
        assert dpt_out is result.get_array()
        assert_allclose(result, expected)

        # out is a numpy array -> TypeError
        result = numpy.empty_like(expected)
        with pytest.raises(TypeError):
            dpnp.prod(ia, axis=0, out=result)

        # incorrect shape for out
        result = dpnp.array(numpy.empty((2, 3)))
        with pytest.raises(ValueError):
            dpnp.prod(ia, axis=0, out=result)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("in_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    def test_out_dtype(self, in_dt, out_dt):
        a = generate_random_numpy_array((2, 3), dtype=in_dt, low=-5, high=5)
        out = numpy.zeros_like(a, shape=(2,), dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        result = dpnp.prod(ia, out=iout, axis=1)
        expected = numpy.prod(a, out=out, axis=1)
        assert_allclose(result, expected, rtol=1e-06)
        assert result is iout

    def test_error(self):
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
        a = generate_random_numpy_array(10, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.real_if_close(ia + 1e-15j)
        expected = numpy.real_if_close(a + 1e-15j)
        assert_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_singlecomplex(self, dt):
        a = generate_random_numpy_array(10, dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.real_if_close(ia + 1e-7j)
        expected = numpy.real_if_close(a + 1e-7j)
        assert_equal(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    def test_tol(self, dt):
        a = generate_random_numpy_array(10, dtype=dt)
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
        if is_intel_numpy():
            assert_allclose(result, expected)
        else:
            # numpy.spacing(-0.0) == numpy.spacing(0.0), i.e. the stock NumPy
            # returns positive value (looks as a bug), because for any other
            # negative input the NumPy result will be also a negative value.
            expected[1] *= -1
            assert_allclose(result, expected)

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

    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True, no_bool=True))
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
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("y_dt", get_all_dtypes(no_none=True, no_bool=True))
    @pytest.mark.parametrize("x_dt", get_all_dtypes(no_none=True, no_bool=True))
    @pytest.mark.parametrize("y_arr", [[1, 2, 4, 5], [1.0, 2.5, 6.0, 7.0]])
    @pytest.mark.parametrize("x_arr", [[2, 5, 6, 9]])
    def test_x_samples(self, y_arr, x_arr, y_dt, x_dt):
        y = numpy.array(y_arr, dtype=y_dt)
        x = numpy.array(x_arr, dtype=x_dt)
        iy, ix = dpnp.array(y), dpnp.array(x)

        result = dpnp.trapezoid(iy, ix)
        expected = self.get_numpy_func()(y, x)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("data", [[1, 2, 3], [4, 5, 6]])
    def test_2d_with_x_samples(self, data):
        a = numpy.array(data)
        ia = dpnp.array(a)

        result = dpnp.trapezoid(ia, ia)
        expected = self.get_numpy_func()(a, a)
        assert_array_equal(result, expected)

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
        assert_array_equal(result, expected)

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

    @pytest.mark.parametrize("dt", get_integer_float_dtypes(no_unsigned=True))
    def test_period(self, dt):
        a = numpy.array([1, 1 + 108], dtype=dt)
        ia = dpnp.array(a)

        # unwrap removes jumps greater than 107
        result = dpnp.unwrap(ia, period=107)
        expected = numpy.unwrap(a, period=107)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dt", get_integer_float_dtypes(no_unsigned=True))
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

    @pytest.mark.parametrize("dt", get_integer_float_dtypes(no_unsigned=True))
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
@pytest.mark.parametrize("val_type", [bool, int, float])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
@pytest.mark.parametrize(
    "func", ["add", "divide", "multiply", "power", "subtract"]
)
@pytest.mark.parametrize("val", [0, 1, 5])
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
    ids=["2D-zeros", "2D-repetitive", "2D", "3D", "4D"],
)
def test_op_with_scalar(array, val, func, dtype, val_type):
    a = numpy.array(array, dtype=dtype)
    ia = dpnp.array(a)
    val_ = val_type(val)

    if func == "power":
        if (
            val_ == 0
            and numpy.issubdtype(dtype, numpy.complexfloating)
            and not dpnp.all(ia)
        ):
            pytest.skip(
                "(0j ** 0) is different: (NaN + NaNj) in dpnp and (1 + 0j) in numpy"
            )

    if func == "subtract" and val_type == bool and dtype == dpnp.bool:
        with pytest.raises(TypeError):
            result = getattr(dpnp, func)(ia, val_)
            expected = getattr(numpy, func)(a, val_)

            result = getattr(dpnp, func)(val_, ia)
            expected = getattr(numpy, func)(val_, a)
    else:
        result = getattr(dpnp, func)(ia, val_)
        expected = getattr(numpy, func)(a, val_)
        assert_allclose(result, expected, rtol=1e-06)

        result = getattr(dpnp, func)(val_, ia)
        expected = getattr(numpy, func)(val_, a)
        assert_allclose(result, expected)


@pytest.mark.parametrize("shape", [(), (3, 2)], ids=["0D", "2D"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
def test_multiply_scalar(shape, dtype):
    a = numpy.ones(shape, dtype=dtype)
    ia = dpnp.array(a)

    result = 0.5 * ia * 1.7
    expected = 0.5 * a * 1.7
    assert_allclose(result, expected)


@pytest.mark.parametrize("shape", [(), (3, 2)], ids=["0D", "2D"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
def test_add_scalar(shape, dtype):
    a = numpy.ones(shape, dtype=dtype)
    ia = dpnp.array(a)

    result = 0.5 + ia + 1.7
    expected = 0.5 + a + 1.7
    assert_allclose(result, expected)


@pytest.mark.parametrize("shape", [(), (3, 2)], ids=["0D", "2D"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
def test_subtract_scalar(shape, dtype):
    a = numpy.ones(shape, dtype=dtype)
    ia = dpnp.array(a)

    result = 0.5 - ia - 1.7
    expected = 0.5 - a - 1.7
    assert_allclose(result, expected)


@pytest.mark.parametrize("shape", [(), (3, 2)], ids=["0D", "2D"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
def test_divide_scalar(shape, dtype):
    a = numpy.ones(shape, dtype=dtype)
    ia = dpnp.array(a)

    result = 0.5 / ia / 1.7
    expected = 0.5 / a / 1.7
    assert_allclose(result, expected, rtol=1e-06)


@pytest.mark.parametrize(
    "data", [[[1.0, -1.0], [0.1, -0.1]], [-2, -1, 0, 1, 2]], ids=["2D", "1D"]
)
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_none=True, no_bool=True, no_unsigned=True)
)
def test_negative(data, dtype):
    a = numpy.array(data, dtype=dtype)
    ia = dpnp.array(a)

    result = dpnp.negative(ia)
    expected = numpy.negative(a)
    assert_allclose(result, expected)

    result = -ia
    expected = -a
    assert_dtype_allclose(result, expected)

    # out keyword
    if dtype is not None:
        iout = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.negative(ia, out=iout)
        assert result is iout
        assert_allclose(result, expected)


def test_negative_boolean():
    ia = dpnp.array([True, False])

    with pytest.raises(TypeError):
        dpnp.negative(ia)


@pytest.mark.parametrize(
    "data", [[[1.0, -1.0], [0.1, -0.1]], [-2, -1, 0, 1, 2]], ids=["2D", "1D"]
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
def test_positive(data, dtype):
    a = get_abs_array(data, dtype=dtype)
    ia = dpnp.array(a)

    result = dpnp.positive(ia)
    expected = numpy.positive(a)
    assert_allclose(result, expected)

    result = +ia
    expected = +a
    assert_allclose(result, expected)

    # out keyword
    if dtype is not None:
        iout = dpnp.empty(expected.shape, dtype=dtype)
        result = dpnp.positive(ia, out=iout)
        assert result is iout
        assert_allclose(result, expected)


def test_positive_boolean():
    ia = dpnp.array([True, False])

    with pytest.raises(TypeError):
        dpnp.positive(ia)


@pytest.mark.parametrize("dtype", get_float_dtypes(no_float16=False))
def test_float_remainder_magnitude(dtype):
    b = numpy.array(1.0, dtype=dtype)
    a = numpy.nextafter(numpy.array(0.0, dtype=dtype), -b)
    ia, ib = dpnp.array(a), dpnp.array(b)

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
    ia, ib = dpnp.array(a), dpnp.array(b)

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
    "dtype", get_all_dtypes(no_none=True, no_complex=True, no_unsigned=True)
)
def test_signbit(data, dtype):
    a = numpy.array(data, dtype=dtype)
    ia = dpnp.array(a)

    result = dpnp.signbit(ia)
    expected = numpy.signbit(a)
    assert_dtype_allclose(result, expected)

    # out keyword
    iout = dpnp.empty(expected.shape, dtype=expected.dtype)
    result = dpnp.signbit(ia, out=iout)
    assert iout is result
    assert_dtype_allclose(result, expected)


class TestRealImag:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
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
        assert dpnp.allclose(result, expected)

        # out keyword
        iout = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = dpnp.proj(a, out=iout)
        assert iout is result
        assert dpnp.allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_projection(self, dtype):
        result = dpnp.proj(dpnp.array(1, dtype=dtype))
        expected = dpnp.array(complex(1, 0))
        assert dpnp.allclose(result, expected)


@pytest.mark.parametrize("func", ["ceil", "floor", "trunc", "fix"])
class TestRoundingFuncs:
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_basic(self, func, dt):
        a = generate_random_numpy_array((2, 4), dt)
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia)
        expected = getattr(numpy, func)(a)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_complex(self, func, xp, dt):
        a = xp.array([1.1, -1.1], dtype=dt)
        with pytest.raises((ValueError, TypeError)):
            getattr(xp, func)(a)

    @testing.with_requires("numpy>=2.1.0")
    @pytest.mark.parametrize(
        "dt_in", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "dt_out", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_out(self, func, dt_in, dt_out):
        a = generate_random_numpy_array(10, dt_in)
        out = numpy.empty(a.shape, dtype=dt_out)
        ia, iout = dpnp.array(a), dpnp.array(out)

        if dt_in != dt_out:
            if numpy.can_cast(dt_in, dt_out, casting="same_kind"):
                # NumPy allows "same_kind" casting, dpnp does not
                assert_raises(ValueError, getattr(dpnp, func), ia, out=iout)
            else:
                assert_raises(ValueError, getattr(dpnp, func), ia, out=iout)
                assert_raises(TypeError, getattr(numpy, func), a, out=out)
        else:
            expected = getattr(numpy, func)(a, out=out)
            result = getattr(dpnp, func)(ia, out=iout)
            assert result is iout
            assert_array_equal(result, expected)

    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
    def test_out_float16(self, func):
        a = generate_random_numpy_array((4, 2), numpy.float16)
        out = numpy.zeros_like(a, dtype=numpy.float16)
        ia, iout = dpnp.array(a), dpnp.array(out)

        result = getattr(dpnp, func)(ia, out=iout)
        expected = getattr(numpy, func)(a, out=out)
        assert result is iout
        assert_array_equal(result, expected)

    @testing.with_requires("numpy>=2.1.0")
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_out_usm_ndarray(self, func, dt):
        a = generate_random_numpy_array(10, dt)
        out = numpy.empty(a.shape, dtype=dt)
        ia, usm_out = dpnp.array(a), dpt.asarray(out)

        expected = getattr(numpy, func)(a, out=out)
        result = getattr(dpnp, func)(ia, out=usm_out)
        assert result.get_array() is usm_out
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15,)", "(2, 2)"]
    )
    def test_invalid_shape(self, func, xp, shape):
        a = xp.arange(10, dtype=xp.float32)
        out = xp.empty(shape, dtype=xp.float32)
        assert_raises(ValueError, getattr(xp, func), a, out=out)

    def test_error(self, func):
        # scalar, unsupported input
        assert_raises(TypeError, getattr(dpnp, func), -3.4)

        # unsupported out
        a = dpnp.array([1, 2, 3])
        out = numpy.empty_like(3, dtype=a.dtype)
        assert_raises(TypeError, getattr(dpnp, func), a, out=out)


class TestHypot:
    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    def test_hypot(self, dtype):
        a = generate_random_numpy_array(10, dtype, low=0)
        b = generate_random_numpy_array(10, dtype, low=0)
        expected = numpy.hypot(a, b)

        ia, ib = dpnp.array(a), dpnp.array(b)
        out_dt = map_dtype_to_device(expected.dtype, ia.sycl_device)
        iout = dpnp.empty(expected.shape, dtype=out_dt)
        result = dpnp.hypot(ia, ib, out=iout)

        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_out_overlap(self, dtype):
        size = 15
        ia = dpnp.arange(2 * size, dtype=dtype)
        dpnp.hypot(ia[size::], ia[::2], out=ia[:size:])

        a = numpy.arange(2 * size, dtype=dtype)
        numpy.hypot(a[size::], a[::2], out=a[:size:])

        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15,)", "(2, 2)"]
    )
    def test_invalid_shape(self, shape):
        ia = dpnp.arange(10)
        iout = dpnp.empty(shape)
        assert_raises(ValueError, dpnp.hypot, ia, ia, iout)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["scalar", "empty_tuple", "empty_list", "tuple", "list"],
    )
    def test_invalid_out(self, xp, out):
        a = xp.arange(10)
        assert_raises(TypeError, xp.hypot, a, 2, out)


class TestLogSumExp:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("axis", [None, 2, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic(self, dtype, axis, keepdims):
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

    @pytest.mark.parametrize("in_dt", get_integer_float_dtypes())
    @pytest.mark.parametrize("dt", get_all_dtypes(no_bool=True))
    def test_dtype(self, in_dt, dt):
        a = dpnp.ones(100, dtype=in_dt)
        res = dpnp.logsumexp(a, dtype=dt)
        exp = numpy.logaddexp.reduce(dpnp.asnumpy(a)).astype(res.dtype)

        dtype_list = [dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        tol = 1e-2 if in_dt in dtype_list else 1e-6
        assert_allclose(res, exp, rtol=tol)

    @testing.with_requires("numpy>=1.26.4")
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "in_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "out_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_out(self, in_dt, out_dt):
        a = numpy.arange(1, 11).reshape(2, 5).astype(dtype=in_dt)
        out = numpy.zeros_like(a, shape=(2,), dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        result = dpnp.logsumexp(ia, out=iout, axis=1)
        assert result is iout
        if numpy.issubdtype(out_dt, numpy.uint64):
            # NumPy returns incorrect results for this case if out kwarg is used
            exp = numpy.logaddexp.reduce(a, axis=1).astype(out_dt)
        else:
            exp = numpy.logaddexp.reduce(a, out=out, axis=1)

        dtype_list = [dpnp.int8, dpnp.uint8, dpnp.int16, dpnp.uint16]
        tol = 1e-2 if in_dt in dtype_list else 1e-6
        assert_allclose(result, exp, rtol=tol)


class TestReduceHypot:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("axis", [None, 2, -1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic(self, dtype, axis, keepdims):
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

    @pytest.mark.parametrize("in_dt", get_integer_float_dtypes())
    @pytest.mark.parametrize("dt", get_all_dtypes(no_bool=True))
    def test_dtype(self, in_dt, dt):
        a = dpnp.ones(99, dtype=in_dt)
        res = dpnp.reduce_hypot(a, dtype=dt)
        exp = numpy.hypot.reduce(dpnp.asnumpy(a)).astype(res.dtype)

        dtype_list = [dpnp.int8, dpnp.uint8]
        tol = 1e-2 if in_dt in dtype_list else 1e-6
        assert_allclose(res, exp, rtol=tol)

    @pytest.mark.parametrize(
        "in_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "out_dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_out(self, in_dt, out_dt):
        a = numpy.arange(10, 20).reshape(2, 5).astype(dtype=in_dt)
        out = numpy.zeros_like(a, shape=(2,), dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        result = dpnp.reduce_hypot(ia, out=iout, axis=1)
        exp = numpy.hypot.reduce(a, out=out, axis=1)

        assert result is iout
        assert_allclose(result, exp, rtol=1e-06)


@pytest.mark.parametrize("dtype", get_integer_float_dtypes())
def test_inplace_remainder(dtype):
    size = 21
    a = numpy.arange(size, dtype=dtype)
    ia = dpnp.arange(size, dtype=dtype)

    a %= 4
    ia %= 4

    assert_allclose(ia, a)


@pytest.mark.parametrize("dtype", get_integer_float_dtypes())
def test_inplace_floor_divide(dtype):
    size = 21
    a = numpy.arange(size, dtype=dtype)
    ia = dpnp.arange(size, dtype=dtype)

    a //= 4
    ia //= 4

    assert_allclose(ia, a)


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
    [(), (2), (3, 4), (3, 4, 5)],
)
@pytest.mark.parametrize(
    "y_shape",
    [(), (2), (3, 4), (3, 4, 5)],
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
    assert dpnp.allclose(result, expected)
