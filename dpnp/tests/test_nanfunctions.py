import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)

import dpnp

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_abs_array,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    has_support_aspect64,
    numpy_version,
)
from .third_party.cupy.testing import with_requires


@pytest.mark.parametrize("func", ["nanargmin", "nanargmax"])
class TestNanArgmaxNanArgmin:
    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2])
    @pytest.mark.parametrize("keepdims", [False, True])
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_basic(self, func, axis, keepdims, dtype):
        a = numpy.arange(768, dtype=dtype).reshape((4, 4, 6, 8))
        a[2:3, 2, 3:4, 4] = numpy.nan
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
        result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    def test_out(self, func):
        a = numpy.arange(12, dtype=numpy.float32).reshape((2, 2, 3))
        a[1, 0, 2] = numpy.nan
        ia = dpnp.array(a)

        # out is dpnp_array
        expected = getattr(numpy, func)(a, axis=0)
        iout = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=iout)
        assert iout is result
        assert_allclose(result, expected)

        # out is usm_ndarray
        dpt_out = dpt.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=dpt_out)
        assert dpt_out is result.get_array()
        assert_allclose(result, expected)

        # out is a numpy array -> TypeError
        iout = numpy.empty_like(expected)
        with pytest.raises(TypeError):
            getattr(dpnp, func)(ia, axis=0, out=iout)

        # out shape is incorrect -> ValueError
        iout = dpnp.array(numpy.zeros((2, 2)), dtype=dpnp.intp)
        with pytest.raises(ValueError):
            getattr(dpnp, func)(ia, axis=0, out=iout)

    @pytest.mark.parametrize("in_dt", get_float_complex_dtypes())
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    def test_out_dtype(self, func, in_dt, out_dt):
        a = generate_random_numpy_array((2, 3, 4), dtype=in_dt)
        a[:, :, 2] = numpy.nan
        out = numpy.zeros_like(a, shape=(2, 3), dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        if numpy.can_cast(out.dtype, numpy.intp, casting="safe"):
            result = getattr(dpnp, func)(ia, out=iout, axis=2)
            expected = getattr(numpy, func)(a, out=out, axis=2)
            assert_array_equal(result, expected)
            assert result is iout
        else:
            assert_raises(TypeError, getattr(numpy, func), a, out=out, axis=2)
            assert_raises(TypeError, getattr(dpnp, func), ia, out=iout, axis=2)

    def test_error(self, func):
        ia = dpnp.arange(12, dtype=dpnp.float32).reshape((2, 2, 3))
        ia[:, :, 2] = dpnp.nan

        # All-NaN slice encountered -> ValueError
        with pytest.raises(ValueError):
            getattr(dpnp, func)(ia, axis=0)


@pytest.mark.parametrize("func", ["nancumsum", "nancumprod"])
class TestNanCumSumProd:
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize(
        "array",
        [numpy.array(numpy.nan), numpy.full((3, 3), numpy.nan)],
        ids=["0d", "2d"],
    )
    def test_allnans(self, func, dtype, array):
        a = numpy.array(array, dtype=dtype)
        ia = dpnp.array(a, dtype=dtype)

        result = getattr(dpnp, func)(ia)
        expected = getattr(numpy, func)(a)
        # for "0d" case, dpnp returns 0D array, numpy returns 1D array
        # Array API indicates that the behavior is unspecified
        assert_dtype_allclose(result, expected, check_shape=False)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_empty(self, func, axis):
        a = numpy.zeros((0, 3))
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia, axis=axis)
        expected = getattr(numpy, func)(a, axis=axis)
        assert_equal(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_keepdims(self, func, axis):
        a = numpy.eye(3)
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia, axis=axis, out=None)
        expected = getattr(numpy, func)(a, axis=axis, out=None)
        assert_equal(result, expected)
        assert result.ndim == expected.ndim

    @pytest.mark.parametrize("axis", [None] + list(range(4)))
    def test_keepdims_random(self, func, axis):
        a = numpy.ones((3, 5, 7, 11))
        # Randomly set some elements to NaN:
        rs = numpy.random.RandomState(0)
        a[rs.rand(*a.shape) < 0.5] = numpy.nan
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia, axis=axis)
        expected = getattr(numpy, func)(a, axis=axis)
        assert_equal(result, expected)

    @pytest.mark.parametrize("axis", [-2, -1, 0, 1, None])
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_ndat_ones(self, func, axis, dtype):
        a = numpy.array(
            [
                [0.6244, 1.0, 0.2692, 0.0116, 1.0, 0.1170],
                [0.5351, -0.9403, 1.0, 0.2100, 0.4759, 0.2833],
                [1.0, 1.0, 1.0, 0.1042, 1.0, -0.5954],
                [0.1610, 1.0, 1.0, 0.1859, 0.3146, 1.0],
            ]
        )
        a = a.astype(dtype=dtype)
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia, axis=axis)
        expected = getattr(numpy, func)(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [-2, -1, 0, 1, None])
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_ndat_zeros(self, func, axis, dtype):
        a = numpy.array(
            [
                [0.6244, 0.0, 0.2692, 0.0116, 0.0, 0.1170],
                [0.5351, -0.9403, 0.0, 0.2100, 0.4759, 0.2833],
                [0.0, 0.0, 0.0, 0.1042, 0.0, -0.5954],
                [0.1610, 0.0, 0.0, 0.1859, 0.3146, 0.0],
            ]
        )
        a = a.astype(dtype=dtype)
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia, axis=axis)
        expected = getattr(numpy, func)(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [-2, -1, 0, 1])
    def test_out(self, func, axis):
        a = numpy.eye(3)
        out = numpy.eye(3)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = getattr(dpnp, func)(ia, axis=axis, out=iout)
        expected = getattr(numpy, func)(a, axis=axis, out=out)
        assert_almost_equal(result, expected)
        assert result is iout


@pytest.mark.parametrize("func", ["nanmax", "nanmin"])
class TestNanMaxNanMin:
    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
    @pytest.mark.parametrize("keepdims", [False, True])
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_basic(self, func, axis, keepdims, dtype):
        a = generate_random_numpy_array((4, 4, 6, 8), dtype=dtype)
        a[2:3, 2, 3:4, 4] = numpy.nan
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
        result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_strided(self, func, dtype):
        a = numpy.arange(20, dtype=dtype)
        a[::3] = numpy.nan
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a[::-1])
        result = getattr(dpnp, func)(ia[::-1])
        assert_dtype_allclose(result, expected)

        expected = getattr(numpy, func)(a[::2])
        result = getattr(dpnp, func)(ia[::2])
        assert_dtype_allclose(result, expected)

    def test_out(self, func):
        a = numpy.arange(12, dtype=numpy.float32).reshape((2, 2, 3))
        a[1, 0, 2] = numpy.nan
        ia = dpnp.array(a)

        # out is dpnp_array
        expected = getattr(numpy, func)(a, axis=0)
        iout = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=iout)
        assert iout is result
        assert_allclose(result, expected)

        # out is usm_ndarray
        dpt_out = dpt.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=dpt_out)
        assert dpt_out is result.get_array()
        assert_allclose(result, expected)

        # output is numpy array -> Error
        iout = numpy.empty_like(expected)
        with pytest.raises(TypeError):
            getattr(dpnp, func)(ia, axis=0, out=iout)

        # output has incorrect shape -> Error
        iout = dpnp.array(numpy.zeros((4, 2)))
        with pytest.raises(ValueError):
            getattr(dpnp, func)(ia, axis=0, out=iout)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize("in_dt", get_float_complex_dtypes())
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    def test_out_dtype(self, func, in_dt, out_dt):
        # if out_dt is unsigned, input cannot be signed otherwise overflow occurs
        low = 0 if dpnp.issubdtype(out_dt, dpnp.unsignedinteger) else -10
        a = generate_random_numpy_array((2, 3, 4), dtype=in_dt, low=low)
        a[:, :, 2] = numpy.nan
        out = numpy.zeros_like(a, shape=(2, 3), dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        result = getattr(dpnp, func)(ia, out=iout, axis=2)
        expected = getattr(numpy, func)(a, out=out, axis=2)
        assert_array_equal(result, expected)
        assert result is iout

    def test_error(self, func):
        ia = dpnp.arange(5)
        # where is not supported
        with pytest.raises(NotImplementedError):
            getattr(dpnp, func)(ia, where=False)

        # initial is not supported
        with pytest.raises(NotImplementedError):
            getattr(dpnp, func)(ia, initial=6)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_no_NaN(self, func, dtype):
        a = generate_random_numpy_array((4, 4, 6, 8), dtype=dtype)
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=0)
        result = getattr(dpnp, func)(ia, axis=0)
        assert_dtype_allclose(result, expected)

    def test_all_NaN(self, recwarn, func):
        a = numpy.arange(12, dtype=numpy.float32).reshape((2, 2, 3))
        a[:, :, 2] = numpy.nan
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=0)
        result = getattr(dpnp, func)(ia, axis=0)
        assert_dtype_allclose(result, expected)

        assert len(recwarn) == 2
        assert all(
            "All-NaN slice encountered" in str(r.message) for r in recwarn
        )
        assert all(r.category is RuntimeWarning for r in recwarn)


class TestNanMean:
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic(self, dtype, axis, keepdims):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        result = dpnp.nanmean(ia, axis=axis, keepdims=keepdims)
        expected = numpy.nanmean(a, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("axis", [0, 1])
    def test_out(self, dtype, axis):
        ia = dpnp.array([[dpnp.nan, 1, 2], [3, dpnp.nan, 0]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        expected = numpy.nanmean(a, axis=axis)
        out = dpnp.empty_like(dpnp.asarray(expected))
        result = dpnp.nanmean(ia, axis=axis, out=out)
        assert out is result
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_complex(self, dtype):
        a = generate_random_numpy_array(10, dtype=dtype)
        a[::3] = numpy.nan
        ia = dpnp.array(a)

        expected = numpy.nanmean(a)
        result = dpnp.nanmean(ia)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_dtype(self, dtype):
        ia = dpnp.array([[dpnp.nan, 1, 2], [3, dpnp.nan, 0]])
        a = dpnp.asnumpy(ia)

        expected = numpy.nanmean(a, dtype=dtype)
        result = dpnp.nanmean(ia, dtype=dtype)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_strided(self, dtype):
        ia = dpnp.arange(20, dtype=dtype)
        ia[::3] = dpnp.nan
        a = dpnp.asnumpy(ia)

        result = dpnp.nanmean(ia[::-1])
        expected = numpy.nanmean(a[::-1])
        assert_dtype_allclose(result, expected)

        result = dpnp.nanmean(ia[::2])
        expected = numpy.nanmean(a[::2])
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_mean_empty_slice_numpy_warnings")
    def test_scalar(self):
        ia = dpnp.array(dpnp.nan)
        a = dpnp.asnumpy(ia)

        result = dpnp.nanmean(ia)
        expected = numpy.nanmean(a)
        assert_allclose(result, expected)

    def test_error(self):
        ia = dpnp.arange(5, dtype=dpnp.float32)
        ia[0] = dpnp.nan
        # where keyword is not implemented
        with pytest.raises(NotImplementedError):
            dpnp.nanmean(ia, where=False)

        # dtype should be floating
        with pytest.raises(TypeError):
            dpnp.nanmean(ia, dtype=dpnp.int32)

        # out dtype should be inexact
        res = dpnp.empty((1,), dtype=dpnp.int32)
        with pytest.raises(TypeError):
            dpnp.nanmean(ia, out=res)


class TestNanMedian:
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("axis", [None, 0, (-1,), [0, 1], (0, -2, -1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_basic(self, dtype, axis, keepdims):
        a = generate_random_numpy_array((2, 3, 4), dtype=dtype)
        a[0, 0, 0] = a[-2, -2, -2] = numpy.nan
        ia = dpnp.array(a)

        expected = numpy.nanmedian(a, axis=axis, keepdims=keepdims)
        result = dpnp.nanmedian(ia, axis=axis, keepdims=keepdims)

        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_mean_empty_slice_numpy_warnings",
    )
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 0), (0, 3)])
    def test_empty(self, axis, shape):
        a = numpy.empty(shape)
        ia = dpnp.array(a)

        result = dpnp.nanmedian(ia, axis=axis)
        expected = numpy.nanmedian(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [None, 0, (-1,), [0, 1], (0, -2, -1)])
    def test_no_nan(self, dtype, axis):
        a = generate_random_numpy_array((2, 3, 4), dtype=dtype)
        ia = dpnp.array(a)

        expected = numpy.nanmedian(a, axis=axis)
        result = dpnp.nanmedian(ia, axis=axis)

        assert_dtype_allclose(result, expected)

    @pytest.mark.filterwarnings("ignore:All-NaN slice:RuntimeWarning")
    def test_all_nan(self):
        a = numpy.array(numpy.nan)
        ia = dpnp.array(a)

        result = dpnp.nanmedian(ia)
        expected = numpy.nanmedian(a)
        assert_dtype_allclose(result, expected)

        a = generate_random_numpy_array((2, 3, 4))
        a[:, :, 2] = numpy.nan
        ia = dpnp.array(a)

        result = dpnp.nanmedian(ia, axis=1)
        expected = numpy.nanmedian(a, axis=1)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, -1, (0, -2, -1)])
    def test_overwrite_input(self, axis):
        a = generate_random_numpy_array((2, 3, 4))
        a[0, 0, 0] = a[-2, -2, -2] = numpy.nan
        ia = dpnp.array(a)

        b = a.copy()
        ib = ia.copy()
        expected = numpy.nanmedian(b, axis=axis, overwrite_input=True)
        result = dpnp.nanmedian(ib, axis=axis, overwrite_input=True)
        assert not numpy.all(a == b)
        assert not dpnp.all(ia == ib)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, (-1,), [0, 1]])
    @pytest.mark.parametrize("overwrite_input", [True, False])
    def test_usm_ndarray(self, axis, overwrite_input):
        a = generate_random_numpy_array((2, 3, 4))
        a[0, 0, 0] = a[-2, -2, -2] = numpy.nan
        ia = dpt.asarray(a)

        expected = numpy.nanmedian(
            a, axis=axis, overwrite_input=overwrite_input
        )
        result = dpnp.nanmedian(ia, axis=axis, overwrite_input=overwrite_input)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize(
        "axis, out_shape", [(0, (3,)), (1, (2,)), ((0, 1), ())]
    )
    def test_out(self, dtype, axis, out_shape):
        a = numpy.array([[5, numpy.nan, 2], [8, 4, numpy.nan]], dtype=dtype)
        ia = dpnp.array(a)

        out_np = numpy.empty_like(a, shape=out_shape)
        out_dp = dpnp.empty_like(ia, shape=out_shape)
        expected = numpy.nanmedian(a, axis=axis, out=out_np)
        result = dpnp.nanmedian(ia, axis=axis, out=out_dp)
        assert result is out_dp
        assert_dtype_allclose(result, expected)

    def test_error(self):
        a = dpnp.arange(6.0).reshape(2, 3)
        a[0, 0] = a[-1, -1] = numpy.nan

        # out shape is incorrect
        res = dpnp.empty(3, dtype=a.dtype)
        with pytest.raises(ValueError):
            dpnp.nanmedian(a, axis=1, out=res)

        # out has a different queue
        exec_q = dpctl.SyclQueue()
        res = dpnp.empty(2, dtype=a.dtype, sycl_queue=exec_q)
        with pytest.raises(ExecutionPlacementError):
            dpnp.nanmedian(a, axis=1, out=res)


@pytest.mark.parametrize("func", ["nanprod", "nansum"])
class TestNanProdSum:
    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
    @pytest.mark.parametrize("keepdims", [False, True])
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_basic(self, func, axis, keepdims, dtype):
        a = generate_random_numpy_array((2, 3, 4), dtype=dtype)
        a[:, :, 2] = numpy.nan
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
        result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)

        assert_allclose(result, expected, rtol=1e-5)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize("in_dt", get_float_complex_dtypes())
    @pytest.mark.parametrize("dt", get_all_dtypes())
    def test_dtype(self, func, in_dt, dt):
        a = generate_random_numpy_array((2, 3, 4), dtype=in_dt)
        a[:, :, 2] = numpy.nan
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, dtype=dt)
        result = getattr(dpnp, func)(ia, dtype=dt)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_overflow_encountered_in_cast_numpy_warnings"
    )
    def test_out(self, func):
        ia = dpnp.arange(1, 7).reshape((2, 3))
        ia = ia.astype(dpnp.default_float_type(ia.device))
        ia[:, 1] = dpnp.nan
        a = dpnp.asnumpy(ia)

        # out is dpnp_array
        expected = getattr(numpy, func)(a, axis=0)
        iout = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=iout)
        assert iout is result
        assert_allclose(result, expected)

        # out is usm_ndarray
        dpt_out = dpt.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=dpt_out)
        assert dpt_out is result.get_array()
        assert_allclose(result, expected)

        # out is a numpy array -> TypeError
        iout = numpy.empty_like(expected)
        with pytest.raises(TypeError):
            getattr(dpnp, func)(ia, axis=0, out=iout)

        # incorrect shape for out
        iout = dpnp.array(numpy.empty((2, 3)))
        with pytest.raises(ValueError):
            getattr(dpnp, func)(ia, axis=0, out=iout)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("in_dt", get_float_complex_dtypes())
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    def test_out_dtype(self, func, in_dt, out_dt):
        # if out_dt is unsigned, input cannot be signed otherwise overflow occurs
        low = 0 if dpnp.issubdtype(out_dt, dpnp.unsignedinteger) else -5
        a = generate_random_numpy_array((2, 3, 4), dtype=in_dt, low=low, high=5)
        a[:, :, 2] = numpy.nan
        out = numpy.zeros_like(a, shape=(2, 3), dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        result = getattr(dpnp, func)(ia, out=iout, axis=2)
        expected = getattr(numpy, func)(a, out=out, axis=2)
        assert_allclose(result, expected, rtol=1e-06)
        assert result is iout

    @pytest.mark.parametrize("stride", [-1, 2])
    def test_strided(self, func, stride):
        a = numpy.arange(20.0)
        a[::3] = numpy.nan
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia[::stride])
        expected = getattr(numpy, func)(a[::stride])
        assert_allclose(result, expected)


@pytest.mark.parametrize("func", ["nanstd", "nanvar"])
class TestNanStdVar:
    @pytest.mark.parametrize(
        "array",
        [
            [2, 0, 6, 2],
            [2, 0, 6, 2, 5, 6, 7, 8],
            [],
            [2, 1, numpy.nan, 5, 3],
            [-1, numpy.nan, 1, numpy.inf],
            [3, 6, 0, 1],
            [3, 6, 0, 1, 8],
            [3, 2, 9, 6, numpy.nan],
            [numpy.nan, numpy.nan, numpy.inf, numpy.nan],
            [[2, 0], [6, 2]],
            [[2, 0, 6, 2], [5, 6, 7, 8]],
            [[[2, 0], [6, 2]], [[5, 6], [7, 8]]],
            [[-1, numpy.nan], [1, numpy.inf]],
            [[numpy.nan, numpy.nan], [numpy.inf, numpy.nan]],
        ],
        ids=[
            "[2, 0, 6, 2]",
            "[2, 0, 6, 2, 5, 6, 7, 8]",
            "[]",
            "[2, 1, np.nan, 5, 3]",
            "[-1, np.nan, 1, np.inf]",
            "[3, 6, 0, 1]",
            "[3, 6, 0, 1, 8]",
            "[3, 2, 9, 6, np.nan]",
            "[np.nan, np.nan, np.inf, np.nan]",
            "[[2, 0], [6, 2]]",
            "[[2, 0, 6, 2], [5, 6, 7, 8]]",
            "[[[2, 0], [6, 2]], [[5, 6], [7, 8]]]",
            "[[-1, np.nan], [1, np.inf]]",
            "[[np.nan, np.nan], [np.inf, np.nan]]",
        ],
    )
    @pytest.mark.usefixtures(
        "suppress_invalid_numpy_warnings", "suppress_dof_numpy_warnings"
    )
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_basic(self, func, array, dtype):
        try:
            a = get_abs_array(array, dtype=dtype)
        except:
            pytest.skip("floating data type is needed to store NaN")
        ia = dpnp.array(a)

        for ddof in range(a.ndim):
            expected = getattr(numpy, func)(a, ddof=ddof)
            result = getattr(dpnp, func)(ia, ddof=ddof)
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_complex(self, func, dtype):
        a = generate_random_numpy_array(10, dtype=dtype)
        a[::3] = numpy.nan
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a)
        result = getattr(dpnp, func)(ia)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_dof_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 1), (1, 2)])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("ddof", [0, 0.5, 1, 1.5, 2, 3])
    def test_out(self, func, dtype, axis, keepdims, ddof):
        a = numpy.arange(4 * 3 * 5, dtype=dtype)
        a[::2] = numpy.nan
        a = a.reshape(4, 3, 5)
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(
            a, axis=axis, ddof=ddof, keepdims=keepdims
        )
        if has_support_aspect64():
            res_dtype = expected.dtype
        else:
            res_dtype = dpnp.default_float_type(ia.device)
        out = dpnp.empty(expected.shape, dtype=res_dtype)
        result = getattr(dpnp, func)(
            ia, out=out, axis=axis, ddof=ddof, keepdims=keepdims
        )
        assert result is out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    def test_strided(self, func, dtype):
        a = numpy.arange(20, dtype=dtype)
        a[::3] = numpy.nan
        ia = dpnp.array(a)

        result = getattr(dpnp, func)(ia[::-1])
        expected = getattr(numpy, func)(a[::-1])
        assert_dtype_allclose(result, expected)

        result = getattr(dpnp, func)(ia[::2])
        expected = getattr(numpy, func)(a[::2])
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("dt_in", get_float_complex_dtypes())
    @pytest.mark.parametrize("dt_out", get_float_complex_dtypes())
    def test_dtype(self, func, dt_in, dt_out):
        a = numpy.arange(4 * 3 * 5, dtype=dt_in)
        a[::2] = numpy.nan
        a = a.reshape(4, 3, 5)
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, dtype=dt_out)
        result = getattr(dpnp, func)(ia, dtype=dt_out)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("axis", [1, (0, 2), None])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_mean_keyword(self, func, dtype, axis, keepdims):
        a = generate_random_numpy_array((10, 20, 5), dtype)
        mask = numpy.random.choice([True, False], size=a.size, p=[0.3, 0.7])
        numpy.place(a, mask, numpy.nan)
        ia = dpnp.array(a)

        mean = numpy.nanmean(a, axis=axis, keepdims=True)
        imean = dpnp.nanmean(ia, axis=axis, keepdims=True)

        mean_kw = {"mean": mean} if numpy_version() >= "2.0.0" else {}
        expected = getattr(numpy, func)(
            a, axis=axis, keepdims=keepdims, **mean_kw
        )
        result = getattr(dpnp, func)(
            ia, axis=axis, keepdims=keepdims, mean=imean
        )
        assert_dtype_allclose(result, expected)

    @with_requires("numpy>=2.0")
    def test_correction(self, func):
        a = numpy.array([127, numpy.nan, numpy.nan, 39, 93, 87, numpy.nan, 46])
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, correction=0.5)
        result = getattr(dpnp, func)(ia, correction=0.5)
        assert_dtype_allclose(result, expected)

    @with_requires("numpy>=2.0")
    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_both_ddof_correction_are_set(self, func, xp):
        a = xp.array([5, xp.nan, -2])

        err_msg = "ddof and correction can't be provided simultaneously."

        with assert_raises_regex(ValueError, err_msg):
            getattr(xp, func)(a, ddof=0.5, correction=0.5)

        with assert_raises_regex(ValueError, err_msg):
            getattr(xp, func)(a, ddof=1, correction=0)

    def test_error(self, func):
        ia = dpnp.arange(5, dtype=dpnp.float32)
        ia[0] = dpnp.nan

        # where keyword is not implemented
        with pytest.raises(NotImplementedError):
            getattr(dpnp, func)(ia, where=False)

        # dtype should be floating
        with pytest.raises(TypeError):
            getattr(dpnp, func)(ia, dtype=dpnp.int32)

        # out dtype should be inexact
        res = dpnp.empty((1,), dtype=dpnp.int32)
        with pytest.raises(TypeError):
            getattr(dpnp, func)(ia, out=res)

        # ddof should be an integer or float
        with pytest.raises(TypeError):
            getattr(dpnp, func)(ia, ddof="1")
