import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)

import dpnp

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    has_support_aspect64,
)
from .third_party.cupy.testing import with_requires


class TestAverage:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("returned", [True, False])
    def test_avg_no_wgt(self, dtype, axis, returned):
        ia = dpnp.array([[1, 1, 2], [3, 4, 5]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        result = dpnp.average(ia, axis=axis, returned=returned)
        expected = numpy.average(a, axis=axis, returned=returned)
        if returned:
            assert_dtype_allclose(result[0], expected[0])
            assert_dtype_allclose(result[1], expected[1])
        else:
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("returned", [True, False])
    def test_avg(self, dtype, axis, returned):
        ia = dpnp.array([[1, 1, 2], [3, 4, 5]], dtype=dtype)
        iw = dpnp.array([[3, 1, 2], [3, 4, 2]], dtype=dtype)
        a = dpnp.asnumpy(ia)
        w = dpnp.asnumpy(iw)

        result = dpnp.average(ia, axis=axis, weights=iw, returned=returned)
        expected = numpy.average(a, axis=axis, weights=w, returned=returned)

        if returned:
            assert_dtype_allclose(result[0], expected[0])
            assert_dtype_allclose(result[1], expected[1])
        else:
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_avg_complex(self, dtype):
        x1 = numpy.random.rand(10)
        x2 = numpy.random.rand(10)
        a = numpy.array(x1 + 1j * x2, dtype=dtype)
        w = numpy.array(x2 + 1j * x1, dtype=dtype)
        ia = dpnp.array(a)
        iw = dpnp.array(w)

        expected = numpy.average(a, weights=w)
        result = dpnp.average(ia, weights=iw)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "weight",
        [[[3, 1, 2], [3, 4, 2]], ((3, 1, 2), (3, 4, 2))],
        ids=["list", "tuple"],
    )
    def test_avg_weight_array_like(self, weight):
        ia = dpnp.array([[1, 1, 2], [3, 4, 5]])
        a = dpnp.asnumpy(ia)

        res = dpnp.average(ia, weights=weight)
        exp = numpy.average(a, weights=weight)
        assert_dtype_allclose(res, exp)

    def test_avg_weight_1D(self):
        ia = dpnp.arange(12).reshape(3, 4)
        wgt = [1, 2, 3]
        a = dpnp.asnumpy(ia)

        res = dpnp.average(ia, axis=0, weights=wgt)
        exp = numpy.average(a, axis=0, weights=wgt)
        assert_dtype_allclose(res, exp)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_avg_strided(self, dtype):
        ia = dpnp.arange(20, dtype=dtype)
        iw = dpnp.arange(-10, 10, dtype=dtype)
        a = dpnp.asnumpy(ia)
        w = dpnp.asnumpy(iw)

        result = dpnp.average(ia[::-1], weights=iw[::-1])
        expected = numpy.average(a[::-1], weights=w[::-1])
        assert_allclose(result, expected)

        result = dpnp.average(ia[::2], weights=iw[::2])
        expected = numpy.average(a[::2], weights=w[::2])
        assert_allclose(result, expected)

    def test_avg_error(self):
        a = dpnp.arange(5)
        w = dpnp.zeros(5)
        # Weights sum to zero
        with pytest.raises(ZeroDivisionError):
            dpnp.average(a, weights=w)

        a = dpnp.arange(12).reshape(3, 4)
        w = dpnp.ones(12)
        # Axis must be specified when shapes of input array and weights differ
        with pytest.raises(TypeError):
            dpnp.average(a, weights=w)

        a = dpnp.arange(12).reshape(3, 4)
        w = dpnp.ones(12).reshape(2, 6)
        # 1D weights expected when shapes of input array and weights differ.
        with pytest.raises(TypeError):
            dpnp.average(a, axis=0, weights=w)

        a = dpnp.arange(12).reshape(3, 4)
        w = dpnp.ones(12)
        # Length of weights not compatible with specified axis.
        with pytest.raises(ValueError):
            dpnp.average(a, axis=0, weights=w)

        a = dpnp.arange(12, sycl_queue=dpctl.SyclQueue())
        w = dpnp.ones(12, sycl_queue=dpctl.SyclQueue())
        # Execution placement can not be unambiguously inferred
        with pytest.raises(ValueError):
            dpnp.average(a, axis=0, weights=w)


class TestMaxMin:
    @pytest.mark.parametrize("func", ["max", "min"])
    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
    @pytest.mark.parametrize("keepdims", [False, True])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_func(self, func, axis, keepdims, dtype):
        a = numpy.arange(768, dtype=dtype).reshape((4, 4, 6, 8))
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
        result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("func", ["max", "min"])
    @pytest.mark.parametrize("axis", [None, 0, 1, -1])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_bool(self, func, axis, keepdims):
        a = numpy.arange(2, dtype=numpy.bool_)
        a = numpy.tile(a, (2, 2))
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
        result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("func", ["max", "min"])
    def test_out(self, func):
        a = numpy.arange(12, dtype=numpy.float32).reshape((2, 2, 3))
        ia = dpnp.array(a)

        # out is dpnp_array
        expected = getattr(numpy, func)(a, axis=0)
        dpnp_out = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=dpnp_out)
        assert dpnp_out is result
        assert_allclose(result, expected)

        # out is usm_ndarray
        dpt_out = dpt.empty(expected.shape, dtype=expected.dtype)
        result = getattr(dpnp, func)(ia, axis=0, out=dpt_out)
        assert dpt_out is result.get_array()
        assert_allclose(result, expected)

        # output is numpy array -> Error
        result = numpy.empty_like(expected)
        with pytest.raises(TypeError):
            getattr(dpnp, func)(ia, axis=0, out=result)

        # output has incorrect shape -> Error
        result = dpnp.array(numpy.zeros((4, 2)))
        with pytest.raises(ValueError):
            getattr(dpnp, func)(ia, axis=0, out=result)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("func", ["max", "min"])
    @pytest.mark.parametrize("arr_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    def test_out_dtype(self, func, arr_dt, out_dt):
        a = numpy.arange(12).reshape(2, 2, 3).astype(arr_dt)
        out = numpy.zeros_like(a, shape=(2, 3), dtype=out_dt)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        result = getattr(dpnp, func)(ia, out=iout, axis=1)
        expected = getattr(numpy, func)(a, out=out, axis=1)
        assert_array_equal(result, expected)
        assert result is iout

    @pytest.mark.parametrize("func", ["max", "min"])
    def test_error(self, func):
        ia = dpnp.arange(5)
        # where is not supported
        with pytest.raises(NotImplementedError):
            getattr(dpnp, func)(ia, where=False)

        # initial is not supported
        with pytest.raises(NotImplementedError):
            getattr(dpnp, func)(ia, initial=6)


class TestMean:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_mean(self, dtype, axis, keepdims):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        result = dpnp.mean(ia, axis=axis, keepdims=keepdims)
        expected = numpy.mean(a, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "axis, out_shape", [(0, (3,)), (1, (2,)), ((0, 1), ())]
    )
    def test_mean_out(self, dtype, axis, out_shape):
        ia = dpnp.array([[5, 1, 2], [8, 4, 3]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        out_np = numpy.empty_like(a, shape=out_shape)
        out_dp = dpnp.empty_like(ia, shape=out_shape)
        expected = numpy.mean(a, axis=axis, out=out_np)
        result = dpnp.mean(ia, axis=axis, out=out_dp)
        assert result is out_dp
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_mean_complex(self, dtype):
        x1 = numpy.random.rand(10)
        x2 = numpy.random.rand(10)
        a = numpy.array(x1 + 1j * x2, dtype=dtype)
        ia = dpnp.array(a)

        expected = numpy.mean(a)
        result = dpnp.mean(ia)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_invalid_numpy_warnings",
        "suppress_mean_empty_slice_numpy_warnings",
    )
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 0), (0, 3)])
    def test_mean_empty(self, axis, shape):
        ia = dpnp.empty(shape, dtype=dpnp.int64)
        a = dpnp.asnumpy(ia)

        result = dpnp.mean(ia, axis=axis)
        expected = numpy.mean(a, axis=axis)
        assert_allclose(result, expected)

    def test_mean_scalar(self):
        ia = dpnp.array(5)
        a = dpnp.asnumpy(ia)

        result = ia.mean()
        expected = a.mean()
        assert_allclose(result, expected)

    def test_mean_NotImplemented(self):
        ia = dpnp.arange(5)
        with pytest.raises(NotImplementedError):
            dpnp.mean(ia, where=False)


class TestMedian:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 9])
    def test_basic(self, dtype, size):
        if dtype == dpnp.bool:
            a = numpy.arange(2, dtype=dtype)
            a = numpy.repeat(a, size)
        else:
            a = numpy.array(numpy.random.uniform(-5, 5, size), dtype=dtype)
        ia = dpnp.array(a)

        expected = numpy.median(a)
        result = dpnp.median(ia)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, (-1,), [0, 1], (0, -2, -1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_axis(self, axis, keepdims):
        a = numpy.random.uniform(-5, 5, 24).reshape(2, 3, 4)
        ia = dpnp.array(a)

        expected = numpy.median(a, axis=axis, keepdims=keepdims)
        result = dpnp.median(ia, axis=axis, keepdims=keepdims)

        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_invalid_numpy_warnings",
        "suppress_mean_empty_slice_numpy_warnings",
    )
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 0), (0, 3)])
    def test_empty(self, axis, shape):
        a = numpy.empty(shape)
        ia = dpnp.array(a)

        result = dpnp.median(ia, axis=axis)
        expected = numpy.median(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "axis, out_shape", [(0, (3,)), (1, (2,)), ((0, 1), ())]
    )
    def test_out(self, dtype, axis, out_shape):
        a = numpy.array([[5, 1, 2], [8, 4, 3]], dtype=dtype)
        ia = dpnp.array(a)

        out_np = numpy.empty_like(a, shape=out_shape)
        out_dp = dpnp.empty_like(ia, shape=out_shape)
        expected = numpy.median(a, axis=axis, out=out_np)
        result = dpnp.median(ia, axis=axis, out=out_dp)
        assert result is out_dp
        assert_dtype_allclose(result, expected)

    def test_0d_array(self):
        a = numpy.array(20)
        ia = dpnp.array(a)

        result = dpnp.median(ia)
        expected = numpy.median(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, (0, 1), (0, -2, -1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_nan(self, axis, keepdims):
        a = numpy.random.uniform(-5, 5, 24).reshape(2, 3, 4)
        a[0, 0, 0] = a[-1, -1, -1] = numpy.nan
        ia = dpnp.array(a)

        expected = numpy.median(a, axis=axis, keepdims=keepdims)
        result = dpnp.median(ia, axis=axis, keepdims=keepdims)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, -1, (0, -2, -1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_overwrite_input(self, axis, keepdims):
        a = numpy.random.uniform(-5, 5, 24).reshape(2, 3, 4)
        ia = dpnp.array(a)

        b = a.copy()
        ib = ia.copy()
        expected = numpy.median(
            b, axis=axis, keepdims=keepdims, overwrite_input=True
        )
        result = dpnp.median(
            ib, axis=axis, keepdims=keepdims, overwrite_input=True
        )
        assert not numpy.all(a == b)
        assert not dpnp.all(ia == ib)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, (-1,), [0, 1]])
    @pytest.mark.parametrize("overwrite_input", [True, False])
    def test_usm_ndarray(self, axis, overwrite_input):
        a = numpy.random.uniform(-5, 5, 24).reshape(2, 3, 4)
        ia = dpt.asarray(a)

        expected = numpy.median(a, axis=axis, overwrite_input=overwrite_input)
        result = dpnp.median(ia, axis=axis, overwrite_input=overwrite_input)
        assert_dtype_allclose(result, expected)


class TestVar:
    @pytest.mark.usefixtures(
        "suppress_divide_invalid_numpy_warnings", "suppress_dof_numpy_warnings"
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("ddof", [0, 0.5, 1, 1.5, 2])
    def test_var(self, dtype, axis, keepdims, ddof):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        expected = numpy.var(a, axis=axis, keepdims=keepdims, ddof=ddof)
        result = dpnp.var(ia, axis=axis, keepdims=keepdims, ddof=ddof)

        if axis == 0 and ddof == 2:
            assert dpnp.all(dpnp.isnan(result))
        else:
            assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_divide_invalid_numpy_warnings", "suppress_dof_numpy_warnings"
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("ddof", [0, 1])
    def test_var_out(self, dtype, axis, ddof):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        expected = numpy.var(a, axis=axis, ddof=ddof)
        if has_support_aspect64():
            res_dtype = expected.dtype
        else:
            res_dtype = dpnp.default_float_type(ia.device)
        out = dpnp.empty(expected.shape, dtype=res_dtype)
        result = dpnp.var(ia, axis=axis, out=out, ddof=ddof)
        assert result is out
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_invalid_numpy_warnings", "suppress_dof_numpy_warnings"
    )
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 0), (0, 3)])
    def test_var_empty(self, axis, shape):
        ia = dpnp.empty(shape, dtype=dpnp.int64)
        a = dpnp.asnumpy(ia)

        result = dpnp.var(ia, axis=axis)
        expected = numpy.var(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("dt_in", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("dt_out", get_float_complex_dtypes())
    def test_var_dtype(self, dt_in, dt_out):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dt_in)
        a = dpnp.asnumpy(ia)

        expected = numpy.var(a, dtype=dt_out)
        result = dpnp.var(ia, dtype=dt_out)
        assert expected.dtype == result.dtype
        assert_allclose(result, expected, rtol=1e-06)

    def test_var_scalar(self):
        ia = dpnp.array(5)
        a = dpnp.asnumpy(ia)

        result = ia.var()
        expected = a.var()
        assert_allclose(result, expected)

    def test_var_error(self):
        ia = dpnp.arange(5)
        # where keyword is not implemented
        with pytest.raises(NotImplementedError):
            dpnp.var(ia, where=False)

        # ddof should be an integer or float
        with pytest.raises(TypeError):
            dpnp.var(ia, ddof="1")


class TestStd:
    @pytest.mark.usefixtures(
        "suppress_divide_invalid_numpy_warnings", "suppress_dof_numpy_warnings"
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("ddof", [0, 0.5, 1, 1.5, 2])
    def test_std(self, dtype, axis, keepdims, ddof):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        expected = numpy.std(a, axis=axis, keepdims=keepdims, ddof=ddof)
        result = dpnp.std(ia, axis=axis, keepdims=keepdims, ddof=ddof)
        if axis == 0 and ddof == 2:
            assert dpnp.all(dpnp.isnan(result))
        else:
            assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_divide_invalid_numpy_warnings", "suppress_dof_numpy_warnings"
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("ddof", [0, 1])
    def test_std_out(self, dtype, axis, ddof):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        expected = numpy.std(a, axis=axis, ddof=ddof)
        if has_support_aspect64():
            res_dtype = expected.dtype
        else:
            res_dtype = dpnp.default_float_type(ia.device)
        out = dpnp.empty(expected.shape, dtype=res_dtype)
        result = dpnp.std(ia, axis=axis, out=out, ddof=ddof)
        assert out is result
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_invalid_numpy_warnings", "suppress_dof_numpy_warnings"
    )
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 0), (0, 3)])
    def test_std_empty(self, axis, shape):
        ia = dpnp.empty(shape, dtype=dpnp.int64)
        a = dpnp.asnumpy(ia)

        result = dpnp.std(ia, axis=axis)
        expected = numpy.std(a, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("dt_in", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("dt_out", get_float_complex_dtypes())
    def test_std_dtype(self, dt_in, dt_out):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dt_in)
        a = dpnp.asnumpy(ia)

        expected = numpy.std(a, dtype=dt_out)
        result = dpnp.std(ia, dtype=dt_out)
        assert expected.dtype == result.dtype
        assert_allclose(result, expected, rtol=1e-6)

    def test_std_scalar(self):
        ia = dpnp.array(5)
        a = dpnp.asnumpy(ia)

        result = ia.std()
        expected = a.std()
        assert_dtype_allclose(result, expected)

    def test_std_error(self):
        ia = dpnp.arange(5)
        # where keyword is not implemented
        with pytest.raises(NotImplementedError):
            dpnp.std(ia, where=False)

        # ddof should be an integer or float
        with pytest.raises(TypeError):
            dpnp.std(ia, ddof="1")


class TestCorrcoef:
    @pytest.mark.usefixtures(
        "suppress_divide_invalid_numpy_warnings",
        "suppress_dof_numpy_warnings",
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("rowvar", [True, False])
    def test_corrcoef(self, dtype, rowvar):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        a = dpnp.asnumpy(ia)

        expected = numpy.corrcoef(a, rowvar=rowvar)
        result = dpnp.corrcoef(ia, rowvar=rowvar)

        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_divide_invalid_numpy_warnings",
        "suppress_dof_numpy_warnings",
        "suppress_mean_empty_slice_numpy_warnings",
    )
    @pytest.mark.parametrize("shape", [(2, 0), (0, 2)])
    def test_corrcoef_empty(self, shape):
        ia = dpnp.empty(shape, dtype=dpnp.int64)
        a = dpnp.asnumpy(ia)

        result = dpnp.corrcoef(ia)
        expected = numpy.corrcoef(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("dt_in", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("dt_out", get_float_complex_dtypes())
    def test_corrcoef_dtype(self, dt_in, dt_out):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dt_in)
        a = dpnp.asnumpy(ia)

        expected = numpy.corrcoef(a, dtype=dt_out)
        result = dpnp.corrcoef(ia, dtype=dt_out)
        assert expected.dtype == result.dtype
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.usefixtures(
        "suppress_divide_invalid_numpy_warnings",
        "suppress_dof_numpy_warnings",
    )
    def test_corrcoef_scalar(self):
        ia = dpnp.array(5)
        a = dpnp.asnumpy(ia)

        result = dpnp.corrcoef(ia)
        expected = numpy.corrcoef(a)
        assert_dtype_allclose(result, expected)


class TestCorrelate:
    @pytest.mark.parametrize(
        "a, v", [([1], [1, 2, 3]), ([1, 2, 3], [1]), ([1, 2, 3], [1, 2])]
    )
    @pytest.mark.parametrize("mode", [None, "full", "valid", "same"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_correlate(self, a, v, mode, dtype):
        an = numpy.array(a, dtype=dtype)
        vn = numpy.array(v, dtype=dtype)
        ad = dpnp.array(an)
        vd = dpnp.array(vn)

        if mode is None:
            expected = numpy.correlate(an, vn)
            result = dpnp.correlate(ad, vd)
        else:
            expected = numpy.correlate(an, vn, mode=mode)
            result = dpnp.correlate(ad, vd, mode=mode)

        assert_dtype_allclose(result, expected)

    def test_correlate_mode_error(self):
        a = dpnp.arange(5)
        v = dpnp.arange(3)

        # invalid mode
        with pytest.raises(ValueError):
            dpnp.correlate(a, v, mode="unknown")

    @pytest.mark.parametrize("a, v", [([], [1]), ([1], []), ([], [])])
    def test_correlate_empty(self, a, v):
        a = dpnp.asarray(a)
        v = dpnp.asarray(v)

        with pytest.raises(ValueError):
            dpnp.correlate(a, v)

    @pytest.mark.parametrize(
        "a, v",
        [
            ([[1, 2], [2, 3]], [1]),
            ([1], [[1, 2], [2, 3]]),
            ([[1, 2], [2, 3]], [[1, 2], [2, 3]]),
        ],
    )
    def test_correlate_shape_error(self, a, v):
        a = dpnp.asarray(a)
        v = dpnp.asarray(v)

        with pytest.raises(ValueError):
            dpnp.correlate(a, v)

    @pytest.mark.parametrize("size", [2, 10**1, 10**2, 10**3, 10**4, 10**5])
    def test_correlate_different_sizes(self, size):
        a = numpy.random.rand(size).astype(numpy.float32)
        v = numpy.random.rand(size // 2).astype(numpy.float32)

        ad = dpnp.array(a)
        vd = dpnp.array(v)

        expected = numpy.correlate(a, v)
        result = dpnp.correlate(ad, vd)

        assert_dtype_allclose(result, expected, factor=20)

    def test_correlate_another_sycl_queue(self):
        a = dpnp.arange(5, sycl_queue=dpctl.SyclQueue())
        v = dpnp.arange(3, sycl_queue=dpctl.SyclQueue())

        with pytest.raises(ValueError):
            dpnp.correlate(a, v)


class TestCov:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_false_rowvar_dtype(self, dtype):
        a = numpy.array([[0, 2], [1, 1], [2, 0]], dtype=dtype)
        ia = dpnp.array(a)

        assert_allclose(dpnp.cov(ia.T), dpnp.cov(ia, rowvar=False))
        assert_allclose(dpnp.cov(ia, rowvar=False), numpy.cov(a, rowvar=False))

    # numpy 2.2 properly transposes 2d array when rowvar=False
    @with_requires("numpy>=2.2")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_false_rowvar_1x3(self):
        a = numpy.array([[0, 1, 2]])
        ia = dpnp.array(a)

        expected = numpy.cov(a, rowvar=False)
        result = dpnp.cov(ia, rowvar=False)
        assert_allclose(expected, result)

    # numpy 2.2 properly transposes 2d array when rowvar=False
    @with_requires("numpy>=2.2")
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_true_rowvar(self):
        a = numpy.ones((3, 1))
        ia = dpnp.array(a)

        expected = numpy.cov(a, ddof=0, rowvar=True)
        result = dpnp.cov(ia, ddof=0, rowvar=True)
        assert_allclose(expected, result)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize(
    "v",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    ],
    ids=[
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
        "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]",
    ],
)
def test_ptp(v, axis):
    a = numpy.array(v)
    ia = dpnp.array(a)
    expected = numpy.ptp(a, axis)
    result = dpnp.ptp(ia, axis)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize(
    "v",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    ],
    ids=[
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
        "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]",
    ],
)
def test_ptp_out(v, axis):
    a = numpy.array(v)
    ia = dpnp.array(a)
    expected = numpy.ptp(a, axis)
    result = dpnp.array(numpy.empty_like(expected))
    dpnp.ptp(ia, axis, out=result)
    assert_array_equal(result, expected)
