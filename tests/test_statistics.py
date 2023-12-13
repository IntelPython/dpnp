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
    get_float_complex_dtypes,
    has_support_aspect64,
)


@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_none=True, no_bool=True, no_complex=True)
)
@pytest.mark.parametrize("size", [2, 4, 8, 16, 3, 9, 27, 81])
def test_median(dtype, size):
    a = numpy.arange(size, dtype=dtype)
    ia = dpnp.array(a)

    np_res = numpy.median(a)
    dpnp_res = dpnp.median(ia)

    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["max", "min"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
def test_max_min(func, axis, keepdims, dtype):
    a = numpy.arange(768, dtype=dtype).reshape((4, 4, 6, 8))
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
    dpnp_res = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["max", "min"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_max_min_bool(func, axis, keepdims):
    a = numpy.arange(2, dtype=dpnp.bool)
    a = numpy.tile(a, (2, 2))
    ia = dpnp.array(a)

    np_res = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
    dpnp_res = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)

    assert dpnp_res.shape == np_res.shape
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["max", "min"])
def test_max_min_out(func):
    a = numpy.arange(6).reshape((2, 3))
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


@pytest.mark.parametrize("func", ["max", "min"])
def test_max_min_NotImplemented(func):
    ia = dpnp.arange(5)

    with pytest.raises(NotImplementedError):
        getattr(dpnp, func)(ia, where=False)
    with pytest.raises(NotImplementedError):
        getattr(dpnp, func)(ia, initial=6)


class TestMean:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_mean_axis_tuple(self, dtype):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.mean(dp_array, axis=(0, 1))
        expected = numpy.mean(np_array, axis=(0, 1))
        assert_allclose(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    def test_mean_out(self, dtype, axis):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.mean(np_array, axis=axis)
        result = dpnp.empty_like(dpnp.asarray(expected))
        dpnp.mean(dp_array, axis=axis, out=result)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_mean_dtype(self, dtype):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype="i4")
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.mean(np_array, dtype=dtype)
        result = dpnp.mean(dp_array, dtype=dtype)
        assert_allclose(expected, result)

    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 0), (0, 3)])
    def test_mean_empty(self, axis, shape):
        dp_array = dpnp.empty(shape, dtype=dpnp.int64)
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.mean(dp_array, axis=axis)
        expected = numpy.mean(np_array, axis=axis)
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

    def test_mean_NotImplemented(self):
        ia = dpnp.arange(5)
        with pytest.raises(NotImplementedError):
            dpnp.mean(ia, where=False)


class TestVar:
    @pytest.mark.usefixtures("suppress_divide_invalid_dof_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("ddof", [0, 0.5, 1, 1.5, 2])
    def test_var(self, dtype, axis, keepdims, ddof):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.var(np_array, axis=axis, keepdims=keepdims, ddof=ddof)
        result = dpnp.var(dp_array, axis=axis, keepdims=keepdims, ddof=ddof)

        if axis == 0 and ddof == 2:
            assert dpnp.all(dpnp.isnan(result))
        else:
            assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_invalid_dof_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("ddof", [0, 1])
    def test_var_out(self, dtype, axis, ddof):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.var(np_array, axis=axis, ddof=ddof)
        if has_support_aspect64():
            res_dtype = expected.dtype
        else:
            res_dtype = dpnp.default_float_type(dp_array.device)
        result = dpnp.empty(expected.shape, dtype=res_dtype)
        dpnp.var(dp_array, axis=axis, out=result, ddof=ddof)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_dof_invalid_numpy_warnings")
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 0), (0, 3)])
    def test_var_empty(self, axis, shape):
        dp_array = dpnp.empty(shape, dtype=dpnp.int64)
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.var(dp_array, axis=axis)
        expected = numpy.var(np_array, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_var_strided(self, dtype):
        dp_array = dpnp.array([-2, -1, 0, 1, 0, 2], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.var(dp_array[::-1])
        expected = numpy.var(np_array[::-1])
        assert_dtype_allclose(result, expected)

        result = dpnp.var(dp_array[::2])
        expected = numpy.var(np_array[::2])
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("dt_in", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("dt_out", get_float_complex_dtypes())
    def test_var_dtype(self, dt_in, dt_out):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dt_in)
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.var(np_array, dtype=dt_out)
        result = dpnp.var(dp_array, dtype=dt_out)
        assert expected.dtype == result.dtype
        assert_allclose(result, expected, rtol=1e-06)

    def test_var_scalar(self):
        dp_array = dpnp.array(5)
        np_array = dpnp.asnumpy(dp_array)

        result = dp_array.var()
        expected = np_array.var()
        assert_allclose(expected, result)

    def test_var_error(self):
        ia = dpnp.arange(5)
        # where keyword is not implemented
        with pytest.raises(NotImplementedError):
            dpnp.var(ia, where=False)

        # ddof should be an integer
        with pytest.raises(TypeError):
            dpnp.var(ia, ddof="1")


class TestStd:
    @pytest.mark.usefixtures("suppress_divide_invalid_dof_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("ddof", [0, 0.5, 1, 1.5, 2])
    def test_std(self, dtype, axis, keepdims, ddof):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.std(np_array, axis=axis, keepdims=keepdims, ddof=ddof)
        result = dpnp.std(dp_array, axis=axis, keepdims=keepdims, ddof=ddof)
        if axis == 0 and ddof == 2:
            assert dpnp.all(dpnp.isnan(result))
        else:
            assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_divide_invalid_dof_numpy_warnings")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("ddof", [0, 1])
    def test_std_out(self, dtype, axis, ddof):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.std(np_array, axis=axis, ddof=ddof)
        if has_support_aspect64():
            res_dtype = expected.dtype
        else:
            res_dtype = dpnp.default_float_type(dp_array.device)
        result = dpnp.empty(expected.shape, dtype=res_dtype)
        dpnp.std(dp_array, axis=axis, out=result, ddof=ddof)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_dof_invalid_numpy_warnings")
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 0), (0, 3)])
    def test_std_empty(self, axis, shape):
        dp_array = dpnp.empty(shape, dtype=dpnp.int64)
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.std(dp_array, axis=axis)
        expected = numpy.std(np_array, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_std_strided(self, dtype):
        dp_array = dpnp.array([-2, -1, 0, 1, 0, 2], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        result = dpnp.std(dp_array[::-1])
        expected = numpy.std(np_array[::-1])
        assert_dtype_allclose(result, expected)

        result = dpnp.std(dp_array[::2])
        expected = numpy.std(np_array[::2])
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("dt_in", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("dt_out", get_float_complex_dtypes())
    def test_std_dtype(self, dt_in, dt_out):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dt_in)
        np_array = dpnp.asnumpy(dp_array)

        expected = numpy.std(np_array, dtype=dt_out)
        result = dpnp.std(dp_array, dtype=dt_out)
        assert expected.dtype == result.dtype
        assert_allclose(result, expected, rtol=1e-6)

    def test_std_scalar(self):
        dp_array = dpnp.array(5)
        np_array = dpnp.asnumpy(dp_array)

        result = dp_array.std()
        expected = np_array.std()
        assert_dtype_allclose(result, expected)

    def test_std_error(self):
        ia = dpnp.arange(5)
        # where keyword is not implemented
        with pytest.raises(NotImplementedError):
            dpnp.std(ia, where=False)

        # ddof should be an integer
        with pytest.raises(TypeError):
            dpnp.std(ia, ddof="1")


class TestNanVar:
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
    @pytest.mark.usefixtures("suppress_dof_invalid_numpy_warnings")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_nanvar(self, array, dtype):
        try:
            a = numpy.array(array, dtype=dtype)
        except:
            pytest.skip("floating datat type is needed to store NaN")
        ia = dpnp.array(a)
        for ddof in range(a.ndim):
            expected = numpy.nanvar(a, ddof=ddof)
            result = dpnp.nanvar(ia, ddof=ddof)
            assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_dof_numpy_warning")
    @pytest.mark.parametrize("dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("axis", [None, 0, 1, 2, (0, 1), (1, 2)])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("ddof", [0, 0.5, 1, 1.5, 2, 3])
    def test_nanvar_out(self, dtype, axis, keepdims, ddof):
        a = numpy.arange(4 * 3 * 5, dtype=dtype)
        a[::2] = numpy.nan
        a = a.reshape(4, 3, 5)
        ia = dpnp.array(a)

        expected = numpy.nanvar(a, axis=axis, ddof=ddof, keepdims=keepdims)
        if has_support_aspect64():
            res_dtype = expected.dtype
        else:
            res_dtype = dpnp.default_float_type(ia.device)
        result = dpnp.empty(expected.shape, dtype=res_dtype)
        dpnp.nanvar(ia, out=result, axis=axis, ddof=ddof, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("dt_in", get_float_complex_dtypes())
    @pytest.mark.parametrize("dt_out", get_float_complex_dtypes())
    def test_nanvar_dtype(self, dt_in, dt_out):
        a = numpy.arange(4 * 3 * 5, dtype=dt_in)
        a[::2] = numpy.nan
        a = a.reshape(4, 3, 5)
        ia = dpnp.array(a)

        expected = numpy.nanvar(a, dtype=dt_out)
        result = dpnp.nanvar(ia, dtype=dt_out)
        assert_dtype_allclose(result, expected)

    def test_nanvar_error(self):
        ia = dpnp.arange(5, dtype=dpnp.float32)
        ia[0] = dpnp.nan
        # where keyword is not implemented
        with pytest.raises(NotImplementedError):
            dpnp.nanvar(ia, where=False)

        # dtype should be floating
        with pytest.raises(TypeError):
            dpnp.nanvar(ia, dtype=dpnp.int32)

        # out dtype should be inexact
        res = dpnp.empty((1,), dtype=dpnp.int32)
        with pytest.raises(TypeError):
            dpnp.nanvar(ia, out=res)

        # ddof should be an integer
        with pytest.raises(TypeError):
            dpnp.nanvar(ia, ddof="1")


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestBincount:
    @pytest.mark.parametrize(
        "array",
        [[1, 2, 3], [1, 2, 2, 1, 2, 4], [2, 2, 2, 2]],
        ids=["[1, 2, 3]", "[1, 2, 2, 1, 2, 4]", "[2, 2, 2, 2]"],
    )
    @pytest.mark.parametrize(
        "minlength", [0, 1, 3, 5], ids=["0", "1", "3", "5"]
    )
    def test_bincount_minlength(self, array, minlength):
        np_a = numpy.array(array)
        dpnp_a = dpnp.array(array)

        expected = numpy.bincount(np_a, minlength=minlength)
        result = dpnp.bincount(dpnp_a, minlength=minlength)
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "array", [[1, 2, 2, 1, 2, 4]], ids=["[1, 2, 2, 1, 2, 4]"]
    )
    @pytest.mark.parametrize(
        "weights",
        [None, [0.3, 0.5, 0.2, 0.7, 1.0, -0.6], [2, 2, 2, 2, 2, 2]],
        ids=["None", "[0.3, 0.5, 0.2, 0.7, 1., -0.6]", "[2, 2, 2, 2, 2, 2]"],
    )
    def test_bincount_weights(self, array, weights):
        np_a = numpy.array(array)
        dpnp_a = dpnp.array(array)

        expected = numpy.bincount(np_a, weights=weights)
        result = dpnp.bincount(dpnp_a, weights=weights)
        assert_allclose(expected, result)


@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
)
def test_cov_rowvar(dtype):
    a = dpnp.array([[0, 2], [1, 1], [2, 0]], dtype=dtype)
    b = numpy.array([[0, 2], [1, 1], [2, 0]], dtype=dtype)
    assert_allclose(dpnp.cov(a.T), dpnp.cov(a, rowvar=False))
    assert_allclose(numpy.cov(b, rowvar=False), dpnp.cov(a, rowvar=False))


@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
)
def test_cov_1D_rowvar(dtype):
    a = dpnp.array([[0, 1, 2]], dtype=dtype)
    b = numpy.array([[0, 1, 2]], dtype=dtype)
    assert_allclose(numpy.cov(b, rowvar=False), dpnp.cov(a, rowvar=False))


@pytest.mark.parametrize(
    "axis",
    [None, 0, 1],
    ids=["None", "0", "1"],
)
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
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "axis",
    [None, 0, 1],
    ids=["None", "0", "1"],
)
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
    assert_array_equal(expected, result)
