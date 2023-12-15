import dpctl.tensor as dpt
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)

import dpnp

from .helper import assert_dtype_allclose, get_all_dtypes


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

    def test_mean_NotImplemented(func):
        ia = dpnp.arange(5)
        with pytest.raises(NotImplementedError):
            dpnp.mean(ia, where=False)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
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
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_none=True, no_bool=True, no_complex=True)
)
def test_nanvar(array, dtype):
    dtype = dpnp.default_float_type()
    a = numpy.array(array, dtype=dtype)
    ia = dpnp.array(a)
    for ddof in range(a.ndim):
        expected = numpy.nanvar(a, ddof=ddof)
        result = dpnp.nanvar(ia, ddof=ddof)
        assert_allclose(expected, result, rtol=1e-06)

    expected = numpy.nanvar(a, axis=None, ddof=0)
    result = dpnp.nanvar(ia, axis=None, ddof=0)
    assert_allclose(expected, result, rtol=1e-06)


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
