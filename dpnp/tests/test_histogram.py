import dpctl
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
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
    get_integer_dtypes,
    get_integer_float_dtypes,
    has_support_aspect64,
    numpy_version,
)


class TestDigitize:
    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    @pytest.mark.parametrize("right", [True, False])
    @pytest.mark.parametrize(
        "x, bins",
        [
            # Negative values
            (
                numpy.array([-5, -3, -1, 0, 1, 3, 5]),
                numpy.array([-4, -2, 0, 2, 4]),
            ),
            # Non-uniform bins
            (
                numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
                numpy.array([1, 4, 6, 7]),
            ),
            # Repeated elements
            (numpy.array([1, 2, 2, 3, 3, 3, 4, 5]), numpy.array([1, 2, 3, 4])),
        ],
    )
    def test_digitize(self, x, bins, dtype, right):
        x = get_abs_array(x, dtype)
        if numpy.issubdtype(dtype, numpy.unsignedinteger):
            min_bin = bins.min()
            if min_bin < 0:
                # bins should be monotonically increasing, cannot use get_abs_array
                bins -= min_bin
        bins = bins.astype(dtype)
        x_dp = dpnp.array(x)
        bins_dp = dpnp.array(bins)

        result = dpnp.digitize(x_dp, bins_dp, right=right)
        expected = numpy.digitize(x, bins, right=right)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("right", [True, False])
    def test_digitize_inf(self, dtype, right):
        x = numpy.array([-numpy.inf, -1, 0, 1, numpy.inf], dtype=dtype)
        bins = numpy.array([-2, -1, 0, 1, 2], dtype=dtype)
        x_dp = dpnp.array(x)
        bins_dp = dpnp.array(bins)

        result = dpnp.digitize(x_dp, bins_dp, right=right)
        expected = numpy.digitize(x, bins, right=right)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype_x", get_integer_float_dtypes())
    @pytest.mark.parametrize("dtype_bins", get_integer_float_dtypes())
    @pytest.mark.parametrize("right", [True, False])
    def test_digitize_diff_types(self, dtype_x, dtype_bins, right):
        x = numpy.array([1, 2, 3, 4, 5], dtype=dtype_x)
        bins = numpy.array([1, 3, 5], dtype=dtype_bins)
        x_dp = dpnp.array(x)
        bins_dp = dpnp.array(bins)

        result = dpnp.digitize(x_dp, bins_dp, right=right)
        expected = numpy.digitize(x, bins, right=right)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_integer_float_dtypes())
    @pytest.mark.parametrize(
        "x, bins",
        [
            # Empty array
            (numpy.array([]), numpy.array([1, 2, 3])),
            # Empty bins
            (numpy.array([1, 2, 3]), numpy.array([])),
        ],
    )
    def test_digitize_empty(self, x, bins, dtype):
        x = x.astype(dtype)
        bins = bins.astype(dtype)
        x_dp = dpnp.array(x)
        bins_dp = dpnp.array(bins)

        result = dpnp.digitize(x_dp, bins_dp)
        expected = numpy.digitize(x, bins)
        assert_dtype_allclose(result, expected)

    def test_digitize_error(self):
        x_dp = dpnp.array([1, 2, 3], dtype="float32")
        bins_dp = dpnp.array([1, 2, 3], dtype="float32")

        # unsupported type
        x_np = dpnp.asnumpy(x_dp)
        bins_np = dpnp.asnumpy(bins_dp)
        with pytest.raises(TypeError):
            dpnp.digitize(x_np, bins_dp)
        with pytest.raises(TypeError):
            dpnp.digitize(x_dp, bins_np)

        # bins ndim < 1
        bins_scalar = dpnp.array(1)
        with pytest.raises(ValueError):
            dpnp.digitize(x_dp, bins_scalar)


class TestHistogram:
    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_rand_data(self, dtype):
        n = 100
        v = numpy.random.rand(n).astype(dtype=dtype)
        iv = dpnp.array(v, dtype=dtype)

        expected_hist, _ = numpy.histogram(v)
        result_hist, _ = dpnp.histogram(iv)
        assert_array_equal(result_hist, expected_hist)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_linspace_data(self, dtype):
        v = numpy.linspace(0, 10, 100, dtype=dtype)
        iv = dpnp.array(v)

        expected_hist, _ = numpy.histogram(v)
        result_hist, _ = dpnp.histogram(iv)
        assert_array_equal(result_hist, expected_hist)

    @pytest.mark.parametrize(
        "data, bins_data",
        [
            pytest.param([1, 2, 3, 4], [1, 2], id="1d-1d"),
            pytest.param([1, 2], 1, id="1d-0d"),
        ],
    )
    def test_one_bin(self, data, bins_data):
        a = numpy.array(data)
        bins = numpy.array(bins_data)

        ia = dpnp.array(a)
        ibins = dpnp.array(bins)
        expected_hist, expected_edges = numpy.histogram(a, bins=bins)
        result_hist, result_edges = dpnp.histogram(ia, bins=ibins)
        assert_array_equal(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_zero_bin(self, xp):
        a = xp.array([1, 2])
        assert_raises(ValueError, xp.histogram, a, bins=0)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_density(self, dtype):
        n = 100
        v = numpy.random.rand(n).astype(dtype=dtype)
        iv = dpnp.array(v, dtype=dtype)

        expected_hist, expected_edges = numpy.histogram(v, density=True)
        result_hist, result_edges = dpnp.histogram(iv, density=True)

        if numpy.issubdtype(dtype, numpy.inexact):
            tol = 4 * numpy.finfo(dtype).resolution
            assert_allclose(result_hist, expected_hist, rtol=tol, atol=tol)
            assert_allclose(result_edges, expected_edges, rtol=tol, atol=tol)
        else:
            assert_dtype_allclose(result_hist, expected_hist)
            assert_dtype_allclose(result_edges, expected_edges)

    @pytest.mark.parametrize("density", [True, False])
    def test_bin_density(self, density):
        bins = [0, 1, 3, 6, 10]
        v = numpy.arange(10)
        iv = dpnp.array(v)

        expected_hist, expected_edges = numpy.histogram(
            v, bins, density=density
        )
        result_hist, result_edges = dpnp.histogram(iv, bins, density=density)
        assert_allclose(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

    @pytest.mark.parametrize(
        "bins", [[0, 1, 3, 6, numpy.inf], [0.5, 1.5, numpy.inf]]
    )
    def test_bin_inf(self, bins):
        v = numpy.arange(10)
        iv = dpnp.array(v)

        expected_hist, expected_edges = numpy.histogram(v, bins, density=True)
        result_hist, result_edges = dpnp.histogram(iv, bins, density=True)
        assert_allclose(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

    @pytest.mark.parametrize("range", [[0, 9], [1, 10]], ids=["lower", "upper"])
    def test_outliers(self, range):
        a = numpy.arange(10) + 0.5
        ia = dpnp.array(a)

        expected_hist, expected_edges = numpy.histogram(a, range=range)
        result_hist, result_edges = dpnp.histogram(ia, range=range)
        assert_allclose(result_hist, expected_hist)
        assert_allclose(result_edges, expected_edges)

    def test_outliers_normalization_weights(self):
        range = [1, 9]
        a = numpy.arange(10) + 0.5
        ia = dpnp.array(a)

        # Normalization
        expected_hist, expected_edges = numpy.histogram(a, range, density=True)
        result_hist, result_edges = dpnp.histogram(ia, range, density=True)
        assert_allclose(result_hist, expected_hist)
        assert_allclose(result_edges, expected_edges)

        w = numpy.arange(10) + 0.5
        iw = dpnp.array(w)

        # Weights
        expected_hist, expected_edges = numpy.histogram(
            a, range, weights=w, density=True
        )
        result_hist, result_edges = dpnp.histogram(
            ia, range, weights=iw, density=True
        )
        assert_allclose(result_hist, expected_hist)
        assert_allclose(result_edges, expected_edges)

        expected_hist, expected_edges = numpy.histogram(
            a, bins=8, range=range, weights=w, density=True
        )
        result_hist, result_edges = dpnp.histogram(
            ia, bins=8, range=range, weights=iw, density=True
        )
        assert_allclose(result_hist, expected_hist)
        assert_allclose(result_edges, expected_edges)

    @pytest.mark.parametrize("density", [True, False])
    def test_weights(self, density):
        v = numpy.random.rand(100)
        w = numpy.ones(100) * 5

        iv = dpnp.array(v)
        iw = dpnp.array(w)

        expected_hist, expected_edges = numpy.histogram(
            v, weights=w, density=density
        )
        result_hist, result_edges = dpnp.histogram(
            iv, weights=iw, density=density
        )
        assert_dtype_allclose(result_hist, expected_hist)
        assert_dtype_allclose(result_edges, expected_edges)

    @pytest.mark.parametrize("dt", get_integer_dtypes(all_int_types=True))
    def test_integer_weights(self, dt):
        v = numpy.array([1, 2, 2, 4])
        w = numpy.array([4, 3, 2, 1], dtype=dt)

        iv = dpnp.array(v)
        iw = dpnp.array(w)

        expected_hist, expected_edges = numpy.histogram(v, bins=4, weights=w)
        result_hist, result_edges = dpnp.histogram(iv, bins=4, weights=iw)
        assert_array_equal(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

    def test_weights_non_uniform_bin_widths(self):
        bins = [0, 1, 3, 6, 10]
        v = numpy.arange(9)
        w = numpy.array([2, 1, 1, 1, 1, 1, 1, 1, 1])

        iv = dpnp.array(v)
        iw = dpnp.array(w)

        expected_hist, expected_edges = numpy.histogram(
            v, bins, weights=w, density=True
        )
        result_hist, result_edges = dpnp.histogram(
            iv, bins, weights=iw, density=True
        )
        assert_allclose(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

    def test_weights_complex_dtype(self):
        bins = [0, 2, 3]
        v = numpy.array([1.3, 2.5, 2.3])
        w = numpy.array([1, -1, 2]) + 1j * numpy.array([2, 1, 2])

        iv = dpnp.array(v)
        iw = dpnp.array(w)

        # with custom bins
        expected_hist, expected_edges = numpy.histogram(v, bins, weights=w)
        result_hist, result_edges = dpnp.histogram(iv, bins, weights=iw)
        assert_array_equal(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

        # with even bins
        expected_hist, expected_edges = numpy.histogram(
            v, bins=2, range=[1, 3], weights=w
        )
        result_hist, result_edges = dpnp.histogram(
            iv, bins=2, range=[1, 3], weights=iw
        )
        assert_array_equal(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

    def test_no_side_effects(self):
        v = dpnp.array([1.3, 2.5, 2.3])
        copy_v = v.copy()

        # check that ensures that values passed to ``histogram`` are unchanged
        _, _ = dpnp.histogram(v, range=[-10, 10], bins=100)
        assert (v == copy_v).all()

    def test_empty(self):
        expected_hist, expected_edges = numpy.histogram(
            numpy.array([]), bins=([0, 1])
        )
        result_hist, result_edges = dpnp.histogram(
            dpnp.array([]), bins=([0, 1])
        )
        assert_array_equal(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error_binnum_type(self, xp):
        vals = xp.linspace(0.0, 1.0, num=100)

        # `bins` must be an integer, a string, or an array
        _, _ = xp.histogram(vals, 5)
        assert_raises(TypeError, xp.histogram, vals, 2.4)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_finite_range(self, xp):
        vals = xp.linspace(0.0, 1.0, num=100)

        # normal ranges should be fine
        _, _ = xp.histogram(vals, range=[0.25, 0.75])
        assert_raises(ValueError, xp.histogram, vals, range=[xp.nan, 0.75])
        assert_raises(ValueError, xp.histogram, vals, range=[0.25, xp.inf])

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_range(self, xp):
        # start of range must be < end of range
        vals = xp.linspace(0.0, 1.0, num=100)
        with assert_raises_regex(ValueError, "max must be larger than"):
            xp.histogram(vals, range=[0.1, 0.01])

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_range_size(self, xp):
        # range shape must be [2]
        vals = xp.linspace(0.0, 1.0, num=100)
        assert_raises(ValueError, xp.histogram, vals, range=[[0, 1, 2]])

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("inf_val", [-numpy.inf, numpy.inf])
    def test_infinite_edge(self, xp, inf_val):
        v = xp.array([0.5, 1.5, inf_val])
        min, max = v.min(), v.max()

        # both first and last ranges must be finite
        with assert_raises_regex(
            ValueError,
            f"autodetected range of \\[{min}, {max}\\] is not finite",
        ):
            xp.histogram(v)

    def test_bin_edge_cases(self):
        v = dpnp.array([337, 404, 739, 806, 1007, 1811, 2012])

        hist, edges = dpnp.histogram(v, bins=8296, range=(2, 2280))
        mask = hist > 0
        left_edges = edges[:-1][mask]
        right_edges = edges[1:][mask]

        # floating-point computations correctly place edge cases
        for x, left, right in zip(v, left_edges, right_edges):
            assert x >= left
            assert x < right

    @pytest.mark.skipif(not has_support_aspect64(), reason="fp64 required")
    def test_last_bin_inclusive_range(self):
        v = numpy.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
        iv = dpnp.array(v)

        expected_hist, expected_edges = numpy.histogram(
            v, bins=30, range=(-0.5, 5)
        )
        result_hist, result_edges = dpnp.histogram(iv, bins=30, range=(-0.5, 5))
        assert_allclose(result_hist, expected_hist)
        assert_allclose(result_edges, expected_edges)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_bin_array_dims(self, xp):
        # gracefully handle bins object > 1 dimension
        vals = xp.linspace(0.0, 1.0, num=100)
        bins = xp.array([[0, 0.5], [0.6, 1.0]])
        with assert_raises_regex(ValueError, "must be 1d"):
            xp.histogram(vals, bins=bins)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_unsigned_monotonicity_check(self, xp):
        # bins must increase monotonically when bins contain unsigned values
        arr = xp.array([2])
        bins = xp.array([1, 3, 1], dtype="uint64")
        with assert_raises(ValueError):
            xp.histogram(arr, bins=bins)

    def test_nan_values(self):
        one_nan = numpy.array([0, 1, numpy.nan])
        all_nan = numpy.array([numpy.nan, numpy.nan])

        ione_nan = dpnp.array(one_nan)
        iall_nan = dpnp.array(all_nan)

        # NaN is not counted
        expected_hist, expected_edges = numpy.histogram(one_nan, bins=[0, 1])
        result_hist, result_edges = dpnp.histogram(ione_nan, bins=[0, 1])
        assert_array_equal(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

        # NaN is not counted
        expected_hist, expected_edges = numpy.histogram(all_nan, bins=[0, 1])
        result_hist, result_edges = dpnp.histogram(iall_nan, bins=[0, 1])
        assert_array_equal(result_hist, expected_hist)
        assert_array_equal(result_edges, expected_edges)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.byte, numpy.short, numpy.intc, numpy.int_, numpy.longlong],
    )
    def test_signed_overflow_bounds(self, dtype):
        exponent = 8 * numpy.dtype(dtype).itemsize - 1
        v = numpy.array([-(2**exponent) + 4, 2**exponent - 4], dtype=dtype)
        iv = dpnp.array(v)

        expected_hist, expected_edges = numpy.histogram(v, bins=2)
        result_hist, result_edges = dpnp.histogram(iv, bins=2)
        assert_array_equal(result_hist, expected_hist)
        assert_allclose(result_edges, expected_edges)

    def test_string_bins_not_implemented(self):
        v = dpnp.arange(5)

        # numpy support string bins, but not dpnp
        _, _ = numpy.histogram(v.asnumpy(), bins="auto")
        with assert_raises(NotImplementedError):
            dpnp.histogram(v, bins="auto")

    def test_bins_another_sycl_queue(self):
        v = dpnp.arange(7, 12, sycl_queue=dpctl.SyclQueue())
        bins = dpnp.arange(4, sycl_queue=dpctl.SyclQueue())
        with assert_raises(ValueError):
            dpnp.histogram(v, bins=bins)

    def test_weights_another_sycl_queue(self):
        v = dpnp.arange(5, sycl_queue=dpctl.SyclQueue())
        w = dpnp.arange(7, 12, sycl_queue=dpctl.SyclQueue())
        with assert_raises(ValueError):
            dpnp.histogram(v, weights=w)

    @pytest.mark.parametrize(
        "bins_count",
        [10, 10**2, 10**3, 10**4, 10**5, 10**6],
    )
    def test_different_bins_amount(self, bins_count):
        v = numpy.linspace(0, bins_count, bins_count, dtype=numpy.float32)
        iv = dpnp.array(v)

        expected_hist, expected_edges = numpy.histogram(v, bins=bins_count)
        result_hist, result_edges = dpnp.histogram(iv, bins=bins_count)
        assert_array_equal(result_hist, expected_hist)
        assert_allclose(result_edges, expected_edges)


class TestHistogramBinEdges:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_basic(self, dtype):
        bins = [1, 2]
        v = numpy.array([1, 2, 3, 4], dtype=dtype)
        iv = dpnp.array(v)

        expected_edges = numpy.histogram_bin_edges(v, bins=bins)
        result_edges = dpnp.histogram_bin_edges(iv, bins=bins)
        assert_array_equal(result_edges, expected_edges)

    @pytest.mark.parametrize("range", [(-0.5, 5), (0, 1)])
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_range(self, range, dtype):
        bins = 30
        v = numpy.array(
            [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0], dtype=dtype
        )
        iv = dpnp.array(v)

        expected_edges = numpy.histogram_bin_edges(v, bins=bins, range=range)
        result_edges = dpnp.histogram_bin_edges(iv, bins=bins, range=range)
        assert_dtype_allclose(result_edges, expected_edges)


class TestBincount:
    @pytest.mark.parametrize("dt", get_integer_dtypes() + [numpy.bool_])
    def test_rand_data(self, dt):
        v = generate_random_numpy_array(100, dtype=dt, low=0)
        iv = dpnp.array(v)

        if numpy.issubdtype(dt, numpy.uint64) and numpy_version() < "2.2.4":
            v = v.astype(numpy.int64)

        expected_hist = numpy.bincount(v)
        result_hist = dpnp.bincount(iv)
        assert_array_equal(result_hist, expected_hist)

    @pytest.mark.parametrize("dt", get_integer_dtypes())
    def test_arange_data(self, dt):
        v = numpy.arange(100, dtype=dt)
        iv = dpnp.array(v)

        if numpy.issubdtype(dt, numpy.uint64) and numpy_version() < "2.2.4":
            v = v.astype(numpy.int64)

        expected_hist = numpy.bincount(v)
        result_hist = dpnp.bincount(iv)
        assert_array_equal(result_hist, expected_hist)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_negative_values(self, xp):
        x = xp.array([-1, 2])
        assert_raises(ValueError, xp.bincount, x)

    def test_no_side_effects(self):
        v = dpnp.array([1, 2, 3], dtype=dpnp.int64)
        copy_v = v.copy()

        # check that ensures that values passed to ``bincount`` are unchanged
        _ = dpnp.bincount(v)
        assert (v == copy_v).all()

    def test_weights_another_sycl_queue(self):
        v = dpnp.arange(5, sycl_queue=dpctl.SyclQueue())
        w = dpnp.arange(7, 12, sycl_queue=dpctl.SyclQueue())
        with assert_raises(ValueError):
            dpnp.bincount(v, weights=w)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", get_float_complex_dtypes())
    def test_data_unsupported_dtype(self, xp, dt):
        v = xp.arange(5, dtype=dt)
        assert_raises(TypeError, xp.bincount, v)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_weights_unsupported_dtype(self, xp, dt):
        v = xp.arange(5)
        w = xp.arange(5, dtype=dt)
        assert_raises((TypeError, ValueError), xp.bincount, v, weights=w)

    @pytest.mark.parametrize(
        "bins_count",
        [10, 10**2, 10**3, 10**4, 10**5, 10**6],
    )
    def test_different_bins_amount(self, bins_count):
        v = numpy.arange(0, bins_count, dtype=int)
        iv = dpnp.array(v)

        expected_hist = numpy.bincount(v)
        result_hist = dpnp.bincount(iv)
        assert_array_equal(result_hist, expected_hist)

    @pytest.mark.parametrize(
        "array",
        [[1, 2, 3], [1, 2, 2, 1, 2, 4], [2, 2, 2, 2]],
        ids=["size=3", "size=6", "size=4"],
    )
    @pytest.mark.parametrize("minlength", [0, 1, 3, 5])
    def test_minlength(self, array, minlength):
        a = numpy.array(array)
        ia = dpnp.array(a)

        expected = numpy.bincount(a, minlength=minlength)
        result = dpnp.bincount(ia, minlength=minlength)
        assert_allclose(result, expected)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize(
        "xp",
        [
            dpnp,
            pytest.param(
                numpy,
                marks=pytest.mark.xfail(
                    numpy_version() < "2.3.0",
                    reason="numpy deprecates but accepts that",
                    strict=True,
                ),
            ),
        ],
    )
    def test_minlength_none(self, xp):
        a = xp.array([1, 2, 3])
        assert_raises_regex(
            TypeError,
            "use 0 instead of None for minlength",
            xp.bincount,
            a,
            minlength=None,
        )

    @pytest.mark.parametrize(
        "weights",
        [None, [0.3, 0.5, 0, 0.7, 1.0, -0.6], [2, 2, 2, 2, 2, 2]],
        ids=["None", "float_data", "int_data"],
    )
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_weights(self, weights, dt):
        a = numpy.array([1, 2, 2, 1, 2, 4])
        ia = dpnp.array(a)
        w = iw = None
        if weights is not None:
            w = numpy.array(weights, dtype=dt)
            iw = dpnp.array(w)

        expected = numpy.bincount(a, weights=w)
        result = dpnp.bincount(ia, weights=iw)
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "data",
        [numpy.arange(5), 3, [2, 1]],
        ids=["numpy.ndarray", "scalar", "list"],
    )
    def test_unsupported_data_weights(self, data):
        # check input array
        msg = "An array must be any of supported type"
        assert_raises_regex(TypeError, msg, dpnp.bincount, data)

        # check array of weights
        a = dpnp.ones(5, dtype=dpnp.int32)
        assert_raises_regex(TypeError, msg, dpnp.bincount, a, weights=data)


class TestHistogramDd:
    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_rand_data(self, dtype):
        n = 100
        dims = 3
        v = numpy.random.rand(n, dims).astype(dtype=dtype)
        iv = dpnp.array(v, dtype=dtype)

        expected_hist, _ = numpy.histogramdd(v)
        result_hist, _ = dpnp.histogramdd(iv)
        assert_array_equal(result_hist, expected_hist)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_linspace_data(self, dtype):
        n = 100
        dims = 3
        v = numpy.linspace(0, 10, n * dims, dtype=dtype).reshape(n, dims)
        iv = dpnp.array(v)

        expected_hist, _ = numpy.histogramdd(v)
        result_hist, _ = dpnp.histogramdd(iv)
        assert_array_equal(result_hist, expected_hist)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_bin_float(self, xp):
        a = xp.array([[1, 2]])
        assert_raises(ValueError, xp.histogramdd, a, bins=0.1)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_bin_2d_array(self, xp):
        a = xp.array([[1, 2]])
        assert_raises(ValueError, xp.histogramdd, a, bins=[[[10]], 10])

    @pytest.mark.parametrize(
        "bins",
        [
            11,
            [11] * 3,
            [[0, 20, 40, 60, 80, 100]] * 3,
            [[0, 20, 40, 60, 80, 300]] * 3,
        ],
    )
    def test_bins(self, bins):
        n = 100
        dims = 3
        v = numpy.arange(100 * 3).reshape(n, dims)
        iv = dpnp.array(v)

        bins_dpnp = bins
        if isinstance(bins, list):
            if isinstance(bins[0], list):
                bins = [numpy.array(b) for b in bins]
                bins_dpnp = [dpnp.array(b) for b in bins]

        expected_hist, expected_edges = numpy.histogramdd(v, bins)
        result_hist, result_edges = dpnp.histogramdd(iv, bins_dpnp)
        assert_allclose(result_hist, expected_hist)
        for x, y in zip(result_edges, expected_edges):
            assert_allclose(x, y)

    def test_no_side_effects(self):
        v = dpnp.array([[1.3, 2.5, 2.3]])
        copy_v = v.copy()

        # check that ensures that values passed to ``histogramdd`` are unchanged
        _, _ = dpnp.histogramdd(v)
        assert (v == copy_v).all()

    @pytest.mark.parametrize("data", [[], 1, [0, 1, 1, 3, 3]])
    def test_01d(self, data):
        a = numpy.array(data)
        ia = dpnp.array(a)

        expected_hist, expected_edges = numpy.histogramdd(a)
        result_hist, result_edges = dpnp.histogramdd(ia)

        assert_allclose(result_hist, expected_hist)
        for x, y in zip(result_edges, expected_edges):
            assert_allclose(x, y)

    def test_3d(self):
        a = dpnp.ones((10, 10, 10))

        with assert_raises_regex(ValueError, "no more than 2 dimensions"):
            dpnp.histogramdd(a)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_finite_range(self, xp):
        vals = xp.linspace(0.0, 1.0, num=100)

        # normal ranges should be fine
        _, _ = xp.histogramdd(vals, range=[[0.25, 0.75]])
        assert_raises(ValueError, xp.histogramdd, vals, range=[[xp.nan, 0.75]])
        assert_raises(ValueError, xp.histogramdd, vals, range=[[0.25, xp.inf]])

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_range(self, xp):
        # start of range must be < end of range
        vals = xp.linspace(0.0, 1.0, num=100)
        with assert_raises_regex(ValueError, "max must be larger than"):
            xp.histogramdd(vals, range=[[0.1, 0.01]])

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_range_dims(self, xp):
        # start of range must be < end of range
        vals = xp.linspace(0.0, 1.0, num=100)
        assert_raises(ValueError, xp.histogramdd, vals, range=[[0, 1]] * 2)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_range_size(self, xp):
        # range shape must be [2, 2]
        x = y = xp.linspace(0.0, 1.0, num=100)
        assert_raises(ValueError, xp.histogramdd, x, y, range=[[0, 1, 2]])

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("inf_val", [-numpy.inf, numpy.inf])
    def test_infinite_edge(self, xp, inf_val):
        v = xp.array([0.5, 1.5, inf_val])
        min, max = v.min(), v.max()

        # both first and last ranges must be finite
        with assert_raises_regex(
            ValueError,
            f"autodetected range of \\[{min}, {max}\\] is not finite",
        ):
            xp.histogramdd(v)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_unsigned_monotonicity_check(self, xp):
        # bins must increase monotonically when bins contain unsigned values
        arr = xp.array([2])
        bins = [xp.array([1, 3, 1], dtype="uint64")]
        with assert_raises(ValueError):
            xp.histogramdd(arr, bins=bins)

    def test_nan_values(self):
        one_nan = numpy.array([0, 1, numpy.nan])
        all_nan = numpy.array([numpy.nan, numpy.nan])

        ione_nan = dpnp.array(one_nan)
        iall_nan = dpnp.array(all_nan)

        # NaN is not counted
        expected_hist, expected_edges = numpy.histogramdd(
            one_nan, bins=[[0, 1]]
        )
        result_hist, result_edges = dpnp.histogramdd(ione_nan, bins=[[0, 1]])
        assert_allclose(result_hist, expected_hist)
        # dpnp returns both result_hist and result_edges as float64 while
        # numpy returns result_hist as float64 but result_edges as int64
        for x, y in zip(result_edges, expected_edges):
            assert_allclose(x, y, strict=False)

        # NaN is not counted
        expected_hist, expected_edges = numpy.histogramdd(
            all_nan, bins=[[0, 1]]
        )
        result_hist, result_edges = dpnp.histogramdd(iall_nan, bins=[[0, 1]])
        assert_allclose(result_hist, expected_hist)
        # dpnp returns both result_hist and result_edges as float64 while
        # numpy returns result_hist as float64 but result_edges as int64
        for x, y in zip(result_edges, expected_edges):
            assert_allclose(x, y, strict=False)

    def test_bins_another_sycl_queue(self):
        v = dpnp.arange(7, 12, sycl_queue=dpctl.SyclQueue())
        bins = dpnp.arange(4, sycl_queue=dpctl.SyclQueue())
        with assert_raises(ValueError):
            dpnp.histogramdd(v, bins=[bins])

    def test_sample_array_like(self):
        v = [0, 1, 2, 3, 4]
        with assert_raises(TypeError):
            dpnp.histogramdd(v)

    def test_weights_array_like(self):
        v = dpnp.arange(5)
        w = [1, 2, 3, 4, 5]
        with assert_raises(TypeError):
            dpnp.histogramdd(v, weights=w)

    def test_weights_another_sycl_queue(self):
        v = dpnp.arange(5, sycl_queue=dpctl.SyclQueue())
        w = dpnp.arange(7, 12, sycl_queue=dpctl.SyclQueue())
        with assert_raises(ValueError):
            dpnp.histogramdd(v, weights=w)

    @pytest.mark.parametrize(
        "bins_count",
        [10, 10**2, 10**3, 10**4, 10**5, 10**6],
    )
    def test_different_bins_amount(self, bins_count):
        v = numpy.linspace(0, bins_count, bins_count, dtype=numpy.float32)
        iv = dpnp.array(v)

        expected_hist, expected_edges = numpy.histogramdd(v, bins=[bins_count])
        result_hist, result_edges = dpnp.histogramdd(iv, bins=[bins_count])
        assert_array_equal(result_hist, expected_hist)
        # dpnp returns both result_hist and result_edges as float64 while
        # numpy returns result_hist as float64 but result_edges as float32
        for x, y in zip(result_edges, expected_edges):
            assert_allclose(x, y, strict=False)


class TestHistogram2d:
    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_rand_data(self, dtype):
        n = 100
        x, y = numpy.random.rand(2, n).astype(dtype=dtype)
        ix = dpnp.array(x, dtype=dtype)
        iy = dpnp.array(y, dtype=dtype)

        expected_hist, _, _ = numpy.histogram2d(x, y)
        result_hist, _, _ = dpnp.histogram2d(ix, iy)
        assert_array_equal(result_hist, expected_hist)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_linspace_data(self, dtype):
        n = 100
        x, y = numpy.linspace(0, 10, 2 * n, dtype=dtype).reshape(2, n)
        ix = dpnp.array(x)
        iy = dpnp.array(y)

        expected_hist, _, _ = numpy.histogram2d(x, y)
        result_hist, _, _ = dpnp.histogram2d(ix, iy)
        assert_array_equal(result_hist, expected_hist)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_bin_float(self, xp):
        x = y = xp.array([[1, 2]])
        assert_raises(ValueError, xp.histogram2d, x, y, bins=0.1)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_bin_2d_array(self, xp):
        x = y = xp.array([[1, 2]])
        assert_raises(ValueError, xp.histogram2d, x, y, bins=[10, 10, 10])

    @pytest.mark.parametrize(
        "bins",
        [
            11,
            [11] * 2,
            [[0, 20, 40, 60, 80, 100]] * 2,
            [[0, 20, 40, 60, 80, 300]] * 2,
        ],
    )
    def test_bins(self, bins):
        n = 100
        dims = 2
        x, y = numpy.arange(n * dims).reshape(dims, n)
        ix = dpnp.array(x)
        iy = dpnp.array(y)

        bins_dpnp = bins
        if isinstance(bins, list):
            if isinstance(bins[0], list):
                bins = [numpy.array(b) for b in bins]
                bins_dpnp = [dpnp.array(b) for b in bins]

        expected_hist, expected_edges_x, expected_edges_y = numpy.histogram2d(
            x, y, bins
        )
        result_hist, result_edges_x, result_edges_y = dpnp.histogram2d(
            ix, iy, bins_dpnp
        )
        assert_allclose(result_hist, expected_hist)
        assert_allclose(result_edges_x, expected_edges_x)
        assert_allclose(result_edges_y, expected_edges_y)

    def test_no_side_effects(self):
        x = dpnp.array([1.3, 2.5, 2.3])
        y = dpnp.array([2.3, 3.5, 4.3])
        copy_x = x.copy()
        copy_y = y.copy()

        # check that ensures that values passed to ``histogram2d`` are unchanged
        _, _, _ = dpnp.histogram2d(x, y)
        assert (x == copy_x).all()
        assert (y == copy_y).all()

    def test_empty(self):
        x = y = numpy.array([])
        ix = dpnp.array(x)
        iy = dpnp.array(y)

        expected_hist, expected_edges_x, expected_edges_y = numpy.histogram2d(
            x, y
        )
        result_hist, result_edges_x, result_edges_y = dpnp.histogram2d(ix, iy)

        assert_allclose(result_hist, expected_hist)
        assert_allclose(result_edges_x, expected_edges_x)
        assert_allclose(result_edges_y, expected_edges_y)

    def test_0d(self):
        x = dpnp.array(1)
        y = dpnp.array(2)

        assert_raises(ValueError, dpnp.histogram2d, x, y)

    def test_2d(self):
        x = dpnp.ones((10, 10))
        y = dpnp.ones((10, 10))

        assert_raises(ValueError, dpnp.histogram2d, x, y)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_finite_range(self, xp):
        x = y = xp.linspace(0.0, 1.0, num=100)

        # normal ranges should be finite
        _, _, _ = xp.histogram2d(x, y, range=[[0.25, 0.75]] * 2)
        assert_raises(
            ValueError, xp.histogram2d, x, y, range=[[xp.nan, 0.75]] * 2
        )
        assert_raises(
            ValueError, xp.histogram2d, x, y, range=[[0.25, xp.inf]] * 2
        )

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_range(self, xp):
        # start of range must be < end of range
        x = y = xp.linspace(0.0, 1.0, num=100)
        with assert_raises_regex(ValueError, "max must be larger than"):
            xp.histogram2d(x, y, range=[[0.1, 0.01]] * 2)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_range_dims(self, xp):
        # range shape must be [2, 2]
        x = y = xp.linspace(0.0, 1.0, num=100)
        assert_raises(ValueError, xp.histogram2d, x, y, range=[[0, 1]])

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_range_size(self, xp):
        # range shape must be [2, 2]
        x = y = xp.linspace(0.0, 1.0, num=100)
        assert_raises(ValueError, xp.histogram2d, x, y, range=[[0, 1, 2]] * 2)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("inf_val", [-numpy.inf, numpy.inf])
    def test_infinite_edge(self, xp, inf_val):
        x = y = xp.array([0.5, 1.5, inf_val])
        min, max = x.min(), x.max()

        # both first and last ranges must be finite
        with assert_raises_regex(
            ValueError,
            f"autodetected range of \\[{min}, {max}\\] is not finite",
        ):
            xp.histogram2d(x, y)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_unsigned_monotonicity_check(self, xp):
        # bins must increase monotonically when bins contain unsigned values
        x = y = xp.array([2])
        bins = [xp.array([1, 3, 1], dtype="uint64")] * 2
        with assert_raises(ValueError):
            xp.histogram2d(x, y, bins=bins)

    def test_nan_values(self):
        one_nan = numpy.array([0, 1, numpy.nan])
        all_nan = numpy.array([numpy.nan, numpy.nan])

        ione_nan = dpnp.array(one_nan)
        iall_nan = dpnp.array(all_nan)

        # NaN is not counted
        expected_hist, expected_edges_x, expected_edges_y = numpy.histogram2d(
            one_nan, one_nan, bins=[[0, 1]] * 2
        )
        result_hist, result_edges_x, result_edges_y = dpnp.histogram2d(
            ione_nan, ione_nan, bins=[[0, 1]] * 2
        )
        assert_allclose(result_hist, expected_hist)
        # dpnp returns both result_hist and result_edges as float64 while
        # numpy returns result_hist as float64 but result_edges as int64
        assert_allclose(result_edges_x, expected_edges_x, strict=False)
        assert_allclose(result_edges_y, expected_edges_y, strict=False)

        # NaN is not counted
        expected_hist, expected_edges_x, expected_edges_y = numpy.histogram2d(
            all_nan, all_nan, bins=[[0, 1]] * 2
        )
        result_hist, result_edges_x, result_edges_y = dpnp.histogram2d(
            iall_nan, iall_nan, bins=[[0, 1]] * 2
        )
        assert_allclose(result_hist, expected_hist)
        # dpnp returns both result_hist and result_edges as float64 while
        # numpy returns result_hist as float64 but result_edges as int64
        assert_allclose(result_edges_x, expected_edges_x, strict=False)
        assert_allclose(result_edges_y, expected_edges_y, strict=False)

    def test_bins_another_sycl_queue(self):
        x = y = dpnp.arange(7, 12, sycl_queue=dpctl.SyclQueue())
        bins = dpnp.arange(4, sycl_queue=dpctl.SyclQueue())
        with assert_raises(ValueError):
            dpnp.histogram2d(x, y, bins=[bins] * 2)

    def test_sample_array_like(self):
        x = y = [0, 1, 2, 3, 4]
        with assert_raises(TypeError):
            dpnp.histogram2d(x, y)

    def test_weights_array_like(self):
        x = y = dpnp.arange(5)
        w = [1, 2, 3, 4, 5]
        with assert_raises(TypeError):
            dpnp.histogram2d(x, y, weights=w)

    def test_weights_another_sycl_queue(self):
        x = y = dpnp.arange(5, sycl_queue=dpctl.SyclQueue())
        w = dpnp.arange(7, 12, sycl_queue=dpctl.SyclQueue())
        with assert_raises(ValueError):
            dpnp.histogram2d(x, y, weights=w)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_size_mismatch(self, xp):
        # x and y must have same shape
        x = xp.linspace(0.0, 1.0, num=10)
        y = xp.linspace(0.0, 1.0, num=20)
        assert_raises(ValueError, xp.histogram2d, x, y)

    @pytest.mark.parametrize(
        "bins_count",
        [10, 10**2, 10**3],
    )
    def test_different_bins_amount(self, bins_count):
        x, y = numpy.linspace(
            0, bins_count, 2 * bins_count, dtype=numpy.float32
        ).reshape(2, bins_count)
        ix = dpnp.array(x)
        iy = dpnp.array(y)

        expected_hist, expected_edges_x, expected_edges_y = numpy.histogram2d(
            x, y, bins=bins_count
        )
        result_hist, result_edges_x, result_edges_y = dpnp.histogram2d(
            ix, iy, bins=bins_count
        )
        assert_array_equal(result_hist, expected_hist)
        # dpnp returns both result_hist and result_edges as float64 while
        # numpy returns result_hist as float64 but result_edges as float32
        assert_allclose(
            result_edges_x, expected_edges_x, rtol=1e-6, strict=False
        )
        assert_allclose(
            result_edges_y, expected_edges_y, rtol=1e-6, strict=False
        )
