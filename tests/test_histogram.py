import dpctl
import numpy
import pytest
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_array_equal,
    assert_raises,
    assert_raises_regex,
    suppress_warnings,
)

import dpnp

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_float_dtypes,
    has_support_aspect64,
)


class TestDigitize:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
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
        x = x.astype(dtype)
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

    @pytest.mark.parametrize(
        "dtype_x", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "dtype_bins", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize("right", [True, False])
    def test_digitize_diff_types(self, dtype_x, dtype_bins, right):
        x = numpy.array([1, 2, 3, 4, 5], dtype=dtype_x)
        bins = numpy.array([1, 3, 5], dtype=dtype_bins)
        x_dp = dpnp.array(x)
        bins_dp = dpnp.array(bins)

        result = dpnp.digitize(x_dp, bins_dp, right=right)
        expected = numpy.digitize(x, bins, right=right)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
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

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_bool_conversion(self, xp):
        a = xp.array([1, 1, 0], dtype=numpy.uint8)
        int_hist, int_edges = xp.histogram(a)

        with suppress_warnings() as sup:
            rec = sup.record(RuntimeWarning, "Converting input from .*")

            v = xp.array([True, True, False])
            hist, edges = xp.histogram(v)

            # A warning should be issued
            assert len(rec) == 1
            assert_array_equal(hist, int_hist)
            assert_array_equal(edges, int_edges)

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

    def test_integer_weights(self):
        v = numpy.array([1, 2, 2, 4])
        w = numpy.array([4, 3, 2, 1])

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
            assert_(x >= left)
            assert_(x < right)

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
