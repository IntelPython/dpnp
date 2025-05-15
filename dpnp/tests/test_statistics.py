import dpctl
import dpctl.tensor as dpt
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
    factor_to_tol,
    generate_random_numpy_array,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
    has_support_aspect64,
    numpy_version,
)
from .third_party.cupy.testing import with_requires


class TestAverage:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("returned", [True, False])
    def test_avg_no_wgt(self, dtype, axis, returned):
        a = generate_random_numpy_array((2, 3), dtype)
        ia = dpnp.array(a)

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
        a = generate_random_numpy_array((2, 3), dtype)
        w = generate_random_numpy_array((2, 3), dtype, low=0, high=10)
        ia = dpnp.array(a)
        iw = dpnp.array(w)

        result = dpnp.average(ia, axis=axis, weights=iw, returned=returned)
        expected = numpy.average(a, axis=axis, weights=w, returned=returned)

        if returned:
            assert_dtype_allclose(result[0], expected[0])
            assert_dtype_allclose(result[1], expected[1])
        else:
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
        a = generate_random_numpy_array(20, dtype)
        w = generate_random_numpy_array(20, dtype)
        ia = dpnp.array(a)
        iw = dpnp.array(w)

        result = dpnp.average(ia[::-1], weights=iw[::-1])
        expected = numpy.average(a[::-1], weights=w[::-1])
        assert_dtype_allclose(result, expected)

        result = dpnp.average(ia[::2], weights=iw[::2])
        expected = numpy.average(a[::2], weights=w[::2])
        assert_dtype_allclose(result, expected)

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


class TestConvolve:
    @staticmethod
    def _get_kwargs(mode=None, method=None):
        dpnp_kwargs = {}
        numpy_kwargs = {}
        if mode is not None:
            dpnp_kwargs["mode"] = mode
            numpy_kwargs["mode"] = mode
        if method is not None:
            dpnp_kwargs["method"] = method
        return dpnp_kwargs, numpy_kwargs

    @pytest.mark.parametrize(
        "a, v", [([1], [1, 2, 3]), ([1, 2, 3], [1]), ([1, 2, 3], [1, 2])]
    )
    @pytest.mark.parametrize("mode", [None, "full", "valid", "same"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("method", [None, "auto", "direct", "fft"])
    def test_convolve(self, a, v, mode, dtype, method):
        an = numpy.array(a, dtype=dtype)
        vn = numpy.array(v, dtype=dtype)
        ad = dpnp.array(an)
        vd = dpnp.array(vn)

        dpnp_kwargs, numpy_kwargs = self._get_kwargs(mode, method)

        expected = numpy.convolve(an, vn, **numpy_kwargs)
        result = dpnp.convolve(ad, vd, **dpnp_kwargs)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("a_size", [1, 100, 10000])
    @pytest.mark.parametrize("v_size", [1, 100, 10000])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("method", ["auto", "direct", "fft"])
    def test_convolve_random(self, a_size, v_size, mode, dtype, method):
        if dtype in [numpy.int8, numpy.uint8, numpy.int16, numpy.uint16]:
            pytest.skip("avoid overflow.")
        an = generate_random_numpy_array(
            a_size, dtype, low=-3, high=3, probability=0.9, seed_value=0
        )
        vn = generate_random_numpy_array(
            v_size, dtype, low=-3, high=3, probability=0.9, seed_value=1
        )

        ad = dpnp.array(an)
        vd = dpnp.array(vn)

        dpnp_kwargs, numpy_kwargs = self._get_kwargs(mode, method)

        result = dpnp.convolve(ad, vd, **dpnp_kwargs)
        expected = numpy.convolve(an, vn, **numpy_kwargs)

        if method != "fft" and (
            dpnp.issubdtype(dtype, dpnp.integer) or dtype == dpnp.bool
        ):
            # For 'direct' and 'auto' methods, we expect exact results for integer types
            assert_array_equal(result, expected)
        else:
            if method == "direct":
                # For 'direct' method we can use standard validation
                # acceptable error depends on the kernel size
                # while error grows linearly with the kernel size,
                # this empirically found formula provides a good balance
                # the resulting factor is 40 for kernel size = 1,
                # 400 for kernel size = 100 and 4000 for kernel size = 10000
                factor = int(40 * (min(a_size, v_size) ** 0.5))
                assert_dtype_allclose(result, expected, factor=factor)
            else:
                rdtype = result.dtype
                if dpnp.issubdtype(rdtype, dpnp.integer):
                    # 'fft' do its calculations in float
                    # and 'auto' could use fft
                    # also assert_dtype_allclose for integer types is
                    # always check for exact match
                    rdtype = dpnp.default_float_type(ad.device)

                result = result.astype(rdtype)

                if rdtype == dpnp.bool:
                    result = result.astype(dpnp.int32)
                    rdtype = result.dtype

                expected = expected.astype(rdtype)

                factor = 1000
                rtol = atol = factor_to_tol(rdtype, factor)
                invalid = numpy.logical_not(
                    numpy.isclose(
                        result.asnumpy(), expected, rtol=rtol, atol=atol
                    )
                )

                # When using the 'fft' method, we might encounter outliers.
                # This usually happens when the resulting array contains values close to zero.
                # For these outliers, the relative error can be significant.
                # We can tolerate a few such outliers.
                # max_outliers = 10 if expected.size > 1 else 0
                max_outliers = 10 if expected.size > 1 else 0
                if invalid.sum() > max_outliers:
                    # we already failed check,
                    # call assert_dtype_allclose just to report error nicely
                    assert_dtype_allclose(result, expected, factor=factor)

    def test_convolve_mode_error(self):
        a = dpnp.arange(5)
        v = dpnp.arange(3)

        # invalid mode
        with pytest.raises(ValueError):
            dpnp.convolve(a, v, mode="unknown")

    @pytest.mark.parametrize("a, v", [([], [1]), ([1], []), ([], [])])
    def test_convolve_empty(self, a, v):
        a = dpnp.asarray(a)
        v = dpnp.asarray(v)

        with pytest.raises(ValueError):
            dpnp.convolve(a, v)

    @pytest.mark.parametrize("a, v", [([1], 2), (3, [4]), (5, 6)])
    def test_convolve_scalar(self, a, v):
        an = numpy.asarray(a, dtype=numpy.float32)
        vn = numpy.asarray(v, dtype=numpy.float32)

        ad = dpnp.asarray(a, dtype=numpy.float32)
        vd = dpnp.asarray(v, dtype=numpy.float32)

        expected = numpy.convolve(an, vn)
        result = dpnp.convolve(ad, vd)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "a, v",
        [
            ([[1, 2], [2, 3]], [1]),
            ([1], [[1, 2], [2, 3]]),
            ([[1, 2], [2, 3]], [[1, 2], [2, 3]]),
        ],
    )
    def test_convolve_shape_error(self, a, v):
        a = dpnp.asarray(a)
        v = dpnp.asarray(v)

        with pytest.raises(ValueError):
            dpnp.convolve(a, v)

    @pytest.mark.parametrize("size", [2, 10**1, 10**2, 10**3, 10**4, 10**5])
    def test_convolve_different_sizes(self, size):
        a = generate_random_numpy_array(
            size, dtype=numpy.float32, low=0, high=1, seed_value=0
        )
        v = generate_random_numpy_array(
            size // 2, dtype=numpy.float32, low=0, high=1, seed_value=1
        )

        ad = dpnp.array(a)
        vd = dpnp.array(v)

        expected = numpy.convolve(a, v)
        result = dpnp.convolve(ad, vd, method="direct")

        assert_dtype_allclose(result, expected, factor=20)

    def test_convolve_another_sycl_queue(self):
        a = dpnp.arange(5, sycl_queue=dpctl.SyclQueue())
        v = dpnp.arange(3, sycl_queue=dpctl.SyclQueue())

        with pytest.raises(ValueError):
            dpnp.convolve(a, v)

    def test_convolve_unkown_method(self):
        a = dpnp.arange(5)
        v = dpnp.arange(3)

        with pytest.raises(ValueError):
            dpnp.convolve(a, v, method="unknown")


class TestCorrcoef:
    @pytest.mark.usefixtures(
        "suppress_divide_invalid_numpy_warnings",
        "suppress_dof_numpy_warnings",
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
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
    @staticmethod
    def _get_kwargs(mode=None, method=None):
        dpnp_kwargs = {}
        numpy_kwargs = {}
        if mode is not None:
            dpnp_kwargs["mode"] = mode
            numpy_kwargs["mode"] = mode
        if method is not None:
            dpnp_kwargs["method"] = method
        return dpnp_kwargs, numpy_kwargs

    @pytest.mark.parametrize(
        "a, v", [([1], [1, 2, 3]), ([1, 2, 3], [1]), ([1, 2, 3], [1, 2])]
    )
    @pytest.mark.parametrize("mode", [None, "full", "valid", "same"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("method", [None, "auto", "direct", "fft"])
    def test_correlate(self, a, v, mode, dtype, method):
        an = numpy.array(a, dtype=dtype)
        vn = numpy.array(v, dtype=dtype)
        ad = dpnp.array(an)
        vd = dpnp.array(vn)

        dpnp_kwargs, numpy_kwargs = self._get_kwargs(mode, method)

        expected = numpy.correlate(an, vn, **numpy_kwargs)
        result = dpnp.correlate(ad, vd, **dpnp_kwargs)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("a_size", [1, 100, 10000])
    @pytest.mark.parametrize("v_size", [1, 100, 10000])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("method", ["auto", "direct", "fft"])
    def test_correlate_random(self, a_size, v_size, mode, dtype, method):
        if dtype in [numpy.int8, numpy.uint8, numpy.int16, numpy.uint16]:
            pytest.skip("avoid overflow.")
        an = generate_random_numpy_array(
            a_size, dtype, low=-3, high=3, probability=0.9, seed_value=0
        )
        vn = generate_random_numpy_array(
            v_size, dtype, low=-3, high=3, probability=0.9, seed_value=1
        )

        ad = dpnp.array(an)
        vd = dpnp.array(vn)

        dpnp_kwargs, numpy_kwargs = self._get_kwargs(mode, method)

        result = dpnp.correlate(ad, vd, **dpnp_kwargs)
        expected = numpy.correlate(an, vn, **numpy_kwargs)

        if method != "fft" and (
            dpnp.issubdtype(dtype, dpnp.integer) or dtype == dpnp.bool
        ):
            # For 'direct' and 'auto' methods, we expect exact results for integer types
            assert_array_equal(result, expected)
        else:
            if method == "direct":
                expected = numpy.correlate(an, vn, **numpy_kwargs)
                # For 'direct' method we can use standard validation
                # acceptable error depends on the kernel size
                # while error grows linearly with the kernel size,
                # this empirically found formula provides a good balance
                # the resulting factor is 40 for kernel size = 1,
                # 400 for kernel size = 100 and 4000 for kernel size = 10000
                factor = int(40 * (min(a_size, v_size) ** 0.5))
                assert_dtype_allclose(result, expected, factor=factor)
            else:
                rdtype = result.dtype
                if dpnp.issubdtype(rdtype, dpnp.integer):
                    # 'fft' do its calculations in float
                    # and 'auto' could use fft
                    # also assert_dtype_allclose for integer types is
                    # always check for exact match
                    rdtype = dpnp.default_float_type(ad.device)

                result = result.astype(rdtype)

                if rdtype == dpnp.bool:
                    result = result.astype(dpnp.int32)
                    rdtype = result.dtype

                expected = expected.astype(rdtype)

                factor = 1000
                rtol = atol = factor_to_tol(rdtype, factor)
                invalid = numpy.logical_not(
                    numpy.isclose(
                        result.asnumpy(), expected, rtol=rtol, atol=atol
                    )
                )

                # When using the 'fft' method, we might encounter outliers.
                # This usually happens when the resulting array contains values close to zero.
                # For these outliers, the relative error can be significant.
                # We can tolerate a few such outliers.
                # max_outliers = 10 if expected.size > 1 else 0
                max_outliers = 10 if expected.size > 1 else 0
                if invalid.sum() > max_outliers:
                    # we already failed check,
                    # call assert_dtype_allclose just to report error nicely
                    assert_dtype_allclose(result, expected, factor=factor)

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
        a = generate_random_numpy_array(
            size, dtype=numpy.float32, low=0, high=1, seed_value=0
        )
        v = generate_random_numpy_array(
            size // 2, dtype=numpy.float32, low=0, high=1, seed_value=1
        )

        ad = dpnp.array(a)
        vd = dpnp.array(v)

        expected = numpy.correlate(a, v)
        result = dpnp.correlate(ad, vd, method="direct")

        assert_dtype_allclose(result, expected, factor=20)

    def test_correlate_another_sycl_queue(self):
        a = dpnp.arange(5, sycl_queue=dpctl.SyclQueue())
        v = dpnp.arange(3, sycl_queue=dpctl.SyclQueue())

        with pytest.raises(ValueError):
            dpnp.correlate(a, v)

    def test_correlate_unkown_method(self):
        a = dpnp.arange(5)
        v = dpnp.arange(3)

        with pytest.raises(ValueError):
            dpnp.correlate(a, v, method="unknown")


class TestCov:
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    def test_basic(self, dt):
        a = numpy.array([[0, 2], [1, 1], [2, 0]], dtype=dt)
        ia = dpnp.array(a)

        expected = numpy.cov(a.T)
        result = dpnp.cov(ia.T)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_complex(self, dt):
        a = numpy.array([[1, 2, 3], [1j, 2j, 3j]], dtype=dt)
        ia = dpnp.array(a)

        expected = numpy.cov(a)
        result = dpnp.cov(ia)
        assert_allclose(result, expected)

        expected = numpy.cov(a, aweights=numpy.ones(3))
        result = dpnp.cov(ia, aweights=dpnp.ones(3))
        assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("y_dt", get_complex_dtypes())
    def test_y(self, dt, y_dt):
        a = numpy.array([[1, 2, 3]], dtype=dt)
        y = numpy.array([[1j, 2j, 3j]], dtype=y_dt)
        ia, iy = dpnp.array(a), dpnp.array(y)

        expected = numpy.cov(a, y)
        result = dpnp.cov(ia, iy)
        assert_allclose(result, expected)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("sh", [None, (0, 2), (2, 0)])
    def test_empty(self, sh):
        a = numpy.array([]).reshape(sh)
        ia = dpnp.array(a)

        expected = numpy.cov(a)
        result = dpnp.cov(ia)
        assert_allclose(result, expected)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_wrong_ddof(self):
        a = numpy.array([[0, 2], [1, 1], [2, 0]])
        ia = dpnp.array(a)

        expected = numpy.cov(a.T, ddof=5)
        result = dpnp.cov(ia.T, ddof=5)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", get_float_dtypes())
    @pytest.mark.parametrize("rowvar", [True, False])
    def test_1D_rowvar(self, dt, rowvar):
        a = numpy.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964], dtype=dt)
        y = numpy.array([0.0780, 0.3107, 0.2111, 0.0334, 0.8501])
        ia, iy = dpnp.array(a), dpnp.array(y)

        expected = numpy.cov(a, rowvar=rowvar)
        result = dpnp.cov(ia, rowvar=rowvar)
        assert_allclose(result, expected)

        expected = numpy.cov(a, y, rowvar=rowvar)
        result = dpnp.cov(ia, iy, rowvar=rowvar)
        assert_allclose(result, expected)

    def test_1D_variance(self):
        a = numpy.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964])
        ia = dpnp.array(a)

        expected = numpy.cov(a, ddof=1)
        result = dpnp.cov(ia, ddof=1)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("freq_data", [[1, 4, 1], [1, 1, 1]])
    def test_fweights(self, freq_data):
        a = numpy.array([0.0, 1.0, 2.0], ndmin=2)
        freq = numpy.array(freq_data)
        ia, ifreq = dpnp.array(a), dpnp.array(freq_data)

        expected = numpy.cov(a, fweights=freq)
        result = dpnp.cov(ia, fweights=ifreq)
        assert_allclose(result, expected)

        a = numpy.array([[0, 2], [1, 1], [2, 0]])
        ia = dpnp.array(a)

        expected = numpy.cov(a.T, fweights=freq)
        result = dpnp.cov(ia.T, fweights=ifreq)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_float_fweights(self, xp):
        a = xp.array([[0, 2], [1, 1], [2, 0]])
        freq = xp.array([1, 4, 1]) + 0.5
        assert_raises(TypeError, xp.cov, a, fweights=freq)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize("sh", [(2, 3), 2])
    def test_fweights_wrong_shapes(self, xp, sh):
        a = xp.array([[0, 2], [1, 1], [2, 0]])
        freq = xp.ones(sh, dtype=xp.int_)
        assert_raises((ValueError, RuntimeError), xp.cov, a.T, fweights=freq)

    @pytest.mark.parametrize("freq", [numpy.array([1, 4, 1]), 2])
    def test_fweights_wrong_type(self, freq):
        a = dpnp.array([[0, 2], [1, 1], [2, 0]]).T
        assert_raises(TypeError, dpnp.cov, a, fweights=freq)

    @pytest.mark.parametrize("weights_data", [[1.0, 4.0, 1.0], [1.0, 1.0, 1.0]])
    def test_aweights(self, weights_data):
        a = numpy.array([[0, 2], [1, 1], [2, 0]])
        weights = numpy.array(weights_data)
        ia, iweights = dpnp.array(a), dpnp.array(weights_data)

        expected = numpy.cov(a.T, aweights=weights)
        result = dpnp.cov(ia.T, aweights=iweights)
        assert_allclose(result, expected)

        expected = numpy.cov(a.T, aweights=3.0 * weights)
        result = dpnp.cov(ia.T, aweights=3.0 * iweights)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize("sh", [(2, 3), 2])
    def test_aweights_wrong_shapes(self, xp, sh):
        a = xp.array([[0, 2], [1, 1], [2, 0]])
        weights = xp.ones(sh)
        assert_raises((ValueError, RuntimeError), xp.cov, a.T, aweights=weights)

    @pytest.mark.parametrize("weights", [numpy.array([1.0, 4.0, 1.0]), 2.0])
    def test_aweights_wrong_type(self, weights):
        a = dpnp.array([[0, 2], [1, 1], [2, 0]]).T
        assert_raises(TypeError, dpnp.cov, a, aweights=weights)

    def test_unit_fweights_and_aweights(self):
        a = numpy.array([0.0, 1.0, 2.0], ndmin=2)
        freq = numpy.array([1, 4, 1])
        weights = numpy.ones(3)
        ia, ifreq, iweights = (
            dpnp.array(a),
            dpnp.array(freq),
            dpnp.array(weights),
        )

        # unit weights
        expected = numpy.cov(a, fweights=freq, aweights=weights)
        result = dpnp.cov(ia, fweights=ifreq, aweights=iweights)
        assert_allclose(result, expected)

        a = numpy.array([[0, 2], [1, 1], [2, 0]])
        ia = dpnp.array(a)

        # unit weights
        expected = numpy.cov(a.T, fweights=freq, aweights=weights)
        result = dpnp.cov(ia.T, fweights=ifreq, aweights=iweights)
        assert_allclose(result, expected)

        freq = numpy.ones(3, dtype=numpy.int_)
        ifreq = dpnp.array(freq)

        # unit frequencies and weights
        expected = numpy.cov(a.T, fweights=freq, aweights=weights)
        result = dpnp.cov(ia.T, fweights=ifreq, aweights=iweights)
        assert_allclose(result, expected)

        weights = numpy.array([1.0, 4.0, 1.0])
        iweights = dpnp.array(weights)

        # unit frequencies
        expected = numpy.cov(a.T, fweights=freq, aweights=weights)
        result = dpnp.cov(ia.T, fweights=ifreq, aweights=iweights)
        assert_allclose(result, expected)

        expected = numpy.cov(a.T, fweights=freq, aweights=3.0 * weights)
        result = dpnp.cov(ia.T, fweights=ifreq, aweights=3.0 * iweights)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", get_float_complex_dtypes())
    def test_dtype(self, dt):
        a = numpy.array([[0, 2], [1, 1], [2, 0]])
        ia = dpnp.array(a)

        expected = numpy.cov(a.T, dtype=dt)
        result = dpnp.cov(ia.T, dtype=dt)
        assert_allclose(result, expected)
        assert result.dtype == dt

    @pytest.mark.parametrize("dt", get_float_complex_dtypes())
    @pytest.mark.parametrize("bias", [True, False])
    def test_bias(self, dt, bias):
        a = generate_random_numpy_array((3, 4), dtype=dt)
        ia = dpnp.array(a)

        expected = numpy.cov(a, bias=bias)
        result = dpnp.cov(ia, bias=bias)
        assert_dtype_allclose(result, expected)

        # with rowvar
        expected = numpy.cov(a, rowvar=False, bias=bias)
        result = dpnp.cov(ia, rowvar=False, bias=bias)
        assert_dtype_allclose(result, expected)

        freq = numpy.array([1, 4, 1, 7])
        ifreq = dpnp.array(freq)

        # with frequency
        expected = numpy.cov(a, bias=bias, fweights=freq)
        result = dpnp.cov(ia, bias=bias, fweights=ifreq)
        assert_dtype_allclose(result, expected)

        weights = numpy.array([1.2, 3.7, 5.0, 1.1])
        iweights = dpnp.array(weights)

        # with weights
        expected = numpy.cov(a, bias=bias, aweights=weights)
        result = dpnp.cov(ia, bias=bias, aweights=iweights)
        assert_dtype_allclose(result, expected)

    def test_usm_ndarray(self):
        a = numpy.array([[0, 2], [1, 1], [2, 0]])
        ia = dpt.asarray(a)

        expected = numpy.cov(a.T)
        result = dpnp.cov(ia.T)
        assert_allclose(result, expected)

    # numpy 2.2 properly transposes 2d array when rowvar=False
    @with_requires("numpy>=2.2")
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_false_rowvar_1x3(self):
        a = numpy.array([[0, 1, 2]])
        ia = dpnp.array(a)

        expected = numpy.cov(a, rowvar=False)
        result = dpnp.cov(ia, rowvar=False)
        assert_allclose(result, expected)

    # numpy 2.2 properly transposes 2d array when rowvar=False
    @with_requires("numpy>=2.2")
    def test_true_rowvar(self):
        a = numpy.ones((3, 1))
        ia = dpnp.array(a)

        expected = numpy.cov(a, ddof=0, rowvar=True)
        result = dpnp.cov(ia, ddof=0, rowvar=True)
        assert_allclose(result, expected)


@pytest.mark.parametrize("func", ["max", "min"])
class TestMaxMin:
    @pytest.mark.parametrize("axis", [None, 0, 1, -1, 2, -2, (1, 2), (0, -2)])
    @pytest.mark.parametrize("keepdims", [False, True])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_func(self, func, axis, keepdims, dtype):
        a = generate_random_numpy_array((4, 4, 6, 8), dtype=dtype)
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=axis, keepdims=keepdims)
        result = getattr(dpnp, func)(ia, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

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
    @pytest.mark.parametrize("arr_dt", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
    def test_out_dtype(self, func, arr_dt, out_dt):
        # if out_dt is unsigned, input cannot be signed otherwise overflow occurs
        low = 0 if dpnp.issubdtype(out_dt, dpnp.unsignedinteger) else -10
        a = generate_random_numpy_array((2, 2, 3), dtype=arr_dt, low=low)
        out = numpy.zeros_like(a, shape=(2, 3), dtype=out_dt)
        ia, iout = dpnp.array(a), dpnp.array(out)

        result = getattr(dpnp, func)(ia, out=iout, axis=1)
        expected = getattr(numpy, func)(a, out=out, axis=1)
        assert_dtype_allclose(result, expected)
        assert result is iout

    def test_error(self, func):
        ia = dpnp.arange(5)
        # where is not supported
        with pytest.raises(NotImplementedError):
            getattr(dpnp, func)(ia, where=False)

        # initial is not supported
        with pytest.raises(NotImplementedError):
            getattr(dpnp, func)(ia, initial=6)


class TestMean:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_mean(self, dtype, axis, keepdims):
        a = generate_random_numpy_array((2, 3), dtype)
        ia = dpnp.array(a)

        result = dpnp.mean(ia, axis=axis, keepdims=keepdims)
        expected = numpy.mean(a, axis=axis, keepdims=keepdims)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
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
        a = generate_random_numpy_array(size, dtype)
        ia = dpnp.array(a)

        expected = numpy.median(a)
        result = dpnp.median(ia)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, (-1,), [0, 1], (0, -2, -1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_axis(self, axis, keepdims):
        a = generate_random_numpy_array((2, 3, 4))
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
        a = generate_random_numpy_array((2, 3, 4))
        a[0, 0, 0] = a[-1, -1, -1] = numpy.nan
        ia = dpnp.array(a)

        expected = numpy.median(a, axis=axis, keepdims=keepdims)
        result = dpnp.median(ia, axis=axis, keepdims=keepdims)

        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [None, 0, -1, (0, -2, -1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_overwrite_input(self, axis, keepdims):
        a = generate_random_numpy_array((2, 3, 4))
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
        a = generate_random_numpy_array((2, 3, 4))
        ia = dpt.asarray(a)

        expected = numpy.median(a, axis=axis, overwrite_input=overwrite_input)
        result = dpnp.median(ia, axis=axis, overwrite_input=overwrite_input)
        assert_dtype_allclose(result, expected)


class TestPtp:
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
    def test_basic(self, v, axis):
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
    def test_out(self, v, axis):
        a = numpy.array(v)
        ia = dpnp.array(a)

        expected = numpy.ptp(a, axis)
        result = dpnp.array(numpy.empty_like(expected))
        dpnp.ptp(ia, axis, out=result)
        assert_array_equal(result, expected)


@pytest.mark.parametrize("func", ["std", "var"])
class TestStdVar:
    @pytest.mark.usefixtures(
        "suppress_divide_invalid_numpy_warnings", "suppress_dof_numpy_warnings"
    )
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    @pytest.mark.parametrize("ddof", [0, 0.5, 1, 1.5, 2])
    def test_basic(self, func, dtype, axis, keepdims, ddof):
        a = generate_random_numpy_array((2, 3), dtype)
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(
            a, axis=axis, keepdims=keepdims, ddof=ddof
        )
        result = getattr(dpnp, func)(
            ia, axis=axis, keepdims=keepdims, ddof=ddof
        )
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
    def test_out(self, func, dtype, axis, ddof):
        a = generate_random_numpy_array((2, 3), dtype)
        ia = dpnp.array(a)

        expected = getattr(numpy, func)(a, axis=axis, ddof=ddof)
        if has_support_aspect64():
            res_dtype = expected.dtype
        else:
            res_dtype = dpnp.default_float_type(ia.device)

        out = dpnp.empty(expected.shape, dtype=res_dtype)
        result = getattr(dpnp, func)(ia, axis=axis, out=out, ddof=ddof)
        assert result is out
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures(
        "suppress_invalid_numpy_warnings", "suppress_dof_numpy_warnings"
    )
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 0), (0, 3)])
    def test_empty(self, func, axis, shape):
        ia = dpnp.empty(shape, dtype=dpnp.int64)
        a = dpnp.asnumpy(ia)

        expected = getattr(numpy, func)(a, axis=axis)
        result = getattr(dpnp, func)(ia, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("dt_in", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("dt_out", get_float_complex_dtypes())
    def test_dtype(self, func, dt_in, dt_out):
        ia = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dt_in)
        a = dpnp.asnumpy(ia)

        expected = getattr(numpy, func)(a, dtype=dt_out)
        result = getattr(dpnp, func)(ia, dtype=dt_out)
        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axis", [1, (0, 2), None])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_mean_keyword(self, func, dtype, axis, keepdims):
        a = generate_random_numpy_array((10, 20, 5), dtype)
        ia = dpnp.array(a)

        mean = numpy.mean(a, axis=axis, keepdims=True)
        imean = dpnp.mean(ia, axis=axis, keepdims=True)

        mean_kw = {"mean": mean} if numpy_version() >= "2.0.0" else {}
        expected = getattr(a, func)(axis=axis, keepdims=keepdims, **mean_kw)
        result = getattr(ia, func)(axis=axis, keepdims=keepdims, mean=imean)
        assert_dtype_allclose(result, expected)

    def test_scalar(self, func):
        ia = dpnp.array(5)
        a = dpnp.asnumpy(ia)

        expected = getattr(a, func)()
        result = getattr(ia, func)()
        assert_dtype_allclose(result, expected)

    @with_requires("numpy>=2.0")
    def test_correction(self, func):
        a = numpy.array([1, -1, 1, -1])
        ia = dpnp.array(a)

        # numpy doesn't support `correction` keyword in std/var methods
        expected = getattr(numpy, func)(a, correction=1)
        result = getattr(ia, func)(correction=1)
        assert_dtype_allclose(result, expected)

    @with_requires("numpy>=2.0")
    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_both_ddof_correction_are_set(self, func, xp):
        a = xp.array([1, -1, 1, -1])

        err_msg = "ddof and correction can't be provided simultaneously."

        with assert_raises_regex(ValueError, err_msg):
            getattr(xp, func)(a, ddof=1, correction=0)

        with assert_raises_regex(ValueError, err_msg):
            getattr(xp, func)(a, ddof=1, correction=1)

    def test_error(self, func):
        ia = dpnp.arange(5)
        # where keyword is not implemented
        with pytest.raises(NotImplementedError):
            getattr(dpnp, func)(ia, where=False)

        # ddof should be an integer or float
        with pytest.raises(TypeError):
            getattr(dpnp, func)(ia, ddof="1")
