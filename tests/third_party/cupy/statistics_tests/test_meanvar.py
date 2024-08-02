import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError

import dpnp as cupy
from tests.helper import has_support_aspect16, has_support_aspect64
from tests.third_party.cupy import testing

ignore_runtime_warnings = pytest.mark.filterwarnings(
    "ignore", category=RuntimeWarning
)


class TestMedian:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_median_noaxis(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_median_axis1(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, axis=1)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_median_axis2(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, axis=2)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_median_overwrite_input(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, overwrite_input=True)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_median_keepdims_axis1(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, axis=1, keepdims=True)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_median_keepdims_noaxis(self, xp, dtype):
        a = testing.shaped_random((3, 4, 5), xp, dtype)
        return xp.median(a, keepdims=True)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_median_invalid_axis(self):
        for xp in [numpy, cupy]:
            a = testing.shaped_random((3, 4, 5), xp)
            with pytest.raises(AxisError):
                return xp.median(a, -a.ndim - 1, keepdims=False)

            with pytest.raises(AxisError):
                return xp.median(a, a.ndim, keepdims=False)

            with pytest.raises(AxisError):
                return xp.median(a, (-a.ndim - 1, 1), keepdims=False)

            with pytest.raises(AxisError):
                return xp.median(
                    a,
                    (
                        0,
                        a.ndim,
                    ),
                    keepdims=False,
                )


@testing.parameterize(
    *testing.product(
        {
            "shape": [(3, 4, 5)],
            "axis": [(0, 1), (0, -1), (1, 2), (1,)],
            "keepdims": [True, False],
        }
    )
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestMedianAxis:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_median_axis_sequence(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        return xp.median(a, self.axis, keepdims=self.keepdims)


class TestAverage:
    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_average_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.average(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_average_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.average(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_average_weights(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        w = testing.shaped_arange((2, 3), xp, dtype)
        return xp.average(a, weights=w)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=2e-7, type_check=has_support_aspect64())
    @pytest.mark.parametrize(
        "axis, weights", [(1, False), (None, True), (1, True)]
    )
    def test_returned(self, xp, dtype, axis, weights):
        a = testing.shaped_arange((2, 3), xp, dtype)
        if weights:
            w = testing.shaped_arange((2, 3), xp, dtype)
        else:
            w = None
        return xp.average(a, axis=axis, weights=w, returned=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=5e-7, type_check=has_support_aspect64())
    @pytest.mark.parametrize("returned", [True, False])
    @testing.with_requires("numpy>=1.23.1")
    def test_average_keepdims_axis1(self, xp, dtype, returned):
        a = testing.shaped_random((2, 3), xp, dtype)
        w = testing.shaped_random((2, 3), xp, dtype)
        return xp.average(
            a, axis=1, weights=w, returned=returned, keepdims=True
        )

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-7, type_check=has_support_aspect64())
    @pytest.mark.parametrize("returned", [True, False])
    @testing.with_requires("numpy>=1.23.1")
    def test_average_keepdims_noaxis(self, xp, dtype, returned):
        a = testing.shaped_random((2, 3), xp, dtype)
        w = testing.shaped_random((2, 3), xp, dtype)
        return xp.average(a, weights=w, returned=returned, keepdims=True)


class TestMeanVar:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_mean_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.mean()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_mean_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.mean(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_mean_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.mean(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_mean_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.mean(a, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-06)
    def test_mean_all_float32_dtype(self, xp, dtype):
        a = xp.full((2, 3, 4), 123456789, dtype=dtype)
        return xp.mean(a, dtype=numpy.float32)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose()
    def test_mean_all_int64_dtype(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.mean(a, dtype=numpy.int64)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_mean_all_complex_dtype(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.mean(a, dtype=numpy.complex64)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_var_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.var()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_var_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.var(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_var_all_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.var(ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_var_all_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.var(a, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_var_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.var(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_var_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.var(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_var_axis_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.var(axis=1, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_var_axis_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.var(a, axis=1, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_std_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.std()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_std_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.std(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_std_all_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.std(ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_std_all_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.std(a, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_std_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.std(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_std_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.std(a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_std_axis_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.std(axis=1, ddof=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_std_axis_ddof(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.std(a, axis=1, ddof=1)


@testing.parameterize(
    *testing.product(
        {
            "shape": [(3, 4), (30, 40, 50)],
            "axis": [None, 0, 1],
            "keepdims": [True, False],
        }
    )
)
class TestNanMean:
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanmean_without_nan(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        return xp.nanmean(a, axis=self.axis, keepdims=self.keepdims)

    @pytest.mark.usefixtures("suppress_mean_empty_slice_numpy_warnings")
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanmean_with_nan_float(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)

        if a.dtype.kind not in "biu":
            a[1, :] = xp.nan
            a[:, 3] = xp.nan

        return xp.nanmean(a, axis=self.axis, keepdims=self.keepdims)


class TestNanMeanAdditional:
    @pytest.mark.usefixtures("suppress_mean_empty_slice_numpy_warnings")
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanmean_out(self, xp, dtype):
        a = testing.shaped_random((10, 20, 30), xp, dtype)
        # `numpy.mean` allows ``unsafe`` casting while `dpnp.mean` does not.
        # So, output data type cannot be the same as input.
        out_dtype = (
            cupy.default_float_type(a.device) if xp == cupy else numpy.float64
        )
        z = xp.zeros((20, 30), dtype=out_dtype)

        if a.dtype.kind not in "biu":
            a[1, :] = xp.nan
            a[:, 3] = xp.nan

        xp.nanmean(a, axis=0, out=z)
        return z

    @testing.slow
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanmean_huge(self, xp, dtype):
        a = testing.shaped_random((1024, 512), xp, dtype)

        if a.dtype.kind not in "biu":
            a[:512, :256] = xp.nan

        return xp.nanmean(a, axis=1)

    @pytest.mark.skipif(
        not has_support_aspect16(), reason="No fp16 support by device"
    )
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_nanmean_float16(self, xp):
        a = testing.shaped_arange((2, 3), xp, numpy.float16)
        a[0][0] = xp.nan
        return xp.nanmean(a)

    @pytest.mark.usefixtures("suppress_mean_empty_slice_numpy_warnings")
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanmean_all_nan(self, xp):
        a = xp.zeros((3, 4))
        a[:] = xp.nan
        return xp.nanmean(a)


@testing.parameterize(
    *testing.product(
        {
            "shape": [(3, 4), (4, 3, 5)],
            "axis": [None, 0, 1],
            "keepdims": [True, False],
            "ddof": [0, 1],
        }
    )
)
class TestNanVarStd:
    @pytest.mark.usefixtures("suppress_dof_numpy_warnings")
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanvar(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype=dtype)
        if a.dtype.kind not in "biu":
            a[0, :] = xp.nan
        return xp.nanvar(
            a, axis=self.axis, ddof=self.ddof, keepdims=self.keepdims
        )

    @pytest.mark.usefixtures("suppress_dof_numpy_warnings")
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanstd(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype=dtype)
        if a.dtype.kind not in "biu":
            a[0, :] = xp.nan
        return xp.nanstd(
            a, axis=self.axis, ddof=self.ddof, keepdims=self.keepdims
        )


class TestNanVarStdAdditional:
    @pytest.mark.usefixtures("suppress_dof_numpy_warnings")
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanvar_out(self, xp, dtype):
        a = testing.shaped_random((10, 20, 30), xp, dtype)
        z = xp.zeros((20, 30))

        if a.dtype.kind not in "biu":
            a[1, :] = xp.nan
            a[:, 3] = xp.nan

        xp.nanvar(a, axis=0, out=z)
        return z

    @testing.slow
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanvar_huge(self, xp, dtype):
        a = testing.shaped_random((1024, 512), xp, dtype)

        if a.dtype.kind not in "biu":
            a[:512, :256] = xp.nan

        return xp.nanvar(a, axis=1)

    @pytest.mark.skipif(
        not has_support_aspect16(), reason="No fp16 support by device"
    )
    @testing.numpy_cupy_allclose(rtol=1e-3)
    def test_nanvar_float16(self, xp):
        a = testing.shaped_arange((4, 5), xp, numpy.float16)
        a[0][0] = xp.nan
        return xp.nanvar(a, axis=0)

    @pytest.mark.usefixtures("suppress_dof_numpy_warnings")
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanstd_out(self, xp, dtype):
        a = testing.shaped_random((10, 20, 30), xp, dtype)
        z = xp.zeros((20, 30))

        if a.dtype.kind not in "biu":
            a[1, :] = xp.nan
            a[:, 3] = xp.nan

        xp.nanstd(a, axis=0, out=z)
        return z

    @testing.slow
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6, type_check=has_support_aspect64())
    def test_nanstd_huge(self, xp, dtype):
        a = testing.shaped_random((1024, 512), xp, dtype)

        if a.dtype.kind not in "biu":
            a[:512, :256] = xp.nan

        return xp.nanstd(a, axis=1)

    @pytest.mark.skipif(
        not has_support_aspect16(), reason="No fp16 support by device"
    )
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_nanstd_float16(self, xp):
        a = testing.shaped_arange((4, 5), xp, numpy.float16)
        a[0][0] = xp.nan
        return xp.nanstd(a, axis=1)


@testing.parameterize(
    *testing.product(
        {
            "params": [
                ((), None),
                ((0,), None),
                ((0, 0), None),
                ((0, 0), 1),
                ((0, 0, 0), None),
                ((0, 0, 0), (0, 2)),
            ],
            "func": ["mean", "std", "var"],
        }
    )
)
@pytest.mark.usefixtures(
    "suppress_invalid_numpy_warnings",
    "suppress_dof_numpy_warnings",
    "suppress_mean_empty_slice_numpy_warnings",
)
class TestProductZeroLength:
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_external_mean_zero_len(self, xp, dtype):
        shape, axis = self.params
        a = testing.shaped_arange(shape, xp, dtype)
        f = getattr(xp, self.func)
        return f(a, axis=axis)
