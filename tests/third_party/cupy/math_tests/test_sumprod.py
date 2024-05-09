import math
import warnings

import numpy
import pytest

import dpnp as cupy
from tests.helper import (
    has_support_aspect16,
    has_support_aspect64,
    is_win_platform,
)
from tests.third_party.cupy import testing


class TestSumprod:
    @pytest.fixture(autouse=True)
    def tearDown(self):
        # Free huge memory for slow test
        # cupy.get_default_memory_pool().free_all_blocks()
        # cupy.get_default_pinned_memory_pool().free_all_blocks()
        pass

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.sum()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_all_keepdims(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.sum(keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_sum_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.sum(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-06)
    def test_sum_all2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype)
        return a.sum()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_all_transposed(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
        return a.sum()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-06)
    def test_sum_all_transposed2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype).transpose(2, 0, 1)
        return a.sum()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.sum(axis=1)

    @testing.slow
    @testing.numpy_cupy_allclose()
    def test_sum_axis_huge(self, xp):
        a = testing.shaped_random((2048, 1, 1024), xp, "b")
        a = xp.broadcast_to(a, (2048, 1024, 1024))
        return a.sum(axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_sum_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.sum(a, axis=1)

    # float16 is omitted, since NumPy's sum on float16 arrays has more error
    # than CuPy's.
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose()
    def test_sum_axis2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype)
        return a.sum(axis=1)

    # test is updated to have exactly the same calls between cupy and numpy,
    # otherwise it is unclear what is verified here
    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
    @testing.numpy_cupy_allclose()
    def test_sum_axis2_float16(self, xp):
        # Note that the above test example overflows in float16. We use a
        # smaller array instead.
        a = testing.shaped_arange((2, 30, 4), xp, dtype="e")
        return a.sum(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_sum_axis_transposed(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype).transpose(2, 0, 1)
        return a.sum(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_sum_axis_transposed2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40), xp, dtype).transpose(2, 0, 1)
        return a.sum(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_axes(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return a.sum(axis=(1, 3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-4)
    def test_sum_axes2(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40, 50), xp, dtype)
        return a.sum(axis=(1, 3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_sum_axes3(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return a.sum(axis=(0, 2, 3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_sum_axes4(self, xp, dtype):
        a = testing.shaped_arange((20, 30, 40, 50), xp, dtype)
        return a.sum(axis=(0, 2, 3))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_empty_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return a.sum(axis=())

    @testing.for_all_dtypes_combination(names=["src_dtype", "dst_dtype"])
    @testing.numpy_cupy_allclose()
    def test_sum_dtype(self, xp, src_dtype, dst_dtype):
        if not xp.can_cast(src_dtype, dst_dtype):
            pytest.skip()
        a = testing.shaped_arange((2, 3, 4), xp, src_dtype)
        return a.sum(dtype=dst_dtype)

    @testing.for_all_dtypes_combination(names=["src_dtype", "dst_dtype"])
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-2, "default": 1e-7})
    def test_sum_keepdims_and_dtype(self, xp, src_dtype, dst_dtype):
        if not xp.can_cast(src_dtype, dst_dtype):
            pytest.skip()
        a = testing.shaped_arange((2, 3, 4), xp, src_dtype)
        return a.sum(axis=2, dtype=dst_dtype, keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_keepdims_multiple_axes(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.sum(axis=(1, 2), keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_sum_out(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty((2, 4), dtype=dtype)
        a.sum(axis=1, out=b)
        return b

    def test_sum_out_wrong_shape(self):
        a = testing.shaped_arange((2, 3, 4))
        b = cupy.empty((2, 3))
        with pytest.raises(ValueError):
            a.sum(axis=1, out=b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_prod_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return a.prod()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_prod_all(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.prod(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_prod_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return a.prod(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_external_prod_axis(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.prod(a, axis=1)

    @testing.for_all_dtypes_combination(names=["src_dtype", "dst_dtype"])
    @testing.numpy_cupy_allclose()
    def test_prod_dtype(self, xp, src_dtype, dst_dtype):
        if not xp.can_cast(src_dtype, dst_dtype):
            pytest.skip()
        a = testing.shaped_arange((2, 3), xp, src_dtype)
        return a.prod(dtype=dst_dtype)

    @pytest.mark.skip("product() is deprecated")
    @testing.numpy_cupy_allclose()
    def test_product_alias(self, xp):
        a = testing.shaped_arange((2, 3), xp, xp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return xp.product(a)


@testing.parameterize(
    *testing.product(
        {
            "shape": [(2, 3, 4), (20, 30, 40)],
            "axis": [0, 1],
            "transpose_axes": [True, False],
            "keepdims": [True, False],
            "func": ["nansum", "nanprod"],
        }
    )
)
class TestNansumNanprodLong:
    def _do_transposed_axis_test(self):
        return not self.transpose_axes and self.axis != 1

    def _numpy_nanprod_implemented(self):
        return (
            self.func == "nanprod"
            and numpy.__version__ >= numpy.lib.NumpyVersion("1.10.0")
        )

    def _test(self, xp, dtype):
        if (
            self.func == "nanprod"
            and self.shape == (20, 30, 40)
            and has_support_aspect64()
        ):
            # If input type is float, NumPy returns the same data type but
            # dpctl (and dpnp) returns default platform float following array api.
            # When input is `float32` and output is a very large number, dpnp returns
            # the number because it is `float64` but NumPy returns `inf` since it is `float32`.
            pytest.skip("Output is a very large number.")

        a = testing.shaped_arange(self.shape, xp, dtype)
        if self.transpose_axes:
            a = a.transpose(2, 0, 1)
        if not issubclass(dtype, xp.integer):
            a[:, 1] = xp.nan
        func = getattr(xp, self.func)
        return func(a, axis=self.axis, keepdims=self.keepdims)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_nansum_all(self, xp, dtype):
        if (
            not self._numpy_nanprod_implemented()
            or not self._do_transposed_axis_test()
        ):
            return xp.array(())
        return self._test(xp, dtype)

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(
        contiguous_check=False, type_check=has_support_aspect64()
    )
    def test_nansum_axis_transposed(self, xp, dtype):
        if (
            not self._numpy_nanprod_implemented()
            or not self._do_transposed_axis_test()
        ):
            return xp.array(())
        return self._test(xp, dtype)


@testing.parameterize(
    *testing.product(
        {
            "shape": [(2, 3, 4), (20, 30, 40)],
        }
    )
)
class TestNansumNanprodExtra:
    def test_nansum_axis_float16(self):
        # Note that the above test example overflows in float16. We use a
        # smaller array instead, just return if array is too large.
        if numpy.prod(self.shape) > 24:
            return
        a = testing.shaped_arange(self.shape, dtype="e")
        a[:, 1] = cupy.nan
        sa = cupy.nansum(a, axis=1)
        b = testing.shaped_arange(self.shape, numpy, dtype="f")
        b[:, 1] = numpy.nan
        sb = numpy.nansum(b, axis=1)
        testing.assert_allclose(sa, sb.astype("e"))

    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose()
    def test_nansum_out(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if not issubclass(dtype, xp.integer):
            a[:, 1] = xp.nan
        b = xp.empty((self.shape[0], self.shape[2]), dtype=dtype)
        xp.nansum(a, axis=1, out=b)
        return b

    def test_nansum_out_wrong_shape(self):
        a = testing.shaped_arange(self.shape)
        a[:, 1] = cupy.nan
        b = cupy.empty((2, 3))
        with pytest.raises(ValueError):
            cupy.nansum(a, axis=1, out=b)


@testing.parameterize(
    *testing.product(
        {
            "shape": [(2, 3, 4, 5), (20, 30, 40, 50)],
            "axis": [(1, 3), (0, 2, 3)],
        }
    )
)
class TestNansumNanprodAxes:
    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_nansum_axes(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        if not issubclass(dtype, xp.integer):
            a[:, 1] = xp.nan
        return xp.nansum(a, axis=self.axis)


class TestNansumNanprodHuge:
    def _test(self, xp, nan_slice):
        a = testing.shaped_random((2048, 1, 1024), xp, "f")
        a[nan_slice] = xp.nan
        a = xp.broadcast_to(a, (2048, 256, 1024))
        return xp.nansum(a, axis=2)

    @testing.slow
    @testing.numpy_cupy_allclose(atol=1e-1)
    def test_nansum_axis_huge(self, xp):
        return self._test(
            xp, (slice(None, None), slice(None, None), slice(1, 2))
        )

    @testing.slow
    @testing.numpy_cupy_allclose(atol=1e-2)
    def test_nansum_axis_huge_halfnan(self, xp):
        return self._test(
            xp, (slice(None, None), slice(None, None), slice(0, 512))
        )


axes = [0, 1, 2]


@testing.parameterize(*testing.product({"axis": axes}))
class TestCumsum:
    def _cumsum(self, xp, a, *args, **kwargs):
        b = a.copy()
        res = xp.cumsum(a, *args, **kwargs)
        testing.assert_array_equal(a, b)  # Check if input array is overwritten
        return res

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return self._cumsum(xp, a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum_out(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        out = xp.zeros((5,), dtype=dtype)
        self._cumsum(xp, a, out=out)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum_out_noncontiguous(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        out = xp.zeros((10,), dtype=dtype)[::2]  # Non contiguous view
        self._cumsum(xp, a, out=out)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum_2dim(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return self._cumsum(xp, a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_cumsum_axis(self, xp, dtype):
        n = len(axes)
        a = testing.shaped_arange(tuple(range(4, 4 + n)), xp, dtype)
        return self._cumsum(xp, a, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum_axis_out(self, xp, dtype):
        n = len(axes)
        shape = tuple(range(4, 4 + n))
        a = testing.shaped_arange(shape, xp, dtype)
        out = xp.zeros(shape, dtype=dtype)
        self._cumsum(xp, a, axis=self.axis, out=out)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum_axis_out_noncontiguous(self, xp, dtype):
        n = len(axes)
        shape = tuple(range(4, 4 + n))
        a = testing.shaped_arange(shape, xp, dtype)
        out = xp.zeros((8,) + shape[1:], dtype=dtype)[
            ::2
        ]  # Non contiguous view
        self._cumsum(xp, a, axis=self.axis, out=out)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(contiguous_check=False)
    def test_ndarray_cumsum_axis(self, xp, dtype):
        n = len(axes)
        a = testing.shaped_arange(tuple(range(4, 4 + n)), xp, dtype)
        return a.cumsum(axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumsum_axis_empty(self, xp, dtype):
        n = len(axes)
        a = testing.shaped_arange(tuple(range(0, n)), xp, dtype)
        return self._cumsum(xp, a, axis=self.axis)

    @testing.for_all_dtypes()
    def test_invalid_axis_lower1(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((4, 5), xp, dtype)
            with pytest.raises(numpy.AxisError):
                xp.cumsum(a, axis=-a.ndim - 1)

    @testing.for_all_dtypes()
    def test_invalid_axis_lower2(self, dtype):
        a = testing.shaped_arange((4, 5), cupy, dtype)
        with pytest.raises(numpy.AxisError):
            return cupy.cumsum(a, axis=-a.ndim - 1)

    @testing.for_all_dtypes()
    def test_invalid_axis_upper1(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((4, 5), xp, dtype)
            with pytest.raises(numpy.AxisError):
                xp.cumsum(a, axis=a.ndim + 1)

    @testing.for_all_dtypes()
    def test_invalid_axis_upper2(self, dtype):
        a = testing.shaped_arange((4, 5), cupy, dtype)
        with pytest.raises(numpy.AxisError):
            return cupy.cumsum(a, axis=a.ndim + 1)

    def test_cumsum_arraylike(self):
        with pytest.raises(TypeError):
            return cupy.cumsum((1, 2, 3))

    @testing.for_float_dtypes()
    def test_cumsum_numpy_array(self, dtype):
        a_numpy = numpy.arange(8, dtype=dtype)
        with pytest.raises(TypeError):
            return cupy.cumsum(a_numpy)


class TestCumprod:
    def _cumprod(self, xp, a, *args, **kwargs):
        b = a.copy()
        res = xp.cumprod(a, *args, **kwargs)
        testing.assert_array_equal(a, b)  # Check if input array is overwritten
        return res

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumprod_1dim(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return self._cumprod(xp, a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumprod_out(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        out = xp.zeros((5,), dtype=dtype)
        self._cumprod(xp, a, out=out)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumprod_out_noncontiguous(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        out = xp.zeros((10,), dtype=dtype)[::2]  # Non contiguous view
        self._cumprod(xp, a, out=out)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-6)
    def test_cumprod_2dim_without_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return self._cumprod(xp, a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_cumprod_2dim_with_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return self._cumprod(xp, a, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_ndarray_cumprod_2dim_with_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return a.cumprod(axis=1)

    @pytest.mark.skip("buffer overflow")
    @testing.slow
    def test_cumprod_huge_array(self):
        size = 2**32
        a = cupy.ones(size, dtype="b")
        result = cupy.cumprod(a, dtype="b")
        del a
        assert (result == 1).all()
        # Free huge memory for slow test
        del result

    @testing.for_all_dtypes()
    def test_invalid_axis_lower1(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((4, 5), xp, dtype)
            with pytest.raises(numpy.AxisError):
                xp.cumprod(a, axis=-a.ndim - 1)

    @testing.for_all_dtypes()
    def test_invalid_axis_lower2(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((4, 5), xp, dtype)
            with pytest.raises(numpy.AxisError):
                xp.cumprod(a, axis=-a.ndim - 1)

    @testing.for_all_dtypes()
    def test_invalid_axis_upper1(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((4, 5), xp, dtype)
            with pytest.raises(numpy.AxisError):
                return xp.cumprod(a, axis=a.ndim)

    @testing.for_all_dtypes()
    def test_invalid_axis_upper2(self, dtype):
        a = testing.shaped_arange((4, 5), cupy, dtype)
        with pytest.raises(numpy.AxisError):
            return cupy.cumprod(a, axis=a.ndim)

    def test_cumprod_arraylike(self):
        with pytest.raises(TypeError):
            return cupy.cumprod((1, 2, 3))

    @testing.for_float_dtypes()
    def test_cumprod_numpy_array(self, dtype):
        a_numpy = numpy.arange(1, 6, dtype=dtype)
        with pytest.raises(TypeError):
            return cupy.cumprod(a_numpy)

    @pytest.mark.skip("cumproduct() is deprecated")
    @testing.numpy_cupy_allclose()
    def test_cumproduct_alias(self, xp):
        a = testing.shaped_arange((2, 3), xp, xp.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return xp.cumproduct(a)


@testing.parameterize(
    *testing.product(
        {
            "shape": [(20,), (7, 6), (3, 4, 5)],
            "axis": [None, 0, 1, 2],
            "func": ("nancumsum", "nancumprod"),
        }
    )
)
class TestNanCumSumProd:
    zero_density = 0.25

    @pytest.fixture(autouse=True)
    def setUp(self):
        if self.func == "nancumprod":
            pytest.skip("nancumprod() is not implemented yet")
        pass

    def _make_array(self, dtype):
        dtype = numpy.dtype(dtype)
        if dtype.char in "efdFD":
            r_dtype = dtype.char.lower()
            a = testing.shaped_random(self.shape, numpy, dtype=r_dtype, scale=1)
            if dtype.char in "FD":
                ai = a
                aj = testing.shaped_random(
                    self.shape, numpy, dtype=r_dtype, scale=1
                )
                ai[ai < math.sqrt(self.zero_density)] = 0
                aj[aj < math.sqrt(self.zero_density)] = 0
                a = ai + 1j * aj
            else:
                a[a < self.zero_density] = 0
            a = a / a
        else:
            a = testing.shaped_random(self.shape, numpy, dtype=dtype)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nancumsumprod(self, xp, dtype):
        if self.axis is not None and self.axis >= len(self.shape):
            pytest.skip()
        a = xp.array(self._make_array(dtype))
        out = getattr(xp, self.func)(a, axis=self.axis)
        return xp.ascontiguousarray(out)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nancumsumprod_out(self, xp, dtype):
        dtype = numpy.dtype(dtype)
        if self.axis is not None and self.axis >= len(self.shape):
            pytest.skip()
        if len(self.shape) > 1 and self.axis is None:
            # Skip the cases where np.nancum{sum|prod} raise AssertionError.
            pytest.skip()
        a = xp.array(self._make_array(dtype))
        out = xp.empty(self.shape, dtype=dtype)
        getattr(xp, self.func)(a, axis=self.axis, out=out)
        return xp.ascontiguousarray(out)


class TestDiff:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diff_1dim(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.diff(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diff_1dim_with_n(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.diff(a, n=3)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diff_2dim_without_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.diff(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diff_2dim_with_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.diff(a, axis=-2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diff_2dim_with_n_and_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.diff(a, 2, 1)

    @testing.with_requires("numpy>=1.16")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diff_2dim_with_prepend(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        b = testing.shaped_arange((4, 1), xp, dtype)
        return xp.diff(a, axis=-1, prepend=b)

    @testing.with_requires("numpy>=1.16")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_diff_2dim_with_append(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        b = testing.shaped_arange((1, 5), xp, dtype)
        return xp.diff(a, axis=0, append=b, n=2)

    @testing.with_requires("numpy>=1.16")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_diff_2dim_with_scalar_append(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.diff(a, prepend=1, append=0)

    @testing.with_requires("numpy>=1.16")
    def test_diff_invalid_axis(self):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 4), xp)
            with pytest.raises(numpy.AxisError):
                xp.diff(a, axis=3)
            with pytest.raises(numpy.AxisError):
                xp.diff(a, axis=-4)


# This class compares CUB results against NumPy's
@testing.parameterize(
    *testing.product_dict(
        testing.product(
            {
                "shape": [()],
                "axis": [None, ()],
                "spacing": [(), (1.2,)],
            }
        )
        + testing.product(
            {
                "shape": [(33,)],
                "axis": [None, 0, -1, (0,)],
                "spacing": [(), (1.2,), "sequence of int", "arrays"],
            }
        )
        + testing.product(
            {
                "shape": [(10, 20), (10, 20, 30)],
                "axis": [None, 0, -1, (0, -1), (1, 0)],
                "spacing": [(), (1.2,), "sequence of int", "arrays", "mixed"],
            }
        ),
        testing.product(
            {
                "edge_order": [1, 2],
            }
        ),
    )
)
@pytest.mark.skip("gradient() is not implemented yet")
class TestGradient:
    def _gradient(self, xp, dtype, shape, spacing, axis, edge_order):
        x = testing.shaped_random(shape, xp, dtype=dtype)
        if axis is None:
            normalized_axes = tuple(range(x.ndim))
        else:
            normalized_axes = axis
            if not isinstance(normalized_axes, tuple):
                normalized_axes = (normalized_axes,)
            normalized_axes = tuple(ax % x.ndim for ax in normalized_axes)
        if spacing == "sequence of int":
            # one scalar per axis
            spacing = tuple((ax + 1) / x.ndim for ax in normalized_axes)
        elif spacing == "arrays":
            # one array per axis
            spacing = tuple(
                xp.arange(x.shape[ax]) * (ax + 0.5) for ax in normalized_axes
            )
            # make at one of the arrays have non-constant spacing
            spacing[-1][5:] *= 2.0
        elif spacing == "mixed":
            # mixture of arrays and scalars
            spacing = [xp.arange(x.shape[normalized_axes[0]])]
            spacing = spacing + [0.5] * (len(normalized_axes) - 1)
        return xp.gradient(x, *spacing, axis=axis, edge_order=edge_order)

    @testing.for_dtypes("fFdD")
    @testing.numpy_cupy_allclose(atol=1e-6, rtol=1e-5)
    def test_gradient_floating(self, xp, dtype):
        return self._gradient(
            xp, dtype, self.shape, self.spacing, self.axis, self.edge_order
        )

    # unsigned int behavior fixed in 1.18.1
    # https://github.com/numpy/numpy/issues/15207
    @testing.with_requires("numpy>=1.18.1")
    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose(atol=1e-6, rtol=1e-5)
    def test_gradient_int(self, xp, dtype):
        return self._gradient(
            xp, dtype, self.shape, self.spacing, self.axis, self.edge_order
        )

    @testing.numpy_cupy_allclose(atol=2e-2, rtol=1e-3)
    def test_gradient_float16(self, xp):
        return self._gradient(
            xp,
            numpy.float16,
            self.shape,
            self.spacing,
            self.axis,
            self.edge_order,
        )


@pytest.mark.skip("gradient() is not implemented yet")
class TestGradientErrors:
    def test_gradient_invalid_spacings1(self):
        # more spacings than axes
        spacing = (1.0, 2.0, 3.0)
        for xp in [numpy, cupy]:
            x = testing.shaped_random((32, 16), xp)
            with pytest.raises(TypeError):
                xp.gradient(x, *spacing)

    def test_gradient_invalid_spacings2(self):
        # wrong length array in spacing
        shape = (32, 16)
        spacing = (15, cupy.arange(shape[1] + 1))
        for xp in [numpy, cupy]:
            x = testing.shaped_random(shape, xp)
            with pytest.raises(ValueError):
                xp.gradient(x, *spacing)

    def test_gradient_invalid_spacings3(self):
        # spacing array with ndim != 1
        shape = (32, 16)
        spacing = (15, cupy.arange(shape[0]).reshape(4, -1))
        for xp in [numpy, cupy]:
            x = testing.shaped_random(shape, xp)
            with pytest.raises(ValueError):
                xp.gradient(x, *spacing)

    def test_gradient_invalid_edge_order1(self):
        # unsupported edge order
        shape = (32, 16)
        for xp in [numpy, cupy]:
            x = testing.shaped_random(shape, xp)
            with pytest.raises(ValueError):
                xp.gradient(x, edge_order=3)

    def test_gradient_invalid_edge_order2(self):
        # shape cannot be < edge_order
        shape = (1, 16)
        for xp in [numpy, cupy]:
            x = testing.shaped_random(shape, xp)
            with pytest.raises(ValueError):
                xp.gradient(x, axis=0, edge_order=2)

    @testing.with_requires("numpy>=1.16")
    def test_gradient_invalid_axis(self):
        # axis out of range
        shape = (4, 16)
        for xp in [numpy, cupy]:
            x = testing.shaped_random(shape, xp)
            for axis in [-3, 2]:
                with pytest.raises(numpy.AxisError):
                    xp.gradient(x, axis=axis)

    def test_gradient_bool_input(self):
        # axis out of range
        shape = (4, 16)
        for xp in [numpy, cupy]:
            x = testing.shaped_random(shape, xp, dtype=numpy.bool_)
            with pytest.raises(TypeError):
                xp.gradient(x)


@pytest.mark.skip("ediff1d() is not implemented yet")
class TestEdiff1d:
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ediff1d_1dim(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.ediff1d(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ediff1d_2dim(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.ediff1d(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ediff1d_3dim(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        return xp.ediff1d(a)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ediff1d_to_begin1(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.ediff1d(a, to_begin=xp.array([0], dtype=dtype))

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ediff1d_to_begin2(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.ediff1d(a, to_begin=xp.array([4, 4], dtype=dtype))

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ediff1d_to_begin3(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.ediff1d(a, to_begin=xp.array([1, 1], dtype=dtype))

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ediff1d_to_end1(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.ediff1d(a, to_end=xp.array([0], dtype=dtype))

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ediff1d_to_end2(self, xp, dtype):
        a = testing.shaped_arange((4, 1), xp, dtype)
        return xp.ediff1d(a, to_end=xp.array([1, 2], dtype=dtype))

    @testing.for_dtypes("bhilqefdFD")
    @testing.numpy_cupy_allclose()
    def test_ediff1d_ed1(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4, 5), xp, dtype)
        return xp.ediff1d(
            a,
            to_begin=xp.array([-1], dtype=dtype),
            to_end=xp.array([0], dtype=dtype),
        )

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ediff1d_ed2(self, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return xp.ediff1d(
            a,
            to_begin=xp.array([0, 4], dtype=dtype),
            to_end=xp.array([1, 1], dtype=dtype),
        )


@pytest.mark.skip("trapz() is not implemented yet")
class TestTrapz:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-1, "default": 1e-7})
    def test_trapz_1dim(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.trapz(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-1, "default": 1e-7})
    def test_trapz_1dim_with_x(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        x = testing.shaped_arange((5,), xp, dtype)
        return xp.trapz(a, x=x)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-1, "default": 1e-7})
    def test_trapz_1dim_with_dx(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        return xp.trapz(a, dx=0.1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-1, "default": 1e-7})
    def test_trapz_2dim_without_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.trapz(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-1, "default": 1e-7})
    def test_trapz_2dim_with_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.trapz(a, axis=-2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-1, "default": 1e-7})
    def test_trapz_2dim_with_x_and_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        x = testing.shaped_arange((5,), xp, dtype)
        return xp.trapz(a, x=x, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-1, "default": 1e-7})
    def test_trapz_2dim_with_dx_and_axis(self, xp, dtype):
        a = testing.shaped_arange((4, 5), xp, dtype)
        return xp.trapz(a, dx=0.1, axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol={numpy.float16: 1e-1, "default": 1e-7})
    def test_trapz_1dim_with_x_and_dx(self, xp, dtype):
        a = testing.shaped_arange((5,), xp, dtype)
        x = testing.shaped_arange((5,), xp, dtype)
        return xp.trapz(a, x=x, dx=0.1)
