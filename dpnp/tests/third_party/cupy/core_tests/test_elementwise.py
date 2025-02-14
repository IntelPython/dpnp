import unittest

import numpy
import pytest

import dpnp as cupy
from dpnp.tests.helper import (
    has_support_aspect64,
    is_win_platform,
    numpy_version,
)
from dpnp.tests.third_party.cupy import testing


class TestElementwise(unittest.TestCase):

    def check_copy(self, dtype, src_id, dst_id):
        with cuda.Device(src_id):
            src = testing.shaped_arange((2, 3, 4), dtype=dtype)
        with cuda.Device(dst_id):
            dst = cupy.empty((2, 3, 4), dtype=dtype)
        _core.elementwise_copy(src, dst)
        testing.assert_allclose(src, dst)

    @pytest.mark.skip("elementwise_copy() argument isn't supported")
    @testing.for_all_dtypes()
    def test_copy(self, dtype):
        device_id = cuda.Device().id
        self.check_copy(dtype, device_id, device_id)

    @pytest.mark.skip("elementwise_copy() argument isn't supported")
    @testing.for_all_dtypes()
    def test_copy_multigpu_nopeer(self, dtype):
        if cuda.runtime.deviceCanAccessPeer(0, 1) == 1:
            pytest.skip("peer access is available")
        with self.assertRaises(ValueError):
            self.check_copy(dtype, 0, 1)

    @pytest.mark.skip("elementwise_copy() argument isn't supported")
    @testing.for_all_dtypes()
    def test_copy_multigpu_peer(self, dtype):
        if cuda.runtime.deviceCanAccessPeer(0, 1) != 1:
            pytest.skip("peer access is unavailable")
        with pytest.warns(cupy._util.PerformanceWarning):
            self.check_copy(dtype, 0, 1)

    @testing.for_orders("CFAK")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_copy_zero_sized_array1(self, xp, dtype, order):
        src = xp.empty((0,), dtype=dtype)
        res = xp.copy(src, order=order)
        assert src is not res
        return res

    @testing.for_orders("CFAK")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_copy_zero_sized_array2(self, xp, dtype, order):
        src = xp.empty((1, 0, 2), dtype=dtype)
        res = xp.copy(src, order=order)
        assert src is not res
        return res

    @testing.for_orders("CFAK")
    def test_copy_orders(self, order):
        a = cupy.empty((2, 3, 4))
        b = cupy.copy(a, order)

        a_cpu = numpy.empty((2, 3, 4))
        b_cpu = numpy.copy(a_cpu, order)

        assert b.strides == tuple(x / b_cpu.itemsize for x in b_cpu.strides)


@pytest.mark.skip("`ElementwiseKernel` isn't supported")
class TestElementwiseInvalidShape(unittest.TestCase):

    def test_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "Out shape is mismatched"):
            f = cupy.ElementwiseKernel("T x", "T y", "y += x")
            x = cupy.arange(12).reshape(3, 4)
            y = cupy.arange(4)
            f(x, y)


@pytest.mark.skip("`ElementwiseKernel` isn't supported")
class TestElementwiseInvalidArgument(unittest.TestCase):

    def test_invalid_kernel_name(self):
        with self.assertRaisesRegex(ValueError, "Invalid kernel name"):
            cupy.ElementwiseKernel("T x", "", "", "1")


class TestElementwiseType(unittest.TestCase):

    @testing.with_requires("numpy>=2.0")
    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(accept_error=OverflowError)
    def test_large_int_upper_1(self, xp, dtype):
        a = xp.array([0], dtype=xp.int8)
        b = xp.iinfo(dtype).max
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(accept_error=OverflowError)
    def test_large_int_upper_2(self, xp, dtype):
        if numpy_version() < "2.0.0":
            flag = dtype in [xp.int16, xp.int32, xp.int64, xp.longlong]
            if xp.issubdtype(dtype, xp.unsignedinteger) or flag:
                pytest.skip("numpy doesn't raise OverflowError")

            if dtype in [xp.int8, xp.intc] and is_win_platform():
                pytest.skip("numpy promotes dtype differently")

        a = xp.array([1], dtype=xp.int8)
        b = xp.iinfo(dtype).max - 1
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_upper_3(self, xp, dtype):
        if (
            numpy.issubdtype(dtype, numpy.unsignedinteger)
            and numpy_version() < "2.0.0"
        ):
            pytest.skip("numpy promotes dtype differently")
        elif (
            dtype in (numpy.uint64, numpy.ulonglong)
            and not has_support_aspect64()
        ):
            pytest.skip("no fp64 support")

        a = xp.array([xp.iinfo(dtype).max], dtype=dtype)
        b = numpy.int8(0)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_upper_4(self, xp, dtype):
        if (
            numpy.issubdtype(dtype, numpy.unsignedinteger)
            and numpy_version() < "2.0.0"
        ):
            pytest.skip("numpy promotes dtype differently")
        elif (
            dtype in (numpy.uint64, numpy.ulonglong)
            and not has_support_aspect64()
        ):
            pytest.skip("no fp64 support")

        a = xp.array([xp.iinfo(dtype).max - 1], dtype=dtype)
        b = numpy.int8(1)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(accept_error=OverflowError)
    def test_large_int_lower_1(self, xp, dtype):
        if numpy_version() < "2.0.0":
            if dtype in [xp.int16, xp.int32, xp.int64, xp.longlong]:
                pytest.skip("numpy doesn't raise OverflowError")

            if dtype in [xp.int8, xp.intc] and is_win_platform():
                pytest.skip("numpy promotes dtype differently")

        a = xp.array([0], dtype=xp.int8)
        b = xp.iinfo(dtype).min
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal(accept_error=OverflowError)
    def test_large_int_lower_2(self, xp, dtype):
        if numpy_version() < "2.0.0":
            if dtype in [xp.int16, xp.int32, xp.int64, xp.longlong]:
                pytest.skip("numpy doesn't raise OverflowError")

            if dtype in [xp.int8, xp.intc] and is_win_platform():
                pytest.skip("numpy promotes dtype differently")

        a = xp.array([-1], dtype=xp.int8)
        b = xp.iinfo(dtype).min + 1
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_lower_3(self, xp, dtype):
        if (
            numpy.issubdtype(dtype, numpy.unsignedinteger)
            and numpy_version() < "2.0.0"
        ):
            pytest.skip("numpy promotes dtype differently")
        elif (
            dtype in (numpy.uint64, numpy.ulonglong)
            and not has_support_aspect64()
        ):
            pytest.skip("no fp64 support")

        a = xp.array([xp.iinfo(dtype).min], dtype=dtype)
        b = numpy.int8(0)
        return a + b

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_large_int_lower_4(self, xp, dtype):
        if (
            dtype in (numpy.uint64, numpy.ulonglong)
            and not has_support_aspect64()
        ):
            pytest.skip("no fp64 support")

        a = xp.array([xp.iinfo(dtype).min + 1], dtype=dtype)
        b = numpy.int8(-1)
        return a + b
