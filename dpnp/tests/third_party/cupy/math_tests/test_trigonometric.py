import unittest

import numpy
import pytest

from dpnp.tests.helper import has_support_aspect64
from dpnp.tests.third_party.cupy import testing


class TestTrigonometric(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(
        atol=1e-4, rtol=0.001, type_check=has_support_aspect64()
    )
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5, type_check=has_support_aspect64())
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    @testing.for_dtypes(["e", "f", "d"])
    @testing.numpy_cupy_allclose(atol={numpy.float16: 1e-3, "default": 1e-5})
    def check_unary_unit(self, name, xp, dtype):
        a = xp.array([0.2, 0.4, 0.6, 0.8], dtype=dtype)
        return getattr(xp, name)(a)

    def test_sin(self):
        self.check_unary("sin")

    def test_cos(self):
        self.check_unary("cos")

    def test_tan(self):
        self.check_unary("tan")

    def test_arcsin(self):
        self.check_unary_unit("arcsin")

    def test_arccos(self):
        self.check_unary_unit("arccos")

    def test_arctan(self):
        self.check_unary("arctan")

    def test_arctan2(self):
        self.check_binary("arctan2")

    def test_hypot(self):
        self.check_binary("hypot")

    def test_deg2rad(self):
        self.check_unary("deg2rad")

    def test_rad2deg(self):
        self.check_unary("rad2deg")


@testing.with_requires("numpy>=1.21.0")
class TestUnwrap(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_unwrap_1dim(self, xp, dtype):
        a = testing.shaped_random((5,), xp, dtype)
        return xp.unwrap(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_unwrap_1dim_with_discont(self, xp, dtype):
        a = testing.shaped_random((5,), xp, dtype)
        return xp.unwrap(a, discont=1.0)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_unwrap_1dim_with_period(self, xp, dtype):
        if not has_support_aspect64() and dtype in [xp.uint8, xp.uint16]:
            # The unwrap function relies on the remainder function, and the
            # result of remainder can vary significantly between float32 and
            # float64. This discrepancy causes test failures when numpy uses
            # float64 and dpnp uses float32, especially with uint8/uint16
            # dtypes where overflow occurs
            pytest.skip("skipping due to large difference of result")
        a = testing.shaped_random((5,), xp, dtype)
        return xp.unwrap(a, period=1.2)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_unwrap_1dim_with_discont_and_period(self, xp, dtype):
        if not has_support_aspect64() and dtype in [xp.uint8, xp.uint16]:
            # The unwrap function relies on the remainder function, and the
            # result of remainder can vary significantly between float32 and
            # float64. This discrepancy causes test failures when numpy uses
            # float64 and dpnp uses float32, especially with uint8/uint16
            # dtypes where overflow occurs
            pytest.skip("skipping due to large difference of result")
        a = testing.shaped_random((5,), xp, dtype)
        return xp.unwrap(a, discont=1.0, period=1.2)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(
        rtol=1e-06, atol=1e-06, type_check=has_support_aspect64()
    )
    def test_unwrap_2dim_without_axis(self, xp, dtype):
        a = testing.shaped_random((4, 5), xp, dtype)
        return xp.unwrap(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(
        rtol=1e-06, atol=1e-06, type_check=has_support_aspect64()
    )
    def test_unwrap_2dim_with_axis(self, xp, dtype):
        a = testing.shaped_random((4, 5), xp, dtype)
        return xp.unwrap(a, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(rtol=1e-06, type_check=has_support_aspect64())
    def test_unwrap_2dim_with_discont(self, xp, dtype):
        a = testing.shaped_random((4, 5), xp, dtype)
        return xp.unwrap(a, discont=5.0, axis=1)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_unwrap_2dim_with_period(self, xp, dtype):
        a = testing.shaped_random((4, 5), xp, dtype)
        return xp.unwrap(a, axis=1, period=4.5)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_unwrap_2dim_with_discont_and_period(self, xp, dtype):
        a = testing.shaped_random((4, 5), xp, dtype)
        return xp.unwrap(a, discont=5.0, axis=1, period=4.5)
