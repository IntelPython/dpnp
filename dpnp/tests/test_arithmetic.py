import unittest

from .third_party.cupy import testing


class TestArithmetic(unittest.TestCase):
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf_part1(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        b, _ = xp.modf(a)
        return b

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf_part2(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        _, c = xp.modf(a)
        return c

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(type_check=False)
    def test_nanprod(self, xp, dtype):
        a = xp.array([-2.5, -1.5, xp.nan, 10.5, 1.5, xp.nan], dtype=dtype)
        return xp.nanprod(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nansum(self, xp, dtype):
        a = xp.array([-2.5, -1.5, xp.nan, 10.5, 1.5, xp.nan], dtype=dtype)
        return xp.nansum(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_remainder(self, xp, dtype):
        a = xp.array([5, -3, -2, -1, -5], dtype=dtype)
        b = xp.full(a.size, 3, dtype=dtype)
        return xp.remainder(a, b)
