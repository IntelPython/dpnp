import unittest

from tests.third_party.cupy import testing


class TestArithmetic(unittest.TestCase):

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        b, c = xp.modf(a)

        d = xp.empty(2 * a.size, dtype=dtype)
        for i in range(a.size):
            d[i] = b[i]
            d[i + a.size] = c[i]

        return d

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nanprod(self, xp, dtype):
        a = xp.array([-2.5, -1.5, xp.nan, 10.5, 1.5, xp.nan], dtype=dtype)
        return xp.nanprod(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_remainder(self, xp, dtype):
        a = xp.array([5, -3, -2, -1, -5], dtype=dtype)
        b = xp.full(a.size, 3, dtype=dtype)

        return xp.remainder(a, b)
