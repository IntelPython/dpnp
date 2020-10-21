import unittest

from tests.third_party.cupy import testing


class TestArithmeticModf(unittest.TestCase):

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
