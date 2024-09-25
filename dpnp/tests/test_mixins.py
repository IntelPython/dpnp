import unittest

from .third_party.cupy import testing


class TestMatMul(unittest.TestCase):
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matmul(self, xp, dtype):
        data = [1.0, 2.0, 3.0, 4.0]
        shape = (2, 2)

        a = xp.array(data, dtype=dtype).reshape(shape)
        b = xp.array(data, dtype=dtype).reshape(shape)

        return xp.matmul(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matmul2(self, xp, dtype):
        data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        a = xp.array(data1, dtype=dtype).reshape(3, 2)
        b = xp.array(data2, dtype=dtype).reshape(2, 4)

        return xp.matmul(a, b)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_matmul3(self, xp, dtype):
        data1 = xp.full((513, 513), 5)
        data2 = xp.full((513, 513), 2)
        out = xp.empty((513, 513), dtype=dtype)

        a = xp.array(data1, dtype=dtype)
        b = xp.array(data2, dtype=dtype)

        xp.matmul(a, b, out=out)

        return out


if __name__ == "__main__":
    unittest.main()
