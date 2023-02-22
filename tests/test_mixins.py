import unittest
from .helper import skip_or_check_if_dtype_not_supported

import dpnp as inp

import numpy


class TestMatMul(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dtype = inp.float64 if skip_or_check_if_dtype_not_supported(
            inp.float64, check_dtype=True
        ) else inp.float32

    def test_matmul(self):
        array_data = [1., 2., 3., 4.]
        size = 2

        # DPNP
        array1 = inp.reshape(inp.array(array_data, dtype=self.dtype), (size, size))
        array2 = inp.reshape(inp.array(array_data, dtype=self.dtype), (size, size))
        result = inp.matmul(array1, array2)
        # print(result)

        # original
        array_1 = numpy.array(array_data, dtype=self.dtype).reshape((size, size))
        array_2 = numpy.array(array_data, dtype=self.dtype).reshape((size, size))
        expected = numpy.matmul(array_1, array_2)
        # print(expected)

        # passed
        numpy.testing.assert_array_equal(expected, result)
        # still failed
        # self.assertEqual(expected, result)

    def test_matmul2(self):
        array_data1 = [1., 2., 3., 4., 5., 6.]
        array_data2 = [1., 2., 3., 4., 5., 6., 7., 8.]

        # DPNP
        array1 = inp.reshape(inp.array(array_data1, dtype=self.dtype), (3, 2))
        array2 = inp.reshape(inp.array(array_data2, dtype=self.dtype), (2, 4))
        result = inp.matmul(array1, array2)
        # print(result)

        # original
        array_1 = numpy.array(array_data1, dtype=self.dtype).reshape((3, 2))
        array_2 = numpy.array(array_data2, dtype=self.dtype).reshape((2, 4))
        expected = numpy.matmul(array_1, array_2)
        # print(expected)

        numpy.testing.assert_array_equal(expected, result)

    def test_matmul3(self):
        array_data1 = numpy.full((513, 513), 5)
        array_data2 = numpy.full((513, 513), 2)
        out = numpy.empty((513, 513), dtype=self.dtype)

        # DPNP
        array1 = inp.array(array_data1, dtype=self.dtype)
        array2 = inp.array(array_data2, dtype=self.dtype)
        out1 = inp.array(out, dtype=self.dtype)
        result = inp.matmul(array1, array2, out=out1)

        # original
        array_1 = numpy.array(array_data1, dtype=self.dtype)
        array_2 = numpy.array(array_data2, dtype=self.dtype)
        expected = numpy.matmul(array_1, array_2, out=out)

        numpy.testing.assert_array_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
