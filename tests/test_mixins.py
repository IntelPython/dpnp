import unittest

import numpy

import dpnp as inp

from .helper import get_float_dtypes


class TestMatMul(unittest.TestCase):
    def test_matmul(self):
        array_data = [1.0, 2.0, 3.0, 4.0]
        size = 2

        for dtype in get_float_dtypes():
            # DPNP
            array1 = inp.reshape(
                inp.array(array_data, dtype=dtype), (size, size)
            )
            array2 = inp.reshape(
                inp.array(array_data, dtype=dtype), (size, size)
            )
            result = inp.matmul(array1, array2)
            # print(result)

            # original
            array_1 = numpy.array(array_data, dtype=dtype).reshape((size, size))
            array_2 = numpy.array(array_data, dtype=dtype).reshape((size, size))
            expected = numpy.matmul(array_1, array_2)
            # print(expected)

            # passed
            numpy.testing.assert_array_equal(expected, result)
            # still failed
            # self.assertEqual(expected, result)

    def test_matmul2(self):
        array_data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        array_data2 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        for dtype in get_float_dtypes():
            # DPNP
            array1 = inp.reshape(inp.array(array_data1, dtype=dtype), (3, 2))
            array2 = inp.reshape(inp.array(array_data2, dtype=dtype), (2, 4))
            result = inp.matmul(array1, array2)
            # print(result)

            # original
            array_1 = numpy.array(array_data1, dtype=dtype).reshape((3, 2))
            array_2 = numpy.array(array_data2, dtype=dtype).reshape((2, 4))
            expected = numpy.matmul(array_1, array_2)
            # print(expected)

            numpy.testing.assert_array_equal(expected, result)

    def test_matmul3(self):
        array_data1 = numpy.full((513, 513), 5)
        array_data2 = numpy.full((513, 513), 2)

        for dtype in get_float_dtypes():
            out = numpy.empty((513, 513), dtype=dtype)

            # DPNP
            array1 = inp.array(array_data1, dtype=dtype)
            array2 = inp.array(array_data2, dtype=dtype)
            out1 = inp.array(out, dtype=dtype)
            result = inp.matmul(array1, array2, out=out1)

            # original
            array_1 = numpy.array(array_data1, dtype=dtype)
            array_2 = numpy.array(array_data2, dtype=dtype)
            expected = numpy.matmul(array_1, array_2, out=out)

            numpy.testing.assert_array_equal(expected, result)


if __name__ == "__main__":
    unittest.main()
