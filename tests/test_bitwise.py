import unittest

import dpnp as inp

import numpy


class TestBitwise(unittest.TestCase):

    def _test_unary_int(self, name):
        data = [-3, -2, -1, 0, 1, 2, 3]
        for dtype in (numpy.int32, numpy.int64):
            with self.subTest(dtype=dtype):
                a = inp.array(data, dtype=dtype)
                result = getattr(inp, name)(a)

                a = numpy.array(data, dtype=dtype)
                expected = getattr(numpy, name)(a)

                numpy.testing.assert_array_equal(result, expected)

    def _test_binary_int(self, name):
        data1, data2 = [-3, -2, -1, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6]
        for dtype in (numpy.int32, numpy.int64):
            with self.subTest(dtype=dtype):
                a = inp.array(data1, dtype=dtype)
                b = inp.array(data2, dtype=dtype)
                result = getattr(inp, name)(a, b)

                a = numpy.array(data1, dtype=dtype)
                b = numpy.array(data2, dtype=dtype)
                expected = getattr(numpy, name)(a, b)

                numpy.testing.assert_array_equal(result, expected)

    def test_bitwise_and(self):
        self._test_binary_int('bitwise_and')

    def test_bitwise_or(self):
        self._test_binary_int('bitwise_or')

    def test_bitwise_xor(self):
        self._test_binary_int('bitwise_xor')

    def test_invert(self):
        self._test_unary_int('invert')

    def test_left_shift(self):
        self._test_binary_int('left_shift')

    def test_right_shift(self):
        self._test_binary_int('right_shift')


if __name__ == '__main__':
    unittest.main()
