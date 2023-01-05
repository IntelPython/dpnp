import pytest

import dpnp as inp

import numpy


@pytest.mark.parametrize("lhs", [[[-7, -6, -5, -4, -3, -2, -1], [0, 1, 2, 3, 4, 5, 6]], [-3, -2, -1, 0, 1, 2, 3], 0])
@pytest.mark.parametrize("rhs", [[[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]], [0, 1, 2, 3, 4, 5, 6], 3])
@pytest.mark.parametrize("dtype", [numpy.int32, numpy.int64])
class TestBitwise:

    @staticmethod
    def array_or_scalar(xp, data, dtype=None):
        if numpy.isscalar(data):
            return data

        return xp.array(data, dtype=dtype)

    def _test_unary_int(self, name, data, dtype):
        a = self.array_or_scalar(inp, data, dtype=dtype)
        result = getattr(inp, name)(a)

        a = self.array_or_scalar(numpy, data, dtype=dtype)
        expected = getattr(numpy, name)(a)

        numpy.testing.assert_array_equal(result, expected)

    def _test_binary_int(self, name, lhs, rhs, dtype):
        a = self.array_or_scalar(inp, lhs, dtype=dtype)
        b = self.array_or_scalar(inp, rhs, dtype=dtype)
        result = getattr(inp, name)(a, b)

        a = self.array_or_scalar(numpy, lhs, dtype=dtype)
        b = self.array_or_scalar(numpy, rhs, dtype=dtype)
        expected = getattr(numpy, name)(a, b)

        numpy.testing.assert_array_equal(result, expected)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_bitwise_and(self, lhs, rhs, dtype):
        self._test_binary_int('bitwise_and', lhs, rhs, dtype)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_bitwise_or(self, lhs, rhs, dtype):
        self._test_binary_int('bitwise_or', lhs, rhs, dtype)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_bitwise_xor(self, lhs, rhs, dtype):
        self._test_binary_int('bitwise_xor', lhs, rhs, dtype)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_invert(self, lhs, rhs, dtype):
        self._test_unary_int('invert', lhs, dtype)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_left_shift(self, lhs, rhs, dtype):
        self._test_binary_int('left_shift', lhs, rhs, dtype)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_right_shift(self, lhs, rhs, dtype):
        self._test_binary_int('right_shift', lhs, rhs, dtype)
