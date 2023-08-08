import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp as inp


@pytest.mark.parametrize(
    "lhs",
    [
        [[-7, -6, -5, -4, -3, -2, -1], [0, 1, 2, 3, 4, 5, 6]],
        [-3, -2, -1, 0, 1, 2, 3],
        0,
    ],
)
@pytest.mark.parametrize(
    "rhs",
    [
        [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13]],
        [0, 1, 2, 3, 4, 5, 6],
        3,
    ],
)
@pytest.mark.parametrize("dtype", [inp.bool, inp.int32, inp.int64])
class TestBitwise:
    @staticmethod
    def array_or_scalar(xp, data, dtype=None):
        if numpy.isscalar(data):
            if dtype == inp.bool:
                return numpy.dtype(dtype).type(data)
            return data

        return xp.array(data, dtype=dtype)

    def _test_unary_int(self, name, data, dtype):
        if numpy.isscalar(data):
            pytest.skip("Input can't be scalar")
        dp_a = self.array_or_scalar(inp, data, dtype=dtype)
        result = getattr(inp, name)(dp_a)

        np_a = self.array_or_scalar(numpy, data, dtype=dtype)
        expected = getattr(numpy, name)(np_a)

        assert_array_equal(result, expected)
        return (dp_a, np_a)

    def _test_binary_int(self, name, lhs, rhs, dtype):
        if name in ("left_shift", "right_shift") and dtype == inp.bool:
            pytest.skip("A shift operation isn't implemented for bool type")
        elif numpy.isscalar(lhs) and numpy.isscalar(rhs):
            pytest.skip("Both inputs can't be scalars")

        dp_a = self.array_or_scalar(inp, lhs, dtype=dtype)
        dp_b = self.array_or_scalar(inp, rhs, dtype=dtype)
        result = getattr(inp, name)(dp_a, dp_b)

        np_a = self.array_or_scalar(numpy, lhs, dtype=dtype)
        np_b = self.array_or_scalar(numpy, rhs, dtype=dtype)
        expected = getattr(numpy, name)(np_a, np_b)

        assert_array_equal(result, expected)
        return (dp_a, dp_b, np_a, np_b)

    def test_bitwise_and(self, lhs, rhs, dtype):
        dp_a, dp_b, np_a, np_b = self._test_binary_int(
            "bitwise_and", lhs, rhs, dtype
        )
        assert_array_equal(dp_a & dp_b, np_a & np_b)

        """
        TODO: unmute once dpctl support that
        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a &= dp_b
            np_a &= np_b
            assert_array_equal(dp_a, np_a)
        """

    def test_bitwise_or(self, lhs, rhs, dtype):
        dp_a, dp_b, np_a, np_b = self._test_binary_int(
            "bitwise_or", lhs, rhs, dtype
        )
        assert_array_equal(dp_a | dp_b, np_a | np_b)

        """
        TODO: unmute once dpctl support that
        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a |= dp_b
            np_a |= np_b
            assert_array_equal(dp_a, np_a)
        """

    def test_bitwise_xor(self, lhs, rhs, dtype):
        dp_a, dp_b, np_a, np_b = self._test_binary_int(
            "bitwise_xor", lhs, rhs, dtype
        )
        assert_array_equal(dp_a ^ dp_b, np_a ^ np_b)

        """
        TODO: unmute once dpctl support that
        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a ^= dp_b
            np_a ^= np_b
            assert_array_equal(dp_a, np_a)
        """

    def test_invert(self, lhs, rhs, dtype):
        dp_a, np_a = self._test_unary_int("invert", lhs, dtype)
        assert_array_equal(~dp_a, ~np_a)

    def test_left_shift(self, lhs, rhs, dtype):
        dp_a, dp_b, np_a, np_b = self._test_binary_int(
            "left_shift", lhs, rhs, dtype
        )
        assert_array_equal(dp_a << dp_b, np_a << np_b)

        """
        TODO: unmute once dpctl support that
        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a <<= dp_b
            np_a <<= np_b
            assert_array_equal(dp_a, np_a)
        """

    def test_right_shift(self, lhs, rhs, dtype):
        dp_a, dp_b, np_a, np_b = self._test_binary_int(
            "right_shift", lhs, rhs, dtype
        )
        assert_array_equal(dp_a >> dp_b, np_a >> np_b)

        """
        TODO: unmute once dpctl support that
        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a >>= dp_b
            np_a >>= np_b
            assert_array_equal(dp_a, np_a)
        """
