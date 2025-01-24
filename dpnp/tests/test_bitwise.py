import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp as inp

from .helper import assert_dtype_allclose, get_abs_array, get_integer_dtypes


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
@pytest.mark.parametrize("dtype", [inp.bool] + get_integer_dtypes())
class TestBitwise:
    @staticmethod
    def array_or_scalar(xp, data, dtype=None):
        if numpy.isscalar(data):
            if dtype == inp.bool:
                return numpy.dtype(dtype).type(data)
            return data

        if numpy.issubdtype(dtype, numpy.unsignedinteger):
            data = xp.abs(xp.array(data))
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

        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a &= dp_b
            np_a &= np_b
            assert_array_equal(dp_a, np_a)

    def test_bitwise_or(self, lhs, rhs, dtype):
        dp_a, dp_b, np_a, np_b = self._test_binary_int(
            "bitwise_or", lhs, rhs, dtype
        )
        assert_array_equal(dp_a | dp_b, np_a | np_b)

        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a |= dp_b
            np_a |= np_b
            assert_array_equal(dp_a, np_a)

    def test_bitwise_xor(self, lhs, rhs, dtype):
        dp_a, dp_b, np_a, np_b = self._test_binary_int(
            "bitwise_xor", lhs, rhs, dtype
        )
        assert_array_equal(dp_a ^ dp_b, np_a ^ np_b)

        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a ^= dp_b
            np_a ^= np_b
            assert_array_equal(dp_a, np_a)

    def test_invert(self, lhs, rhs, dtype):
        dp_a, np_a = self._test_unary_int("invert", lhs, dtype)
        assert_array_equal(~dp_a, ~np_a)

    def test_left_shift(self, lhs, rhs, dtype):
        dp_a, dp_b, np_a, np_b = self._test_binary_int(
            "left_shift", lhs, rhs, dtype
        )
        assert_array_equal(dp_a << dp_b, np_a << np_b)

        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a <<= dp_b
            np_a <<= np_b
            assert_array_equal(dp_a, np_a)

    def test_right_shift(self, lhs, rhs, dtype):
        dp_a, dp_b, np_a, np_b = self._test_binary_int(
            "right_shift", lhs, rhs, dtype
        )
        assert_array_equal(dp_a >> dp_b, np_a >> np_b)

        if (
            not (inp.isscalar(dp_a) or inp.isscalar(dp_b))
            and dp_a.shape == dp_b.shape
        ):
            dp_a >>= dp_b
            np_a >>= np_b
            assert_array_equal(dp_a, np_a)

    def test_bitwise_aliase1(self, lhs, rhs, dtype):
        if numpy.isscalar(lhs):
            pytest.skip("Input can't be scalar")
        dp_a = self.array_or_scalar(inp, lhs, dtype=dtype)
        result1 = inp.invert(dp_a)
        result2 = inp.bitwise_invert(dp_a)
        assert_array_equal(result1, result2)

        result2 = inp.bitwise_not(dp_a)
        assert_array_equal(result1, result2)

    def test_bitwise_aliase2(self, lhs, rhs, dtype):
        if dtype == inp.bool:
            pytest.skip("A shift operation isn't implemented for bool type")
        elif numpy.isscalar(lhs) and numpy.isscalar(rhs):
            pytest.skip("Both inputs can't be scalars")

        dp_a = self.array_or_scalar(inp, lhs, dtype=dtype)
        dp_b = self.array_or_scalar(inp, rhs, dtype=dtype)
        result1 = inp.left_shift(dp_a, dp_b)
        result2 = inp.bitwise_left_shift(dp_a, dp_b)
        assert_array_equal(result1, result2)

        result1 = inp.right_shift(dp_a, dp_b)
        result2 = inp.bitwise_right_shift(dp_a, dp_b)
        assert_array_equal(result1, result2)


@pytest.mark.parametrize("dtype", get_integer_dtypes())
def test_invert_out(dtype):
    low = 0 if numpy.issubdtype(dtype, numpy.unsignedinteger) else -5
    np_a = numpy.arange(low, 5, dtype=dtype)
    dp_a = inp.array(np_a)

    expected = numpy.invert(np_a)
    dp_out = inp.empty(expected.shape, dtype=expected.dtype)
    result = inp.invert(dp_a, out=dp_out)
    assert result is dp_out
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("dtype1", [inp.bool] + get_integer_dtypes())
@pytest.mark.parametrize("dtype2", [inp.bool] + get_integer_dtypes())
class TestBitwiseInplace:
    def test_bitwise_and(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = inp.array(a), inp.array(b)

        a &= True
        ia &= True
        assert_array_equal(ia, a)

        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a &= b
            ia &= ib
            assert_array_equal(ia, a)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            a &= b
            ia &= ib
            assert_array_equal(ia, a)
        else:
            with pytest.raises(TypeError):
                a &= b

            with pytest.raises(ValueError):
                ia &= ib

    def test_bitwise_or(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = inp.array(a), inp.array(b)

        a |= False
        ia |= False
        assert_array_equal(ia, a)

        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a |= b
            ia |= ib
            assert_array_equal(ia, a)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            a |= b
            ia |= ib
            assert_array_equal(ia, a)
        else:
            with pytest.raises(TypeError):
                a |= b

            with pytest.raises(ValueError):
                ia |= ib

    def test_bitwise_xor(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = inp.array(a), inp.array(b)

        a ^= False
        ia ^= False
        assert_array_equal(ia, a)

        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = inp.array(a), inp.array(b)
        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a ^= b
            ia ^= ib
            assert_array_equal(ia, a)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            a ^= b
            ia ^= ib
            assert_array_equal(ia, a)
        else:
            with pytest.raises(TypeError):
                a ^= b

            with pytest.raises(ValueError):
                ia ^= ib


@pytest.mark.parametrize("dtype1", get_integer_dtypes())
@pytest.mark.parametrize("dtype2", get_integer_dtypes())
class TestBitwiseShiftInplace:
    def test_bitwise_left_shift(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, 2, 0, 1, 0], dtype=dtype2)
        ia, ib = inp.array(a), inp.array(b)

        a <<= True
        ia <<= True
        assert_array_equal(ia, a)

        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a <<= b
            ia <<= ib
            assert_array_equal(ia, a)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            a <<= b
            ia <<= ib
            assert_array_equal(ia, a)
        else:
            with pytest.raises(TypeError):
                a <<= b

            with pytest.raises(ValueError):
                ia <<= ib

    def test_bitwise_right_shift(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, 2, 0, 1, 0], dtype=dtype2)
        ia, ib = inp.array(a), inp.array(b)

        a >>= True
        ia >>= True
        assert_array_equal(ia, a)

        if numpy.issubdtype(dtype1, numpy.signedinteger) and numpy.issubdtype(
            dtype2, numpy.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a >>= b
            ia >>= ib
            assert_array_equal(ia, a)
        elif numpy.can_cast(dtype2, dtype1, casting="same_kind"):
            a >>= b
            ia >>= ib
            assert_array_equal(ia, a)
        else:
            with pytest.raises(TypeError):
                a >>= b

            with pytest.raises(ValueError):
                ia >>= ib
