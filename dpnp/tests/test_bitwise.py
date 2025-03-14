import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp

from .helper import (
    assert_dtype_allclose,
    get_abs_array,
    get_integer_dtypes,
    numpy_version,
)
from .third_party.cupy import testing


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
@pytest.mark.parametrize("dtype", [dpnp.bool] + get_integer_dtypes())
class TestBitwiseBinary:
    @staticmethod
    def array_or_scalar(data, dtype=None):
        if numpy.isscalar(data):
            if dtype == dpnp.bool:
                data = numpy.dtype(dtype).type(data)
            return data, data

        data = get_abs_array(data, dtype=dtype)
        return data, dpnp.array(data)

    def _test_binary(self, name, lhs, rhs, dtype):
        if numpy.isscalar(lhs) and numpy.isscalar(rhs):
            pytest.skip("Both inputs can't be scalars")

        a, ia = self.array_or_scalar(lhs, dtype=dtype)
        b, ib = self.array_or_scalar(rhs, dtype=dtype)

        result = getattr(dpnp, name)(ia, ib)
        expected = getattr(numpy, name)(a, b)
        assert_array_equal(result, expected)

        iout = dpnp.empty_like(result)
        result = getattr(dpnp, name)(ia, ib, out=iout)
        assert result is iout
        assert_array_equal(result, expected)

        return (ia, ib, a, b)

    def test_bitwise_and(self, lhs, rhs, dtype):
        ia, ib, a, b = self._test_binary("bitwise_and", lhs, rhs, dtype)
        assert_array_equal(ia & ib, a & b)

    def test_bitwise_or(self, lhs, rhs, dtype):
        ia, ib, a, b = self._test_binary("bitwise_or", lhs, rhs, dtype)
        assert_array_equal(ia | ib, a | b)

    def test_bitwise_xor(self, lhs, rhs, dtype):
        ia, ib, a, b = self._test_binary("bitwise_xor", lhs, rhs, dtype)
        assert_array_equal(ia ^ ib, a ^ b)

    def test_left_shift(self, lhs, rhs, dtype):
        if numpy_version() >= "2.0.0":
            _ = self._test_binary("bitwise_left_shift", lhs, rhs, dtype)
        ia, ib, a, b = self._test_binary("left_shift", lhs, rhs, dtype)
        assert_array_equal(ia << ib, a << b)

    def test_right_shift(self, lhs, rhs, dtype):
        if numpy_version() >= "2.0.0":
            _ = self._test_binary("bitwise_right_shift", lhs, rhs, dtype)
        ia, ib, a, b = self._test_binary("right_shift", lhs, rhs, dtype)
        assert_array_equal(ia >> ib, a >> b)


@pytest.mark.parametrize("dtype1", [dpnp.bool] + get_integer_dtypes())
@pytest.mark.parametrize("dtype2", [dpnp.bool] + get_integer_dtypes())
class TestBitwiseInplace:
    def test_bitwise_and(self, dtype1, dtype2):
        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

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
        ia, ib = dpnp.array(a), dpnp.array(b)

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
        ia, ib = dpnp.array(a), dpnp.array(b)

        a ^= False
        ia ^= False
        assert_array_equal(ia, a)

        a = get_abs_array([[-7, 6, -3, 2, -1], [0, -3, 4, 5, -6]], dtype=dtype1)
        b = get_abs_array([5, -2, 0, 1, 0], dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)
        if dpnp.issubdtype(dtype1, dpnp.signedinteger) and dpnp.issubdtype(
            dtype2, dpnp.uint64
        ):
            # For this special case, NumPy raises an error but dpnp works
            b = b.astype(numpy.int64)
            a ^= b
            ia ^= ib
            assert_array_equal(ia, a)
        elif dpnp.can_cast(dtype2, dtype1, casting="same_kind"):
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
        ia, ib = dpnp.array(a), dpnp.array(b)

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
        ia, ib = dpnp.array(a), dpnp.array(b)

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


@pytest.mark.parametrize(
    "val",
    [
        [[-7, -6, -5, -4, -3, -2, -1], [0, 1, 2, 3, 4, 5, 6]],
        [-3, -2, -1, 0, 1, 2, 3],
    ],
)
@pytest.mark.parametrize("dtype", [dpnp.bool] + get_integer_dtypes())
class TestBitwiseUnary:
    def _test_unary(self, name, data, dtype):
        a = get_abs_array(data, dtype=dtype)
        ia = dpnp.array(a)

        result = getattr(dpnp, name)(ia)
        expected = getattr(numpy, name)(a)
        assert_array_equal(result, expected)

        iout = dpnp.empty_like(result)
        result = getattr(dpnp, name)(ia, out=iout)
        assert result is iout
        assert_array_equal(result, expected)

        return (ia, a)

    @testing.with_requires("numpy>=2.0")
    def test_bitwise_count(self, val, dtype):
        _ = self._test_unary("bitwise_count", val, dtype)

    def test_invert(self, val, dtype):
        if numpy_version() >= "2.0.0":
            _ = self._test_unary("bitwise_not", val, dtype)
            _ = self._test_unary("bitwise_invert", val, dtype)
        ia, a = self._test_unary("invert", val, dtype)
        assert_array_equal(~ia, ~a)
