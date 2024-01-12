import unittest

import numpy

from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing


# Note: numpy.sum() always upcast integers to (u)int64 and float32 to
# float64 for dtype=None. `np.sum` does that too for integers, but not for
# float32, so we need to special-case it for these tests
def _get_dtype_kwargs(xp, dtype):
    if xp is numpy and dtype == numpy.float32 and has_support_aspect64():
        return {"dtype": numpy.float64}
    return {}


class TestArithmetic(unittest.TestCase):
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf_part1(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        b, _ = xp.modf(a)

        return b

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_modf_part2(self, xp, dtype):
        a = xp.array([-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=dtype)
        _, c = xp.modf(a)

        return c

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(type_check=False)
    def test_nanprod(self, xp, dtype):
        a = xp.array([-2.5, -1.5, xp.nan, 10.5, 1.5, xp.nan], dtype=dtype)
        return xp.nanprod(a)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_nansum(self, xp, dtype):
        a = xp.array([-2.5, -1.5, xp.nan, 10.5, 1.5, xp.nan], dtype=dtype)
        return xp.nansum(a, **_get_dtype_kwargs(xp, a.dtype))

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_remainder(self, xp, dtype):
        a = xp.array([5, -3, -2, -1, -5], dtype=dtype)
        b = xp.full(a.size, 3, dtype=dtype)

        return xp.remainder(a, b)
