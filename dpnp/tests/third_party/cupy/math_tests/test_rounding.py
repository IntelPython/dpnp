import unittest

import numpy
import pytest

import dpnp as cupy
from dpnp.tests.helper import has_support_aspect64
from dpnp.tests.third_party.cupy import testing


class TestRounding(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(type_check=False, atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_complex(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_complex_dtypes()
    def check_unary_complex_unsupported(self, name, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3), xp, dtype)
            with pytest.raises((TypeError, ValueError)):
                getattr(xp, name)(a)

    @testing.for_dtypes(["?", "b", "h", "i", "q", "e", "f", "d"])
    @testing.numpy_cupy_allclose(type_check=False, atol=1e-5)
    def check_unary_negative(self, name, xp, dtype):
        a = xp.array([-3, -2, -1, 1, 2, 3], dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary_negative_complex(self, name, xp, dtype):
        a = xp.array(
            [-3 - 3j, -2 - 2j, -1 - 1j, 1 + 1j, 2 + 2j, 3 + 3j], dtype=dtype
        )
        return getattr(xp, name)(a)

    def test_rint(self):
        self.check_unary("rint")
        self.check_unary_complex("rint")

    def test_rint_negative(self):
        self.check_unary_negative("rint")
        self.check_unary_negative_complex("rint")

    @testing.with_requires("numpy>=2.1")
    def test_floor(self):
        self.check_unary("floor")
        self.check_unary_complex_unsupported("floor")

    @testing.with_requires("numpy>=2.1")
    def test_ceil(self):
        self.check_unary("ceil")
        self.check_unary_complex_unsupported("ceil")

    @testing.with_requires("numpy>=2.1")
    def test_trunc(self):
        self.check_unary("trunc")
        self.check_unary_complex_unsupported("trunc")

    @testing.with_requires("numpy>=2.1")
    def test_fix(self):
        self.check_unary("fix")
        self.check_unary_complex_unsupported("fix")

    def test_around(self):
        self.check_unary("around")
        self.check_unary_complex("around")

    def test_round(self):
        self.check_unary("round")
        self.check_unary_complex("round")


@testing.parameterize(
    *testing.product(
        {
            "decimals": [-2, -1, 0, 1, 2],
        }
    )
)
class TestRound(unittest.TestCase):

    shape = (20,)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_round(self, xp, dtype):
        if dtype == numpy.bool_:
            # avoid cast problem
            a = testing.shaped_random(self.shape, xp, scale=10, dtype=dtype)
            return xp.around(a, 0)
        if dtype == numpy.float16:
            # avoid accuracy problem
            a = testing.shaped_random(self.shape, xp, scale=10, dtype=dtype)
            return xp.around(a, 0)
        a = testing.shaped_random(self.shape, xp, scale=100, dtype=dtype)
        return xp.around(a, self.decimals)

    @testing.numpy_cupy_allclose()
    def test_round_out(self, xp):
        a = testing.shaped_random(
            self.shape, xp, scale=100, dtype=cupy.default_float_type()
        )
        out = xp.empty_like(a)
        xp.around(a, self.decimals, out)
        return out


@testing.parameterize(
    *testing.product(
        {
            "decimals": [-100, -99, -90, 0, 90, 99, 100],
        }
    )
)
@pytest.mark.skipif(
    not has_support_aspect64(), reason="overflow encountered for float32 dtype"
)
class TestRoundExtreme(unittest.TestCase):

    shape = (20,)

    dtype_ = (
        [numpy.float64, numpy.complex128]
        if has_support_aspect64()
        else [numpy.float32, numpy.complex64]
    )

    @testing.for_dtypes(dtype_)
    @testing.numpy_cupy_allclose()
    def test_round_large(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, scale=1e100, dtype=dtype)
        return xp.around(a, self.decimals)

    @testing.for_dtypes(dtype_)
    @testing.numpy_cupy_allclose()
    def test_round_small(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, scale=1e-100, dtype=dtype)
        return xp.around(a, self.decimals)


@testing.parameterize(
    *testing.product(
        {
            "value": [
                (14, -1),
                (15, -1),
                (16, -1),
                (14.0, -1),
                (15.0, -1),
                (16.0, -1),
                (1.4, 0),
                (1.5, 0),
                (1.6, 0),
            ]
        }
    )
)
class TestRoundBorder(unittest.TestCase):

    @pytest.mark.skip("scalar input is not supported")
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_around_positive1(self, xp):
        a, decimals = self.value
        return xp.around(a, decimals)

    @testing.numpy_cupy_allclose(atol=1e-5, type_check=has_support_aspect64())
    def test_around_positive2(self, xp):
        a, decimals = self.value
        a = xp.asarray(a)
        return xp.around(a, decimals)

    @pytest.mark.skip("scalar input is not supported")
    @testing.numpy_cupy_allclose(atol=1e-5)
    def test_around_negative1(self, xp):
        a, decimals = self.value
        return xp.around(-a, decimals)

    @testing.numpy_cupy_allclose(atol=1e-5, type_check=has_support_aspect64())
    def test_around_negative2(self, xp):
        a, decimals = self.value
        a = xp.asarray(a)
        return xp.around(-a, decimals)
