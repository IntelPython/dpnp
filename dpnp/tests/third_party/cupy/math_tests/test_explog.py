import numpy
import pytest

from dpnp.tests.helper import has_support_aspect64
from dpnp.tests.third_party.cupy import testing


class TestExplog:

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, type_check=has_support_aspect64())
    def check_unary(self, name, xp, dtype, no_complex=False):
        if no_complex:
            if numpy.dtype(dtype).kind == "c":
                return xp.array(True)
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    # rtol=1e-3 is added for dpnp to pass the test when dtype is int8/unint8
    # for such a case, output dtype is float16
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-3, atol=1e-5, type_check=has_support_aspect64()
    )
    def check_binary(self, name, xp, dtype, no_complex=False):
        if no_complex:
            if numpy.dtype(dtype).kind == "c":
                return xp.array(True)
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    def test_exp(self):
        self.check_unary("exp")

    def test_expm1(self):
        self.check_unary("expm1")

    def test_exp2(self):
        self.check_unary("exp2")

    def test_log(self):
        with numpy.errstate(divide="ignore"):
            self.check_unary("log")

    def test_log10(self):
        with numpy.errstate(divide="ignore"):
            self.check_unary("log10")

    def test_log2(self):
        with numpy.errstate(divide="ignore"):
            self.check_unary("log2")

    def test_log1p(self):
        self.check_unary("log1p")

    def test_logaddexp(self):
        self.check_binary("logaddexp", no_complex=True)

    @pytest.mark.parametrize("val", [numpy.inf, -numpy.inf])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_logaddexp_infinities(self, xp, dtype, val):
        a = xp.full((2, 3), val, dtype=dtype)
        return xp.logaddexp(a, a)

    def test_logaddexp2(self):
        self.check_binary("logaddexp2", no_complex=True)

    @pytest.mark.parametrize("val", [numpy.inf, -numpy.inf])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_logaddexp2_infinities(self, xp, dtype, val):
        a = xp.full((2, 3), val, dtype=dtype)
        return xp.logaddexp2(a, a)
