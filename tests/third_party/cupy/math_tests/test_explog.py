import unittest
import warnings

import numpy
import pytest

from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing


class TestExplog(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, type_check=has_support_aspect64())
    def check_unary(self, name, xp, dtype, no_complex=False):
        if no_complex:
            if numpy.dtype(dtype).kind == "c":
                return xp.array(True)
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, type_check=has_support_aspect64())
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
        with testing.NumpyError(divide="ignore"):
            self.check_unary("log")

    def test_log10(self):
        with testing.NumpyError(divide="ignore"):
            self.check_unary("log10")

    def test_log2(self):
        with testing.NumpyError(divide="ignore"):
            self.check_unary("log2")

    def test_log1p(self):
        self.check_unary("log1p")

    def test_logaddexp(self):
        self.check_binary("logaddexp", no_complex=True)

    def test_logaddexp2(self):
        self.check_binary("logaddexp2", no_complex=True)

    @pytest.mark.parametrize("val", [numpy.inf, -numpy.inf])
    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_logaddexp2_infinities(self, xp, dtype, val):
        a = xp.full((2, 3), val, dtype=dtype)
        return xp.logaddexp2(a, a)


@pytest.mark.parametrize("val", [numpy.inf, -numpy.inf])
@testing.for_float_dtypes()
@testing.numpy_cupy_allclose()
def test_logaddexp_infinities(xp, dtype, val):
    a = xp.full((2, 3), val, dtype=dtype)
    return xp.logaddexp(a, a)


@testing.for_float_dtypes()
@testing.numpy_cupy_allclose()
def test_logaddexp_nan(xp, dtype):
    a = xp.full((2, 3), xp.nan, dtype=dtype)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = xp.logaddexp(a, a)
    return result
