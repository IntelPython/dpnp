import unittest
import pytest

import numpy

from tests.third_party.cupy import testing


@testing.gpu
class TestContent(unittest.TestCase):

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_array_equal()
    def check_unary_inf(self, name, xp, dtype):
        a = xp.array([-3, numpy.inf, -1, -numpy.inf, 0, 1, 2],
                     dtype=dtype)
        return getattr(xp, name)(a)

    @testing.for_dtypes('fd')
    @testing.numpy_cupy_array_equal()
    def check_unary_nan(self, name, xp, dtype):
        a = xp.array(
            [-3, numpy.NAN, -1, numpy.NAN, 0, numpy.NAN, numpy.inf],
            dtype=dtype)
        return getattr(xp, name)(a)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_isfinite(self):
        self.check_unary_inf('isfinite')

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_isinf(self):
        self.check_unary_inf('isinf')

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_isnan(self):
        self.check_unary_nan('isnan')
