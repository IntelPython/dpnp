import unittest
import pytest

from tests.third_party.cupy import testing


@testing.gpu
class TestOps(unittest.TestCase):

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_unary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a)

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(atol=1e-5)
    def check_binary(self, name, xp, dtype):
        a = testing.shaped_arange((2, 3), xp, dtype)
        b = testing.shaped_reverse_arange((2, 3), xp, dtype)
        return getattr(xp, name)(a, b)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_logical_and(self):
        self.check_binary('logical_and')

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_logical_or(self):
        self.check_binary('logical_or')

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_logical_xor(self):
        self.check_binary('logical_xor')

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_logical_not(self):
        self.check_unary('logical_not')
