import unittest

import numpy
import pytest

import dpnp as cupy
from tests.third_party.cupy import testing


@testing.parameterize(
    {"shape": (10,), "shift": 2, "axis": None},
    {"shape": (5, 2), "shift": 1, "axis": None},
    {"shape": (5, 2), "shift": -2, "axis": None},
    {"shape": (5, 2), "shift": 1, "axis": 0},
    {"shape": (5, 2), "shift": 1, "axis": -1},
    {"shape": (10,), "shift": 35, "axis": None},
    {"shape": (5, 2), "shift": 11, "axis": 0},
    {"shape": (), "shift": 5, "axis": None},
    {"shape": (5, 2), "shift": 1, "axis": (0, 1)},
    {"shape": (5, 2), "shift": 1, "axis": (0, 0)},
    {"shape": (5, 2), "shift": 50, "axis": 0},
    {"shape": (5, 2), "shift": (2, 1), "axis": (0, 1)},
    {"shape": (5, 2), "shift": (2, 1), "axis": (0, -1)},
    {"shape": (5, 2), "shift": (2, 1), "axis": (1, -1)},
    {"shape": (5, 2), "shift": (2, 1, 3), "axis": 0},
    {"shape": (5, 2), "shift": (2, 1, 3), "axis": None},
)
class TestRoll(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype)
        return xp.roll(x, self.shift, axis=self.axis)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_roll_cupy_shift(self, xp, dtype):
        x = testing.shaped_arange(self.shape, xp, dtype)
        shift = self.shift
        return xp.roll(x, shift, axis=self.axis)


class TestRollTypeError(unittest.TestCase):
    def test_roll_invalid_shift(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(TypeError):
                xp.roll(x, "0", axis=0)

    def test_roll_invalid_axis_type(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange((5, 2), xp)
            with pytest.raises(TypeError):
                xp.roll(x, 2, axis="0")


@testing.parameterize(
    {"shape": (5, 2, 3), "shift": (2, 2, 2), "axis": (0, 1)},
    {"shape": (5, 2), "shift": 1, "axis": 2},
    {"shape": (5, 2), "shift": 1, "axis": -3},
    {"shape": (5, 2, 2), "shift": (1, 0), "axis": (0, 1, 2)},
    {"shape": (5, 2), "shift": 1, "axis": -3},
    {"shape": (5, 2), "shift": 1, "axis": (1, -3)},
)
class TestRollValueError(unittest.TestCase):
    def test_roll_invalid(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange(self.shape, xp)
            with pytest.raises(ValueError):
                xp.roll(x, self.shift, axis=self.axis)

    def test_roll_invalid_cupy_shift(self):
        for xp in (numpy, cupy):
            x = testing.shaped_arange(self.shape, xp)
            shift = self.shift
            with pytest.raises(ValueError):
                xp.roll(x, shift, axis=self.axis)
