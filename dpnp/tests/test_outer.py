import unittest

import numpy as np
import pytest
from numpy.testing import assert_raises

import dpnp as dp

from .third_party.cupy import testing


class TestOuter(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_two_vectors(self, xp, dtype):
        a = xp.ones((10,), dtype=dtype)
        b = xp.linspace(-2, 2, 5, dtype=dtype)

        return xp.outer(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_two_matrix(self, xp, dtype):
        a = xp.ones((10, 10, 10), dtype=dtype)
        b = xp.full(shape=(3, 7), fill_value=42, dtype=dtype)

        return xp.outer(a, b)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_the_same_vector(self, xp, dtype):
        a = xp.full(shape=(100,), fill_value=7, dtype=dtype)
        return xp.outer(a, a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_the_same_matrix(self, xp, dtype):
        a = xp.arange(27, dtype=dtype).reshape(3, 3, 3)
        return xp.outer(a, a)

    @testing.with_requires("numpy>=2.0")
    @testing.numpy_cupy_allclose()
    def test_linalg_outer(self, xp):
        a = xp.arange(10)
        b = xp.arange(10) - 5

        return xp.linalg.outer(a, b)

    @testing.with_requires("numpy>=2.0")
    def test_linalg_outer_error(self):
        for xp in (np, dp):
            a = xp.arange(9).reshape(3, 3)
            with pytest.raises(ValueError):
                xp.linalg.outer(a, a)


class TestScalarOuter(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=False)
    def test_first_is_scalar(self, xp, dtype):
        scalar = 4
        a = xp.arange(24, dtype=dtype).reshape(2, 3, 4)
        return xp.outer(scalar, a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(type_check=False)
    def test_second_is_scalar(self, xp, dtype):
        scalar = 5
        a = xp.arange(24, dtype=dtype).reshape(2, 3, 4)
        return xp.outer(a, scalar)


class TestListOuter(unittest.TestCase):
    def test_list(self):
        a = np.arange(27).reshape(3, 3, 3)
        b: list[list[list[int]]] = a.tolist()
        dp_a = dp.array(a)

        with assert_raises(TypeError):
            dp.outer(b, dp_a)
            dp.outer(dp_a, b)
            dp.outer(b, b)
