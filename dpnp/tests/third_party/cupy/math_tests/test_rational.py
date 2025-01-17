import unittest

import numpy
import pytest

import dpnp as cupy
from dpnp.tests.helper import is_cuda_device
from dpnp.tests.third_party.cupy import testing


class TestRational(unittest.TestCase):

    @testing.for_dtypes(["?", "e", "f", "d", "F", "D"])
    def test_gcd_dtype_check(self, dtype):
        # TODO: remove it once the issue with CUDA support is resolved
        # for dpnp.random
        if is_cuda_device():
            a = cupy.asarray(
                numpy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
            )
            b = cupy.asarray(
                numpy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
            )
        else:
            a = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
            b = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        with pytest.raises(ValueError):
            cupy.gcd(a, b)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_gcd_check_boundary_cases(self, xp, dtype):
        a = xp.array([0, -10, -5, 10, 410, 1, 6, 33])
        b = xp.array([0, 5, -10, -5, 20, 51, 6, 42])
        return xp.gcd(a, b)

    @testing.for_dtypes(["?", "e", "f", "d", "F", "D"])
    def test_lcm_dtype_check(self, dtype):
        if is_cuda_device():
            a = cupy.asarray(
                numpy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
            )
            b = cupy.asarray(
                numpy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
            )
        else:
            a = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
            b = cupy.random.randint(-10, 10, size=(10, 10)).astype(dtype)
        with pytest.raises(ValueError):
            cupy.lcm(a, b)

    @testing.for_int_dtypes(no_bool=True)
    @testing.numpy_cupy_array_equal()
    def test_lcm_check_boundary_cases(self, xp, dtype):
        a = xp.array([0, -10, -5, 10, 410, 1, 6, 33])
        b = xp.array([0, 5, -10, -5, 20, 51, 6, 42])
        return xp.lcm(a, b)
