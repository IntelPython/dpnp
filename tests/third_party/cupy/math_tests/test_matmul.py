import operator
import unittest

import numpy
import pytest

import dpnp
from tests.third_party.cupy import testing


@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                # dot test
                ((3, 2), (2, 4)),
                ((3, 0), (0, 4)),
                ((0, 2), (2, 4)),
                ((3, 2), (2, 0)),
                ((2,), (2, 4)),
                ((0,), (0, 4)),
                ((3, 2), (2,)),
                ((3, 0), (0,)),
                ((2,), (2,)),
                ((0,), (0,)),
                # matmul test
                ((5, 3, 2), (5, 2, 4)),
                # ((0, 3, 2), (0, 2, 4)),
                # ((5, 3, 2), (2, 4)),
                # ((0, 3, 2), (2, 4)),
                # ((3, 2), (5, 2, 4)),
                # ((3, 2), (0, 2, 4)),
                # ((5, 3, 2), (1, 2, 4)),
                # ((0, 3, 2), (1, 2, 4)),
                # ((1, 3, 2), (5, 2, 4)),
                # ((1, 3, 2), (0, 2, 4)),
                # ((5, 3, 2), (2,)),
                # ((5, 3, 0), (0,)),
                # ((2,), (5, 2, 4)),
                # ((0,), (5, 0, 4)),
                # ((2, 2, 3, 2), (2, 2, 2, 4)),
                # ((5, 0, 3, 2), (5, 0, 2, 4)),
                # ((6, 5, 3, 2), (2, 4)),
                # ((5, 0, 3, 2), (2, 4)),
                # ((3, 2), (6, 5, 2, 4)),
                # ((3, 2), (5, 0, 2, 4)),
                # ((1, 5, 3, 2), (6, 1, 2, 4)),
                # ((1, 0, 3, 2), (6, 1, 2, 4)),
                # ((6, 1, 3, 2), (1, 5, 2, 4)),
                # ((6, 1, 3, 2), (1, 0, 2, 4)),
                # ((6, 5, 3, 2), (2,)),
                # ((6, 5, 3, 0), (0,)),
                # ((2,), (6, 5, 2, 4)),
                # ((0,), (6, 5, 0, 4)),
                ((1, 3, 3), (10, 1, 3, 1)),
            ],
        }
    )
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@testing.gpu
class TestMatmul(unittest.TestCase):
    @testing.for_all_dtypes(name="dtype1")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype1)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name="dtype1")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype1)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                ((6, 5, 3, 2), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (6, 1, 2, 4)),
                ((6, 5, 3, 2), (1, 5, 2, 4)),
                ((6, 5, 3, 2), (1, 1, 2, 4)),
                ((6, 1, 3, 2), (6, 5, 2, 4)),
                ((1, 5, 3, 2), (6, 5, 2, 4)),
                ((1, 1, 3, 2), (6, 5, 2, 4)),
                ((3, 2), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (2, 4)),
                ((2,), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (2,)),
            ],
        }
    )
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@testing.gpu
class TestMatmulLarge(unittest.TestCase):
    # Avoid overflow
    skip_dtypes = {
        (numpy.int8, numpy.uint8),
        (numpy.int8, numpy.int16),
        (numpy.int8, numpy.float16),
        (numpy.uint8, numpy.uint8),
        (numpy.uint8, numpy.int16),
        (numpy.uint8, numpy.uint16),
        (numpy.int16, numpy.int16),
        (numpy.uint16, numpy.uint16),
    }

    @testing.for_all_dtypes(name="dtype1")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1):
        if (dtype1, dtype1) in self.skip_dtypes or (
            dtype1,
            dtype1,
        ) in self.skip_dtypes:
            return xp.array([])
        x1 = testing.shaped_random(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_random(self.shape_pair[1], xp, dtype1)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name="dtype1")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1):
        if (dtype1, dtype1) in self.skip_dtypes or (
            dtype1,
            dtype1,
        ) in self.skip_dtypes:
            return xp.array([])
        shape1, shape2 = self.shape_pair
        x1 = testing.shaped_random(shape1, xp, dtype1)
        x2 = testing.shaped_random(shape2, xp, dtype1)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                ((5, 3, 1), (3, 1, 4)),
                ((3, 2, 3), (3, 2, 4)),
                ((3, 2), ()),
                ((), (3, 2)),
                ((), ()),
                ((3, 2), (1,)),
                ((0, 2), (3, 0)),
                ((0, 1, 1), (2, 1, 1)),
            ],
        }
    )
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@testing.gpu
class TestMatmulInvalidShape(unittest.TestCase):
    def test_invalid_shape(self):
        for xp in (numpy, dpnp):
            shape1, shape2 = self.shape_pair
            x1 = testing.shaped_arange(shape1, xp, numpy.float32)
            x2 = testing.shaped_arange(shape2, xp, numpy.float32)
            with pytest.raises(ValueError):
                xp.matmul(x1, x2)
