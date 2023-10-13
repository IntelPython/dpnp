import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing


@testing.parameterize(
    *testing.product(
        {
            "order": ["C", "F"],
        }
    )
)
class TestSolve(unittest.TestCase):
    # TODO: add get_batched_gesv_limit
    # def setUp(self):
    #     if self.batched_gesv_limit is not None:
    #         self.old_limit = get_batched_gesv_limit()
    #         set_batched_gesv_limit(self.batched_gesv_limit)

    # def tearDown(self):
    #     if self.batched_gesv_limit is not None:
    #         set_batched_gesv_limit(self.old_limit)

    @testing.for_dtypes("ifdFD")
    @testing.numpy_cupy_allclose(
        atol=1e-3, contiguous_check=False, type_check=has_support_aspect64()
    )
    def check_x(self, a_shape, b_shape, xp, dtype):
        a = testing.shaped_random(a_shape, xp, dtype=dtype, seed=0, scale=20)
        b = testing.shaped_random(b_shape, xp, dtype=dtype, seed=1)
        a = a.copy(order=self.order)
        b = b.copy(order=self.order)
        a_copy = a.copy()
        b_copy = b.copy()
        result = xp.linalg.solve(a, b)
        numpy.testing.assert_array_equal(a_copy, a)
        numpy.testing.assert_array_equal(b_copy, b)
        return result

    def test_solve(self):
        self.check_x((4, 4), (4,))
        self.check_x((5, 5), (5, 2))
        self.check_x((2, 4, 4), (2, 4))
        self.check_x((2, 5, 5), (2, 5, 2))
        self.check_x((2, 3, 2, 2), (2, 3, 2))
        self.check_x((2, 3, 3, 3), (2, 3, 3, 2))
        self.check_x((0, 0), (0,))
        self.check_x((0, 0), (0, 2))
        self.check_x((0, 2, 2), (0, 2))
        self.check_x((0, 2, 2), (0, 2, 3))

    def check_shape(self, a_shape, b_shape, error_types):
        for xp, error_type in error_types.items():
            a = xp.random.rand(*a_shape)
            b = xp.random.rand(*b_shape)
            with pytest.raises(error_type):
                xp.linalg.solve(a, b)

    def test_solve_singular_empty(self):
        for xp in (numpy, cupy):
            a = xp.zeros((3, 3))  # singular
            b = xp.empty((3, 0))  # nrhs = 0
            # numpy <= 1.24.* raises LinAlgError when b.size == 0
            # numpy >= 1.25 returns an empty array
            if xp == numpy and testing.numpy_satisfies("<1.25.0"):
                with pytest.raises(numpy.linalg.LinAlgError):
                    xp.linalg.solve(a, b)
            else:
                result = xp.linalg.solve(a, b)
                assert result.size == 0

    def test_invalid_shape(self):
        linalg_errors = {
            numpy: numpy.linalg.LinAlgError,
            cupy: cupy.linalg.LinAlgError,
        }
        value_errors = {
            numpy: ValueError,
            cupy: ValueError,
        }

        self.check_shape((2, 3), (4,), linalg_errors)
        self.check_shape((3, 3), (2,), value_errors)
        self.check_shape((3, 3), (2, 2), value_errors)
        self.check_shape((3, 3, 4), (3,), linalg_errors)
        self.check_shape((2, 3, 3), (3,), value_errors)
        self.check_shape((3, 3), (0,), value_errors)
        self.check_shape((0, 3, 4), (3,), linalg_errors)
