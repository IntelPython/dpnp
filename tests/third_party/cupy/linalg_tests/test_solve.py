import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import (
    assert_dtype_allclose,
    has_support_aspect64,
    is_cpu_device,
)
from tests.third_party.cupy import testing
from tests.third_party.cupy.testing import condition


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
        testing.assert_array_equal(a_copy, a)
        testing.assert_array_equal(b_copy, b)
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

    # Undefined behavior is implementation-dependent:
    # Numpy with OpenBLAS returns an empty array
    # while numpy with OneMKL raises LinAlgError
    @pytest.mark.skip("Undefined behavior")
    def test_solve_singular_empty(self, xp):
        a = xp.zeros((3, 3))  # singular
        b = xp.empty((3, 0))  # nrhs = 0
        # LinAlgError("Singular matrix") is not raised
        return xp.linalg.solve(a, b)

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


@testing.parameterize(
    *testing.product(
        {
            "order": ["C", "F"],
        }
    )
)
class TestInv(unittest.TestCase):
    @testing.for_dtypes("ifdFD")
    @condition.retry(10)
    def check_x(self, a_shape, dtype):
        a_cpu = numpy.random.randint(0, 10, size=a_shape)
        a_cpu = a_cpu.astype(dtype, order=self.order)
        a_gpu = cupy.asarray(a_cpu, order=self.order)
        a_gpu_copy = a_gpu.copy()
        result_cpu = numpy.linalg.inv(a_cpu)
        result_gpu = cupy.linalg.inv(a_gpu)

        assert_dtype_allclose(result_gpu, result_cpu)
        testing.assert_array_equal(a_gpu_copy, a_gpu)

    def check_shape(self, a_shape):
        a = cupy.random.rand(*a_shape)
        with self.assertRaises(
            (numpy.linalg.LinAlgError, cupy.linalg.LinAlgError)
        ):
            cupy.linalg.inv(a)

    def test_inv(self):
        self.check_x((3, 3))
        self.check_x((4, 4))
        self.check_x((5, 5))
        self.check_x((2, 5, 5))
        self.check_x((3, 4, 4))
        self.check_x((4, 2, 3, 3))
        self.check_x((0, 0))
        self.check_x((3, 0, 0))
        self.check_x((2, 0, 3, 4, 4))

    def test_invalid_shape(self):
        self.check_shape((2, 3))
        self.check_shape((4, 1))
        self.check_shape((4, 3, 2))
        self.check_shape((2, 4, 3))
        self.check_shape((2, 0))
        self.check_shape((0, 2, 3))


class TestInvInvalid(unittest.TestCase):
    # TODO: remove skipif when MKLD-16626 is resolved
    # _gesv does not raise an error with singular matrices on CPU.
    @pytest.mark.skipif(is_cpu_device(), reason="MKLD-16626")
    @testing.for_dtypes("ifdFD")
    def test_inv(self, dtype):
        for xp in (numpy, cupy):
            a = xp.array([[1, 2], [2, 4]]).astype(dtype)
            with pytest.raises(
                (numpy.linalg.LinAlgError, cupy.linalg.LinAlgError)
            ):
                xp.linalg.inv(a)

    # TODO: remove skipif when MKLD-16626 is resolved
    # _getrf_batch does not raise an error with singular matrices.
    @pytest.mark.skip("MKLD-16626")
    @testing.for_dtypes("ifdFD")
    def test_batched_inv(self, dtype):
        for xp in (numpy, cupy):
            a = xp.array([[[1, 2], [2, 4]]]).astype(dtype)
            assert a.ndim >= 3  # CuPy internally uses a batched function.
            with pytest.raises(xp.linalg.LinAlgError):
                xp.linalg.inv(a)
