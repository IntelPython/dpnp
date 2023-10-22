import unittest

import numpy
import pytest

import dpnp as cupy

# from cupy.core import _accelerator
from tests.third_party.cupy import testing


@testing.gpu
class TestArrayReduction(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.max()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_all_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.max(keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.max(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.max(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.max(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.max(axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_multiple_axes(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.max(axis=(1, 2))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_multiple_axes_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.max(axis=(1, 2), keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_nan(self, xp, dtype):
        a = xp.array([float("nan"), 1, -1], dtype)
        return a.max()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_nan_real(self, xp, dtype):
        a = xp.array([float("nan"), 1, -1], dtype)
        return a.max()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_max_nan_imag(self, xp, dtype):
        a = xp.array([float("nan") * 1.0j, 1.0j, -1.0j], dtype)
        return a.max()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.min()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_all_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return a.min(keepdims=True)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return a.min(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.min(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.min(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.min(axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_multiple_axes(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.min(axis=(1, 2))

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_multiple_axes_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return a.min(axis=(1, 2), keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_nan(self, xp, dtype):
        a = xp.array([float("nan"), 1, -1], dtype)
        return a.min()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_nan_real(self, xp, dtype):
        a = xp.array([float("nan"), 1, -1], dtype)
        return a.min()

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_min_nan_imag(self, xp, dtype):
        a = xp.array([float("nan") * 1.0j, 1.0j, -1.0j], dtype)
        return a.min()

    # skip bool: numpy's ptp raises a TypeError on bool inputs
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_all(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.ptp(a)

    @testing.with_requires("numpy>=1.15")
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_all_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3), xp, dtype)
        return xp.ptp(a, keepdims=True)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_axis_large(self, xp, dtype):
        a = testing.shaped_random((3, 1000), xp, dtype)
        return xp.ptp(a, axis=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_axis0(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.ptp(a, axis=0)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_axis1(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.ptp(a, axis=1)

    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_axis2(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.ptp(a, axis=2)

    @testing.with_requires("numpy>=1.15")
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_multiple_axes(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.ptp(a, axis=(1, 2))

    @testing.with_requires("numpy>=1.15")
    @testing.for_all_dtypes(no_bool=True)
    @testing.numpy_cupy_allclose()
    def test_ptp_multiple_axes_keepdims(self, xp, dtype):
        a = testing.shaped_random((2, 3, 4), xp, dtype)
        return xp.ptp(a, axis=(1, 2), keepdims=True)

    @testing.for_float_dtypes()
    @testing.numpy_cupy_allclose()
    def test_ptp_nan(self, xp, dtype):
        a = xp.array([float("nan"), 1, -1], dtype)
        return xp.ptp(a)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_ptp_nan_real(self, xp, dtype):
        a = xp.array([float("nan"), 1, -1], dtype)
        return xp.ptp(a)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose()
    def test_ptp_nan_imag(self, xp, dtype):
        a = xp.array([float("nan") * 1.0j, 1.0j, -1.0j], dtype)
        return xp.ptp(a)


@testing.parameterize(
    *testing.product(
        {
            # TODO(leofang): make a @testing.for_all_axes decorator
            "shape_and_axis": [
                ((), None),
                ((0,), (0,)),
                ((0, 2), (0,)),
                ((0, 2), (1,)),
                ((0, 2), (0, 1)),
                ((2, 0), (0,)),
                ((2, 0), (1,)),
                ((2, 0), (0, 1)),
                ((0, 2, 3), (0,)),
                ((0, 2, 3), (1,)),
                ((0, 2, 3), (2,)),
                ((0, 2, 3), (0, 1)),
                ((0, 2, 3), (1, 2)),
                ((0, 2, 3), (0, 2)),
                ((0, 2, 3), (0, 1, 2)),
                ((2, 0, 3), (0,)),
                ((2, 0, 3), (1,)),
                ((2, 0, 3), (2,)),
                ((2, 0, 3), (0, 1)),
                ((2, 0, 3), (1, 2)),
                ((2, 0, 3), (0, 2)),
                ((2, 0, 3), (0, 1, 2)),
                ((2, 3, 0), (0,)),
                ((2, 3, 0), (1,)),
                ((2, 3, 0), (2,)),
                ((2, 3, 0), (0, 1)),
                ((2, 3, 0), (1, 2)),
                ((2, 3, 0), (0, 2)),
                ((2, 3, 0), (0, 1, 2)),
            ],
            "order": ("C", "F"),
            "func": ("min", "max"),
        }
    )
)
class TestArrayReductionZeroSize:
    @testing.numpy_cupy_allclose(
        contiguous_check=False, accept_error=ValueError
    )
    def test_zero_size(self, xp):
        shape, axis = self.shape_and_axis
        # NumPy only supports axis being an int
        if self.func in ("argmax", "argmin"):
            if axis is not None and len(axis) == 1:
                axis = axis[0]
            else:
                pytest.skip(
                    f"NumPy does not support axis={axis} for {self.func}"
                )
        # dtype is irrelevant here, just pick one
        a = testing.shaped_random(shape, xp, xp.float32, order=self.order)
        return getattr(a, self.func)(axis=axis)


# This class compares CUB results against NumPy's
@testing.parameterize(
    *testing.product(
        {
            "shape": [(10,), (10, 20), (10, 20, 30), (10, 20, 30, 40)],
            "order": ("C", "F"),
        }
    )
)
@testing.gpu
# @unittest.skipUnless(cupy.cuda.cub.available, 'The CUB routine is not enabled')
class TestCubReduction(unittest.TestCase):
    # def setUp(self):
    #     self.old_accelerators = _accelerator.get_routine_accelerators()
    #     _accelerator.set_routine_accelerators(['cub'])
    #
    # def tearDown(self):
    #     _accelerator.set_routine_accelerators(self.old_accelerators)

    # @testing.for_contiguous_axes()
    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_cub_min(self, xp, dtype, axis):
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.order in ("c", "C"):
            a = xp.ascontiguousarray(a)
        elif self.order in ("f", "F"):
            a = xp.asfortranarray(a)

        if xp is numpy:
            return a.min(axis=axis)

        # xp is cupy, first ensure we really use CUB
        ret = cupy.empty(())  # Cython checks return type, need to fool it
        if len(axis) == len(self.shape):
            func = "cupy.core._routines_statistics.cub.device_reduce"
        else:
            func = "cupy.core._routines_statistics.cub.device_segmented_reduce"
        with testing.AssertFunctionIsCalled(func, return_value=ret):
            a.min(axis=axis)
        # ...then perform the actual computation
        return a.min(axis=axis)

    # @testing.for_contiguous_axes()
    @testing.for_all_dtypes(no_bool=True, no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-5)
    def test_cub_max(self, xp, dtype, axis):
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.order in ("c", "C"):
            a = xp.ascontiguousarray(a)
        elif self.order in ("f", "F"):
            a = xp.asfortranarray(a)

        if xp is numpy:
            return a.max(axis=axis)

        # xp is cupy, first ensure we really use CUB
        ret = cupy.empty(())  # Cython checks return type, need to fool it
        if len(axis) == len(self.shape):
            func = "cupy.core._routines_statistics.cub.device_reduce"
        else:
            func = "cupy.core._routines_statistics.cub.device_segmented_reduce"
        with testing.AssertFunctionIsCalled(func, return_value=ret):
            a.max(axis=axis)
        # ...then perform the actual computation
        return a.max(axis=axis)
