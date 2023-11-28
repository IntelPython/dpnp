import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import is_cpu_device
from tests.third_party.cupy import testing


class TestDet(unittest.TestCase):
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det(self, xp, dtype):
        a = testing.shaped_arange((2, 2), xp, dtype) + 1
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_3(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2), xp, dtype) + 1
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_4(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2, 2), xp, dtype) + 1
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_empty_batch(self, xp, dtype):
        a = xp.empty((2, 0, 3, 3), dtype=dtype)
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_empty_matrix(self, xp, dtype):
        a = xp.empty((0, 0), dtype=dtype)
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_empty_matrices(self, xp, dtype):
        a = xp.empty((2, 3, 0, 0), dtype=dtype)
        return xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    def test_det_different_last_two_dims(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 2), xp, dtype)
            # TODO: replace ValueError with dpnp.linalg.LinAlgError
            with pytest.raises((numpy.linalg.LinAlgError, ValueError)):
                xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    def test_det_different_last_two_dims_empty_batch(self, dtype):
        for xp in (numpy, cupy):
            a = xp.empty((0, 3, 2), dtype=dtype)
            # TODO: replace ValueError with dpnp.linalg.LinAlgError
            with pytest.raises((numpy.linalg.LinAlgError, ValueError)):
                xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    def test_det_one_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2,), xp, dtype)
            # TODO: replace ValueError with dpnp.linalg.LinAlgError
            with pytest.raises((numpy.linalg.LinAlgError, ValueError)):
                xp.linalg.det(a)

    @testing.for_float_dtypes(no_float16=True)
    def test_det_zero_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp, dtype)
            # TODO: replace ValueError with dpnp.linalg.LinAlgError
            with pytest.raises((numpy.linalg.LinAlgError, ValueError)):
                xp.linalg.det(a)

    # TODO: remove skipif when MKLD-16626 is resolved
    @pytest.mark.skipif(is_cpu_device(), reason="MKL bug MKLD-16626")
    @testing.for_float_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_singular(self, xp, dtype):
        a = xp.zeros((2, 3, 3), dtype=dtype)
        return xp.linalg.det(a)


class TestSlogdet(unittest.TestCase):
    @testing.for_dtypes("fd")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet(self, xp, dtype):
        a = testing.shaped_arange((2, 2), xp, dtype) + 1
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes("fd")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_3(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2), xp, dtype) + 1
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes("fd")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_4(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2, 2), xp, dtype) + 1
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    # TODO: remove skipif when MKLD-16626 is resolved
    @pytest.mark.skipif(is_cpu_device(), reason="MKL bug MKLD-16626")
    @testing.for_dtypes("fd")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_singular(self, xp, dtype):
        a = xp.zeros((3, 3), dtype=dtype)
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    # @testing.for_dtypes("fd")
    # @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    # def test_slogdet_singular_errstate(self, xp, dtype):
    #     a = xp.zeros((3, 3), dtype)
    #     with cupyx.errstate(linalg="raise"):
    #         # `cupy.linalg.slogdet` internally catches `dev_info < 0` from
    #         # cuSOLVER, which should not affect `dev_info > 0` cases.
    #         sign, logdet = xp.linalg.slogdet(a)
    #     return sign, logdet

    @testing.for_dtypes("fd")
    def test_slogdet_one_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2,), xp, dtype)
            if xp is numpy:
                with pytest.raises(numpy.linalg.LinAlgError):
                    xp.linalg.slogdet(a)
            else:
                # Replace with dpnp.linalg.LinAlgError
                with pytest.raises(ValueError):
                    xp.linalg.slogdet(a)
