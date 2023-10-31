import unittest

import numpy
import pytest

import dpnp as cupy
from tests.third_party.cupy import testing


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
