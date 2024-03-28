import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import is_cpu_device
from tests.third_party.cupy import testing


@testing.parameterize(
    *testing.product(
        {
            "array": [
                [[1, 2], [3, 4]],
                [[1, 2], [1, 2]],
                [[0, 0], [0, 0]],
                [1, 2],
                [0, 1],
                [0, 0],
            ],
            "tol": [None, 1],
        }
    )
)
class TestMatrixRank(unittest.TestCase):
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_array_equal(type_check=True)
    def test_matrix_rank(self, xp, dtype):
        a = xp.array(self.array, dtype=dtype)
        y = xp.linalg.matrix_rank(a, tol=self.tol)
        if xp is cupy:
            assert isinstance(y, cupy.ndarray)
            assert y.shape == ()
        else:
            # Note numpy returns numpy scalar or python int
            y = xp.array(y)
        return y


# TODO: Remove the use of fixture for all tests in this file
# when dpnp.prod() will support complex dtypes on Gen9
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDet(unittest.TestCase):
    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det(self, xp, dtype):
        a = testing.shaped_arange((2, 2), xp, dtype) + 1
        return xp.linalg.det(a)

    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_3(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2), xp, dtype) + 1
        return xp.linalg.det(a)

    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_4(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2, 2), xp, dtype) + 1
        return xp.linalg.det(a)

    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_empty_batch(self, xp, dtype):
        a = xp.empty((2, 0, 3, 3), dtype=dtype)
        return xp.linalg.det(a)

    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_empty_matrix(self, xp, dtype):
        a = xp.empty((0, 0), dtype=dtype)
        return xp.linalg.det(a)

    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_empty_matrices(self, xp, dtype):
        a = xp.empty((2, 3, 0, 0), dtype=dtype)
        return xp.linalg.det(a)

    @testing.for_dtypes("fdFD")
    def test_det_different_last_two_dims(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2, 3, 2), xp, dtype)
            with pytest.raises(xp.linalg.LinAlgError):
                xp.linalg.det(a)

    @testing.for_dtypes("fdFD")
    def test_det_different_last_two_dims_empty_batch(self, dtype):
        for xp in (numpy, cupy):
            a = xp.empty((0, 3, 2), dtype=dtype)
            with pytest.raises(xp.linalg.LinAlgError):
                xp.linalg.det(a)

    @testing.for_dtypes("fdFD")
    def test_det_one_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2,), xp, dtype)
            with pytest.raises(xp.linalg.LinAlgError):
                xp.linalg.det(a)

    @testing.for_dtypes("fdFD")
    def test_det_zero_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((), xp, dtype)
            with pytest.raises(xp.linalg.LinAlgError):
                xp.linalg.det(a)

    # TODO: remove skipif when MKLD-16626 is resolved
    # _getrf_batch does not raise an error with singular matrices.
    # Skip running on cpu because dpnp uses _getrf_batch only on cpu.
    @pytest.mark.skipif(is_cpu_device(), reason="MKLD-16626")
    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_det_singular(self, xp, dtype):
        a = xp.zeros((2, 3, 3), dtype=dtype)
        return xp.linalg.det(a)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestSlogdet(unittest.TestCase):
    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet(self, xp, dtype):
        a = testing.shaped_arange((2, 2), xp, dtype) + 1
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_3(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2), xp, dtype) + 1
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_4(self, xp, dtype):
        a = testing.shaped_arange((2, 2, 2, 2), xp, dtype) + 1
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_singular(self, xp, dtype):
        a = xp.zeros((3, 3), dtype=dtype)
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes("fdFD")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4)
    def test_slogdet_singular_errstate(self, xp, dtype):
        a = xp.zeros((3, 3), dtype=dtype)
        # TODO: dpnp has no errstate. Probably to be implemented later
        # with cupyx.errstate(linalg="raise"):
        # `cupy.linalg.slogdet` internally catches `dev_info < 0` from
        # cuSOLVER, which should not affect `dev_info > 0` cases.
        sign, logdet = xp.linalg.slogdet(a)
        return sign, logdet

    @testing.for_dtypes("fdFD")
    def test_slogdet_one_dim(self, dtype):
        for xp in (numpy, cupy):
            a = testing.shaped_arange((2,), xp, dtype)
            with pytest.raises(xp.linalg.LinAlgError):
                xp.linalg.slogdet(a)
