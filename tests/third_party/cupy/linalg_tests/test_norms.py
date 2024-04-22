import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import is_cpu_device
from tests.third_party.cupy import testing


@testing.parameterize(
    *testing.product(
        {
            "shape": [(1,), (2,)],
            "ord": [-numpy.inf, -2, -1, 0, 1, 2, 3, numpy.inf],
            "axis": [0, None],
            "keepdims": [True, False],
        }
    )
    + testing.product(
        {
            "shape": [(1, 2), (2, 2)],
            "ord": [-numpy.inf, -2, -1, 1, 2, numpy.inf, "fro", "nuc"],
            "axis": [(0, 1), None],
            "keepdims": [True, False],
        }
    )
    + testing.product(
        {
            "shape": [(2, 2, 2)],
            "ord": [-numpy.inf, -2, -1, 0, 1, 2, 3, numpy.inf],
            "axis": [0, 1, 2],
            "keepdims": [True, False],
        }
    )
    + testing.product(
        {
            "shape": [(2, 2, 2)],
            "ord": [-numpy.inf, -1, 1, numpy.inf, "fro"],
            "axis": [(0, 1), (0, 2), (1, 2)],
            "keepdims": [True, False],
        }
    )
)
class TestNorm(unittest.TestCase):
    @testing.for_all_dtypes(no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-4, type_check=False)
    # since dtype of sum is different in dpnp and NumPy, type_check=False
    def test_norm(self, xp, dtype):
        a = testing.shaped_arange(self.shape, xp, dtype)
        res = xp.linalg.norm(a, self.ord, self.axis, self.keepdims)
        if xp == numpy and not isinstance(res, numpy.ndarray):
            real_dtype = a.real.dtype
            if issubclass(real_dtype.type, numpy.inexact):
                # Avoid numpy bug. See numpy/numpy#10667
                res = res.astype(a.real.dtype)
        return res


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

    # TODO: remove skipif when MKLD-13852 is resolved
    # _getrf_batch does not raise an error with singular matrices.
    # Skip running on cpu because dpnp uses _getrf_batch only on cpu.
    @pytest.mark.skipif(is_cpu_device(), reason="MKLD-13852")
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
