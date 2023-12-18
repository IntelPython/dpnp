import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing
from tests.third_party.cupy.testing import condition


def stacked_identity(xp, batch_shape, n, dtype):
    shape = batch_shape + (n, n)
    idx = xp.arange(n)
    x = xp.zeros(shape, dtype=dtype)
    x[..., idx, idx] = 1
    return x


@testing.parameterize(
    *testing.product(
        {
            "full_matrices": [True, False],
        }
    )
)
# @testing.fix_random()
class TestSVD(unittest.TestCase):
    def setUp(self):
        self.seed = numpy.random.randint(0x7FFFFFFF)

    @testing.for_dtypes(
        [
            numpy.int32,
            numpy.int64,
            numpy.uint32,
            numpy.uint64,
            numpy.float32,
            numpy.float64,
            numpy.complex64,
            numpy.complex128,
        ]
    )
    def check_usv(self, shape, dtype):
        array = testing.shaped_random(shape, numpy, dtype=dtype, seed=self.seed)
        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        result_cpu = numpy.linalg.svd(a_cpu, full_matrices=self.full_matrices)
        result_gpu = cupy.linalg.svd(a_gpu, full_matrices=self.full_matrices)
        # Check if the input matrix is not broken
        testing.assert_allclose(a_gpu, a_cpu)

        assert len(result_gpu) == 3
        for i in range(3):
            assert result_gpu[i].shape == result_cpu[i].shape
            if has_support_aspect64():
                assert result_gpu[i].dtype == result_cpu[i].dtype
            else:
                assert result_gpu[i].dtype.kind == result_cpu[i].dtype.kind
        u_cpu, s_cpu, vh_cpu = result_cpu
        u_gpu, s_gpu, vh_gpu = result_gpu
        testing.assert_allclose(s_gpu, s_cpu, rtol=1e-5, atol=1e-4)

        # reconstruct the matrix
        k = s_cpu.shape[-1]

        # dpnp.dot/matmul does not support complex type and unstable on cpu
        # TODO: remove it when dpnp.dot/matmul is updated
        u_gpu = cupy.asnumpy(u_gpu)
        vh_gpu = cupy.asnumpy(vh_gpu)
        s_gpu = cupy.asnumpy(s_gpu)
        xp = numpy

        if len(shape) == 2:
            if self.full_matrices:
                a_gpu_usv = xp.dot(u_gpu[:, :k] * s_gpu, vh_gpu[:k, :])
            else:
                a_gpu_usv = xp.dot(u_gpu * s_gpu, vh_gpu)
        else:
            if self.full_matrices:
                a_gpu_usv = xp.matmul(
                    u_gpu[..., :k] * s_gpu[..., None, :], vh_gpu[..., :k, :]
                )
            else:
                a_gpu_usv = xp.matmul(u_gpu * s_gpu[..., None, :], vh_gpu)
        testing.assert_allclose(a_gpu, a_gpu_usv, rtol=1e-4, atol=1e-4)

        # assert unitary
        u_len = u_gpu.shape[-1]
        vh_len = vh_gpu.shape[-2]
        testing.assert_allclose(
            xp.matmul(u_gpu.swapaxes(-1, -2).conj(), u_gpu),
            stacked_identity(xp, shape[:-2], u_len, dtype),
            atol=1e-4,
        )
        testing.assert_allclose(
            xp.matmul(vh_gpu, vh_gpu.swapaxes(-1, -2).conj()),
            stacked_identity(xp, shape[:-2], vh_len, dtype),
            atol=1e-4,
        )

    @testing.for_dtypes(
        [
            numpy.int32,
            numpy.int64,
            numpy.uint32,
            numpy.uint64,
            numpy.float32,
            numpy.float64,
            numpy.complex64,
            numpy.complex128,
        ]
    )
    @testing.numpy_cupy_allclose(
        rtol=1e-5, atol=1e-4, type_check=has_support_aspect64()
    )
    def check_singular(self, shape, xp, dtype):
        array = testing.shaped_random(shape, xp, dtype=dtype, seed=self.seed)
        a = xp.asarray(array, dtype=dtype)
        a_copy = a.copy()
        result = xp.linalg.svd(
            a, full_matrices=self.full_matrices, compute_uv=False
        )
        # Check if the input matrix is not broken
        assert (a == a_copy).all()
        return result

    # @condition.repeat(3, 10)
    def test_svd_rank2(self):
        self.check_usv((3, 7))
        self.check_usv((2, 2))
        self.check_usv((7, 3))

    # @condition.repeat(3, 10)
    def test_svd_rank2_no_uv(self):
        self.check_singular((3, 7))
        self.check_singular((2, 2))
        self.check_singular((7, 3))

    @testing.with_requires("numpy>=1.16")
    # dpnp.matmul does not support input empty arrays
    # TODO: remove it when support is added
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_svd_rank2_empty_array(self):
        self.check_usv((0, 3))
        self.check_usv((3, 0))
        self.check_usv((1, 0))

    @testing.with_requires("numpy>=1.16")
    @testing.numpy_cupy_array_equal(type_check=has_support_aspect64())
    def test_svd_rank2_empty_array_compute_uv_false(self, xp):
        array = xp.empty((3, 0))
        return xp.linalg.svd(
            array, full_matrices=self.full_matrices, compute_uv=False
        )

    # @condition.repeat(3, 10)
    def test_svd_rank3(self):
        self.check_usv((2, 3, 4))
        self.check_usv((2, 3, 7))
        self.check_usv((2, 4, 4))
        self.check_usv((2, 7, 3))
        self.check_usv((2, 4, 3))
        self.check_usv((2, 32, 32))

    # @condition.repeat(3, 10)
    def test_svd_rank3_loop(self):
        # This tests the loop-based batched gesvd on CUDA (_gesvd_batched)
        self.check_usv((2, 64, 64))
        self.check_usv((2, 64, 32))
        self.check_usv((2, 32, 64))

    # @condition.repeat(3, 10)
    def test_svd_rank3_no_uv(self):
        self.check_singular((2, 3, 4))
        self.check_singular((2, 3, 7))
        self.check_singular((2, 4, 4))
        self.check_singular((2, 7, 3))
        self.check_singular((2, 4, 3))

    # @condition.repeat(3, 10)
    def test_svd_rank3_no_uv_loop(self):
        # This tests the loop-based batched gesvd on CUDA (_gesvd_batched)
        self.check_singular((2, 64, 64))
        self.check_singular((2, 64, 32))
        self.check_singular((2, 32, 64))

    @testing.with_requires("numpy>=1.16")
    def test_svd_rank3_empty_array(self):
        self.check_usv((0, 3, 4))
        self.check_usv((3, 0, 4))
        self.check_usv((3, 4, 0))
        self.check_usv((3, 0, 0))
        self.check_usv((0, 3, 0))
        self.check_usv((0, 0, 3))

    @testing.with_requires("numpy>=1.16")
    @testing.numpy_cupy_array_equal(type_check=has_support_aspect64())
    def test_svd_rank3_empty_array_compute_uv_false1(self, xp):
        array = xp.empty((3, 0, 4))
        return xp.linalg.svd(
            array, full_matrices=self.full_matrices, compute_uv=False
        )

    @testing.with_requires("numpy>=1.16")
    @testing.numpy_cupy_array_equal(type_check=has_support_aspect64())
    def test_svd_rank3_empty_array_compute_uv_false2(self, xp):
        array = xp.empty((0, 3, 4))
        return xp.linalg.svd(
            array, full_matrices=self.full_matrices, compute_uv=False
        )

    # @condition.repeat(3, 10)
    def test_svd_rank4(self):
        self.check_usv((2, 2, 3, 4))
        self.check_usv((2, 2, 3, 7))
        self.check_usv((2, 2, 4, 4))
        self.check_usv((2, 2, 7, 3))
        self.check_usv((2, 2, 4, 3))
        self.check_usv((2, 2, 32, 32))

    # @condition.repeat(3, 10)
    def test_svd_rank4_loop(self):
        # This tests the loop-based batched gesvd on CUDA (_gesvd_batched)
        self.check_usv((3, 2, 64, 64))
        self.check_usv((3, 2, 64, 32))
        self.check_usv((3, 2, 32, 64))

    # @condition.repeat(3, 10)
    def test_svd_rank4_no_uv(self):
        self.check_singular((2, 2, 3, 4))
        self.check_singular((2, 2, 3, 7))
        self.check_singular((2, 2, 4, 4))
        self.check_singular((2, 2, 7, 3))
        self.check_singular((2, 2, 4, 3))

    # @condition.repeat(3, 10)
    def test_svd_rank4_no_uv_loop(self):
        # This tests the loop-based batched gesvd on CUDA (_gesvd_batched)
        self.check_singular((3, 2, 64, 64))
        self.check_singular((3, 2, 64, 32))
        self.check_singular((3, 2, 32, 64))

    @testing.with_requires("numpy>=1.16")
    def test_svd_rank4_empty_array(self):
        self.check_usv((0, 2, 3, 4))
        self.check_usv((1, 2, 0, 4))
        self.check_usv((1, 2, 3, 0))
