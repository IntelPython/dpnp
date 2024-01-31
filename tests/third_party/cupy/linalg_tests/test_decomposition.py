import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import has_support_aspect64, is_cpu_device
from tests.third_party.cupy import testing
from tests.third_party.cupy.testing import _condition


def random_matrix(shape, dtype, scale, sym=False):
    m, n = shape[-2:]
    dtype = numpy.dtype(dtype)
    assert dtype.kind in "iufc"
    low_s, high_s = scale
    bias = None
    if dtype.kind in "iu":
        # For an m \times n matrix M whose element is in [-0.5, 0.5], it holds
        # (singular value of M) <= \sqrt{mn} / 2
        err = numpy.sqrt(m * n) / 2.0
        low_s += err
        high_s -= err
        if dtype.kind in "u":
            assert sym, (
                "generating nonsymmetric matrix with uint cells is not"
                " supported."
            )
            # (singular value of numpy.ones((m, n))) <= \sqrt{mn}
            high_s = bias = high_s / (1 + numpy.sqrt(m * n))
    assert low_s <= high_s
    a = numpy.random.standard_normal(shape)
    if dtype.kind == "c":
        a = a + 1j * numpy.random.standard_normal(shape)
    u, s, vh = numpy.linalg.svd(a)
    if sym:
        assert m == n
        vh = u.conj().swapaxes(-1, -2)
    new_s = numpy.random.uniform(low_s, high_s, s.shape)
    new_a = numpy.einsum("...ij,...j,...jk->...ik", u, new_s, vh)
    if bias is not None:
        new_a += bias
    if dtype.kind in "iu":
        new_a = numpy.rint(new_a)
    return new_a.astype(dtype)


class TestCholeskyDecomposition:
    @testing.numpy_cupy_allclose(atol=1e-3, type_check=has_support_aspect64())
    def check_L(self, array, xp):
        a = xp.asarray(array)
        return xp.linalg.cholesky(a)

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
    def test_decomposition(self, dtype):
        # A positive definite matrix
        A = random_matrix((5, 5), dtype, scale=(10, 10000), sym=True)
        self.check_L(A)
        # np.linalg.cholesky only uses a lower triangle of an array
        self.check_L(numpy.array([[1, 2], [1, 9]], dtype))

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
    def test_batched_decomposition(self, dtype):
        Ab1 = random_matrix((3, 5, 5), dtype, scale=(10, 10000), sym=True)
        self.check_L(Ab1)
        Ab2 = random_matrix((2, 2, 5, 5), dtype, scale=(10, 10000), sym=True)
        self.check_L(Ab2)

    @pytest.mark.parametrize(
        "shape",
        [
            # empty square
            (0, 0),
            (3, 0, 0),
            # empty batch
            (2, 0, 3, 4, 4),
        ],
    )
    @testing.for_dtypes(
        [
            numpy.int32,
            numpy.uint16,
            numpy.float32,
            numpy.float64,
            numpy.complex64,
            numpy.complex128,
        ]
    )
    @testing.numpy_cupy_allclose(type_check=has_support_aspect64())
    def test_empty(self, shape, xp, dtype):
        a = xp.empty(shape, dtype=dtype)
        return xp.linalg.cholesky(a)


class TestCholeskyInvalid(unittest.TestCase):
    def check_L(self, array):
        for xp in (numpy, cupy):
            a = xp.asarray(array)
            with pytest.raises(xp.linalg.LinAlgError):
                xp.linalg.cholesky(a)

    # TODO: remove skipif when MKLD-16626 is resolved
    @pytest.mark.skipif(is_cpu_device(), reason="MKLD-16626")
    @testing.for_dtypes(
        [
            numpy.int32,
            numpy.int64,
            numpy.uint32,
            numpy.uint64,
            numpy.float32,
            numpy.float64,
        ]
    )
    def test_decomposition(self, dtype):
        A = numpy.array([[1, -2], [-2, 1]]).astype(dtype)
        self.check_L(A)


@testing.parameterize(
    *testing.product(
        {
            "mode": ["r", "raw", "complete", "reduced"],
        }
    )
)
class TestQRDecomposition(unittest.TestCase):
    @testing.for_dtypes("fdFD")
    def check_mode(self, array, mode, dtype):
        if dtype in (numpy.complex64, numpy.complex128):
            pytest.skip("ungqr unsupported")

        a_cpu = numpy.asarray(array, dtype=dtype)
        a_gpu = cupy.asarray(array, dtype=dtype)
        result_gpu = cupy.linalg.qr(a_gpu, mode=mode)
        if (
            mode != "raw"
            or numpy.lib.NumpyVersion(numpy.__version__) >= "1.22.0rc1"
        ):
            result_cpu = numpy.linalg.qr(a_cpu, mode=mode)
            self._check_result(result_cpu, result_gpu)

    def _check_result(self, result_cpu, result_gpu):
        if isinstance(result_cpu, tuple):
            for b_cpu, b_gpu in zip(result_cpu, result_gpu):
                assert b_cpu.dtype == b_gpu.dtype
                testing.assert_allclose(b_cpu, b_gpu, atol=1e-4)
        else:
            assert result_cpu.dtype == result_gpu.dtype
            testing.assert_allclose(result_cpu, result_gpu, atol=1e-4)

    @testing.fix_random()
    @_condition.repeat(3, 10)
    def test_mode(self):
        self.check_mode(numpy.random.randn(2, 4), mode=self.mode)
        self.check_mode(numpy.random.randn(3, 3), mode=self.mode)
        self.check_mode(numpy.random.randn(5, 4), mode=self.mode)

    @testing.with_requires("numpy>=1.22")
    @testing.fix_random()
    def test_mode_rank3(self):
        self.check_mode(numpy.random.randn(3, 2, 4), mode=self.mode)
        self.check_mode(numpy.random.randn(4, 3, 3), mode=self.mode)
        self.check_mode(numpy.random.randn(2, 5, 4), mode=self.mode)

    @testing.with_requires("numpy>=1.22")
    @testing.fix_random()
    def test_mode_rank4(self):
        self.check_mode(numpy.random.randn(2, 3, 2, 4), mode=self.mode)
        self.check_mode(numpy.random.randn(2, 4, 3, 3), mode=self.mode)
        self.check_mode(numpy.random.randn(2, 2, 5, 4), mode=self.mode)

    @testing.with_requires("numpy>=1.16")
    def test_empty_array(self):
        self.check_mode(numpy.empty((0, 3)), mode=self.mode)
        self.check_mode(numpy.empty((3, 0)), mode=self.mode)

    @testing.with_requires("numpy>=1.22")
    def test_empty_array_rank3(self):
        self.check_mode(numpy.empty((0, 3, 2)), mode=self.mode)
        self.check_mode(numpy.empty((3, 0, 2)), mode=self.mode)
        self.check_mode(numpy.empty((3, 2, 0)), mode=self.mode)
        self.check_mode(numpy.empty((0, 3, 3)), mode=self.mode)
        self.check_mode(numpy.empty((3, 0, 3)), mode=self.mode)
        self.check_mode(numpy.empty((3, 3, 0)), mode=self.mode)
        self.check_mode(numpy.empty((0, 2, 3)), mode=self.mode)
        self.check_mode(numpy.empty((2, 0, 3)), mode=self.mode)
        self.check_mode(numpy.empty((2, 3, 0)), mode=self.mode)
