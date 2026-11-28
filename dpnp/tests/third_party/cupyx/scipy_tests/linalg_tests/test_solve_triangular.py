from __future__ import annotations

import unittest

import numpy
import pytest

import dpnp as cupy
import dpnp as cupyx
from dpnp.tests.third_party.cupy import testing

# import cupyx.scipy.linalg

try:
    import scipy.linalg

    _scipy_available = True
except ImportError:
    _scipy_available = False


@testing.parameterize(
    *testing.product(
        {
            "trans": [0, 1, 2, "N", "T", "C"],
            "lower": [True, False],
            "unit_diagonal": [True, False],
            "overwrite_b": [True, False],
            "check_finite": [True, False],
        }
    )
)
@testing.with_requires("scipy")
class TestSolveTriangular(unittest.TestCase):

    @testing.for_dtypes("fdFD")
    def check_x(self, a_shape, b_shape, dtype):
        a_cpu = numpy.random.randint(1, 10, size=a_shape).astype(dtype)
        b_cpu = numpy.random.randint(1, 10, size=b_shape).astype(dtype)
        a_cpu = numpy.tril(a_cpu)

        if self.lower is False:
            a_cpu = a_cpu.T
        if self.unit_diagonal is True:
            numpy.fill_diagonal(a_cpu, 1)

        a_gpu = cupy.asarray(a_cpu)
        b_gpu = cupy.asarray(b_cpu)
        a_gpu_copy = a_gpu.copy()
        b_gpu_copy = b_gpu.copy()
        result_cpu = scipy.linalg.solve_triangular(
            a_cpu,
            b_cpu,
            trans=self.trans,
            lower=self.lower,
            unit_diagonal=self.unit_diagonal,
            overwrite_b=self.overwrite_b,
            check_finite=self.check_finite,
        )
        result_gpu = cupyx.scipy.linalg.solve_triangular(
            a_gpu,
            b_gpu,
            trans=self.trans,
            lower=self.lower,
            unit_diagonal=self.unit_diagonal,
            overwrite_b=self.overwrite_b,
            check_finite=self.check_finite,
        )
        assert result_cpu.dtype == result_gpu.dtype
        testing.assert_allclose(result_cpu, result_gpu, atol=1e-3)
        testing.assert_array_equal(a_gpu_copy, a_gpu)
        if not self.overwrite_b:
            testing.assert_array_equal(b_gpu_copy, b_gpu)

    def test_solve(self):
        self.check_x((4, 4), (4,))
        self.check_x((5, 5), (5, 2))
        self.check_x((5, 5), (5, 5))

    # dpnp supports batches as SciPy does, so the test passes for
    # lower=True, unit_diagonal=False and overwrite_b=False; in the other
    # cases it fails by itself: a 3-D `.T` makes the matrices singular,
    # `fill_diagonal` needs equal dimensions, and SciPy 1.18 solves a 1-D `b`
    # in place for each matrix, so the solutions of the batch are all equal
    # @pytest.mark.xfail(strict=True, reason="not implemented")
    def test_solve_batchwise(self):
        if not self.lower or self.unit_diagonal or self.overwrite_b:
            pytest.xfail("the test fails with SciPy too")
        self.check_x((3, 3, 3), (3,))
        self.check_x((3, 3, 3), (3, 2))
        self.check_x((8, 3, 3), (3, 2))
        self.check_x((3, 3), (8, 3, 2))
        self.check_x((8, 3, 3), (8, 3, 2))
        self.check_x((2, 1, 3, 3), (1, 4, 3, 2))

    def check_shape(self, a_shape, b_shape):
        for xp, sp in ((numpy, scipy), (cupy, cupyx.scipy)):
            a = xp.random.rand(*a_shape)
            b = xp.random.rand(*b_shape)
            with pytest.raises(ValueError):
                sp.linalg.solve_triangular(
                    a,
                    b,
                    trans=self.trans,
                    lower=self.lower,
                    unit_diagonal=self.unit_diagonal,
                    overwrite_b=self.overwrite_b,
                    check_finite=self.check_finite,
                )

    def test_invalid_shape(self):
        self.check_shape((2, 3), (4,))
        self.check_shape((3, 3), (2,))
        self.check_shape((3, 3), (2, 2))
        self.check_shape((3, 3, 4), (3,))
        self.check_shape((3, 3, 3), (2, 3))

    def check_infinite(self, a_shape, b_shape):
        for xp, sp in ((numpy, scipy), (cupy, cupyx.scipy)):
            a = xp.random.rand(*a_shape)
            b = xp.random.rand(*b_shape)
            a[(0,) * a.ndim] = numpy.inf
            b[(0,) * b.ndim] = numpy.inf
            with pytest.raises(ValueError):
                sp.linalg.solve_triangular(
                    a,
                    b,
                    trans=self.trans,
                    lower=self.lower,
                    unit_diagonal=self.unit_diagonal,
                    overwrite_b=self.overwrite_b,
                    check_finite=self.check_finite,
                )

    def test_infinite(self):
        if self.check_finite:
            self.check_infinite((4, 4), (4,))
            self.check_infinite((5, 5), (5, 2))
            self.check_infinite((5, 5), (5, 5))

    @testing.for_dtypes("fdFD")
    def check_batched(self, a_shape, b_shape, dtype):
        a_cpu = numpy.random.randint(1, 10, size=a_shape).astype(dtype)
        b_cpu = numpy.random.randint(1, 10, size=b_shape).astype(dtype)
        a_cpu = numpy.tril(a_cpu)

        if self.lower is False:
            a_dim = tuple(range(len(a_cpu.shape)))
            a_dim = a_dim[:-2] + (a_dim[-1], a_dim[-2])
            a_cpu = numpy.transpose(a_cpu, a_dim)
        if self.unit_diagonal is True:
            for i in range(a_cpu.shape[0]):
                numpy.fill_diagonal(a_cpu[i], 1)

        a_gpu = cupy.asarray(a_cpu)
        b_gpu = cupy.asarray(b_cpu)
        a_gpu_copy = a_gpu.copy()
        b_gpu_copy = b_gpu.copy()

        result_cpu = []
        for i in range(a_cpu.shape[0]):
            result_cpu_single = scipy.linalg.solve_triangular(
                a_cpu[i],
                b_cpu[i],
                trans=self.trans,
                lower=self.lower,
                unit_diagonal=self.unit_diagonal,
                overwrite_b=self.overwrite_b,
                check_finite=self.check_finite,
            )
            result_cpu.append(result_cpu_single)
        result_cpu = numpy.array(result_cpu)
        # result_gpu = cupyx.scipy.linalg.solve_triangular(
        #     a_gpu,
        #     b_gpu,
        #     trans=self.trans,
        #     lower=self.lower,
        #     unit_diagonal=self.unit_diagonal,
        #     overwrite_b=self.overwrite_b,
        #     check_finite=self.check_finite,
        # )
        # unlike CuPy, dpnp follows SciPy, where a (..., M) `b` is a matrix,
        # not a batch of vectors: pass the vectors as (..., M, 1)
        is_vec = b_gpu.ndim == a_gpu.ndim - 1
        result_gpu = cupyx.scipy.linalg.solve_triangular(
            a_gpu,
            b_gpu[..., None] if is_vec else b_gpu,
            trans=self.trans,
            lower=self.lower,
            unit_diagonal=self.unit_diagonal,
            overwrite_b=self.overwrite_b,
            check_finite=self.check_finite,
        )
        if is_vec:
            result_gpu = result_gpu[..., 0]
        assert result_cpu.dtype == result_gpu.dtype
        testing.assert_allclose(result_cpu, result_gpu, atol=1e-3)
        testing.assert_array_equal(a_gpu_copy, a_gpu)
        if not self.overwrite_b:
            testing.assert_array_equal(b_gpu_copy, b_gpu)

    def test_solve_batched(self):
        self.check_batched((4, 4, 4), (4, 4))
        self.check_batched((5, 5, 5), (5, 5, 2))
        self.check_batched((5, 5, 5), (5, 5, 5))
