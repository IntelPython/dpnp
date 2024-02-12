import operator
import unittest

import numpy
import pytest

import dpnp as cupy
from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing


@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                # dot test
                ((3, 2), (2, 4)),
                ((3, 0), (0, 4)),
                ((0, 2), (2, 4)),
                ((3, 2), (2, 0)),
                ((2,), (2, 4)),
                ((0,), (0, 4)),
                ((3, 2), (2,)),
                ((3, 0), (0,)),
                ((2,), (2,)),
                ((0,), (0,)),
                # matmul test
                ((5, 3, 2), (5, 2, 4)),
                ((0, 3, 2), (0, 2, 4)),
                ((5, 3, 2), (2, 4)),
                ((0, 3, 2), (2, 4)),
                ((3, 2), (5, 2, 4)),
                ((3, 2), (0, 2, 4)),
                ((5, 3, 2), (1, 2, 4)),
                ((0, 3, 2), (1, 2, 4)),
                ((1, 3, 2), (5, 2, 4)),
                ((1, 3, 2), (0, 2, 4)),
                ((5, 3, 2), (2,)),
                ((5, 3, 0), (0,)),
                ((2,), (5, 2, 4)),
                ((0,), (5, 0, 4)),
                ((2, 2, 3, 2), (2, 2, 2, 4)),
                ((5, 0, 3, 2), (5, 0, 2, 4)),
                ((6, 5, 3, 2), (2, 4)),
                ((5, 0, 3, 2), (2, 4)),
                ((3, 2), (6, 5, 2, 4)),
                ((3, 2), (5, 0, 2, 4)),
                ((1, 5, 3, 2), (6, 1, 2, 4)),
                ((1, 0, 3, 2), (6, 1, 2, 4)),
                ((6, 1, 3, 2), (1, 5, 2, 4)),
                ((6, 1, 3, 2), (1, 0, 2, 4)),
                ((6, 5, 3, 2), (2,)),
                ((6, 5, 3, 0), (0,)),
                ((2,), (6, 5, 2, 4)),
                ((0,), (6, 5, 0, 4)),
                ((1, 3, 3), (10, 1, 3, 1)),
            ],
        }
    )
)
class TestMatmul(unittest.TestCase):
    @testing.for_all_dtypes(name="dtype1")
    @testing.for_all_dtypes(name="dtype2")
    @testing.numpy_cupy_allclose(
        rtol=1e-3, atol=1e-3, type_check=has_support_aspect64()
    )  # required for uint8
    def test_operator_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name="dtype1")
    @testing.for_all_dtypes(name="dtype2")
    @testing.numpy_cupy_allclose(
        rtol=1e-3, atol=1e-3, type_check=has_support_aspect64()
    )  # required for uint8
    def test_cupy_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                # dot test
                ((2, 3), (3, 4), (2, 4)),
                ((0,), (0,), (0,)),
                # matmul test
                ((5, 3, 2), (5, 2, 4), (5, 3, 4)),
                ((0, 3, 2), (0, 2, 4), (0, 3, 4)),
            ],
        }
    )
)
class TestMatmulOut(unittest.TestCase):
    @testing.for_all_dtypes(name="dtype1")
    @testing.for_all_dtypes(name="dtype2")
    @testing.numpy_cupy_allclose(
        rtol=1e-3, atol=1e-3, accept_error=TypeError  # required for uint8
    )
    def test_cupy_matmul_noncontiguous(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        out = xp.zeros(self.shape_pair[2], dtype=dtype1)[::-1]
        ret = xp.matmul(x1, x2, out=out)
        assert ret is out
        return ret

    @testing.for_all_dtypes(name="dtype1")
    @testing.for_all_dtypes(name="dtype2")
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul_out_cast(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        out = xp.zeros(self.shape_pair[2], dtype=bool)
        ret = xp.matmul(x1, x2, out=out, casting="unsafe")
        assert ret is out
        return ret


class TestMatmulOutOverlap:
    @pytest.mark.parametrize(
        "shape",
        [
            (900, 900),
            (2, 600, 600),
        ],
    )
    @testing.for_dtypes([numpy.int32, numpy.float64])
    @testing.numpy_cupy_allclose(rtol=1e-5, atol=1e-5)
    def test_overlap_both(self, xp, dtype, shape):
        a = xp.ones(shape, dtype=dtype)
        return xp.matmul(a, a, out=a)


class TestMatmulStrides:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_relaxed_c_contiguous_input(self, xp, dtype):
        x1 = testing.shaped_arange((2, 2, 3), xp, dtype)[:, None, :, :]
        x2 = testing.shaped_arange((2, 1, 3, 1), xp, dtype)
        return x1 @ x2


@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                ((6, 5, 3, 2), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (6, 1, 2, 4)),
                ((6, 5, 3, 2), (1, 5, 2, 4)),
                ((6, 5, 3, 2), (1, 1, 2, 4)),
                ((6, 1, 3, 2), (6, 5, 2, 4)),
                ((1, 5, 3, 2), (6, 5, 2, 4)),
                ((1, 1, 3, 2), (6, 5, 2, 4)),
                ((3, 2), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (2, 4)),
                ((2,), (6, 5, 2, 4)),
                ((6, 5, 3, 2), (2,)),
            ],
        }
    )
)
class TestMatmulLarge(unittest.TestCase):
    # Avoid overflow
    skip_dtypes = {
        (numpy.int8, numpy.uint8),
        (numpy.int8, numpy.int16),
        (numpy.int8, numpy.float16),
        (numpy.uint8, numpy.uint8),
        (numpy.uint8, numpy.int16),
        (numpy.uint8, numpy.uint16),
        (numpy.int16, numpy.int16),
        (numpy.uint16, numpy.uint16),
    }

    @testing.for_all_dtypes(name="dtype1")
    @testing.for_all_dtypes(name="dtype2")
    @testing.numpy_cupy_allclose(
        rtol=1e-3, atol=1e-3, type_check=has_support_aspect64()
    )  # required for uint8
    def test_operator_matmul(self, xp, dtype1, dtype2):
        if (dtype1, dtype1) in self.skip_dtypes or (
            dtype1,
            dtype1,
        ) in self.skip_dtypes:
            return xp.array([])
        x1 = testing.shaped_random(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_random(self.shape_pair[1], xp, dtype2)
        return operator.matmul(x1, x2)

    @testing.for_all_dtypes(name="dtype1")
    @testing.for_all_dtypes(name="dtype2")
    @testing.numpy_cupy_allclose(
        rtol=1e-3, atol=1e-3, type_check=has_support_aspect64()
    )  # required for uint8
    def test_cupy_matmul(self, xp, dtype1, dtype2):
        if (dtype1, dtype1) in self.skip_dtypes or (
            dtype1,
            dtype1,
        ) in self.skip_dtypes:
            return xp.array([])
        shape1, shape2 = self.shape_pair
        x1 = testing.shaped_random(shape1, xp, dtype1)
        x2 = testing.shaped_random(shape2, xp, dtype2)
        return xp.matmul(x1, x2)


@pytest.mark.parametrize(
    "shape1, shape2",
    [
        # TODO: include it when issue #1540 in dpctl is resolved
        # ((256, 256, 3, 2), (256, 256, 2, 4)),
        ((256, 256, 3, 2), (2, 4)),
        ((3, 2), (256, 256, 2, 4)),
    ],
)
class TestMatmulIntegralLargeBatch:
    @testing.for_int_dtypes(name="dtype")
    @testing.numpy_cupy_array_equal()
    def test_operator_matmul(self, xp, dtype, shape1, shape2):
        x1 = testing.shaped_random(shape1, xp, dtype)
        x2 = testing.shaped_random(shape2, xp, dtype)
        return operator.matmul(x1, x2)

    @testing.for_int_dtypes(name="dtype")
    @testing.numpy_cupy_array_equal()
    def test_cupy_matmul(self, xp, dtype, shape1, shape2):
        x1 = testing.shaped_random(shape1, xp, dtype)
        x2 = testing.shaped_random(shape2, xp, dtype)
        return xp.matmul(x1, x2)


@pytest.mark.skip("until issue #1540 in dpctl is resolved")
class TestMatmulOverflow(unittest.TestCase):
    @testing.for_int_dtypes(name="dtype", no_bool=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_overflow(self, xp, dtype):
        value = numpy.iinfo(dtype).max
        a = xp.array([value - 10]).astype(dtype)
        b = xp.array([value - 10]).astype(dtype)
        return xp.matmul(a, b)


@testing.parameterize(
    *testing.product(
        {
            "shape_pair": [
                ((5, 3, 1), (3, 1, 4)),
                ((3, 2, 3), (3, 2, 4)),
                ((3, 2), ()),
                ((), (3, 2)),
                ((), ()),
                ((3, 2), (1,)),
                ((0, 2), (3, 0)),
                ((0, 1, 1), (2, 1, 1)),
            ],
        }
    )
)
class TestMatmulInvalidShape(unittest.TestCase):
    def test_invalid_shape(self):
        for xp in (numpy, cupy):
            shape1, shape2 = self.shape_pair
            x1 = testing.shaped_arange(shape1, xp, numpy.float32)
            x2 = testing.shaped_arange(shape2, xp, numpy.float32)
            with pytest.raises(ValueError):
                xp.matmul(x1, x2)


@testing.parameterize(
    *testing.product(
        {
            "shapes_axes": [
                (
                    (
                        (2, 5, 3, 2, 3, 4),
                        (3, 5, 1, 1, 1, 4),
                        (5, 5, 2, 2, 3, 4),
                    ),
                    [(1, 2), (0, 1), (0, 1)],
                ),
                (
                    (
                        (2, 5, 3, 2, 3, 4),
                        (2, 5, 3, 1, 4, 1),
                        (3, 1, 2, 5, 3, 2),
                    ),
                    [(-2, -1), (-2, -1), (0, 1)],
                ),
                (
                    ((3, 2, 4, 4), (4, 4, 3, 2), (4, 4, 3, 3)),
                    [(0, 1), (-1, -2), (-2, -1)],
                ),
                (
                    ((3, 2, 4, 4), (2, 3, 4, 4), (4, 3, 3, 4)),
                    [(0, 1), (0, 1), (1, 2)],
                ),
            ],
        }
    )
)
class TestMatmulAxes(unittest.TestCase):
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul_axes(self, xp):
        x1 = testing.shaped_arange(self.shapes_axes[0][0], xp)
        x2 = testing.shaped_arange(self.shapes_axes[0][1], xp)
        return xp.matmul(x1, x2, axes=self.shapes_axes[1])

    @testing.numpy_cupy_allclose(
        rtol=1e-3, atol=1e-3, type_check=has_support_aspect64()
    )  # required for uint8
    def test_cupy_matmul_axes_out(self, xp):
        x1 = testing.shaped_arange(self.shapes_axes[0][0], xp)
        x2 = testing.shaped_arange(self.shapes_axes[0][1], xp)
        out = xp.zeros(self.shapes_axes[0][2])
        result = xp.matmul(x1, x2, axes=self.shapes_axes[1], out=out)
        assert out is result
        return out
