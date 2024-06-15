import dpctl
import numpy
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_raises

import dpnp

from .helper import assert_dtype_allclose, get_all_dtypes, get_complex_dtypes


def _assert_selective_dtype_allclose(result, expected, dtype):
    # For numpy.dot, numpy.vdot, numpy.kron, numpy.inner, and numpy.tensordot,
    # when inputs are an scalar (which has the default dtype of platform) and
    # an array, the scalar dtype precision determines the output dtype
    # precision. In dpnp, we rely on dpnp.multiply for scalar-array product
    # and array (not scalar) determines output dtype precision of dpnp.multiply
    if dtype in [numpy.int32, numpy.float32, numpy.complex64]:
        assert_dtype_allclose(result, expected, check_only_type_kind=True)
    else:
        assert_dtype_allclose(result, expected)


class TestCross:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("axis", [None, 0], ids=["None", "0"])
    @pytest.mark.parametrize("axisc", [-1, 0], ids=["-1", "0"])
    @pytest.mark.parametrize("axisb", [-1, 0], ids=["-1", "0"])
    @pytest.mark.parametrize("axisa", [-1, 0], ids=["-1", "0"])
    @pytest.mark.parametrize(
        "x1",
        [[1, 2, 3], [1.0, 2.5, 6.0], [2, 4, 6]],
        ids=["[1, 2, 3]", "[1., 2.5, 6.]", "[2, 4, 6]"],
    )
    @pytest.mark.parametrize(
        "x2",
        [[4, 5, 6], [1.0, 5.0, 2.0], [6, 4, 3]],
        ids=["[4, 5, 6]", "[1., 5., 2.]", "[6, 4, 3]"],
    )
    def test_cross_3x3(self, x1, x2, axisa, axisb, axisc, axis):
        np_x1 = numpy.array(x1)
        dpnp_x1 = dpnp.array(x1)

        np_x2 = numpy.array(x2)
        dpnp_x2 = dpnp.array(x2)

        result = dpnp.cross(dpnp_x1, dpnp_x2, axisa, axisb, axisc, axis)
        expected = numpy.cross(np_x1, np_x2, axisa, axisb, axisc, axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "shape1, shape2, axis_a, axis_b, axis_c",
        [
            ((4, 2, 3, 5), (2, 4, 3, 5), 1, 0, -2),
            ((2, 2, 4, 5), (2, 4, 3, 5), 1, 2, -1),
            ((2, 3, 4, 5), (2, 4, 2, 5), 1, 2, -1),
            ((2, 3, 4, 5), (2, 4, 3, 5), 1, 2, -1),
            ((2, 3, 4, 5), (2, 4, 3, 5), -3, -2, 0),
        ],
    )
    def test_cross(self, dtype, shape1, shape2, axis_a, axis_b, axis_c):
        a = numpy.array(
            numpy.random.uniform(-5, 5, numpy.prod(shape1)), dtype=dtype
        ).reshape(shape1)
        b = numpy.array(
            numpy.random.uniform(-5, 5, numpy.prod(shape2)), dtype=dtype
        ).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia, ib, axis_a, axis_b, axis_c)
        expected = numpy.cross(a, b, axis_a, axis_b, axis_c)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize(
        "shape1, shape2, axis_a, axis_b, axis_c",
        [
            ((4, 2, 3, 5), (2, 4, 3, 5), 1, 0, -2),
            ((2, 2, 4, 5), (2, 4, 3, 5), 1, 2, -1),
            ((2, 3, 4, 5), (2, 4, 2, 5), 1, 2, -1),
            ((2, 3, 4, 5), (2, 4, 3, 5), 1, 2, -1),
            ((2, 3, 4, 5), (2, 4, 3, 5), -3, -2, 0),
        ],
    )
    def test_cross_complex(self, dtype, shape1, shape2, axis_a, axis_b, axis_c):
        x11 = numpy.random.uniform(-5, 5, numpy.prod(shape1))
        x12 = numpy.random.uniform(-5, 5, numpy.prod(shape1))
        x21 = numpy.random.uniform(-5, 5, numpy.prod(shape2))
        x22 = numpy.random.uniform(-5, 5, numpy.prod(shape2))
        a = numpy.array(x11 + 1j * x12, dtype=dtype).reshape(shape1)
        b = numpy.array(x21 + 1j * x22, dtype=dtype).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia, ib, axis_a, axis_b, axis_c)
        expected = numpy.cross(a, b, axis_a, axis_b, axis_c)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape1, shape2, axis",
        [
            ((2, 3, 4, 5), (2, 3, 4, 5), 0),
            ((2, 3, 4, 5), (2, 3, 4, 5), 1),
        ],
    )
    def test_cross_axis(self, dtype, shape1, shape2, axis):
        a = numpy.array(
            numpy.random.uniform(-5, 5, numpy.prod(shape1)), dtype=dtype
        ).reshape(shape1)
        b = numpy.array(
            numpy.random.uniform(-5, 5, numpy.prod(shape2)), dtype=dtype
        ).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia, ib, axis=axis)
        expected = numpy.cross(a, b, axis=axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_cross_input_dtype_matrix(self, dtype1, dtype2):
        if dtype1 == dpnp.bool and dtype2 == dpnp.bool:
            pytest.skip("boolean input arrays is not supported.")
        a = numpy.array(numpy.random.uniform(-5, 5, 3), dtype=dtype1)
        b = numpy.array(numpy.random.uniform(-5, 5, 3), dtype=dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia, ib)
        expected = numpy.cross(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "shape1, shape2, axis_a, axis_b, axis_c",
        [
            ((4, 2, 1, 5), (2, 4, 3, 5), 1, 0, -2),
            ((2, 2, 4, 5), (2, 4, 3, 1), 1, 2, -1),
            ((2, 3, 4, 1), (2, 4, 2, 5), 1, 2, -1),
            ((1, 3, 4, 5), (2, 4, 3, 5), 1, 2, -1),
            ((2, 3, 4, 5), (1, 1, 3, 1), -3, -2, 0),
        ],
    )
    def test_cross_broadcast(
        self, dtype, shape1, shape2, axis_a, axis_b, axis_c
    ):
        a = numpy.array(
            numpy.random.uniform(-5, 5, numpy.prod(shape1)), dtype=dtype
        ).reshape(shape1)
        b = numpy.array(
            numpy.random.uniform(-5, 5, numpy.prod(shape2)), dtype=dtype
        ).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia, ib, axis_a, axis_b, axis_c)
        expected = numpy.cross(a, b, axis_a, axis_b, axis_c)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("stride", [3, -3], ids=["3", "-3"])
    def test_cross_strided(self, dtype, stride):
        a = numpy.arange(1, 10, dtype=dtype)
        b = numpy.arange(1, 10, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia[::stride], ib[::stride])
        expected = numpy.cross(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)

    def test_cross_error(self):
        a = dpnp.arange(3)
        b = dpnp.arange(4)
        # Incompatible vector dimensions
        with pytest.raises(ValueError):
            dpnp.cross(a, b)

        a = dpnp.arange(3)
        b = dpnp.arange(4)
        # axis should be an integer
        with pytest.raises(TypeError):
            dpnp.cross(a, b, axis=0.0)

        a = dpnp.arange(2, dtype=dpnp.bool)
        # Input arrays with boolean data type are not supported
        with pytest.raises(TypeError):
            dpnp.cross(a, a)


class TestDot:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_dot_ones(self, dtype):
        n = 10**5
        a = numpy.ones(n, dtype=dtype)
        b = numpy.ones(n, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia, ib)
        expected = numpy.dot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_dot_arange(self, dtype):
        n = 10**2
        m = 10**3 if dtype is not dpnp.float32 else 10**2
        a = numpy.hstack((numpy.arange(n, dtype=dtype),) * m)
        b = numpy.flipud(a)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia, ib)
        expected = numpy.dot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_dot_scalar(self, dtype):
        a = 2
        b = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype)
        ib = dpnp.array(b)

        result = dpnp.dot(a, ib)
        expected = numpy.dot(a, b)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.dot(ib, a)
        expected = numpy.dot(b, a)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize(
        "shape_pair",
        [
            ((), (10,)),
            ((10,), ()),
            ((), ()),
            ((10,), (10,)),
            ((4, 3), (3, 2)),
            ((4, 3), (3,)),
            ((4,), (4, 2)),
            ((5, 4, 3), (3,)),
            ((4,), (5, 4, 3)),
            ((5, 3, 4), (6, 4, 2)),
        ],
        ids=[
            "0d_1d",
            "1d_0d",
            "0d_0d",
            "1d_1d",
            "2d_2d",
            "2d_1d",
            "1d_2d",
            "3d_1d",
            "1d_3d",
            "3d_3d",
        ],
    )
    def test_dot(self, dtype, shape_pair):
        shape1, shape2 = shape_pair
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        a = numpy.array(
            numpy.random.uniform(-5, 5, size1), dtype=dtype
        ).reshape(shape1)
        b = numpy.array(
            numpy.random.uniform(-5, 5, size2), dtype=dtype
        ).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia, ib)
        expected = numpy.dot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize(
        "shape_pair",
        [
            ((), (10,)),
            ((10,), ()),
            ((), ()),
            ((10,), (10,)),
            ((4, 3), (3, 2)),
            ((4, 3), (3,)),
            ((5, 4, 3), (3,)),
            ((4,), (4, 2)),
            ((5, 3, 4), (6, 4, 2)),
        ],
        ids=[
            "0d_1d",
            "1d_0d",
            "0d_0d",
            "1d_1d",
            "2d_2d",
            "2d_1d",
            "3d_1d",
            "1d_2d",
            "3d_3d",
        ],
    )
    def test_dot_complex(self, dtype, shape_pair):
        shape1, shape2 = shape_pair
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        x11 = numpy.random.uniform(-5, 5, size1)
        x12 = numpy.random.uniform(-5, 5, size1)
        x21 = numpy.random.uniform(-5, 5, size2)
        x22 = numpy.random.uniform(-5, 5, size2)
        a = numpy.array(x11 + 1j * x12, dtype=dtype).reshape(shape1)
        b = numpy.array(x21 + 1j * x22, dtype=dtype).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia, ib)
        expected = numpy.dot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "shape_pair",
        [
            ((), (10,)),
            ((10,), ()),
            ((), ()),
            ((10,), (10,)),
            ((4, 3), (3, 2)),
            ((4, 3), (3,)),
            ((5, 4, 3), (3,)),
            ((4,), (4, 2)),
            ((5, 3, 4), (6, 4, 2)),
        ],
        ids=[
            "0d_1d",
            "1d_0d",
            "0d_0d",
            "1d_1d",
            "2d_2d",
            "2d_1d",
            "3d_1d",
            "1d_2d",
            "3d_3d",
        ],
    )
    def test_dot_ndarray(self, dtype, shape_pair):
        shape1, shape2 = shape_pair
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        a = numpy.array(
            numpy.random.uniform(-5, 5, size1), dtype=dtype
        ).reshape(shape1)
        b = numpy.array(
            numpy.random.uniform(-5, 5, size2), dtype=dtype
        ).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = ia.dot(ib)
        expected = a.dot(b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "stride", [3, -1, -2, -5], ids=["3", "-1", "-2", "-5"]
    )
    def test_dot_strided(self, dtype, stride):
        a = numpy.arange(25, dtype=dtype)
        b = numpy.arange(25, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia[::stride], ib[::stride])
        expected = numpy.dot(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_dot_out_scalar(self, dtype):
        size = 10
        a = 2
        b = numpy.array(numpy.random.uniform(-5, 5, size), dtype=dtype)
        ia = 2
        ib = dpnp.array(b)

        dp_out = dpnp.empty((size,), dtype=dtype)
        result = dpnp.dot(ia, ib, out=dp_out)
        expected = numpy.dot(a, b)

        assert result is dp_out
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "shape_pair",
        [
            ((), (10,), (10,)),
            ((10,), (), (10,)),
            ((), (), ()),
            ((10,), (10,), ()),
            ((4, 3), (3, 2), (4, 2)),
            ((4, 3), (3,), (4,)),
            ((4,), (4, 2), (2,)),
            ((5, 4, 3), (3,), (5, 4)),
            ((4,), (5, 4, 3), (5, 3)),
            ((5, 3, 4), (6, 4, 2), (5, 3, 6, 2)),
        ],
        ids=[
            "0d_1d",
            "1d_0d",
            "0d_0d",
            "1d_1d",
            "2d_2d",
            "2d_1d",
            "1d_2d",
            "3d_1d",
            "1d_3d",
            "3d_3d",
        ],
    )
    def test_dot_out(self, dtype, shape_pair):
        shape1, shape2, out_shape = shape_pair
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        a = numpy.array(
            numpy.random.uniform(-5, 5, size1), dtype=dtype
        ).reshape(shape1)
        b = numpy.array(
            numpy.random.uniform(-5, 5, size2), dtype=dtype
        ).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        dp_out = dpnp.empty(out_shape, dtype=dtype)
        result = dpnp.dot(ia, ib, out=dp_out)
        expected = numpy.dot(a, b)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_dot_input_dtype_matrix(self, dtype1, dtype2):
        a = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype1)
        b = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia, ib)
        expected = numpy.dot(a, b)
        assert_dtype_allclose(result, expected)

    def test_dot_1d_error(self):
        a = dpnp.ones(25)
        b = dpnp.ones(24)
        # size of input arrays differ
        with pytest.raises(ValueError):
            dpnp.dot(a, b)

    def test_dot_sycl_queue_error(self):
        a = dpnp.ones((5,), sycl_queue=dpctl.SyclQueue())
        b = dpnp.ones((5,), sycl_queue=dpctl.SyclQueue())
        with pytest.raises(ValueError):
            dpnp.dot(a, b)

        a = dpnp.ones((5,))
        b = dpnp.ones((5,))
        out = dpnp.empty((), sycl_queue=dpctl.SyclQueue())
        with pytest.raises(ExecutionPlacementError):
            dpnp.dot(a, b, out=out)

    @pytest.mark.parametrize("ia", [1, dpnp.ones((), dtype=dpnp.float32)])
    def test_dot_out_error_scalar(self, ia):
        a = ia if dpnp.isscalar(ia) else ia.asnumpy()
        ib = dpnp.ones(10, dtype=dpnp.float32)
        b = ib.asnumpy()

        # output data type is incorrect
        dp_out = dpnp.empty((10,), dtype=dpnp.complex64)
        out = numpy.empty((10,), dtype=numpy.complex64)
        assert_raises(ValueError, dpnp.dot, ia, ib, out=dp_out)
        assert_raises(ValueError, numpy.dot, a, b, out=out)

        # output shape is incorrect
        dp_out = dpnp.empty((2,), dtype=dpnp.int32)
        out = numpy.empty((2,), dtype=numpy.int32)
        assert_raises(ValueError, dpnp.dot, ia, ib, out=dp_out)
        assert_raises(ValueError, numpy.dot, a, b, out=out)

    @pytest.mark.parametrize(
        "shape_pair",
        [
            ((10,), (10,), ()),
            ((3, 4), (4, 2), (3, 2)),
            ((3, 4), (4,), (3,)),
            ((5, 4, 3), (3,), (5, 4)),
            ((4,), (3, 4, 2), (3, 2)),
            ((5, 3, 4), (6, 4, 2), (5, 3, 6, 2)),
        ],
        ids=["1d_1d", "2d_2d", "2d_1d", "3d_1d", "1d_3d", "3d_3d"],
    )
    def test_dot_out_error(self, shape_pair):
        shape1, shape2, shape_out = shape_pair
        a = numpy.ones(shape1, dtype=numpy.int32)
        b = numpy.ones(shape2, dtype=numpy.int32)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        # output data type is incorrect
        np_out = numpy.empty(shape_out, dtype=numpy.int64)
        dp_out = dpnp.empty(shape_out, dtype=dpnp.int64)
        with pytest.raises(TypeError):
            dpnp.dot(ia, ib, out=dp_out)
        with pytest.raises(ValueError):
            numpy.dot(a, b, out=np_out)

        # output shape is incorrect
        np_out = numpy.empty((2, 3), dtype=numpy.int32)
        dp_out = dpnp.empty((2, 3), dtype=dpnp.int32)
        with pytest.raises(ValueError):
            dpnp.dot(ia, ib, out=dp_out)
        with pytest.raises(ValueError):
            numpy.dot(a, b, out=np_out)

        # "F" or "C" is irrelevant for 0d or 1d arrays
        if not (len(shape_out) in [0, 1]):
            # output should be C-contiguous
            np_out = numpy.empty(shape_out, dtype=numpy.int32, order="F")
            dp_out = dpnp.empty(shape_out, dtype=dpnp.int32, order="F")
            with pytest.raises(ValueError):
                dpnp.dot(ia, ib, out=dp_out)
            with pytest.raises(ValueError):
                numpy.dot(a, b, out=np_out)


class TestInner:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_inner_scalar(self, dtype):
        a = 2
        b = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype)
        ib = dpnp.array(b)

        result = dpnp.inner(a, ib)
        expected = numpy.inner(a, b)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.inner(ib, a)
        expected = numpy.inner(b, a)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((5,), (5,)),
            ((3, 5), (3, 5)),
            ((2, 4, 3, 5), (2, 4, 3, 5)),
            ((), (3, 4)),
            ((5,), ()),
        ],
    )
    def test_inner(self, dtype, shape1, shape2):
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        a = numpy.array(
            numpy.random.uniform(-5, 5, size1), dtype=dtype
        ).reshape(shape1)
        b = numpy.array(
            numpy.random.uniform(-5, 5, size2), dtype=dtype
        ).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.inner(ia, ib)
        expected = numpy.inner(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((5,), (5,)),
            ((3, 5), (3, 5)),
            ((2, 4, 3, 5), (2, 4, 3, 5)),
            ((), (3, 4)),
            ((5,), ()),
        ],
    )
    def test_inner_complex(self, dtype, shape1, shape2):
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        x11 = numpy.random.uniform(-5, 5, size1)
        x12 = numpy.random.uniform(-5, 5, size1)
        x21 = numpy.random.uniform(-5, 5, size2)
        x22 = numpy.random.uniform(-5, 5, size2)
        a = numpy.array(x11 + 1j * x12, dtype=dtype).reshape(shape1)
        b = numpy.array(x21 + 1j * x22, dtype=dtype).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.inner(ia, ib)
        expected = numpy.inner(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_inner_input_dtype_matrix(self, dtype1, dtype2):
        a = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype1)
        b = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.inner(ia, ib)
        expected = numpy.inner(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "stride", [3, -1, -2, -4], ids=["3", "-1", "-2", "-4"]
    )
    def test_inner_strided(self, dtype, stride):
        a = numpy.arange(20, dtype=dtype)
        b = numpy.arange(20, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.inner(ia[::stride], ib[::stride])
        expected = numpy.inner(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)

    def test_inner_error(self):
        a = dpnp.arange(24)
        b = dpnp.arange(23)
        # shape of input arrays is not similar at the last axis
        with pytest.raises(ValueError):
            dpnp.inner(a, b)


class TestKron:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_kron_scalar(self, dtype):
        a = 2
        b = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype)
        ib = dpnp.array(b)

        result = dpnp.kron(a, ib)
        expected = numpy.kron(a, b)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.kron(ib, a)
        expected = numpy.kron(b, a)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((5,), (5,)),
            ((3, 5), (4, 6)),
            ((2, 4, 3, 5), (3, 5, 6, 2)),
            ((4, 3, 5), (3, 5, 6, 2)),
            ((2, 4, 3, 5), (3, 5, 6)),
            ((2, 4, 3, 5), (3,)),
            ((), (3, 4)),
            ((5,), ()),
        ],
    )
    def test_kron(self, dtype, shape1, shape2):
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        a = numpy.array(
            numpy.random.uniform(-5, 5, size1), dtype=dtype
        ).reshape(shape1)
        b = numpy.array(
            numpy.random.uniform(-5, 5, size2), dtype=dtype
        ).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.kron(ia, ib)
        expected = numpy.kron(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((5,), (5,)),
            ((3, 5), (4, 6)),
            ((2, 4, 3, 5), (3, 5, 6, 2)),
            ((4, 3, 5), (3, 5, 6, 2)),
            ((2, 4, 3, 5), (3, 5, 6)),
            ((2, 4, 3, 5), (3,)),
            ((), (3, 4)),
            ((5,), ()),
        ],
    )
    def test_kron(self, dtype, shape1, shape2):
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        x11 = numpy.random.uniform(-5, 5, size1)
        x12 = numpy.random.uniform(-5, 5, size1)
        x21 = numpy.random.uniform(-5, 5, size2)
        x22 = numpy.random.uniform(-5, 5, size2)
        a = numpy.array(x11 + 1j * x12, dtype=dtype).reshape(shape1)
        b = numpy.array(x21 + 1j * x22, dtype=dtype).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.kron(ia, ib)
        expected = numpy.kron(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_kron_input_dtype_matrix(self, dtype1, dtype2):
        a = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype1)
        b = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.kron(ia, ib)
        expected = numpy.kron(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "stride", [3, -1, -2, -4], ids=["3", "-1", "-2", "-4"]
    )
    def test_kron_strided(self, dtype, stride):
        a = numpy.arange(20, dtype=dtype)
        b = numpy.arange(20, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.kron(ia[::stride], ib[::stride])
        expected = numpy.kron(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)


class TestMultiDot:
    def setup_method(self):
        numpy.random.seed(70)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_multi_dot_all_2d(self, dtype):
        n = 16
        a = dpnp.reshape(dpnp.arange(n, dtype=dtype), (4, 4))
        b = dpnp.reshape(dpnp.arange(n, dtype=dtype), (4, 4))
        c = dpnp.reshape(dpnp.arange(n, dtype=dtype), (4, 4))
        d = dpnp.reshape(dpnp.arange(n, dtype=dtype), (4, 4))

        a1 = numpy.arange(n, dtype=dtype).reshape((4, 4))
        b1 = numpy.arange(n, dtype=dtype).reshape((4, 4))
        c1 = numpy.arange(n, dtype=dtype).reshape((4, 4))
        d1 = numpy.arange(n, dtype=dtype).reshape((4, 4))

        result = dpnp.linalg.multi_dot([a, b, c, d])
        expected = numpy.linalg.multi_dot([a1, b1, c1, d1])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize(
        "shapes",
        [
            ((4, 5), (5, 4)),
            ((4,), (4, 6), (6, 8)),
            ((4, 8), (8, 6), (6,)),
            ((6,), (6, 8), (8,)),
            ((2, 10), (10, 5), (5, 8)),
            ((8, 5), (5, 10), (10, 2)),
            ((4, 6), (6, 9), (9, 7), (7, 8)),
            ((6,), (6, 10), (10, 7), (7, 8)),
            ((4, 6), (6, 10), (10, 7), (7,)),
            ((6,), (6, 10), (10, 7), (7,)),
            ((4, 6), (6, 9), (9, 7), (7, 8), (8, 3)),
        ],
        ids=[
            "two_arrays",
            "three_arrays_1st_1D",
            "three_arrays_last_1D",
            "three_arrays_1st_last_1D",
            "three_arrays_cost1",
            "three_arrays_cost2",
            "four_arrays",
            "four_arrays_1st_1D",
            "four_arrays_last_1D",
            "four_arrays_1st_last_1D",
            "five_arrays",
        ],
    )
    def test_multi_dot(self, shapes, dtype):
        numpy_array_list = []
        dpnp_array_list = []
        for shape in shapes:
            a = numpy.array(
                numpy.random.uniform(-5, 5, numpy.prod(shape)), dtype=dtype
            ).reshape(shape)
            ia = dpnp.array(a)

            numpy_array_list.append(a)
            dpnp_array_list.append(ia)

        result = dpnp.linalg.multi_dot(dpnp_array_list)
        expected = numpy.linalg.multi_dot(numpy_array_list)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize(
        "shapes",
        [
            ((4, 5), (5, 4)),
            ((4,), (4, 6), (6, 8)),
            ((4, 8), (8, 6), (6,)),
            ((6,), (6, 8), (8,)),
            ((2, 10), (10, 5), (5, 8)),
            ((8, 5), (5, 10), (10, 2)),
            ((4, 6), (6, 9), (9, 7), (7, 8)),
            ((6,), (6, 10), (10, 7), (7, 8)),
            ((4, 6), (6, 10), (10, 7), (7,)),
            ((6,), (6, 10), (10, 7), (7,)),
            ((4, 6), (6, 9), (9, 7), (7, 8), (8, 3)),
        ],
        ids=[
            "two_arrays",
            "three_arrays_1st_1D",
            "three_arrays_last_1D",
            "three_arrays_1st_last_1D",
            "three_arrays_cost1",
            "three_arrays_cost2",
            "four_arrays",
            "four_arrays_1st_1D",
            "four_arrays_last_1D",
            "four_arrays_1st_last_1D",
            "five_arrays",
        ],
    )
    def test_multi_dot_complex(self, shapes, dtype):
        numpy_array_list = []
        dpnp_array_list = []
        for shape in shapes:
            x1 = numpy.random.uniform(-5, 5, numpy.prod(shape))
            x2 = numpy.random.uniform(-5, 5, numpy.prod(shape))
            a = numpy.array(x1 + 1j * x2, dtype=dtype).reshape(shape)
            ia = dpnp.array(a)

            numpy_array_list.append(a)
            dpnp_array_list.append(ia)

        result = dpnp.linalg.multi_dot(dpnp_array_list)
        expected = numpy.linalg.multi_dot(numpy_array_list)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "shapes",
        [
            ((4, 5), (5, 4), (4, 4)),
            ((4,), (4, 6), (6, 8), (8,)),
            ((4, 8), (8, 6), (6,), (4,)),
            ((6,), (6, 8), (8,), ()),
            ((2, 10), (10, 5), (5, 8), (2, 8)),
            ((8, 5), (5, 10), (10, 2), (8, 2)),
            ((4, 6), (6, 9), (9, 7), (7, 8), (4, 8)),
            ((6,), (6, 10), (10, 7), (7, 8), (8,)),
            ((4, 6), (6, 10), (10, 7), (7,), (4,)),
            ((6,), (6, 10), (10, 7), (7,), ()),
            ((4, 6), (6, 9), (9, 7), (7, 8), (8, 3), (4, 3)),
        ],
        ids=[
            "two_arrays",
            "three_arrays_1st_1D",
            "three_arrays_last_1D",
            "three_arrays_1st_last_1D",
            "three_arrays_cost1",
            "three_arrays_cost2",
            "four_arrays",
            "four_arrays_1st_1D",
            "four_arrays_last_1D",
            "four_arrays_1st_last_1D",
            "five_arrays",
        ],
    )
    def test_multi_dot_out(self, shapes, dtype):
        numpy_array_list = []
        dpnp_array_list = []
        for shape in shapes[:-1]:
            a = numpy.array(
                numpy.random.uniform(-5, 5, numpy.prod(shape)), dtype=dtype
            ).reshape(shape)
            ia = dpnp.array(a)

            numpy_array_list.append(a)
            dpnp_array_list.append(ia)

        dp_out = dpnp.empty(shapes[-1], dtype=dtype)
        result = dpnp.linalg.multi_dot(dpnp_array_list, out=dp_out)
        assert result is dp_out
        expected = numpy.linalg.multi_dot(numpy_array_list)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "stride",
        [(-2, -2), (2, 2), (-2, 2), (2, -2)],
        ids=["(-2, -2)", "(2, 2)", "(-2, 2)", "(2, -2)"],
    )
    def test_multi_dot_strided(self, stride):
        numpy_array_list = []
        dpnp_array_list = []
        for num_array in [2, 3, 4, 5]:  # number of arrays in multi_dot
            for _ in range(num_array):  # creat arrays one by one
                A = numpy.random.rand(20, 20)
                B = dpnp.array(A)

                slices = (
                    slice(None, None, stride[0]),
                    slice(None, None, stride[1]),
                )
                a = A[slices]
                b = B[slices]

                numpy_array_list.append(a)
                dpnp_array_list.append(b)

            result = dpnp.linalg.multi_dot(dpnp_array_list)
            expected = numpy.linalg.multi_dot(numpy_array_list)
            assert_dtype_allclose(result, expected)

    def test_multi_dot_error(self):
        a = dpnp.ones(25)
        # Expecting at least two arrays
        with pytest.raises(ValueError):
            dpnp.linalg.multi_dot([a])

        a = dpnp.ones((5, 8, 10))
        b = dpnp.ones((10, 5))
        c = dpnp.ones((5, 15))
        # First array must be 1-D or 2-D
        with pytest.raises(dpnp.linalg.LinAlgError):
            dpnp.linalg.multi_dot([a, b, c])

        a = dpnp.ones((5, 10))
        b = dpnp.ones((10, 5))
        c = dpnp.ones((5, 15, 6))
        # Last array must be 1-D or 2-D
        with pytest.raises(dpnp.linalg.LinAlgError):
            dpnp.linalg.multi_dot([a, b, c])

        a = dpnp.ones((5, 10))
        b = dpnp.ones((10, 5, 8))
        c = dpnp.ones((8, 15))
        # Inner array must be 2-D
        with pytest.raises(dpnp.linalg.LinAlgError):
            dpnp.linalg.multi_dot([a, b, c])

        a = dpnp.ones((5, 10))
        b = dpnp.ones((10, 8))
        c = dpnp.ones((8, 15))
        # output should be C-contiguous
        dp_out = dpnp.empty((5, 15), order="F")
        with pytest.raises(ValueError):
            dpnp.linalg.multi_dot([a, b, c], out=dp_out)


class TestTensordot:
    def setup_method(self):
        numpy.random.seed(87)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_tensordot_scalar(self, dtype):
        a = 2
        b = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype)
        ib = dpnp.array(b)

        result = dpnp.tensordot(a, ib, axes=0)
        expected = numpy.tensordot(a, b, axes=0)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.tensordot(ib, a, axes=0)
        expected = numpy.tensordot(b, a, axes=0)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("axes", [0, 1, 2])
    def test_tensordot(self, dtype, axes):
        a = numpy.array(numpy.random.uniform(-10, 10, 64), dtype=dtype).reshape(
            4, 4, 4
        )
        b = numpy.array(numpy.random.uniform(-10, 10, 64), dtype=dtype).reshape(
            4, 4, 4
        )
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.tensordot(ia, ib, axes=axes)
        expected = numpy.tensordot(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("axes", [0, 1, 2])
    def test_tensordot_complex(self, dtype, axes):
        x11 = numpy.random.uniform(-10, 10, 64)
        x12 = numpy.random.uniform(-10, 10, 64)
        x21 = numpy.random.uniform(-10, 10, 64)
        x22 = numpy.random.uniform(-10, 10, 64)
        a = numpy.array(x11 + 1j * x12, dtype=dtype).reshape(4, 4, 4)
        b = numpy.array(x21 + 1j * x22, dtype=dtype).reshape(4, 4, 4)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.tensordot(ia, ib, axes=axes)
        expected = numpy.tensordot(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "axes",
        [
            ([0, 1]),
            ([0, 1], [1, 2]),
            (2, 3),
            ([-2, -3], [3, 2]),
            ((3, 1), (0, 2)),
        ],
    )
    def test_tensordot_axes(self, dtype, axes):
        a = numpy.array(
            numpy.random.uniform(-10, 10, 120), dtype=dtype
        ).reshape(2, 5, 3, 4)
        b = numpy.array(
            numpy.random.uniform(-10, 10, 120), dtype=dtype
        ).reshape(4, 2, 5, 3)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.tensordot(ia, ib, axes=axes)
        expected = numpy.tensordot(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_tensordot_input_dtype_matrix(self, dtype1, dtype2):
        a = numpy.array(
            numpy.random.uniform(-10, 10, 60), dtype=dtype1
        ).reshape(3, 4, 5)
        b = numpy.array(
            numpy.random.uniform(-10, 10, 40), dtype=dtype2
        ).reshape(4, 5, 2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.tensordot(ia, ib)
        expected = numpy.tensordot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "stride",
        [(-2, -2, -2, -2), (2, 2, 2, 2), (-2, 2, -2, 2), (2, -2, 2, -2)],
        ids=["-2", "2", "(-2, 2)", "(2, -2)"],
    )
    def test_tensordot_strided(self, stride):
        for dim in [1, 2, 3, 4]:
            axes = 1 if dim == 1 else 2
            A = numpy.random.rand(*([20] * dim))
            B = dpnp.asarray(A)
            slices = tuple(slice(None, None, stride[i]) for i in range(A.ndim))
            a = A[slices]
            b = B[slices]

            result = dpnp.tensordot(b, b, axes=axes)
            expected = numpy.tensordot(a, a, axes=axes)
            assert_dtype_allclose(result, expected)

    def test_tensordot_error(self):
        a = 5
        b = 2
        # both inputs are scalar
        with pytest.raises(TypeError):
            dpnp.tensordot(a, b, axes=0)

        a = dpnp.arange(24).reshape(2, 3, 4)
        b = dpnp.arange(24).reshape(3, 4, 2)
        # axes should be an integer
        with pytest.raises(TypeError):
            dpnp.tensordot(a, b, axes=2.0)

        # Axes must consist of two sequences
        with pytest.raises(ValueError):
            dpnp.tensordot(a, b, axes=([0, 2],))

        # Axes length mismatch
        with pytest.raises(ValueError):
            dpnp.tensordot(a, b, axes=([0, 2], [2]))

        # shape of input arrays is not similar at requested axes
        with pytest.raises(ValueError):
            dpnp.tensordot(a, b, axes=([0, 2], [2, 0]))

        # out of range index
        with pytest.raises(IndexError):
            dpnp.tensordot(a, b, axes=([0, 3], [2, 0]))

        # incorrect axes for scalar
        with pytest.raises(ValueError):
            dpnp.tensordot(dpnp.arange(4), 5, axes=1)

        # negative axes
        with pytest.raises(ValueError):
            dpnp.tensordot(dpnp.arange(4), dpnp.array(5), axes=-1)


class TestVdot:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_vdot_scalar(self, dtype):
        a = numpy.array([3.5], dtype=dtype)
        ia = dpnp.array(a)
        b = 2 + 3j

        result = dpnp.vdot(ia, b)
        expected = numpy.vdot(a, b)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.vdot(b, ia)
        expected = numpy.vdot(b, a)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize(
        "shape_pair",
        [
            ((), ()),
            ((10,), (10,)),
            ((4, 3), (3, 4)),
            ((4, 3), (12,)),
            ((5, 4, 3), (60,)),
            ((8,), (4, 2)),
            ((5, 3, 4), (3, 4, 5)),
        ],
        ids=[
            "0d_0d",
            "1d_1d",
            "2d_2d",
            "2d_1d",
            "3d_1d",
            "1d_2d",
            "3d_3d",
        ],
    )
    def test_vdot(self, dtype, shape_pair):
        shape1, shape2 = shape_pair
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        a = numpy.array(
            numpy.random.uniform(-5, 5, size1), dtype=dtype
        ).reshape(shape1)
        b = numpy.array(
            numpy.random.uniform(-5, 5, size2), dtype=dtype
        ).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.vdot(ia, ib)
        expected = numpy.vdot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize(
        "shape_pair",
        [
            ((), ()),
            ((10,), (10,)),
            ((4, 3), (3, 4)),
            ((4, 3), (12,)),
            ((5, 4, 3), (60,)),
            ((8,), (4, 2)),
            ((5, 3, 4), (3, 4, 5)),
        ],
        ids=[
            "0d_0d",
            "1d_1d",
            "2d_2d",
            "2d_1d",
            "3d_1d",
            "1d_2d",
            "3d_3d",
        ],
    )
    def test_vdot_complex(self, dtype, shape_pair):
        shape1, shape2 = shape_pair
        size1 = numpy.prod(shape1, dtype=int)
        size2 = numpy.prod(shape2, dtype=int)
        x11 = numpy.random.uniform(-5, 5, size1)
        x12 = numpy.random.uniform(-5, 5, size1)
        x21 = numpy.random.uniform(-5, 5, size2)
        x22 = numpy.random.uniform(-5, 5, size2)
        a = numpy.array(x11 + 1j * x12, dtype=dtype).reshape(shape1)
        b = numpy.array(x21 + 1j * x22, dtype=dtype).reshape(shape2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.vdot(ia, ib)
        expected = numpy.vdot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "stride", [3, -1, -2, -4], ids=["3", "-1", "-2", "-4"]
    )
    def test_vdot_strided(self, dtype, stride):
        a = numpy.arange(25, dtype=dtype)
        b = numpy.arange(25, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.vdot(ia[::stride], ib[::stride])
        expected = numpy.vdot(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_vdot_input_dtype_matrix(self, dtype1, dtype2):
        a = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype1)
        b = numpy.array(numpy.random.uniform(-5, 5, 10), dtype=dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.vdot(ia, ib)
        expected = numpy.vdot(a, b)
        assert_dtype_allclose(result, expected)

    def test_vdot_error(self):
        a = dpnp.ones(25)
        b = dpnp.ones(24)
        # size of input arrays differ
        with pytest.raises(ValueError):
            dpnp.vdot(a, b)

        a = dpnp.ones(25)
        b = 2
        # The first array should be of size one
        with pytest.raises(ValueError):
            dpnp.vdot(a, b)

        # The second array should be of size one
        with pytest.raises(ValueError):
            dpnp.vdot(b, a)
