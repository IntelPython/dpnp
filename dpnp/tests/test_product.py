import dpctl
import numpy
import pytest
from dpctl.tensor._numpy_helper import AxisError
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_array_equal, assert_raises

import dpnp

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_complex_dtypes,
)
from .third_party.cupy import testing


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

    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("axisc", [-1, 0])
    @pytest.mark.parametrize("axisb", [-1, 0])
    @pytest.mark.parametrize("axisa", [-1, 0])
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
    def test_3x3(self, x1, x2, axisa, axisb, axisc, axis):
        np_x1 = numpy.array(x1)
        dpnp_x1 = dpnp.array(x1)

        np_x2 = numpy.array(x2)
        dpnp_x2 = dpnp.array(x2)

        result = dpnp.cross(dpnp_x1, dpnp_x2, axisa, axisb, axisc, axis)
        expected = numpy.cross(np_x1, np_x2, axisa, axisb, axisc, axis)
        assert_dtype_allclose(result, expected)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
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
    def test_basic(self, dtype, shape1, shape2, axis_a, axis_b, axis_c):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia, ib, axis_a, axis_b, axis_c)
        expected = numpy.cross(a, b, axis_a, axis_b, axis_c)
        assert_dtype_allclose(result, expected)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape1, shape2, axis",
        [
            ((2, 3, 4, 5), (2, 3, 4, 5), 0),
            ((2, 3, 4, 5), (2, 3, 4, 5), 1),
        ],
    )
    def test_axis(self, dtype, shape1, shape2, axis):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia, ib, axis=axis)
        expected = numpy.cross(a, b, axis=axis)
        assert_dtype_allclose(result, expected, factor=24)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_input_dtype_matrix(self, dtype1, dtype2):
        if dtype1 == dpnp.bool and dtype2 == dpnp.bool:
            pytest.skip("boolean input arrays is not supported.")
        a = generate_random_numpy_array(3, dtype1)
        b = generate_random_numpy_array(3, dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia, ib)
        expected = numpy.cross(a, b)
        assert_dtype_allclose(result, expected, factor=24)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
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
    def test_broadcast(self, dtype, shape1, shape2, axis_a, axis_b, axis_c):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia, ib, axis_a, axis_b, axis_c)
        expected = numpy.cross(a, b, axis_a, axis_b, axis_c)
        assert_dtype_allclose(result, expected, factor=24)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("stride", [3, -3])
    def test_strided(self, dtype, stride):
        a = numpy.arange(1, 10, dtype=dtype)
        b = numpy.arange(1, 10, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.cross(ia[::stride], ib[::stride])
        expected = numpy.cross(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_linalg(self, axis):
        a = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = numpy.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.linalg.cross(ia, ib, axis=axis)
        expected = numpy.linalg.cross(a, b, axis=axis)
        assert_dtype_allclose(result, expected)

    def test_error(self):
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

    @testing.with_requires("numpy>=2.0")
    def test_linalg_error(self):
        a = dpnp.arange(4)
        b = dpnp.arange(4)
        # Both input arrays must be (arrays of) 3-dimensional vectors
        with pytest.raises(ValueError):
            dpnp.linalg.cross(a, b)


class TestDot:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_ones(self, dtype):
        n = 10**5
        a = numpy.ones(n, dtype=dtype)
        b = numpy.ones(n, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia, ib)
        expected = numpy.dot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_arange(self, dtype):
        n = 10**2
        m = 10**3 if dtype is not dpnp.float32 else 10**2
        a = numpy.hstack((numpy.arange(n, dtype=dtype),) * m)
        b = numpy.flipud(a)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia, ib)
        expected = numpy.dot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_scalar(self, dtype):
        a = 2
        b = generate_random_numpy_array(10, dtype)
        ib = dpnp.array(b)

        result = dpnp.dot(a, ib)
        expected = numpy.dot(a, b)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.dot(ib, a)
        expected = numpy.dot(b, a)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
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
    def test_basic(self, dtype, shape1, shape2):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia, ib)
        expected = numpy.dot(a, b)
        assert_dtype_allclose(result, expected)

        # ndarray
        result = ia.dot(ib)
        expected = a.dot(b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("stride", [3, -1, -2, -5])
    def test_strided(self, dtype, stride):
        a = numpy.arange(25, dtype=dtype)
        b = numpy.arange(25, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia[::stride], ib[::stride])
        expected = numpy.dot(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    def test_out_scalar(self, dtype):
        a = 2
        b = generate_random_numpy_array(10, dtype)
        ib = dpnp.array(b)

        dp_out = dpnp.empty(10, dtype=dtype)
        result = dpnp.dot(a, ib, out=dp_out)
        expected = numpy.dot(a, b)

        assert result is dp_out
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "shape1, shape2, out_shape",
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
    def test_out(self, dtype, shape1, shape2, out_shape):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        dp_out = dpnp.empty(out_shape, dtype=dtype)
        result = dpnp.dot(ia, ib, out=dp_out)
        expected = numpy.dot(a, b)

        assert result is dp_out
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_input_dtype_matrix(self, dtype1, dtype2):
        a = generate_random_numpy_array(10, dtype1)
        b = generate_random_numpy_array(10, dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia, ib)
        expected = numpy.dot(a, b)
        assert_dtype_allclose(result, expected)

    def test_1d_error(self):
        a = dpnp.ones(25)
        b = dpnp.ones(24)
        # size of input arrays differ
        with pytest.raises(ValueError):
            dpnp.dot(a, b)

    def test_sycl_queue_error(self):
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
    def test_out_error_scalar(self, ia):
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
        "shape1, shape2, out_shape",
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
    def test_out_error(self, shape1, shape2, out_shape):
        a = numpy.ones(shape1, dtype=numpy.int32)
        b = numpy.ones(shape2, dtype=numpy.int32)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        # output data type is incorrect
        np_out = numpy.empty(out_shape, dtype=numpy.int64)
        dp_out = dpnp.empty(out_shape, dtype=dpnp.int64)
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
        if not (len(out_shape) in [0, 1]):
            # output should be C-contiguous
            np_out = numpy.empty(out_shape, dtype=numpy.int32, order="F")
            dp_out = dpnp.empty(out_shape, dtype=dpnp.int32, order="F")
            with pytest.raises(ValueError):
                dpnp.dot(ia, ib, out=dp_out)
            with pytest.raises(ValueError):
                numpy.dot(a, b, out=np_out)


class TestInner:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_scalar(self, dtype):
        a = 2
        b = generate_random_numpy_array(10, dtype)
        ib = dpnp.array(b)

        result = dpnp.inner(a, ib)
        expected = numpy.inner(a, b)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.inner(ib, a)
        expected = numpy.inner(b, a)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
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
    def test_basic(self, dtype, shape1, shape2):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.inner(ia, ib)
        expected = numpy.inner(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_input_dtype_matrix(self, dtype1, dtype2):
        a = generate_random_numpy_array(10, dtype1)
        b = generate_random_numpy_array(10, dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.inner(ia, ib)
        expected = numpy.inner(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("stride", [3, -1, -2, -4])
    def test_strided(self, dtype, stride):
        a = numpy.arange(20, dtype=dtype)
        b = numpy.arange(20, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.inner(ia[::stride], ib[::stride])
        expected = numpy.inner(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)

    def test_error(self):
        a = dpnp.arange(24)
        b = dpnp.arange(23)
        # shape of input arrays is not similar at the last axis
        with pytest.raises(ValueError):
            dpnp.inner(a, b)


class TestKron:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_scalar(self, dtype):
        a = 2
        b = generate_random_numpy_array(10, dtype)
        ib = dpnp.array(b)

        result = dpnp.kron(a, ib)
        expected = numpy.kron(a, b)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.kron(ib, a)
        expected = numpy.kron(b, a)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
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
    def test_basic(self, dtype, shape1, shape2):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.kron(ia, ib)
        expected = numpy.kron(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_input_dtype_matrix(self, dtype1, dtype2):
        a = generate_random_numpy_array(10, dtype1)
        b = generate_random_numpy_array(10, dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.kron(ia, ib)
        expected = numpy.kron(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("stride", [3, -1, -2, -4])
    def test_strided1(self, dtype, stride):
        a = numpy.arange(20, dtype=dtype)
        b = numpy.arange(20, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.kron(ia[::stride], ib[::stride])
        expected = numpy.kron(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("stride", [2, -1, -2])
    def test_strided2(self, stride):
        a = numpy.arange(48).reshape(6, 8)
        b = numpy.arange(480).reshape(6, 8, 10)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.kron(
            ia[::stride, ::stride], ib[::stride, ::stride, ::stride]
        )
        expected = numpy.kron(
            a[::stride, ::stride], b[::stride, ::stride, ::stride]
        )
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("order", ["C", "F", "A"])
    def test_order(self, order):
        a = numpy.arange(48).reshape(6, 8, order=order)
        b = numpy.arange(480).reshape(6, 8, 10, order=order)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.kron(ia, ib)
        expected = numpy.kron(a, b)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)


class TestMatmul:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize(
        "order1, order2", [("C", "C"), ("C", "F"), ("F", "C"), ("F", "F")]
    )
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((4,), (4,)),
            ((1, 4), (4, 1)),
            ((4,), (4, 2)),
            ((1, 4), (4, 2)),
            ((2, 4), (4,)),
            ((2, 4), (4, 1)),
            ((1, 4), (4,)),  # output should be 1-d not 0-d
            ((4,), (4, 1)),
            ((1, 4), (4, 1)),
            ((2, 4), (4, 3)),
            ((1, 2, 3), (1, 3, 5)),
            ((4, 2, 3), (4, 3, 5)),
            ((1, 2, 3), (4, 3, 5)),
            ((2, 3), (4, 3, 5)),
            ((4, 2, 3), (1, 3, 5)),
            ((4, 2, 3), (3, 5)),
            ((1, 1, 4, 3), (1, 1, 3, 5)),
            ((6, 7, 4, 3), (6, 7, 3, 5)),
            ((6, 7, 4, 3), (1, 1, 3, 5)),
            ((6, 7, 4, 3), (1, 3, 5)),
            ((6, 7, 4, 3), (3, 5)),
            ((6, 7, 4, 3), (1, 7, 3, 5)),
            ((6, 7, 4, 3), (7, 3, 5)),
            ((6, 7, 4, 3), (6, 1, 3, 5)),
            ((1, 1, 4, 3), (6, 7, 3, 5)),
            ((1, 4, 3), (6, 7, 3, 5)),
            ((4, 3), (6, 7, 3, 5)),
            ((6, 1, 4, 3), (6, 7, 3, 5)),
            ((1, 7, 4, 3), (6, 7, 3, 5)),
            ((7, 4, 3), (6, 7, 3, 5)),
            ((1, 5, 3, 2), (6, 5, 2, 4)),
            ((5, 3, 2), (6, 5, 2, 4)),
            ((1, 3, 3), (10, 1, 3, 1)),
            ((2, 3, 3), (10, 1, 3, 1)),
            ((10, 2, 3, 3), (10, 1, 3, 1)),
            ((10, 2, 3, 3), (10, 2, 3, 1)),
            ((10, 1, 1, 3), (1, 3, 3)),
            ((10, 1, 1, 3), (2, 3, 3)),
            ((10, 1, 1, 3), (10, 2, 3, 3)),
            ((10, 2, 1, 3), (10, 2, 3, 3)),
            ((3, 3, 1), (3, 1, 2)),
            ((3, 3, 1), (1, 1, 2)),
            ((1, 3, 1), (3, 1, 2)),
            ((4,), (3, 4, 1)),
            ((3, 1, 4), (4,)),
            ((3, 1, 4), (3, 4, 1)),
            ((4, 1, 3, 1), (1, 3, 1, 2)),
            ((1, 3, 3, 1), (4, 1, 1, 2)),
            # empty arrays
            ((2, 0), (0, 3)),
            ((0, 4), (4, 3)),
            ((2, 4), (4, 0)),
            ((1, 2, 3), (0, 3, 5)),
            ((0, 2, 3), (1, 3, 5)),
            ((2, 3), (0, 3, 5)),
            ((0, 2, 3), (3, 5)),
            ((0, 0, 4, 3), (1, 1, 3, 5)),
            ((6, 0, 4, 3), (1, 3, 5)),
            ((0, 7, 4, 3), (3, 5)),
            ((0, 7, 4, 3), (1, 7, 3, 5)),
            ((0, 7, 4, 3), (7, 3, 5)),
            ((6, 0, 4, 3), (6, 1, 3, 5)),
            ((1, 1, 4, 3), (0, 0, 3, 5)),
            ((1, 4, 3), (6, 0, 3, 5)),
            ((4, 3), (0, 0, 3, 5)),
            ((6, 1, 4, 3), (6, 0, 3, 5)),
            ((1, 7, 4, 3), (0, 7, 3, 5)),
            ((7, 4, 3), (0, 7, 3, 5)),
        ],
    )
    def test_basic(self, order1, order2, shape1, shape2):
        # input should be float type otherwise they are copied to c-contigous array
        # so testing order becomes meaningless
        dtype = dpnp.default_float_type()
        a = numpy.arange(numpy.prod(shape1), dtype=dtype).reshape(shape1)
        b = numpy.arange(numpy.prod(shape2), dtype=dtype).reshape(shape2)
        a = numpy.array(a, order=order1)
        b = numpy.array(b, order=order2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.matmul(ia, ib)
        expected = numpy.matmul(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((2, 4), (4, 3)),
            ((4, 2, 3), (4, 3, 5)),
            ((6, 7, 4, 3), (6, 7, 3, 5)),
        ],
        ids=[
            "((2, 4), (4, 3))",
            "((4, 2, 3), (4, 3, 5))",
            "((6, 7, 4, 3), (6, 7, 3, 5))",
        ],
    )
    def test_bool(self, shape1, shape2):
        x = numpy.arange(2, dtype=numpy.bool_)
        a = numpy.resize(x, numpy.prod(shape1)).reshape(shape1)
        b = numpy.resize(x, numpy.prod(shape2)).reshape(shape2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.matmul(ia, ib)
        expected = numpy.matmul(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            [(-3, -1), (0, 2), (-2, -3)],
            [(3, 1), (2, 0), (3, 1)],
            [(3, 1), (2, 0), (0, 1)],
        ],
    )
    def test_axes_ND_ND(self, axes):
        a = generate_random_numpy_array((2, 5, 3, 4))
        b = generate_random_numpy_array((4, 2, 5, 3))
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.matmul(ia, ib, axes=axes)
        expected = numpy.matmul(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.2")
    @pytest.mark.parametrize("func", ["matmul", "matvec"])
    @pytest.mark.parametrize(
        "axes",
        [
            [(1, 0), (0), (0)],
            [(1, 0), 0, 0],
            [(1, 0), (0,), (0,)],
        ],
    )
    def test_axes_ND_1D(self, func, axes):
        a = numpy.arange(3 * 4 * 5).reshape(3, 4, 5)
        b = numpy.arange(3)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = getattr(dpnp, func)(ia, ib, axes=axes)
        expected = getattr(numpy, func)(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.2")
    @pytest.mark.parametrize("func", ["matmul", "vecmat"])
    @pytest.mark.parametrize(
        "axes",
        [
            [(0,), (0, 1), (0)],
            [(0), (0, 1), 0],
            [0, (0, 1), (0,)],
        ],
    )
    def test_axes_1D_ND(self, func, axes):
        a = numpy.arange(3)
        b = numpy.arange(3 * 4 * 5).reshape(3, 4, 5)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = getattr(dpnp, func)(ia, ib, axes=axes)
        expected = getattr(numpy, func)(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    def test_axes_1D_1D(self):
        a = numpy.arange(3)
        ia = dpnp.array(a)

        axes = [0, 0, ()]
        result = dpnp.matmul(ia, ia, axes=axes)
        expected = numpy.matmul(a, a, axes=axes)
        assert_dtype_allclose(result, expected)

        iout = dpnp.empty((), dtype=ia.dtype)
        result = dpnp.matmul(ia, ia, axes=axes, out=iout)
        assert iout is result
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "axes, out_shape",
        [
            ([(-3, -1), (0, 2), (-2, -3)], (2, 5, 5, 3)),
            ([(3, 1), (2, 0), (3, 1)], (2, 4, 3, 4)),
            ([(3, 1), (2, 0), (1, 2)], (2, 4, 4, 3)),
        ],
    )
    def test_axes_out(self, dtype, axes, out_shape):
        a = generate_random_numpy_array((2, 5, 3, 4), dtype)
        b = generate_random_numpy_array((4, 2, 5, 3), dtype)
        ia, ib = dpnp.array(a), dpnp.array(b)

        iout = dpnp.empty(out_shape, dtype=dtype)
        result = dpnp.matmul(ia, ib, axes=axes, out=iout)
        assert result is iout
        expected = numpy.matmul(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes, b_shape, out_shape",
        [
            ([(1, 0), 0, 0], (3,), (4, 5)),
            ([(1, 0), 0, 1], (3,), (5, 4)),
            ([(1, 0), (0, 1), (1, 2)], (3, 1), (5, 4, 1)),
            ([(1, 0), (0, 1), (0, 2)], (3, 1), (4, 5, 1)),
            ([(1, 0), (0, 1), (1, 0)], (3, 1), (1, 4, 5)),
        ],
    )
    def test_axes_out_1D(self, axes, b_shape, out_shape):
        a = numpy.arange(3 * 4 * 5).reshape(3, 4, 5)
        b = numpy.arange(3).reshape(b_shape)
        ia, ib = dpnp.array(a), dpnp.array(b)

        iout = dpnp.empty(out_shape)
        out = numpy.empty(out_shape)
        result = dpnp.matmul(ia, ib, axes=axes, out=iout)
        assert result is iout
        expected = numpy.matmul(a, b, axes=axes, out=out)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("in_dt", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "out_dt", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((2, 4), (4, 3)),
            ((4, 2, 3), (4, 3, 5)),
            ((6, 7, 4, 3), (6, 7, 3, 5)),
        ],
        ids=[
            "((2, 4), (4, 3))",
            "((4, 2, 3), (4, 3, 5))",
            "((6, 7, 4, 3), (6, 7, 3, 5))",
        ],
    )
    def test_dtype_matrix_inout(self, in_dt, out_dt, shape1, shape2):
        a = generate_random_numpy_array(shape1, in_dt)
        b = generate_random_numpy_array(shape2, in_dt)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if dpnp.can_cast(dpnp.result_type(ia, ib), out_dt, casting="same_kind"):
            result = dpnp.matmul(ia, ib, dtype=out_dt)
            expected = numpy.matmul(a, b, dtype=out_dt)
            assert_dtype_allclose(result, expected)
        else:
            assert_raises(TypeError, dpnp.matmul, ia, ib, dtype=out_dt)
            assert_raises(TypeError, numpy.matmul, a, b, dtype=out_dt)

    @pytest.mark.parametrize("dtype1", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("dtype2", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((2, 4), (4, 3)),
            ((4, 2, 3), (4, 3, 5)),
            ((6, 7, 4, 3), (6, 7, 3, 5)),
        ],
        ids=[
            "((2, 4), (4, 3))",
            "((4, 2, 3), (4, 3, 5))",
            "((6, 7, 4, 3), (6, 7, 3, 5))",
        ],
    )
    def test_dtype_matrix_inputs(self, dtype1, dtype2, shape1, shape2):
        a = generate_random_numpy_array(shape1, dtype1)
        b = generate_random_numpy_array(shape2, dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.matmul(ia, ib)
        expected = numpy.matmul(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("order1", ["C", "F", "A"])
    @pytest.mark.parametrize("order2", ["C", "F", "A"])
    @pytest.mark.parametrize("order", ["C", "F", "K", "A"])
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((2, 4), (4, 3)),
            ((4, 2, 3), (4, 3, 5)),
            ((6, 7, 4, 3), (6, 7, 3, 5)),
        ],
        ids=[
            "((2, 4), (4, 3))",
            "((4, 2, 3), (4, 3, 5))",
            "((6, 7, 4, 3), (6, 7, 3, 5))",
        ],
    )
    def test_order(self, order1, order2, order, shape1, shape2):
        a = numpy.arange(numpy.prod(shape1)).reshape(shape1, order=order1)
        b = numpy.arange(numpy.prod(shape2)).reshape(shape2, order=order2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.matmul(ia, ib, order=order)
        expected = numpy.matmul(a, b, order=order)
        # For the special case of shape1 = (6, 7, 4, 3), shape2 = (6, 7, 3, 5)
        # and order1 = "F" and order2 = "F", NumPy result is not c-contiguous
        # nor f-contiguous, while dpnp (and cupy) results are c-contiguous
        if not (
            shape1 == (6, 7, 4, 3)
            and shape2 == (6, 7, 3, 5)
            and order1 == "F"
            and order2 == "F"
            and order == "K"
        ):
            assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "stride",
        [(-2, -2, -2, -2), (2, 2, 2, 2), (-2, 2, -2, 2), (2, -2, 2, -2)],
        ids=["-2", "2", "(-2, 2)", "(2, -2)"],
    )
    def test_strided1(self, stride):
        for dim in [1, 2, 3, 4]:
            shape = tuple(20 for _ in range(dim))
            A = numpy.random.rand(*shape)
            iA = dpnp.asarray(A)
            slices = tuple(slice(None, None, stride[i]) for i in range(dim))
            a = A[slices]
            ia = iA[slices]
            # input arrays will be copied into c-contiguous arrays
            # the 2D base is not c-contiguous nor f-contigous
            result = dpnp.matmul(ia, ia)
            expected = numpy.matmul(a, a)
            assert_dtype_allclose(result, expected)

            iOUT = dpnp.empty(shape, dtype=result.dtype)
            iout = iOUT[slices]
            result = dpnp.matmul(ia, ia, out=iout)
            assert result is iout
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "shape", [(10, 3, 3), (12, 10, 3, 3)], ids=["3D", "4D"]
    )
    @pytest.mark.parametrize("stride", [-1, -2, 2], ids=["-1", "-2", "2"])
    @pytest.mark.parametrize("transpose", [False, True], ids=["False", "True"])
    def test_strided2(self, shape, stride, transpose):
        # one dimension (axis=-3) is strided
        # if negative stride, copy is needed and the base becomes c-contiguous
        # otherwise the base remains the same as input in gemm_batch
        A = numpy.random.rand(*shape)
        iA = dpnp.asarray(A)
        if transpose:
            A = numpy.moveaxis(A, (-2, -1), (-1, -2))
            iA = dpnp.moveaxis(iA, (-2, -1), (-1, -2))
        index = [slice(None)] * len(shape)
        index[-3] = slice(None, None, stride)
        index = tuple(index)
        a = A[index]
        ia = iA[index]
        result = dpnp.matmul(ia, ia)
        expected = numpy.matmul(a, a)
        assert_dtype_allclose(result, expected)

        iOUT = dpnp.empty(shape, dtype=result.dtype)
        iout = iOUT[index]
        result = dpnp.matmul(ia, ia, out=iout)
        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "stride",
        [(-2, -2), (2, 2), (-2, 2), (2, -2)],
        ids=["(-2, -2)", "(2, 2)", "(-2, 2)", "(2, -2)"],
    )
    @pytest.mark.parametrize("transpose", [False, True])
    def test_strided3(self, stride, transpose):
        # 4D case, the 1st and 2nd dimensions are strided
        # For negative stride, copy is needed and the base becomes c-contiguous.
        # For positive stride, no copy but reshape makes the base c-contiguous.
        stride0, stride1 = stride
        shape = (12, 10, 3, 3)  # 4D array
        A = numpy.random.rand(*shape)
        iA = dpnp.asarray(A)
        if transpose:
            A = numpy.moveaxis(A, (-2, -1), (-1, -2))
            iA = dpnp.moveaxis(iA, (-2, -1), (-1, -2))
        a = A[::stride0, ::stride1]
        ia = iA[::stride0, ::stride1]
        result = dpnp.matmul(ia, ia)
        expected = numpy.matmul(a, a)
        assert_dtype_allclose(result, expected)

        iOUT = dpnp.empty(shape, dtype=result.dtype)
        iout = iOUT[::stride0, ::stride1]
        result = dpnp.matmul(ia, ia, out=iout)
        assert result is iout
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.2")
    @pytest.mark.parametrize("func", ["matmul", "matvec"])
    @pytest.mark.parametrize("incx", [-2, 2])
    @pytest.mark.parametrize("incy", [-2, 2])
    @pytest.mark.parametrize("transpose", [False, True])
    def test_strided_mat_vec(self, func, incx, incy, transpose):
        # vector is strided
        shape = (8, 10)  # 2D
        if transpose:
            s1 = shape[-2]
            s2 = shape[-1]
        else:
            s1 = shape[-1]
            s2 = shape[-2]
        a = numpy.random.rand(*shape)
        ia = dpnp.asarray(a)
        if transpose:
            a = numpy.moveaxis(a, (-2, -1), (-1, -2))
            ia = dpnp.moveaxis(ia, (-2, -1), (-1, -2))
        B = numpy.random.rand(2 * s1)
        b = B[::incx]
        iB = dpnp.asarray(B)
        ib = iB[::incx]

        result = getattr(dpnp, func)(ia, ib)
        expected = getattr(numpy, func)(a, b)
        assert_dtype_allclose(result, expected)

        out_shape = shape[:-2] + (2 * s2,)
        iOUT = dpnp.empty(out_shape, dtype=result.dtype)
        iout = iOUT[..., ::incy]
        result = getattr(dpnp, func)(ia, ib, out=iout)
        assert result is iout
        assert_dtype_allclose(result, expected)

    @testing.with_requires("numpy>=2.2")
    @pytest.mark.parametrize("func", ["matmul", "vecmat"])
    @pytest.mark.parametrize("incx", [-2, 2])
    @pytest.mark.parametrize("incy", [-2, 2])
    @pytest.mark.parametrize("transpose", [False, True])
    def test_strided_vec_mat(self, func, incx, incy, transpose):
        # vector is strided
        shape = (8, 10)  # 2D
        if transpose:
            s1 = shape[-2]
            s2 = shape[-1]
        else:
            s1 = shape[-1]
            s2 = shape[-2]
        a = numpy.random.rand(*shape)
        ia = dpnp.asarray(a)
        if transpose:
            a = numpy.moveaxis(a, (-2, -1), (-1, -2))
            ia = dpnp.moveaxis(ia, (-2, -1), (-1, -2))
        B = numpy.random.rand(2 * s2)
        b = B[::incx]
        iB = dpnp.asarray(B)
        ib = iB[::incx]

        result = getattr(dpnp, func)(ib, ia)
        expected = getattr(numpy, func)(b, a)
        assert_dtype_allclose(result, expected)

        out_shape = shape[:-2] + (2 * s1,)
        iOUT = dpnp.empty(out_shape, dtype=result.dtype)
        iout = iOUT[..., ::incy]
        result = getattr(dpnp, func)(ib, ia, out=iout)
        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "order1, order2, out_order",
        [
            ("C", "C", "C"),
            ("C", "C", "F"),
            ("C", "F", "C"),
            ("C", "F", "F"),
            ("F", "C", "C"),
            ("F", "C", "F"),
            ("F", "F", "F"),
            ("F", "F", "C"),
        ],
    )
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_out1(self, order1, order2, out_order, dtype):
        # test gemm with out keyword
        a1 = numpy.arange(20, dtype=dtype).reshape(5, 4, order=order1)
        a2 = numpy.arange(28, dtype=dtype).reshape(4, 7, order=order2)
        b1, b2 = dpnp.array(a1), dpnp.array(a2)

        iout = dpnp.empty((5, 7), dtype=dtype, order=out_order)
        result = dpnp.matmul(b1, b2, out=iout)
        assert result is iout

        out = numpy.empty((5, 7), dtype=dtype, order=out_order)
        expected = numpy.matmul(a1, a2, out=out)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("trans", [True, False])
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_out2(self, trans, dtype):
        # test gemm_batch with out keyword
        # the base of input arrays is c-contiguous
        # the base of output array is c-contiguous or f-contiguous
        a = numpy.arange(24, dtype=dtype).reshape(2, 3, 4)
        b = numpy.arange(40, dtype=dtype).reshape(2, 4, 5)
        ia, ib = dpnp.array(a), dpnp.array(b)

        if trans:
            iout = dpnp.empty((2, 5, 3), dtype=dtype).transpose(0, 2, 1)
            out = numpy.empty((2, 5, 3), dtype=dtype).transpose(0, 2, 1)
        else:
            iout = dpnp.empty((2, 3, 5), dtype=dtype)
            out = numpy.empty((2, 3, 5), dtype=dtype)

        result = dpnp.matmul(ia, ib, out=iout)
        assert result is iout

        expected = numpy.matmul(a, b, out=out)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("trans", [True, False])
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_out3(self, trans, dtype):
        # test gemm_batch with out keyword
        # the base of input arrays is f-contiguous
        # the base of output array is c-contiguous or f-contiguous
        a = numpy.arange(24, dtype=dtype).reshape(2, 4, 3)
        b = numpy.arange(40, dtype=dtype).reshape(2, 5, 4)
        ia, ib = dpnp.array(a), dpnp.array(b)

        a = numpy.asarray(a).transpose(0, 2, 1)
        b = numpy.asarray(b).transpose(0, 2, 1)
        ia = ia.transpose(0, 2, 1)
        ib = ib.transpose(0, 2, 1)

        if trans:
            iout = dpnp.empty((2, 5, 3), dtype=dtype).transpose(0, 2, 1)
            out = numpy.empty((2, 5, 3), dtype=dtype).transpose(0, 2, 1)
        else:
            iout = dpnp.empty((2, 3, 5), dtype=dtype)
            out = numpy.empty((2, 3, 5), dtype=dtype)

        result = dpnp.matmul(ia, ib, out=iout)
        assert result is iout

        expected = numpy.matmul(a, b, out=out)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "out_shape",
        [
            ((4, 5)),
            ((6,)),
            ((4, 7, 2)),
        ],
    )
    def test_out_0D(self, out_shape):
        # for matmul of 1-D arrays, output is 0-D and if out keyword is given
        # NumPy repeats the data to match the shape of output array
        a = numpy.arange(3)
        ia = dpnp.asarray(a)

        numpy_out = numpy.empty(out_shape)
        iout = dpnp.empty(out_shape)
        result = dpnp.matmul(ia, ia, out=iout)
        expected = numpy.matmul(a, a, out=numpy_out)
        assert result is iout
        assert_dtype_allclose(result, expected)

    @testing.slow
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((5000, 5000, 2, 2), (5000, 5000, 2, 2)),
            ((2, 2), (5000, 5000, 2, 2)),
            ((5000, 5000, 2, 2), (2, 2)),
        ],
    )
    def test_large(self, shape1, shape2):
        a = generate_random_numpy_array(shape1)
        b = generate_random_numpy_array(shape2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.matmul(ia, ib)
        expected = numpy.matmul(a, b)
        assert_dtype_allclose(result, expected, factor=24)

    @testing.with_requires("numpy>=2.0")
    def test_linalg_matmul(self):
        a = numpy.ones((3, 4))
        b = numpy.ones((4, 5))
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.linalg.matmul(ia, ib)
        expected = numpy.linalg.matmul(a, b)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "sh1, sh2",
        [
            ((2, 3, 3), (2, 3, 3)),
            ((3, 3, 3, 3), (3, 3, 3, 3)),
        ],
        ids=["gemm", "gemm_batch"],
    )
    def test_matmul_with_offsets(self, sh1, sh2):
        a = generate_random_numpy_array(sh1)
        b = generate_random_numpy_array(sh2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = ia[1] @ ib[1]
        expected = a[1] @ b[1]
        assert_dtype_allclose(result, expected)


class TestMatmulInplace:
    ALL_DTYPES = get_all_dtypes(no_none=True)
    DTYPES = {}
    for i in ALL_DTYPES:
        for j in ALL_DTYPES:
            if numpy.can_cast(j, i):
                DTYPES[f"{i}-{j}"] = (i, j)

    @pytest.mark.parametrize("dtype1, dtype2", DTYPES.values())
    def test_basic(self, dtype1, dtype2):
        a = numpy.arange(10).reshape(5, 2).astype(dtype1)
        b = numpy.ones((2, 2), dtype=dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)
        ia_id = id(ia)

        a @= b
        ia @= ib
        assert id(ia) == ia_id
        assert_dtype_allclose(ia, a)

    @pytest.mark.parametrize(
        "a_sh, b_sh",
        [
            pytest.param((10**5, 10), (10, 10), id="2d_large"),
            pytest.param((10**4, 10, 10), (1, 10, 10), id="3d_large"),
            pytest.param((3,), (3,), id="1d"),
            pytest.param((3, 3), (3,), id="2d_1d"),
            pytest.param((3,), (3, 3), id="1d_2d"),
            pytest.param((3, 3), (3, 1), id="2d_broadcast"),
            pytest.param((1, 3), (3, 3), id="2d_broadcast_reverse"),
            pytest.param((3, 3, 3), (1, 3, 1), id="3d_broadcast1"),
            pytest.param((3, 3, 3), (1, 3, 3), id="3d_broadcast2"),
            pytest.param((3, 3, 3), (3, 3, 1), id="3d_broadcast3"),
            pytest.param((1, 3, 3), (3, 3, 3), id="3d_broadcast_reverse1"),
            pytest.param((3, 1, 3), (3, 3, 3), id="3d_broadcast_reverse2"),
            pytest.param((1, 1, 3), (3, 3, 3), id="3d_broadcast_reverse3"),
        ],
    )
    def test_shapes(self, a_sh, b_sh):
        a_sz, b_sz = numpy.prod(a_sh), numpy.prod(b_sh)
        a = numpy.arange(a_sz).reshape(a_sh).astype(numpy.float64)
        b = numpy.arange(b_sz).reshape(b_sh)

        ia, ib = dpnp.array(a), dpnp.array(b)
        ia_id = id(ia)

        expected = a @ b
        if expected.shape != a_sh:
            if len(b_sh) == 1:
                # check the exception matches NumPy
                match = "inplace matrix multiplication requires"
            else:
                match = None

            with pytest.raises(ValueError, match=match):
                a @= b

            with pytest.raises(ValueError, match=match):
                ia @= ib
        else:
            ia @= ib
            assert id(ia) == ia_id
            assert_dtype_allclose(ia, expected)


class TestMatmulInvalidCases:
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((3, 2), ()),
            ((), (3, 2)),
            ((), ()),
        ],
    )
    def test_zero_dim(self, xp, shape1, shape2):
        a = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
        b = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)

        assert_raises(ValueError, xp.matmul, a, b)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((3,), (4,)),
            ((2, 3), (4, 5)),
            ((2, 4), (3, 5)),
            ((2, 3), (4,)),
            ((3,), (4, 5)),
            ((2, 2, 3), (2, 4, 5)),
            ((3, 2, 3), (2, 4, 5)),
            ((4, 3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (3, 2, 4)),
        ],
    )
    def test_invalid_shape(self, xp, shape1, shape2):
        a = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
        b = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)

        assert_raises(ValueError, xp.matmul, a, b)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize(
        "shape1, shape2, out_shape",
        [
            ((5, 4, 3), (3, 1), (3, 4, 1)),
            ((5, 4, 3), (3, 1), (5, 6, 1)),
            ((5, 4, 3), (3, 1), (5, 4, 2)),
            ((5, 4, 3), (3, 1), (4, 1)),
            ((5, 4, 3), (3,), (5, 3)),
            ((5, 4, 3), (3,), (6, 4)),
            ((4,), (3, 4, 5), (4, 5)),
            ((4,), (3, 4, 5), (3, 6)),
        ],
    )
    def test_invalid_shape_out(self, xp, shape1, shape2, out_shape):
        a = xp.arange(numpy.prod(shape1), dtype=xp.float32).reshape(shape1)
        b = xp.arange(numpy.prod(shape2), dtype=xp.float32).reshape(shape2)
        out = xp.empty(out_shape)

        assert_raises(ValueError, xp.matmul, a, b, out=out)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True)[:-2])
    def test_invalid_dtype(self, xp, dtype):
        in_dtype = get_all_dtypes(no_none=True)[-1]
        a = xp.arange(5 * 4, dtype=in_dtype).reshape(5, 4)
        b = xp.arange(7 * 4, dtype=in_dtype).reshape(4, 7)
        out = xp.empty((5, 7), dtype=dtype)

        assert_raises(TypeError, xp.matmul, a, b, out=out)

    def test_exe_q(self):
        a = dpnp.ones((5, 4), sycl_queue=dpctl.SyclQueue())
        b = dpnp.ones((4, 7), sycl_queue=dpctl.SyclQueue())
        assert_raises(ValueError, dpnp.matmul, a, b)

        a = dpnp.ones((5, 4))
        b = dpnp.ones((4, 7))
        out = dpnp.empty((5, 7), sycl_queue=dpctl.SyclQueue())
        assert_raises(ExecutionPlacementError, dpnp.matmul, a, b, out=out)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_matmul_casting(self, xp):
        a = xp.arange(2 * 4, dtype=xp.float32).reshape(2, 4)
        b = xp.arange(4 * 3).reshape(4, 3)

        res = xp.empty((2, 3), dtype=xp.int64)
        assert_raises(TypeError, xp.matmul, a, b, out=res, casting="safe")

    def test_matmul_not_implemented(self):
        a = dpnp.arange(2 * 4).reshape(2, 4)
        b = dpnp.arange(4 * 3).reshape(4, 3)

        assert_raises(NotImplementedError, dpnp.matmul, a, b, subok=False)

        signature = (dpnp.float32, dpnp.float32, dpnp.float32)
        assert_raises(
            NotImplementedError, dpnp.matmul, a, b, signature=signature
        )

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_invalid_axes(self, xp):
        a = xp.arange(120).reshape(2, 5, 3, 4)
        b = xp.arange(120).reshape(4, 2, 5, 3)

        # axes must be a list
        axes = ((3, 1), (2, 0), (0, 1))
        assert_raises(TypeError, xp.matmul, a, b, axes=axes)

        # axes must be be a list of three tuples
        axes = [(3, 1), (2, 0)]
        assert_raises(ValueError, xp.matmul, a, b, axes=axes)

        # axes item should be a tuple
        axes = [(3, 1), (2, 0), [0, 1]]
        assert_raises(TypeError, xp.matmul, a, b, axes=axes)

        # axes item should be a tuple with 2 elements
        axes = [(3, 1), (2, 0), (0, 1, 2)]
        assert_raises(AxisError, xp.matmul, a, b, axes=axes)

        # axes must be an integer
        axes = [(3, 1), (2, 0), (0.0, 1)]
        assert_raises(TypeError, xp.matmul, a, b, axes=axes)

        # axes item 2 should be an empty tuple
        a = xp.arange(3)
        axes = [0, 0, 0]
        assert_raises(AxisError, xp.matmul, a, a, axes=axes)

        # axes should be a list of three tuples
        axes = [0, 0]
        assert_raises(ValueError, xp.matmul, a, a, axes=axes)

        a = xp.arange(3 * 4 * 5).reshape(3, 4, 5)
        b = xp.arange(3)
        # list object cannot be interpreted as an integer
        axes = [(1, 0), (0), [0]]
        assert_raises(TypeError, xp.matmul, a, b, axes=axes)

        # axes item should be a tuple with a single element, or an integer
        axes = [(1, 0), (0), (0, 1)]
        assert_raises(AxisError, xp.matmul, a, b, axes=axes)


@testing.with_requires("numpy>=2.2")
class TestMatvec:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((3, 4), (4,)),
            ((2, 3, 4), (4,)),
            ((3, 4), (2, 4)),
            ((5, 1, 3, 4), (2, 4)),
            ((2, 1, 4), (4,)),
        ],
    )
    def test_basic(self, dtype, shape1, shape2):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.matvec(ia, ib)
        expected = numpy.matvec(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            [(-1, -3), (-2,), -3],
            [(3, 1), 2, (0,)],
        ],
    )
    def test_axes(self, axes):
        a = generate_random_numpy_array((2, 5, 3, 4))
        b = generate_random_numpy_array((4, 2, 5, 3))
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.matvec(ia, ib, axes=axes)
        expected = numpy.matvec(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        a = xp.ones((5,))
        # first input does not have enough dimensions
        assert_raises(ValueError, xp.matvec, a, a)

        a = xp.ones((3, 4))
        b = xp.ones((5,))
        # core dimensions do not match
        assert_raises(ValueError, xp.matvec, a, b)

        a = xp.ones((2, 3, 4))
        b = xp.ones((5, 4))
        # broadcasting is not possible
        assert_raises(ValueError, xp.matvec, a, b)

        a = xp.ones((3, 2))
        b = xp.ones((3, 4))
        # two distinct core dimensions are needed, axis cannot be used
        assert_raises(TypeError, xp.matvec, a, b, axis=-2)


class TestMultiDot:
    def setup_method(self):
        numpy.random.seed(70)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
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
    def test_basic(self, shapes, dtype):
        numpy_array_list = []
        dpnp_array_list = []
        for shape in shapes:
            a = generate_random_numpy_array(shape, dtype)
            ia = dpnp.array(a)

            numpy_array_list.append(a)
            dpnp_array_list.append(ia)

        result = dpnp.linalg.multi_dot(dpnp_array_list)
        expected = numpy.linalg.multi_dot(numpy_array_list)
        assert_dtype_allclose(result, expected, factor=24)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
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
    def test_out(self, shapes, dtype):
        numpy_array_list = []
        dpnp_array_list = []
        for shape in shapes[:-1]:
            a = generate_random_numpy_array(shape, dtype)
            ia = dpnp.array(a)

            numpy_array_list.append(a)
            dpnp_array_list.append(ia)

        dp_out = dpnp.empty(shapes[-1], dtype=dtype)
        result = dpnp.linalg.multi_dot(dpnp_array_list, out=dp_out)
        assert result is dp_out
        expected = numpy.linalg.multi_dot(numpy_array_list)
        assert_dtype_allclose(result, expected, factor=24)

    @pytest.mark.parametrize(
        "stride",
        [(-2, -2), (2, 2), (-2, 2), (2, -2)],
        ids=["(-2, -2)", "(2, 2)", "(-2, 2)", "(2, -2)"],
    )
    def test_strided(self, stride):
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

    def test_error(self):
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

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_scalar(self, dtype):
        a = 2
        b = generate_random_numpy_array(10, dtype)
        ib = dpnp.array(b)

        result = dpnp.tensordot(a, ib, axes=0)
        expected = numpy.tensordot(a, b, axes=0)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.tensordot(ib, a, axes=0)
        expected = numpy.tensordot(b, a, axes=0)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axes", [0, 1, 2])
    def test_basic(self, dtype, axes):
        a = generate_random_numpy_array((4, 4, 4), dtype)
        b = generate_random_numpy_array((4, 4, 4), dtype)
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
    def test_axes(self, dtype, axes):
        a = generate_random_numpy_array((2, 5, 3, 4), dtype)
        b = generate_random_numpy_array((4, 2, 5, 3), dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.tensordot(ia, ib, axes=axes)
        expected = numpy.tensordot(a, b, axes=axes)
        assert_dtype_allclose(result, expected, factor=9)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_input_dtype_matrix(self, dtype1, dtype2):
        a = generate_random_numpy_array((3, 4, 5), dtype1)
        b = generate_random_numpy_array((4, 5, 2), dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.tensordot(ia, ib)
        expected = numpy.tensordot(a, b)
        assert_dtype_allclose(result, expected, factor=16)

    @pytest.mark.parametrize(
        "stride",
        [(-2, -2, -2, -2), (2, 2, 2, 2), (-2, 2, -2, 2), (2, -2, 2, -2)],
        ids=["-2", "2", "(-2, 2)", "(2, -2)"],
    )
    def test_strided(self, stride):
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

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize(
        "axes",
        [([0, 1]), ([0, 1], [1, 2]), ([-2, -3], [3, 2])],
    )
    def test_linalg(self, axes):
        a = generate_random_numpy_array((2, 5, 3, 4))
        b = generate_random_numpy_array((4, 2, 5, 3))
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.linalg.tensordot(ia, ib, axes=axes)
        expected = numpy.linalg.tensordot(a, b, axes=axes)
        assert_dtype_allclose(result, expected, factor=9)

    def test_error(self):
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

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_scalar(self, dtype):
        a = numpy.array([3.5], dtype=dtype)
        ia = dpnp.array(a)
        b = 2 + 3j

        result = dpnp.vdot(ia, b)
        expected = numpy.vdot(a, b)
        _assert_selective_dtype_allclose(result, expected, dtype)

        result = dpnp.vdot(b, ia)
        expected = numpy.vdot(b, a)
        _assert_selective_dtype_allclose(result, expected, dtype)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
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
    def test_basic(self, dtype, shape1, shape2):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.vdot(ia, ib)
        expected = numpy.vdot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize("stride", [3, -1, -2, -4])
    def test_strided(self, dtype, stride):
        a = numpy.arange(25, dtype=dtype)
        b = numpy.arange(25, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.vdot(ia[::stride], ib[::stride])
        expected = numpy.vdot(a[::stride], b[::stride])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_input_dtype_matrix(self, dtype1, dtype2):
        a = generate_random_numpy_array(10, dtype1)
        b = generate_random_numpy_array(10, dtype2)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.vdot(ia, ib)
        expected = numpy.vdot(a, b)
        assert_dtype_allclose(result, expected)

    def test_error(self):
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


@testing.with_requires("numpy>=2.0")
class TestVecdot:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((4,), (4,)),  # call_flag: dot
            ((1, 1, 4), (1, 1, 4)),  # call_flag: dot
            ((3, 1), (3, 1)),
            ((2, 0), (2, 0)),  # zero-size inputs, 1D output
            ((3, 0, 4), (3, 0, 4)),  # zero-size output
            ((3, 4), (3, 4)),
            ((1, 4), (3, 4)),
            ((4,), (3, 4)),
            ((3, 4), (1, 4)),
            ((3, 4), (4,)),
            ((1, 4, 5), (3, 1, 5)),
            ((1, 1, 4, 5), (3, 1, 5)),
            ((1, 4, 5), (1, 3, 1, 5)),
        ],
    )
    def test_basic(self, dtype, shape1, shape2):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.vecdot(ia, ib)
        expected = numpy.vecdot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [0, 2, -2])
    @pytest.mark.parametrize(
        "shape1, shape2",
        [((4,), (4, 4, 4)), ((3, 4, 5), (3, 4, 5))],
    )
    def test_axis1(self, axis, shape1, shape2):
        a = generate_random_numpy_array(shape1)
        b = generate_random_numpy_array(shape2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.vecdot(ia, ib)
        expected = numpy.vecdot(a, b)
        assert_dtype_allclose(result, expected)

    def test_axis2(self):
        # This is a special case, `a` cannot be broadcast_to `b`
        a = numpy.arange(4).reshape(1, 4)
        b = numpy.arange(60).reshape(3, 4, 5)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.vecdot(ia, ib, axis=1)
        expected = numpy.vecdot(a, b, axis=1)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            [(1,), (1,), ()],
            [(0), (0), ()],
            [0, 1, ()],
            [-2, -1],
        ],
    )
    def test_axes(self, axes):
        a = generate_random_numpy_array((5, 5, 5))
        ia = dpnp.array(a)

        result = dpnp.vecdot(ia, ia, axes=axes)
        expected = numpy.vecdot(a, a, axes=axes)
        assert_dtype_allclose(result, expected)

        iout = dpnp.empty((5, 5), dtype=ia.dtype)
        result = dpnp.vecdot(ia, ia, axes=axes, out=iout)
        assert iout is result
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes, b_shape",
        [
            ([1, 0, ()], (3,)),
            ([1, 0, ()], (3, 1)),
            ([1, 1, ()], (1, 3)),
        ],
    )
    def test_axes_out_1D(self, axes, b_shape):
        a = numpy.arange(60).reshape(4, 3, 5)
        b = numpy.arange(3).reshape(b_shape)
        ia, ib = dpnp.array(a), dpnp.array(b)

        iout = dpnp.empty((4, 5))
        out = numpy.empty((4, 5))
        result = dpnp.vecdot(ia, ib, axes=axes, out=iout)
        assert result is iout
        expected = numpy.vecdot(a, b, axes=axes, out=out)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("stride", [2, -1, -2])
    def test_strided(self, stride):
        a = numpy.arange(100).reshape(10, 10)
        b = numpy.arange(100).reshape(10, 10)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.vecdot(ia[::stride, ::stride], ib[::stride, ::stride])
        expected = numpy.vecdot(a[::stride, ::stride], b[::stride, ::stride])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes())
    @pytest.mark.parametrize("dtype2", get_all_dtypes())
    def test_input_dtype_matrix(self, dtype1, dtype2):
        a = generate_random_numpy_array(10, dtype1)
        b = generate_random_numpy_array(10, dtype2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.vecdot(ia, ib)
        expected = numpy.vecdot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("order1", ["C", "F", "A"])
    @pytest.mark.parametrize("order2", ["C", "F", "A"])
    @pytest.mark.parametrize("order", ["C", "F", "K", "A"])
    @pytest.mark.parametrize(
        "shape",
        [(4, 3), (4, 3, 5), (6, 7, 3, 5)],
        ids=["(4, 3)", "(4, 3, 5)", "(6, 7, 3, 5)"],
    )
    def test_order(self, order1, order2, order, shape):
        a = numpy.arange(numpy.prod(shape)).reshape(shape, order=order1)
        b = numpy.arange(numpy.prod(shape)).reshape(shape, order=order2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.vecdot(ia, ib, order=order)
        expected = numpy.vecdot(a, b, order=order)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("order", ["C", "F", "K", "A"])
    @pytest.mark.parametrize(
        "shape", [(2, 4, 0), (4, 0, 5)], ids=["(2, 4, 0)", "(4, 0, 5)"]
    )
    def test_order_trivial(self, order, shape):
        # input is both c-contiguous and f-contiguous
        a = numpy.ones(shape)
        ia = dpnp.asarray(a)

        result = dpnp.vecdot(ia, ia, order=order)
        expected = numpy.vecdot(a, a, order=order)
        if shape == (2, 4, 0) and order == "A":
            # NumPy does not behave correctly for this case, for order="A",
            # if input is both c- and f-contiguous, output is c-contiguous
            assert result.flags.c_contiguous
            assert not result.flags.f_contiguous
        else:
            assert result.flags.c_contiguous == expected.flags.c_contiguous
            assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "order1, order2, out_order",
        [
            ("C", "C", "C"),
            ("C", "C", "F"),
            ("C", "F", "C"),
            ("C", "F", "F"),
            ("F", "C", "C"),
            ("F", "C", "F"),
            ("F", "F", "F"),
            ("F", "F", "C"),
        ],
    )
    def test_out_order(self, order1, order2, out_order):
        a = numpy.arange(20).reshape(5, 4, order=order1)
        b = numpy.arange(20).reshape(5, 4, order=order2)
        ia, ib = dpnp.array(a), dpnp.array(b)

        iout = dpnp.empty(5, order=out_order)
        result = dpnp.vecdot(ia, ib, out=iout)
        assert result is iout

        out = numpy.empty(5, order=out_order)
        expected = numpy.vecdot(a, b, out=out)
        assert result.flags.c_contiguous == expected.flags.c_contiguous
        assert result.flags.f_contiguous == expected.flags.f_contiguous
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype1", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("dtype2", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((4,), ()),
            ((1, 1, 4), (1, 1)),
            ((6, 7, 4, 3), (6, 7, 4)),
            ((2, 0), (2,)),  # zero-size inputs, 1D output
            ((3, 0, 4), (3, 0)),  # zero-size output
        ],
    )
    def test_out_dtype(self, dtype1, dtype2, shape1, shape2):
        a = numpy.ones(shape1, dtype=dtype1)
        ia = dpnp.array(a)

        out = numpy.empty(shape2, dtype=dtype2)
        iout = dpnp.array(out)

        if dpnp.can_cast(dtype1, dtype2, casting="same_kind"):
            result = dpnp.vecdot(ia, ia, out=iout)
            expected = numpy.vecdot(a, a, out=out)
            assert_dtype_allclose(result, expected)
        else:
            assert_raises(TypeError, dpnp.vecdot, ia, ia, out=iout)
            assert_raises(TypeError, numpy.vecdot, a, a, out=iout)

    @pytest.mark.parametrize(
        "out_shape",
        [
            ((4, 5)),
            ((6,)),
            ((4, 7, 2)),
        ],
    )
    def test_out_0D(self, out_shape):
        # for vecdot of 1-D arrays, output is 0-D and if out keyword is given
        # NumPy repeats the data to match the shape of output array
        a = numpy.arange(3)
        ia = dpnp.array(a)

        out = numpy.empty(out_shape)
        iout = dpnp.array(out)
        result = dpnp.vecdot(ia, ia, out=iout)
        expected = numpy.vecdot(a, a, out=out)
        assert result is iout
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
    def test_linalg(self, axis):
        a = generate_random_numpy_array(4)
        b = generate_random_numpy_array((4, 4, 4))
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.linalg.vecdot(ia, ib)
        expected = numpy.linalg.vecdot(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        a = xp.ones((5, 4))
        b = xp.ones((5, 5))
        # core dimension differs
        assert_raises(ValueError, xp.vecdot, a, b)

        a = xp.ones((5, 4))
        b = xp.ones((3, 4))
        # input arrays are not compatible
        assert_raises(ValueError, xp.vecdot, a, b)

        a = xp.ones((3, 4))
        b = xp.ones((3, 4))
        c = xp.empty((5,))
        # out array shape is incorrect
        assert_raises(ValueError, xp.vecdot, a, b, out=c)

        # both axes and axis cannot be specified
        a = xp.ones((5, 5))
        assert_raises(TypeError, xp.vecdot, a, a, axes=[0, 0, ()], axis=-1)

        # axes should be a list of three tuples
        a = xp.ones(5)
        axes = [0, 0, 0, 0]
        assert_raises(ValueError, xp.vecdot, a, a, axes=axes)


@testing.with_requires("numpy>=2.2")
class TestVecmat:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "shape1, shape2",
        [
            ((3,), (3, 4)),
            ((3,), (2, 3, 4)),
            ((2, 3), (3, 4)),
            ((2, 3), (5, 1, 3, 4)),
            ((3,), (2, 3, 1)),
        ],
    )
    def test_basic(self, dtype, shape1, shape2):
        a = generate_random_numpy_array(shape1, dtype)
        b = generate_random_numpy_array(shape2, dtype)
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.vecmat(ia, ib)
        expected = numpy.vecmat(a, b)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            [-2, (-1, -3), (-3,)],
            [(2,), (3, 1), 0],
        ],
    )
    def test_axes(self, axes):
        a = generate_random_numpy_array((2, 4, 3, 5))
        b = generate_random_numpy_array((4, 2, 5, 3))
        ia, ib = dpnp.array(a), dpnp.array(b)

        result = dpnp.vecmat(ia, ib, axes=axes)
        expected = numpy.vecmat(a, b, axes=axes)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        a = xp.ones((5,))
        # second input does not have enough dimensions
        assert_raises(ValueError, xp.vecmat, a, a)

        a = xp.ones((4,))
        b = xp.ones((3, 5))
        # core dimensions do not match
        assert_raises(ValueError, xp.vecmat, a, b)

        a = xp.ones((3, 4))
        b = xp.ones((2, 4, 5))
        # broadcasting is not possible
        assert_raises(ValueError, xp.vecmat, a, b)
