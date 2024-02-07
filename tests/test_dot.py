import dpctl
import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpnp

from .helper import assert_dtype_allclose, get_all_dtypes, get_complex_dtypes


class TestDot:
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
        assert_allclose(result, expected)

    # TODO: get rid of falls back on NumPy when tensordot
    # is implemented using OneMKL
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize(
        "array_info",
        [
            (1, 10, (), (10,)),
            (10, 1, (10,), ()),
            (1, 1, (), ()),
            (10, 10, (10,), (10,)),
            (12, 6, (4, 3), (3, 2)),
            (12, 3, (4, 3), (3,)),
            (60, 3, (5, 4, 3), (3,)),
            (4, 8, (4,), (4, 2)),
            (60, 48, (5, 3, 4), (6, 4, 2)),
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
    def test_dot(self, dtype, array_info):
        size1, size2, shape1, shape2 = array_info
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

    # TODO: get rid of falls back on NumPy when tensordot
    # is implemented using OneMKL
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize(
        "array_info",
        [
            (1, 10, (), (10,)),
            (10, 1, (10,), ()),
            (1, 1, (), ()),
            (10, 10, (10,), (10,)),
            (12, 6, (4, 3), (3, 2)),
            (12, 3, (4, 3), (3,)),
            (60, 3, (5, 4, 3), (3,)),
            (4, 8, (4,), (4, 2)),
            (60, 48, (5, 3, 4), (6, 4, 2)),
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
    def test_dot_complex(self, dtype, array_info):
        size1, size2, shape1, shape2 = array_info
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

    # TODO: get rid of falls back on NumPy when tensordot
    # is implemented using OneMKL
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "array_info",
        [
            (1, 10, (), (10,)),
            (10, 1, (10,), ()),
            (1, 1, (), ()),
            (10, 10, (10,), (10,)),
            (12, 6, (4, 3), (3, 2)),
            (12, 3, (4, 3), (3,)),
            (60, 3, (5, 4, 3), (3,)),
            (4, 8, (4,), (4, 2)),
            (60, 48, (5, 3, 4), (6, 4, 2)),
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
    def test_dot_ndarray(self, dtype, array_info):
        size1, size2, shape1, shape2 = array_info
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
    def test_dot_strided(self, dtype):
        a = numpy.arange(25, dtype=dtype)
        b = numpy.arange(25, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.dot(ia[::3], ib[::3])
        expected = numpy.dot(a[::3], b[::3])
        assert_dtype_allclose(result, expected)

        result = dpnp.dot(ia, ib[::-1])
        expected = numpy.dot(a, b[::-1])
        assert_dtype_allclose(result, expected)

        result = dpnp.dot(ia[::-2], ib[::-2])
        expected = numpy.dot(a[::-2], b[::-2])
        assert_dtype_allclose(result, expected)

        result = dpnp.dot(ia[::-5], ib[::-5])
        expected = numpy.dot(a[::-5], b[::-5])
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
        assert_allclose(result, expected)

    # TODO: get rid of falls back on NumPy when tensordot
    # is implemented using OneMKL
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize(
        "array_info",
        [
            (1, 10, (), (10,), (10,)),
            (10, 1, (10,), (), (10,)),
            (1, 1, (), (), ()),
            (10, 10, (10,), (10,), ()),
            (12, 6, (4, 3), (3, 2), (4, 2)),
            (12, 3, (4, 3), (3,), (4,)),
            (60, 3, (5, 4, 3), (3,), (5, 4)),
            (4, 8, (4,), (4, 2), (2,)),
            (60, 48, (5, 3, 4), (6, 4, 2), (5, 3, 6, 2)),
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
    def test_dot_out(self, dtype, array_info):
        size1, size2, shape1, shape2, out_shape = array_info
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

    # NumPy does not raise an error for the following test.
    # it just does not update the out keyword if it as not properly defined
    @pytest.mark.parametrize("ia", [1, dpnp.ones((), dtype=dpnp.int32)])
    def test_dot_out_error_scalar(self, ia):
        ib = dpnp.ones(10, dtype=dpnp.int32)

        # output data type is incorrect
        dp_out = dpnp.empty((10,), dtype=dpnp.int64)
        # TODO: change it to ValueError, when updated
        # dpctl is being used in internal CI
        with pytest.raises((ValueError, TypeError)):
            dpnp.dot(ia, ib, out=dp_out)

        # output shape is incorrect
        dp_out = dpnp.empty((2,), dtype=dpnp.int32)
        # TODO: change it to ValueError, when updated
        # dpctl is being used in internal CI
        with pytest.raises((ValueError, TypeError)):
            dpnp.dot(ia, ib, out=dp_out)

    # TODO: get rid of falls back on NumPy when tensordot
    # is implemented using OneMKL
    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
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


@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
def test_multi_dot(type):
    n = 16
    a = dpnp.reshape(dpnp.arange(n, dtype=type), (4, 4))
    b = dpnp.reshape(dpnp.arange(n, dtype=type), (4, 4))
    c = dpnp.reshape(dpnp.arange(n, dtype=type), (4, 4))
    d = dpnp.reshape(dpnp.arange(n, dtype=type), (4, 4))

    a1 = numpy.arange(n, dtype=type).reshape((4, 4))
    b1 = numpy.arange(n, dtype=type).reshape((4, 4))
    c1 = numpy.arange(n, dtype=type).reshape((4, 4))
    d1 = numpy.arange(n, dtype=type).reshape((4, 4))

    result = dpnp.linalg.multi_dot([a, b, c, d])
    expected = numpy.linalg.multi_dot([a1, b1, c1, d1])
    assert_array_equal(expected, result)


class TestVdot:
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_vdot_scalar(self, dtype):
        a = numpy.array([3.5], dtype=dtype)
        ia = dpnp.array(a)
        b = 2 + 3j

        result = dpnp.vdot(ia, b)
        expected = numpy.vdot(a, b)
        assert_allclose(result, expected)

        result = dpnp.vdot(b, ia)
        expected = numpy.vdot(b, a)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize(
        "array_info",
        [
            (1, 1, (), ()),
            (10, 10, (10,), (10,)),
            (12, 12, (4, 3), (3, 4)),
            (12, 12, (4, 3), (12,)),
            (60, 60, (5, 4, 3), (60,)),
            (8, 8, (8,), (4, 2)),
            (60, 60, (5, 3, 4), (3, 4, 5)),
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
    def test_vdot(self, dtype, array_info):
        size1, size2, shape1, shape2 = array_info
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
        "array_info",
        [
            (1, 1, (), ()),
            (10, 10, (10,), (10,)),
            (12, 12, (4, 3), (3, 4)),
            (12, 12, (4, 3), (12,)),
            (60, 60, (5, 4, 3), (60,)),
            (8, 8, (8,), (4, 2)),
            (60, 60, (5, 3, 4), (3, 4, 5)),
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
    def test_vdot_complex(self, dtype, array_info):
        size1, size2, shape1, shape2 = array_info
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
    def test_vdot_strided(self, dtype):
        a = numpy.arange(25, dtype=dtype)
        b = numpy.arange(25, dtype=dtype)
        ia = dpnp.array(a)
        ib = dpnp.array(b)

        result = dpnp.vdot(ia[::3], ib[::3])
        expected = numpy.vdot(a[::3], b[::3])
        assert_dtype_allclose(result, expected)

        result = dpnp.vdot(ia, ib[::-1])
        expected = numpy.vdot(a, b[::-1])
        assert_dtype_allclose(result, expected)

        result = dpnp.vdot(ia[::-2], ib[::-2])
        expected = numpy.vdot(a[::-2], b[::-2])
        assert_dtype_allclose(result, expected)

        result = dpnp.vdot(ia[::-5], ib[::-5])
        expected = numpy.vdot(a[::-5], b[::-5])
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
