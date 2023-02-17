import pytest
from .helper import get_all_dtypes

import dpnp
import numpy


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize(
    "data", [[1, 2, 3], [1.0, 2.0, 3.0]], ids=["[1, 2, 3]", "[1., 2., 3.]"]
)
def test_asfarray(dtype, data):
    expected = numpy.asfarray(data, dtype)
    result = dpnp.asfarray(data, dtype)

    numpy.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize("data", [[1.0, 2.0, 3.0]], ids=["[1., 2., 3.]"])
@pytest.mark.parametrize("data_dtype", get_all_dtypes(no_none=True))
def test_asfarray2(dtype, data, data_dtype):
    expected = numpy.asfarray(numpy.array(data, dtype=data_dtype), dtype)
    result = dpnp.asfarray(dpnp.array(data, dtype=data_dtype), dtype)

    numpy.testing.assert_array_equal(result, expected)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestConcatenate:
    def test_returns_copy(self):
        a = dpnp.array(numpy.eye(3))
        b = dpnp.concatenate([a])
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_large_concatenate_axis_None(self):
        x = dpnp.arange(1, 100)
        r = dpnp.concatenate(x, None)
        numpy.testing.assert_array_equal(x, r)
        r = dpnp.concatenate(x, 100)
        numpy.testing.assert_array_equal(x, r)

    def test_concatenate(self):
        # Test concatenate function
        # One sequence returns unmodified (but as array)
        r4 = list(range(4))
        numpy.testing.assert_array_equal(dpnp.concatenate((r4,)), r4)
        # Any sequence
        numpy.testing.assert_array_equal(dpnp.concatenate((tuple(r4),)), r4)
        numpy.testing.assert_array_equal(dpnp.concatenate((dpnp.array(r4),)), r4)
        # 1D default concatenation
        r3 = list(range(3))
        numpy.testing.assert_array_equal(dpnp.concatenate((r4, r3)), r4 + r3)
        # Mixed sequence types
        numpy.testing.assert_array_equal(dpnp.concatenate((tuple(r4), r3)), r4 + r3)
        numpy.testing.assert_array_equal(
            dpnp.concatenate((dpnp.array(r4), r3)), r4 + r3
        )
        # Explicit axis specification
        numpy.testing.assert_array_equal(dpnp.concatenate((r4, r3), 0), r4 + r3)
        # Including negative
        numpy.testing.assert_array_equal(dpnp.concatenate((r4, r3), -1), r4 + r3)
        # 2D
        a23 = dpnp.array([[10, 11, 12], [13, 14, 15]])
        a13 = dpnp.array([[0, 1, 2]])
        res = dpnp.array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        numpy.testing.assert_array_equal(dpnp.concatenate((a23, a13)), res)
        numpy.testing.assert_array_equal(dpnp.concatenate((a23, a13), 0), res)
        numpy.testing.assert_array_equal(dpnp.concatenate((a23.T, a13.T), 1), res.T)
        numpy.testing.assert_array_equal(dpnp.concatenate((a23.T, a13.T), -1), res.T)
        # Arrays much match shape
        numpy.testing.assert_raises(ValueError, dpnp.concatenate, (a23.T, a13.T), 0)
        # 3D
        res = dpnp.reshape(dpnp.arange(2 * 3 * 7), (2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        numpy.testing.assert_array_equal(dpnp.concatenate((a0, a1, a2), 2), res)
        numpy.testing.assert_array_equal(dpnp.concatenate((a0, a1, a2), -1), res)
        numpy.testing.assert_array_equal(dpnp.concatenate((a0.T, a1.T, a2.T), 0), res.T)

        out = dpnp.copy(res)
        rout = dpnp.concatenate((a0, a1, a2), 2, out=out)
        numpy.testing.assert_(out is rout)
        numpy.testing.assert_equal(res, rout)


class TestHstack:
    def test_non_iterable(self):
        numpy.testing.assert_raises(TypeError, dpnp.hstack, 1)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_empty_input(self):
        numpy.testing.assert_raises(ValueError, dpnp.hstack, ())

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_0D_array(self):
        b = dpnp.array(2)
        a = dpnp.array(1)
        res = dpnp.hstack([a, b])
        desired = dpnp.array([1, 2])
        numpy.testing.assert_array_equal(res, desired)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_1D_array(self):
        a = dpnp.array([1])
        b = dpnp.array([2])
        res = dpnp.hstack([a, b])
        desired = dpnp.array([1, 2])
        numpy.testing.assert_array_equal(res, desired)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_2D_array(self):
        a = dpnp.array([[1], [2]])
        b = dpnp.array([[1], [2]])
        res = dpnp.hstack([a, b])
        desired = dpnp.array([[1, 1], [2, 2]])
        numpy.testing.assert_array_equal(res, desired)

    def test_generator(self):
        with numpy.testing.assert_warns(FutureWarning):
            dpnp.hstack((numpy.arange(3) for _ in range(2)))
        with numpy.testing.assert_warns(FutureWarning):
            dpnp.hstack(map(lambda x: x, numpy.ones((3, 2))))


class TestVstack:
    def test_non_iterable(self):
        numpy.testing.assert_raises(TypeError, dpnp.vstack, 1)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_empty_input(self):
        numpy.testing.assert_raises(ValueError, dpnp.vstack, ())

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_0D_array(self):
        a = dpnp.array(1)
        b = dpnp.array(2)
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2]])
        numpy.testing.assert_array_equal(res, desired)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_1D_array(self):
        a = dpnp.array([1])
        b = dpnp.array([2])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2]])
        numpy.testing.assert_array_equal(res, desired)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_2D_array(self):
        a = dpnp.array([[1], [2]])
        b = dpnp.array([[1], [2]])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2], [1], [2]])
        numpy.testing.assert_array_equal(res, desired)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_2D_array2(self):
        a = dpnp.array([1, 2])
        b = dpnp.array([1, 2])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1, 2], [1, 2]])
        numpy.testing.assert_array_equal(res, desired)

    def test_generator(self):
        with numpy.testing.assert_warns(FutureWarning):
            dpnp.vstack((numpy.arange(3) for _ in range(2)))
