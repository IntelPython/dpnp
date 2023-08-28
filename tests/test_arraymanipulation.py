import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_warns,
)

import dpnp

from .helper import get_all_dtypes, get_float_complex_dtypes


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize(
    "data", [[1, 2, 3], [1.0, 2.0, 3.0]], ids=["[1, 2, 3]", "[1., 2., 3.]"]
)
def test_asfarray(dtype, data):
    expected = numpy.asfarray(data, dtype)
    result = dpnp.asfarray(data, dtype)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize("data", [[1.0, 2.0, 3.0]], ids=["[1., 2., 3.]"])
@pytest.mark.parametrize("data_dtype", get_all_dtypes(no_none=True))
def test_asfarray2(dtype, data, data_dtype):
    expected = numpy.asfarray(numpy.array(data, dtype=data_dtype), dtype)
    result = dpnp.asfarray(dpnp.array(data, dtype=data_dtype), dtype)

    assert_array_equal(result, expected)


class TestDims:
    @pytest.mark.parametrize("dt", get_all_dtypes())
    @pytest.mark.parametrize(
        "sh", [(0,), (1,), (3,)], ids=["(0,)", "(1,)", "(3,)"]
    )
    def test_broadcast_array(self, sh, dt):
        np_a = numpy.array(0, dtype=dt)
        dp_a = dpnp.array(0, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize("dt", get_all_dtypes())
    @pytest.mark.parametrize(
        "sh", [(1,), (2,), (1, 2, 3)], ids=["(1,)", "(2,)", "(1, 2, 3)"]
    )
    def test_broadcast_ones(self, sh, dt):
        np_a = numpy.ones(1, dtype=dt)
        dp_a = dpnp.ones(1, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize("dt", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "sh", [(3,), (1, 3), (2, 3)], ids=["(3,)", "(1, 3)", "(2, 3)"]
    )
    def test_broadcast_arange(self, sh, dt):
        np_a = numpy.arange(3, dtype=dt)
        dp_a = dpnp.arange(3, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize("dt", get_all_dtypes())
    @pytest.mark.parametrize(
        "sh1, sh2",
        [
            pytest.param([0], [0], id="(0)"),
            pytest.param([1], [1], id="(1)"),
            pytest.param([1], [2], id="(2)"),
        ],
    )
    def test_broadcast_not_tuple(self, sh1, sh2, dt):
        np_a = numpy.ones(sh1, dtype=dt)
        dp_a = dpnp.ones(sh1, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh2)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize("dt", get_all_dtypes())
    @pytest.mark.parametrize(
        "sh1, sh2",
        [
            pytest.param([1], (0,), id="(0,)"),
            pytest.param((1, 2), (0, 2), id="(0, 2)"),
            pytest.param((2, 1), (2, 0), id="(2, 0)"),
        ],
    )
    def test_broadcast_zero_shape(self, sh1, sh2, dt):
        np_a = numpy.ones(sh1, dtype=dt)
        dp_a = dpnp.ones(sh1, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh2)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize(
        "sh1, sh2",
        [
            pytest.param((0,), (), id="(0,)-()"),
            pytest.param((1,), (), id="(1,)-()"),
            pytest.param((3,), (), id="(3,)-()"),
            pytest.param((3,), (1,), id="(3,)-(1,)"),
            pytest.param((3,), (2,), id="(3,)-(2,)"),
            pytest.param((3,), (4,), id="(3,)-(4,)"),
            pytest.param((1, 2), (2, 1), id="(1, 2)-(2, 1)"),
            pytest.param((1, 2), (1,), id="(1, 2)-(1,)"),
            pytest.param((1,), -1, id="(1,)--1"),
            pytest.param((1,), (-1,), id="(1,)-(-1,)"),
            pytest.param((1, 2), (-1, 2), id="(1, 2)-(-1, 2)"),
        ],
    )
    def test_broadcast_raise(self, sh1, sh2):
        np_a = numpy.zeros(sh1)
        dp_a = dpnp.zeros(sh1)
        func = lambda xp, a: xp.broadcast_to(a, sh2)

        with pytest.raises(ValueError):
            func(numpy, np_a)
            func(dpnp, dp_a)


class TestConcatenate:
    def test_returns_copy(self):
        a = dpnp.eye(3)
        b = dpnp.concatenate([a])
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_axis_exceptions(self, ndim):
        dp_a = dpnp.ones((1,) * ndim)
        np_a = numpy.ones((1,) * ndim)

        dp_res = dpnp.concatenate((dp_a, dp_a), axis=0)
        np_res = numpy.concatenate((np_a, np_a), axis=0)
        assert_equal(dp_res.asnumpy(), np_res)

        for axis in [ndim, -(ndim + 1)]:
            assert_raises(
                numpy.AxisError, dpnp.concatenate, (dp_a, dp_a), axis=axis
            )
            assert_raises(
                numpy.AxisError, numpy.concatenate, (np_a, np_a), axis=axis
            )

    def test_scalar_exceptions(self):
        assert_raises(TypeError, dpnp.concatenate, (0,))
        assert_raises(ValueError, numpy.concatenate, (0,))

        for xp in [dpnp, numpy]:
            with pytest.raises(ValueError):
                xp.concatenate((xp.array(0),))

    def test_dims_exception(self):
        for xp in [dpnp, numpy]:
            with pytest.raises(ValueError):
                xp.concatenate((xp.zeros(1), xp.zeros((1, 1))))

    def test_shapes_match_exception(self):
        axis = list(range(3))
        np_a = numpy.ones((1, 2, 3))
        np_b = numpy.ones((2, 2, 3))

        dp_a = dpnp.array(np_a)
        dp_b = dpnp.array(np_b)

        for _ in range(3):
            # shapes must match except for concatenation axis
            np_res = numpy.concatenate((np_a, np_b), axis=axis[0])
            dp_res = dpnp.concatenate((dp_a, dp_b), axis=axis[0])
            assert_equal(dp_res.asnumpy(), np_res)

            for i in range(1, 3):
                assert_raises(
                    ValueError, numpy.concatenate, (np_a, np_b), axis=axis[i]
                )
                assert_raises(
                    ValueError, dpnp.concatenate, (dp_a, dp_b), axis=axis[i]
                )

            np_a = numpy.moveaxis(np_a, -1, 0)
            dp_a = dpnp.moveaxis(dp_a, -1, 0)

            np_b = numpy.moveaxis(np_b, -1, 0)
            dp_b = dpnp.moveaxis(dp_b, -1, 0)
            axis.append(axis.pop(0))

    def test_no_array_exception(self):
        with pytest.raises(TypeError):
            dpnp.concatenate(())

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_concatenate_axis_None(self, dtype):
        stop, sh = (4, (2, 2)) if dtype is not dpnp.bool else (2, (2, 1))
        np_a = numpy.arange(stop, dtype=dtype).reshape(sh)
        dp_a = dpnp.arange(stop, dtype=dtype).reshape(sh)

        np_res = numpy.concatenate((np_a, np_a), axis=None)
        dp_res = dpnp.concatenate((dp_a, dp_a), axis=None)
        assert_equal(dp_res.asnumpy(), np_res)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_large_concatenate_axis_None(self, dtype):
        start, stop = (1, 100)
        np_a = numpy.arange(start, stop, dtype=dtype)
        dp_a = dpnp.arange(start, stop, dtype=dtype)

        np_res = numpy.concatenate(np_a, axis=None)
        dp_res = dpnp.concatenate(dp_a, axis=None)
        assert_array_equal(dp_res.asnumpy(), np_res)

        # numpy doesn't raise an exception here but probably should
        with pytest.raises(numpy.AxisError):
            dpnp.concatenate(dp_a, axis=100)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_concatenate(self, dtype):
        # Test concatenate function
        # One sequence returns unmodified (but as array)
        r4 = list(range(4))
        np_r4 = numpy.array(r4, dtype=dtype)
        dp_r4 = dpnp.array(r4, dtype=dtype)

        np_res = numpy.concatenate((np_r4,))
        dp_res = dpnp.concatenate((dp_r4,))
        assert_array_equal(dp_res.asnumpy(), np_res)

        # 1D default concatenation
        r3 = list(range(3))
        np_r3 = numpy.array(r3, dtype=dtype)
        dp_r3 = dpnp.array(r3, dtype=dtype)

        np_res = numpy.concatenate((np_r4, np_r3))
        dp_res = dpnp.concatenate((dp_r4, dp_r3))
        assert_array_equal(dp_res.asnumpy(), np_res)

        # Explicit axis specification
        np_res = numpy.concatenate((np_r4, np_r3), axis=0)
        dp_res = dpnp.concatenate((dp_r4, dp_r3), axis=0)
        assert_array_equal(dp_res.asnumpy(), np_res)

        # Including negative
        np_res = numpy.concatenate((np_r4, np_r3), axis=-1)
        dp_res = dpnp.concatenate((dp_r4, dp_r3), axis=-1)
        assert_array_equal(dp_res.asnumpy(), np_res)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_concatenate_2d(self, dtype):
        np_a23 = numpy.array([[10, 11, 12], [13, 14, 15]], dtype=dtype)
        np_a13 = numpy.array([[0, 1, 2]], dtype=dtype)

        dp_a23 = dpnp.array([[10, 11, 12], [13, 14, 15]], dtype=dtype)
        dp_a13 = dpnp.array([[0, 1, 2]], dtype=dtype)

        np_res = numpy.concatenate((np_a23, np_a13))
        dp_res = dpnp.concatenate((dp_a23, dp_a13))
        assert_array_equal(dp_res.asnumpy(), np_res)

        np_res = numpy.concatenate((np_a23, np_a13), axis=0)
        dp_res = dpnp.concatenate((dp_a23, dp_a13), axis=0)
        assert_array_equal(dp_res.asnumpy(), np_res)

        for axis in [1, -1]:
            np_res = numpy.concatenate((np_a23.T, np_a13.T), axis=axis)
            dp_res = dpnp.concatenate((dp_a23.T, dp_a13.T), axis=axis)
            assert_array_equal(dp_res.asnumpy(), np_res)

        # Arrays much match shape
        assert_raises(
            ValueError, numpy.concatenate, (np_a23.T, np_a13.T), axis=0
        )
        assert_raises(
            ValueError, dpnp.concatenate, (dp_a23.T, dp_a13.T), axis=0
        )

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_concatenate_3d(self, dtype):
        np_a = numpy.arange(2 * 3 * 7, dtype=dtype).reshape((2, 3, 7))
        np_a0 = np_a[..., :4]
        np_a1 = np_a[..., 4:6]
        np_a2 = np_a[..., 6:]

        dp_a = dpnp.arange(2 * 3 * 7, dtype=dtype).reshape((2, 3, 7))
        dp_a0 = dp_a[..., :4]
        dp_a1 = dp_a[..., 4:6]
        dp_a2 = dp_a[..., 6:]

        for axis in [2, -1]:
            np_res = numpy.concatenate((np_a0, np_a1, np_a2), axis=axis)
            dp_res = dpnp.concatenate((dp_a0, dp_a1, dp_a2), axis=axis)
            assert_array_equal(dp_res.asnumpy(), np_res)

        np_res = numpy.concatenate((np_a0.T, np_a1.T, np_a2.T), axis=0)
        dp_res = dpnp.concatenate((dp_a0.T, dp_a1.T, dp_a2.T), axis=0)
        assert_array_equal(dp_res.asnumpy(), np_res)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_concatenate_out(self, dtype):
        np_a = numpy.arange(2 * 3 * 7, dtype=dtype).reshape((2, 3, 7))
        np_a0 = np_a[..., :4]
        np_a1 = np_a[..., 4:6]
        np_a2 = np_a[..., 6:]
        np_out = numpy.empty_like(np_a)

        dp_a = dpnp.arange(2 * 3 * 7, dtype=dtype).reshape((2, 3, 7))
        dp_a0 = dp_a[..., :4]
        dp_a1 = dp_a[..., 4:6]
        dp_a2 = dp_a[..., 6:]
        dp_out = dpnp.empty_like(dp_a)

        np_res = numpy.concatenate((np_a0, np_a1, np_a2), axis=2, out=np_out)
        dp_res = dpnp.concatenate((dp_a0, dp_a1, dp_a2), axis=2, out=dp_out)

        assert dp_out is dp_res
        assert_array_equal(dp_out.asnumpy(), np_out)
        assert_array_equal(dp_res.asnumpy(), np_res)


class TestHstack:
    def test_non_iterable(self):
        assert_raises(TypeError, dpnp.hstack, 1)

    def test_empty_input(self):
        assert_raises(TypeError, dpnp.hstack, ())

    def test_0D_array(self):
        b = dpnp.array(2)
        a = dpnp.array(1)
        res = dpnp.hstack([a, b])
        desired = dpnp.array([1, 2])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = dpnp.array([1])
        b = dpnp.array([2])
        res = dpnp.hstack([a, b])
        desired = dpnp.array([1, 2])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = dpnp.array([[1], [2]])
        b = dpnp.array([[1], [2]])
        res = dpnp.hstack([a, b])
        desired = dpnp.array([[1, 1], [2, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with assert_warns(FutureWarning):
            dpnp.hstack((numpy.arange(3) for _ in range(2)))
        with assert_warns(FutureWarning):
            dpnp.hstack(map(lambda x: x, numpy.ones((3, 2))))


class TestStack:
    def test_non_iterable_input(self):
        with pytest.raises(TypeError):
            dpnp.stack(1)

    @pytest.mark.parametrize(
        "input", [(1, 2, 3), [dpnp.int32(1), dpnp.int32(2), dpnp.int32(3)]]
    )
    def test_scalar_input(self, input):
        with pytest.raises(TypeError):
            dpnp.stack(input)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_0d_array_input(self, dtype):
        np_arrays = []
        dp_arrays = []

        for x in (1, 2, 3):
            np_arrays.append(numpy.array(x, dtype=dtype))
            dp_arrays.append(dpnp.array(x, dtype=dtype))

        np_res = numpy.stack(np_arrays)
        dp_res = dpnp.stack(dp_arrays)
        assert_array_equal(dp_res.asnumpy(), np_res)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_1d_array_input(self, dtype):
        np_a = numpy.array([1, 2, 3], dtype=dtype)
        np_b = numpy.array([4, 5, 6], dtype=dtype)
        dp_a = dpnp.array(np_a, dtype=dtype)
        dp_b = dpnp.array(np_b, dtype=dtype)

        np_res = numpy.stack((np_a, np_b))
        dp_res = dpnp.stack((dp_a, dp_b))
        assert_array_equal(dp_res.asnumpy(), np_res)

        np_res = numpy.stack((np_a, np_b), axis=1)
        dp_res = dpnp.stack((dp_a, dp_b), axis=1)
        assert_array_equal(dp_res.asnumpy(), np_res)

        np_res = numpy.stack(list([np_a, np_b]))
        dp_res = dpnp.stack(list([dp_a, dp_b]))
        assert_array_equal(dp_res.asnumpy(), np_res)

        np_res = numpy.stack(numpy.array([np_a, np_b]))
        dp_res = dpnp.stack(dpnp.array([dp_a, dp_b]))
        assert_array_equal(dp_res.asnumpy(), np_res)

    @pytest.mark.parametrize("axis", [0, 1, -1, -2])
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_1d_array_axis(self, axis, dtype):
        arrays = [numpy.random.randn(3) for _ in range(10)]
        np_arrays = numpy.array(arrays, dtype=dtype)
        dp_arrays = dpnp.array(arrays, dtype=dtype)

        np_res = numpy.stack(np_arrays, axis=axis)
        dp_res = dpnp.stack(dp_arrays, axis=axis)
        assert_array_equal(dp_res.asnumpy(), np_res)

    @pytest.mark.parametrize("axis", [2, -3])
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_1d_array_invalid_axis(self, axis, dtype):
        arrays = [numpy.random.randn(3) for _ in range(10)]
        np_arrays = numpy.array(arrays, dtype=dtype)
        dp_arrays = dpnp.array(arrays, dtype=dtype)

        assert_raises(numpy.AxisError, numpy.stack, np_arrays, axis=axis)
        assert_raises(numpy.AxisError, dpnp.stack, dp_arrays, axis=axis)

    @pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_2d_array_axis(self, axis, dtype):
        arrays = [numpy.random.randn(3, 4) for _ in range(10)]
        np_arrays = numpy.array(arrays, dtype=dtype)
        dp_arrays = dpnp.array(arrays, dtype=dtype)

        np_res = numpy.stack(np_arrays, axis=axis)
        dp_res = dpnp.stack(dp_arrays, axis=axis)
        assert_array_equal(dp_res.asnumpy(), np_res)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_empty_arrays_input(self, dtype):
        arrays = [[], [], []]
        np_arrays = numpy.array(arrays, dtype=dtype)
        dp_arrays = dpnp.array(arrays, dtype=dtype)

        np_res = numpy.stack(np_arrays)
        dp_res = dpnp.stack(dp_arrays)
        assert_array_equal(dp_res.asnumpy(), np_res)

        np_res = numpy.stack(np_arrays, axis=1)
        dp_res = dpnp.stack(dp_arrays, axis=1)
        assert_array_equal(dp_res.asnumpy(), np_res)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_out(self, dtype):
        np_a = numpy.array([1, 2, 3], dtype=dtype)
        np_b = numpy.array([4, 5, 6], dtype=dtype)
        dp_a = dpnp.array(np_a, dtype=dtype)
        dp_b = dpnp.array(np_b, dtype=dtype)

        np_out = numpy.empty_like(np_a, shape=(2, 3))
        dp_out = dpnp.empty_like(dp_a, shape=(2, 3))

        np_res = numpy.stack((np_a, np_b), out=np_out)
        dp_res = dpnp.stack((dp_a, dp_b), out=dp_out)

        assert dp_out is dp_res
        assert_array_equal(dp_out.asnumpy(), np_out)
        assert_array_equal(dp_res.asnumpy(), np_res)

    def test_empty_list_input(self):
        with pytest.raises(TypeError):
            dpnp.stack([])

    @pytest.mark.parametrize(
        "sh1, sh2",
        [
            pytest.param((), (3,), id="()-(3,)"),
            pytest.param((3,), (), id="(3,)-()"),
            pytest.param((3, 3), (3,), id="(3, 3)-(3,)"),
            pytest.param((2,), (3,), id="(2,)-(3,)"),
        ],
    )
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_invalid_shapes_input(self, sh1, sh2, dtype):
        np_a = numpy.ones(sh1, dtype=dtype)
        np_b = numpy.ones(sh2, dtype=dtype)
        dp_a = dpnp.array(np_a, dtype=dtype)
        dp_b = dpnp.array(np_b, dtype=dtype)

        assert_raises(ValueError, numpy.stack, [np_a, np_b])
        assert_raises(ValueError, dpnp.stack, [dp_a, dp_b])
        assert_raises(ValueError, numpy.stack, [np_a, np_b], axis=1)
        assert_raises(ValueError, dpnp.stack, [dp_a, dp_b], axis=1)

    def test_generator_input(self):
        with pytest.raises(TypeError):
            dpnp.stack((x for x in range(3)))

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.usefixtures("suppress_complex_warning")
    @pytest.mark.parametrize("arr_dtype", get_all_dtypes())
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_casting_dtype(self, arr_dtype, dtype):
        np_a = numpy.array([1, 2, 3], dtype=arr_dtype)
        np_b = numpy.array([2.5, 3.5, 4.5], dtype=arr_dtype)
        dp_a = dpnp.array(np_a, dtype=arr_dtype)
        dp_b = dpnp.array(np_b, dtype=arr_dtype)

        np_res = numpy.stack(
            (np_a, np_b), axis=1, casting="unsafe", dtype=dtype
        )
        dp_res = dpnp.stack((dp_a, dp_b), axis=1, casting="unsafe", dtype=dtype)
        assert_array_equal(dp_res.asnumpy(), np_res)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("arr_dtype", get_float_complex_dtypes())
    @pytest.mark.parametrize("dtype", [dpnp.bool, dpnp.int32, dpnp.int64])
    def test_invalid_casting_dtype(self, arr_dtype, dtype):
        np_a = numpy.array([1, 2, 3], dtype=arr_dtype)
        np_b = numpy.array([2.5, 3.5, 4.5], dtype=arr_dtype)
        dp_a = dpnp.array(np_a, dtype=arr_dtype)
        dp_b = dpnp.array(np_b, dtype=arr_dtype)

        assert_raises(
            TypeError,
            numpy.stack,
            (np_a, np_b),
            axis=1,
            casting="safe",
            dtype=dtype,
        )
        assert_raises(
            TypeError,
            dpnp.stack,
            (dp_a, dp_b),
            axis=1,
            casting="safe",
            dtype=dtype,
        )


class TestVstack:
    def test_non_iterable(self):
        assert_raises(TypeError, dpnp.vstack, 1)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_empty_input(self):
        assert_raises(ValueError, dpnp.vstack, ())

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_0D_array(self):
        a = dpnp.array(1)
        b = dpnp.array(2)
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2]])
        assert_array_equal(res, desired)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_1D_array(self):
        a = dpnp.array([1])
        b = dpnp.array([2])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2]])
        assert_array_equal(res, desired)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_2D_array(self):
        a = dpnp.array([[1], [2]])
        b = dpnp.array([[1], [2]])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2], [1], [2]])
        assert_array_equal(res, desired)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_2D_array2(self):
        a = dpnp.array([1, 2])
        b = dpnp.array([1, 2])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1, 2], [1, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with assert_warns(FutureWarning):
            dpnp.vstack((numpy.arange(3) for _ in range(2)))


@pytest.mark.parametrize("arys", [1.0, dpnp.arange(9.0).reshape(3, 3), [3, 4]])
def test_atleast_1d(arys):
    expected = numpy.atleast_1d(arys)
    result = dpnp.atleast_1d(arys)
    assert_array_equal(result, expected)
