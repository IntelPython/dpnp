import pytest
from .helper import get_all_dtypes

import dpnp

import numpy
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_warns
)


@pytest.mark.usefixtures('allow_fall_back_on_numpy')
@pytest.mark.parametrize('dtype', get_all_dtypes())
@pytest.mark.parametrize(
    'data', [[1, 2, 3], [1.0, 2.0, 3.0]], ids=['[1, 2, 3]', '[1., 2., 3.]']
)
def test_asfarray(dtype, data):
    expected = numpy.asfarray(data, dtype)
    result = dpnp.asfarray(data, dtype)

    assert_array_equal(result, expected)


@pytest.mark.parametrize('dtype', get_all_dtypes())
@pytest.mark.parametrize('data', [[1.0, 2.0, 3.0]], ids=['[1., 2., 3.]'])
@pytest.mark.parametrize('data_dtype', get_all_dtypes(no_none=True))
def test_asfarray2(dtype, data, data_dtype):
    expected = numpy.asfarray(numpy.array(data, dtype=data_dtype), dtype)
    result = dpnp.asfarray(dpnp.array(data, dtype=data_dtype), dtype)

    assert_array_equal(result, expected)


class TestDims:
    @pytest.mark.parametrize('dt', get_all_dtypes())
    @pytest.mark.parametrize('sh',
                             [(0,), (1,), (3,)],
                             ids=['(0,)', '(1,)', '(3,)'])
    def test_broadcast_array(self, sh, dt):
        np_a = numpy.array(0, dtype=dt)
        dp_a = dpnp.array(0, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize('dt', get_all_dtypes())
    @pytest.mark.parametrize('sh',
                             [(1,), (2,), (1, 2, 3)],
                             ids=['(1,)', '(2,)', '(1, 2, 3)'])
    def test_broadcast_ones(self, sh, dt):
        np_a = numpy.ones(1, dtype=dt)
        dp_a = dpnp.ones(1, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize('dt', get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize('sh',
                             [(3,), (1, 3), (2, 3)],
                             ids=['(3,)', '(1, 3)', '(2, 3)'])
    def test_broadcast_arange(self, sh, dt):
        np_a = numpy.arange(3, dtype=dt)
        dp_a = dpnp.arange(3, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize('dt', get_all_dtypes())
    @pytest.mark.parametrize(
        'sh1, sh2',
        [
            pytest.param([0], [0], id='(0)'),
            pytest.param([1], [1], id='(1)'),
            pytest.param([1], [2], id='(2)'),
        ],
    )
    def test_broadcast_not_tuple(self, sh1, sh2, dt):
        np_a = numpy.ones(sh1, dtype=dt)
        dp_a = dpnp.ones(sh1, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh2)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize('dt', get_all_dtypes())
    @pytest.mark.parametrize(
        'sh1, sh2',
        [
            pytest.param([1], (0,), id='(0,)'),
            pytest.param((1, 2), (0, 2), id='(0, 2)'),
            pytest.param((2, 1), (2, 0), id='(2, 0)'),
        ],
    )
    def test_broadcast_zero_shape(self, sh1, sh2, dt):
        np_a = numpy.ones(sh1, dtype=dt)
        dp_a = dpnp.ones(sh1, dtype=dt)
        func = lambda xp, a: xp.broadcast_to(a, sh2)

        assert_allclose(func(numpy, np_a), func(dpnp, dp_a))

    @pytest.mark.parametrize(
        'sh1, sh2',
        [
            pytest.param((0,), (), id='(0,)-()'),
            pytest.param((1,), (), id='(1,)-()'),
            pytest.param((3,), (), id='(3,)-()'),
            pytest.param((3,), (1,), id='(3,)-(1,)'),
            pytest.param((3,), (2,), id='(3,)-(2,)'),
            pytest.param((3,), (4,), id='(3,)-(4,)'),
            pytest.param((1, 2), (2, 1), id='(1, 2)-(2, 1)'),
            pytest.param((1, 2), (1,), id='(1, 2)-(1,)'),
            pytest.param((1,), -1, id='(1,)--1'),
            pytest.param((1,), (-1,), id='(1,)-(-1,)'),
            pytest.param((1, 2), (-1, 2), id='(1, 2)-(-1, 2)'),
        ],
    )
    def test_broadcast_raise(self, sh1, sh2):
        np_a = numpy.zeros(sh1)
        dp_a = dpnp.zeros(sh1)
        func = lambda xp, a: xp.broadcast_to(a, sh2)

        with pytest.raises(ValueError):
            func(numpy, np_a)
            func(dpnp, dp_a)


@pytest.mark.usefixtures('allow_fall_back_on_numpy')
class TestConcatenate:
    def test_returns_copy(self):
        a = dpnp.array(numpy.eye(3))
        b = dpnp.concatenate([a])
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_large_concatenate_axis_None(self):
        x = dpnp.arange(1, 100)
        r = dpnp.concatenate(x, None)
        assert_array_equal(x, r)
        r = dpnp.concatenate(x, 100)
        assert_array_equal(x, r)

    def test_concatenate(self):
        # Test concatenate function
        # One sequence returns unmodified (but as array)
        r4 = list(range(4))
        assert_array_equal(dpnp.concatenate((r4,)), r4)
        # Any sequence
        assert_array_equal(dpnp.concatenate((tuple(r4),)), r4)
        assert_array_equal(dpnp.concatenate((dpnp.array(r4),)), r4)
        # 1D default concatenation
        r3 = list(range(3))
        assert_array_equal(dpnp.concatenate((r4, r3)), r4 + r3)
        # Mixed sequence types
        assert_array_equal(dpnp.concatenate((tuple(r4), r3)), r4 + r3)
        assert_array_equal(
            dpnp.concatenate((dpnp.array(r4), r3)), r4 + r3
        )
        # Explicit axis specification
        assert_array_equal(dpnp.concatenate((r4, r3), 0), r4 + r3)
        # Including negative
        assert_array_equal(dpnp.concatenate((r4, r3), -1), r4 + r3)
        # 2D
        a23 = dpnp.array([[10, 11, 12], [13, 14, 15]])
        a13 = dpnp.array([[0, 1, 2]])
        res = dpnp.array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        assert_array_equal(dpnp.concatenate((a23, a13)), res)
        assert_array_equal(dpnp.concatenate((a23, a13), 0), res)
        assert_array_equal(dpnp.concatenate((a23.T, a13.T), 1), res.T)
        assert_array_equal(dpnp.concatenate((a23.T, a13.T), -1), res.T)
        # Arrays much match shape
        assert_raises(ValueError, dpnp.concatenate, (a23.T, a13.T), 0)
        # 3D
        res = dpnp.reshape(dpnp.arange(2 * 3 * 7), (2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        assert_array_equal(dpnp.concatenate((a0, a1, a2), 2), res)
        assert_array_equal(dpnp.concatenate((a0, a1, a2), -1), res)
        assert_array_equal(dpnp.concatenate((a0.T, a1.T, a2.T), 0), res.T)

        out = dpnp.copy(res)
        rout = dpnp.concatenate((a0, a1, a2), 2, out=out)
        assert_(out is rout)
        assert_equal(res, rout)


class TestHstack:
    def test_non_iterable(self):
        assert_raises(TypeError, dpnp.hstack, 1)

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_empty_input(self):
        assert_raises(ValueError, dpnp.hstack, ())

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_0D_array(self):
        b = dpnp.array(2)
        a = dpnp.array(1)
        res = dpnp.hstack([a, b])
        desired = dpnp.array([1, 2])
        assert_array_equal(res, desired)

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_1D_array(self):
        a = dpnp.array([1])
        b = dpnp.array([2])
        res = dpnp.hstack([a, b])
        desired = dpnp.array([1, 2])
        assert_array_equal(res, desired)

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
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


class TestVstack:
    def test_non_iterable(self):
        assert_raises(TypeError, dpnp.vstack, 1)

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_empty_input(self):
        assert_raises(ValueError, dpnp.vstack, ())

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_0D_array(self):
        a = dpnp.array(1)
        b = dpnp.array(2)
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2]])
        assert_array_equal(res, desired)

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_1D_array(self):
        a = dpnp.array([1])
        b = dpnp.array([2])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2]])
        assert_array_equal(res, desired)

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_2D_array(self):
        a = dpnp.array([[1], [2]])
        b = dpnp.array([[1], [2]])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1], [2], [1], [2]])
        assert_array_equal(res, desired)

    @pytest.mark.usefixtures('allow_fall_back_on_numpy')
    def test_2D_array2(self):
        a = dpnp.array([1, 2])
        b = dpnp.array([1, 2])
        res = dpnp.vstack([a, b])
        desired = dpnp.array([[1, 2], [1, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with assert_warns(FutureWarning):
            dpnp.vstack((numpy.arange(3) for _ in range(2)))
