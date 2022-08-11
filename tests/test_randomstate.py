import pytest
import unittest

import dpnp
import numpy

from dpnp.random import RandomState
from numpy.testing import (assert_allclose, assert_raises, assert_array_equal, assert_array_almost_equal)


class TestSeed:
    def test_scalar(self):
        seed = 28041997
        size = (3, 2, 4)
        rs = RandomState(seed)
        a1 = dpnp.asnumpy(rs.uniform(size=size))
        rs = RandomState(seed)
        a2 = dpnp.asnumpy(rs.uniform(size=size))
        assert_allclose(a1, a2, rtol=1e-07, atol=0)

    @pytest.mark.parametrize("seed",
                             [range(3),
                              numpy.arange(3, dtype=numpy.int32),
                              dpnp.arange(3, dtype=numpy.int32),
                              [0], [4294967295], [2, 7, 15], (1,), (85, 6, 17)],
                             ids=['range(2)',
                                  'numpy.arange(2)',
                                  'dpnp.arange(2)',
                                  '[0]', '[4294967295]', '[2, 7, 15]', '(1,)', '(85, 6, 17)'])
    def test_array_range(self, seed):
        size = 15
        a1 = dpnp.asnumpy(RandomState(seed).uniform(size=size))
        a2 = dpnp.asnumpy(RandomState(seed).uniform(size=size))
        assert_allclose(a1, a2, rtol=1e-07, atol=0)

    @pytest.mark.parametrize("seed",
                             [0.5, -1.5, [-0.3], (1.7, 3),
                              'text',
                              numpy.arange(0, 1, 0.5),
                              dpnp.arange(3),
                              dpnp.arange(3, dtype=numpy.float32)],
                             ids=['0.5', '-1.5', '[-0.3]', '(1.7, 3)',
                                  'text',
                                  'numpy.arange(0, 1, 0.5)',
                                  'dpnp.arange(3)',
                                  'dpnp.arange(3, dtype=numpy.float32)'])
    def test_invalid_type(self, seed):
        # seed must be an unsigned 32-bit integer
        assert_raises(TypeError, RandomState, seed)

    @pytest.mark.parametrize("seed",
                             [-1, [-3, 7], (17, 3, -5), [4, 3, 2, 1], (7, 6, 5, 1),
                              range(-1, -11, -1),
                              numpy.arange(4, dtype=numpy.int32),
                              dpnp.arange(-3, 3, dtype=numpy.int32),
                              numpy.iinfo(numpy.uint32).max + 1,
                              (1, 7, numpy.iinfo(numpy.uint32).max + 1)],
                             ids=['-1', '[-3, 7]', '(17, 3, -5)', '[4, 3, 2, 1]', '(7, 6, 5, 1)',
                                  'range(-1, -11, -1)',
                                  'numpy.arange(4, dtype=numpy.int32)',
                                  'dpnp.arange(-3, 3, dtype=numpy.int32)',
                                  'numpy.iinfo(numpy.uint32).max + 1',
                                  '(1, 7, numpy.iinfo(numpy.uint32).max + 1)'])
    def test_invalid_value(self, seed):
        # seed must be an unsigned 32-bit integer
        assert_raises(ValueError, RandomState, seed)

    @pytest.mark.parametrize("seed",
                             [[], (),
                              [[1, 2, 3]],
                              [[1, 2, 3], [4, 5, 6]],
                              numpy.array([], dtype=numpy.int64),
                              dpnp.array([], dtype=numpy.int64)],
                             ids=['[]', '()',
                                  '[[1, 2, 3]]',
                                  '[[1, 2, 3], [4, 5, 6]]',
                                  'numpy.array([], dtype=numpy.int64)',
                                  'dpnp.array([], dtype=numpy.int64)'])
    def test_invalid_shape(self, seed):
        # seed must be an unsigned or 1-D array
        assert_raises(ValueError, RandomState, seed)

class TestUniform:
    @pytest.mark.parametrize("dtype",
                             [dpnp.float32, dpnp.float64, numpy.float32, numpy.float64],
                             ids=['dpnp.float32', 'dpnp.float64', 'numpy.float32', 'numpy.float64'])
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_uniform_float(self, dtype, usm_type):
        seed = 28041997
        actual = dpnp.asnumpy(RandomState(seed).uniform(low=1.23, high=10.54, size=(3, 2), dtype=dtype, usm_type=usm_type))
        desired = numpy.array([[3.700744485249743, 8.390019132522866],
                               [2.60340195777826,  4.473366308724508],
                               [1.773701806552708, 4.193498786306009]])
        assert_array_almost_equal(actual, desired, decimal=6)

    @pytest.mark.parametrize("dtype",
                             [dpnp.int32, numpy.int32, numpy.intc],
                             ids=['dpnp.int32', 'numpy.int32', 'numpy.intc'])
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_uniform_int(self, dtype, usm_type):
        seed = 28041997
        actual = dpnp.asnumpy(RandomState(seed).uniform(low=1.23, high=10.54, size=(3, 2), dtype=dtype, usm_type=usm_type))
        desired = numpy.array([[3, 8],
                               [2,  4],
                               [1, 4]])
        assert_array_equal(actual, desired)

    @pytest.mark.parametrize("high",
                             [dpnp.array([3]), numpy.array([3])],
                             ids=['dpnp.array([3])', 'numpy.array([3])'])
    @pytest.mark.parametrize("low",
                             [[2], dpnp.array([2]), numpy.array([2])],
                             ids=['[2]', 'dpnp.array([2])', 'numpy.array([2])'])
    def test_fallback(self, low, high):
        seed = 15
        # dpnp accepts only scalar as low and/or high, in other case it will be a fallback to numpy
        actual = dpnp.asnumpy(RandomState(seed).uniform(low=low, high=high, size=(3, 2, 5)))
        desired = numpy.random.RandomState(seed).uniform(low=low, high=high, size=(3, 2, 5))
        assert_array_almost_equal(actual, desired, decimal=15)

    @pytest.mark.parametrize("dtype",
                             [dpnp.float16, numpy.integer, dpnp.int, dpnp.bool, numpy.int64],
                             ids=['dpnp.float16', 'numpy.integer', 'dpnp.int', 'dpnp.bool', 'numpy.int64'])
    def test_invalid_dtype(self, dtype):
        # dtype must be float32 or float64
        assert_raises(TypeError, RandomState().uniform, dtype=dtype)
