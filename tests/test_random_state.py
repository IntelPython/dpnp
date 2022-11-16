import pytest
from unittest import mock

import dpnp
import numpy
import dpctl

from dpnp.random import RandomState
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises
)


def assert_cfd(data, exp_sycl_queue, exp_usm_type=None):
    assert exp_sycl_queue == data.sycl_queue
    if exp_usm_type:
        assert exp_usm_type == data.usm_type


class TestNormal:
    @pytest.mark.parametrize("dtype",
                             [dpnp.float32, dpnp.float64, None],
                             ids=['float32', 'float64', 'None'])
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_distr(self, dtype, usm_type):
        seed = 1234567
        sycl_queue = dpctl.SyclQueue()

        data = RandomState(seed, sycl_queue=sycl_queue).normal(loc=.12345,
                                                               scale=2.71,
                                                               size=(3, 2),
                                                               dtype=dtype,
                                                               usm_type=usm_type)

        if dtype is None:
            # dtype depends on fp64 support by the device
            dtype = dpnp.float64 if data.sycl_device.has_aspect_fp64 else dpnp.float32

        desired = numpy.array([[0.428205496031286, -0.55383273779227 ],
                               [2.027017795643378,  4.318888073163015],
                               [2.69080893259102,  -1.047967253719708]], dtype=dtype)

        # TODO: discuss with opneMKL: there is a difference between CPU and GPU
        # generated samples since 9 digit while precision=15 for float64
        # precision = numpy.finfo(dtype=dtype).precision
        precision = 8 if dtype == dpnp.float64 else numpy.finfo(dtype=dtype).precision
        assert_array_almost_equal(dpnp.asnumpy(data), desired, decimal=precision)

        # check if compute follows data isn't broken
        assert_cfd(data, sycl_queue, usm_type)


    @pytest.mark.parametrize("dtype",
                             [dpnp.float32, dpnp.float64, None],
                             ids=['float32', 'float64', 'None'])
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_scale(self, dtype, usm_type):
        rs = RandomState(39567)

        # zero scale means full ndarray of mean values
        assert_equal(rs.normal(loc=7, scale=0, dtype=dtype, usm_type=usm_type), 7)

        # scale must be non-negative ('-0.0' is negative value)
        assert_raises(ValueError, rs.normal, scale=-0., dtype=dtype, usm_type=usm_type)
        assert_raises(ValueError, rs.normal, scale=-3.71, dtype=dtype, usm_type=usm_type)


    @pytest.mark.parametrize("loc",
                             [numpy.inf, -numpy.inf,
                              numpy.nextafter(numpy.finfo('double').max, 0),
                              numpy.nextafter(numpy.finfo('double').min, 0)],
                             ids=['numpy.inf', '-numpy.inf', 'nextafter(max, 0)', 'nextafter(min, 0)'])
    def test_inf_loc(self, loc):
        assert_equal(RandomState(6531).normal(loc=loc, scale=1, size=1000), loc)


    def test_inf_scale(self):
        a = dpnp.asnumpy(RandomState().normal(0, numpy.inf, size=1000))
        assert_equal(numpy.isnan(a).any(), False)
        assert_equal(numpy.isinf(a).all(), True)
        assert_equal(a.max(), numpy.inf)
        assert_equal(a.min(), -numpy.inf)


    @pytest.mark.parametrize("loc",
                             [numpy.inf, -numpy.inf],
                             ids=['numpy.inf', '-numpy.inf'])
    def test_inf_loc_scale(self, loc):
        a = dpnp.asnumpy(RandomState().normal(loc=loc, scale=numpy.inf, size=1000))
        assert_equal(numpy.isnan(a).all(), False)
        assert_equal(numpy.nanmin(a), loc)
        assert_equal(numpy.nanmax(a), loc)


    def test_extreme_bounds(self):
        fmin = numpy.finfo('double').min
        fmax = numpy.finfo('double').max

        size = 1000
        func = RandomState(34567).normal

        assert_raises(OverflowError, func, fmin, 1)
        assert_raises(OverflowError, func, fmax, 1)
        assert_raises(OverflowError, func, 1, fmax)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, fmax, fmax)

        try:
            # acceptable extreme range: -fmin < loc < fmax and 0 <= scale < fmax
            func(loc=1, scale=numpy.nextafter(fmax, 0), size=size)
            func(loc=numpy.nextafter(fmin, 0), scale=numpy.nextafter(fmax, 0), size=size)
            func(loc=numpy.nextafter(fmax, 0), scale=numpy.nextafter(fmax, 0), size=size)
        except Exception as e:
            raise AssertionError("No error should have been raised, but one was "
                                 "with the following message:\n\n%s" % str(e))


    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("scale",
                             [dpnp.array([3]), numpy.array([3])],
                             ids=['dpnp.array([3])', 'numpy.array([3])'])
    @pytest.mark.parametrize("loc",
                             [[2], dpnp.array([2]), numpy.array([2])],
                             ids=['[2]', 'dpnp.array([2])', 'numpy.array([2])'])
    def test_fallback(self, loc, scale):
        seed = 15
        size = (3, 2, 5)

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).normal(loc=loc, scale=scale, size=size)

        # dpnp accepts only scalar as low and/or high, in other cases it will be a fallback to numpy
        actual = dpnp.asnumpy(data)
        desired = numpy.random.RandomState(seed).normal(loc=loc, scale=scale, size=size)

        precision = numpy.finfo(dtype=numpy.float64).precision
        assert_array_almost_equal(actual, desired, decimal=precision)

        # check if compute follows data isn't broken
        assert_cfd(data, sycl_queue)


    @pytest.mark.parametrize("dtype",
                             [dpnp.float16, dpnp.float, float, dpnp.integer, dpnp.int64, dpnp.int32, dpnp.int, int,
                              dpnp.longcomplex, dpnp.complex128, dpnp.complex64, dpnp.bool, dpnp.bool_],
                             ids=['dpnp.float16', 'dpnp.float', 'float', 'dpnp.integer', 'dpnp.int64', 'dpnp.int32', 'dpnp.int', 'int',
                                  'dpnp.longcomplex', 'dpnp.complex128', 'dpnp.complex64', 'dpnp.bool', 'dpnp.bool_'])
    def test_invalid_dtype(self, dtype):
        # dtype must be float32 or float64
        assert_raises(TypeError, RandomState().normal, dtype=dtype)


    @pytest.mark.parametrize("usm_type",
                             ["", "unknown"],
                             ids=['Empty', 'Unknown'])
    def test_invalid_usm_type(self, usm_type):
        # dtype must be float32 or float64
        assert_raises(ValueError, RandomState().normal, usm_type=usm_type)


class TestRand:
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_distr(self, usm_type):
        seed = 28042
        sycl_queue = dpctl.SyclQueue()

        data = RandomState(seed, sycl_queue=sycl_queue).rand(3, 2, usm_type=usm_type)
        desired = numpy.array([[0.7592552667483687, 0.5937560645397753],
                               [0.257010098779574 , 0.749422621447593 ],
                               [0.6316644293256104, 0.7411410815548152]], dtype=numpy.float64)

        precision = numpy.finfo(dtype=numpy.float64).precision
        assert_array_almost_equal(dpnp.asnumpy(data), desired, decimal=precision)
        assert_cfd(data, sycl_queue, usm_type)

        # call with the same seed has to draw the same values
        data = RandomState(seed, sycl_queue=sycl_queue).rand(3, 2, usm_type=usm_type)
        assert_array_almost_equal(dpnp.asnumpy(data), desired, decimal=precision)
        assert_cfd(data, sycl_queue, usm_type)

        # call with omitted dimensions has to draw the first element from desired
        data = RandomState(seed, sycl_queue=sycl_queue).rand(usm_type=usm_type)
        assert_array_almost_equal(dpnp.asnumpy(data), desired[0, 0], decimal=precision)
        assert_cfd(data, sycl_queue, usm_type)

        # rand() is an alias on random_sample(), map arguments
        with mock.patch('dpnp.random.RandomState.random_sample') as m:
            RandomState(seed).rand(3, 2, usm_type=usm_type)
            m.assert_called_once_with(size=(3, 2),
                                      usm_type=usm_type)


    @pytest.mark.parametrize("dims",
                             [(), (5,), (10, 2, 7), (0,)],
                             ids=['()', '(5,)', '(10, 2, 7)', '(0,)'])
    def test_dims(self, dims):
        size = None if len(dims) == 0 else dims

        with mock.patch('dpnp.random.RandomState.uniform') as m:
            RandomState().rand(*dims)
            m.assert_called_once_with(low=0.0,
                                      high=1.0,
                                      size=size,
                                      dtype=None,
                                      usm_type="device")


    @pytest.mark.parametrize("zero_dims",
                             [(0,), (3, 0), (3, 0, 10), ()],
                             ids=['(0,)', '(3, 0)', '(3, 0, 10)', '()'])
    def test_zero_dims(self, zero_dims):
        assert_equal(RandomState().rand(*zero_dims).shape, zero_dims)


    def test_wrong_dims(self):
        # non-positive number in size
        assert_raises(ValueError, RandomState().rand, 2, -1)

        # non-integer number in size
        assert_raises(TypeError, RandomState().rand, 3.5, 2)


class TestRandInt:
    @pytest.mark.parametrize("dtype",
                             [int, dpnp.int32, dpnp.int],
                             ids=['int', 'dpnp.int32', 'dpnp.int'])
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_distr(self, dtype, usm_type):
        seed = 9864
        low = 1
        high = 10

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).randint(low=low,
                                                                high=high,
                                                                size=(3, 2),
                                                                dtype=dtype,
                                                                usm_type=usm_type)
        desired = numpy.array([[4, 1],
                               [5, 3],
                               [5, 7]], dtype=numpy.int32)
        assert_array_equal(dpnp.asnumpy(data), desired)
        assert_cfd(data, sycl_queue, usm_type)

        # call with the same seed has to draw the same values
        data = RandomState(seed, sycl_queue=sycl_queue).randint(low=low,
                                                                high=high,
                                                                size=(3, 2),
                                                                dtype=dtype,
                                                                usm_type=usm_type)
        assert_array_equal(dpnp.asnumpy(data), desired)
        assert_cfd(data, sycl_queue, usm_type)

        # call with omitted dimensions has to draw the first element from desired
        data = RandomState(seed, sycl_queue=sycl_queue).randint(low=low,
                                                                high=high,
                                                                dtype=dtype,
                                                                usm_type=usm_type)
        assert_array_equal(dpnp.asnumpy(data), desired[0, 0])
        assert_cfd(data, sycl_queue, usm_type)

        # rand() is an alias on random_sample(), map arguments
        with mock.patch('dpnp.random.RandomState.uniform') as m:
            RandomState(seed).randint(low=low, high=high, size=(3, 2), dtype=dtype, usm_type=usm_type)
            m.assert_called_once_with(low=low,
                                      high=high,
                                      size=(3, 2),
                                      dtype=dpnp.int32,
                                      usm_type=usm_type)


    def test_float_bounds(self):
        actual = dpnp.asnumpy(RandomState(365852).randint(low=0.6, high=6.789102534, size=(7,)))
        desired = numpy.array([4, 4, 3, 3, 1, 0, 3], dtype=numpy.int32)
        assert_array_equal(actual, desired)


    def test_negative_bounds(self):
        actual = dpnp.asnumpy(RandomState(5143).randint(low=-15.74, high=-3, size=(2, 7)))
        desired = numpy.array([[-9, -12, -4,  -12, -5, -13, -9],
                               [-4, -6,  -13, -9,  -9,  -6, -15]], dtype=numpy.int32)
        assert_array_equal(actual, desired)


    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_negative_interval(self):
        rs = RandomState(3567)

        assert_equal(-5 <= rs.randint(-5, -1) < -1, True)

        x = rs.randint(-7, -1, 5)
        assert_equal(-7 <= x, True)
        assert_equal(x < -1, True)        


    def test_bounds_checking(self):
        dtype = dpnp.int32
        func = RandomState().randint
        low = numpy.iinfo(dtype).min
        high = numpy.iinfo(dtype).max

        # inf can't be converted to int boundary
        assert_raises(OverflowError, func, -numpy.inf, 0)
        assert_raises(OverflowError, func, 0, numpy.inf)
        assert_raises(OverflowError, func, -numpy.inf, numpy.inf)

        # either low < min_int or high > max_int is prohibeted
        assert_raises(OverflowError, func, low - 1, 0, dtype=dtype)
        assert_raises(OverflowError, func, -1, high + 1, dtype=dtype)
        assert_raises(OverflowError, func, low - 1, high + 1, dtype=dtype)

        # low >= high is prohibeted
        assert_raises(ValueError, func, 1, 1, dtype=dtype)
        assert_raises(ValueError, func, high, low, dtype=dtype)
        assert_raises(ValueError, func, 1, 0, dtype=dtype)
        assert_raises(ValueError, func, high, low/2, dtype=dtype)


    def test_rng_zero_and_extremes(self):
        dtype = dpnp.int32
        func = RandomState().randint
        low = numpy.iinfo(dtype).min
        high = numpy.iinfo(dtype).max

        tgt = high - 1
        assert_equal(func(tgt, tgt + 1, size=1000), tgt)

        tgt = low
        assert_equal(func(tgt, tgt + 1, size=1000), tgt)

        tgt = (low + high)//2
        assert_equal(func(tgt, tgt + 1, size=1000), tgt)


    def test_full_range(self):
        dtype = dpnp.int32
        low = numpy.iinfo(dtype).min
        high = numpy.iinfo(dtype).max

        try:
            RandomState().randint(low, high)
        except Exception as e:
            raise AssertionError("No error should have been raised, but one was "
                                 "with the following message:\n\n%s" % str(e))


    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_in_bounds_fuzz(self):
        for high in [4, 8, 16]:
            vals = RandomState().randint(2, high, size=2**16)
            assert_equal(vals.max() < high, True)
            assert_equal(vals.min() >= 2, True)


    @pytest.mark.parametrize("zero_size",
                             [(3, 0, 4), 0, (0,), ()],
                             ids=['(3, 0, 4)', '0', '(0,)', '()'])
    def test_zero_size(self, zero_size):
        exp_shape = zero_size if zero_size != 0 else (0,)
        assert_equal(RandomState().randint(0, 10, size=zero_size).shape, exp_shape)


    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("high",
                             [dpnp.array([3]), numpy.array([3])],
                             ids=['dpnp.array([3])', 'numpy.array([3])'])
    @pytest.mark.parametrize("low",
                             [[2], dpnp.array([2]), numpy.array([2])],
                             ids=['[2]', 'dpnp.array([2])', 'numpy.array([2])'])
    def test_bounds_fallback(self, low, high):
        seed = 15
        size = (3, 2, 5)

        # dpnp accepts only scalar as low and/or high, in other cases it will be a fallback to numpy
        actual = dpnp.asnumpy(RandomState(seed).randint(low=low, high=high, size=size))
        desired = numpy.random.RandomState(seed).randint(low=low, high=high, size=size)
        assert_equal(actual, desired)


    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("dtype",
                             [dpnp.int64, dpnp.integer, dpnp.bool, dpnp.bool_, bool],
                             ids=['dpnp.int64', 'dpnp.integer', 'dpnp.bool', 'dpnp.bool_', 'bool'])
    def test_dtype_fallback(self, dtype):
        seed = 157
        low = -3 if not dtype in {dpnp.bool_, bool} else 0
        high = 37 if not dtype in {dpnp.bool_, bool} else 2
        size = (3, 2, 5)

        if dtype == dpnp.integer and dtype == dpnp.dtype('int32'):
            pytest.skip("dpnp.integer is alias on dpnp.int32 on the target OS, so no fallback here")

        # dtype must be int or dpnp.int32, in other cases it will be a fallback to numpy
        actual = dpnp.asnumpy(RandomState(seed).randint(low=low, high=high, size=size, dtype=dtype))
        desired = numpy.random.RandomState(seed).randint(low=low, high=high, size=size, dtype=dtype)
        assert_equal(actual, desired)
        assert_raises(TypeError, RandomState().randint, dtype=dtype)


    @pytest.mark.parametrize("usm_type",
                             ["", "unknown"],
                             ids=['Empty', 'Unknown'])
    def test_invalid_usm_type(self, usm_type):
        # dtype must be float32 or float64
        assert_raises(ValueError, RandomState().uniform, usm_type=usm_type)


class TestRandN:
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_distr(self, usm_type):
        seed = 3649

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).randn(3, 2, usm_type=usm_type)
        desired = numpy.array([[-0.862485623762009,  1.169492612490272],
                                [-0.405876118480338,  0.939006537666719],
                                [-0.615075625641019,  0.555260469834381]], dtype=numpy.float64)

        # TODO: discuss with opneMKL: there is a difference between CPU and GPU
        # generated samples since 9 digit while precision=15 for float64
        # precision = numpy.finfo(dtype=numpy.float64).precision
        precision = 8
        assert_array_almost_equal(dpnp.asnumpy(data), desired, decimal=precision)

        # call with the same seed has to draw the same values
        data = RandomState(seed, sycl_queue=sycl_queue).randn(3, 2, usm_type=usm_type)
        assert_array_almost_equal(dpnp.asnumpy(data), desired, decimal=precision)

        # TODO: discuss with oneMKL: return 0.0 instead of the 1st element
        # call with omitted dimensions has to draw the first element from desired
        # actual = dpnp.asnumpy(RandomState(seed).randn(usm_type=usm_type))
        # assert_array_almost_equal(actual, desired[0, 0], decimal=precision)

        # randn() is an alias on standard_normal(), map arguments
        with mock.patch('dpnp.random.RandomState.standard_normal') as m:
            RandomState(seed).randn(2, 7, usm_type=usm_type)
            m.assert_called_once_with(size=(2, 7),
                                      usm_type=usm_type)


    @pytest.mark.parametrize("dims",
                             [(), (5,), (10, 2, 7), (0,)],
                             ids=['()', '(5,)', '(10, 2, 7)', '(0,)'])
    def test_dims(self, dims):
        size = None if len(dims) == 0 else dims

        with mock.patch('dpnp.random.RandomState.normal') as m:
            RandomState().randn(*dims)
            m.assert_called_once_with(loc=0.0,
                                      scale=1.0,
                                      size=size,
                                      dtype=None,
                                      usm_type="device")


    @pytest.mark.parametrize("zero_dims",
                             [(0,), (3, 0), (3, 0, 10), ()],
                             ids=['(0,)', '(3, 0)', '(3, 0, 10)', '()'])
    def test_zero_dims(self, zero_dims):
        assert_equal(RandomState().randn(*zero_dims).shape, zero_dims)


    def test_wrong_dims(self):
        # non-positive number in size
        assert_raises(ValueError, RandomState().randn, 2, -1)

        # non-integer number in size
        assert_raises(TypeError, RandomState().randn, 3.5, 2)


class TestSeed:
    @pytest.mark.parametrize("func",
                             ['normal', 'standard_normal', 'random_sample', 'uniform'],
                             ids=['normal', 'standard_normal', 'random_sample', 'uniform'])
    def test_scalar(self, func):
        seed = 28041997
        size = (3, 2, 4)

        rs = RandomState(seed)
        a1 = dpnp.asnumpy(getattr(rs, func)(size=size))

        rs = RandomState(seed)
        a2 = dpnp.asnumpy(getattr(rs, func)(size=size))

        precision = numpy.finfo(dtype=numpy.float64).precision
        assert_array_almost_equal(a1, a2, decimal=precision)


    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
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


    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
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


class TestStandardNormal:
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_distr(self, usm_type):
        seed = 1234567

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).standard_normal(size=(4, 2), usm_type=usm_type)
        desired = numpy.array([[0.112455902594571, -0.249919829443642],
                               [0.702423540827815,  1.548132130318456],
                               [0.947364919775284, -0.432257289195464],
                               [0.736848611436872,  1.557284323302839]], dtype=numpy.float64)

        # TODO: discuss with opneMKL: there is a difference between CPU and GPU
        # generated samples since 9 digit while precision=15 for float64
        # precision = numpy.finfo(dtype=numpy.float64).precision
        precision = 8
        assert_array_almost_equal(dpnp.asnumpy(data), desired, decimal=precision)

        # TODO: discuss with oneMKL: return 0.0 instead of the 1st element
        # call with omitted dimensions has to draw the first element from desired
        # actual = dpnp.asnumpy(RandomState(seed).standard_normal(usm_type=usm_type))
        # assert_array_almost_equal(actual, desired[0, 0], decimal=precision)

        # random_sample() is an alias on uniform(), map arguments
        with mock.patch('dpnp.random.RandomState.normal') as m:
            RandomState(seed).standard_normal((4, 2), usm_type=usm_type)
            m.assert_called_once_with(loc=0.0,
                                      scale=1.0,
                                      size=(4, 2),
                                      dtype=None,
                                      usm_type=usm_type)


    @pytest.mark.parametrize("size",
                             [(), (5,), (10, 2, 7), (0,)],
                             ids=['()', '(5,)', '(10, 2, 7)', '(0,)'])
    def test_sizes(self, size):
        with mock.patch('dpnp.random.RandomState.normal') as m:
            RandomState().standard_normal(size)
            m.assert_called_once_with(loc=0.0,
                                      scale=1.0,
                                      size=size,
                                      dtype=None,
                                      usm_type="device")


    def test_wrong_dims(self):
        # non-positive number in size
        assert_raises(ValueError, RandomState().standard_normal, (2, -1))

        # non-integer number in size
        assert_raises(TypeError, RandomState().standard_normal, (3.5, 2))


class TestRandSample:
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_distr(self, usm_type):
        seed = 12657

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).random_sample(size=(4, 2), usm_type=usm_type)
        desired = numpy.array([[0.1887628440745175, 0.2763057765550911],
                               [0.3973943444434553, 0.2975987731479108],
                               [0.4144027342554182, 0.2636592474300414],
                               [0.6129623607266694, 0.2596735346596688]], dtype=numpy.float64)
        
        precision = numpy.finfo(dtype=numpy.float64).precision
        assert_array_almost_equal(dpnp.asnumpy(data), desired, decimal=precision)

        # call with omitted dimensions has to draw the first element from desired
        data = RandomState(seed, sycl_queue=sycl_queue).random_sample(usm_type=usm_type)
        assert_array_almost_equal(dpnp.asnumpy(data), desired[0, 0], decimal=precision)

        # random_sample() is an alias on uniform(), map arguments
        with mock.patch('dpnp.random.RandomState.uniform') as m:
            RandomState(seed).random_sample((4, 2), usm_type=usm_type)
            m.assert_called_once_with(low=0.0,
                                      high=1.0,
                                      size=(4, 2),
                                      dtype=None,
                                      usm_type=usm_type)


    @pytest.mark.parametrize("size",
                             [(), (5,), (10, 2, 7), (0,)],
                             ids=['()', '(5,)', '(10, 2, 7)', '(0,)'])
    def test_sizes(self, size):
        with mock.patch('dpnp.random.RandomState.uniform') as m:
            RandomState().random_sample(size)
            m.assert_called_once_with(low=0.0,
                                      high=1.0,
                                      size=size,
                                      dtype=None,
                                      usm_type="device")


    def test_wrong_dims(self):
        # non-positive number in size
        assert_raises(ValueError, RandomState().random_sample, (2, -1))

        # non-integer number in size
        assert_raises(TypeError, RandomState().random_sample, (3.5, 2))


class TestUniform:
    @pytest.mark.parametrize("bounds",
                             [[1.23, 10.54], [10.54, 1.23]],
                             ids=['(low, high)=[1.23, 10.54]', '(low, high)=[10.54, 1.23]'])
    @pytest.mark.parametrize("dtype",
                             [dpnp.float32, dpnp.float64, dpnp.int32, None],
                             ids=['float32', 'float64', 'int32', 'None'])
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_distr(self, bounds, dtype, usm_type):
        seed = 28041997
        low = bounds[0]
        high = bounds[1]

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).uniform(low=low,
                                                                high=high,
                                                                size=(3, 2),
                                                                dtype=dtype,
                                                                usm_type=usm_type)

        if dtype is None:
            # dtype depends on fp64 support by the device
            dtype = dpnp.float64 if data.sycl_device.has_aspect_fp64 else dpnp.float32

        if dtype != dpnp.int32:
            desired = numpy.array([[4.023770128630567, 8.87456423597643 ],
                                   [2.888630247435067, 4.823004481580574],
                                   [2.030351535445079, 4.533497077834326]])
            assert_array_almost_equal(dpnp.asnumpy(data), desired, decimal=numpy.finfo(dtype=dtype).precision)
        else:
            desired = numpy.array([[3, 8],
                                   [2, 4],
                                   [1, 4]])
            assert_array_equal(dpnp.asnumpy(data), desired)

        # check if compute follows data isn't broken
        assert_cfd(data, sycl_queue, usm_type)


    @pytest.mark.parametrize("dtype",
                             [dpnp.float32, dpnp.float64, dpnp.int32, None],
                             ids=['float32', 'float64', 'int32', 'None'])
    @pytest.mark.parametrize("usm_type",
                             ["host", "device", "shared"],
                             ids=['host', 'device', 'shared'])
    def test_low_high_equal(self, dtype, usm_type):
        seed = 28045
        low = high = 3.75
        shape = (7, 6, 20)

        data = RandomState(seed).uniform(low=low, high=high, size=shape, dtype=dtype, usm_type=usm_type)

        if dtype is None:
            # dtype depends on fp64 support by the device
            dtype = dpnp.float64 if data.sycl_device.has_aspect_fp64 else dpnp.float32

        actual = dpnp.asnumpy(data)
        desired = numpy.full(shape=shape, fill_value=low, dtype=dtype)

        if dtype == dpnp.int32:
            assert_array_equal(actual, desired)
        else:
            assert_array_almost_equal(actual, desired, decimal=numpy.finfo(dtype=dtype).precision)


    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_range_bounds(self):
        fmin = numpy.finfo('double').min
        fmax = numpy.finfo('double').max
        func = RandomState().uniform

        assert_raises(OverflowError, func, -numpy.inf, 0)
        assert_raises(OverflowError, func, 0, numpy.inf)
        assert_raises(OverflowError, func, -numpy.inf, numpy.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-numpy.inf], [0])
        assert_raises(OverflowError, func, [0], [numpy.inf])

        func(low=numpy.nextafter(fmin, 0), high=numpy.nextafter(fmax, 0))


    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize("high",
                             [dpnp.array([3]), numpy.array([3])],
                             ids=['dpnp.array([3])', 'numpy.array([3])'])
    @pytest.mark.parametrize("low",
                             [[2], dpnp.array([2]), numpy.array([2])],
                             ids=['[2]', 'dpnp.array([2])', 'numpy.array([2])'])
    def test_fallback(self, low, high):
        seed = 15
        size = (3, 2, 5)

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).uniform(low=low, high=high, size=size)

        # dpnp accepts only scalar as low and/or high, in other cases it will be a fallback to numpy
        actual = dpnp.asnumpy(data)
        desired = numpy.random.RandomState(seed).uniform(low=low, high=high, size=size)

        precision = numpy.finfo(dtype=numpy.float64).precision
        assert_array_almost_equal(actual, desired, decimal=precision)

        # check if compute follows data isn't broken
        assert_cfd(data, sycl_queue)


    @pytest.mark.parametrize("dtype",
                             [dpnp.float16, dpnp.float, float, dpnp.integer, dpnp.int64, dpnp.int, int,
                              dpnp.longcomplex, dpnp.complex128, dpnp.complex64, dpnp.bool, dpnp.bool_],
                             ids=['dpnp.float16', 'dpnp.float', 'float', 'dpnp.integer', 'dpnp.int64', 'dpnp.int', 'int',
                                  'dpnp.longcomplex', 'dpnp.complex128', 'dpnp.complex64', 'dpnp.bool', 'dpnp.bool_'])
    def test_invalid_dtype(self, dtype):
        # dtype must be float32 or float64
        assert_raises(TypeError, RandomState().uniform, dtype=dtype)


    @pytest.mark.parametrize("usm_type",
                             ["", "unknown"],
                             ids=['Empty', 'Unknown'])
    def test_invalid_usm_type(self, usm_type):
        # dtype must be float32 or float64
        assert_raises(ValueError, RandomState().uniform, usm_type=usm_type)
