from unittest import mock

import dpctl
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

import dpnp
from dpnp.dpnp_array import dpnp_array
from dpnp.random import RandomState

from .helper import (
    assert_dtype_allclose,
    get_array,
    is_cpu_device,
    is_gpu_device,
)

# aspects of default device:
_def_device = dpctl.SyclQueue().sycl_device
_def_dev_has_fp64 = _def_device.has_aspect_fp64

list_of_usm_types = ["host", "device", "shared"]


def assert_cfd(data, exp_sycl_queue, exp_usm_type=None):
    assert exp_sycl_queue == data.sycl_queue
    if exp_usm_type:
        assert exp_usm_type == data.usm_type


def get_default_floating():
    if not _def_dev_has_fp64:
        return dpnp.float32
    return dpnp.float64


class TestNormal:
    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64, None])
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    def test_distr(self, dtype, usm_type):
        seed = 1234567
        sycl_queue = dpctl.SyclQueue()
        func = lambda: RandomState(seed, sycl_queue=sycl_queue).normal(
            loc=0.12345, scale=2.71, size=(3, 2), dtype=dtype, usm_type=usm_type
        )

        if dtype is dpnp.float64 and not _def_dev_has_fp64:
            # default device doesn't support 'float64' type
            assert_raises(RuntimeError, func)
            return

        dpnp_data = func()

        # default dtype depends on fp64 support by the device
        dtype = get_default_floating() if dtype is None else dtype
        if sycl_queue.sycl_device.is_cpu:
            expected = numpy.array(
                [
                    [0.428205496031286, -0.55383273779227],
                    [2.027017795643378, 4.318888073163015],
                    [2.69080893259102, -1.047967253719708],
                ],
                dtype=dtype,
            )
        else:
            if dtype == dpnp.float64:
                expected = numpy.array(
                    [
                        [-15.73532523, -11.84163022],
                        [0.548032, 1.41296207],
                        [-2.63250381, 2.77542322],
                    ],
                    dtype=dtype,
                )
            else:
                expected = numpy.array(
                    [
                        [-15.735329, -11.841626],
                        [0.548032, 1.4129621],
                        [-2.6325033, 2.7754242],
                    ],
                    dtype=dtype,
                )
        # TODO: discuss with oneMKL: there is a difference between CPU and GPU
        # generated samples since 9 digit while precision=15 for float64
        # precision = dpnp.finfo(dtype).precision
        precision = 8 if dtype == dpnp.float64 else dpnp.finfo(dtype).precision
        assert_array_almost_equal(dpnp_data, expected, decimal=precision)

        # check if compute follows data isn't broken
        assert_cfd(dpnp_data, sycl_queue, usm_type)

    @pytest.mark.parametrize("dtype", [dpnp.float32, dpnp.float64, None])
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    def test_scale(self, dtype, usm_type):
        mean = dtype(7.0) if dtype == dpnp.float32 else 7.0
        rs = RandomState(39567)
        func = lambda scale: rs.normal(
            loc=mean, scale=scale, dtype=dtype, usm_type=usm_type
        )

        if dtype is dpnp.float64 and not _def_dev_has_fp64:
            # default device doesn't support 'float64' type
            assert_raises(RuntimeError, func, scale=0)
            return

        # zero scale means full ndarray of mean values
        assert_equal(func(scale=0), mean)

        # scale must be non-negative ('-0.0' is negative value)
        assert_raises(ValueError, func, scale=-0.0)
        assert_raises(ValueError, func, scale=-3.71)

    @pytest.mark.parametrize(
        "loc",
        [
            numpy.inf,
            -numpy.inf,
            numpy.nextafter(dpnp.finfo(get_default_floating()).max, 0),
            numpy.nextafter(dpnp.finfo(get_default_floating()).min, 0),
        ],
        ids=[
            "numpy.inf",
            "-numpy.inf",
            "nextafter(max, 0)",
            "nextafter(min, 0)",
        ],
    )
    def test_inf_loc(self, loc):
        a = RandomState(6531).normal(loc=loc, scale=1, size=1000)
        assert_equal(a, get_default_floating()(loc), strict=False)

    def test_inf_scale(self):
        a = RandomState().normal(0, numpy.inf, size=1000)
        assert not dpnp.isnan(a).any()
        assert dpnp.isinf(a).all()
        assert_equal(a.max(), numpy.inf)
        assert_equal(a.min(), -numpy.inf)

    @pytest.mark.parametrize("loc", [numpy.inf, -numpy.inf])
    def test_inf_loc_scale(self, loc):
        a = RandomState().normal(loc=loc, scale=numpy.inf, size=1000)
        assert not dpnp.isnan(a).all()
        assert_equal(dpnp.nanmin(a), loc)
        assert_equal(dpnp.nanmax(a), loc)

    def test_extreme_bounds(self):
        dtype = get_default_floating()
        fmin = dpnp.finfo(dtype).min
        fmax = dpnp.finfo(dtype).max

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
            func(
                loc=numpy.nextafter(fmin, 0),
                scale=numpy.nextafter(fmax, 0),
                size=size,
            )
            func(
                loc=numpy.nextafter(fmax, 0),
                scale=numpy.nextafter(fmax, 0),
                size=size,
            )
        except Exception as e:
            raise AssertionError(
                "No error should have been raised, but one was "
                "with the following message:\n\n%s" % str(e)
            )

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "scale",
        [dpnp.array([3]), numpy.array([3])],
        ids=["dpnp.array([3])", "numpy.array([3])"],
    )
    @pytest.mark.parametrize(
        "loc",
        [[2], dpnp.array([2]), numpy.array([2])],
        ids=["[2]", "dpnp.array([2])", "numpy.array([2])"],
    )
    def test_fallback(self, loc, scale):
        seed = 15
        size = (3, 2, 5)

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).normal(
            loc=loc, scale=scale, size=size
        )

        # dpnp accepts only scalar as low and/or high, in other cases it will be a fallback to numpy
        expected = numpy.random.RandomState(seed).normal(
            loc=get_array(numpy, loc), scale=get_array(numpy, scale), size=size
        )

        precision = dpnp.finfo(get_default_floating()).precision
        assert_array_almost_equal(data, expected, decimal=precision)

        # check if compute follows data isn't broken
        assert_cfd(data, sycl_queue)

    @pytest.mark.parametrize(
        "dtype",
        [
            dpnp.float16,
            float,
            dpnp.int64,
            dpnp.int32,
            dpnp.int_,
            int,
            numpy.clongdouble,
            dpnp.complex128,
            dpnp.complex64,
            dpnp.bool,
            dpnp.bool_,
        ],
        ids=[
            "dpnp.float16",
            "float",
            "dpnp.int64",
            "dpnp.int32",
            "dpnp.int_",
            "int",
            "numpy.clongdouble",
            "dpnp.complex128",
            "dpnp.complex64",
            "dpnp.bool",
            "dpnp.bool_",
        ],
    )
    def test_invalid_dtype(self, dtype):
        # dtype must be float32 or float64
        assert_raises(TypeError, RandomState().normal, dtype=dtype)

    @pytest.mark.parametrize(
        "usm_type", ["", "unknown"], ids=["Empty", "Unknown"]
    )
    def test_invalid_usm_type(self, usm_type):
        # dtype must be float32 or float64
        assert_raises(ValueError, RandomState().normal, usm_type=usm_type)


class TestRand:
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    def test_distr(self, usm_type):
        seed = 28042
        sycl_queue = dpctl.SyclQueue()
        dtype = get_default_floating()

        data = RandomState(seed, sycl_queue=sycl_queue).rand(
            3, 2, usm_type=usm_type
        )
        if sycl_queue.sycl_device.is_cpu:
            expected = numpy.array(
                [
                    [0.7592552667483687, 0.5937560645397753],
                    [0.257010098779574, 0.749422621447593],
                    [0.6316644293256104, 0.7411410815548152],
                ],
                dtype=dtype,
            )
        else:
            expected = numpy.array(
                [
                    [4.864511571334162e-14, 7.333946068708259e-01],
                    [8.679067575689537e-01, 5.627257087965376e-01],
                    [4.413379518222594e-01, 4.334482843514076e-01],
                ],
                dtype=dtype,
            )

        precision = dpnp.finfo(dtype).precision
        assert_array_almost_equal(data, expected, decimal=precision)
        assert_cfd(data, sycl_queue, usm_type)

        # call with the same seed has to draw the same values
        data = RandomState(seed, sycl_queue=sycl_queue).rand(
            3, 2, usm_type=usm_type
        )
        assert_array_almost_equal(data, expected, decimal=precision)
        assert_cfd(data, sycl_queue, usm_type)

        # call with omitted dimensions has to draw the first element from expected
        data = RandomState(seed, sycl_queue=sycl_queue).rand(usm_type=usm_type)
        assert_array_almost_equal(data, expected[0, 0], decimal=precision)
        assert_cfd(data, sycl_queue, usm_type)

        # rand() is an alias on random_sample(), map arguments
        with mock.patch("dpnp.random.RandomState.random_sample") as m:
            RandomState(seed).rand(3, 2, usm_type=usm_type)
            m.assert_called_once_with(size=(3, 2), usm_type=usm_type)

    @pytest.mark.parametrize(
        "dims",
        [(), (5,), (10, 2, 7), (0,)],
        ids=["()", "(5,)", "(10, 2, 7)", "(0,)"],
    )
    def test_dims(self, dims):
        size = None if len(dims) == 0 else dims

        with mock.patch("dpnp.random.RandomState.uniform") as m:
            RandomState().rand(*dims)
            m.assert_called_once_with(
                low=0.0, high=1.0, size=size, dtype=None, usm_type="device"
            )

    @pytest.mark.parametrize(
        "zero_dims",
        [(0,), (3, 0), (3, 0, 10), ()],
        ids=["(0,)", "(3, 0)", "(3, 0, 10)", "()"],
    )
    def test_zero_dims(self, zero_dims):
        assert_equal(RandomState().rand(*zero_dims).shape, zero_dims)

    def test_wrong_dims(self):
        # non-positive number in size
        assert_raises(ValueError, RandomState().rand, 2, -1)

        # non-integer number in size
        assert_raises(TypeError, RandomState().rand, 3.5, 2)


class TestRandInt:
    @pytest.mark.parametrize(
        "dtype",
        [int, dpnp.int32, dpnp.int_],
        ids=["int", "dpnp.int32", "dpnp.int_"],
    )
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    def test_distr(self, dtype, usm_type):
        seed = 9864
        low = 1
        high = 10

        if dtype == dpnp.int_ and dtype != dpnp.dtype("int32"):
            pytest.skip(
                "dtype isn't alias on dpnp.int32 on the target OS, so there will be a fallback"
            )

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).randint(
            low=low, high=high, size=(3, 2), dtype=dtype, usm_type=usm_type
        )
        if sycl_queue.sycl_device.is_cpu:
            expected = numpy.array([[4, 1], [5, 3], [5, 7]], dtype=numpy.int32)
        else:
            expected = numpy.array([[1, 2], [1, 5], [3, 7]], dtype=numpy.int32)
        assert_array_equal(data, expected)
        assert_cfd(data, sycl_queue, usm_type)

        # call with the same seed has to draw the same values
        data = RandomState(seed, sycl_queue=sycl_queue).randint(
            low=low, high=high, size=(3, 2), dtype=dtype, usm_type=usm_type
        )
        assert_array_equal(data, expected)
        assert_cfd(data, sycl_queue, usm_type)

        # call with omitted dimensions has to draw the first element from expected
        data = RandomState(seed, sycl_queue=sycl_queue).randint(
            low=low, high=high, dtype=dtype, usm_type=usm_type
        )
        assert_array_equal(data, expected[0, 0])
        assert_cfd(data, sycl_queue, usm_type)

        # rand() is an alias on random_sample(), map arguments
        with mock.patch("dpnp.random.RandomState.uniform") as m:
            RandomState(seed).randint(
                low=low, high=high, size=(3, 2), dtype=dtype, usm_type=usm_type
            )
            m.assert_called_once_with(
                low=low,
                high=high,
                size=(3, 2),
                dtype=dpnp.int32,
                usm_type=usm_type,
            )

    def test_float_bounds(self):
        actual = RandomState(365852).randint(
            low=0.6, high=6.789102534, size=(7,)
        )
        if actual.sycl_device.is_cpu:
            expected = numpy.array([4, 4, 3, 3, 1, 0, 3], dtype=numpy.int32)
        else:
            expected = numpy.array([0, 1, 4, 0, 3, 3, 3], dtype=numpy.int32)
        assert_array_equal(actual, expected)

    def test_negative_bounds(self):
        actual = RandomState(5143).randint(low=-15.74, high=-3, size=(2, 7))
        if actual.sycl_device.is_cpu:
            expected = numpy.array(
                [
                    [-9, -12, -4, -12, -5, -13, -9],
                    [-4, -6, -13, -9, -9, -6, -15],
                ],
                dtype=numpy.int32,
            )
        else:
            expected = numpy.array(
                [
                    [-15, -7, -12, -5, -10, -11, -11],
                    [-14, -7, -7, -10, -14, -9, -6],
                ],
                dtype=numpy.int32,
            )
        assert_array_equal(actual, expected)

    def test_negative_interval(self):
        rs = RandomState(3567)

        assert -5 <= rs.randint(-5, -1) < -1

        x = rs.randint(-7, -1, 5)
        assert_equal(-7 <= x, True, strict=False)
        assert_equal(x < -1, True, strict=False)

    def test_bounds_checking(self):
        dtype = dpnp.int32
        func = RandomState().randint
        low = dpnp.iinfo(dtype).min
        high = dpnp.iinfo(dtype).max

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
        assert_raises(ValueError, func, high, low / 2, dtype=dtype)

    def test_rng_zero_and_extremes(self):
        dtype = dpnp.int32
        func = RandomState().randint
        low = dpnp.iinfo(dtype).min
        high = dpnp.iinfo(dtype).max

        sycl_device = dpctl.SyclQueue().sycl_device
        if sycl_device.has_aspect_gpu and not sycl_device.has_aspect_fp64:
            # TODO: discuss with oneMKL
            pytest.skip(
                f"Due to some reason, oneMKL wrongly returns high value instead of low"
            )

        tgt = high - 1
        assert_equal(func(tgt, tgt + 1, size=1000), tgt, strict=False)

        tgt = low
        assert_equal(func(tgt, tgt + 1, size=1000), tgt, strict=False)

        tgt = (low + high) // 2
        assert_equal(func(tgt, tgt + 1, size=1000), tgt, strict=False)

    def test_full_range(self):
        dtype = dpnp.int32
        low = dpnp.iinfo(dtype).min
        high = dpnp.iinfo(dtype).max

        try:
            RandomState().randint(low, high)
        except Exception as e:
            raise AssertionError(
                "No error should have been raised, but one was "
                "with the following message:\n\n%s" % str(e)
            )

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_in_bounds_fuzz(self):
        for high in [4, 8, 16]:
            vals = RandomState().randint(2, high, size=2**16)
            assert vals.max() < high
            assert vals.min() >= 2

    @pytest.mark.parametrize(
        "zero_size",
        [(3, 0, 4), 0, (0,), ()],
        ids=["(3, 0, 4)", "0", "(0,)", "()"],
    )
    def test_zero_size(self, zero_size):
        exp_shape = zero_size if zero_size != 0 else (0,)
        assert_equal(
            RandomState().randint(0, 10, size=zero_size).shape, exp_shape
        )

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "high",
        [dpnp.array([3]), numpy.array([3])],
        ids=["dpnp.array([3])", "numpy.array([3])"],
    )
    @pytest.mark.parametrize(
        "low",
        [[2], dpnp.array([2]), numpy.array([2])],
        ids=["[2]", "dpnp.array([2])", "numpy.array([2])"],
    )
    def test_bounds_fallback(self, low, high):
        if isinstance(high, dpnp_array) and high.dtype == numpy.int32:
            pytest.skip("NumPy fails: 'high' is out of bounds for int32")

        seed = 15
        size = (3, 2, 5)

        # dpnp accepts only scalar as low and/or high, in other cases it will be a fallback to numpy
        actual = RandomState(seed).randint(low=low, high=high, size=size)
        expected = numpy.random.RandomState(seed).randint(
            low=get_array(numpy, low), high=get_array(numpy, high), size=size
        )
        assert_equal(actual, expected)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "dtype",
        [dpnp.int64, dpnp.int_, dpnp.bool, dpnp.bool_, bool],
        ids=[
            "dpnp.int64",
            "dpnp.int_",
            "dpnp.bool",
            "dpnp.bool_",
            "bool",
        ],
    )
    def test_dtype_fallback(self, dtype):
        seed = 157
        low = -3 if not dtype in {dpnp.bool_, bool} else 0
        high = 37 if not dtype in {dpnp.bool_, bool} else 2
        size = (3, 2, 5)

        if dtype == dpnp.int_ and dtype == dpnp.dtype("int32"):
            pytest.skip(
                "dtype is alias on dpnp.int32 on the target OS, so no fallback here"
            )

        # dtype must be int or dpnp.int32, in other cases it will be a fallback to numpy
        actual = RandomState(seed).randint(
            low=low, high=high, size=size, dtype=dtype
        )
        expected = numpy.random.RandomState(seed).randint(
            low=low, high=high, size=size, dtype=dtype
        )
        assert_array_equal(actual, expected)
        assert_raises(TypeError, RandomState().randint, dtype=dtype)

    @pytest.mark.parametrize(
        "usm_type", ["", "unknown"], ids=["Empty", "Unknown"]
    )
    def test_invalid_usm_type(self, usm_type):
        # dtype must be float32 or float64
        assert_raises(ValueError, RandomState().uniform, usm_type=usm_type)


class TestRandN:
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    def test_distr(self, usm_type):
        seed = 3649
        sycl_queue = dpctl.SyclQueue()
        dtype = get_default_floating()

        data = RandomState(seed, sycl_queue=sycl_queue).randn(
            3, 2, usm_type=usm_type
        )
        if sycl_queue.sycl_device.is_cpu:
            expected = numpy.array(
                [
                    [-0.862485623762009, 1.169492612490272],
                    [-0.405876118480338, 0.939006537666719],
                    [-0.615075625641019, 0.555260469834381],
                ],
                dtype=dtype,
            )
        else:
            expected = numpy.array(
                [
                    [-4.019566117504177, 7.016412093100934],
                    [-1.044015254820266, -0.839721616192757],
                    [0.545079768980527, 0.380676324099473],
                ],
                dtype=dtype,
            )

        # TODO: discuss with oneMKL: there is a difference between CPU and GPU
        # generated samples since 9 digit while precision=15 for float64
        # precision = dpnp.finfo(numpy.float64).precision
        precision = dpnp.finfo(numpy.float32).precision
        assert_array_almost_equal(data, expected, decimal=precision)

        # call with the same seed has to draw the same values
        data = RandomState(seed, sycl_queue=sycl_queue).randn(
            3, 2, usm_type=usm_type
        )
        assert_array_almost_equal(data, expected, decimal=precision)

        # call with omitted dimensions has to draw the first element from expected
        actual = RandomState(seed).randn(usm_type=usm_type)
        assert_array_almost_equal(actual, expected[0, 0], decimal=precision)

        # randn() is an alias on standard_normal(), map arguments
        with mock.patch("dpnp.random.RandomState.standard_normal") as m:
            RandomState(seed).randn(2, 7, usm_type=usm_type)
            m.assert_called_once_with(size=(2, 7), usm_type=usm_type)

    @pytest.mark.parametrize(
        "dims",
        [(), (5,), (10, 2, 7), (0,)],
        ids=["()", "(5,)", "(10, 2, 7)", "(0,)"],
    )
    def test_dims(self, dims):
        size = None if len(dims) == 0 else dims

        with mock.patch("dpnp.random.RandomState.normal") as m:
            RandomState().randn(*dims)
            m.assert_called_once_with(
                loc=0.0, scale=1.0, size=size, dtype=None, usm_type="device"
            )

    @pytest.mark.parametrize(
        "zero_dims",
        [(0,), (3, 0), (3, 0, 10), ()],
        ids=["(0,)", "(3, 0)", "(3, 0, 10)", "()"],
    )
    def test_zero_dims(self, zero_dims):
        assert_equal(RandomState().randn(*zero_dims).shape, zero_dims)

    def test_wrong_dims(self):
        # non-positive number in size
        assert_raises(ValueError, RandomState().randn, 2, -1)

        # non-integer number in size
        assert_raises(TypeError, RandomState().randn, 3.5, 2)


class TestSeed:
    @pytest.mark.parametrize(
        "func", ["normal", "standard_normal", "random_sample", "uniform"]
    )
    def test_scalar(self, func):
        seed = 28041997
        size = (3, 2, 4)

        rs = RandomState(seed)
        a1 = getattr(rs, func)(size=size)

        rs = RandomState(seed)
        a2 = getattr(rs, func)(size=size)

        precision = dpnp.finfo(numpy.float64).precision
        assert_array_almost_equal(a1, a2, decimal=precision)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "seed",
        [
            range(3),
            numpy.arange(3, dtype=numpy.int32),
            dpnp.arange(3, dtype=numpy.int32),
            [0],
            [4294967295],
            [2, 7, 15],
            (1,),
            (85, 6, 17),
        ],
        ids=[
            "range(2)",
            "numpy.arange(2)",
            "dpnp.arange(2)",
            "[0]",
            "[4294967295]",
            "[2, 7, 15]",
            "(1,)",
            "(85, 6, 17)",
        ],
    )
    def test_array_range(self, seed):
        if is_gpu_device():
            pytest.skip("seed as a scalar is only supported on GPU")

        size = 15
        a1 = RandomState(seed).uniform(size=size)
        a2 = RandomState(seed).uniform(size=size)
        assert dpnp.allclose(a1, a2)

    @pytest.mark.parametrize(
        "seed",
        [
            0.5,
            -1.5,
            [-0.3],
            (1.7, 3),
            "text",
            numpy.arange(0, 1, 0.5),
            dpnp.arange(3, dtype=numpy.float32),
            dpnp.arange(3, dtype=numpy.complex64),
        ],
        ids=[
            "0.5",
            "-1.5",
            "[-0.3]",
            "(1.7, 3)",
            "text",
            "numpy.arange(0, 1, 0.5)",
            "dpnp.arange(3, dtype=numpy.float32)",
            "dpnp.arange(3, dtype=numpy.complex64)",
        ],
    )
    def test_invalid_type(self, seed):
        # seed must be an unsigned 32-bit integer
        assert_raises(TypeError, RandomState, seed)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "seed",
        [
            -1,
            [-3, 7],
            (17, 3, -5),
            [4, 3, 2, 1],
            (7, 6, 5, 1),
            range(-1, -11, -1),
            numpy.arange(4, dtype=numpy.int32),
            dpnp.arange(-3, 3, dtype=numpy.int32),
            dpnp.iinfo(numpy.uint32).max + 1,
            (1, 7, dpnp.iinfo(numpy.uint32).max + 1),
        ],
        ids=[
            "-1",
            "[-3, 7]",
            "(17, 3, -5)",
            "[4, 3, 2, 1]",
            "(7, 6, 5, 1)",
            "range(-1, -11, -1)",
            "numpy.arange(4, dtype=numpy.int32)",
            "dpnp.arange(-3, 3, dtype=numpy.int32)",
            "dpnp.iinfo(numpy.uint32).max + 1",
            "(1, 7, dpnp.iinfo(numpy.uint32).max + 1)",
        ],
    )
    def test_invalid_value(self, seed):
        if is_cpu_device():
            # seed must be an unsigned 32-bit integer
            assert_raises(ValueError, RandomState, seed)
        else:
            if dpnp.isscalar(seed):
                # seed must be an unsigned 64-bit integer
                assert_raises(ValueError, RandomState, seed)
            else:
                # seed must be a scalar
                assert_raises(TypeError, RandomState, seed)

    @pytest.mark.parametrize(
        "seed",
        [
            [],
            (),
            [[1, 2, 3]],
            [[1, 2, 3], [4, 5, 6]],
            numpy.array([], dtype=numpy.int64),
            dpnp.array([], dtype=numpy.int64),
        ],
        ids=[
            "[]",
            "()",
            "[[1, 2, 3]]",
            "[[1, 2, 3], [4, 5, 6]]",
            "numpy.array([], dtype=numpy.int64)",
            "dpnp.array([], dtype=numpy.int64)",
        ],
    )
    def test_invalid_shape(self, seed):
        if is_cpu_device():
            # seed must be an unsigned or 1-D array
            assert_raises(ValueError, RandomState, seed)
        else:
            if dpnp.isscalar(seed):
                # seed must be an unsigned 64-bit scalar
                assert_raises(ValueError, RandomState, seed)
            else:
                # seed must be a scalar
                assert_raises(TypeError, RandomState, seed)


class TestStandardNormal:
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    def test_distr(self, usm_type):
        seed = 1234567
        sycl_queue = dpctl.SyclQueue()
        dtype = get_default_floating()

        data = RandomState(seed, sycl_queue=sycl_queue).standard_normal(
            size=(4, 2), usm_type=usm_type
        )
        if sycl_queue.sycl_device.is_cpu:
            expected = numpy.array(
                [
                    [0.112455902594571, -0.249919829443642],
                    [0.702423540827815, 1.548132130318456],
                    [0.947364919775284, -0.432257289195464],
                    [0.736848611436872, 1.557284323302839],
                ],
                dtype=dtype,
            )
        else:
            expected = numpy.array(
                [
                    [-5.851946579836138, -4.415158753007455],
                    [0.156672323326223, 0.475834711471613],
                    [-1.016957125278234, 0.978587902851975],
                    [-0.295425067084912, 1.438622345507964],
                ],
                dtype=dtype,
            )

        # TODO: discuss with oneMKL: there is a difference between CPU and GPU
        # generated samples since 9 digit while precision=15 for float64
        # precision = dpnp.finfo(numpy.float64).precision
        precision = dpnp.finfo(numpy.float32).precision
        assert_array_almost_equal(data, expected, decimal=precision)

        # call with the same seed has to draw the same values
        data = RandomState(seed, sycl_queue=sycl_queue).standard_normal(
            size=(4, 2), usm_type=usm_type
        )
        assert_array_almost_equal(data, expected, decimal=precision)

        # call with omitted dimensions has to draw the first element from expected
        actual = RandomState(seed).standard_normal(usm_type=usm_type)
        assert_array_almost_equal(actual, expected[0, 0], decimal=precision)

        # random_sample() is an alias on uniform(), map arguments
        with mock.patch("dpnp.random.RandomState.normal") as m:
            RandomState(seed).standard_normal((4, 2), usm_type=usm_type)
            m.assert_called_once_with(
                loc=0.0, scale=1.0, size=(4, 2), dtype=None, usm_type=usm_type
            )

    @pytest.mark.parametrize(
        "size",
        [(), (5,), (10, 2, 7), (0,)],
        ids=["()", "(5,)", "(10, 2, 7)", "(0,)"],
    )
    def test_sizes(self, size):
        with mock.patch("dpnp.random.RandomState.normal") as m:
            RandomState().standard_normal(size)
            m.assert_called_once_with(
                loc=0.0, scale=1.0, size=size, dtype=None, usm_type="device"
            )

    def test_wrong_dims(self):
        # non-positive number in size
        assert_raises(ValueError, RandomState().standard_normal, (2, -1))

        # non-integer number in size
        assert_raises(TypeError, RandomState().standard_normal, (3.5, 2))


class TestRandSample:
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    def test_distr(self, usm_type):
        seed = 12657
        sycl_queue = dpctl.SyclQueue()
        dtype = get_default_floating()

        data = RandomState(seed, sycl_queue=sycl_queue).random_sample(
            size=(4, 2), usm_type=usm_type
        )
        if sycl_queue.sycl_device.is_cpu:
            expected = numpy.array(
                [
                    [0.1887628440745175, 0.2763057765550911],
                    [0.3973943444434553, 0.2975987731479108],
                    [0.4144027342554182, 0.2636592474300414],
                    [0.6129623607266694, 0.2596735346596688],
                ],
                dtype=dtype,
            )
        else:
            expected = numpy.array(
                [
                    [0.219563950354e-13, 0.6500454867400344],
                    [0.8847833902913576, 0.9030532521302965],
                    [0.2943803743033427, 0.2688879158061396],
                    [0.2730219631925900, 0.8695396883048091],
                ],
                dtype=dtype,
            )

        precision = dpnp.finfo(dtype).precision
        assert_array_almost_equal(data, expected, decimal=precision)

        # call with omitted dimensions has to draw the first element from expected
        data = RandomState(seed, sycl_queue=sycl_queue).random_sample(
            usm_type=usm_type
        )
        assert_array_almost_equal(data, expected[0, 0], decimal=precision)
        # random_sample() is an alias on uniform(), map arguments
        with mock.patch("dpnp.random.RandomState.uniform") as m:
            RandomState(seed).random_sample((4, 2), usm_type=usm_type)
            m.assert_called_once_with(
                low=0.0, high=1.0, size=(4, 2), dtype=None, usm_type=usm_type
            )

    @pytest.mark.parametrize(
        "size",
        [(), (5,), (10, 2, 7), (0,)],
        ids=["()", "(5,)", "(10, 2, 7)", "(0,)"],
    )
    def test_sizes(self, size):
        with mock.patch("dpnp.random.RandomState.uniform") as m:
            RandomState().random_sample(size)
            m.assert_called_once_with(
                low=0.0, high=1.0, size=size, dtype=None, usm_type="device"
            )

    def test_wrong_dims(self):
        # non-positive number in size
        assert_raises(ValueError, RandomState().random_sample, (2, -1))

        # non-integer number in size
        assert_raises(TypeError, RandomState().random_sample, (3.5, 2))


class TestUniform:
    @pytest.mark.parametrize(
        "bounds",
        [[1.23, 10.54], [10.54, 1.23]],
        ids=["(low, high)=[1.23, 10.54]", "(low, high)=[10.54, 1.23]"],
    )
    @pytest.mark.parametrize(
        "dtype", [dpnp.float32, dpnp.float64, dpnp.int32, None]
    )
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    def test_distr(self, bounds, dtype, usm_type):
        seed = 28041997
        low = bounds[0]
        high = bounds[1]

        sycl_queue = dpctl.SyclQueue()
        func = lambda: RandomState(seed, sycl_queue=sycl_queue).uniform(
            low=low, high=high, size=(3, 2), dtype=dtype, usm_type=usm_type
        )

        if dtype is dpnp.float64 and not _def_dev_has_fp64:
            # default device doesn't support 'float64' type
            assert_raises(RuntimeError, func)
            return

        # get drawn samples by dpnp
        actual = func()

        # default dtype depends on fp64 support by the device
        dtype = get_default_floating() if dtype is None else dtype
        if sycl_queue.sycl_device.is_cpu:
            if dtype != dpnp.int32:
                data = [
                    [4.023770128630567, 8.87456423597643],
                    [2.888630247435067, 4.823004481580574],
                    [2.030351535445079, 4.533497077834326],
                ]
                expected = numpy.array(data, dtype=dtype)
                precision = dpnp.finfo(dtype).precision
                assert_array_almost_equal(actual, expected, decimal=precision)
            else:
                expected = numpy.array([[3, 8], [2, 4], [1, 4]], dtype=dtype)
                assert_array_equal(actual, expected)
        else:
            if dtype != dpnp.int32:
                data = [
                    [1.230000000452886, 4.889115418092382],
                    [6.084098950993071, 1.682066500463302],
                    [3.316473517549554, 8.428297791221597],
                ]
                expected = numpy.array(data, dtype=dtype)
                precision = dpnp.finfo(dtype).precision
                assert_array_almost_equal(actual, expected, decimal=precision)
            else:
                expected = numpy.array([[1, 4], [5, 1], [3, 7]], dtype=dtype)
                assert_array_equal(actual, expected)

        # check if compute follows data isn't broken
        assert_cfd(actual, sycl_queue, usm_type)

    @pytest.mark.parametrize(
        "dtype", [dpnp.float32, dpnp.float64, dpnp.int32, None]
    )
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    def test_low_high_equal(self, dtype, usm_type):
        seed = 28045
        low = high = 3.75
        shape = (7, 6, 20)

        func = lambda: RandomState(seed).uniform(
            low=low, high=high, size=shape, dtype=dtype, usm_type=usm_type
        )

        if dtype is dpnp.float64 and not _def_dev_has_fp64:
            # default device doesn't support 'float64' type
            assert_raises(RuntimeError, func)
            return

        # get drawn samples by dpnp
        actual = func()

        # default dtype depends on fp64 support by the device
        dtype = get_default_floating() if dtype is None else dtype
        expected = numpy.full(shape=shape, fill_value=low, dtype=dtype)

        assert_dtype_allclose(actual, expected)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    def test_range_bounds(self):
        fmin = dpnp.finfo("double").min
        fmax = dpnp.finfo("double").max
        func = RandomState().uniform

        assert_raises(OverflowError, func, -numpy.inf, 0)
        assert_raises(OverflowError, func, 0, numpy.inf)
        assert_raises(OverflowError, func, -numpy.inf, numpy.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-numpy.inf], [0])
        assert_raises(OverflowError, func, [0], [numpy.inf])

        func(low=numpy.nextafter(fmin, 0), high=numpy.nextafter(fmax, 0))

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "high",
        [dpnp.array([3]), numpy.array([3])],
        ids=["dpnp.array([3])", "numpy.array([3])"],
    )
    @pytest.mark.parametrize(
        "low",
        [[2], dpnp.array([2]), numpy.array([2])],
        ids=["[2]", "dpnp.array([2])", "numpy.array([2])"],
    )
    def test_fallback(self, low, high):
        seed = 15
        size = (3, 2, 5)

        sycl_queue = dpctl.SyclQueue()
        data = RandomState(seed, sycl_queue=sycl_queue).uniform(
            low=low, high=high, size=size
        )

        # dpnp accepts only scalar as low and/or high, in other cases it will be a fallback to numpy
        expected = numpy.random.RandomState(seed).uniform(
            low=get_array(numpy, low), high=get_array(numpy, high), size=size
        )

        precision = dpnp.finfo(get_default_floating()).precision
        assert_array_almost_equal(data, expected, decimal=precision)

        # check if compute follows data isn't broken
        assert_cfd(data, sycl_queue)

    @pytest.mark.parametrize(
        "dtype",
        [
            dpnp.float16,
            float,
            dpnp.int64,
            dpnp.int_,
            int,
            numpy.clongdouble,
            dpnp.complex128,
            dpnp.complex64,
            dpnp.bool,
            dpnp.bool_,
        ],
        ids=[
            "dpnp.float16",
            "float",
            "dpnp.int64",
            "dpnp.int_",
            "int",
            "numpy.clongdouble",
            "dpnp.complex128",
            "dpnp.complex64",
            "dpnp.bool",
            "dpnp.bool_",
        ],
    )
    def test_invalid_dtype(self, dtype):
        if dtype == dpnp.int_ and dtype == dpnp.dtype("int32"):
            pytest.skip(
                "dtype is alias on dpnp.int32 on the target OS, so no error here"
            )

        # dtype must be int32, float32 or float64
        assert_raises(TypeError, RandomState().uniform, dtype=dtype)

    @pytest.mark.parametrize(
        "usm_type", ["", "unknown"], ids=["Empty", "Unknown"]
    )
    def test_invalid_usm_type(self, usm_type):
        # dtype must be float32 or float64
        assert_raises(ValueError, RandomState().uniform, usm_type=usm_type)
