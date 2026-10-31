import warnings

import dpctl
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

import dpnp
from dpnp.random import RandomState

from .helper import get_all_dtypes, get_float_dtypes
from .test_random_generator import _assert_same_error

float_dtypes = get_float_dtypes(no_float16=True)
list_of_usm_types = ["host", "device", "shared"]

# method name, positional parameters, whether the samples are floats
distributions = [
    ("beta", (0.5, 2.0), True),
    ("binomial", (10, 0.3), False),
    ("chisquare", (3.0,), True),
    ("exponential", (2.0,), True),
    ("f", (5.0, 7.0), True),
    ("gamma", (3.0, 2.0), True),
    ("geometric", (0.3,), False),
    ("gumbel", (1.0, 2.0), True),
    ("hypergeometric", (10, 5, 7), False),
    ("laplace", (1.0, 2.0), True),
    ("logistic", (1.0, 2.0), True),
    ("lognormal", (0.0, 0.5), True),
    ("logseries", (0.6,), False),
    ("negative_binomial", (3.0, 0.4), False),
    ("noncentral_chisquare", (3.0, 2.0), True),
    ("noncentral_f", (5.0, 7.0, 2.0), True),
    ("normal", (1.0, 2.0), True),
    ("pareto", (5.0,), True),
    ("poisson", (4.0,), False),
    ("power", (3.0,), True),
    ("rayleigh", (2.0,), True),
    ("standard_cauchy", (), True),
    ("standard_exponential", (), True),
    ("standard_gamma", (0.5,), True),
    ("standard_normal", (), True),
    ("standard_t", (5.0,), True),
    ("triangular", (0.0, 1.0, 3.0), True),
    ("uniform", (-1.0, 2.0), True),
    ("vonmises", (0.5, 2.0), True),
    ("wald", (1.0, 2.0), True),
    ("weibull", (2.0,), True),
    ("zipf", (3.0,), False),
    ("random_sample", (), True),
    ("randint", (-3, 7), False),
    ("tomaxint", (), False),
]
float_distributions = [d[:2] for d in distributions if d[2]]


def _all_devices():
    return [
        dev for dev in dpctl.get_devices() if not dev.has_aspect_accelerator
    ]


class TestAPI:
    def test_methods(self):
        public = lambda c: {n for n in dir(c) if not n.startswith("_")}
        missing = public(numpy.random.RandomState) - public(RandomState)
        assert not missing

    def test_module_functions(self):
        # NumPy's other bit generators are not provided
        bit_generators = {"MT19937", "PCG64", "Philox", "SFC64"}
        missing = set(numpy.random.__all__) - set(dpnp.random.__all__)
        assert missing == bit_generators


class TestSeed:
    @pytest.mark.parametrize(
        "seed",
        [
            0,
            4294967295,
            numpy.uint32(5),
            [0],
            [4294967295],
            [2, 7, 15],
            [4, 3, 2, 1],
            (85, 6, 17),
            range(3),
            numpy.arange(3, dtype=numpy.int32),
            dpnp.arange(3, dtype=dpnp.int32),
        ],
    )
    def test_reproducible(self, seed):
        a1 = RandomState(seed).random_sample(15)
        a2 = RandomState(seed).random_sample(15)
        assert_array_equal(a1, a2)

    def test_distinct(self):
        a1 = RandomState(1).random_sample(8)
        a2 = RandomState(2).random_sample(8)
        assert not (a1 == a2).any()

    def test_none(self):
        a1 = RandomState().random_sample(8)
        a2 = RandomState(None).random_sample(8)
        assert not (a1 == a2).any()

    @pytest.mark.parametrize(
        "seed",
        [
            -1,
            4294967296,
            [-3, 7],
            (1, 7, 4294967296),
            [],
            (),
            [[1, 2, 3]],
            [[1, 2, 3], [4, 5, 6]],
            numpy.array([], dtype=numpy.int64),
            0.5,
            -1.5,
            [-0.3],
            (1.7, 3),
            "text",
            numpy.arange(0, 1, 0.5),
        ],
    )
    def test_invalid(self, seed):
        _assert_same_error(
            lambda: numpy.random.RandomState(seed),
            lambda: RandomState(seed),
        )

    @pytest.mark.parametrize(
        "seed",
        [
            dpnp.arange(3, dtype=dpnp.float32),
            dpnp.array([], dtype=dpnp.int64),
            dpnp.arange(-3, 3),
        ],
    )
    def test_invalid_dpnp_array(self, seed):
        np_seed = dpnp.asnumpy(seed)
        _assert_same_error(
            lambda: numpy.random.RandomState(np_seed),
            lambda: RandomState(seed),
        )

    def test_reseed(self):
        rs = RandomState(3)
        rs.random_sample(5)
        rs.seed(7)
        assert_array_equal(rs.random_sample(5), RandomState(7).random_sample(5))

    def test_module_seed(self):
        dpnp.random.seed(7)
        a1 = dpnp.random.random_sample(5)
        a2 = dpnp.random.random_sample(5)
        dpnp.random.seed(7)
        assert_array_equal(dpnp.random.random_sample(5), a1)
        assert_array_equal(dpnp.random.random_sample(5), a2)
        assert_array_equal(a1, RandomState(7).random_sample(5))


class TestState:
    def test_roundtrip(self):
        rs = RandomState(12)
        rs.normal(size=3)
        state = rs.get_state()
        a1 = rs.normal(size=7)
        rs.random_sample(3)
        rs.set_state(state)
        assert_array_equal(rs.normal(size=7), a1)

    def test_legacy_false(self):
        rs = RandomState(12)
        assert rs.get_state(legacy=False) == rs.get_state()

    def test_restore_on_new_instance(self):
        rs = RandomState(12)
        rs.poisson(3.0, size=10)
        state = rs.get_state()
        a1 = rs.poisson(3.0, size=10)
        rs2 = RandomState()
        rs2.set_state(state)
        assert_array_equal(rs2.poisson(3.0, size=10), a1)

    def test_module_state(self):
        dpnp.random.seed(5)
        state = dpnp.random.get_state()
        a1 = dpnp.random.random_sample(5)
        dpnp.random.set_state(state)
        assert_array_equal(dpnp.random.random_sample(5), a1)

    def test_numpy_state(self):
        state = numpy.random.RandomState(3).get_state()
        assert_raises(ValueError, RandomState().set_state, state)

    def test_not_a_state(self):
        assert_raises(ValueError, RandomState().set_state, 1)



@pytest.fixture
def restore_bit_generator():
    saved = dpnp.random.get_bit_generator()
    yield
    dpnp.random.set_bit_generator(saved)


@pytest.mark.usefixtures("restore_bit_generator")
class TestBitGenerator:
    def test_get_same_object(self):
        bg = dpnp.random.get_bit_generator()
        assert isinstance(bg, dpnp.random.Philox4x3210)
        assert dpnp.random.get_bit_generator() is bg

    def test_module_seed_keeps_object(self):
        bg = dpnp.random.get_bit_generator()
        dpnp.random.seed(3)
        assert dpnp.random.get_bit_generator() is bg
        assert_array_equal(
            dpnp.random.random_sample(5), RandomState(3).random_sample(5)
        )

    def test_set(self):
        bg = dpnp.random.PCG64DXSM(5)
        dpnp.random.set_bit_generator(bg)
        assert dpnp.random.get_bit_generator() is bg
        ref = dpnp.random.Generator(dpnp.random.PCG64DXSM(5)).random(6)
        assert_array_equal(dpnp.random.random_sample(6), ref)

    def test_shared_stream(self):
        bg = dpnp.random.PCG64DXSM(5)
        dpnp.random.set_bit_generator(bg)
        a = dpnp.random.random_sample(3)
        b = dpnp.random.Generator(bg).random(3)
        ref = dpnp.random.Generator(dpnp.random.PCG64DXSM(5)).random(6)
        assert_array_equal(dpnp.concatenate((a, b)), ref)

    def test_module_seed_reseeds_in_place(self):
        bg = dpnp.random.PCG64DXSM(5)
        dpnp.random.set_bit_generator(bg)
        dpnp.random.random_sample(3)
        dpnp.random.seed(9)
        assert dpnp.random.get_bit_generator() is bg
        assert bg.state == dpnp.random.PCG64DXSM(9).state
        ref = dpnp.random.Generator(dpnp.random.PCG64DXSM(9)).random(4)
        assert_array_equal(dpnp.random.random_sample(4), ref)

    def test_queue(self):
        queue = dpctl.SyclQueue()
        bg = dpnp.random.PCG64DXSM(1, sycl_queue=queue)
        dpnp.random.set_bit_generator(bg)
        assert dpnp.random.get_bit_generator(sycl_queue=queue) is bg
        assert dpnp.random.get_bit_generator() is not bg

        res = dpnp.random.random_sample(4, sycl_queue=queue)
        assert res.sycl_queue == queue
        gen = dpnp.random.Generator(dpnp.random.PCG64DXSM(1, sycl_queue=queue))
        assert_array_equal(res, gen.random(4))

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_class(self, xp):
        bg_type = xp.random.PCG64DXSM
        assert_raises(ValueError, xp.random.set_bit_generator, bg_type)

    @pytest.mark.parametrize("bitgen", [None, 5, numpy.random.PCG64DXSM(1)])
    def test_not_a_bit_generator(self, bitgen):
        assert_raises(TypeError, dpnp.random.set_bit_generator, bitgen)


class TestBitGeneratorSeed:
    def test_random_state(self):
        bg = dpnp.random.PCG64DXSM(7)
        rs = RandomState(bg)
        assert str(rs) == "RandomState(PCG64DXSM)"
        assert rs.get_sycl_queue() == bg.sycl_queue
        ref = dpnp.random.Generator(dpnp.random.PCG64DXSM(7)).random(5)
        assert_array_equal(rs.random_sample(5), ref)

    def test_state(self):
        rs = RandomState(dpnp.random.PCG64DXSM(7))
        state = rs.get_state()
        assert state["bit_generator"] == "PCG64DXSM"
        a1 = rs.random_sample(5)
        rs.set_state(state)
        assert_array_equal(rs.random_sample(5), a1)
        philox_state = RandomState(7).get_state()
        assert_raises(ValueError, rs.set_state, philox_state)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_reseed(self, xp):
        # only the legacy bit generator can be reseeded
        rs = xp.random.RandomState(xp.random.PCG64DXSM(7))
        assert_raises(TypeError, rs.seed, 3)

    @pytest.mark.parametrize("kw", ["device", "sycl_queue"])
    def test_queue_kwarg(self, kw):
        bg = dpnp.random.PCG64DXSM(7)
        queue = dpctl.SyclQueue()
        assert_raises(ValueError, RandomState, bg, **{kw: queue})


class TestQueue:
    @pytest.mark.parametrize("device", _all_devices(), ids=str)
    def test_device(self, device):
        rs = RandomState(1, device=device)
        assert rs.get_sycl_device() == device
        res = rs.normal(size=5)
        assert res.sycl_device == device

    @pytest.mark.parametrize("device", _all_devices(), ids=str)
    def test_sycl_queue(self, device):
        queue = dpctl.SyclQueue(device)
        rs = RandomState(1, sycl_queue=queue)
        assert rs.get_sycl_queue() == queue
        assert rs.normal(size=5).sycl_queue == queue

    @pytest.mark.parametrize("name, params, is_float", distributions)
    def test_device_independent(self, name, params, is_float):
        devices = _all_devices()
        if len(devices) < 2:
            pytest.skip("a single device")

        results = []
        for device in devices:
            rs = RandomState(1234, device=device)
            kw = {"dtype": dpnp.float32} if is_float else {}
            results.append(dpnp.asnumpy(getattr(rs, name)(*params, 64, **kw)))
        for res in results[1:]:
            if is_float:
                # the streams match, but the math functions of the devices
                # round differently, which e.g. vonmises amplifies
                assert_allclose(res, results[0], rtol=1e-3, atol=1e-5)
            else:
                # a float rounding may rarely take a rejection sampler down
                # another path
                assert numpy.mean(res != results[0]) < 0.05

    def test_follows_param_queue(self):
        queue = dpctl.SyclQueue()
        lam = dpnp.asarray([1.0, 2.0], sycl_queue=queue)
        res = dpnp.random.poisson(lam)
        assert res.sycl_queue == queue

        x = dpnp.arange(5, sycl_queue=queue)
        assert dpnp.random.permutation(x).sycl_queue == queue
        assert dpnp.random.choice(x, 3).sycl_queue == queue


class TestDistributions:
    @pytest.mark.parametrize("usm_type", list_of_usm_types)
    @pytest.mark.parametrize("name, params, is_float", distributions)
    def test_shape_dtype(self, name, params, is_float, usm_type):
        rs = RandomState(0)
        meth = getattr(rs, name)

        res = meth(*params, size=(3, 4), usm_type=usm_type)
        assert isinstance(res, dpnp.ndarray)
        assert res.shape == (3, 4)
        assert res.usm_type == usm_type
        if is_float:
            assert res.dtype == dpnp.default_float_type()
        else:
            assert res.dtype == dpnp.int64

        assert meth(*params).shape == ()
        assert meth(*params, size=0).shape == (0,)

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize("name, params", float_distributions)
    def test_dtype(self, name, params, dtype):
        res = getattr(RandomState(0), name)(*params, size=5, dtype=dtype)
        assert res.dtype == dtype

    @pytest.mark.parametrize(
        "dtype", [dpnp.float16, dpnp.int32, dpnp.complex64, dpnp.bool]
    )
    @pytest.mark.parametrize("name, params", float_distributions)
    def test_bad_dtype(self, name, params, dtype):
        meth = getattr(RandomState(0), name)
        assert_raises(TypeError, meth, *params, size=5, dtype=dtype)

    @pytest.mark.parametrize("usm_type", ["", "unknown"])
    def test_bad_usm_type(self, usm_type):
        assert_raises(ValueError, RandomState().normal, usm_type=usm_type)

    @pytest.mark.parametrize("name, params, is_float", distributions)
    def test_moments(self, name, params, is_float):
        # the sample means agree with the ones of NumPy within six standard
        # deviations
        n = 2**16
        np_res = getattr(numpy.random.RandomState(0), name)(*params, n)
        res = dpnp.asnumpy(getattr(RandomState(0), name)(*params, n))
        if name == "standard_cauchy":
            np_res, res = numpy.median(np_res), numpy.median(res)
            assert abs(res - np_res) < 0.05
            return
        tol = 6 * numpy.sqrt(np_res.var() / n + res.var() / n)
        assert abs(res.mean() - np_res.mean()) <= tol

    @pytest.mark.parametrize(
        "name, args",
        [
            ("dirichlet", ([1.0, 2.0, 3.0],)),
            ("multinomial", (10, [0.2, 0.3, 0.5])),
            ("multivariate_normal", ([0.0, 1.0], [[2.0, 0.5], [0.5, 1.0]])),
        ],
    )
    @pytest.mark.parametrize("size", [None, 4, (2, 3)])
    def test_multivariate_shape(self, name, args, size):
        np_res = getattr(numpy.random.RandomState(0), name)(*args, size)
        res = getattr(RandomState(0), name)(*args, size)
        assert res.shape == np_res.shape

    @pytest.mark.parametrize("name, params, is_float", distributions)
    def test_module_function(self, name, params, is_float):
        if name == "tomaxint":
            pytest.skip("not a module function")

        dpnp.random.seed(11)
        res = getattr(dpnp.random, name)(*params, size=5, usm_type="shared")
        assert res.usm_type == "shared"
        assert_array_equal(res, getattr(RandomState(11), name)(*params, 5))


class TestSize:
    @pytest.mark.parametrize(
        "size",
        [
            True,
            [True],
            numpy.bool_(True),
            numpy.array(True),
            numpy.array([True]),
            -1,
            (2, -1),
            2.0,
            (3.5, 2),
            numpy.array([[2]]),
        ],
    )
    def test_invalid(self, size):
        _assert_same_error(
            lambda: numpy.random.RandomState(0).uniform(size=size),
            lambda: RandomState(0).uniform(size=size),
        )

    @pytest.mark.parametrize(
        "size",
        [
            (numpy.int64(31), 63),
            (numpy.int64(31), numpy.int64(31)),
            numpy.array(1),
            numpy.array([2]),
            dpnp.array(3),
        ],
    )
    def test_castable_to_integer(self, size):
        res = RandomState().uniform(size=size)
        exp = numpy.random.RandomState().uniform(size=dpnp.asnumpy(size))
        assert res.shape == exp.shape

    @pytest.mark.parametrize("dims", [(), (5,), (10, 2, 7), (0,), (3, 0)])
    @pytest.mark.parametrize("name", ["rand", "randn"])
    def test_rand_dims(self, name, dims):
        res = getattr(RandomState(), name)(*dims)
        assert res.shape == dims

    @pytest.mark.parametrize("dims", [(2, -1), (3.5, 2)])
    @pytest.mark.parametrize("name", ["rand", "randn"])
    def test_rand_bad_dims(self, name, dims):
        _assert_same_error(
            lambda: getattr(numpy.random.RandomState(), name)(*dims),
            lambda: getattr(RandomState(), name)(*dims),
        )

    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_rand_dtype(self, dtype):
        assert RandomState().rand(2, 3, dtype=dtype).dtype == dtype
        assert RandomState().randn(2, 3, dtype=dtype).dtype == dtype


class TestIntegers:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_float16=True, no_none=True, no_complex=True)
    )
    def test_full_range(self, dtype):
        if dpnp.issubdtype(dtype, dpnp.floating):
            assert_raises(TypeError, RandomState().randint, 5, dtype=dtype)
            return

        info = numpy.iinfo(dtype) if dtype != dpnp.bool else None
        low, high = (0, 2) if info is None else (info.min, info.max + 1)
        res = RandomState(0).randint(low, high, size=2**12, dtype=dtype)
        assert res.dtype == dtype
        assert int(res.min()) >= low
        assert int(res.max()) < high
        assert len(numpy.unique(dpnp.asnumpy(res))) > 1

    def test_default_dtype(self):
        assert RandomState().randint(5).dtype == dpnp.int64
        assert RandomState().randint(5, dtype=int).dtype == dpnp.int64

    @pytest.mark.parametrize("low, high", [(2, 4), (-7, -1), (0.6, 6.78)])
    def test_bounds(self, low, high):
        res = dpnp.asnumpy(RandomState(0).randint(low, high, 2**12))
        exp = numpy.random.RandomState(0).randint(low, high, 2**12)
        assert_equal(numpy.unique(res), numpy.unique(exp))

    def test_one_bound(self):
        res = dpnp.asnumpy(RandomState(0).randint(5, size=2**12))
        assert_equal(numpy.unique(res), numpy.arange(5))

    def test_zero_range(self):
        res = RandomState(0).randint(-3, -2, size=100)
        assert (res == -3).all()

    def test_array_bounds(self):
        low = dpnp.asarray([0, 10])
        res = RandomState(0).randint(low, numpy.array([2, 12]), size=(64, 2))
        res = dpnp.asnumpy(res)
        assert_equal(numpy.unique(res[:, 0]), [0, 1])
        assert_equal(numpy.unique(res[:, 1]), [10, 11])

    @pytest.mark.parametrize(
        "args, kwargs",
        [
            ((-numpy.inf, 0), {}),
            ((0, numpy.inf), {}),
            ((1, 1), {}),
            ((5, 2), {}),
            ((-(2**31) - 1, 0), {"dtype": numpy.int32}),
            ((0, 2**31 + 1), {"dtype": numpy.int32}),
            ((0, 3), {"dtype": numpy.bool_}),
            ((5,), {"dtype": numpy.float64}),
            ((0,), {}),
            ((1, [0, 1]), {}),
            ((0, [0, 1]), {}),
            (([-1], 3), {"dtype": numpy.uint8}),
            (([0], 257), {"dtype": numpy.uint8}),
        ],
    )
    def test_errors(self, args, kwargs):
        _assert_same_error(
            lambda: numpy.random.RandomState().randint(*args, **kwargs),
            lambda: RandomState().randint(*args, **kwargs),
        )

    def test_random_integers_none(self):
        with pytest.warns(DeprecationWarning):
            _assert_same_error(
                lambda: numpy.random.RandomState().random_integers(None),
                lambda: RandomState().random_integers(None),
            )

    @pytest.mark.parametrize("args", [(5,), (-2, 3)])
    def test_random_integers(self, args):
        with pytest.warns(DeprecationWarning) as np_warn:
            exp = numpy.random.RandomState(0).random_integers(*args, size=2**12)
        with pytest.warns(DeprecationWarning) as dp_warn:
            res = RandomState(0).random_integers(*args, size=2**12)
        assert str(dp_warn[0].message) == str(np_warn[0].message)
        assert res.dtype == dpnp.int64
        assert_equal(numpy.unique(dpnp.asnumpy(res)), numpy.unique(exp))

    def test_tomaxint(self):
        res = RandomState(0).tomaxint(2**12)
        assert res.dtype == dpnp.int64
        assert (res >= 0).all()
        # the high bit is random too
        assert (res > 2**62).any()


class TestLegacy:
    """The behaviors of RandomState that differ from the ones of Generator."""

    nan = numpy.nan

    @pytest.mark.parametrize(
        "name, args",
        [
            ("uniform", (-numpy.inf, 0.0)),
            ("uniform", (0.0, numpy.inf)),
            ("uniform", (-1e308, 1e308)),
            ("uniform", ([-numpy.inf], [0.0])),
            ("uniform", ([0.0], nan)),
            ("normal", (0.0, -1.0)),
            ("normal", (0.0, -0.0)),
            ("exponential", ([-1.0],)),
            ("gamma", (1.0, -1.0)),
            ("chisquare", (0.0,)),
            ("beta", (0.0, 1.0)),
            ("wald", (1.0, 0.0)),
            ("triangular", (1.0, 0.0, 2.0)),
            ("vonmises", (0.0, -1.0)),
            ("poisson", (-1.0,)),
            ("poisson", (nan,)),
            ("poisson", (1e19,)),
            ("binomial", (-1, 0.5)),
            ("binomial", (5, 1.5)),
            ("binomial", (5, nan)),
            ("negative_binomial", (0.0, 0.5)),
            ("negative_binomial", (-1.0, 0.5)),
            ("negative_binomial", (1.0, 1.5)),
            ("negative_binomial", (1.0, nan)),
            ("negative_binomial", ([1.0], [nan])),
            ("geometric", (0.0,)),
            ("logseries", (1.0,)),
            ("zipf", (1.0,)),
            ("hypergeometric", (1, 2, 4)),
            ("hypergeometric", ([1], 2, 4)),
            ("hypergeometric", (-1, 5, 3)),
            ("hypergeometric", (5, -1, 3)),
            ("hypergeometric", (5, 3, 0)),
            ("hypergeometric", (5, 3, -1)),
            ("hypergeometric", ([5], [3], [0])),
            ("dirichlet", ([-1.0, 1.0],)),
            ("dirichlet", ([0.0, 1.0],)),
            ("multinomial", ([5], [0.5, 0.5])),
            ("multinomial", (5, 0.5)),
            ("multinomial", (5, [[0.5, 0.5]])),
            ("multinomial", (5, [0.7, 0.7, 0.1])),
            ("multinomial", (-1, [0.2, 0.8])),
            ("multinomial", (5, [-0.1, 1.1])),
            ("multivariate_normal", ([[0.0, 0.0]], numpy.eye(2))),
            ("multivariate_normal", ([0.0, 0.0], numpy.eye(3))),
            (
                "multivariate_normal",
                ([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]], None, "raise"),
            ),
            ("permutation", (1.5,)),
            ("permutation", (numpy.zeros(()),)),
            ("choice", (0,)),
            ("choice", (-1,)),
            ("choice", (1.5,)),
            ("choice", ([],)),
            ("choice", ([[1, 2]],)),
            ("choice", (3, 4, False)),
            ("choice", (3, None, True, [0.5, 0.6, -0.1])),
            ("choice", (3, None, True, [0.5, 0.6, 0.1])),
            ("choice", (3, None, True, [0.5, nan, 0.5])),
            ("choice", (3, None, True, [0.5, 0.5])),
            ("choice", (3, None, True, [[0.5, 0.5, 0.0]])),
            ("choice", (3, 2, False, [1.0, 0.0, 0.0])),
            ("poisson", (None,)),
            ("normal", (0.0, None)),
            ("uniform", (None, 1.0)),
            ("negative_binomial", (None, 0.5)),
            ("exponential", ("x",)),
            ("multinomial", (None, [0.5, 0.5])),
            ("multinomial", ("5", [0.5, 0.5])),
            ("multinomial", (5, "x")),
        ],
    )
    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    def test_same_error_as_numpy(self, name, args):
        _assert_same_error(
            lambda: getattr(numpy.random.RandomState(0), name)(*args),
            lambda: getattr(RandomState(0), name)(*args),
        )

    @pytest.mark.parametrize(
        "name, args",
        [
            ("beta", ([1.0, -1.0], 1.0)),
            ("binomial", ([3, -1], 0.5)),
            ("binomial", (3, [0.5, 1.5])),
            ("exponential", ([1.0, -1.0],)),
            ("geometric", ([0.5, 0.0],)),
            ("logseries", ([0.5, 1.0],)),
            ("negative_binomial", ([1.0, 0.0], 0.5)),
            ("noncentral_f", (1.0, [1.0, 0.0], 1.0)),
            ("normal", (0.0, [1.0, -1.0])),
            ("poisson", ([1.0, 1e19],)),
            ("poisson", ([1.0, nan],)),
            ("standard_gamma", ([1.0, -1.0],)),
            ("zipf", ([2.0, 1.0],)),
            ("hypergeometric", ([3, -1], 3, 2)),
            ("hypergeometric", ([3, 1], 3, [2, 5])),
            ("hypergeometric", ([3, 1], 3, [2, 0])),
            ("triangular", ([0.0, 2.0], 1.0, 3.0)),
            ("triangular", ([0.0, 3.0], 3.0, 3.0)),
            ("uniform", ([0.0, -numpy.inf], 1.0)),
            ("randint", ([0, 5], [3, 4])),
            ("randint", ([0, -1], 3, None, "uint8")),
            ("randint", (0, [3, 300], None, "uint8")),
            ("choice", (3, 2, True, [0.5, 0.5, 0.5])),
            ("choice", (3, 2, True, [0.5, -0.5, 1.0])),
            ("choice", (3, 2, True, [0.5, nan, 0.5])),
            ("choice", (3, 3, False, [0.5, 0.5, 0.0])),
            ("multinomial", (3, [0.5, 1.5])),
            ("multinomial", (3, [0.7, 0.7, 0.0])),
            ("dirichlet", ([1.0, 0.0],)),
        ],
    )
    def test_same_error_as_numpy_on_device(self, name, args):
        # unlike Generator, RandomState checks the parameters on the device
        dp_args = [dpnp.asarray(a) if isinstance(a, list) else a for a in args]
        _assert_same_error(
            lambda: getattr(numpy.random.RandomState(0), name)(*args),
            lambda: getattr(RandomState(0), name)(*dp_args),
        )

    def test_uniform_reversed(self):
        res = RandomState(0).uniform(2.0, 1.0, size=100)
        assert ((res > 1.0) & (res <= 2.0)).all()

    def test_negative_binomial_accepted(self):
        # NumPy's legacy checks allow p == 0 and a NaN n
        rs = RandomState(0)
        rs.negative_binomial(1.0, 0.0, size=4)
        rs.negative_binomial(self.nan, 0.5, size=4)

    def test_hypergeometric_limit(self):
        # unlike NumPy's legacy checks, the ones of Generator apply
        assert_raises(ValueError, RandomState().hypergeometric, 10**9, 1, 1)

    def test_multinomial_empty(self):
        res = RandomState(0).multinomial(5, [], size=3)
        exp = numpy.random.RandomState(0).multinomial(5, [], size=3)
        assert res.shape == exp.shape
        assert res.dtype == dpnp.int64

    def test_multinomial_float_n(self):
        res = RandomState(0).multinomial(5.7, [0.5, 0.5], size=10)
        assert (res.sum(axis=-1) == 5).all()

    def test_dirichlet_nan(self):
        RandomState(0).dirichlet([self.nan, 1.0])

    def test_multivariate_normal_svd(self):
        cov = [[2.0, 1.0], [1.0, 2.0]]
        res = RandomState(0).multivariate_normal([1.0, -1.0], cov, 2**16)
        assert_allclose(dpnp.mean(res, axis=0), [1.0, -1.0], atol=0.05)
        assert_allclose(numpy.cov(dpnp.asnumpy(res).T), cov, atol=0.05)

    def test_not_positive_semidefinite_warns(self):
        cov = [[1.0, 2.0], [2.0, 1.0]]
        with pytest.warns(RuntimeWarning, match="positive-semidefinite"):
            RandomState(0).multivariate_normal([0.0, 0.0], cov)


class TestPermutations:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    def test_shuffle(self, dtype):
        x = dpnp.arange(10, dtype=dtype)
        RandomState(0).shuffle(x)
        assert not (x == dpnp.arange(10, dtype=dtype)).all()
        assert_array_equal(dpnp.sort(x), dpnp.arange(10, dtype=dtype))

    def test_shuffle_reproducible(self):
        x1 = dpnp.arange(10)
        x2 = dpnp.arange(10)
        RandomState(4).shuffle(x1)
        RandomState(4).shuffle(x2)
        assert_array_equal(x1, x2)

    def test_shuffle_rows(self):
        x = dpnp.arange(20).reshape(10, 2)
        RandomState(0).shuffle(x)
        x = dpnp.asnumpy(x)
        assert (x[:, 1] - x[:, 0] == 1).all()
        assert_array_equal(numpy.sort(x[:, 0]), numpy.arange(0, 20, 2))

    @pytest.mark.parametrize(
        "x", [[1, 2, 3], numpy.arange(3)], ids=["list", "numpy.ndarray"]
    )
    def test_shuffle_not_dpnp_array(self, x):
        # unlike NumPy, only a dpnp.ndarray is shuffled, like in CuPy
        assert_raises(TypeError, RandomState().shuffle, x)

    def test_permutation_int(self):
        res = RandomState(0).permutation(10)
        assert res.dtype == dpnp.int64
        assert_array_equal(dpnp.sort(res), dpnp.arange(10))

    def test_permutation_array(self):
        x = dpnp.arange(20).reshape(10, 2)
        res = RandomState(0).permutation(x)
        assert_array_equal(x, dpnp.arange(20).reshape(10, 2))
        assert_array_equal(dpnp.sort(res[:, 0]), dpnp.arange(0, 20, 2))

    def test_choice(self):
        res = RandomState(0).choice(5, 2**12)
        assert_equal(numpy.unique(dpnp.asnumpy(res)), numpy.arange(5))

        res = RandomState(0).choice(5, 5, replace=False)
        assert_array_equal(dpnp.sort(res), dpnp.arange(5))

        res = RandomState(0).choice(5, 2**12, p=[0.0, 0.5, 0.0, 0.5, 0.0])
        assert_equal(numpy.unique(dpnp.asnumpy(res)), [1, 3])

    def test_bytes(self):
        res = RandomState(0).bytes(13)
        assert isinstance(res, bytes)
        assert len(res) == 13
