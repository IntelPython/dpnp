import math
import threading

import dpctl
import dpctl.utils as dpu
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_raises,
    assert_raises_regex,
)

import dpnp
import dpnp.libtensor._rng_impl as ri
from dpnp.random import Generator, Philox4x3210, default_rng

from .helper import get_float_dtypes, has_support_aspect64

float_dtypes = get_float_dtypes(no_float16=True)

# the number of output elements each work-item of a kernel produces, with
# Philox4x3210
ITEMS_PER_WORK_ITEM = 8

MASK32 = 0xFFFFFFFF


def _philox4x32x10(key, counter):
    """Reference Philox4x32-10 (Salmon et al., 2011) for one counter."""

    k0, k1 = key & MASK32, key >> 32
    c = [(counter >> (32 * i)) & MASK32 for i in range(4)]
    for r in range(10):
        if r:
            k0 = (k0 + 0x9E3779B9) & MASK32
            k1 = (k1 + 0xBB67AE85) & MASK32
        p0 = 0xD2511F53 * c[0]
        p1 = 0xCD9E8D57 * c[2]
        c = [
            (p1 >> 32) ^ c[1] ^ k0,
            p1 & MASK32,
            (p0 >> 32) ^ c[3] ^ k1,
            p0 & MASK32,
        ]
    return c


def _reference_raw(key, counter, n):
    """The expected output of Philox4x3210.random_raw(n)."""

    n_wi = -(-n // ITEMS_PER_WORK_ITEM)
    res = numpy.empty(n, dtype=numpy.uint64)
    for w in range(n_wi):
        stream = []
        block = (counter + (w << 64)) & ((1 << 128) - 1)
        for i in range(w, n, n_wi):
            while len(stream) < 2:
                stream.extend(_philox4x32x10(key, block))
                block = (block + 1) & ((1 << 128) - 1)
            hi, lo = stream.pop(0), stream.pop(0)
            res[i] = (hi << 32) | lo
    return res


def _all_devices():
    return [
        d for d in dpctl.get_devices() if d.device_type.name in ("cpu", "gpu")
    ]


class TestPhilox4x3210:
    @pytest.mark.parametrize("n", [1, 7, 8, 9, 100])
    @pytest.mark.parametrize(
        "counter", [0, 5, (1 << 64) - 1, (3 << 64) + 2, (1 << 128) - 2]
    )
    def test_known_answer(self, n, counter):
        key = 0x0123456789ABCDEF
        bg = Philox4x3210(key=key, counter=counter)
        res = bg.random_raw(n)
        assert res.dtype == dpnp.uint64
        assert_array_equal(dpnp.asnumpy(res), _reference_raw(key, counter, n))

    def test_state_advance(self):
        bg = Philox4x3210(1234)
        state = bg.state
        assert state["bit_generator"] == "Philox4x3210"
        assert state["state"]["counter"] == 0

        bg.random_raw(17)
        assert bg.state["state"]["counter"] == 3 << 64
        bg.random_raw(output=False)
        assert bg.state["state"]["counter"] == 4 << 64

        # an empty draw does not advance the state
        bg.random_raw((0, 5))
        assert bg.state["state"]["counter"] == 4 << 64

    def test_state_roundtrip(self):
        bg = Philox4x3210(42)
        bg.random_raw(10)
        state = bg.state
        a = bg.random_raw(50)
        bg.state = state
        b = bg.random_raw(50)
        assert_array_equal(a, b)

    def test_state_errors(self):
        bg = Philox4x3210(42)
        assert_raises(TypeError, setattr, bg, "state", 1)
        assert_raises(
            ValueError, setattr, bg, "state", {"bit_generator": "PCG64"}
        )
        assert_raises(
            ValueError,
            setattr,
            bg,
            "state",
            {
                "bit_generator": "Philox4x3210",
                "state": {"key": -1, "counter": 0},
            },
        )

    def test_seed_and_key(self):
        assert_raises(ValueError, Philox4x3210, 1, key=1)
        assert_raises(ValueError, Philox4x3210, key=1 << 64)
        assert_raises(ValueError, Philox4x3210, counter=-1)

    def test_seed_sequence(self):
        ss = numpy.random.SeedSequence(7)
        bg = Philox4x3210(ss)
        assert bg.seed_seq is ss
        key = int(ss.generate_state(1, numpy.uint64)[0])
        assert bg.state["state"]["key"] == key
        assert Philox4x3210(7).state["state"]["key"] == key

    def test_spawn(self):
        bg = Philox4x3210(5)
        children = bg.spawn(2)
        keys = {c.state["state"]["key"] for c in children}
        keys.add(bg.state["state"]["key"])
        assert len(keys) == 3
        for c in children:
            assert c.sycl_queue == bg.sycl_queue

    def test_base_class(self):
        assert_raises(TypeError, dpnp.random.BitGenerator)

    def test_lock_is_reentrant(self):
        bg = Philox4x3210(1)

        def _use_state():
            with bg.lock:
                bg.state = bg.state

        t = threading.Thread(target=_use_state)
        t.start()
        t.join(timeout=10)
        assert not t.is_alive()

    def test_launch_takes_key_under_lock(self):
        bg = Philox4x3210(key=5, counter=0)

        class _SwapKeyOnRelease:
            """A lock after whose release another thread sets a new key."""

            def __init__(self):
                self._lock = threading.RLock()

            def __enter__(self):
                self._lock.acquire()

            def __exit__(self, *args):
                self._lock.release()
                bg._key = 6

        bg._lock = _SwapKeyOnRelease()
        seed, counter_lo, counter_hi = bg._launch(
            ri._random_raw, numpy.dtype(numpy.uint64), 8
        )
        assert (seed, counter_lo, counter_hi) == ([5], 0, 0)

    @pytest.mark.parametrize("size", [-8, (2, -8), 2.0, True])
    def test_random_raw_bad_size(self, size):
        bg = Philox4x3210(1)
        bg.random_raw(8)
        state = bg.state

        # NumPy raises only with output=True
        np_bg = numpy.random.Philox(1)
        for output in (True, False):
            _assert_same_error(
                lambda: np_bg.random_raw(size),
                lambda: bg.random_raw(size, output=output),
            )
            assert bg.state == state

    def test_random_raw_skip_large(self):
        # the number of values does not fit in 64 bits
        bg = Philox4x3210(key=1, counter=0)
        bg.random_raw((3**30, 3**30), output=False)
        n_blocks = -(-math.prod((3**30, 3**30)) // ITEMS_PER_WORK_ITEM)
        expected = (n_blocks << 64) & ((1 << 128) - 1)
        assert bg.state["state"]["counter"] == expected

    def test_spawn_with_key(self):
        # as in NumPy, a bit generator given its key has no seed sequence
        bg = Philox4x3210(key=5)
        assert bg.seed_seq is None
        assert_raises_regex(
            TypeError, "does not implement spawning", bg.spawn, 1
        )
        assert_raises(TypeError, default_rng(bg).spawn, 1)

    def test_float32_normal_reference(self):
        # float32 normals take u1 from a whole 32-bit word
        key, counter, n = 0x0123456789ABCDEF, 3, 32
        res = Generator(Philox4x3210(key=key, counter=counter))
        res = dpnp.asnumpy(res.standard_normal(n, dtype=dpnp.float32))

        n_wi = n // ITEMS_PER_WORK_ITEM
        expected = numpy.empty(n)
        for w in range(n_wi):
            block = counter + (w << 64)
            words = []
            for _ in range(ITEMS_PER_WORK_ITEM // 4):
                words.extend(_philox4x32x10(key, block))
                block += 1
            for k in range(0, ITEMS_PER_WORK_ITEM, 2):
                u1 = (numpy.float32(words[k]) + numpy.float32(0.5)) * (
                    numpy.float32(2.0**-32)
                )
                u2 = (words[k + 1] >> 8) * 2.0**-24
                r = numpy.sqrt(-2 * numpy.log(numpy.float64(u1)))
                expected[w + k * n_wi] = r * numpy.sin(2 * numpy.pi * u2)
                expected[w + (k + 1) * n_wi] = r * numpy.cos(2 * numpy.pi * u2)
        expected = expected.astype(numpy.float32)
        assert_allclose(res, expected, rtol=1e-5, atol=1e-6)

    def test_param_overlaps_result(self):
        x = dpnp.get_usm_ndarray(dpnp.zeros(9, dtype=dpnp.float32))
        q = x.sycl_queue
        engine = int(ri.engine_id.philox4x32x10)
        _manager = dpu.SequentialOrderManager[q]

        assert_raises_regex(
            ValueError,
            "overlap",
            ri._normal,
            q,
            engine,
            [1],
            0,
            0,
            [x[:-1], 1.0],
            x[1:],
            depends=_manager.submitted_events,
        )

        # an exact alias is safe, as every element is read before written
        ht_ev, ev = ri._normal(
            q, engine, [1], 0, 0, [x, 1.0], x, depends=_manager.submitted_events
        )
        _manager.add_event_pair(ht_ev, ev)
        assert dpnp.isfinite(dpnp.asarray(x)).all()


class TestDefaultRng:
    def test_reproducible(self):
        a = default_rng(123).random(100)
        b = default_rng(123).random(100)
        c = default_rng(124).random(100)
        assert_array_equal(a, b)
        assert not (a == c).all()

    def test_passthrough(self):
        rng = default_rng(0)
        assert default_rng(rng) is rng

        bg = Philox4x3210(0)
        assert default_rng(bg).bit_generator is bg

    def test_stream_advances(self):
        rng = default_rng(0)
        a = rng.random(10)
        b = rng.random(10)
        assert not (a == b).any()

    def test_spawn(self):
        rng = default_rng(0)
        children = rng.spawn(2)
        a, b = (c.random(10) for c in children)
        assert isinstance(children[0], Generator)
        assert not (a == b).any()

    def test_bad_bit_generator(self):
        assert_raises(TypeError, Generator, numpy.random.PCG64(0))


class TestFloatMethods:
    # method name, positional parameters
    cases = [
        ("random", ()),
        ("uniform", (-1.0, 2.0)),
        ("standard_normal", ()),
        ("normal", (1.0, 2.0)),
        ("lognormal", (0.0, 0.5)),
        ("standard_exponential", ()),
        ("exponential", (2.0,)),
        ("standard_gamma", (0.5,)),
        ("gamma", (3.0, 2.0)),
        ("chisquare", (3.0,)),
        ("beta", (0.5, 2.0)),
        ("standard_cauchy", ()),
        ("wald", (1.0, 2.0)),
        ("pareto", (3.0,)),
        ("weibull", (2.0,)),
        ("power", (3.0,)),
        ("rayleigh", (2.0,)),
        ("laplace", (1.0, 2.0)),
        ("gumbel", (1.0, 2.0)),
        ("logistic", (1.0, 2.0)),
        ("triangular", (0.0, 1.0, 3.0)),
        ("vonmises", (0.5, 2.0)),
        ("standard_t", (5.0,)),
        ("f", (5.0, 7.0)),
    ]

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize("name, params", cases)
    def test_shape_dtype(self, name, params, dtype):
        rng = default_rng(0)
        meth = getattr(rng, name)

        res = meth(*params, size=(3, 4), dtype=dtype)
        assert isinstance(res, dpnp.ndarray)
        assert res.shape == (3, 4)
        assert res.dtype == dtype
        assert dpnp.isfinite(res).all()

        res = meth(*params, dtype=dtype)
        assert res.shape == ()

        res = meth(*params, size=0, dtype=dtype)
        assert res.shape == (0,)

    @pytest.mark.parametrize("name, params", cases)
    def test_default_dtype(self, name, params):
        rng = default_rng(0)
        res = getattr(rng, name)(*params, size=2)
        assert res.dtype == dpnp.default_float_type(sycl_queue=rng.sycl_queue)

    @pytest.mark.parametrize("name, params", cases)
    def test_bad_dtype(self, name, params):
        rng = default_rng(0)
        meth = getattr(rng, name)
        assert_raises(TypeError, meth, *params, size=2, dtype=numpy.int32)

    @pytest.mark.parametrize("usm_type", ["device", "shared", "host"])
    def test_usm_type(self, usm_type):
        res = default_rng(0).normal(size=4, usm_type=usm_type)
        assert res.usm_type == usm_type

    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_random_range(self, dtype):
        res = dpnp.asnumpy(default_rng(1).random(10**6, dtype=dtype))
        assert res.min() >= 0
        assert res.max() < 1

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize(
        "name, params, mean, var",
        [
            ("random", (), 0.5, 1 / 12),
            ("uniform", (-1.0, 3.0), 1.0, 16 / 12),
            ("standard_normal", (), 0.0, 1.0),
            ("normal", (1.0, 2.0), 1.0, 4.0),
            (
                "lognormal",
                (0.0, 0.5),
                numpy.exp(0.125),
                (numpy.exp(0.25) - 1) * numpy.exp(0.25),
            ),
            ("standard_exponential", (), 1.0, 1.0),
            ("exponential", (2.0,), 2.0, 4.0),
            ("standard_gamma", (0.5,), 0.5, 0.5),
            ("standard_gamma", (1.0,), 1.0, 1.0),
            ("gamma", (3.0, 2.0), 6.0, 12.0),
            ("chisquare", (3.0,), 3.0, 6.0),
            ("beta", (0.5, 2.0), 0.2, 8.0 / 175),
            ("beta", (0.5, 0.5), 0.5, 0.125),
            ("beta", (2.0, 3.0), 0.4, 0.04),
            ("wald", (1.0, 2.0), 1.0, 0.5),
            ("wald", (3.0, 50.0), 3.0, 27 / 50),
            ("pareto", (20.0,), 1 / 19, 20 / (19**2 * 18)),
            (
                "weibull",
                (2.0,),
                math.gamma(1.5),
                math.gamma(2.0) - math.gamma(1.5) ** 2,
            ),
            ("weibull", (0.0,), 0.0, 0.0),
            ("power", (3.0,), 0.75, 0.0375),
            ("rayleigh", (2.0,), 2 * math.sqrt(math.pi / 2), 2 * (4 - math.pi)),
            ("laplace", (1.0, 2.0), 1.0, 8.0),
            (
                "gumbel",
                (1.0, 2.0),
                1 + 2 * 0.5772156649015329,
                math.pi**2 * 4 / 6,
            ),
            ("logistic", (1.0, 2.0), 1.0, math.pi**2 * 4 / 3),
            ("triangular", (0.0, 1.0, 3.0), 4 / 3, 7 / 18),
            ("triangular", (0.0, 0.0, 1.0), 1 / 3, 1 / 18),
            # uniform on the circle, whatever mu is, as in NumPy
            ("vonmises", (0.5, 0.0), 0.0, math.pi**2 / 3),
            ("standard_t", (10.0,), 0.0, 1.25),
            ("f", (10.0, 30.0), 30 / 28, 2 * 900 * 38 / (10 * 28**2 * 26)),
        ],
    )
    def test_moments(self, name, params, mean, var, dtype):
        n = 10**6
        res = getattr(default_rng(2), name)(*params, size=n, dtype=dtype)
        res = dpnp.asnumpy(res).astype(numpy.float64)

        # five standard errors of the mean and of the variance, taking a
        # generous bound on the fourth central moment
        assert abs(res.mean() - mean) <= 5 * numpy.sqrt(var / n)
        assert abs(res.var() - var) <= 5 * var * numpy.sqrt(10 / n)

    @pytest.mark.parametrize("dtype", float_dtypes)
    def test_standard_cauchy(self, dtype):
        res = default_rng(2).standard_cauchy(size=10**6, dtype=dtype)
        q1, q2, q3 = numpy.quantile(dpnp.asnumpy(res), [0.25, 0.5, 0.75])
        assert abs(q1 + 1) < 0.01 and abs(q2) < 0.01 and abs(q3 - 1) < 0.01

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize("kappa", [1e-9, 1e-6, 0.5, 2.0, 50.0, 1e7])
    def test_vonmises(self, kappa, dtype):
        n = 10**6
        mu = 3.0
        res = default_rng(2).vonmises(mu, kappa, size=n, dtype=dtype)
        res = dpnp.asnumpy(res).astype(numpy.float64)
        assert ((res >= -numpy.pi) & (res <= numpy.pi)).all()

        # E[cos(x - mu)] is I1(kappa) / I0(kappa)
        t = numpy.linspace(0, numpy.pi, 20001)
        w = numpy.exp(kappa * (numpy.cos(t) - 1))
        expected = numpy.trapezoid(numpy.cos(t) * w, t) / numpy.trapezoid(w, t)
        assert abs(numpy.cos(res - mu).mean() - expected) < 5 / numpy.sqrt(n)
        # the angles are symmetric around mu
        assert abs(numpy.sin(res - mu).mean()) < 5 / numpy.sqrt(n)

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize(
        "name, params, lo, hi",
        [
            ("pareto", (1e-3,), 0, numpy.inf),
            ("weibull", (1e-3,), 0, numpy.inf),
            ("power", (1e-3,), 0, 1),
            ("power", (1e3,), 0, 1),
            ("wald", (1e10, 1e-10), 0, numpy.inf),
            ("wald", (1e-10, 1e10), 0, numpy.inf),
            ("triangular", (0.0, 1.0, 1.0), 0, 1),
            ("vonmises", (-3.0, 1e-3), -numpy.pi, numpy.pi),
        ],
    )
    def test_extreme_range(self, name, params, lo, hi, dtype):
        res = getattr(default_rng(1), name)(*params, size=10**5, dtype=dtype)
        res = dpnp.asnumpy(res)
        assert ((res >= lo) & (res <= hi)).all()

    @pytest.mark.parametrize(
        "name, params",
        [
            ("wald", (1.0, 1.0)),
            ("pareto", (1.0,)),
            ("weibull", (1.0,)),
            ("power", (1.0,)),
            ("rayleigh", (1.0,)),
            ("laplace", (0.0, 1.0)),
            ("gumbel", (0.0, 1.0)),
            ("logistic", (0.0, 1.0)),
            ("vonmises", (0.0, 1.0)),
            ("chisquare", (1.0,)),
            ("standard_t", (1.0,)),
            ("f", (1.0, 1.0)),
            ("gamma", (1.0, 1.0)),
            ("beta", (1.0, 1.0)),
        ],
    )
    def test_nan_param(self, name, params):
        # NaN passes NumPy's checks and gives NaN; every parameter is tried
        rng = default_rng(0)
        for i in range(len(params)):
            args = list(params)
            args[i] = numpy.nan
            res = dpnp.asnumpy(getattr(rng, name)(*args, size=100))
            assert numpy.isnan(res).all()

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize(
        "a, b", [(1e-3, 1e-3), (0.01, 0.01), (0.3, 1e-4), (50.0, 0.01)]
    )
    def test_beta_range(self, a, b, dtype):
        # the samples underflow, which devices flushing subnormals to zero
        # and with inexact division once turned into inf or 1 + ulp
        res = dpnp.asnumpy(default_rng(1).beta(a, b, size=10**5, dtype=dtype))
        assert ((res >= 0) & (res <= 1)).all()

    def test_array_params(self):
        rng = default_rng(3)
        loc = dpnp.arange(4.0)[:, None] * 10
        res = rng.normal(loc, 1.0, size=(4, 10000))
        assert res.shape == (4, 10000)
        assert_allclose(
            dpnp.asnumpy(res.mean(axis=1)), [0.0, 10.0, 20.0, 30.0], atol=0.1
        )

        # the output shape is the broadcast shape of the parameters
        res = rng.gamma(dpnp.ones(3), numpy.ones((2, 1)))
        assert res.shape == (2, 3)

        # parameters given as lists
        res = rng.uniform([0.0, 10.0], [1.0, 11.0], size=(1000, 2))
        res = dpnp.asnumpy(res)
        assert (res[:, 0] < 1).all() and (res[:, 1] >= 10).all()

    def test_array_params_chisquare(self):
        df = dpnp.asarray([1.0, 10.0])
        res = default_rng(4).chisquare(df, size=(10**5, 2))
        assert_allclose(dpnp.asnumpy(res.mean(axis=0)), [1.0, 10.0], rtol=0.05)

    @pytest.mark.parametrize(
        "size",
        [
            True,
            [True],
            numpy.bool_(True),
            numpy.array(True),
            -1,
            (2, -1),
            2.0,
            (2.0,),
            numpy.array([[2]]),
        ],
    )
    def test_bad_size(self, size):
        _assert_same_error(
            lambda: numpy.random.default_rng(0).normal(size=size),
            lambda: default_rng(0).normal(size=size),
        )

    @pytest.mark.parametrize(
        "size",
        [numpy.int64(3), numpy.array(3), dpnp.array(3), (numpy.int64(2), 3)],
    )
    def test_size_castable_to_integer(self, size):
        res = default_rng(0).normal(size=size)
        assert res.shape == numpy.random.default_rng(0).normal(size=size).shape

    def test_size_not_broadcastable(self):
        rng = default_rng(0)
        assert_raises_regex(
            ValueError,
            "not compatible with broadcast",
            rng.normal,
            numpy.zeros(3),
            1.0,
            size=(2,),
        )

    @pytest.mark.parametrize(
        "name, params, msg",
        [
            ("normal", (0.0, -1.0), "scale < 0"),
            ("lognormal", (0.0, -1.0), "sigma < 0"),
            ("exponential", (-1.0,), "scale < 0"),
            ("standard_gamma", (-1.0,), "shape < 0"),
            ("gamma", (1.0, -1.0), "scale < 0"),
            ("chisquare", (0.0,), "df <= 0"),
            ("beta", (0.0, 1.0), "a <= 0"),
            ("beta", (1.0, 0.0), "b <= 0"),
        ],
    )
    def test_scalar_checks(self, name, params, msg):
        rng = default_rng(0)
        assert_raises_regex(ValueError, msg, getattr(rng, name), *params)

    def test_uniform_overflow(self):
        rng = default_rng(0)
        big = numpy.finfo(numpy.float64).max
        assert_raises(OverflowError, rng.uniform, -big, big)

    def test_uniform_overflow_float32(self):
        # the kernel subtracts the bounds in the sampling type
        rng = default_rng(0)
        args = (-3e38, 3e38)
        assert_raises(OverflowError, rng.uniform, *args, dtype=dpnp.float32)
        assert_raises(
            OverflowError, rng.uniform, 0.0, [1e39], dtype=dpnp.float32
        )
        if has_support_aspect64():
            res = rng.uniform(*args, size=100, dtype=dpnp.float64)
            assert dpnp.isfinite(res).all()

    @pytest.mark.parametrize("bad", [numpy.inf, -numpy.inf, numpy.nan])
    def test_uniform_device_and_host_bounds(self, bad):
        # a host bound is checked even when the other one is on the device
        rng = default_rng(0)
        low = dpnp.zeros(1)
        assert_raises_regex(
            OverflowError,
            "Range exceeds valid bounds",
            rng.uniform,
            low,
            numpy.array([bad]),
        )
        assert_raises(OverflowError, rng.uniform, bad, dpnp.ones(1))
        assert_raises(OverflowError, rng.uniform, low, 1e39, dtype=dpnp.float32)
        assert rng.uniform(low, 1.0).shape == (1,)

    @pytest.mark.parametrize("name", ["normal", "uniform"])
    def test_device_param_dtype(self, name):
        rng = default_rng(0)
        assert_raises_regex(
            TypeError,
            "Cannot cast array data",
            getattr(rng, name),
            dpnp.arange(3) * 1j,
        )
        res = getattr(rng, name)(dpnp.arange(3), dtype=dpnp.float32)
        assert res.dtype == dpnp.float32
        if has_support_aspect64():
            # float64 parameters feed a float32 draw
            res = getattr(rng, name)(
                dpnp.zeros(3, dtype=dpnp.float64), dtype=dpnp.float32
            )
            assert res.dtype == dpnp.float32

    @pytest.mark.parametrize(
        "usm_types, expected",
        [
            (["shared"], "shared"),
            (["host"], "host"),
            (["shared", "host"], "shared"),
            (["device", "host"], "device"),
        ],
    )
    def test_usm_type_follows_params(self, usm_types, expected):
        rng = default_rng(0)
        params = [dpnp.ones(2, usm_type=u) for u in usm_types]
        if len(params) == 1:
            params.append(1.0)
        assert rng.normal(*params).usm_type == expected
        assert rng.uniform(*params).usm_type == expected
        # an explicit usm_type wins
        assert rng.normal(*params, usm_type="device").usm_type == "device"
        # host parameters alone give a device array
        assert rng.normal([1.0, 2.0]).usm_type == "device"

    def test_standard_exponential_method(self):
        rng = default_rng(0)
        rng.standard_exponential(3, method="inv")
        assert_raises(ValueError, rng.standard_exponential, 3, method="x")

    def test_out_overlapping_param(self):
        # a parameter overlapping `out` is read before `out` is written
        dt = dpnp.default_float_type()
        x = dpnp.arange(1, 10, dtype=dt)
        expected = default_rng(3).standard_gamma(dpnp.copy(x[:-1]))
        out = x[1:]
        res = default_rng(3).standard_gamma(x[:-1], out=out)
        assert res is out
        assert_array_equal(out, expected)
        assert x[0] == 1

        # an exact alias needs no copy
        y = dpnp.arange(1, 9, dtype=dt)
        expected = default_rng(3).standard_gamma(dpnp.copy(y))
        default_rng(3).standard_gamma(y, out=y)
        assert_array_equal(y, expected)

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize(
        "name, params",
        [
            ("random", ()),
            ("standard_normal", ()),
            ("standard_exponential", ()),
            ("standard_gamma", (2.0,)),
        ],
    )
    def test_out(self, name, params, dtype):
        out = dpnp.empty((3, 5), dtype=dtype)
        res = getattr(default_rng(5), name)(*params, dtype=dtype, out=out)
        assert res is out

        expected = getattr(default_rng(5), name)(
            *params, size=(3, 5), dtype=dtype
        )
        assert_array_equal(out, expected)

    def test_out_errors(self):
        rng = default_rng(0)
        out = dpnp.empty(3, dtype=dpnp.float32)
        assert_raises(TypeError, rng.random, out=numpy.empty(3))
        assert_raises(TypeError, rng.random, dtype=dpnp.int32, out=out)
        assert_raises(TypeError, rng.random, dtype=dpnp.float64, out=out)
        assert_raises(ValueError, rng.random, size=4, out=out, dtype=out.dtype)

        # the default dtype applies even when out is given
        default = dpnp.default_float_type(sycl_queue=rng.sycl_queue)
        other = dpnp.float32 if default == dpnp.float64 else None
        if other is not None:
            assert_raises(TypeError, rng.random, out=dpnp.empty(3, dtype=other))

        # out must be contiguous, as in NumPy, each with its default dtype
        np_big = numpy.empty((3, 10))
        big = dpnp.empty((3, 10), dtype=default)
        _assert_same_error(
            lambda: numpy.random.default_rng(0).random(out=np_big[:, ::2]),
            lambda: rng.random(out=big[:, ::2]),
        )
        _assert_same_error(
            lambda: numpy.random.default_rng(0).random(out=np_big[:, :3].T),
            lambda: rng.random(out=big[:, :3].T),
        )

    @pytest.mark.parametrize("name", ["random", "standard_normal"])
    def test_out_usm_ndarray(self, name):
        out = dpnp.get_usm_ndarray(dpnp.empty(6, dtype=dpnp.float32))
        res = getattr(default_rng(5), name)(dtype=dpnp.float32, out=out)
        assert res is out
        expected = getattr(default_rng(5), name)(6, dtype=dpnp.float32)
        assert_array_equal(dpnp.asnumpy(out), expected)

    @pytest.mark.parametrize("name", ["random", "standard_normal"])
    def test_out_f_contiguous(self, name):
        # as in NumPy, an F-contiguous out is filled in memory order
        out = dpnp.empty((3, 4), dtype=dpnp.float32, order="F")
        res = getattr(default_rng(5), name)(dtype=dpnp.float32, out=out)
        assert res is out
        expected = getattr(default_rng(5), name)((4, 3), dtype=dpnp.float32)
        assert_array_equal(out, expected.T)

        usm_out = dpnp.get_usm_ndarray(dpnp.empty_like(out, order="F"))
        res = getattr(default_rng(5), name)(dtype=dpnp.float32, out=usm_out)
        assert res is usm_out
        assert_array_equal(dpnp.asnumpy(usm_out), expected.T)

    def test_out_read_only(self):
        out = dpnp.empty(3, dtype=dpnp.float32)
        out.flags.writable = False
        assert_raises_regex(
            ValueError,
            "must be contiguous, writable",
            default_rng(0).random,
            dtype=dpnp.float32,
            out=out,
        )

    def test_device_params_terminate(self):
        # the parameters on which the rejection samplers could loop forever
        # give NaN
        rng = default_rng(0)
        nan, inf = numpy.nan, numpy.inf

        res = rng.beta(dpnp.asarray([-inf, -1.0, 0.0, nan]), 1.0, size=(8, 4))
        assert dpnp.isnan(res).all()

        # a shape just below zero would reject nearly every draw
        shape = dpnp.asarray([-1e-8, -1e-12, -1e-30, -1.0])
        res = rng.standard_gamma(shape, size=(8, 4))
        assert dpnp.isnan(res).all()
        res = rng.chisquare(shape, size=(8, 4))
        assert dpnp.isnan(res).all()
        res = rng.standard_gamma(shape.astype(dpnp.float32), dtype="f4")
        assert dpnp.isnan(res).all()
        rng.standard_t(shape)

    def test_host_params_terminate(self):
        # valid parameters on which rounding in single precision made every
        # draw of the rejection sampler fail
        rng = default_rng(0)
        kappa = [5e-5, 1e-4, 3e-4, 4e-4]
        res = rng.vonmises(0.0, kappa, size=(8, 4), dtype=dpnp.float32)
        assert (dpnp.abs(res) <= numpy.pi).all()


def _assert_same_error(np_call, dp_call):
    """Assert `dp_call` raises the same exception as `np_call`."""

    with pytest.raises(Exception) as np_exc:
        np_call()
    with pytest.raises(type(np_exc.value)) as dp_exc:
        dp_call()
    assert str(dp_exc.value) == str(np_exc.value)


class TestChecks:
    nan = numpy.nan

    @pytest.mark.parametrize(
        "name, args",
        [
            ("uniform", (2.0, 1.0)),
            ("uniform", ([2.0], [1.0])),
            ("uniform", (-1e308, 1e308)),
            ("uniform", ([-1e308], 1e308)),
            ("uniform", (0.0, nan)),
            ("uniform", ([0.0], nan)),
            ("normal", (0.0, -1.0)),
            ("normal", (0.0, -0.0)),
            ("normal", (0.0, [1.0, -1.0])),
            ("lognormal", (0.0, -1.0)),
            ("exponential", (-1.0,)),
            ("exponential", ([-1.0],)),
            ("standard_gamma", (-0.0,)),
            ("gamma", (1.0, -1.0)),
            ("gamma", ([-1.0], 1.0)),
            ("chisquare", (0.0,)),
            ("chisquare", ([1.0, 0.0],)),
            ("standard_t", (0.0,)),
            ("f", (0.0, 1.0)),
            ("f", (1.0, [0.0])),
            ("beta", (0.0, 1.0)),
            ("beta", (1.0, [0.0])),
            ("pareto", (0.0,)),
            ("power", (0.0,)),
            ("weibull", (-1.0,)),
            ("weibull", (-0.0,)),
            ("rayleigh", (-1.0,)),
            ("laplace", (0.0, -1.0)),
            ("gumbel", (0.0, -1.0)),
            ("logistic", (0.0, -1.0)),
            ("wald", (0.0, 1.0)),
            ("wald", (1.0, 0.0)),
            ("wald", ([1.0], [0.0])),
            ("vonmises", (0.0, -1.0)),
            ("triangular", (1.0, 0.0, 2.0)),
            ("triangular", (0.0, 2.0, 1.0)),
            ("triangular", (1.0, 1.0, 1.0)),
            ("triangular", ([0.0, 1.0], 1.0, [2.0, 1.0])),
            ("normal", (0.0, None)),
            ("uniform", (None, 1.0)),
            # complex and object arrays do not cast safely to float64
            ("normal", (numpy.arange(3) * 1j,)),
            ("normal", (numpy.array(1j),)),
            ("normal", (0.0, numpy.array([1.0, None], dtype=object))),
            ("normal", ([1j, 2j],)),
            ("uniform", (numpy.array([1j]), 2.0)),
            ("uniform", (0.0, numpy.array(2j))),
            ("triangular", (0.0, None, 2.0)),
            ("exponential", ("x",)),
            ("exponential", (1j,)),
        ],
    )
    @pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
    def test_same_as_numpy(self, name, args):
        np_rng = numpy.random.default_rng(0)
        rng = default_rng(0)
        _assert_same_error(
            lambda: getattr(np_rng, name)(*args),
            lambda: getattr(rng, name)(*args),
        )

    @pytest.mark.parametrize(
        "name, args",
        [
            ("normal", (None, [1.0, 2.0])),
            ("laplace", (0.0, [None, 1.0])),
            ("gamma", (None, [1.0, 2.0])),
            ("triangular", (None, [1.0, 2.0], 3.0)),
        ],
    )
    def test_none_with_array_is_nan(self, name, args):
        # as in NumPy, None is NaN when another parameter is an array
        res = getattr(default_rng(0), name)(*args)
        assert dpnp.isnan(res).any()


@pytest.mark.skipif(len(_all_devices()) < 2, reason="needs two devices")
class TestDeviceIndependence:
    def _draw(self, device, bitgen):
        rng = Generator(bitgen(12, device=device))
        return (
            rng.random(1000, dtype=dpnp.float32),
            rng.bit_generator.random_raw(1000),
            rng.normal(size=1000, dtype=dpnp.float32),
            rng.gamma(0.7, size=1000, dtype=dpnp.float32),
        )

    @pytest.mark.parametrize("bitgen", [Philox4x3210])
    def test_same_values(self, bitgen):
        devices = _all_devices()
        ref = [dpnp.asnumpy(x) for x in self._draw(devices[0], bitgen)]
        for dev in devices[1:]:
            res = [dpnp.asnumpy(x) for x in self._draw(dev, bitgen)]
            # uniform bits are exact
            for a, b in zip(ref[:2], res[:2]):
                assert_array_equal(a, b)
            # transcendental functions may differ in the last bits
            for a, b in zip(ref[2:], res[2:]):
                assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_queue(self):
        for dev in _all_devices():
            rng = default_rng(0, device=dev)
            res = rng.random(3, dtype=dpnp.float32)
            assert res.sycl_device == dev
            assert res.sycl_queue == rng.sycl_queue
