import math
import re

import dpctl
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)

import dpnp
from dpnp.random import Generator, Philox4x3210, default_rng

from .helper import get_float_dtypes, has_support_aspect64

float_dtypes = get_float_dtypes(no_float16=True)
integer_dtypes = [
    numpy.bool_,
    numpy.int8,
    numpy.uint8,
    numpy.int16,
    numpy.uint16,
    numpy.int32,
    numpy.uint32,
    numpy.int64,
    numpy.uint64,
]

# the number of output elements each work-item of a kernel produces
ITEMS_PER_WORK_ITEM = 8

MASK32 = 0xFFFFFFFF
MASK64 = (1 << 64) - 1


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
        ("noncentral_chisquare", (3.0, 2.0)),
        ("standard_t", (5.0,)),
        ("f", (5.0, 7.0)),
        ("noncentral_f", (5.0, 7.0, 2.0)),
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
            ("noncentral_chisquare", (3.0, 2.0), 5.0, 14.0),
            ("noncentral_chisquare", (0.5, 1.0), 1.5, 5.0),
            ("noncentral_chisquare", (3.0, 0.0), 3.0, 6.0),
            ("standard_t", (10.0,), 0.0, 1.25),
            ("f", (10.0, 30.0), 30 / 28, 2 * 900 * 38 / (10 * 28**2 * 26)),
            (
                "noncentral_f",
                (10.0, 30.0, 2.0),
                30 * 12 / (10 * 28),
                2 * 9 * (144 + 14 * 28) / (28**2 * 26),
            ),
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
            ("noncentral_chisquare", (1e-3, 1e-3), 0, numpy.inf),
            ("noncentral_f", (1e-3, 1e-3, 1e3), 0, numpy.inf),
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
            ("noncentral_chisquare", (1.0, 1.0)),
            ("standard_t", (1.0,)),
            ("f", (1.0, 1.0)),
            ("noncentral_f", (1.0, 1.0, 1.0)),
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

    def test_standard_exponential_method(self):
        rng = default_rng(0)
        rng.standard_exponential(3, method="inv")
        assert_raises(ValueError, rng.standard_exponential, 3, method="x")

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

        # out must be C-contiguous
        big = dpnp.empty((3, 10), dtype=default)
        assert_raises(ValueError, rng.random, out=big[:, ::2])
        assert_raises(ValueError, rng.random, out=big[:, :3].T)


class TestIntegers:
    @pytest.mark.parametrize("dtype", integer_dtypes)
    def test_full_range(self, dtype):
        if dtype == numpy.bool_:
            lo, hi = 0, 1
        else:
            info = numpy.iinfo(dtype)
            lo, hi = int(info.min), int(info.max)

        rng = default_rng(6)
        res = rng.integers(lo, hi, size=10**5, dtype=dtype, endpoint=True)
        assert res.dtype == dtype
        res = dpnp.asnumpy(res)
        assert res.min() >= lo and res.max() <= hi

        if numpy.dtype(dtype).itemsize == 1:
            # every value of a small type is drawn
            assert len(numpy.unique(res)) == hi - lo + 1

        # an exclusive high may be one past the maximum
        res = rng.integers(lo, hi + 1, size=10, dtype=dtype)
        assert res.dtype == dtype

    @pytest.mark.parametrize("dtype", integer_dtypes[1:])
    def test_bounds(self, dtype):
        rng = default_rng(7)
        res = dpnp.asnumpy(rng.integers(3, 10, size=10**5, dtype=dtype))
        assert_equal(numpy.unique(res), numpy.arange(3, 10, dtype=dtype))

        res = dpnp.asnumpy(
            rng.integers(3, 10, size=10**5, dtype=dtype, endpoint=True)
        )
        assert_equal(numpy.unique(res), numpy.arange(3, 11, dtype=dtype))

        res = dpnp.asnumpy(rng.integers(5, size=1000, dtype=dtype))
        assert_equal(numpy.unique(res), numpy.arange(5, dtype=dtype))

        # a single-value range
        res = dpnp.asnumpy(rng.integers(4, 5, size=10, dtype=dtype))
        assert (res == 4).all()

    def test_uniformity(self):
        # a range that is not a power of two is not biased
        n, k = 10**6, 3
        res = dpnp.asnumpy(default_rng(8).integers(0, k, size=n))
        counts = numpy.bincount(res, minlength=k)
        assert (abs(counts - n / k) < 5 * numpy.sqrt(n / k)).all()

    def test_large_range(self):
        rng = default_rng(9)
        hi = (1 << 64) - 12345
        res = dpnp.asnumpy(rng.integers(0, hi, size=10**4, dtype=numpy.uint64))
        assert (res < hi).all()
        # about half of the values are at least 2**63
        assert 0.4 < (res >= (1 << 63)).mean() < 0.6

    def test_default_dtype(self):
        res = default_rng(0).integers(10, size=3)
        assert res.dtype == dpnp.int64

    def test_array_bounds(self):
        rng = default_rng(10)
        high = dpnp.asarray([1, 10, 100])
        res = rng.integers(0, high, size=(1000, 3))
        assert res.shape == (1000, 3)
        res = dpnp.asnumpy(res)
        assert (res < dpnp.asnumpy(high)).all()
        assert res[:, 0].max() == 0
        assert res[:, 2].max() > 50

        # an exclusive high of max + 1 does not overflow
        res = rng.integers(numpy.array([0, 250]), 256, dtype=numpy.uint8)
        assert res.dtype == numpy.uint8
        assert dpnp.asnumpy(res)[1] >= 250

    @pytest.mark.parametrize(
        "args, kwargs, msg",
        [
            ((5, 5), {}, "low >= high"),
            ((5, 4), {"endpoint": True}, "low > high"),
            ((-1, 5), {"dtype": numpy.uint8}, "low is out of bounds for uint8"),
            (
                (0, 257),
                {"dtype": numpy.uint8},
                "high is out of bounds for uint8",
            ),
            ((0, 256), {"dtype": numpy.uint8, "endpoint": True}, "high is out"),
            ((0, 3), {"dtype": numpy.bool_}, "high is out of bounds for bool"),
        ],
    )
    def test_errors(self, args, kwargs, msg):
        rng = default_rng(0)
        assert_raises_regex(ValueError, msg, rng.integers, *args, **kwargs)

    def test_bad_dtype(self):
        rng = default_rng(0)
        assert_raises(TypeError, rng.integers, 0, 5, dtype=numpy.float32)

    def test_bytes(self):
        rng = default_rng(11)
        b = rng.bytes(10)
        assert isinstance(b, bytes) and len(b) == 10
        assert default_rng(11).bytes(10) == b


def _zeta(a, n=10**5):
    """The Riemann zeta function, by a partial sum and Euler-Maclaurin."""
    k = numpy.arange(1, n, dtype=numpy.float64)
    return (k**-a).sum() + n ** (1 - a) / (a - 1) + 0.5 * n**-a


def _lchoose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _xlog1py(x, y):
    """x * log1p(y), taking 0 * log(0) to be 0."""
    if x == 0:
        return 0.0
    return -math.inf if y == -1 else x * math.log1p(y)


# log-pmfs of the discrete distributions, for the support values k
_log_pmfs = {
    "poisson": lambda k, lam: k * math.log(lam) - lam - math.lgamma(k + 1),
    "binomial": lambda k, n, p: (
        _lchoose(n, k) + k * math.log(p) + _xlog1py(n - k, -p)
        if k <= n
        else -math.inf
    ),
    "negative_binomial": lambda k, n, p: (
        math.lgamma(k + n)
        - math.lgamma(k + 1)
        - math.lgamma(n)
        + n * math.log(p)
        + k * math.log1p(-p)
    ),
    "geometric": lambda k, p: (
        _xlog1py(k - 1, -p) + math.log(p) if k >= 1 else -math.inf
    ),
    "logseries": lambda k, p: (
        k * math.log(p) - math.log(k) - math.log(-math.log1p(-p))
        if k >= 1
        else -math.inf
    ),
    "zipf": lambda k, a: (
        -a * math.log(k) - math.log(_zeta(a)) if k >= 1 else -math.inf
    ),
    "hypergeometric": lambda k, g, b, s: (
        _lchoose(g, k) + _lchoose(b, s - k) - _lchoose(g + b, s)
        if max(0, s - b) <= k <= min(g, s)
        else -math.inf
    ),
}


@pytest.fixture(params=["double", "single"])
def discrete_precision(request, monkeypatch):
    """Run a test with both floating-point types of discrete sampling."""

    if request.param == "double" and not has_support_aspect64():
        pytest.skip("no fp64 support on the device")
    if request.param == "single":
        monkeypatch.setattr(
            Generator,
            "_discrete_float_type",
            lambda self: numpy.dtype(numpy.float32),
        )
    return request.param


class TestDiscreteMethods:
    cases = [
        ("poisson", (3.0,)),
        ("binomial", (10, 0.3)),
        ("negative_binomial", (5.0, 0.3)),
        ("geometric", (0.5,)),
        ("logseries", (0.6,)),
        ("zipf", (2.5,)),
        ("hypergeometric", (5, 7, 3)),
    ]

    @pytest.mark.parametrize("name, params", cases)
    def test_shape_dtype(self, name, params):
        rng = default_rng(0)
        meth = getattr(rng, name)

        res = meth(*params, size=(3, 4))
        assert isinstance(res, dpnp.ndarray)
        assert res.shape == (3, 4)
        assert res.dtype == dpnp.int64
        assert meth(*params).shape == ()
        assert meth(*params, size=0).shape == (0,)

        res = meth(*params, size=2, usm_type="host")
        assert res.usm_type == "host"

    @pytest.mark.parametrize(
        "name, params",
        [
            ("poisson", (0.5,)),
            ("poisson", (5.0,)),
            ("poisson", (10.0,)),
            ("poisson", (15.5,)),
            ("poisson", (100.0,)),
            ("binomial", (10, 0.3)),
            ("binomial", (20, 0.9)),
            ("binomial", (100, 0.4)),
            ("binomial", (1000, 0.5)),
            ("binomial", (5, 1.0)),
            ("negative_binomial", (5.0, 0.3)),
            ("negative_binomial", (0.5, 0.8)),
            ("negative_binomial", (100.0, 0.5)),
            ("geometric", (0.5,)),
            ("geometric", (0.9,)),
            ("geometric", (0.1,)),
            ("geometric", (1.0,)),
            ("logseries", (0.3,)),
            ("logseries", (0.9,)),
            ("zipf", (1.5,)),
            ("zipf", (2.5,)),
            ("zipf", (6.0,)),
            ("hypergeometric", (5, 7, 3)),
            ("hypergeometric", (7, 5, 10)),
            ("hypergeometric", (100, 200, 50)),
            ("hypergeometric", (1000, 5, 100)),
            ("hypergeometric", (300, 300, 590)),
        ],
    )
    def test_pmf(self, name, params, discrete_precision):
        n = 10**6
        res = dpnp.asnumpy(getattr(default_rng(3), name)(*params, size=n))
        counts = numpy.bincount(res[res < 10**4], minlength=1)

        for k in range(len(counts) + 1):
            prob = math.exp(_log_pmfs[name](k, *params))
            freq = counts[k] / n if k < len(counts) else 0.0
            # five standard errors of the frequency
            assert (
                abs(freq - prob) <= 5 * math.sqrt(prob * (1 - prob) / n) + 5e-6
            )

    @pytest.mark.parametrize(
        "name, params, mean, var",
        [
            ("poisson", (1e7,), 1e7, 1e7),
            ("poisson", (1e15,), 1e15, 1e15),
            ("binomial", (10**9, 0.3), 3e8, 2.1e8),
            ("binomial", (10**7, 1e-6), 10.0, 10.0),
            ("negative_binomial", (1e6, 0.5), 1e6, 2e6),
            ("geometric", (1e-6,), 1e6, (1 - 1e-6) / 1e-12),
            ("logseries", (0.999,), None, None),
            (
                "hypergeometric",
                (10**8, 3 * 10**8, 10**6),
                2.5e5,
                10**6 * 0.25 * 0.75 * (4e8 - 1e6) / (4e8 - 1),
            ),
        ],
    )
    def test_large_params(self, name, params, mean, var, discrete_precision):
        n = 10**6
        res = dpnp.asnumpy(getattr(default_rng(4), name)(*params, size=n))
        assert (res >= 0).all()
        if name == "logseries":
            p = params[0]
            mean = -p / ((1 - p) * math.log1p(-p))
            var = -p * (p + math.log1p(-p)) / ((1 - p) * math.log1p(-p)) ** 2
            # a heavy tail
            var_tol = 0.2
        else:
            var_tol = 5 * math.sqrt(10 / n)

        res = res.astype(numpy.float64)
        # the rounding of the parameter to the compute type
        eps = 2**-23 if discrete_precision == "single" else 2**-52
        assert abs(res.mean() - mean) <= 5 * math.sqrt(var / n) + eps * mean
        assert abs(res.var() - var) <= var_tol * var

    def test_edge_values(self):
        rng = default_rng(5)
        assert (dpnp.asnumpy(rng.poisson(0.0, size=100)) == 0).all()
        assert (dpnp.asnumpy(rng.binomial(0, 0.5, size=100)) == 0).all()
        assert (dpnp.asnumpy(rng.binomial(7, 0.0, size=100)) == 0).all()
        assert (dpnp.asnumpy(rng.binomial(7, 1.0, size=100)) == 7).all()
        assert (dpnp.asnumpy(rng.geometric(1.0, size=100)) == 1).all()
        assert (dpnp.asnumpy(rng.logseries(0.0, size=100)) == 1).all()
        assert (dpnp.asnumpy(rng.zipf(1025.0, size=100)) == 1).all()
        assert (dpnp.asnumpy(rng.hypergeometric(5, 0, 3, size=100)) == 3).all()
        assert (dpnp.asnumpy(rng.hypergeometric(0, 5, 3, size=100)) == 0).all()
        assert (dpnp.asnumpy(rng.hypergeometric(5, 5, 10, size=100)) == 5).all()

        # values that do not fit an int64 are clamped or rejected
        res = dpnp.asnumpy(rng.geometric(1e-300, size=1000))
        assert (res > 0).all()
        res = dpnp.asnumpy(rng.zipf(1.0001, size=1000))
        assert (res > 0).all()

    def test_array_params(self, discrete_precision):
        rng = default_rng(6)
        lam = dpnp.asarray([1.0, 100.0, 1e6])
        res = rng.poisson(lam, size=(10**5, 3))
        assert_allclose(
            dpnp.asnumpy(res.mean(axis=0)), [1.0, 100.0, 1e6], rtol=0.02
        )

        res = rng.binomial([10, 1000], dpnp.asarray([[0.5], [0.1]]))
        assert res.shape == (2, 2)

        res = rng.hypergeometric(numpy.array([5, 500]), 5, [3, 7], size=(4, 2))
        assert res.shape == (4, 2)
        res = dpnp.asnumpy(res)
        assert (res[:, 0] <= 3).all() and (res[:, 1] <= 7).all()

        # integer parameters are truncated, as in NumPy
        res = dpnp.asnumpy(rng.binomial([2.9], 1.0, size=3))
        assert (res == 2).all()
        res = dpnp.asnumpy(rng.binomial(dpnp.asarray([2.9]), 1.0, size=3))
        assert (res == 2).all()

    def test_device_params_not_checked(self):
        # invalid parameters on the device give undefined values
        rng = default_rng(0)
        rng.poisson(dpnp.asarray([-1.0, numpy.nan]))
        rng.binomial(dpnp.asarray([-1]), 0.5)
        rng.zipf(dpnp.asarray([0.5, numpy.nan]))
        rng.hypergeometric(dpnp.asarray([10**10]), 1, 5)
        rng.normal(0, dpnp.asarray([-1.0]))

    def test_device_params_terminate(self):
        # the parameters on which the rejection samplers could loop forever
        # give INT64_MIN, or NaN for beta
        rng = default_rng(0)
        nan, inf = numpy.nan, numpy.inf
        int64_min = numpy.iinfo(numpy.int64).min

        res = rng.poisson(dpnp.asarray([inf, 1e300, 2.0**63]), size=(8, 3))
        assert_array_equal(res, int64_min, strict=False)
        res = rng.binomial(5, dpnp.asarray([nan, -1.0, 2.0]), size=(8, 3))
        assert_array_equal(res, int64_min, strict=False)
        n = dpnp.asarray([2**40, -1], dtype=dpnp.int64)
        res = rng.hypergeometric(n, n.max(), 10**9, size=(8, 2))
        assert_array_equal(res, int64_min, strict=False)
        res = rng.beta(dpnp.asarray([-inf, -1.0, 0.0, nan]), 1.0, size=(8, 4))
        assert dpnp.isnan(res).all()
        rng.negative_binomial(dpnp.asarray([inf, nan]), 0.5)
        rng.noncentral_chisquare(1.0, dpnp.asarray([inf, nan]))
        rng.noncentral_f(1.0, 1.0, dpnp.asarray([inf, nan]))
        res = rng.multinomial(10, dpnp.asarray([nan, 0.5, nan]))
        assert res.shape == (3,)


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
            ("noncentral_chisquare", (0.0, 1.0)),
            ("noncentral_chisquare", (1.0, -1.0)),
            ("standard_t", (0.0,)),
            ("f", (0.0, 1.0)),
            ("f", (1.0, [0.0])),
            ("noncentral_f", (0.0, 1.0, 1.0)),
            ("noncentral_f", (1.0, 0.0, 1.0)),
            ("noncentral_f", (1.0, 1.0, -1.0)),
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
            ("poisson", (-1.0,)),
            ("poisson", (nan,)),
            ("poisson", ([nan],)),
            ("poisson", ([-1.0],)),
            ("poisson", (1e19,)),
            ("poisson", ([1e19],)),
            ("binomial", (-1, 0.5)),
            ("binomial", ([-1], 0.5)),
            ("binomial", (5, 1.5)),
            ("binomial", (5, nan)),
            ("binomial", ([5], [nan])),
            ("binomial", (nan, 0.5)),
            ("binomial", (-1, 2.0)),
            ("negative_binomial", (0.0, 0.5)),
            ("negative_binomial", (nan, 0.5)),
            ("negative_binomial", ([nan], 0.5)),
            ("negative_binomial", (1.0, 0.0)),
            ("negative_binomial", (1.0, [1.5])),
            ("negative_binomial", (-1.0, 2.0)),
            ("negative_binomial", (1e20, 1e-10)),
            ("negative_binomial", ([1e20], 1e-10)),
            ("geometric", (0.0,)),
            ("geometric", (1.5,)),
            ("geometric", ([nan],)),
            ("logseries", (1.0,)),
            ("logseries", (-0.1,)),
            ("logseries", (nan,)),
            ("logseries", ([nan],)),
            ("zipf", (1.0,)),
            ("zipf", (nan,)),
            ("zipf", ([nan],)),
            ("hypergeometric", (10**9, 1, 1)),
            ("hypergeometric", (1, [10**9], 1)),
            ("hypergeometric", (1, 2, 4)),
            ("hypergeometric", ([1], 2, 4)),
            ("hypergeometric", (-1, 5, 3)),
            ("hypergeometric", (5, -1, 3)),
            ("hypergeometric", (5, 3, -1)),
            ("hypergeometric", ([5], [3], [-1])),
            ("hypergeometric", (nan, 3, 1)),
            ("dirichlet", ([-1.0, 1.0],)),
            ("dirichlet", ([[1.0, 2.0]],)),
            ("dirichlet", (1.0,)),
            ("multinomial", (5, 0.5)),
            ("multinomial", (5, [])),
            ("multinomial", (5, [0.7, 0.7, 0.1])),
            ("multinomial", (5, [[0.7, 0.7, 0.1]])),
            ("multinomial", (-1, [0.2, 0.8])),
            ("multinomial", (5, [-0.1, 1.1])),
            ("multinomial", (5, [nan, 1.0])),
            ("multivariate_hypergeometric", ([1, 2], 5, None, "x")),
            ("multivariate_hypergeometric", ([1, 2], 1.5)),
            ("multivariate_hypergeometric", ([1, 2], -1)),
            ("multivariate_hypergeometric", ([1.0, 2.0], 1)),
            ("multivariate_hypergeometric", ([[1, 2]], 1)),
            ("multivariate_hypergeometric", ([1, -2], 1)),
            ("multivariate_hypergeometric", ([1, 2], 4)),
            ("multivariate_hypergeometric", ([10**9], 1)),
            ("multivariate_normal", ([[0.0, 0.0]], numpy.eye(2))),
            ("multivariate_normal", ([0.0, 0.0], [1.0, 1.0])),
            ("multivariate_normal", ([0.0, 0.0], numpy.ones((2, 3)))),
            ("multivariate_normal", ([0.0, 0.0], numpy.eye(3))),
            ("multivariate_normal", ([0j, 0.0], numpy.eye(2))),
            ("multivariate_normal", ([0.0, 0.0], numpy.eye(2), None, "x")),
            (
                "multivariate_normal",
                ([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]], None, "raise"),
            ),
            ("permutation", (1.5,)),
            ("permutation", (numpy.zeros(()),)),
            ("permutation", (numpy.zeros((2, 3)), 2)),
            ("choice", (0,)),
            ("choice", (-1,)),
            ("choice", (1.5,)),
            ("choice", (numpy.array(1.5),)),
            ("choice", ([],)),
            ("choice", (3, 4, False)),
            ("choice", (3, None, True, [0.5, 0.6, -0.1])),
            ("choice", (3, None, True, [0.5, 0.6, 0.1])),
            ("choice", (3, None, True, [0.5, nan, 0.5])),
            ("choice", (3, None, True, [0.5, 0.5])),
            ("choice", (3, None, True, [[0.5, 0.5, 0.0]])),
            ("choice", (3, 2, False, [1.0, 0.0, 0.0])),
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

    def test_multivariate_hypergeometric_method(self):
        _assert_same_error(
            lambda: numpy.random.default_rng(0).multivariate_hypergeometric(
                [10**9], 1, method="x"
            ),
            lambda: default_rng(0).multivariate_hypergeometric(
                [10**9], 1, method="x"
            ),
        )
        # unlike NumPy, the count method is limited as the marginals one is
        assert_raises_regex(
            ValueError,
            'When method is "count", sum',
            default_rng(0).multivariate_hypergeometric,
            [10**9],
            1,
            method="count",
        )

    def test_multivariate_normal_method(self):
        _assert_same_error(
            lambda: numpy.random.default_rng(0).multivariate_normal(
                [0.0], [[1.0]], method="x"
            ),
            lambda: default_rng(0).multivariate_normal(
                [0.0], [[1.0]], method="x"
            ),
        )

    def test_shuffle(self):
        np_rng = numpy.random.default_rng(0)
        rng = default_rng(0)
        _assert_same_error(
            lambda: np_rng.shuffle(numpy.zeros(())),
            lambda: rng.shuffle(dpnp.zeros(())),
        )
        _assert_same_error(
            lambda: np_rng.shuffle(numpy.zeros(3), axis=1),
            lambda: rng.shuffle(dpnp.zeros(3), axis=1),
        )
        assert_raises(TypeError, rng.shuffle, [1, 2, 3])
        assert_raises(TypeError, rng.shuffle, numpy.arange(3))

    def test_permuted(self):
        np_rng = numpy.random.default_rng(0)
        rng = default_rng(0)
        _assert_same_error(
            lambda: np_rng.permuted(numpy.zeros(())),
            lambda: rng.permuted(dpnp.zeros(())),
        )
        _assert_same_error(
            lambda: np_rng.permuted(numpy.zeros(3), axis=1),
            lambda: rng.permuted(dpnp.zeros(3), axis=1),
        )
        _assert_same_error(
            lambda: np_rng.permuted(numpy.zeros(3), out=numpy.zeros(4)),
            lambda: rng.permuted(dpnp.zeros(3), out=dpnp.zeros(4)),
        )
        assert_raises(
            TypeError, rng.permuted, dpnp.zeros(3), out=numpy.zeros(3)
        )
        assert_raises(
            TypeError,
            rng.permuted,
            dpnp.zeros(3),
            out=dpnp.zeros(3, dtype=dpnp.int32),
        )

    def test_choice_axis(self):
        # NumPy raises IndexError, and AxisError is one
        assert_raises(IndexError, default_rng(0).choice, [1, 2], axis=1)

    def test_cholesky_not_positive_definite(self):
        rng = default_rng(0)
        assert_raises(
            dpnp.linalg.LinAlgError,
            rng.multivariate_normal,
            [0.0, 0.0],
            [[1.0, 2.0], [2.0, 1.0]],
            method="cholesky",
        )

    def test_not_positive_semidefinite_warns(self):
        rng = default_rng(0)
        cov = [[1.0, 2.0], [2.0, 1.0]]
        for method in ("svd", "eigh"):
            with pytest.warns(RuntimeWarning, match="positive-semidefinite"):
                rng.multivariate_normal([0.0, 0.0], cov, method=method)
            rng.multivariate_normal(
                [0.0, 0.0], cov, check_valid="ignore", method=method
            )


class TestMultivariate:
    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize(
        "alpha", [[0.5, 1.0, 2.0], [0.01, 0.05, 0.02], [3.0], [0.0, 1.0, 0.5]]
    )
    def test_dirichlet(self, alpha, dtype):
        n = 10**5
        res = default_rng(7).dirichlet(alpha, size=(n, 2), dtype=dtype)
        assert res.shape == (n, 2, len(alpha))
        assert res.dtype == dtype
        res = dpnp.asnumpy(res).reshape(-1, len(alpha)).astype(numpy.float64)
        assert ((res >= 0) & (res <= 1)).all()
        assert_allclose(res.sum(axis=-1), numpy.ones(2 * n), rtol=1e-5)

        alpha = numpy.asarray(alpha)
        a0 = alpha.sum()
        mean = alpha / a0
        var = mean * (1 - mean) / (a0 + 1)
        assert (
            abs(res.mean(axis=0) - mean) <= 5 * numpy.sqrt(var / (2 * n))
        ).all()

    @pytest.mark.parametrize(
        "alpha", [[0.0], [0.0, 0.0], [0.05, 0.0], [0.0, 0.0, 0.05], [0.0, 2.0]]
    )
    def test_dirichlet_degenerate(self, alpha):
        expected = numpy.random.default_rng(0).dirichlet(alpha)
        res = default_rng(0).dirichlet(alpha, size=3)
        assert_allclose(
            dpnp.asnumpy(res), numpy.tile(expected, (3, 1)), rtol=1e-15
        )

    def test_dirichlet_shapes(self):
        rng = default_rng(0)
        assert rng.dirichlet([1.0, 2.0]).shape == (2,)
        assert rng.dirichlet([]).shape == (0,)
        assert rng.dirichlet([], size=3).shape == (3, 0)
        assert rng.dirichlet(dpnp.asarray([1.0, 2.0]), size=3).shape == (3, 2)
        res = dpnp.asnumpy(rng.dirichlet([1.0, numpy.nan], size=3))
        assert numpy.isnan(res).all()

    def test_multinomial(self, discrete_precision):
        n = 10**5
        pvals = [0.2, 0.3, 0.0, 0.5]
        res = default_rng(8).multinomial(100, pvals, size=n)
        assert res.shape == (n, 4)
        assert res.dtype == dpnp.int64
        res = dpnp.asnumpy(res)
        assert (res.sum(axis=1) == 100).all()
        assert (res[:, 2] == 0).all()

        p = numpy.asarray(pvals)
        assert (
            abs(res.mean(axis=0) - 100 * p)
            <= 5 * numpy.sqrt(100 * p * (1 - p) / n) + 1e-12
        ).all()
        assert_allclose(
            res.var(axis=0), 100 * p * (1 - p), rtol=0.05, atol=1e-12
        )

    def test_multinomial_shapes(self, discrete_precision):
        rng = default_rng(9)
        assert rng.multinomial(5, [0.5, 0.5]).shape == (2,)
        assert rng.multinomial(5, [0.5, 0.5], size=(3, 2)).shape == (3, 2, 2)
        assert rng.multinomial([3, 5], [0.2, 0.8], size=(4, 2)).shape == (
            4,
            2,
            2,
        )

        n = [[1], [20]]
        pvals = [[0.5, 0.5], [0.2, 0.8], [0.0, 1.0]]
        for pv in (pvals, dpnp.asarray(pvals)):
            res = rng.multinomial(n, pv)
            assert res.shape == (2, 3, 2)
            res = dpnp.asnumpy(res)
            assert_array_equal(res.sum(axis=-1), numpy.broadcast_to(n, (2, 3)))
            assert (res[:, 2, 0] == 0).all()

        res = rng.multinomial(dpnp.asarray([4, 7]), [0.3, 0.7])
        assert_array_equal(dpnp.asnumpy(res).sum(axis=-1), [4, 7])
        assert_raises(ValueError, rng.multinomial, [3, 5], [0.2, 0.8], size=3)

    @pytest.mark.parametrize("method", ["marginals", "count"])
    @pytest.mark.parametrize("nsample", [0, 7, 25, 45, 60])
    def test_multivariate_hypergeometric(
        self, method, nsample, discrete_precision
    ):
        n = 10**5
        colors = numpy.array([10, 20, 0, 30])
        res = default_rng(10).multivariate_hypergeometric(
            colors, nsample, size=n, method=method
        )
        assert res.shape == (n, 4)
        assert res.dtype == dpnp.int64
        res = dpnp.asnumpy(res)
        assert (res.sum(axis=1) == nsample).all()
        assert ((res >= 0) & (res <= colors)).all()

        total = colors.sum()
        p = colors / total
        mean = nsample * p
        var = mean * (1 - p) * (total - nsample) / (total - 1)
        assert (
            abs(res.mean(axis=0) - mean) <= 5 * numpy.sqrt(var / n) + 1e-12
        ).all()

    def test_multivariate_hypergeometric_shapes(self):
        rng = default_rng(0)
        assert rng.multivariate_hypergeometric([1, 2], 2).shape == (2,)
        assert rng.multivariate_hypergeometric([], 0).shape == (0,)
        assert rng.multivariate_hypergeometric([], 0, size=3).shape == (3, 0)
        res = rng.multivariate_hypergeometric(
            dpnp.asarray([3, 4]), 7, size=(2, 3)
        )
        assert res.shape == (2, 3, 2)
        assert (dpnp.asnumpy(res) == [3, 4]).all()

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
    def test_multivariate_normal(self, method, dtype):
        n = 10**5
        mean = numpy.array([1.0, -2.0, 3.0])
        cov = numpy.array([[2.0, 0.5, 0.0], [0.5, 1.0, -0.3], [0.0, -0.3, 0.5]])
        res = default_rng(11).multivariate_normal(
            mean, cov, size=(n, 2), method=method, dtype=dtype
        )
        assert res.shape == (n, 2, 3)
        assert res.dtype == dtype
        res = dpnp.asnumpy(res).reshape(-1, 3).astype(numpy.float64)
        assert_allclose(res.mean(axis=0), mean, atol=0.02)
        assert_allclose(numpy.cov(res.T), cov, atol=0.02)

    def test_multivariate_normal_shapes(self):
        rng = default_rng(0)
        mean = dpnp.zeros(2)
        cov = dpnp.eye(2)
        assert rng.multivariate_normal(mean, cov).shape == (2,)
        assert rng.multivariate_normal(mean, cov, size=3).shape == (3, 2)
        assert rng.multivariate_normal([0.0], [[1.0]], size=0).shape == (0, 1)
        res = rng.multivariate_normal(
            [0.0, 0.0], [[1.0, 0.0], [0.0, 0.0]], size=100
        )
        assert (dpnp.asnumpy(res)[:, 1] == 0).all()
        res = rng.multivariate_normal(mean, cov, size=2, usm_type="host")
        assert res.usm_type == "host"


class TestPermutations:
    def test_permutation(self):
        rng = default_rng(12)
        res = rng.permutation(10)
        assert res.dtype == dpnp.int64
        assert_array_equal(numpy.sort(dpnp.asnumpy(res)), numpy.arange(10))
        assert rng.permutation(-3).shape == (0,)
        assert rng.permutation(0).shape == (0,)

        x = dpnp.arange(12).reshape(3, 4)
        res = rng.permutation(x)
        assert_array_equal(
            numpy.sort(dpnp.asnumpy(res), axis=0), dpnp.asnumpy(x)
        )
        # the rows are moved whole
        assert (
            dpnp.asnumpy(res)[:, 1:] - dpnp.asnumpy(res)[:, :1] == [1, 2, 3]
        ).all()

        res = dpnp.asnumpy(rng.permutation(x, axis=-1))
        assert (res[1:] - res[:1] == [[4], [8]]).all()

        res = rng.permutation([1.5, 2.5, 3.5])
        assert isinstance(res, dpnp.ndarray)
        assert_array_equal(numpy.sort(dpnp.asnumpy(res)), [1.5, 2.5, 3.5])

        # a copy is returned
        x = dpnp.arange(5)
        rng.permutation(x)[...] = 0
        assert_array_equal(x, dpnp.arange(5))

    def test_permutation_uniform(self):
        # all 6 permutations of 3 values are equally likely
        rng = default_rng(13)
        n = 6000
        keys = [tuple(dpnp.asnumpy(rng.permutation(3))) for _ in range(n)]
        _, counts = numpy.unique(keys, axis=0, return_counts=True)
        assert len(counts) == 6
        assert (abs(counts - n / 6) < 5 * numpy.sqrt(n / 6)).all()

    def test_shuffle(self):
        rng = default_rng(14)
        x = dpnp.arange(20)
        assert rng.shuffle(x) is None
        assert not (x == dpnp.arange(20)).all()
        assert_array_equal(numpy.sort(dpnp.asnumpy(x)), numpy.arange(20))

        x = dpnp.arange(12).reshape(3, 4)
        rng.shuffle(x, axis=1)
        res = dpnp.asnumpy(x)
        assert (res[1:] - res[:1] == [[4], [8]]).all()

    def test_permuted(self):
        rng = default_rng(15)
        x = dpnp.arange(24).reshape(4, 6)
        res = dpnp.asnumpy(rng.permuted(x))
        assert res.shape == (4, 6)
        assert_array_equal(numpy.sort(res, axis=None), numpy.arange(24))

        res = dpnp.asnumpy(rng.permuted(x, axis=1))
        assert_array_equal(numpy.sort(res, axis=1), dpnp.asnumpy(x))
        # the rows are shuffled independently
        assert len({tuple(r - r.min()) for r in res}) > 1

        out = dpnp.empty((4, 6), dtype=dpnp.float32)
        res = rng.permuted(x.astype(dpnp.int16), axis=0, out=out)
        assert res is out
        assert_array_equal(
            numpy.sort(dpnp.asnumpy(out), axis=0), dpnp.asnumpy(x), strict=False
        )

        # in place
        x = dpnp.arange(10)
        rng.permuted(x, out=x)
        assert_array_equal(numpy.sort(dpnp.asnumpy(x)), numpy.arange(10))

        res = rng.permuted([[1, 2], [3, 4]], axis=0)
        assert isinstance(res, dpnp.ndarray)


class TestChoice:
    def test_int(self):
        rng = default_rng(16)
        res = rng.choice(5, size=1000)
        assert res.dtype == dpnp.int64
        assert_array_equal(numpy.unique(dpnp.asnumpy(res)), numpy.arange(5))

        res = rng.choice(5)
        assert isinstance(res, dpnp.ndarray) and res.shape == ()
        assert rng.choice(5, size=()).shape == ()
        assert rng.choice(0, size=0).shape == (0,)
        assert rng.choice(5, size=(2, 3)).shape == (2, 3)

    def test_without_replacement(self):
        rng = default_rng(17)
        res = dpnp.asnumpy(rng.choice(100, size=(10, 10), replace=False))
        assert_array_equal(numpy.sort(res, axis=None), numpy.arange(100))
        res = dpnp.asnumpy(rng.choice(1000, size=5, replace=False))
        assert len(numpy.unique(res)) == 5

    def test_weighted(self):
        n = 10**5
        p = numpy.array([0.1, 0.0, 0.3, 0.6])
        rng = default_rng(18)
        for pp in (p, dpnp.asarray(p)):
            res = dpnp.asnumpy(rng.choice(4, size=n, p=pp))
            freq = numpy.bincount(res, minlength=4) / n
            assert (abs(freq - p) <= 5 * numpy.sqrt(p * (1 - p) / n)).all()

    def test_weighted_without_replacement(self):
        n = 20000
        p = numpy.array([0.1, 0.0, 0.3, 0.6])
        rng = default_rng(19)
        res = numpy.array(
            [
                dpnp.asnumpy(rng.choice(4, size=2, replace=False, p=p))
                for _ in range(n)
            ]
        )
        assert (res != 1).all()
        assert (res[:, 0] != res[:, 1]).all()

        # the first draw is from p, and the second from p conditioned on
        # not being the first
        freq = numpy.bincount(res[:, 0], minlength=4) / n
        assert (abs(freq - p) <= 5 * numpy.sqrt(p * (1 - p) / n)).all()
        pair = (res[:, 0] == 3) & (res[:, 1] == 2)
        prob = 0.6 * 0.3 / 0.4
        assert abs(pair.mean() - prob) <= 5 * numpy.sqrt(prob * (1 - prob) / n)

    def test_array(self):
        rng = default_rng(20)
        a = dpnp.asarray([10, 20, 30])
        res = rng.choice(a, size=100)
        assert set(dpnp.asnumpy(res)) <= {10, 20, 30}
        res = rng.choice(a)
        assert res.shape == ()

        a = numpy.arange(12).reshape(3, 4)
        res = dpnp.asnumpy(rng.choice(a, size=5))
        assert res.shape == (5, 4)
        assert (res[:, 1:] - res[:, :1] == [1, 2, 3]).all()
        res = dpnp.asnumpy(rng.choice(a, size=(2, 2), axis=1, replace=False))
        assert res.shape == (3, 2, 2)
        assert_array_equal(numpy.sort(res[0], axis=None), a[0])
        assert rng.choice(a, axis=1).shape == (3,)


@pytest.mark.skipif(len(_all_devices()) < 2, reason="needs two devices")
class TestDeviceIndependence:
    def _draw(self, device):
        rng = default_rng(12, device=device)
        return (
            rng.random(1000, dtype=dpnp.float32),
            rng.integers(0, 1000, size=1000),
            rng.bit_generator.random_raw(1000),
            rng.normal(size=1000, dtype=dpnp.float32),
            rng.gamma(0.7, size=1000, dtype=dpnp.float32),
            rng.poisson([3.0, 50.0], size=(500, 2)),
            rng.binomial(100, 0.3, size=1000),
            rng.zipf(2.0, size=1000),
            rng.hypergeometric(50, 70, 30, size=1000),
        )

    def test_same_values(self):
        devices = _all_devices()
        ref = [dpnp.asnumpy(x) for x in self._draw(devices[0])]
        for dev in devices[1:]:
            res = [dpnp.asnumpy(x) for x in self._draw(dev)]
            # uniform bits and integers are exact
            for a, b in zip(ref[:3], res[:3]):
                assert_array_equal(a, b)
            # transcendental functions may differ in the last bits
            for a, b in zip(ref[3:5], res[3:5]):
                assert_allclose(a, b, rtol=1e-4, atol=1e-5)
            # and so may flip a rejection test, which shifts the rest of
            # the work-item's stream
            for a, b in zip(ref[5:], res[5:]):
                assert (a != b).mean() < 0.05

    def test_queue(self):
        for dev in _all_devices():
            rng = default_rng(0, device=dev)
            res = rng.random(3, dtype=dpnp.float32)
            assert res.sycl_device == dev
            assert res.sycl_queue == rng.sycl_queue
