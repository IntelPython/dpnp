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

    @pytest.mark.parametrize("name", ["poisson", "binomial", "shuffle"])
    def test_not_implemented(self, name):
        rng = default_rng(0)
        assert_raises(NotImplementedError, getattr(rng, name), 1)


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
        ],
    )
    def test_moments(self, name, params, mean, var, dtype):
        n = 10**6
        res = getattr(default_rng(2), name)(*params, size=n, dtype=dtype)
        res = dpnp.asnumpy(res).astype(numpy.float64)

        # five standard errors of the mean and of the variance, taking a
        # generous bound on the fourth central moment
        assert abs(res.mean() - mean) < 5 * numpy.sqrt(var / n)
        assert abs(res.var() - var) < 5 * var * numpy.sqrt(10 / n)

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
            for a, b in zip(ref[3:], res[3:]):
                assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_queue(self):
        for dev in _all_devices():
            rng = default_rng(0, device=dev)
            res = rng.random(3, dtype=dpnp.float32)
            assert res.sycl_device == dev
            assert res.sycl_queue == rng.sycl_queue
