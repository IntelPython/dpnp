import dpctl
import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_raises,
    assert_raises_regex,
)

import dpnp
from dpnp.random import Generator, Philox4x3210, default_rng

from .helper import get_float_dtypes

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

    def test_array_params(self):
        rng = default_rng(3)
        loc = dpnp.arange(4.0)[:, None] * 10
        res = rng.normal(loc, 1.0, size=(4, 10000))
        assert res.shape == (4, 10000)
        assert_allclose(
            dpnp.asnumpy(res.mean(axis=1)), [0.0, 10.0, 20.0, 30.0], atol=0.1
        )

        # parameters given as lists
        res = rng.uniform([0.0, 10.0], [1.0, 11.0], size=(1000, 2))
        res = dpnp.asnumpy(res)
        assert (res[:, 0] < 1).all() and (res[:, 1] >= 10).all()

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
        ],
    )
    def test_scalar_checks(self, name, params, msg):
        rng = default_rng(0)
        assert_raises_regex(ValueError, msg, getattr(rng, name), *params)

    def test_uniform_overflow(self):
        rng = default_rng(0)
        big = numpy.finfo(numpy.float64).max
        assert_raises(OverflowError, rng.uniform, -big, big)

    @pytest.mark.parametrize("dtype", float_dtypes)
    @pytest.mark.parametrize(
        "name, params",
        [
            ("random", ()),
            ("standard_normal", ()),
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
            ("normal", (0.0, None)),
            ("uniform", (None, 1.0)),
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
