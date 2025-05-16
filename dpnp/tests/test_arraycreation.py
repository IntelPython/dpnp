import tempfile
from math import prod

import dpctl
import dpctl.tensor as dpt
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

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_array,
)
from .third_party.cupy import testing


class TestArray:
    @pytest.mark.parametrize(
        "x",
        [numpy.ones(5), numpy.ones((3, 4)), numpy.ones((0, 4)), [1, 2, 3], []],
    )
    @pytest.mark.parametrize("ndmin", [-5, -1, 0, 1, 2, 3, 4, 9, 21])
    def test_ndmin(self, x, ndmin):
        a = numpy.array(x, ndmin=ndmin)
        ia = dpnp.array(x, ndmin=ndmin)
        assert_array_equal(ia, a)

    @pytest.mark.parametrize(
        "x",
        [
            numpy.ones((2, 3, 4, 5)),
            numpy.ones((3, 4)),
            numpy.ones((0, 4)),
            [1, 2, 3],
            [],
        ],
    )
    @pytest.mark.parametrize("order", ["C", "F", "K", "A"])
    @pytest.mark.parametrize("ndmin", [1, 2, 3, 4, 9, 21])
    def test_ndmin_order(self, x, order, ndmin):
        a = numpy.array(x, order=order, ndmin=ndmin)
        ia = dpnp.array(x, order=order, ndmin=ndmin)
        assert a.flags.c_contiguous == ia.flags.c_contiguous
        assert a.flags.f_contiguous == ia.flags.f_contiguous
        assert_array_equal(ia, a)

    def test_error(self):
        x = numpy.ones((3, 4))
        assert_raises(TypeError, dpnp.array, x, ndmin=3.0)


class TestAsType:
    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_validate_positional_args(self, xp):
        x = xp.ones(4)
        assert_raises_regex(
            TypeError,
            "got some positional-only arguments passed as keyword arguments",
            xp.astype,
            x,
            dtype="f4",
        )
        assert_raises_regex(
            TypeError,
            "takes 2 positional arguments but 3 were given",
            xp.astype,
            x,
            "f4",
            None,
        )


class TestTrace:
    @pytest.mark.parametrize("a_sh", [(3, 4), (2, 2, 2)])
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    @pytest.mark.parametrize("offset_arg", [[], [0], [1], [-1]])
    def test_offset(self, a_sh, dtype, offset_arg):
        a = numpy.arange(prod(a_sh), dtype=dtype).reshape(a_sh)
        ia = dpnp.array(a)
        assert_equal(ia.trace(*offset_arg), a.trace(*offset_arg))

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_none_offset(self, xp):
        a = xp.arange(12).reshape((3, 4))
        with pytest.raises(TypeError):
            a.trace(offset=None)

    @pytest.mark.parametrize(
        "offset, axis1, axis2",
        [
            pytest.param(0, 0, 1),
            pytest.param(0, 0, 2),
            pytest.param(0, 1, 2),
            pytest.param(1, 0, 2),
        ],
    )
    def test_axis(self, offset, axis1, axis2):
        a = numpy.arange(8).reshape((2, 2, 2))
        ia = dpnp.array(a)

        expected = a.trace(offset=offset, axis1=axis1, axis2=axis2)
        result = ia.trace(offset=offset, axis1=axis1, axis2=axis2)
        assert_equal(result, expected)

    def test_out(self):
        a = numpy.arange(12).reshape((3, 4))
        out = numpy.array(1)

        ia = dpnp.array(a)
        iout = dpnp.array(out)

        expected = a.trace(out=out)
        result = ia.trace(out=iout)
        assert_equal(result, expected)
        assert result is iout

    @testing.with_requires("numpy>=2.0")
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    @pytest.mark.parametrize("offset", [0, 1, -1])
    def test_linalg_trace(self, dtype, offset):
        a = numpy.arange(12, dtype=dtype).reshape(3, 4)
        ia = dpnp.array(a)
        result = dpnp.linalg.trace(ia, offset=offset, dtype=dtype)
        expected = numpy.linalg.trace(a, offset=offset, dtype=dtype)
        assert_equal(result, expected)


@pytest.mark.parametrize(
    "func, args",
    [
        pytest.param("empty_like", [dpnp.ones(10)]),
        pytest.param("full_like", [dpnp.ones(10), 7]),
        pytest.param("ones_like", [dpnp.ones(10)]),
        pytest.param("zeros_like", [dpnp.ones(10)]),
        pytest.param("empty", [3]),
        pytest.param("eye", [3]),
        pytest.param("full", [3, 7]),
        pytest.param("ones", [3]),
        pytest.param("zeros", [3]),
    ],
)
def test_exception_order(func, args):
    with pytest.raises(ValueError):
        getattr(dpnp, func)(*args, order="S")


@pytest.mark.parametrize(
    "func, args",
    [
        pytest.param("arange", [2]),
        pytest.param("array", [2]),
        pytest.param("asanyarray", [2]),
        pytest.param("asarray", [2]),
        pytest.param("ascontiguousarray", [2]),
        pytest.param("asfortranarray", [2]),
        pytest.param("empty", [(2,)]),
        pytest.param("eye", [2]),
        pytest.param("frombuffer", [b"\x01\x02\x03\x04"]),
        pytest.param("full", [(2,), 4]),
        pytest.param("identity", [2]),
        pytest.param("ones", [(2,)]),
        pytest.param("zeros", [(2,)]),
    ],
)
def test_exception_like(func, args):
    like = dpnp.array([1, 2])
    with pytest.raises(NotImplementedError):
        getattr(dpnp, func)(*args, like=like)


@pytest.mark.parametrize(
    "func, args",
    [
        pytest.param("array", []),
        pytest.param("copy", []),
        pytest.param("empty_like", []),
        pytest.param("full_like", [5]),
        pytest.param("ones_like", []),
        pytest.param("zeros_like", []),
    ],
)
def test_exception_subok(func, args):
    x = dpnp.ones((3,))
    with pytest.raises(NotImplementedError):
        getattr(dpnp, func)(x, *args, subok=True)


@pytest.mark.parametrize("start", [0, -5, 10, -2.5, 9.7])
@pytest.mark.parametrize("stop", [None, 10, -2, 20.5, 100])
@pytest.mark.parametrize("step", [None, 1, 2.7, -1.6, 80])
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_float16=False)
)
def test_arange(start, stop, step, dtype):
    if numpy.issubdtype(dtype, numpy.unsignedinteger):
        start = abs(start)
        stop = abs(stop) if stop else None

    # numpy casts to float32 type when computes float16 data
    rtol_mult = 4 if dpnp.issubdtype(dtype, dpnp.float16) else 2

    func = lambda xp: xp.arange(start, stop=stop, step=step, dtype=dtype)

    exp_array = func(numpy)
    res_array = func(dpnp)

    if dtype is None:
        _device = dpctl.SyclQueue().sycl_device
        if not _device.has_aspect_fp64:
            # numpy allocated array with dtype=float64 by default,
            # while dpnp might use float32, if float64 isn't supported by device
            _dtype = dpnp.float32
            rtol_mult *= 150
        else:
            _dtype = dpnp.float64
    else:
        _dtype = dtype

    if dpnp.issubdtype(_dtype, dpnp.floating) or dpnp.issubdtype(
        _dtype, dpnp.complexfloating
    ):
        assert_allclose(
            res_array, exp_array, rtol=rtol_mult * numpy.finfo(_dtype).eps
        )
    else:
        assert_array_equal(exp_array, res_array)


@pytest.mark.parametrize("func", ["diag", "diagflat"])
@pytest.mark.parametrize("k", [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "v",
    [
        [0, 1, 2, 3, 4],
        [1, 1, 1, 1, 1],
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    ],
    ids=[
        "[0, 1, 2, 3, 4]",
        "[1, 1, 1, 1, 1]",
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
        "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]",
    ],
)
def test_diag_diagflat(func, v, k):
    a = numpy.array(v)
    ia = dpnp.array(a)
    expected = getattr(numpy, func)(a, k)
    result = getattr(dpnp, func)(ia, k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("func", ["diag", "diagflat"])
def test_diag_diagflat_raise_error(func):
    ia = dpnp.array([0, 1, 2, 3, 4])
    with pytest.raises(TypeError):
        getattr(dpnp, func)(ia, k=2.0)


@pytest.mark.parametrize("func", ["diag", "diagflat"])
@pytest.mark.parametrize(
    "seq",
    [
        [0, 1, 2, 3, 4],
        (0, 1, 2, 3, 4),
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    ],
    ids=[
        "[0, 1, 2, 3, 4]",
        "(0, 1, 2, 3, 4)",
        "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]",
    ],
)
def test_diag_diagflat_seq(func, seq):
    expected = getattr(numpy, func)(seq)
    result = getattr(dpnp, func)(seq)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("N", [0, 1, 2, 3])
@pytest.mark.parametrize("M", [None, 0, 1, 2, 3])
@pytest.mark.parametrize("k", [-4, -3, -2, -1, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order", [None, "C", "F"])
def test_eye(N, M, k, dtype, order):
    func = lambda xp: xp.eye(N, M, k=k, dtype=dtype, order=order)
    assert_array_equal(func(numpy), func(dpnp))


@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_frombuffer(dtype):
    buffer = b"12345678ABCDEF00"
    func = lambda xp: xp.frombuffer(buffer, dtype=dtype)
    assert_dtype_allclose(func(dpnp), func(numpy))


@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_fromfile(dtype):
    with tempfile.TemporaryFile() as fh:
        fh.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08")
        fh.flush()

        func = lambda xp: xp.fromfile(fh, dtype=dtype)

        fh.seek(0)
        np_res = func(numpy)

        fh.seek(0)
        dpnp_res = func(dpnp)

    assert_dtype_allclose(dpnp_res, np_res)


@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_bool=True, no_float16=False)
)
def test_fromfunction(dtype):
    def func(x, y):
        return x * y

    shape = (3, 3)
    call_func = lambda xp: xp.fromfunction(func, shape=shape, dtype=dtype)
    assert_array_equal(call_func(dpnp), call_func(numpy))


@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_fromiter(dtype):
    _iter = [1, 2, 3, 4]
    func = lambda xp: xp.fromiter(_iter, dtype=dtype)
    assert_array_equal(func(dpnp), func(numpy))


@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_fromstring(dtype):
    string = "1 2 3 4"
    func = lambda xp: xp.fromstring(string, dtype=dtype, sep=" ")
    assert_array_equal(func(dpnp), func(numpy))


@pytest.mark.parametrize("n", [0, 1, 4])
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_identity(n, dtype):
    func = lambda xp: xp.identity(n, dtype=dtype)
    assert_array_equal(func(numpy), func(dpnp))


def test_identity_error():
    # negative dimensions
    assert_raises(ValueError, dpnp.identity, -5)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_loadtxt(dtype):
    func = lambda xp: xp.loadtxt(fh, dtype=dtype)

    with tempfile.TemporaryFile() as fh:
        fh.write(b"1 2 3 4")
        fh.flush()

        fh.seek(0)
        np_res = func(numpy)
        fh.seek(0)
        dpnp_res = func(dpnp)

    assert_array_equal(dpnp_res, np_res)


@pytest.mark.parametrize("N", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("M", [None, 0, 1, 2, 3, 4])
@pytest.mark.parametrize("k", [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_tri(N, M, k, dtype):
    func = lambda xp: xp.tri(N, M, k, dtype=dtype)
    assert_array_equal(func(dpnp), func(numpy))


def test_tri_default_dtype():
    expected = numpy.tri(3, 5, -1)
    result = dpnp.tri(3, 5, -1)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "k",
    [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        numpy.array(1),
        dpnp.array(2),
        dpt.asarray(3),
    ],
    ids=[
        "-3",
        "-2",
        "-1",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "np.array(1)",
        "dpnp.array(2)",
        "dpt.asarray(3)",
    ],
)
@pytest.mark.parametrize(
    "m",
    [
        [[0, 0], [0, 0]],
        [[1, 2], [1, 2]],
        [[1, 2], [3, 4]],
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    ],
    ids=[
        "[[0, 0], [0, 0]]",
        "[[1, 2], [1, 2]]",
        "[[1, 2], [3, 4]]",
        "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]",
    ],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_tril(m, k, dtype):
    a = numpy.array(m, dtype=dtype)
    ia = dpnp.array(a)
    expected = numpy.tril(a, k=k)
    result = dpnp.tril(ia, k=k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "k",
    [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        numpy.array(1),
        dpnp.array(2),
        dpt.asarray(3),
    ],
    ids=[
        "-3",
        "-2",
        "-1",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "np.array(1)",
        "dpnp.array(2)",
        "dpt.asarray(3)",
    ],
)
@pytest.mark.parametrize(
    "m",
    [
        [[1, 2], [3, 4]],
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    ],
    ids=[
        "[[1, 2], [3, 4]]",
        "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]",
        "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]",
    ],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_triu(m, k, dtype):
    a = numpy.array(m, dtype=dtype)
    ia = dpnp.array(a)
    expected = numpy.triu(a, k=k)
    result = dpnp.triu(ia, k=k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("k", [-4, -3, -2, -1, 0, 1, 2, 3, 4])
def test_triu_size_null(k):
    a = numpy.ones(shape=(1, 2, 0))
    ia = dpnp.array(a)
    expected = numpy.triu(a, k=k)
    result = dpnp.triu(ia, k=k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "array",
    [[1, 2, 3, 4], [], [0, 3, 5]],
    ids=["[1, 2, 3, 4]", "[]", "[0, 3, 5]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize("n", [0, 1, 4, None])
@pytest.mark.parametrize("increase", [True, False])
def test_vander(array, dtype, n, increase):
    if dtype in [dpnp.complex64, dpnp.complex128] and array == [0, 3, 5]:
        pytest.skip(
            "per array API dpnp.power(complex(0, 0)), 0) returns nan+nanj while NumPy returns 1+0j"
        )
    vander_func = lambda xp, x: xp.vander(x, N=n, increasing=increase)

    a_np = numpy.array(array, dtype=dtype)
    a_dpnp = dpnp.array(array, dtype=dtype)

    assert_allclose(vander_func(dpnp, a_dpnp), vander_func(numpy, a_np))


def test_vander_raise_error():
    a = dpnp.array([1, 2, 3, 4])
    with pytest.raises(TypeError):
        dpnp.vander(a, N=1.0)

    a = dpnp.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        dpnp.vander(a)


@pytest.mark.parametrize(
    "sequence",
    [[1, 2, 3, 4], (1, 2, 3, 4)],
    ids=["[1, 2, 3, 4]", "(1, 2, 3, 4)"],
)
def test_vander_seq(sequence):
    vander_func = lambda xp, x: xp.vander(x)
    assert_allclose(vander_func(dpnp, sequence), vander_func(numpy, sequence))


@pytest.mark.usefixtures("suppress_complex_warning")
@pytest.mark.parametrize(
    "shape",
    [(), 0, (0,), (2, 0, 3), (3, 2)],
    ids=["()", "0", "(0,)", "(2, 0, 3)", "(3, 2)"],
)
@pytest.mark.parametrize(
    "fill_value", [1.5, 2, 1.5 + 0.0j], ids=["1.5", "2", "1.5+0.j"]
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order", [None, "C", "F"])
def test_full(shape, fill_value, dtype, order):
    func = lambda xp: xp.full(shape, fill_value, dtype=dtype, order=order)
    assert_array_equal(func(numpy), func(dpnp))


@pytest.mark.usefixtures("suppress_complex_warning")
@pytest.mark.parametrize(
    "array",
    [[], 0, [1, 2, 3], [[1, 2], [3, 4]]],
    ids=["[]", "0", "[1, 2, 3]", "[[1, 2], [3, 4]]"],
)
@pytest.mark.parametrize(
    "fill_value", [1.5, 2, 1.5 + 0.0j], ids=["1.5", "2", "1.5+0.j"]
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order", [None, "C", "F", "A", "K"])
def test_full_like(array, fill_value, dtype, order):
    func = lambda xp, x: xp.full_like(x, fill_value, dtype=dtype, order=order)

    a = numpy.array(array)
    ia = dpnp.array(array)
    assert_array_equal(func(numpy, a), func(dpnp, ia))


@pytest.mark.parametrize("order1", ["F", "C"])
@pytest.mark.parametrize("order2", ["F", "C"])
def test_full_order(order1, order2):
    array = numpy.array([1, 2, 3], order=order1)
    a = numpy.full((3, 3), array, order=order2)
    ia = dpnp.full((3, 3), array, order=order2)

    assert ia.flags.c_contiguous == a.flags.c_contiguous
    assert ia.flags.f_contiguous == a.flags.f_contiguous
    assert_equal(ia, a)


def test_full_strides():
    a = numpy.full((3, 3), numpy.arange(3, dtype="i4"))
    ia = dpnp.full((3, 3), dpnp.arange(3, dtype="i4"))
    assert ia.strides == tuple(el // a.itemsize for el in a.strides)
    assert_array_equal(ia, a)

    a = numpy.full((3, 3), numpy.arange(6, dtype="i4")[::2])
    ia = dpnp.full((3, 3), dpnp.arange(6, dtype="i4")[::2])
    assert ia.strides == tuple(el // a.itemsize for el in a.strides)
    assert_array_equal(ia, a)


@pytest.mark.parametrize(
    "fill_value", [[], (), dpnp.full(0, 0)], ids=["[]", "()", "dpnp.full(0, 0)"]
)
def test_full_invalid_fill_value(fill_value):
    with pytest.raises(ValueError):
        dpnp.full(10, fill_value=fill_value)


@pytest.mark.parametrize(
    "shape",
    [(), 0, (0,), (2, 0, 3), (3, 2)],
    ids=["()", "0", "(0,)", "(2, 0, 3)", "(3, 2)"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order", [None, "C", "F"])
def test_zeros(shape, dtype, order):
    func = lambda xp: xp.zeros(shape, dtype=dtype, order=order)
    assert_array_equal(func(numpy), func(dpnp))


@pytest.mark.parametrize(
    "array",
    [[], 0, [1, 2, 3], [[1, 2], [3, 4]]],
    ids=["[]", "0", "[1, 2, 3]", "[[1, 2], [3, 4]]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order", [None, "C", "F", "A", "K"])
def test_zeros_like(array, dtype, order):
    func = lambda xp, x: xp.zeros_like(x, dtype=dtype, order=order)

    a = numpy.array(array)
    ia = dpnp.array(array)
    assert_array_equal(func(numpy, a), func(dpnp, ia))


@pytest.mark.parametrize(
    "shape",
    [(), 0, (0,), (2, 0, 3), (3, 2)],
    ids=["()", "0", "(0,)", "(2, 0, 3)", "(3, 2)"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order", [None, "C", "F"])
def test_empty(shape, dtype, order):
    func = lambda xp: xp.empty(shape, dtype=dtype, order=order)
    assert func(numpy).shape == func(dpnp).shape


@pytest.mark.parametrize(
    "array",
    [[], 0, [1, 2, 3], [[1, 2], [3, 4]]],
    ids=["[]", "0", "[1, 2, 3]", "[[1, 2], [3, 4]]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order", [None, "C", "F", "A", "K"])
def test_empty_like(array, dtype, order):
    func = lambda xp, x: xp.empty_like(x, dtype=dtype, order=order)

    a = numpy.array(array)
    ia = dpnp.array(array)
    assert func(numpy, a).shape == func(dpnp, ia).shape


@pytest.mark.parametrize(
    "shape",
    [(), 0, (0,), (2, 0, 3), (3, 2)],
    ids=["()", "0", "(0,)", "(2, 0, 3)", "(3, 2)"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order", [None, "C", "F"])
def test_ones(shape, dtype, order):
    func = lambda xp: xp.ones(shape, dtype=dtype, order=order)
    assert_array_equal(func(numpy), func(dpnp))


@pytest.mark.parametrize(
    "array",
    [[], 0, [1, 2, 3], [[1, 2], [3, 4]]],
    ids=["[]", "0", "[1, 2, 3]", "[[1, 2], [3, 4]]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order", [None, "C", "F", "A", "K"])
def test_ones_like(array, dtype, order):
    func = lambda xp, x: xp.ones_like(x, dtype=dtype, order=order)

    a = numpy.array(array)
    ia = dpnp.array(array)
    assert_array_equal(func(numpy, a), func(dpnp, ia))


@pytest.mark.parametrize(
    "func, args",
    [
        pytest.param("full_like", ["x0", "4"]),
        pytest.param("zeros_like", ["x0"]),
        pytest.param("ones_like", ["x0"]),
        pytest.param("empty_like", ["x0"]),
    ],
)
def test_dpctl_tensor_input(func, args):
    x0 = dpt.reshape(dpt.arange(9), (3, 3))
    new_args = [eval(val, {"x0": x0}) for val in args]
    X = getattr(dpt, func)(*new_args)
    Y = getattr(dpnp, func)(*new_args)
    if func == "empty_like":
        assert X.shape == Y.shape
    else:
        assert_array_equal(X, Y)


@pytest.mark.parametrize("start", [0, -5, 10, -2.5, 9.7])
@pytest.mark.parametrize("stop", [0, 10, -2, 20.5, 120])
@pytest.mark.parametrize(
    "num",
    [1, 5, numpy.array(10), dpnp.array(17), dpt.asarray(100)],
    ids=["1", "5", "numpy.array(10)", "dpnp.array(17)", "dpt.asarray(100)"],
)
@pytest.mark.parametrize(
    "dtype",
    get_all_dtypes(no_bool=True, no_float16=False),
)
@pytest.mark.parametrize("retstep", [True, False])
def test_linspace(start, stop, num, dtype, retstep):
    if numpy.issubdtype(dtype, numpy.unsignedinteger):
        start = abs(start)
        stop = abs(stop)

    res_np = numpy.linspace(start, stop, num, dtype=dtype, retstep=retstep)
    res_dp = dpnp.linspace(start, stop, num, dtype=dtype, retstep=retstep)

    if retstep:
        [res_np, step_np] = res_np
        [res_dp, step_dp] = res_dp
        assert_allclose(step_np, step_dp)

    if numpy.issubdtype(dtype, dpnp.integer):
        assert_allclose(res_np, res_dp, rtol=1)
    else:
        assert_dtype_allclose(res_dp, res_np)


@pytest.mark.parametrize("func", ["geomspace", "linspace", "logspace"])
@pytest.mark.parametrize(
    "start_dtype", [numpy.float64, numpy.float32, numpy.int64, numpy.int32]
)
@pytest.mark.parametrize(
    "stop_dtype", [numpy.float64, numpy.float32, numpy.int64, numpy.int32]
)
def test_space_numpy_dtype(func, start_dtype, stop_dtype):
    start = numpy.array([1, 2, 3], dtype=start_dtype)
    stop = numpy.array([11, 7, -2], dtype=stop_dtype)
    getattr(dpnp, func)(start, stop, 10)


@pytest.mark.parametrize(
    "start",
    [
        dpnp.array(1),
        dpnp.array([2.6]),
        numpy.array([[-6.7, 3]]),
        [1, -4],
        (3, 5),
    ],
)
@pytest.mark.parametrize(
    "stop",
    [
        dpnp.array([-4]),
        dpnp.array([[2.6], [-4]]),
        numpy.array(2),
        [[-4.6]],
        (3,),
    ],
)
def test_linspace_arrays(start, stop):
    func = lambda xp: xp.linspace(get_array(xp, start), get_array(xp, stop), 10)
    assert func(numpy).shape == func(dpnp).shape


def test_linspace_complex():
    func = lambda xp: xp.linspace(0, 3 + 2j, num=1000)
    assert_allclose(func(dpnp), func(numpy))


@pytest.mark.parametrize("axis", [0, 1])
def test_linspace_axis(axis):
    func = lambda xp: xp.linspace([2, 3], [20, 15], num=10, axis=axis)
    assert_allclose(func(dpnp), func(numpy))


def test_linspace_step_nan():
    func = lambda xp: xp.linspace(1, 2, num=0, endpoint=False)
    assert_allclose(func(dpnp), func(numpy))


@pytest.mark.parametrize("start", [1, [1, 1]])
@pytest.mark.parametrize("stop", [10, [10 + 10]])
def test_linspace_retstep(start, stop):
    func = lambda xp: xp.linspace(start, stop, num=10, retstep=True)
    np_res = func(numpy)
    dpnp_res = func(dpnp)
    assert_allclose(dpnp_res[0], np_res[0])
    assert_allclose(dpnp_res[1], np_res[1])


@pytest.mark.parametrize(
    "arrays",
    [[], [[1]], [[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4], [5, 6]]],
    ids=["[]", "[[1]]", "[[1, 2, 3], [4, 5, 6]]", "[[1, 2], [3, 4], [5, 6]]"],
)
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("indexing", ["ij", "xy"])
def test_meshgrid(arrays, dtype, indexing):
    func = lambda xp, xi: xp.meshgrid(*xi, indexing=indexing)
    a = tuple(numpy.array(array, dtype=dtype) for array in arrays)
    ia = tuple(dpnp.array(array, dtype=dtype) for array in arrays)
    assert_array_equal(func(numpy, a), func(dpnp, ia))


@pytest.mark.parametrize("shape", [(24,), (4, 6), (2, 3, 4), (2, 3, 2, 2)])
def test_set_shape(shape):
    na = numpy.arange(24)
    na.shape = shape
    da = dpnp.arange(24)
    da.shape = shape

    assert_array_equal(na, da)


def test_geomspace_zero_error():
    with pytest.raises(ValueError):
        dpnp.geomspace(0, 5, 3)
        dpnp.geomspace(2, 0, 3)
        dpnp.geomspace(0, 0, 3)


def test_space_num_error():
    with pytest.raises(ValueError):
        dpnp.linspace(2, 5, -3)
        dpnp.geomspace(2, 5, -3)
        dpnp.logspace(2, 5, -3)
        dpnp.linspace([2, 3], 5, -3)
        dpnp.geomspace([2, 3], 5, -3)
        dpnp.logspace([2, 3], 5, -3)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize("num", [2, 4, 8, 3, 9, 27])
@pytest.mark.parametrize("endpoint", [True, False])
def test_geomspace(sign, dtype, num, endpoint):
    start = 2 * sign
    stop = 127 * sign

    func = lambda xp: xp.geomspace(
        start, stop, num, endpoint=endpoint, dtype=dtype
    )

    np_res = func(numpy)
    dpnp_res = func(dpnp)

    assert_allclose(dpnp_res, np_res, rtol=1e-06)


@pytest.mark.parametrize("start", [1j, 1 + 1j])
@pytest.mark.parametrize("stop", [10j, 10 + 10j])
def test_geomspace_complex(start, stop):
    func = lambda xp: xp.geomspace(start, stop, num=10)
    np_res = func(numpy)
    dpnp_res = func(dpnp)
    assert_allclose(dpnp_res, np_res, rtol=1e-06)


@pytest.mark.parametrize("axis", [0, 1])
def test_geomspace_axis(axis):
    func = lambda xp: xp.geomspace([2, 3], [20, 15], num=10, axis=axis)
    np_res = func(numpy)
    dpnp_res = func(dpnp)
    assert_allclose(dpnp_res, np_res, rtol=1e-06)


def test_geomspace_num0():
    func = lambda xp: xp.geomspace(1, 10, num=0, endpoint=False)
    np_res = func(numpy)
    dpnp_res = func(dpnp)
    assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize("num", [2, 4, 8, 3, 9, 27])
@pytest.mark.parametrize("endpoint", [True, False])
def test_logspace(dtype, num, endpoint):
    start = 2
    stop = 5
    base = 2

    func = lambda xp: xp.logspace(
        start, stop, num, endpoint=endpoint, dtype=dtype, base=base
    )

    np_res = func(numpy)
    dpnp_res = func(dpnp)

    assert_allclose(dpnp_res, np_res, rtol=1e-06)


@pytest.mark.parametrize("axis", [0, 1])
def test_logspace_axis(axis):
    if numpy.lib.NumpyVersion(numpy.__version__) < "1.25.0":
        pytest.skip(
            "numpy.logspace supports a non-scalar base argument since 1.25.0"
        )
    func = lambda xp: xp.logspace(
        [2, 3], [20, 15], num=2, base=[[1, 3], [5, 7]], axis=axis
    )
    assert_dtype_allclose(func(dpnp), func(numpy))


def test_logspace_list_input():
    expected = numpy.logspace([0], [2], base=[5])
    result = dpnp.logspace([0], [2], base=[5])
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize(
    "data", [(), 1, (2, 3), [4], numpy.array(5), numpy.array([6, 7])]
)
def test_ascontiguousarray1(data):
    result = dpnp.ascontiguousarray(data)
    expected = numpy.ascontiguousarray(data)
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("data", [(), 1, (2, 3), [4]])
def test_ascontiguousarray2(data):
    result = dpnp.ascontiguousarray(dpnp.array(data))
    expected = numpy.ascontiguousarray(numpy.array(data))
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize(
    "data", [(), 1, (2, 3), [4], numpy.array(5), numpy.array([6, 7])]
)
def test_asfortranarray1(data):
    result = dpnp.asfortranarray(data)
    expected = numpy.asfortranarray(data)
    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("data", [(), 1, (2, 3), [4]])
def test_asfortranarray2(data):
    result = dpnp.asfortranarray(dpnp.array(data))
    expected = numpy.asfortranarray(numpy.array(data))
    assert_dtype_allclose(result, expected)


def test_meshgrid_raise_error():
    a = numpy.array([1, 2, 3, 4])
    with pytest.raises(TypeError):
        dpnp.meshgrid(a)
    b = dpnp.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        dpnp.meshgrid(b, indexing="ab")


class TestMgrid:
    def check_results(self, result, expected):
        if isinstance(result, (list, tuple)):
            assert len(result) == len(expected)
            for dp_arr, np_arr in zip(result, expected):
                assert_allclose(dp_arr, np_arr)
        else:
            assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "slice",
        [
            slice(0, 5, 0.5),  # float step
            slice(0, 5, 1j),  # complex step
            slice(0, 5, 5j),  # complex step
            slice(None, 5, 1),  # no start
            slice(0, 5, None),  # no step
        ],
    )
    def test_single_slice(self, slice):
        dpnp_result = dpnp.mgrid[slice]
        numpy_result = numpy.mgrid[slice]
        self.check_results(dpnp_result, numpy_result)

    @pytest.mark.parametrize(
        "slices",
        [
            (slice(None, 5, 1), slice(None, 10, 2)),  # no start
            (slice(0, 5), slice(0, 10)),  # no step
            (slice(0, 5.5, 1), slice(0, 10, 3j)),  # float stop and complex step
            (
                slice(0.0, 5, 1),
                slice(0, 10, 1j),
            ),  # float start and complex step
        ],
    )
    def test_md_slice(self, slices):
        dpnp_result = dpnp.mgrid[slices]
        numpy_result = numpy.mgrid[slices]
        self.check_results(dpnp_result, numpy_result)


def test_exception_tri():
    x = dpnp.ones((2, 2))
    with pytest.raises(TypeError):
        dpnp.tri(x)
    with pytest.raises(TypeError):
        dpnp.tri(1, x)
    with pytest.raises(TypeError):
        dpnp.tri(1, 1, k=1.2)
    with pytest.raises(TypeError):
        dpnp.tril(x, k=1.2)
    with pytest.raises(TypeError):
        dpnp.triu(x, k=1.2)
    with pytest.raises(TypeError):
        dpnp.tril(1)
    with pytest.raises(TypeError):
        dpnp.triu(1)

    with pytest.raises(ValueError):
        dpnp.tri(-1)
    with pytest.raises(ValueError):
        dpnp.tri(1, -1)
    with pytest.raises(ValueError):
        dpnp.tril(dpnp.array(5))
    with pytest.raises(ValueError):
        dpnp.triu(dpnp.array(5))
