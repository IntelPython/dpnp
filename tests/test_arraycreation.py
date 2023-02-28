import pytest
from .helper import get_all_dtypes

import dpnp

import dpctl
import dpctl.tensor as dpt

import numpy
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_raises
)

import tempfile
import operator


@pytest.mark.parametrize("start",
                         [0, -5, 10, -2.5, 9.7],
                         ids=['0', '-5', '10', '-2.5', '9.7'])
@pytest.mark.parametrize("stop",
                         [None, 10, -2, 20.5, 1000],
                         ids=['None', '10', '-2', '20.5', '10**5'])
@pytest.mark.parametrize("step",
                         [None, 1, 2.7, -1.6, 100],
                         ids=['None', '1', '2.7', '-1.6', '100'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_float16=False))
def test_arange(start, stop, step, dtype):
    rtol_mult = 2
    if numpy.issubdtype(dtype, numpy.float16):
        # numpy casts to float32 type when computes float16 data
        rtol_mult = 4

    func = lambda xp: xp.arange(start, stop=stop, step=step, dtype=dtype)

    exp_array = func(numpy)
    res_array = func(dpnp).asnumpy()

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

    if numpy.issubdtype(_dtype, numpy.floating) or numpy.issubdtype(_dtype, numpy.complexfloating):
        assert_allclose(exp_array, res_array, rtol=rtol_mult*numpy.finfo(_dtype).eps)
    else:
        assert_array_equal(exp_array, res_array)


@pytest.mark.parametrize("k",
                         [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
                         ids=['-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6'])
@pytest.mark.parametrize("v",
                         [[0, 1, 2, 3, 4],
                          [1, 1, 1, 1, 1],
                          [[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],
                          [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]],
                         ids=['[0, 1, 2, 3, 4]',
                              '[1, 1, 1, 1, 1]',
                              '[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]',
                              '[[0, 1, 2], [3, 4, 5], [6, 7, 8]]',
                              '[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]'])
def test_diag(v, k):
    a = numpy.array(v)
    ia = dpnp.array(a)
    expected = numpy.diag(a, k)
    result = dpnp.diag(ia, k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("N",
                         [0, 1, 2, 3],
                         ids=['0', '1', '2', '3'])
@pytest.mark.parametrize("M",
                         [None, 0, 1, 2, 3],
                         ids=['None', '0', '1', '2', '3'])
@pytest.mark.parametrize("k",
                         [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                         ids=['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_eye(N, M, k, dtype, order):
    func = lambda xp: xp.eye(N, M, k=k, dtype=dtype, order=order)
    assert_array_equal(func(numpy), func(dpnp))


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False, no_none=True))
def test_frombuffer(dtype):
    buffer = b'12345678ABCDEF00'
    func = lambda xp: xp.frombuffer(buffer, dtype=dtype)
    assert_allclose(func(dpnp), func(numpy))


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_fromfile(dtype):
    with tempfile.TemporaryFile() as fh:
        fh.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08")
        fh.flush()

        func = lambda xp: xp.fromfile(fh, dtype=dtype)

        fh.seek(0)
        np_res = func(numpy)

        fh.seek(0)
        dpnp_res = func(dpnp)

        assert_almost_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_float16=False))
def test_fromfunction(dtype):
    def func(x, y):
        return x * y

    shape = (3, 3)
    call_func = lambda xp: xp.fromfunction(func, shape=shape, dtype=dtype)
    assert_array_equal(call_func(dpnp), call_func(numpy))


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_fromiter(dtype):
    _iter = [1, 2, 3, 4]
    func = lambda xp: xp.fromiter(_iter, dtype=dtype)
    assert_array_equal(func(dpnp), func(numpy))


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_fromstring(dtype):
    string = "1 2 3 4"
    func = lambda xp: xp.fromstring(string, dtype=dtype, sep=' ')
    assert_array_equal(func(dpnp), func(numpy))


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize("num",
                         [2, 4, 8, 3, 9, 27])
@pytest.mark.parametrize("endpoint",
                         [True, False])
def test_geomspace(dtype, num, endpoint):
    start = 2
    stop = 256

    func = lambda xp: xp.geomspace(start, stop, num, endpoint, dtype)

    np_res = func(numpy)
    dpnp_res = func(dpnp)

    # Note that the above may not produce exact integers:
    # (c) https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html
    if dtype in [numpy.int64, numpy.int32]:
        assert_allclose(dpnp_res, np_res, atol=1)
    else:
        assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("n",
                         [0, 1, 4],
                         ids=['0', '1', '4'])
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_identity(n, dtype):
    func = lambda xp: xp.identity(n, dtype=dtype)
    assert_array_equal(func(numpy), func(dpnp))


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
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


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("type", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize("offset",
                         [0, 1],
                         ids=['0', '1'])
@pytest.mark.parametrize("array",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],
                          [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
                          [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]],
                          [[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]],
                          [[[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]], [
                              [[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]]],
                          [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]]],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]',
                              '[[0, 1, 2], [3, 4, 5], [6, 7, 8]]',
                              '[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]',
                              '[[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]',
                              '[[[[1, 2], [3, 4]], [[1, 2], [2, 1]]], [[[1, 3], [3, 1]], [[0, 1], [1, 3]]]]',
                              '[[[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [2, 1, 0]]], [[[1, 3, 5], [3, 1, 0]], [[0, 1, 2], [1, 3, 4]]]]',
                              '[[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]]'])
def test_trace(array, offset, type, dtype):
    create_array = lambda xp: xp.array(array, type)
    trace_func = lambda xp, x: xp.trace(x, offset=offset, dtype=dtype)

    a = create_array(numpy)
    ia = create_array(dpnp)
    assert_array_equal(trace_func(dpnp, ia), trace_func(numpy, a))


@pytest.mark.parametrize("N",
                         [0, 1, 2, 3, 4],
                         ids=['0', '1', '2', '3', '4'])
@pytest.mark.parametrize("M",
                         [0, 1, 2, 3, 4],
                         ids=['0', '1', '2', '3', '4'])
@pytest.mark.parametrize("k",
                         [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                         ids=['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
def test_tri(N, M, k, dtype):
    func = lambda xp: xp.tri(N, M, k, dtype=dtype)
    assert_array_equal(func(dpnp), func(numpy))


def test_tri_default_dtype():
    expected = numpy.tri(3, 5, -1)
    result = dpnp.tri(3, 5, -1)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("k",
                         [-3, -2, -1, 0, 1, 2, 3, 4, 5,
                          numpy.array(1), dpnp.array(2), dpt.asarray(3)],
                         ids=['-3', '-2', '-1', '0', '1', '2', '3', '4', '5',
                              'np.array(1)', 'dpnp.array(2)', 'dpt.asarray(3)'])
@pytest.mark.parametrize("m",
                         [[[0, 0], [0, 0]],
                          [[1, 2], [1, 2]],
                          [[1, 2], [3, 4]],
                          [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]],
                         ids=['[[0, 0], [0, 0]]',
                              '[[1, 2], [1, 2]]',
                              '[[1, 2], [3, 4]]',
                              '[[0, 1, 2], [3, 4, 5], [6, 7, 8]]',
                              '[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]'])
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_tril(m, k, dtype):
    a = numpy.array(m, dtype=dtype)
    ia = dpnp.array(a)
    expected = numpy.tril(a, k=k)
    result = dpnp.tril(ia, k=k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("k",
                         [-3, -2, -1, 0, 1, 2, 3, 4, 5,
                          numpy.array(1), dpnp.array(2), dpt.asarray(3)],
                         ids=['-3', '-2', '-1', '0', '1', '2', '3', '4', '5',
                              'np.array(1)', 'dpnp.array(2)', 'dpt.asarray(3)'])
@pytest.mark.parametrize("m",
                         [[[1, 2], [3, 4]],
                          [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]],
                         ids=['[[1, 2], [3, 4]]',
                              '[[0, 1, 2], [3, 4, 5], [6, 7, 8]]',
                              '[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]'])
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
def test_triu(m, k, dtype):
    a = numpy.array(m, dtype=dtype)
    ia = dpnp.array(a)
    expected = numpy.triu(a, k=k)
    result = dpnp.triu(ia, k=k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("k",
                         [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                         ids=['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'])
def test_triu_size_null(k):
    a = numpy.ones(shape=(1, 2, 0))
    ia = dpnp.array(a)
    expected = numpy.triu(a, k=k)
    result = dpnp.triu(ia, k=k)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("array",
                         [[1, 2, 3, 4],
                          [],
                          [0, 3, 5]],
                         ids=['[1, 2, 3, 4]',
                              '[]',
                              '[0, 3, 5]'])
@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize("n",
                         [0, 1, 4, None],
                         ids=['0', '1', '4', 'None'])
@pytest.mark.parametrize("increase",
                         [True, False],
                         ids=['True', 'False'])
def test_vander(array, dtype, n, increase):
    create_array = lambda xp: xp.array(array, dtype=dtype)
    vander_func = lambda xp, x: xp.vander(x, N=n, increasing=increase)

    a_np = numpy.array(array, dtype=dtype)
    a_dpnp = dpnp.array(array, dtype=dtype)
    assert_array_equal(vander_func(numpy, a_np), vander_func(dpnp, a_dpnp))


@pytest.mark.parametrize("shape",
                         [(), 0, (0,), (2, 0, 3), (3, 2)],
                         ids=['()', '0', '(0,)', '(2, 0, 3)', '(3, 2)'])
@pytest.mark.parametrize("fill_value",
                         [1.5, 2, 1.5+0.j],
                         ids=['1.5', '2', '1.5+0.j'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_full(shape, fill_value, dtype, order):
    func = lambda xp: xp.full(shape, fill_value, dtype=dtype, order=order)
    assert_array_equal(func(numpy), func(dpnp))


@pytest.mark.parametrize("array",
                         [[], 0,  [1, 2, 3], [[1, 2], [3, 4]]],
                         ids=['[]', '0',  '[1, 2, 3]', '[[1, 2], [3, 4]]'])
@pytest.mark.parametrize("fill_value",
                         [1.5, 2, 1.5+0.j],
                         ids=['1.5', '2', '1.5+0.j'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_full_like(array, fill_value, dtype, order):
    func = lambda xp, x: xp.full_like(x, fill_value, dtype=dtype, order=order)

    a = numpy.array(array)
    ia = dpnp.array(array)
    assert_array_equal(func(numpy, a), func(dpnp, ia))


@pytest.mark.parametrize("order1",
                         ["F", "C"],
                         ids=['F', 'C'])
@pytest.mark.parametrize("order2",
                         ["F", "C"],
                         ids=['F', 'C'])
def test_full_order(order1, order2):
    array = numpy.array([1, 2, 3], order=order1)
    a = numpy.full((3, 3), array, order=order2)
    ia = dpnp.full((3, 3), array, order=order2)

    assert ia.flags.c_contiguous == a.flags.c_contiguous
    assert ia.flags.f_contiguous == a.flags.f_contiguous
    assert numpy.array_equal(dpnp.asnumpy(ia), a)


def test_full_strides():
    a = numpy.full((3, 3), numpy.arange(3, dtype="i4"))
    ia = dpnp.full((3, 3), dpnp.arange(3, dtype="i4"))
    assert ia.strides == tuple(el // a.itemsize for el in a.strides)
    assert_array_equal(dpnp.asnumpy(ia), a)

    a = numpy.full((3, 3), numpy.arange(6, dtype="i4")[::2])
    ia = dpnp.full((3, 3), dpnp.arange(6, dtype="i4")[::2])
    assert ia.strides == tuple(el // a.itemsize for el in a.strides)
    assert_array_equal(dpnp.asnumpy(ia), a)


@pytest.mark.parametrize("fill_value",
                         [[], (), dpnp.full(0, 0)],
                         ids=['[]', '()', 'dpnp.full(0, 0)'])
def test_full_invalid_fill_value(fill_value):
    with pytest.raises(ValueError):
        dpnp.full(10, fill_value=fill_value)


@pytest.mark.parametrize("shape",
                         [(), 0, (0,), (2, 0, 3), (3, 2)],
                         ids=['()', '0', '(0,)', '(2, 0, 3)', '(3, 2)'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_zeros(shape, dtype, order):
    func = lambda xp: xp.zeros(shape, dtype=dtype, order=order)
    assert_array_equal(func(numpy), func(dpnp))


@pytest.mark.parametrize("array",
                         [[], 0,  [1, 2, 3], [[1, 2], [3, 4]]],
                         ids=['[]', '0',  '[1, 2, 3]', '[[1, 2], [3, 4]]'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_zeros_like(array, dtype, order):
    func = lambda xp, x: xp.zeros_like(x, dtype=dtype, order=order)

    a = numpy.array(array)
    ia = dpnp.array(array)
    assert_array_equal(func(numpy, a), func(dpnp, ia))


@pytest.mark.parametrize("shape",
                         [(), 0, (0,), (2, 0, 3), (3, 2)],
                         ids=['()', '0', '(0,)', '(2, 0, 3)', '(3, 2)'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_empty(shape, dtype, order):
    func = lambda xp: xp.empty(shape, dtype=dtype, order=order)
    assert func(numpy).shape == func(dpnp).shape


@pytest.mark.parametrize("array",
                         [[], 0,  [1, 2, 3], [[1, 2], [3, 4]]],
                         ids=['[]', '0',  '[1, 2, 3]', '[[1, 2], [3, 4]]'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_empty_like(array, dtype, order):
    func = lambda xp, x: xp.empty_like(x, dtype=dtype, order=order)

    a = numpy.array(array)
    ia = dpnp.array(array)
    assert func(numpy, a).shape == func(dpnp, ia).shape


@pytest.mark.parametrize("shape",
                         [(), 0, (0,), (2, 0, 3), (3, 2)],
                         ids=['()', '0', '(0,)', '(2, 0, 3)', '(3, 2)'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_ones(shape, dtype, order):
    func = lambda xp: xp.ones(shape, dtype=dtype, order=order)
    assert_array_equal(func(numpy), func(dpnp))


@pytest.mark.parametrize("array",
                         [[], 0,  [1, 2, 3], [[1, 2], [3, 4]]],
                         ids=['[]', '0',  '[1, 2, 3]', '[[1, 2], [3, 4]]'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_ones_like(array, dtype, order):
    func = lambda xp, x: xp.ones_like(x, dtype=dtype, order=order)

    a = numpy.array(array)
    ia = dpnp.array(array)
    assert_array_equal(func(numpy, a), func(dpnp, ia))


@pytest.mark.parametrize(
    "func, args",
    [
        pytest.param("full_like",
                     ['x0', '4']),
        pytest.param("zeros_like",
                     ['x0']),
        pytest.param("ones_like",
                     ['x0']),
        pytest.param("empty_like",
                     ['x0']),
    ])
def test_dpctl_tensor_input(func, args):
    x0 = dpt.reshape(dpt.arange(9), (3,3))
    new_args = [eval(val, {'x0' : x0}) for val in args]
    X = getattr(dpt, func)(*new_args)
    Y = getattr(dpnp, func)(*new_args)
    if func == 'empty_like':
        assert X.shape == Y.shape
    else:
        assert_array_equal(X, Y)


@pytest.mark.parametrize("arrays",
                         [[], [[1]], [[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4], [5, 6]]],
                         ids=['[]', '[[1]]', '[[1, 2, 3], [4, 5, 6]]', '[[1, 2], [3, 4], [5, 6]]'])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False))
@pytest.mark.parametrize("indexing",
                         ["ij", "xy"],
                         ids=["ij", "xy"])
def test_meshgrid(arrays, dtype, indexing):
    func = lambda xp, xi: xp.meshgrid(*xi, indexing=indexing)
    a = tuple(numpy.array(array, dtype=dtype) for array in arrays)
    ia = tuple(dpnp.array(array, dtype=dtype) for array in arrays)
    assert_array_equal(func(numpy, a), func(dpnp, ia))
