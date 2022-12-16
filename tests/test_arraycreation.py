import pytest

import dpnp
import dpctl.tensor as dpt

import numpy

import tempfile


@pytest.mark.parametrize("start",
                         [0, -5, 10, -2.5, 9.7],
                         ids=['0', '-5', '10', '-2.5', '9.7'])
@pytest.mark.parametrize("stop",
                         [None, 10, -2, 20.5, 1000],
                         ids=['None', '10', '-2', '20.5', '10**5'])
@pytest.mark.parametrize("step",
                         [None, 1, 2.7, -1.6, 100],
                         ids=['None', '1', '2.7', '-1.6', '100'])
@pytest.mark.parametrize("dtype",
                         [numpy.complex128, numpy.complex64, numpy.float64, numpy.float32, numpy.float16, numpy.int64, numpy.int32],
                         ids=['complex128', 'complex64', 'float64', 'float32', 'float16', 'int64', 'int32'])
def test_arange(start, stop, step, dtype):
    rtol_mult = 2
    if numpy.issubdtype(dtype, numpy.float16):
        # numpy casts to float32 type when computes float16 data
        rtol_mult = 4

    exp_array = numpy.arange(start, stop=stop, step=step, dtype=dtype)

    dpnp_array = dpnp.arange(start, stop=stop, step=step, dtype=dtype)
    res_array = dpnp.asnumpy(dpnp_array)

    if numpy.issubdtype(dtype, numpy.floating) or numpy.issubdtype(dtype, numpy.complexfloating):
        numpy.testing.assert_allclose(exp_array, res_array, rtol=rtol_mult*numpy.finfo(dtype).eps)
    else:
        numpy.testing.assert_array_equal(exp_array, res_array)


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
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("N",
                         [0, 1, 2, 3, 4],
                         ids=['0', '1', '2', '3', '4'])
@pytest.mark.parametrize("M",
                         [None, 0, 1, 2, 3, 4],
                         ids=['None', '0', '1', '2', '3', '4'])
@pytest.mark.parametrize("k",
                         [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                         ids=['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5'])
@pytest.mark.parametrize("dtype",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_eye(N, M, k, dtype):
    expected = numpy.eye(N, M=M, k=k, dtype=dtype)
    result = dpnp.eye(N, M=M, k=k, dtype=dtype)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_frombuffer(type):
    buffer = b'12345678'

    np_res = numpy.frombuffer(buffer, dtype=type)
    dpnp_res = dpnp.frombuffer(buffer, dtype=type)

    numpy.testing.assert_array_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_fromfile(type):
    with tempfile.TemporaryFile() as fh:
        fh.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08")
        fh.flush()

        fh.seek(0)
        np_res = numpy.fromfile(fh, dtype=type)
        fh.seek(0)
        dpnp_res = dpnp.fromfile(fh, dtype=type)

        numpy.testing.assert_array_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_fromfunction(type):
    def func(x, y):
        return x * y

    shape = (3, 3)

    np_res = numpy.fromfunction(func, shape=shape, dtype=type)
    dpnp_res = dpnp.fromfunction(func, shape=shape, dtype=type)

    numpy.testing.assert_array_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_fromiter(type):
    iter = [1, 2, 3, 4]

    np_res = numpy.fromiter(iter, dtype=type)
    dpnp_res = dpnp.fromiter(iter, dtype=type)

    numpy.testing.assert_array_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_fromstring(type):
    string = "1 2 3 4"

    np_res = numpy.fromstring(string, dtype=type, sep=' ')
    dpnp_res = dpnp.fromstring(string, dtype=type, sep=' ')

    numpy.testing.assert_array_equal(dpnp_res, np_res)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("num",
                         [2, 4, 8, 3, 9, 27])
@pytest.mark.parametrize("endpoint",
                         [True, False])
def test_geomspace(type, num, endpoint):
    start = 2
    stop = 256

    np_res = numpy.geomspace(start, stop, num, endpoint, type)
    dpnp_res = dpnp.geomspace(start, stop, num, endpoint, type)

    # Note that the above may not produce exact integers:
    # (c) https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html
    if type in [numpy.int64, numpy.int32]:
        numpy.testing.assert_allclose(dpnp_res, np_res, atol=1)
    else:
        numpy.testing.assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("n",
                         [0, 1, 4],
                         ids=['0', '1', '4'])
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64,
                          numpy.int32, numpy.bool, numpy.complex128, None],
                         ids=['float64', 'float32', 'int64', 'int32', 'bool', 'complex128', 'None'])
def test_identity(n, type):
    expected = numpy.identity(n, dtype=type)
    result = dpnp.identity(n, dtype=type)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_loadtxt(type):
    with tempfile.TemporaryFile() as fh:
        fh.write(b"1 2 3 4")
        fh.flush()

        fh.seek(0)
        np_res = numpy.loadtxt(fh, dtype=type)
        fh.seek(0)
        dpnp_res = dpnp.loadtxt(fh, dtype=type)

        numpy.testing.assert_array_equal(dpnp_res, np_res)


@pytest.mark.parametrize("dtype",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
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
    a = numpy.array(array, type)
    ia = dpnp.array(array, type)
    expected = numpy.trace(a, offset=offset, dtype=dtype)
    result = dpnp.trace(ia, offset=offset, dtype=dtype)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("N",
                         [0, 1, 2, 3, 4],
                         ids=['0', '1', '2', '3', '4'])
@pytest.mark.parametrize("M",
                         [0, 1, 2, 3, 4],
                         ids=['0', '1', '2', '3', '4'])
@pytest.mark.parametrize("k",
                         [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
                         ids=['-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5'])
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, float, numpy.int64, numpy.int32, numpy.int, numpy.float, int],
                         ids=['float64', 'float32', 'numpy.float', 'float', 'int64', 'int32', 'numpy.int', 'int'])
def test_tri(N, M, k, type):
    expected = numpy.tri(N, M, k, dtype=type)
    result = dpnp.tri(N, M, k, dtype=type)
    numpy.testing.assert_array_equal(result, expected)


def test_tri_default_dtype():
    expected = numpy.tri(3, 5, -1)
    result = dpnp.tri(3, 5, -1)
    numpy.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("k",
                         [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
                         ids=['-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6'])
@pytest.mark.parametrize("m",
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
def test_tril(m, k):
    a = numpy.array(m)
    ia = dpnp.array(a)
    expected = numpy.tril(a, k)
    result = dpnp.tril(ia, k)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("k",
                         [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                         ids=['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'])
@pytest.mark.parametrize("m",
                         [[0, 1, 2, 3, 4],
                          [[1, 2], [3, 4]],
                          [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                          [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]],
                         ids=['[0, 1, 2, 3, 4]',
                              '[[1, 2], [3, 4]]',
                              '[[0, 1, 2], [3, 4, 5], [6, 7, 8]]',
                              '[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]'])
def test_triu(m, k):
    a = numpy.array(m)
    ia = dpnp.array(a)
    expected = numpy.triu(a, k)
    result = dpnp.triu(ia, k)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("k",
                         [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                         ids=['-4', '-3', '-2', '-1', '0', '1', '2', '3', '4'])
def test_triu_size_null(k):
    a = numpy.ones(shape=(1, 2, 0))
    ia = dpnp.array(a)
    expected = numpy.triu(a, k)
    result = dpnp.triu(ia, k)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("array",
                         [[1, 2, 3, 4],
                          [],
                          [0, 3, 5]],
                         ids=['[1, 2, 3, 4]',
                              '[]',
                              '[0, 3, 5]'])
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64,
                          numpy.int32, numpy.bool, numpy.complex128],
                         ids=['float64', 'float32', 'int64', 'int32', 'bool', 'complex128'])
@pytest.mark.parametrize("n",
                         [0, 1, 4, None],
                         ids=['0', '1', '4', 'None'])
@pytest.mark.parametrize("increase",
                         [True, False],
                         ids=['True', 'False'])
def test_vander(array, type, n, increase):
    a_np = numpy.array(array, dtype=type)
    a_dpnp = dpnp.array(array, dtype=type)

    expected = numpy.vander(a_np, N=n, increasing=increase)
    result = dpnp.vander(a_dpnp, N=n, increasing=increase)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("shape",
                         [(), 0, (0,), (2, 0, 3), (3, 2)],
                         ids=['()', '0', '(0,)', '(2, 0, 3)', '(3, 2)'])
@pytest.mark.parametrize("fill_value",
                         [1.5, 2, 1.5+0.j],
                         ids=['1.5', '2', '1.5+0.j'])
@pytest.mark.parametrize("dtype",
                         [None, numpy.complex128, numpy.complex64, numpy.float64, numpy.float32, numpy.float16, numpy.int64, numpy.int32],
                         ids=['None', 'complex128', 'complex64', 'float64', 'float32', 'float16', 'int64', 'int32'])
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_full(shape, fill_value, dtype, order):
    expected = numpy.full(shape, fill_value, dtype=dtype, order=order)
    result = dpnp.full(shape, fill_value, dtype=dtype, order=order)

    assert expected.dtype == result.dtype
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("array",
                         [[], 0,  [1, 2, 3], [[1, 2], [3, 4]]],
                         ids=['[]', '0',  '[1, 2, 3]', '[[1, 2], [3, 4]]'])
@pytest.mark.parametrize("fill_value",
                         [1.5, 2, 1.5+0.j],
                         ids=['1.5', '2', '1.5+0.j'])
@pytest.mark.parametrize("dtype",
                         [None, numpy.complex128, numpy.complex64, numpy.float64, numpy.float32, numpy.float16, numpy.int64, numpy.int32],
                         ids=['None', 'complex128', 'complex64', 'float64', 'float32', 'float16', 'int64', 'int32'])
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_full_like(array, fill_value, dtype, order):
    a = numpy.array(array)
    ia = dpnp.array(array)

    expected = numpy.full_like(a, fill_value, dtype=dtype, order=order)
    result = dpnp.full_like(ia, fill_value, dtype=dtype, order=order)
    
    assert expected.dtype == result.dtype
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.skip(reason="dpnp.ndarray.flags are not implemented")
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
    numpy.testing.assert_array_equal(dpnp.asnumpy(ia), a)

    a = numpy.full((3, 3), numpy.arange(6, dtype="i4")[::2])
    ia = dpnp.full((3, 3), dpnp.arange(6, dtype="i4")[::2])
    assert ia.strides == tuple(el // a.itemsize for el in a.strides)
    numpy.testing.assert_array_equal(dpnp.asnumpy(ia), a)


@pytest.mark.parametrize("fill_value", [[], (), dpnp.full(0, 0)], ids=['[]', '()', 'dpnp.full(0, 0)'])
def test_full_invalid_fill_value(fill_value):
    with pytest.raises(ValueError):
        dpnp.full(10, fill_value=fill_value)


@pytest.mark.parametrize("shape",
                         [(), 0, (0,), (2, 0, 3), (3, 2)],
                         ids=['()', '0', '(0,)', '(2, 0, 3)', '(3, 2)'])
@pytest.mark.parametrize("dtype",
                         [None, numpy.complex128, numpy.complex64, numpy.float64, numpy.float32, numpy.float16, numpy.int64, numpy.int32],
                         ids=['None', 'complex128', 'complex64', 'float64', 'float32', 'float16', 'int64', 'int32'])
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_zeros(shape, dtype, order):
    expected = numpy.zeros(shape, dtype=dtype, order=order)
    result = dpnp.zeros(shape, dtype=dtype, order=order)

    assert expected.dtype == result.dtype
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("array",
                         [[], 0,  [1, 2, 3], [[1, 2], [3, 4]]],
                         ids=['[]', '0',  '[1, 2, 3]', '[[1, 2], [3, 4]]'])
@pytest.mark.parametrize("dtype",
                         [None, numpy.complex128, numpy.complex64, numpy.float64, numpy.float32, numpy.float16, numpy.int64, numpy.int32],
                         ids=['None', 'complex128', 'complex64', 'float64', 'float32', 'float16', 'int64', 'int32'])
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_zeros_like(array, dtype, order):
    a = numpy.array(array)
    ia = dpnp.array(array)

    expected = numpy.zeros_like(a, dtype=dtype, order=order)
    result = dpnp.zeros_like(ia, dtype=dtype, order=order)

    assert expected.dtype == result.dtype
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("shape",
                         [(), 0, (0,), (2, 0, 3), (3, 2)],
                         ids=['()', '0', '(0,)', '(2, 0, 3)', '(3, 2)'])
@pytest.mark.parametrize("dtype",
                         [None, numpy.complex128, numpy.complex64, numpy.float64, numpy.float32, numpy.float16, numpy.int64, numpy.int32],
                         ids=['None', 'complex128', 'complex64', 'float64', 'float32', 'float16', 'int64', 'int32'])
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_ones(shape, dtype, order):
    expected = numpy.ones(shape, dtype=dtype, order=order)
    result = dpnp.ones(shape, dtype=dtype, order=order)

    assert expected.dtype == result.dtype
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("array",
                         [[], 0,  [1, 2, 3], [[1, 2], [3, 4]]],
                         ids=['[]', '0',  '[1, 2, 3]', '[[1, 2], [3, 4]]'])
@pytest.mark.parametrize("dtype",
                         [None, numpy.complex128, numpy.complex64, numpy.float64, numpy.float32, numpy.float16, numpy.int64, numpy.int32],
                         ids=['None', 'complex128', 'complex64', 'float64', 'float32', 'float16', 'int64', 'int32'])
@pytest.mark.parametrize("order",
                         [None, "C", "F"],
                         ids=['None', 'C', 'F'])
def test_ones_like(array, dtype, order):
    a = numpy.array(array)
    ia = dpnp.array(array)

    expected = numpy.ones_like(a, dtype=dtype, order=order)
    result = dpnp.ones_like(ia, dtype=dtype, order=order)

    assert expected.dtype == result.dtype
    numpy.testing.assert_array_equal(expected, result)
