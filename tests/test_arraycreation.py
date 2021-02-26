import pytest

import dpnp

import numpy

import tempfile


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_frombuffer(type):
    buffer = b'12345678'

    np_res = numpy.frombuffer(buffer, dtype=type)
    dpnp_res = dpnp.frombuffer(buffer, dtype=type)

    numpy.testing.assert_array_equal(dpnp_res, np_res)


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


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_fromiter(type):
    iter = [1, 2, 3, 4]

    np_res = numpy.fromiter(iter, dtype=type)
    dpnp_res = dpnp.fromiter(iter, dtype=type)

    numpy.testing.assert_array_equal(dpnp_res, np_res)


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_fromstring(type):
    string = "1 2 3 4"

    np_res = numpy.fromstring(string, dtype=type, sep=' ')
    dpnp_res = dpnp.fromstring(string, dtype=type, sep=' ')

    numpy.testing.assert_array_equal(dpnp_res, np_res)


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
