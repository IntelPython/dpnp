import pytest

import dpnp
import numpy


def _getattr(ex, str_):
    attrs = str_.split(".")
    res = ex
    for attr in attrs:
        res = getattr(res, attr)
    return res


@pytest.mark.parametrize("func_name",
                         ['abs', ])
@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
def test_strides(func_name, type):
    shape = (4, 4)
    a = numpy.arange(shape[0] * shape[1], dtype=type).reshape(shape)
    a_strides = a[0::2, 0::2]
    dpa = dpnp.array(a)
    dpa_strides = dpa[0::2, 0::2]

    dpnp_func = _getattr(dpnp, func_name)
    result = dpnp_func(dpa_strides)

    numpy_func = _getattr(numpy, func_name)
    expected = numpy_func(a_strides)

    numpy.testing.assert_allclose(expected, result)


@pytest.mark.parametrize("func_name",
                         ["add", "arctan2", "hypot", "maximum", "minimum", "multiply", "power", "subtract"])
@pytest.mark.parametrize("dtype",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=["float64", "float32", "int64", "int32"])
@pytest.mark.parametrize("shape",
                         [(3, 3)],
                         ids=["(3, 3)"])
def test_strides_2args(func_name, dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a.T

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa.T

    dpnp_func = _getattr(dpnp, func_name)
    result = dpnp_func(dpa, dpb)

    numpy_func = _getattr(numpy, func_name)
    expected = numpy_func(a, b)

    numpy.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("dtype",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=["float64", "float32", "int64", "int32"])
@pytest.mark.parametrize("shape",
                         [(3, 3)],
                         ids=["(3, 3)"])
def test_strides_copysign(dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = -a.T

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpnp.negative(dpa.T)

    result = dpnp.copysign(dpa, dpb)
    expected = numpy.copysign(a, b)

    numpy.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("dtype",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=["float64", "float32", "int64", "int32"])
@pytest.mark.parametrize("shape",
                         [(3, 3)],
                         ids=["(3, 3)"])
def test_strides_fmod(dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a.T + 1

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa.T + 1

    result = dpnp.fmod(dpa, dpb)
    expected = numpy.fmod(a, b)

    numpy.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("dtype",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=["float64", "float32", "int64", "int32"])
@pytest.mark.parametrize("shape",
                         [(3, 3)],
                         ids=["(3, 3)"])
def test_strides_true_devide(dtype, shape):
    a = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
    b = a.T + 1

    dpa = dpnp.reshape(dpnp.arange(numpy.prod(shape), dtype=dtype), shape)
    dpb = dpa.T + 1

    result = dpnp.fmod(dpa, dpb)
    expected = numpy.fmod(a, b)

    numpy.testing.assert_allclose(result, expected)
