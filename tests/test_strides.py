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
