import math
import dpnp
import numpy
from .helper import skip_or_check_if_dtype_not_supported


dtype = numpy.float64 if skip_or_check_if_dtype_not_supported(
    numpy.float64, check_dtype=True
    ) else numpy.float32


def test_erf():
    a = numpy.linspace(2.0, 3.0, num=10, dtype=dtype)
    ia = dpnp.linspace(2.0, 3.0, num=10, dtype=dtype)

    numpy.testing.assert_array_equal(a, ia)

    expected = numpy.empty_like(a)
    for idx, val in enumerate(a):
        expected[idx] = math.erf(val)

    result = dpnp.erf(ia)

    numpy.testing.assert_array_equal(result, expected)


def test_erf_fallback():
    a = numpy.linspace(2.0, 3.0, num=10, dtype=dtype)
    dpa = dpnp.linspace(2.0, 3.0, num=10, dtype=dtype)

    expected = numpy.empty_like(a)
    for idx, val in enumerate(a):
        expected[idx] = math.erf(val)

    result = dpnp.erf(dpa)

    numpy.testing.assert_array_equal(result, expected)
