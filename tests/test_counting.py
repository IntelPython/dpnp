import numpy
import pytest

import dpnp


@pytest.mark.parametrize(
    "type",
    [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
    ids=["float64", "float32", "int64", "int32"],
)
@pytest.mark.parametrize("size", [2, 4, 8, 16, 3, 9, 27, 81])
def test_count_nonzero(type, size):
    a = numpy.arange(size, dtype=type)
    for i in range(int(size / 2)):
        a[(i * (int(size / 3) - 1)) % size] = 0

    ia = dpnp.array(a)

    np_res = numpy.count_nonzero(a)
    dpnp_res = dpnp.count_nonzero(ia)

    numpy.testing.assert_allclose(dpnp_res, np_res)
