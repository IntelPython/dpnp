import pytest

import dpnp

import numpy


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
