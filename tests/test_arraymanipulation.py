import pytest

import dpnp

import numpy


@pytest.mark.parametrize("type",
                         [numpy.float64, numpy.float32, numpy.int64, numpy.int32],
                         ids=['float64', 'float32', 'int64', 'int32'])
@pytest.mark.parametrize("input",
                         [[1,2,3], [1.,2.,3.], dpnp.array([1,2,3]), dpnp.array([1.,2.,3.])],
                         ids=['intlist', 'floatlist', 'intarray', 'floatarray'])
def test_asfarray(type, input):
    np_res = numpy.asfarray(input, type)
    dpnp_res = dpnp.asfarray(input, type)

    numpy.testing.assert_array_equal(dpnp_res, np_res)
