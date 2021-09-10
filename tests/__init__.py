import dpnp
import numpy

from tests import testing


numpy.testing.assert_allclose = testing.assert_allclose
numpy.testing.assert_array_equal = testing.assert_array_equal
numpy.testing.assert_equal = testing.assert_equal

# patch for shaped_arange func to exclude calls of astype and reshape
# necessary because new data container does not support these functions yet
from tests.third_party.cupy import testing as cupy_testing
orig_shaped_arange = cupy_testing.shaped_arange
def _shaped_arange(shape, xp=dpnp, dtype=dpnp.float64, order='C'):
    res = xp.array(orig_shaped_arange(shape, xp=numpy, dtype=dtype, order=order))
    return res
cupy_testing.shaped_arange = _shaped_arange
