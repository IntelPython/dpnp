from tests.third_party.cupy import testing as cupy_testing
from .helper import has_support_aspect64
import dpnp
import numpy

from tests import testing


numpy.testing.assert_allclose = testing.assert_allclose
numpy.testing.assert_array_equal = testing.assert_array_equal
numpy.testing.assert_equal = testing.assert_equal

# patch for shaped_arange func to exclude calls of astype and reshape
# necessary because new data container does not support these functions yet

orig_shaped_arange = cupy_testing.shaped_arange
orig_shaped_reverse_arange = cupy_testing.shaped_reverse_arange


def _shaped_arange(shape, xp=dpnp, dtype=dpnp.float64, order="C"):
    if dtype is dpnp.float64:
        dtype = dpnp.float32 if not has_support_aspect64() else dtype
    res = xp.array(
        orig_shaped_arange(shape, xp=numpy, dtype=dtype, order=order),
        dtype=dtype,
    )
    return res


def _shaped_reverse_arange(shape, xp=dpnp, dtype=dpnp.float32):
    res = xp.array(
        orig_shaped_reverse_arange(shape, xp=numpy, dtype=dtype), dtype=dtype
    )
    return res


cupy_testing.shaped_arange = _shaped_arange
cupy_testing.shaped_reverse_arange = _shaped_reverse_arange
