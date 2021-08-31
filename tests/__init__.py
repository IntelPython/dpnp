import dpnp
import numpy

from tests import testing


numpy.testing.assert_allclose = testing.assert_allclose
numpy.testing.assert_array_equal = testing.assert_array_equal
