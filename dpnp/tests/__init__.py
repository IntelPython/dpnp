import numpy

from . import testing

numpy.testing.assert_allclose = testing.assert_allclose
numpy.testing.assert_almost_equal = testing.assert_almost_equal
numpy.testing.assert_array_almost_equal = testing.assert_array_almost_equal
numpy.testing.assert_array_equal = testing.assert_array_equal
numpy.testing.assert_equal = testing.assert_equal
