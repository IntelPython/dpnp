import pytest

import dpnp.random
import numpy
from numpy.testing import assert_allclose
import math


class TestDistributionsTestCases:

    def __init__(self):
        pass

    def test_check_extreme_value(self, dist_name, val, params):
        seed = 28041990
        size = 10
        dpnp.random.seed(seed)
        res = numpy.asarray(getattr(dpnp.random, dist_name)(size=size, **params))
        assert len(numpy.unique(res)) == 1
        assert numpy.unique(res)[0] == val

    def test_check_moments(self, dist_name, expected_mean, expected_var, params):
        size = 10**5
        seed = 28041995
        dpnp.random.seed(seed)
        res = numpy.asarray(getattr(dpnp.random, dist_name)(size=size, **params))
        var = numpy.var(res)
        mean = numpy.mean(res)
        assert math.isclose(var, expected_var, abs_tol=0.1)
        assert math.isclose(mean, expected_mean, abs_tol=0.1)

    def test_invalid_args(self, dist_name, params):
        size = 10
        with pytest.raises(ValueError):
            getattr(dpnp.random, dist_name)(size=size, **params)

    def test_seed(self, dist_name, params):
        seed = 28041990
        size = 10
        dpnp.random.seed(seed)
        a1 = numpy.asarray(getattr(dpnp.random, dist_name)(size=size, **params))
        dpnp.random.seed(seed)
        a2 = numpy.asarray(getattr(dpnp.random, dist_name)(size=size, **params))
        assert_allclose(a1, a2, rtol=1e-07, atol=0)
