import pytest
import unittest

import dpnp.random
import numpy
from numpy.testing import assert_allclose
import math


class TestDistribution(unittest.TestCase):

    def check_extreme_value(self, dist_name, val, params):
        seed = 28041990
        size = 10
        dpnp.random.seed(seed)
        res = numpy.asarray(getattr(dpnp.random, dist_name)(size=size, **params))
        assert len(numpy.unique(res)) == 1
        assert numpy.unique(res)[0] == val

    def check_moments(self, dist_name, expected_mean, expected_var, params):
        size = 10**5
        seed = 28041995
        dpnp.random.seed(seed)
        res = numpy.asarray(getattr(dpnp.random, dist_name)(size=size, **params))
        var = numpy.var(res)
        mean = numpy.mean(res)
        assert math.isclose(var, expected_var, abs_tol=0.1)
        assert math.isclose(mean, expected_mean, abs_tol=0.1)

    def check_invalid_args(self, dist_name, params):
        size = 10
        with pytest.raises(ValueError):
            getattr(dpnp.random, dist_name)(size=size, **params)

    def check_seed(self, dist_name, params):
        seed = 28041990
        size = 10
        dpnp.random.seed(seed)
        a1 = numpy.asarray(getattr(dpnp.random, dist_name)(size=size, **params))
        dpnp.random.seed(seed)
        a2 = numpy.asarray(getattr(dpnp.random, dist_name)(size=size, **params))
        assert_allclose(a1, a2, rtol=1e-07, atol=0)


@pytest.mark.parametrize("func",
                         [dpnp.random.chisquare,
                          dpnp.random.rand,
                          dpnp.random.randn],
                         ids=['chisquare', 'rand', 'randn'])
def test_input_size(func):
    output_shape = (10,)
    size = 10
    df = 3  # for dpnp.random.chisquare
    if func == dpnp.random.chisquare:
        res = func(df, size)
    else:
        res = func(size)
    assert output_shape == res.shape


@pytest.mark.parametrize("func",
                         [dpnp.random.chisquare,
                          dpnp.random.random,
                          dpnp.random.random_sample,
                          dpnp.random.ranf,
                          dpnp.random.sample],
                         ids=['chisquare', 'random', 'random_sample',
                              'ranf', 'sample'])
def test_input_shape(func):
    shape = (10, 5)
    df = 3  # for dpnp.random.chisquare
    if func == dpnp.random.chisquare:
        res = func(df, shape)
    else:
        res = func(shape)
    assert shape == res.shape


@pytest.mark.parametrize("func",
                         [dpnp.random.random,
                          dpnp.random.random_sample,
                          dpnp.random.ranf,
                          dpnp.random.sample,
                          dpnp.random.rand],
                         ids=['random', 'random_sample',
                              'ranf', 'sample',
                              'rand'])
def test_check_otput(func):
    shape = (10, 5)
    size = 10 * 5
    if func == dpnp.random.rand:
        res = func(size)
    else:
        res = func(shape)
#    assert numpy.all(res >= 0)
#    assert numpy.all(res < 1)
    for i in range(res.size):
        assert res[i] >= 0.0
        assert res[i] < 1.0


@pytest.mark.parametrize("func",
                         [dpnp.random.random,
                          dpnp.random.random_sample,
                          dpnp.random.ranf,
                          dpnp.random.sample,
                          dpnp.random.rand],
                         ids=['random', 'random_sample',
                              'ranf', 'sample', 'rand'])
def test_seed(func):
    seed = 28041990
    size = 100
    shape = (100, 1)
    if func in [dpnp.random.rand, dpnp.random.randn]:
        args = size
    else:
        args = shape
    dpnp.random.seed(seed)
    a1 = func(args)
    dpnp.random.seed(seed)
    a2 = func(args)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_randn_normal_distribution():
    """
    Check the moments of the normal distribution sample obtained
    from ``dpnp.random.randn``.

    """

    seed = 28041995
    pts = 10**5
    alpha = 0.05
    expected_mean = 0.0
    expected_var = 1.0

    dpnp.random.seed(seed)
    res = dpnp.random.randn(pts)
    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.03)
    assert math.isclose(mean, expected_mean, abs_tol=0.03)


class TestDistributionsBeta(TestDistribution):

    def test_moments(self):
        a = 2.56
        b = 0.8
        expected_mean = a / (a + b)
        expected_var = (a * b) / ((a + b)**2 * (a + b + 1))
        self.check_moments('beta', expected_mean, expected_var, {'a': a, 'b': b})

    def test_invalid_args(self):
        a = 3.0   # OK
        b = -1.0  # positive `b` is expected
        self.check_invalid_args('beta', {'a': a, 'b': b})
        a = -1.0  # positive `a` is expected
        b = 3.0   # OK
        self.check_invalid_args('beta', {'a': a, 'b': b})

    def test_seed(self):
        a = 2.56
        b = 0.8
        self.check_seed('beta', {'a': a, 'b': b})


class TestDistributionsBinomial(TestDistribution):

    def test_extreme_value(self):
        n = 5
        p = 0.0
        expected_val = p
        self.check_extreme_value('binomial', expected_val, {'n': n, 'p': p})
        n = 0
        p = 0.5
        expected_val = n
        self.check_extreme_value('binomial', expected_val, {'n': n, 'p': p})
        n = 5
        p = 1.0
        expected_val = n
        self.check_extreme_value('binomial', expected_val, {'n': n, 'p': p})

    def test_moments(self):
        n = 5
        p = 0.8
        expected_mean = n * p
        expected_var = n * p * (1 - p)
        self.check_moments('binomial', expected_mean,
                           expected_var, {'n': n, 'p': p})

    def test_invalid_args(self):
        n = -5     # non-negative `n` is expected
        p = 0.4    # OK
        self.check_invalid_args('binomial', {'n': n, 'p': p})
        n = 5      # OK
        p = -0.5   # `p` is expected from [0, 1]
        self.check_invalid_args('binomial', {'n': n, 'p': p})

    def test_seed(self):
        n, p = 10, .5  # number of trials, probability of each trial
        self.check_seed('binomial', {'n': n, 'p': p})


class TestDistributionsChisquare(TestDistribution):

    def test_invalid_args(self):
        df = -1  # positive `df` is expected
        self.check_invalid_args('chisquare', {'df': df})

    def test_seed(self):
        df = 3  # number of degrees of freedom
        self.check_seed('chisquare', {'df': df})


class TestDistributionsExponential(TestDistribution):

    def test_invalid_args(self):
        scale = -1  # non-negative `scale` is expected
        self.check_invalid_args('exponential', {'scale': scale})

    def test_seed(self):
        scale = 3  # number of degrees of freedom
        self.check_seed('exponential', {'scale': scale})


class TestDistributionsGamma(TestDistribution):

    def test_moments(self):
        shape = 2.56
        scale = 0.8
        expected_mean = shape * scale
        expected_var = shape * scale * scale
        self.check_moments('gamma', expected_mean, expected_var,
                           {'shape': shape, 'scale': scale})

    def test_invalid_args(self):
        size = 10
        shape = -1   # non-negative `shape` is expected
        self.check_invalid_args('gamma', {'shape': shape})
        shape = 1.0   # OK
        scale = -1.0  # non-negative `shape` is expected
        self.check_invalid_args('gamma', {'shape': shape, 'scale': scale})

    def test_seed(self):
        shape = 3.0  # shape param for gamma distr
        self.check_seed('gamma', {'shape': shape})


class TestDistributionsGeometric(TestDistribution):

    def test_extreme_value(self):
        p = 1.0
        expected_val = p
        self.check_extreme_value('geometric', expected_val, {'p': p})

    def test_moments(self):
        p = 0.8
        expected_mean = (1 - p) / p
        expected_var = (1 - p) / (p**2)
        self.check_moments('geometric', expected_mean, expected_var, {'p': p})

    def test_invalid_args(self):
        size = 10
        p = -1.0  # `p` is expected from (0, 1]
        self.check_invalid_args('geometric', {'p': p})

    def test_seed(self):
        p = 0.8
        self.check_seed('geometric', {'p': p})


class TestDistributionsGumbel(TestDistribution):

    def test_extreme_value(self):
        loc = 5
        scale = 0.0
        expected_val = loc
        self.check_extreme_value('gumbel', expected_val,
                                 {'loc': loc, 'scale': scale})

    def test_moments(self):
        loc = 12
        scale = 0.8
        expected_mean = loc + scale * numpy.euler_gamma
        expected_var = (numpy.pi**2 / 6) * (scale ** 2)
        self.check_moments('gumbel', expected_mean,
                           expected_var, {'loc': loc, 'scale': scale})

    def test_invalid_args(self):
        size = 10
        loc = 3.0     # OK
        scale = -1.0  # non-negative `scale` is expected
        self.check_invalid_args('gumbel', {'loc': loc, 'scale': scale})

    def test_seed(self):
        loc = 2.56
        scale = 0.8
        self.check_seed('gumbel', {'loc': loc, 'scale': scale})


class TestDistributionsHypergeometric(TestDistribution):

    def test_extreme_value(self):
        ngood = 100
        nbad = 0
        nsample = 10
        expected_val = nsample
        self.check_extreme_value('hypergeometric', expected_val,
                                 {'ngood': ngood, 'nbad': nbad, 'nsample': nsample})
        ngood = 0
        nbad = 11
        nsample = 10
        expected_val = 0
        self.check_extreme_value('hypergeometric', expected_val,
                                 {'ngood': ngood, 'nbad': nbad, 'nsample': nsample})

    def test_moments(self):
        ngood = 100
        nbad = 2
        nsample = 10
        expected_mean = nsample * (ngood / (ngood + nbad))
        expected_var = expected_mean * (nbad / (ngood + nbad)) * (((ngood + nbad) - nsample) / ((ngood + nbad) - 1))
        self.check_moments('hypergeometric', expected_mean, expected_var,
                           {'ngood': ngood, 'nbad': nbad, 'nsample': nsample})

    def test_invalid_args(self):
        size = 10
        ngood = 100    # OK
        nbad = 2       # OK
        nsample = -10  # non-negative `nsamp` is expected
        self.check_invalid_args('hypergeometric',
                                {'ngood': ngood, 'nbad': nbad, 'nsample': nsample})

        ngood = 100    # OK
        nbad = -2      # non-negative `nbad` is expected
        nsample = 10   # OK
        self.check_invalid_args('hypergeometric',
                                {'ngood': ngood, 'nbad': nbad, 'nsample': nsample})

        ngood = -100   # non-negative `ngood` is expected
        nbad = 2       # OK
        nsample = 10   # OK
        self.check_invalid_args('hypergeometric',
                                {'ngood': ngood, 'nbad': nbad, 'nsample': nsample})

        ngood = 10
        nbad = 2
        nsample = 100
        # ngood + nbad >= nsample expected
        self.check_invalid_args('hypergeometric',
                                {'ngood': ngood, 'nbad': nbad, 'nsample': nsample})

        ngood = 10   # OK
        nbad = 2     # OK
        nsample = 0  # `nsample` is expected > 0
        self.check_invalid_args('hypergeometric',
                                {'ngood': ngood, 'nbad': nbad, 'nsample': nsample})

    def test_seed(self):
        ngood = 100
        nbad = 2
        nsample = 10
        self.check_seed('hypergeometric',
                        {'ngood': ngood, 'nbad': nbad, 'nsample': nsample})


class TestDistributionsLaplace(TestDistribution):

    def test_extreme_value(self):
        loc = 5
        scale = 0.0
        expected_val = scale
        self.check_extreme_value('laplace', expected_val,
                                 {'loc': loc, 'scale': scale})

    def test_moments(self):
        loc = 2.56
        scale = 0.8
        expected_mean = loc
        expected_var = 2 * scale * scale
        self.check_moments('laplace', expected_mean,
                           expected_var, {'loc': loc, 'scale': scale})

    def test_invalid_args(self):
        loc = 3.0     # OK
        scale = -1.0  # positive `b` is expected
        self.check_invalid_args('laplace',
                                {'loc': loc, 'scale': scale})

    def test_seed(self):
        loc = 2.56
        scale = 0.8
        self.check_seed('laplace', {'loc': loc, 'scale': scale})


class TestDistributionsLognormal(TestDistribution):

    def test_extreme_value(self):
        mean = 0.5
        sigma = 0.0
        expected_val = numpy.exp(mean + (sigma ** 2) / 2)
        self.check_extreme_value('lognormal', expected_val,
                                 {'mean': mean, 'sigma': sigma})

    def test_moments(self):
        mean = 0.5
        sigma = 0.8
        expected_mean = numpy.exp(mean + (sigma ** 2) / 2)
        expected_var = (numpy.exp(sigma**2) - 1) * numpy.exp(2 * mean + sigma**2)
        self.check_moments('lognormal', expected_mean,
                           expected_var, {'mean': mean, 'sigma': sigma})

    def test_invalid_args(self):
        mean = 0.0
        sigma = -1.0  # non-negative `sigma` is expected
        self.check_invalid_args('lognormal', {'mean': mean, 'sigma': sigma})

    def test_seed(self):
        mean = 0.0
        sigma = 0.8
        self.check_seed('lognormal', {'mean': mean, 'sigma': sigma})


class TestDistributionsMultinomial(TestDistribution):

    def test_extreme_value(self):
        n = 0
        pvals = [1 / 6.] * 6
        self.check_extreme_value('multinomial', n, {'n': n, 'pvals': pvals})

    def test_moments(self):
        n = 10
        pvals = [1 / 6.] * 6
        size = 10**5
        expected_mean = n * pvals[0]
        expected_var = n * pvals[0] * (1 - pvals[0])
        self.check_moments('multinomial', expected_mean,
                           expected_var, {'n': n, 'pvals': pvals})

    def test_check_sum(self):
        seed = 28041990
        size = 1
        n = 20
        pvals = [1 / 6.] * 6
        dpnp.random.seed(seed)
        res = dpnp.random.multinomial(n, pvals, size)
        assert_allclose(n, sum(res), rtol=1e-07, atol=0)

    def test_invalid_args(self):
        n = -10                # parameter `n`, non-negative expected
        pvals = [1 / 6.] * 6   # parameter `pvals`, OK
        self.check_invalid_args('multinomial', {'n': n, 'pvals': pvals})
        n = 10                 # parameter `n`, OK
        pvals = [-1 / 6.] * 6  # parameter `pvals`, sum(pvals) expected between [0, 1]
        self.check_invalid_args('multinomial', {'n': n, 'pvals': pvals})

    def test_seed(self):
        n = 20
        pvals = [1 / 6.] * 6
        self.check_seed('multinomial', {'n': n, 'pvals': pvals})


class TestDistributionsMultivariateNormal(TestDistribution):

    def test_moments(self):
        seed = 2804183
        dpnp.random.seed(seed)
        mean = [2.56, 3.23]
        cov = [[1, 0], [0, 1]]
        size = 10**5
        res = numpy.array(dpnp.random.multivariate_normal(mean=mean, cov=cov, size=size))
        res_mean = [numpy.mean(res.T[0]), numpy.mean(res.T[1])]
        assert_allclose(res_mean, mean, rtol=1e-02, atol=0)

    def test_invalid_args(self):
        mean = [2.56, 3.23]  # OK
        cov = [[1, 0]]       # `mean` and `cov` must have same length
        self.check_invalid_args('multivariate_normal', {'mean': mean, 'cov': cov})
        mean = [[2.56, 3.23]]   # `mean` must be 1 dimensional
        cov = [[1, 0], [0, 1]]  # OK
        self.check_invalid_args('multivariate_normal', {'mean': mean, 'cov': cov})
        mean = [2.56, 3.23]  # OK
        cov = [1, 0, 0, 1]   # `cov` must be 2 dimensional and square
        self.check_invalid_args('multivariate_normal', {'mean': mean, 'cov': cov})

    def test_output_shape_check(self):
        seed = 28041990
        size = 100
        mean = [2.56, 3.23]
        cov = [[1, 0], [0, 1]]
        expected_shape = (100, 2)
        dpnp.random.seed(seed)
        res = dpnp.random.multivariate_normal(mean, cov, size=100)
        assert res.shape == expected_shape

    def test_seed(self):
        mean = [2.56, 3.23]
        cov = [[1, 0], [0, 1]]
        self.check_seed('multivariate_normal', {'mean': mean, 'cov': cov})


class TestDistributionsNegativeBinomial(TestDistribution):

    def test_extreme_value(self):
        seed = 28041990
        dpnp.random.seed(seed)
        n = 5
        p = 1.0
        check_val = 0.0
        self.check_extreme_value('negative_binomial', check_val, {'n': n, 'p': p})
        n = 5
        p = 0.0
        res = numpy.asarray(dpnp.random.negative_binomial(n=n, p=p, size=10))
        check_val = numpy.iinfo(res.dtype).min
        assert len(numpy.unique(res)) == 1
        assert numpy.unique(res)[0] == check_val

    def test_invalid_args(self):
        n = 10    # parameter `n`, OK
        p = -0.5  # parameter `p`, expected between [0, 1]
        self.check_invalid_args('negative_binomial', {'n': n, 'p': p})
        n = -10   # parameter `n`, expected non-negative
        p = 0.5   # parameter `p`, OK
        self.check_invalid_args('negative_binomial', {'n': n, 'p': p})

    def test_seed(self):
        n, p = 10, .5  # number of trials, probability of each trial
        self.check_seed('negative_binomial', {'n': n, 'p': p})


class TestDistributionsNormal(TestDistribution):

    def test_extreme_value(self):
        loc = 5
        scale = 0.0
        expected_val = loc
        self.check_extreme_value('normal', expected_val, {'loc': loc, 'scale': scale})

    def test_moments(self):
        loc = 2.56
        scale = 0.8
        expected_mean = loc
        expected_var = scale**2
        self.check_moments('normal', expected_mean,
                           expected_var, {'loc': loc, 'scale': scale})

    def test_invalid_args(self):
        loc = 3.0     # OK
        scale = -1.0  # non-negative `scale` is expected
        self.check_invalid_args('normal', {'loc': loc, 'scale': scale})

    def test_seed(self):
        loc = 2.56
        scale = 0.8
        self.check_seed('normal', {'loc': loc, 'scale': scale})


class TestDistributionsPoisson(TestDistribution):

    def test_extreme_value(self):
        lam = 0.0
        self.check_extreme_value('poisson', lam, {'lam': lam})

    def test_moments(self):
        lam = 0.8
        expected_mean = lam
        expected_var = lam
        self.check_moments('poisson', expected_mean,
                           expected_var, {'lam': lam})

    def test_invalid_args(self):
        lam = -1.0    # non-negative `lam` is expected
        self.check_invalid_args('poisson', {'lam': lam})

    def test_seed(self):
        lam = 0.8
        self.check_seed('poisson', {'lam': lam})


class TestDistributionsRayleigh(TestDistribution):

    def test_extreme_value(self):
        scale = 0.0
        self.check_extreme_value('rayleigh', scale, {'scale': scale})

    def test_moments(self):
        scale = 0.8
        expected_mean = scale * numpy.sqrt(numpy.pi / 2)
        expected_var = ((4 - numpy.pi) / 2) * scale * scale
        self.check_moments('rayleigh', expected_mean,
                           expected_var, {'scale': scale})

    def test_invalid_args(self):
        scale = -1.0  # positive `b` is expected
        self.check_invalid_args('rayleigh', {'scale': scale})

    def test_seed(self):
        scale = 0.8
        self.check_seed('rayleigh', {'scale': scale})


class TestDistributionsStandardCauchy(TestDistribution):

    def test_seed(self):
        self.check_seed('standard_cauchy', {})


class TestDistributionsStandardExponential(TestDistribution):

    def test_moments(self):
        shape = 0.8
        expected_mean = 1.0
        expected_var = 1.0
        self.check_moments('standard_exponential',
                           expected_mean, expected_var, {})

    def test_seed(self):
        self.check_seed('standard_exponential', {})


class TestDistributionsStandardGamma(TestDistribution):

    def test_extreme_value(self):
        self.check_extreme_value('standard_gamma', 0.0, {'shape': 0.0})

    def test_moments(self):
        shape = 0.8
        expected_mean = shape
        expected_var = shape
        self.check_moments('standard_gamma', expected_mean,
                           expected_var, {'shape': shape})

    def test_invalid_args(self):
        shape = -1   # non-negative `shape` is expected
        self.check_invalid_args('standard_gamma', {'shape': shape})

    def test_seed(self):
        self.check_seed('standard_gamma', {'shape': 0.0})


class TestDistributionsStandardNormal(TestDistribution):

    def test_moments(self):
        expected_mean = 0.0
        expected_var = 1.0
        self.check_moments('standard_normal',
                           expected_mean, expected_var, {})

    def test_seed(self):
        self.check_seed('standard_normal', {})


class TestDistributionsUniform(TestDistribution):

    def test_extreme_value(self):
        low = 1.0
        high = 1.0
        expected_val = low
        self.check_extreme_value('uniform', expected_val,
                                 {'low': low, 'high': high})

    def test_moments(self):
        low = 1.0
        high = 2.0
        expected_mean = (low + high) / 2
        expected_var = ((high - low) ** 2) / 12
        self.check_moments('uniform', expected_mean,
                           expected_var, {'low': low, 'high': high})

    def test_seed(self):
        low = 1.0
        high = 2.0
        self.check_seed('uniform', {'low': low, 'high': high})


class TestDistributionsWeibull(TestDistribution):

    def test_extreme_value(self):
        a = 0.0
        expected_val = a
        self.check_extreme_value('weibull', expected_val, {'a': a})

    def test_invalid_args(self):
        a = -1.0  # non-negative `a` is expected
        self.check_invalid_args('weibull', {'a': a})

    def test_seed(self):
        a = 2.56
        self.check_seed('weibull', {'a': a})
