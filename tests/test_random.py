import pytest

import dpnp.random
import numpy
# from scipy import stats
from numpy.testing import assert_allclose
import math


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


def test_beta_seed():
    seed = 28041990
    size = 100
    a = 2.56
    b = 0.8

    dpnp.random.seed(seed)
    a1 = dpnp.random.beta(a, b, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.beta(a, b, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_beta_invalid_args():
    size = 10
    a = 3.0   # OK
    b = -1.0  # positive `b` is expected
    with pytest.raises(ValueError):
        dpnp.random.beta(a=a, b=b, size=size)
    a = -1.0  # positive `a` is expected
    b = 3.0   # OK
    with pytest.raises(ValueError):
        dpnp.random.beta(a=a, b=b, size=size)


def test_beta_check_moments():
    seed = 28041990
    dpnp.random.seed(seed)
    a = 2.56
    b = 0.8

    expected_mean = a / (a + b)
    expected_var = (a * b) / ((a + b)**2 * (a + b + 1))

    var = numpy.var(dpnp.random.beta(a=a, b=b, size=10**6))
    mean = numpy.mean(dpnp.random.beta(a=a, b=b, size=10**6))

    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_binomial_seed():
    seed = 28041990
    size = 100
    n, p = 10, .5  # number of trials, probability of each trial

    dpnp.random.seed(seed)
    a1 = dpnp.random.binomial(n, p, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.binomial(n, p, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_binomial_check_moments():
    seed = 28041990
    dpnp.random.seed(seed)
    n = 5
    p = 0.8
    expected_mean = n * p
    expected_var = n * p * (1 - p)
    var = numpy.var(dpnp.random.binomial(n=n, p=p, size=10**6))
    mean = numpy.mean(dpnp.random.binomial(n=n, p=p, size=10**6))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_binomial_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    n = 5
    p = 0.0
    res = numpy.asarray(dpnp.random.binomial(n=n, p=p, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0

    n = 0
    p = 0.5
    res = numpy.asarray(dpnp.random.binomial(n=n, p=p, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0

    n = 5
    p = 1.0
    res = numpy.asarray(dpnp.random.binomial(n=n, p=p, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 5


def test_chisquare_seed():
    seed = 28041990
    size = 100
    df = 3  # number of degrees of freedom

    dpnp.random.seed(seed)
    a1 = dpnp.random.chisquare(df, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.chisquare(df, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_chisquare_invalid_args():
    size = 10
    df = -1  # positive `df` is expected
    with pytest.raises(ValueError):
        dpnp.random.chisquare(df, size)


def test_exponential_seed():
    seed = 28041990
    size = 100
    scale = 3  # number of degrees of freedom

    dpnp.random.seed(seed)
    a1 = dpnp.random.exponential(scale, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.exponential(scale, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_exponential_invalid_args():
    size = 10
    scale = -1  # non-negative `scale` is expected
    with pytest.raises(ValueError):
        dpnp.random.exponential(scale, size)


def test_gamma_seed():
    seed = 28041990
    size = 100
    shape = 3.0  # shape param for gamma distr

    dpnp.random.seed(seed)
    a1 = dpnp.random.gamma(shape=shape, size=size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.gamma(shape=shape, size=size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_gamma_invalid_args():
    size = 10
    shape = -1   # non-negative `shape` is expected
    with pytest.raises(ValueError):
        dpnp.random.gamma(shape=shape, size=size)
    shape = 1.0   # OK
    scale = -1.0  # non-negative `shape` is expected
    with pytest.raises(ValueError):
        dpnp.random.gamma(shape, scale, size)


def test_gamma_check_moments():
    seed = 28041990
    dpnp.random.seed(seed)
    shape = 2.56
    scale = 0.8
    expected_mean = shape * scale
    expected_var = shape * scale * scale
    var = numpy.var(dpnp.random.gamma(shape=shape, scale=scale, size=10**6))
    mean = numpy.mean(dpnp.random.gamma(shape=shape, scale=scale, size=10**6))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_geometric_seed():
    seed = 28041990
    size = 100
    p = 0.8

    dpnp.random.seed(seed)
    a1 = dpnp.random.geometric(p, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.geometric(p, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_geometric_invalid_args():
    size = 10

    p = -1.0  # `p` is expected from (0, 1]
    with pytest.raises(ValueError):
        dpnp.random.geometric(p=p, size=size)


def test_geometric_check_moments():
    seed = 28041995
    dpnp.random.seed(seed)
    p = 0.8
    size = 10**6
    expected_mean = (1 - p) / p
    expected_var = (1 - p) / (p**2)
    var = numpy.var(dpnp.random.geometric(p=p, size=size))
    mean = numpy.mean(dpnp.random.geometric(p=p, size=size))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_geometric_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    p = 1.0
    expected_val = 1.0
    res = numpy.asarray(dpnp.random.geometric(p=p, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == expected_val


def test_gumbel_seed():
    seed = 28041990
    size = 100
    loc = 2.56
    scale = 0.8

    dpnp.random.seed(seed)
    a1 = dpnp.random.gumbel(loc, scale, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.gumbel(loc, scale, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_gumbel_invalid_args():
    size = 10

    loc = 3.0     # OK
    scale = -1.0  # non-negative `scale` is expected
    with pytest.raises(ValueError):
        dpnp.random.gumbel(loc=loc, scale=scale, size=size)


def test_gumbel_check_moments():
    seed = 28041990
    dpnp.random.seed(seed)
    loc = 12
    scale = 0.8
    size = 10**6
    expected_mean = loc + scale * numpy.euler_gamma
    expected_var = (numpy.pi**2 / 6) * (scale ** 2)

    var = numpy.var(dpnp.random.gumbel(loc=loc, scale=scale, size=size))
    mean = numpy.mean(dpnp.random.gumbel(loc=loc, scale=scale, size=size))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_gumbel_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    loc = 5
    scale = 0.0
    res = numpy.asarray(dpnp.random.gumbel(loc=loc, scale=scale, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == loc


def test_hypergeometric_seed():
    seed = 28041990

    size = 100

    ngood = 100
    nbad = 2
    nsample = 10

    dpnp.random.seed(seed)
    a1 = dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_hypergeometric_invalid_args():
    size = 10

    ngood = 100    # OK
    nbad = 2       # OK
    nsample = -10  # non-negative `nsamp` is expected
    with pytest.raises(ValueError):
        dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)

    ngood = 100    # OK
    nbad = -2      # non-negative `nbad` is expected
    nsample = 10   # OK
    with pytest.raises(ValueError):
        dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)

    ngood = -100   # non-negative `ngood` is expected
    nbad = 2       # OK
    nsample = 10   # OK
    with pytest.raises(ValueError):
        dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)

    ngood = 10
    nbad = 2
    nsample = 100
    # ngood + nbad >= nsample expected
    with pytest.raises(ValueError):
        dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)

    ngood = 10   # OK
    nbad = 2     # OK
    nsample = 0  # `nsample` is expected > 0
    with pytest.raises(ValueError):
        dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)


def test_hypergeometric_check_moments():
    seed = 28041995
    dpnp.random.seed(seed)
    ngood = 100
    nbad = 2
    nsample = 10

    size = 10**5
    expected_mean = nsample * (ngood / (ngood + nbad))
    expected_var = nsample * (ngood / (ngood + nbad)) * (nbad / (ngood + nbad)) * (((ngood + nbad) - nsample) / ((ngood + nbad) - 1))

    var = numpy.var(dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size))
    mean = numpy.mean(dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_hypergeometric_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    ngood = 100
    nbad = 0
    nsample = 10

    expected_val = nsample
    res = numpy.asarray(dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == expected_val

    ngood = 0
    nbad = 11
    nsample = 10

    expected_val = 0
    res = numpy.asarray(dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == expected_val


def test_laplace_seed():
    seed = 28041990
    size = 100
    loc = 2.56
    scale = 0.8

    dpnp.random.seed(seed)
    a1 = dpnp.random.laplace(loc, scale, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.laplace(loc, scale, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_laplace_invalid_args():
    size = 10

    loc = 3.0     # OK
    scale = -1.0  # positive `b` is expected
    with pytest.raises(ValueError):
        dpnp.random.laplace(loc=loc, scale=scale, size=size)


def test_laplace_check_moments():
    seed = 28041995
    dpnp.random.seed(seed)
    loc = 2.56
    scale = 0.8
    size = 10**6
    expected_mean = loc
    expected_var = 2 * scale * scale
    var = numpy.var(dpnp.random.laplace(loc=loc, scale=scale, size=size))
    mean = numpy.mean(dpnp.random.laplace(loc=loc, scale=scale, size=size))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_laplace_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    loc = 5
    scale = 0.0
    res = numpy.asarray(dpnp.random.laplace(loc=loc, scale=scale, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0


def test_lognormal_seed():
    seed = 28041990
    size = 100
    mean = 0.0
    sigma = 0.8

    dpnp.random.seed(seed)
    a1 = dpnp.random.lognormal(mean, sigma, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.lognormal(mean, sigma, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_lognormal_invalid_args():
    size = 10

    mean = 0.0
    sigma = -1.0  # non-negative `sigma` is expected
    with pytest.raises(ValueError):
        dpnp.random.lognormal(mean=mean, sigma=sigma, size=size)


def test_lognormal_check_moments():
    seed = 28041995
    dpnp.random.seed(seed)
    mean = 0.5
    sigma = 0.8
    size = 10**6
    expected_mean = numpy.exp(mean + (sigma ** 2) / 2)
    expected_var = (numpy.exp(sigma**2) - 1) * numpy.exp(2 * mean + sigma**2)
    var = numpy.var(dpnp.random.lognormal(mean=mean, sigma=sigma, size=size))
    mean = numpy.mean(dpnp.random.lognormal(mean=mean, sigma=sigma, size=size))
    assert math.isclose(var, expected_var, abs_tol=0.03)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_lognormal_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    mean = 0.5
    sigma = 0.0
    expected_val = numpy.exp(mean + (sigma ** 2) / 2)
    res = numpy.asarray(dpnp.random.lognormal(mean=mean, sigma=sigma, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == expected_val


def test_negative_binomial_seed():
    seed = 28041990
    size = 100
    n, p = 10, .5  # number of trials, probability of each trial

    dpnp.random.seed(seed)
    a1 = dpnp.random.negative_binomial(n, p, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.negative_binomial(n, p, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_negative_binomial_invalid_args():
    size = 10
    n = 10    # parameter `n`, OK
    p = -0.5  # parameter `p`, expected between [0, 1]
    with pytest.raises(ValueError):
        dpnp.random.negative_binomial(n, p, size)
    n = -10   # parameter `n`, expected non-negative
    p = 0.5   # parameter `p`, OK
    with pytest.raises(ValueError):
        dpnp.random.negative_binomial(n, p, size)


def test_negative_binomial_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    n = 5
    p = 1.0
    res = numpy.asarray(dpnp.random.negative_binomial(n=n, p=p, size=10))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0

    n = 5
    p = 0.0
    res = numpy.asarray(dpnp.random.negative_binomial(n=n, p=p, size=10))
    check_val = numpy.iinfo(res.dtype).min
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == check_val


def test_normal_seed():
    seed = 28041990
    size = 100
    loc = 2.56
    scale = 0.8

    dpnp.random.seed(seed)
    a1 = dpnp.random.normal(loc, scale, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.normal(loc, scale, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_normal_invalid_args():
    size = 10

    loc = 3.0     # OK
    scale = -1.0  # non-negative `scale` is expected
    with pytest.raises(ValueError):
        dpnp.random.normal(loc=loc, scale=scale, size=size)


def test_normal_check_moments():
    seed = 28041995
    dpnp.random.seed(seed)
    loc = 2.56
    scale = 0.8
    size = 10**6
    expected_mean = loc
    expected_var = scale**2
    var = numpy.var(dpnp.random.normal(loc=loc, scale=scale, size=size))
    mean = numpy.mean(dpnp.random.normal(loc=loc, scale=scale, size=size))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_normal_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    loc = 5
    scale = 0.0
    expected_val = loc
    res = numpy.asarray(dpnp.random.normal(loc=loc, scale=scale, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == expected_val


def test_poisson_seed():
    seed = 28041990
    size = 100
    lam = 0.8

    dpnp.random.seed(seed)
    a1 = dpnp.random.poisson(lam, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.poisson(lam, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_poisson_invalid_args():
    size = 10
    lam = -1.0    # non-negative `lam` is expected
    with pytest.raises(ValueError):
        dpnp.random.poisson(lam=lam, size=size)


def test_poisson_check_moments():
    seed = 28041995
    dpnp.random.seed(seed)
    lam = 0.8
    size = 10**6
    expected_mean = lam
    expected_var = lam
    var = numpy.var(dpnp.random.poisson(lam=lam, size=size))
    mean = numpy.mean(dpnp.random.poisson(lam=lam, size=size))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_poisson_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    lam = 0.0
    res = numpy.asarray(dpnp.random.poisson(lam=lam, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0


def test_randn_normal_distribution():
    """ Check if the sample obtained from the dpnp.random.randn differs from
    the normal distribution.
    Using ``scipy.stats.normaltest``.

    It is based on D’Agostino and Pearson’s test that combines skew
    and kurtosis to produce an omnibus test of normality,
    see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    """
    pts = 1000
    alpha = 0.05
    dpnp.random.seed(28041990)
    x = dpnp.random.randn(pts)
    _, p = stats.normaltest(x)
    # null hypothesis: x comes from a normal distribution.
    # The p-value is interpreted against an alpha of 5% and finds that the test
    # dataset does not significantly deviate from normal.
    # If p > alpha, the null hypothesis cannot be rejected.
    assert p > alpha


def test_rayleigh_seed():
    seed = 28041990
    size = 100
    scale = 0.8

    dpnp.random.seed(seed)
    a1 = dpnp.random.rayleigh(scale, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.rayleigh(scale, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_rayleigh_invalid_args():
    size = 10

    scale = -1.0  # positive `b` is expected
    with pytest.raises(ValueError):
        dpnp.random.rayleigh(scale=scale, size=size)


def test_rayleigh_check_moments():
    seed = 28041995
    dpnp.random.seed(seed)
    scale = 0.8
    size = 10**6
    expected_mean = scale * numpy.sqrt(numpy.pi / 2)
    expected_var = ((4 - numpy.pi) / 2) * scale * scale
    var = numpy.var(dpnp.random.rayleigh(scale=scale, size=size))
    mean = numpy.mean(dpnp.random.rayleigh(scale=scale, size=size))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_rayleigh_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    scale = 0.0
    res = numpy.asarray(dpnp.random.rayleigh(scale=scale, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0


def test_standard_cauchy_seed():
    seed = 28041990
    size = 100

    dpnp.random.seed(seed)
    a1 = dpnp.random.standard_cauchy(size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.standard_cauchy(size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_standard_exponential_seed():
    seed = 28041990
    size = 100

    dpnp.random.seed(seed)
    a1 = dpnp.random.standard_exponential(size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.standard_exponential(size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_standard_exponential_check_moments():
    seed = 28041995
    dpnp.random.seed(seed)
    size = 10**6
    expected_mean = 1.0
    expected_var = 1.0
    var = numpy.var(dpnp.random.standard_exponential(size=size))
    mean = numpy.mean(dpnp.random.standard_exponential(size=size))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_standard_normal_seed():
    seed = 28041990
    size = 100

    dpnp.random.seed(seed)
    a1 = dpnp.random.standard_normal(size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.standard_normal(size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_standard_normal_check_moments():
    seed = 28041995
    dpnp.random.seed(seed)
    size = 10**6
    expected_mean = 0.0
    expected_var = 1.0
    var = numpy.var(dpnp.random.standard_normal(size=size))
    mean = numpy.mean(dpnp.random.standard_normal(size=size))
    assert math.isclose(var, expected_var, abs_tol=0.003)
    assert math.isclose(mean, expected_mean, abs_tol=0.003)


def test_weibull_seed():
    seed = 28041990
    size = 100
    a = 2.56

    dpnp.random.seed(seed)
    a1 = dpnp.random.weibull(a, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.weibull(a, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_weibull_invalid_args():
    size = 10
    a = -1.0  # non-negative `a` is expected

    with pytest.raises(ValueError):
        dpnp.random.weibull(a=a, size=size)


def test_weibull_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    a = 0.0
    res = numpy.asarray(dpnp.random.weibull(a=a, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0
