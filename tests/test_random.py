import pytest

import dpnp.random
import numpy
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
    a = 2.56
    b = 0.8

    expected_mean = a / (a + b)
    expected_var = (a * b) / ((a + b)**2 * (a + b + 1))

    seed = 28041990
    dpnp.random.seed(seed)
    res = dpnp.random.beta(a=a, b=b, size=10**5)

    var = numpy.var(res)
    mean = numpy.mean(res)

    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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
    n = 5
    p = 0.8

    expected_mean = n * p
    expected_var = n * p * (1 - p)

    seed = 28041990
    dpnp.random.seed(seed)
    res = dpnp.random.binomial(n=n, p=p, size=10**5)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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


def test_binomial_invalid_args():
    size = 10
    n = -5     # non-negative `n` is expected
    p = 0.4    # OK
    with pytest.raises(ValueError):
        dpnp.random.binomial(n=n, p=p, size=size)

    n = 5      # OK
    p = -0.5   # `p` is expected from [0, 1]
    with pytest.raises(ValueError):
        dpnp.random.binomial(n=n, p=p, size=size)


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
    shape = 2.56
    scale = 0.8

    expected_mean = shape * scale
    expected_var = shape * scale * scale

    seed = 28041990
    dpnp.random.seed(seed)
    res =dpnp.random.gamma(shape=shape, scale=scale, size=10**5)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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
    p = 0.8
    size = 10**5

    expected_mean = (1 - p) / p
    expected_var = (1 - p) / (p**2)

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.geometric(p=p, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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
    loc = 12
    scale = 0.8
    size = 10**5

    expected_mean = loc + scale * numpy.euler_gamma
    expected_var = (numpy.pi**2 / 6) * (scale ** 2)

    seed = 28041990
    dpnp.random.seed(seed)
    res = dpnp.random.gumbel(loc=loc, scale=scale, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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
    ngood = 100
    nbad = 2
    nsample = 10
    size = 10**5

    expected_mean = nsample * (ngood / (ngood + nbad))
    expected_var = expected_mean * (nbad / (ngood + nbad)) * (((ngood + nbad) - nsample) / ((ngood + nbad) - 1))

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.hypergeometric(ngood=ngood, nbad=nbad, nsample=nsample, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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
    loc = 2.56
    scale = 0.8
    size = 10**5

    expected_mean = loc
    expected_var = 2 * scale * scale

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.laplace(loc=loc, scale=scale, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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
    mean = 0.5
    sigma = 0.8
    size = 10**5

    expected_mean = numpy.exp(mean + (sigma ** 2) / 2)
    expected_var = (numpy.exp(sigma**2) - 1) * numpy.exp(2 * mean + sigma**2)

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.lognormal(mean=mean, sigma=sigma, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


def test_lognormal_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    mean = 0.5
    sigma = 0.0
    expected_val = numpy.exp(mean + (sigma ** 2) / 2)
    res = numpy.asarray(dpnp.random.lognormal(mean=mean, sigma=sigma, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == expected_val


def test_multinomial_seed():
    seed = 28041990
    size = 100
    n = 20
    pvals = [1 / 6.] * 6

    dpnp.random.seed(seed)
    a1 = dpnp.random.multinomial(n, pvals, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.multinomial(n, pvals, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_multinomial_check_sum():
    seed = 28041990
    size = 1
    n = 20
    pvals = [1 / 6.] * 6

    dpnp.random.seed(seed)
    res = dpnp.random.multinomial(n, pvals, size)
    assert_allclose(n, sum(res), rtol=1e-07, atol=0)


def test_multinomial_invalid_args():
    size = 10
    n = -10                # parameter `n`, non-negative expected
    pvals = [1 / 6.] * 6   # parameter `pvals`, OK
    with pytest.raises(ValueError):
        dpnp.random.multinomial(n, pvals, size)
    n = 10                 # parameter `n`, OK
    pvals = [-1 / 6.] * 6  # parameter `pvals`, sum(pvals) expected between [0, 1]
    with pytest.raises(ValueError):
        dpnp.random.multinomial(n, pvals, size)
    n = 10                          # parameter `n`, OK
    pvals = [1 / 6.] * 6 + [1 / 6.]  # parameter `pvals`, sum(pvals) expected between [0, 1]
    with pytest.raises(ValueError):
        dpnp.random.multinomial(n, pvals, size)


def test_multinomial_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    n = 0
    pvals = [1 / 6.] * 6

    res = numpy.asarray(dpnp.random.multinomial(n, pvals, size=1))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0


def test_multinomial_check_moments():
    n = 10
    pvals = [1 / 6.] * 6
    size = 10**5

    expected_mean = n * pvals[0]
    expected_var = n * pvals[0] * (1 - pvals[0])

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.multinomial(n=n, pvals=pvals, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


def test_multivariate_normal_output_shape_check():
    seed = 28041990
    size = 100
    mean = [2.56, 3.23]
    cov = [[1, 0], [0, 1]]
    expected_shape = (100, 2)

    dpnp.random.seed(seed)
    res = dpnp.random.multivariate_normal(mean, cov, size=100)
    assert res.shape == expected_shape


def test_multivariate_normal_seed():
    seed = 28041990
    size = 100
    mean = [2.56, 3.23]
    cov = [[1, 0], [0, 1]]

    dpnp.random.seed(seed)
    a1 = dpnp.random.multivariate_normal(mean, cov, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.multivariate_normal(mean, cov, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_multivariate_normal_invalid_args():
    size = 10

    mean = [2.56, 3.23]  # OK
    cov = [[1, 0]]       # `mean` and `cov` must have same length
    with pytest.raises(ValueError):
        dpnp.random.multivariate_normal(mean=mean, cov=cov, size=size)

    mean = [[2.56, 3.23]]   # `mean` must be 1 dimensional
    cov = [[1, 0], [0, 1]]  # OK
    with pytest.raises(ValueError):
        dpnp.random.multivariate_normal(mean=mean, cov=cov, size=size)

    mean = [2.56, 3.23]  # OK
    cov = [1, 0, 0, 1]   # `cov` must be 2 dimensional and square
    with pytest.raises(ValueError):
        dpnp.random.multivariate_normal(mean=mean, cov=cov, size=size)


def test_multivariate_normal_check_moments():
    seed = 2804183
    dpnp.random.seed(seed)

    mean = [2.56, 3.23]
    cov = [[1, 0], [0, 1]]
    size = 10**5

    res = numpy.array(dpnp.random.multivariate_normal(mean=mean, cov=cov, size=size))
    res_mean = [numpy.mean(res.T[0]), numpy.mean(res.T[1])]

    assert_allclose(res_mean, mean, rtol=1e-02, atol=0)


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
    loc = 2.56
    scale = 0.8
    size = 10**5

    expected_mean = loc
    expected_var = scale**2

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.normal(loc=loc, scale=scale, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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
    lam = 0.8
    size = 10**5

    expected_mean = lam
    expected_var = lam

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.poisson(lam=lam, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


def test_poisson_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    lam = 0.0
    res = numpy.asarray(dpnp.random.poisson(lam=lam, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0


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
    scale = 0.8
    size = 10**5

    expected_mean = scale * numpy.sqrt(numpy.pi / 2)
    expected_var = ((4 - numpy.pi) / 2) * scale * scale

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.rayleigh(scale=scale, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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
    expected_mean = 1.0
    expected_var = 1.0
    size = 10**6

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.standard_exponential(size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


def test_standard_gamma_seed():
    seed = 28041990
    size = 100
    shape = 3.0  # shape param for gamma distr

    dpnp.random.seed(seed)
    a1 = dpnp.random.standard_gamma(shape=shape, size=size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.standard_gamma(shape=shape, size=size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_standard_gamma_invalid_args():
    size = 10
    shape = -1   # non-negative `shape` is expected
    with pytest.raises(ValueError):
        dpnp.random.standard_gamma(shape=shape, size=size)


def test_standard_gamma_check_moments():
    shape = 0.8
    size = 10**5

    expected_mean = shape
    expected_var = shape

    seed = 28041990
    dpnp.random.seed(seed)
    res = dpnp.random.gamma(shape=shape, size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


def test_standard_gamma_check_extreme_value():
    seed = 28041990
    dpnp.random.seed(seed)

    shape = 0.0

    res = numpy.asarray(dpnp.random.gamma(shape=shape, size=100))
    assert len(numpy.unique(res)) == 1
    assert numpy.unique(res)[0] == 0.0


def test_standard_normal_seed():
    seed = 28041990
    size = 100

    dpnp.random.seed(seed)
    a1 = dpnp.random.standard_normal(size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.standard_normal(size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_standard_normal_check_moments():
    expected_mean = 0.0
    expected_var = 1.0
    size = 10**5

    seed = 28041995
    dpnp.random.seed(seed)
    res = dpnp.random.standard_normal(size=size)

    var = numpy.var(res)
    mean = numpy.mean(res)
    assert math.isclose(var, expected_var, abs_tol=0.1)
    assert math.isclose(mean, expected_mean, abs_tol=0.1)


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
