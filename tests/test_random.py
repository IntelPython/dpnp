import pytest

import dpnp.random
import numpy
# from scipy import stats
from numpy.testing import assert_allclose


@pytest.mark.parametrize("func",
                         [dpnp.random.chisquare,
                          dpnp.random.rand,
                          dpnp.random.randn],
                         ids=['chisquare', 'rand', 'randn'])
def test_random_input_size(func):
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
def test_random_input_shape(func):
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
def test_random_check_otput(func):
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


@pytest.mark.parametrize("func",
                         [dpnp.random.random,
                          dpnp.random.random_sample,
                          dpnp.random.ranf,
                          dpnp.random.sample,
                          dpnp.random.rand],
                         ids=['random', 'random_sample',
                              'ranf', 'sample', 'rand'])
def test_radnom_seed(func):
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


def test_radnom_exponential_seed():
    seed = 28041990
    size = 100
    scale = 3  # number of degrees of freedom

    dpnp.random.seed(seed)
    a1 = dpnp.random.exponential(scale, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.exponential(scale, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_exponential_invalid_scale():
    size = 10
    scale = -1  # non-negative `scale` is expected
    with pytest.raises(ValueError):
        dpnp.random.exponential(scale, size)


def test_radnom_chisquare_seed():
    seed = 28041990
    size = 100
    df = 3  # number of degrees of freedom

    dpnp.random.seed(seed)
    a1 = dpnp.random.chisquare(df, size)
    dpnp.random.seed(seed)
    a2 = dpnp.random.chisquare(df, size)
    assert_allclose(a1, a2, rtol=1e-07, atol=0)


def test_chisquare_invalid_df():
    size = 10
    df = -1  # positive `df` is expected
    with pytest.raises(ValueError):
        dpnp.random.chisquare(df, size)


def test_radnom_gamma_seed():
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
    shape = -1 # non-negative `shape` is expected
    with pytest.raises(ValueError):
        dpnp.random.gamma(shape=shape, size=size)
    shape = 1.0  # OK
    scale = -1.0 # non-negative `shape` is expected
    with pytest.raises(ValueError):
        dpnp.random.gamma(shape, scale, size)
