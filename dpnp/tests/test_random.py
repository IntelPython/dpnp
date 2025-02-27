import unittest

import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal

import dpnp.random

from .helper import has_support_aspect64


class TestDistribution(unittest.TestCase):
    def check_extreme_value(self, dist_name, val, params):
        seed = 28041990
        size = 10
        dpnp.random.seed(seed)
        res = dpnp.asnumpy(getattr(dpnp.random, dist_name)(size=size, **params))
        assert len(numpy.unique(res)) == 1
        assert numpy.unique(res)[0] == val

    def check_moments(
        self, dist_name, expected_mean, expected_var, params, size=10**5
    ):
        seed = 28041995
        dpnp.random.seed(seed)
        res = getattr(dpnp.random, dist_name)(size=size, **params)
        var = dpnp.var(res)
        mean = dpnp.mean(res)
        assert_allclose(var, expected_var, atol=0.1)
        assert_allclose(mean, expected_mean, atol=0.1)

    def check_invalid_args(self, dist_name, params):
        size = 10
        with pytest.raises(ValueError):
            getattr(dpnp.random, dist_name)(size=size, **params)

    def check_seed(self, dist_name, params):
        seed = 28041990
        size = 10
        dpnp.random.seed(seed)
        a1 = getattr(dpnp.random, dist_name)(size=size, **params)
        dpnp.random.seed(seed)
        a2 = getattr(dpnp.random, dist_name)(size=size, **params)
        assert_allclose(a1, a2)


@pytest.mark.parametrize(
    "func",
    [dpnp.random.chisquare, dpnp.random.rand, dpnp.random.randn],
    ids=["chisquare", "rand", "randn"],
)
def test_input_size(func):
    output_shape = (10,)
    size = 10
    df = 3  # for dpnp.random.chisquare
    if func == dpnp.random.chisquare:
        res = func(df, size)
    else:
        res = func(size)
    assert output_shape == res.shape


@pytest.mark.parametrize(
    "func",
    [
        dpnp.random.chisquare,
        dpnp.random.random,
        dpnp.random.random_sample,
        dpnp.random.ranf,
        dpnp.random.sample,
    ],
    ids=["chisquare", "random", "random_sample", "ranf", "sample"],
)
def test_input_shape(func):
    shape = (10, 5)
    df = 3  # for dpnp.random.chisquare
    if func == dpnp.random.chisquare:
        res = func(df, shape)
    else:
        res = func(shape)
    assert shape == res.shape


@pytest.mark.parametrize(
    "func",
    [
        dpnp.random.random,
        dpnp.random.random_sample,
        dpnp.random.ranf,
        dpnp.random.sample,
        dpnp.random.rand,
    ],
    ids=["random", "random_sample", "ranf", "sample", "rand"],
)
def test_check_output(func):
    shape = (10, 5)
    size = 10 * 5
    if func == dpnp.random.rand:
        res = func(size)
    else:
        res = func(shape)
    assert dpnp.all(res >= 0)
    assert dpnp.all(res < 1)


@pytest.mark.parametrize(
    "func",
    [
        dpnp.random.random,
        dpnp.random.random_sample,
        dpnp.random.ranf,
        dpnp.random.sample,
        dpnp.random.rand,
    ],
    ids=["random", "random_sample", "ranf", "sample", "rand"],
)
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
    assert dpnp.allclose(a1, a2)


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
    var = dpnp.var(res)
    mean = dpnp.mean(res)
    assert_allclose(var, expected_var, atol=1e-02)
    assert_allclose(mean, expected_mean, atol=1e-02)


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsBeta(TestDistribution):
    def test_moments(self):
        a = 2.56
        b = 0.8
        expected_mean = a / (a + b)
        expected_var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        self.check_moments(
            "beta", expected_mean, expected_var, {"a": a, "b": b}
        )

    def test_invalid_args(self):
        a = 3.0  # OK
        b = -1.0  # positive `b` is expected
        self.check_invalid_args("beta", {"a": a, "b": b})
        a = -1.0  # positive `a` is expected
        b = 3.0  # OK
        self.check_invalid_args("beta", {"a": a, "b": b})

    def test_seed(self):
        a = 2.56
        b = 0.8
        self.check_seed("beta", {"a": a, "b": b})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsBinomial(TestDistribution):
    def test_extreme_value(self):
        n = 5
        p = 0.0
        expected_val = p
        self.check_extreme_value("binomial", expected_val, {"n": n, "p": p})
        n = 0
        p = 0.5
        expected_val = n
        self.check_extreme_value("binomial", expected_val, {"n": n, "p": p})
        n = 5
        p = 1.0
        expected_val = n
        self.check_extreme_value("binomial", expected_val, {"n": n, "p": p})

    def test_moments(self):
        n = 5
        p = 0.8
        expected_mean = n * p
        expected_var = n * p * (1 - p)
        self.check_moments(
            "binomial", expected_mean, expected_var, {"n": n, "p": p}
        )

    def test_invalid_args(self):
        n = -5  # non-negative `n` is expected
        p = 0.4  # OK
        self.check_invalid_args("binomial", {"n": n, "p": p})
        n = 5  # OK
        p = -0.5  # `p` is expected from [0, 1]
        self.check_invalid_args("binomial", {"n": n, "p": p})

    def test_seed(self):
        n, p = 10, 0.5  # number of trials, probability of each trial
        self.check_seed("binomial", {"n": n, "p": p})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsChisquare(TestDistribution):
    def test_invalid_args(self):
        df = -1  # positive `df` is expected
        self.check_invalid_args("chisquare", {"df": df})

    def test_seed(self):
        df = 3  # number of degrees of freedom
        self.check_seed("chisquare", {"df": df})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsExponential(TestDistribution):
    def test_invalid_args(self):
        scale = -1  # non-negative `scale` is expected
        self.check_invalid_args("exponential", {"scale": scale})

    def test_seed(self):
        scale = 3  # number of degrees of freedom
        self.check_seed("exponential", {"scale": scale})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsF(TestDistribution):
    def test_moments(self):
        dfnum = 12.56
        dfden = 13.0
        # for dfden > 2
        expected_mean = dfden / (dfden - 2)
        # for dfden > 4
        expected_var = (
            2
            * (dfden**2)
            * (dfnum + dfden - 2)
            / (dfnum * ((dfden - 2) ** 2) * ((dfden - 4)))
        )
        self.check_moments(
            "f", expected_mean, expected_var, {"dfnum": dfnum, "dfden": dfden}
        )

    def test_invalid_args(self):
        size = 10
        dfnum = -1.0  # positive `dfnum` is expected
        dfden = 1.0  # OK
        self.check_invalid_args("f", {"dfnum": dfnum, "dfden": dfden})
        dfnum = 1.0  # OK
        dfden = -1.0  # positive `dfden` is expected
        self.check_invalid_args("f", {"dfnum": dfnum, "dfden": dfden})

    def test_seed(self):
        dfnum = 3.56  # `dfden` param for Wald distr
        dfden = 2.8  # `dfden` param for Wald distr
        self.check_seed("f", {"dfnum": dfnum, "dfden": dfden})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsGamma(TestDistribution):
    def test_moments(self):
        shape = 2.56
        scale = 0.8
        expected_mean = shape * scale
        expected_var = shape * scale * scale
        self.check_moments(
            "gamma",
            expected_mean,
            expected_var,
            {"shape": shape, "scale": scale},
        )

    def test_invalid_args(self):
        size = 10
        shape = -1  # non-negative `shape` is expected
        self.check_invalid_args("gamma", {"shape": shape})
        shape = 1.0  # OK
        scale = -1.0  # non-negative `shape` is expected
        self.check_invalid_args("gamma", {"shape": shape, "scale": scale})

    def test_seed(self):
        shape = 3.0  # shape param for gamma distr
        self.check_seed("gamma", {"shape": shape})


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsGeometric(TestDistribution):
    def test_extreme_value(self):
        p = 1.0
        expected_val = p
        self.check_extreme_value("geometric", expected_val, {"p": p})

    def test_moments(self):
        p = 0.8
        expected_mean = (1 - p) / p
        expected_var = (1 - p) / (p**2)
        self.check_moments("geometric", expected_mean, expected_var, {"p": p})

    def test_invalid_args(self):
        size = 10
        p = -1.0  # `p` is expected from (0, 1]
        self.check_invalid_args("geometric", {"p": p})

    def test_seed(self):
        p = 0.8
        self.check_seed("geometric", {"p": p})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsGumbel(TestDistribution):
    def test_extreme_value(self):
        loc = 5
        scale = 0.0
        expected_val = loc
        self.check_extreme_value(
            "gumbel", expected_val, {"loc": loc, "scale": scale}
        )

    def test_moments(self):
        loc = 12
        scale = 0.8
        expected_mean = loc + scale * numpy.euler_gamma
        expected_var = (numpy.pi**2 / 6) * (scale**2)
        self.check_moments(
            "gumbel", expected_mean, expected_var, {"loc": loc, "scale": scale}
        )

    def test_invalid_args(self):
        size = 10
        loc = 3.0  # OK
        scale = -1.0  # non-negative `scale` is expected
        self.check_invalid_args("gumbel", {"loc": loc, "scale": scale})

    def test_seed(self):
        loc = 2.56
        scale = 0.8
        self.check_seed("gumbel", {"loc": loc, "scale": scale})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsHypergeometric(TestDistribution):
    def test_extreme_value(self):
        ngood = 100
        nbad = 0
        nsample = 10
        expected_val = nsample
        self.check_extreme_value(
            "hypergeometric",
            expected_val,
            {"ngood": ngood, "nbad": nbad, "nsample": nsample},
        )
        ngood = 0
        nbad = 11
        nsample = 10
        expected_val = 0
        self.check_extreme_value(
            "hypergeometric",
            expected_val,
            {"ngood": ngood, "nbad": nbad, "nsample": nsample},
        )

    def test_moments(self):
        ngood = 100
        nbad = 2
        nsample = 10
        expected_mean = nsample * (ngood / (ngood + nbad))
        expected_var = (
            expected_mean
            * (nbad / (ngood + nbad))
            * (((ngood + nbad) - nsample) / ((ngood + nbad) - 1))
        )
        self.check_moments(
            "hypergeometric",
            expected_mean,
            expected_var,
            {"ngood": ngood, "nbad": nbad, "nsample": nsample},
        )

    def test_invalid_args(self):
        size = 10
        ngood = 100  # OK
        nbad = 2  # OK
        nsample = -10  # non-negative `nsamp` is expected
        self.check_invalid_args(
            "hypergeometric", {"ngood": ngood, "nbad": nbad, "nsample": nsample}
        )

        ngood = 100  # OK
        nbad = -2  # non-negative `nbad` is expected
        nsample = 10  # OK
        self.check_invalid_args(
            "hypergeometric", {"ngood": ngood, "nbad": nbad, "nsample": nsample}
        )

        ngood = -100  # non-negative `ngood` is expected
        nbad = 2  # OK
        nsample = 10  # OK
        self.check_invalid_args(
            "hypergeometric", {"ngood": ngood, "nbad": nbad, "nsample": nsample}
        )

        ngood = 10
        nbad = 2
        nsample = 100
        # ngood + nbad >= nsample expected
        self.check_invalid_args(
            "hypergeometric", {"ngood": ngood, "nbad": nbad, "nsample": nsample}
        )

        ngood = 10  # OK
        nbad = 2  # OK
        nsample = 0  # `nsample` is expected > 0
        self.check_invalid_args(
            "hypergeometric", {"ngood": ngood, "nbad": nbad, "nsample": nsample}
        )

    def test_seed(self):
        ngood = 100
        nbad = 2
        nsample = 10
        self.check_seed(
            "hypergeometric", {"ngood": ngood, "nbad": nbad, "nsample": nsample}
        )


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsLaplace(TestDistribution):
    def test_extreme_value(self):
        loc = 5
        scale = 0.0
        expected_val = scale
        self.check_extreme_value(
            "laplace", expected_val, {"loc": loc, "scale": scale}
        )

    def test_moments(self):
        loc = 2.56
        scale = 0.8
        expected_mean = loc
        expected_var = 2 * scale * scale
        self.check_moments(
            "laplace", expected_mean, expected_var, {"loc": loc, "scale": scale}
        )

    def test_invalid_args(self):
        loc = 3.0  # OK
        scale = -1.0  # positive `b` is expected
        self.check_invalid_args("laplace", {"loc": loc, "scale": scale})

    def test_seed(self):
        loc = 2.56
        scale = 0.8
        self.check_seed("laplace", {"loc": loc, "scale": scale})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsLogistic(TestDistribution):
    def test_moments(self):
        loc = 2.56
        scale = 0.8
        expected_mean = loc
        expected_var = (scale**2) * (numpy.pi**2) / 3
        self.check_moments(
            "logistic",
            expected_mean,
            expected_var,
            {"loc": loc, "scale": scale},
        )

    def test_invalid_args(self):
        loc = 3.0  # OK
        scale = -1.0  # non-negative `scale` is expected
        self.check_invalid_args("logistic", {"loc": loc, "scale": scale})

    def test_seed(self):
        loc = 2.56
        scale = 0.8
        self.check_seed("logistic", {"loc": loc, "scale": scale})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsLognormal(TestDistribution):
    def test_extreme_value(self):
        mean = 0.5
        sigma = 0.0
        expected_val = numpy.exp(mean + (sigma**2) / 2)
        self.check_extreme_value(
            "lognormal", expected_val, {"mean": mean, "sigma": sigma}
        )

    def test_moments(self):
        mean = 0.5
        sigma = 0.8
        expected_mean = numpy.exp(mean + (sigma**2) / 2)
        expected_var = (numpy.exp(sigma**2) - 1) * numpy.exp(
            2 * mean + sigma**2
        )
        self.check_moments(
            "lognormal",
            expected_mean,
            expected_var,
            {"mean": mean, "sigma": sigma},
        )

    def test_invalid_args(self):
        mean = 0.0
        sigma = -1.0  # non-negative `sigma` is expected
        self.check_invalid_args("lognormal", {"mean": mean, "sigma": sigma})

    def test_seed(self):
        mean = 0.0
        sigma = 0.8
        self.check_seed("lognormal", {"mean": mean, "sigma": sigma})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsMultinomial(TestDistribution):
    def test_extreme_value(self):
        n = 0
        pvals = [1 / 6.0] * 6
        self.check_extreme_value("multinomial", n, {"n": n, "pvals": pvals})

    def test_moments(self):
        n = 10
        pvals = [1 / 6.0] * 6
        size = 10**5
        expected_mean = n * pvals[0]
        expected_var = n * pvals[0] * (1 - pvals[0])
        self.check_moments(
            "multinomial", expected_mean, expected_var, {"n": n, "pvals": pvals}
        )

    def test_check_sum(self):
        seed = 28041990
        size = 1
        n = 20
        pvals = [1 / 6.0] * 6
        dpnp.random.seed(seed)
        res = dpnp.random.multinomial(n, pvals, size)
        assert_equal(n, res.sum())

    def test_invalid_args(self):
        n = -10  # parameter `n`, non-negative expected
        pvals = [1 / 6.0] * 6  # parameter `pvals`, OK
        self.check_invalid_args("multinomial", {"n": n, "pvals": pvals})
        n = 10  # parameter `n`, OK
        pvals = [
            -1 / 6.0
        ] * 6  # parameter `pvals`, sum(pvals) expected between [0, 1]
        self.check_invalid_args("multinomial", {"n": n, "pvals": pvals})

    def test_seed(self):
        n = 20
        pvals = [1 / 6.0] * 6
        self.check_seed("multinomial", {"n": n, "pvals": pvals})

    def test_seed1(self):
        # pvals_size >= ntrial * 16 && ntrial <= 16
        n = 4
        pvals_size = 16 * n
        pvals = [1 / pvals_size] * pvals_size
        self.check_seed("multinomial", {"n": n, "pvals": pvals})


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsMultivariateNormal(TestDistribution):
    def test_moments(self):
        seed = 2804183
        dpnp.random.seed(seed)
        mean = [2.56, 3.23]
        cov = [[1, 0], [0, 1]]
        size = 10**5
        res = dpnp.random.multivariate_normal(mean=mean, cov=cov, size=size)
        res_mean = [dpnp.mean(res.T[0]), dpnp.mean(res.T[1])]
        assert dpnp.allclose(res_mean, mean)

    def test_invalid_args(self):
        mean = [2.56, 3.23]  # OK
        cov = [[1, 0]]  # `mean` and `cov` must have same length
        self.check_invalid_args(
            "multivariate_normal", {"mean": mean, "cov": cov}
        )
        mean = [[2.56, 3.23]]  # `mean` must be 1 dimensional
        cov = [[1, 0], [0, 1]]  # OK
        self.check_invalid_args(
            "multivariate_normal", {"mean": mean, "cov": cov}
        )
        mean = [2.56, 3.23]  # OK
        cov = [1, 0, 0, 1]  # `cov` must be 2 dimensional and square
        self.check_invalid_args(
            "multivariate_normal", {"mean": mean, "cov": cov}
        )

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
        self.check_seed("multivariate_normal", {"mean": mean, "cov": cov})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsNegativeBinomial(TestDistribution):
    def test_extreme_value(self):
        seed = 28041990
        dpnp.random.seed(seed)
        n = 5
        p = 1.0
        check_val = 0.0
        self.check_extreme_value(
            "negative_binomial", check_val, {"n": n, "p": p}
        )
        n = 5
        p = 0.0
        res = dpnp.random.negative_binomial(n=n, p=p, size=10)
        check_val = dpnp.iinfo(res).min
        assert len(dpnp.unique(res)) == 1
        assert dpnp.unique(res)[0] == check_val

    def test_invalid_args(self):
        n = 10  # parameter `n`, OK
        p = -0.5  # parameter `p`, expected between [0, 1]
        self.check_invalid_args("negative_binomial", {"n": n, "p": p})
        n = -10  # parameter `n`, expected non-negative
        p = 0.5  # parameter `p`, OK
        self.check_invalid_args("negative_binomial", {"n": n, "p": p})

    def test_seed(self):
        n, p = 10, 0.5  # number of trials, probability of each trial
        self.check_seed("negative_binomial", {"n": n, "p": p})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
class TestDistributionsNormal(TestDistribution):
    def test_extreme_value(self):
        loc = 5
        scale = 0.0
        expected_val = loc
        self.check_extreme_value(
            "normal", expected_val, {"loc": loc, "scale": scale}
        )

    def test_moments(self):
        loc = 2.56
        scale = 0.8
        expected_mean = loc
        expected_var = scale**2
        self.check_moments(
            "normal", expected_mean, expected_var, {"loc": loc, "scale": scale}
        )

    def test_invalid_args(self):
        loc = 3.0  # OK
        scale = -1.0  # non-negative `scale` is expected
        self.check_invalid_args("normal", {"loc": loc, "scale": scale})

    def test_seed(self):
        loc = 2.56
        scale = 0.8
        self.check_seed("normal", {"loc": loc, "scale": scale})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsNoncentralChisquare:
    @pytest.mark.parametrize(
        "df", [5.0, 1.0, 0.5], ids=["df_grt_1", "df_eq_1", "df_less_1"]
    )
    def test_moments(self, df):
        nonc = 20.0
        expected_mean = df + nonc
        expected_var = 2 * (df + 2 * nonc)
        size = 10**6
        seed = 28041995
        dpnp.random.seed(seed)
        res = dpnp.random.noncentral_chisquare(df, nonc, size=size)
        var = dpnp.var(res)
        mean = dpnp.mean(res)
        assert_allclose(var, expected_var, atol=0.6)
        assert_allclose(mean, expected_mean, atol=0.6)

    def test_invalid_args(self):
        size = 10
        df = 5.0  # OK
        nonc = -1.0  # non-negative `nonc` is expected
        with pytest.raises(ValueError):
            dpnp.random.noncentral_chisquare(df, nonc, size=size)
        df = -1.0  # positive `df` is expected
        nonc = 1.0  # OK
        with pytest.raises(ValueError):
            dpnp.random.noncentral_chisquare(df, nonc, size=size)

    @pytest.mark.parametrize(
        "df", [5.0, 1.0, 0.5], ids=["df_grt_1", "df_eq_1", "df_less_1"]
    )
    def test_seed(self, df):
        seed = 28041990
        size = 10
        nonc = 1.8
        dpnp.random.seed(seed)
        a1 = dpnp.asarray(dpnp.random.noncentral_chisquare(df, nonc, size=size))
        dpnp.random.seed(seed)
        a2 = dpnp.asarray(dpnp.random.noncentral_chisquare(df, nonc, size=size))
        assert dpnp.allclose(a1, a2)


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsPareto(TestDistribution):
    def test_moments(self):
        a = 30.0
        expected_mean = a / (a - 1)
        expected_var = a / (((a - 1) ** 2) * (a - 2))
        self.check_moments("pareto", expected_mean, expected_var, {"a": a})

    def test_invalid_args(self):
        size = 10
        a = -1.0  # positive `a` is expected
        self.check_invalid_args("pareto", {"a": a})

    def test_seed(self):
        a = 3.0  # a param for pareto distr
        self.check_seed("pareto", {"a": a})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsPoisson(TestDistribution):
    def test_extreme_value(self):
        lam = 0.0
        self.check_extreme_value("poisson", lam, {"lam": lam})

    def test_moments(self):
        lam = 0.8
        expected_mean = lam
        expected_var = lam
        self.check_moments("poisson", expected_mean, expected_var, {"lam": lam})

    def test_invalid_args(self):
        lam = -1.0  # non-negative `lam` is expected
        self.check_invalid_args("poisson", {"lam": lam})

    def test_seed(self):
        lam = 0.8
        self.check_seed("poisson", {"lam": lam})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsPower(TestDistribution):
    def test_moments(self):
        a = 30.0
        neg_a = -a
        expected_mean = neg_a / (neg_a - 1)
        expected_var = neg_a / (((neg_a - 1) ** 2) * (neg_a - 2))
        self.check_moments("power", expected_mean, expected_var, {"a": a})

    def test_invalid_args(self):
        size = 10
        a = -1.0  # positive `a` is expected
        self.check_invalid_args("power", {"a": a})

    def test_seed(self):
        a = 3.0  # a param for pareto distr
        self.check_seed("power", {"a": a})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsRayleigh(TestDistribution):
    def test_extreme_value(self):
        scale = 0.0
        self.check_extreme_value("rayleigh", scale, {"scale": scale})

    def test_moments(self):
        scale = 0.8
        expected_mean = scale * numpy.sqrt(numpy.pi / 2)
        expected_var = ((4 - numpy.pi) / 2) * scale * scale
        self.check_moments(
            "rayleigh", expected_mean, expected_var, {"scale": scale}
        )

    def test_invalid_args(self):
        scale = -1.0  # positive `b` is expected
        self.check_invalid_args("rayleigh", {"scale": scale})

    def test_seed(self):
        scale = 0.8
        self.check_seed("rayleigh", {"scale": scale})


class TestDistributionsStandardCauchy(TestDistribution):
    def test_seed(self):
        self.check_seed("standard_cauchy", {})


class TestDistributionsStandardExponential(TestDistribution):
    def test_moments(self):
        shape = 0.8
        expected_mean = 1.0
        expected_var = 1.0
        self.check_moments(
            "standard_exponential", expected_mean, expected_var, {}
        )

    def test_seed(self):
        self.check_seed("standard_exponential", {})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsStandardGamma(TestDistribution):
    def test_extreme_value(self):
        self.check_extreme_value("standard_gamma", 0.0, {"shape": 0.0})

    def test_moments(self):
        shape = 0.8
        expected_mean = shape
        expected_var = shape
        self.check_moments(
            "standard_gamma", expected_mean, expected_var, {"shape": shape}
        )

    def test_invalid_args(self):
        shape = -1  # non-negative `shape` is expected
        self.check_invalid_args("standard_gamma", {"shape": shape})

    def test_seed(self):
        self.check_seed("standard_gamma", {"shape": 0.0})


class TestDistributionsStandardNormal(TestDistribution):
    def test_moments(self):
        expected_mean = 0.0
        expected_var = 1.0
        self.check_moments("standard_normal", expected_mean, expected_var, {})

    def test_seed(self):
        self.check_seed("standard_normal", {})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsStandardT(TestDistribution):
    def test_moments(self):
        df = 300.0
        expected_mean = 0.0
        expected_var = df / (df - 2)
        self.check_moments(
            "standard_t", expected_mean, expected_var, {"df": df}
        )

    def test_invalid_args(self):
        df = 0.0  # positive `df` is expected
        self.check_invalid_args("standard_t", {"df": df})

    def test_seed(self):
        self.check_seed("standard_t", {"df": 10.0})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsTriangular(TestDistribution):
    def test_moments(self):
        left = 1.0
        mode = 2.0
        right = 3.0
        expected_mean = (left + mode + right) / 3
        expected_var = (
            left**2
            + mode**2
            + right**2
            - left * mode
            - left * right
            - mode * right
        ) / 18
        self.check_moments(
            "triangular",
            expected_mean,
            expected_var,
            {"left": left, "mode": mode, "right": right},
        )

    def test_invalid_args(self):
        left = 2.0  # `left` is expected <= `mode`
        mode = 1.0  # `mode` is expected > `left`
        right = 3.0  # OK
        self.check_invalid_args(
            "triangular", {"left": left, "mode": mode, "right": right}
        )

        left = 1.0  # OK
        mode = 3.0  # `mode` is expected <= `right`
        right = 2.0  # `right` is expected > `mode`
        self.check_invalid_args(
            "triangular", {"left": left, "mode": mode, "right": right}
        )

    def test_seed(self):
        left = 1.0
        mode = 2.0
        right = 3.0
        self.check_seed(
            "triangular", {"left": left, "mode": mode, "right": right}
        )


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
class TestDistributionsUniform(TestDistribution):
    def test_extreme_value(self):
        low = 1.0
        high = 1.0
        expected_val = low
        self.check_extreme_value(
            "uniform", expected_val, {"low": low, "high": high}
        )

    def test_moments(self):
        low = 1.0
        high = 2.0
        expected_mean = (low + high) / 2
        expected_var = ((high - low) ** 2) / 12
        self.check_moments(
            "uniform", expected_mean, expected_var, {"low": low, "high": high}
        )

    def test_seed(self):
        low = 1.0
        high = 2.0
        self.check_seed("uniform", {"low": low, "high": high})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsVonmises:
    @pytest.mark.parametrize(
        "kappa", [5.0, 0.5], ids=["large_kappa", "small_kappa"]
    )
    def test_moments(self, kappa):
        size = 10**6
        mu = 2.0

        numpy_res = numpy.random.vonmises(mu, kappa, size=size)
        expected_mean = numpy.mean(numpy_res)
        expected_var = numpy.var(numpy_res)

        res = dpnp.random.vonmises(mu, kappa, size=size)
        var = dpnp.var(res)
        mean = dpnp.mean(res)
        assert_allclose(var, expected_var, atol=0.6)
        assert_allclose(mean, expected_mean, atol=0.6)

    def test_invalid_args(self):
        size = 10
        mu = 5.0  # OK
        kappa = -1.0  # non-negative `kappa` is expected
        with pytest.raises(ValueError):
            dpnp.random.vonmises(mu, kappa, size=size)

    @pytest.mark.parametrize(
        "kappa", [5.0, 0.5], ids=["large_kappa", "small_kappa"]
    )
    def test_seed(self, kappa):
        seed = 28041990
        size = 10
        mu = 2.0
        dpnp.random.seed(seed)
        a1 = dpnp.asarray(dpnp.random.vonmises(mu, kappa, size=size))
        dpnp.random.seed(seed)
        a2 = dpnp.asarray(dpnp.random.vonmises(mu, kappa, size=size))
        assert dpnp.allclose(a1, a2)


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsWald(TestDistribution):
    def test_moments(self):
        size = 5 * 10**6
        mean = 3.56
        scale = 2.8
        expected_mean = mean
        expected_var = (mean**3) / scale
        self.check_moments(
            "wald",
            expected_mean,
            expected_var,
            {"mean": mean, "scale": scale},
            size=size,
        )

    def test_invalid_args(self):
        size = 10
        mean = -1.0  # positive `mean` is expected
        scale = 1.0  # OK
        self.check_invalid_args("wald", {"mean": mean, "scale": scale})
        mean = 1.0  # OK
        scale = -1.0  # positive `scale` is expected
        self.check_invalid_args("wald", {"mean": mean, "scale": scale})

    def test_seed(self):
        mean = 3.56  # `mean` param for Wald distr
        scale = 2.8  # `scale` param for Wald distr
        self.check_seed("wald", {"mean": mean, "scale": scale})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsWeibull(TestDistribution):
    def test_extreme_value(self):
        a = 0.0
        expected_val = a
        self.check_extreme_value("weibull", expected_val, {"a": a})

    def test_invalid_args(self):
        a = -1.0  # non-negative `a` is expected
        self.check_invalid_args("weibull", {"a": a})

    def test_seed(self):
        a = 2.56
        self.check_seed("weibull", {"a": a})


@pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestDistributionsZipf(TestDistribution):
    def test_invalid_args(self):
        a = 1.0  # parameter `a` is expected greater than 1.
        self.check_invalid_args("zipf", {"a": a})

    def test_seed(self):
        a = 2.56
        self.check_seed("zipf", {"a": a})


class TestPermutationsTestShuffle:
    @pytest.mark.parametrize(
        "dtype", [dpnp.float32, dpnp.float64, dpnp.int32, dpnp.int64]
    )
    def test_shuffle(self, dtype):
        seed = 28041990
        input_x_int64 = dpnp.asarray(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=dpnp.int64
        )
        input_x = dpnp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=dtype)
        dpnp.random.seed(seed)
        dpnp.random.shuffle(input_x_int64)  # inplace
        desired_x = dpnp.astype(input_x_int64, dtype)
        dpnp.random.seed(seed)
        dpnp.random.shuffle(input_x)  # inplace
        actual_x = input_x
        assert_array_equal(actual_x, desired_x)

    @pytest.mark.parametrize(
        "dtype", [dpnp.float32, dpnp.float64, dpnp.int32, dpnp.int64]
    )
    def test_no_miss_numbers(self, dtype):
        seed = 28041990
        input_x = dpnp.asarray([5, 4, 0, 7, 6, 1, 8, 3, 2, 9], dtype=dtype)
        desired_x = dpnp.sort(input_x)
        dpnp.random.seed(seed)
        dpnp.random.shuffle(input_x)  # inplace
        output_x = input_x
        actual_x = dpnp.sort(output_x)
        assert_array_equal(actual_x, desired_x)

    @pytest.mark.skipif(not has_support_aspect64(), reason="Failed on Iris Xe")
    @pytest.mark.parametrize(
        "conv",
        [
            lambda x: dpnp.array([]),
            lambda x: dpnp.astype(dpnp.asarray(x), dpnp.int8),
            lambda x: dpnp.astype(dpnp.asarray(x), dpnp.float32),
            lambda x: dpnp.asarray(x).astype(dpnp.complex64),
            lambda x: dpnp.astype(dpnp.asarray(x), object),
            lambda x: dpnp.asarray([[i, i] for i in x]),
            lambda x: dpnp.vstack([x, x]).T,
            lambda x: (
                dpnp.asarray(
                    [(i, i) for i in x], [("a", int), ("b", int)]
                ).view(dpnp.recarray)
            ),
            lambda x: dpnp.asarray(
                [(i, i) for i in x], [("a", object), ("b", dpnp.int32)]
            ),
        ],
        ids=[
            "lambda x: dpnp.array([])",
            "lambda x: dpnp.astype(dpnp.asarray(x), dpnp.int8)",
            "lambda x: dpnp.astype(dpnp.asarray(x), dpnp.float32)",
            "lambda x: dpnp.asarray(x).astype(dpnp.complex64)",
            "lambda x: dpnp.astype(dpnp.asarray(x), object)",
            "lambda x: dpnp.asarray([[i, i] for i in x])",
            "lambda x: dpnp.vstack([x, x]).T",
            "lambda x: (dpnp.asarray([(i, i) for i in x], ["
            '("a", int), ("b", int)]).view(dpnp.recarray))',
            'lambda x: dpnp.asarray([(i, i) for i in x], [("a", object), ("b", dpnp.int32)])]',
        ],
    )
    def test_shuffle1(self, conv):
        # `conv` contains test lists, arrays (of various dtypes), and multidimensional
        # versions of both, c-contiguous or not.
        #
        # This test is a modification of the original tests of `numpy.random` (both the same):
        # * tests/test_random.py::TestRandomDist::test_shuffle
        # * tests/test_randomstate.py::TestRandomDist::test_shuffle
        #
        # The original tests do not have a parameterized launch and they
        # do not correctly check of the results for the dpnp RNG engine.

        # Computing desired 1 dim list for given 1 dim list
        # on the current device for the given seed number.
        seed = 1234567890

        dpnp.random.seed(seed)
        list_1d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        dpnp_1d = dpnp.array(list_1d)
        dpnp.random.shuffle(dpnp_1d)  # inplace

        dpnp.random.seed(seed)
        alist = conv(list_1d)
        dpnp.random.shuffle(alist)  # inplace
        actual = alist
        desired = conv(dpnp_1d)
        assert_array_equal(actual, desired)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @pytest.mark.parametrize(
        "conv",
        [lambda x: x, lambda x: [(i, i) for i in x]],
        ids=[
            "lambda x: x",
            "lambda x: [(i, i) for i in x]",
        ],
    )
    def test_shuffle1_fallback(self, conv):
        # This is parameterized version of original tests of `numpy.random` (both the same):
        # * tests/test_random.py::TestRandomDist::test_shuffle
        # * tests/test_randomstate.py::TestRandomDist::test_shuffle

        seed = 1234567890

        dpnp.random.seed(seed)
        alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        dpnp.random.shuffle(alist)
        actual = alist
        desired = conv([0, 1, 9, 6, 2, 4, 5, 8, 7, 3])
        assert_array_equal(actual, desired)
