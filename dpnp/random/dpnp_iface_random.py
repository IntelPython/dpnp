# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""
Module Intel NumPy Random

Set of functions to implement NumPy random module API

    .. seealso:: :obj:`numpy.random`

"""


import dpnp
import numpy

from dpnp.backend import *
from dpnp.dparray import dparray
from dpnp.dpnp_utils import *
from dpnp.random._random import *


__all__ = [
    'beta',
    'binomial',
    'bytes',
    'chisquare',
    'choice',
    'dirichlet',
    'exponential',
    'f',
    'gamma',
    'geometric',
    'gumbel',
    'hypergeometric',
    'laplace',
    'logistic',
    'lognormal',
    'logseries',
    'multinomial',
    'multivariate_normal',
    'negative_binomial',
    'normal',
    'noncentral_chisquare',
    'noncentral_f',
    'pareto',
    'permutation',
    'poisson',
    'power',
    'rand',
    'randint',
    'randn',
    'random',
    'random_integers',
    'random_sample',
    'ranf',
    'rayleigh',
    'sample',
    'shuffle',
    'seed',
    'standard_cauchy',
    'standard_exponential',
    'standard_gamma',
    'standard_normal',
    'standard_t',
    'triangular',
    'uniform',
    'vonmises',
    'wald',
    'weibull',
    'zipf'
]


def beta(a, b, size=None):
    """Beta distribution.

    Draw samples from a Beta distribution.

    The Beta distribution is a special case of the Dirichlet distribution,
    and is related to the Gamma distribution.  It has the probability
    distribution function

    .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}
                                                     (1 - x)^{\\beta - 1},

    where the normalization, B, is the beta function,

    .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}
                                 (1 - t)^{\\beta - 1} dt.

    It is often seen in Bayesian inference and order statistics.

    Parameters
    ----------
    a : float
        Alpha, positive (>0).
    b : float
        Beta, positive (>0).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` and ``b`` are both scalars.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized beta distribution.

    """

    # TODO:
    # array_like of floats for `a`, `b`
    if not use_origin_backend(a) and dpnp_queue_is_cpu():
        if size is None:
            size = 1
        if isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    pass
        elif not isinstance(size, int):
            pass
        elif a <= 0:
            pass
        elif b <= 0:
            pass
        else:
            return dpnp_beta(a, b, size)

    return call_origin(numpy.random.beta, a, b, size)


def binomial(n, p, size=None):
    """Binomial distribution.

    Draw samples from a binomial distribution.

    Samples are drawn from a binomial distribution with specified
    parameters, n trials and p probability of success where
    n an integer >= 0 and p is in the interval [0,1]. (n may be
    input as a float, but it is truncated to an integer in use)

    Parameters
    ----------
    n : int
        Parameter of the distribution, >= 0. Floats are also accepted,
        but they will be truncated to integers.
    p : float
        Parameter of the distribution, >= 0 and <=1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``n`` and ``p`` are both scalars.
        Otherwise, ``np.broadcast(n, p).size`` samples are drawn.

    Returns
    -------
    out : dparray, int32
        Drawn samples from the parameterized binomial distribution, where
        each sample is equal to the number of successes over the n trials.

    Notes
    -----
    The probability density for the binomial distribution is

    .. math:: P(N) = \\binom{n}{N}p^N(1-p)^{n-N},

    where :math:`n` is the number of trials, :math:`p` is the probability
    of success, and :math:`N` is the number of successes.

    When estimating the standard error of a proportion in a population by
    using a random sample, the normal distribution works well unless the
    product p*n <=5, where p = population proportion estimate, and n =
    number of samples, in which case the binomial distribution is used
    instead. For example, a sample of 15 people shows 4 who are left
    handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4,
    so the binomial distribution should be used in this case.

    References
    ----------
    .. [1] Dalgaard, Peter, "Introductory Statistics with R",
           Springer-Verlag, 2002.
    .. [2] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
           Fifth Edition, 2002.
    .. [3] Lentner, Marvin, "Elementary Applied Statistics", Bogden
           and Quigley, 1972.
    .. [4] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A
           Wolfram Web Resource.
           http://mathworld.wolfram.com/BinomialDistribution.html
    .. [5] Wikipedia, "Binomial distribution",
           https://en.wikipedia.org/wiki/Binomial_distribution

    Examples
    --------
    Draw samples from the distribution:
    >>> n, p = 10, .5  # number of trials, probability of each trial
    >>> s = dpnp.random.binomial(n, p, 1000)
    # result of flipping a coin 10 times, tested 1000 times.
    A real world example. A company drills 9 wild-cat oil exploration
    wells, each with an estimated probability of success of 0.1. All nine
    wells fail. What is the probability of that happening?
    Let's do 20,000 trials of the model, and count the number that
    generate zero positive results.
    >>> sum(dpnp.random.binomial(9, 0.1, 20000) == 0)/20000.
    # answer = 0.38885, or 38%.

    """

    if not use_origin_backend(n) and dpnp_queue_is_cpu():
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("binomial", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("binomial", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `p` param
        if p > 1 or p < 0:
            checker_throw_value_error("binomial", "p", p, "in [0, 1]")
        if n < 0:
            checker_throw_value_error("binomial", "n", n, "non-negative")

        return dpnp_binomial(int(n), p, size)

    return call_origin(numpy.random.binomial, n, p, size)


def bytes(length):
    """Bytes

    Return random bytes.

    Parameters
    ----------
    length : int
        Number of random bytes.

    Returns
    -------
    out : str
        String of length `length`.

    Examples
    --------
    >>> dpnp.random.bytes(10)
    ' eh\\x85\\x022SZ\\xbf\\xa4' #random

    """

    return call_origin(numpy.random.bytes, length)


def chisquare(df, size=None):
    """
    chisquare(df, size=None)

    Draw samples from a chi-square distribution.

    When `df` independent random variables, each with standard normal
    distributions (mean 0, variance 1), are squared and summed, the
    resulting distribution is chi-square (see Notes).  This distribution
    is often used in hypothesis testing.

    Parameters
    ----------
    df : float
         Number of degrees of freedom, must be > 0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``df`` is a scalar.  Otherwise,
        ``np.array(df).size`` samples are drawn.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized chi-square distribution.

    Raises
    ------
    ValueError
        When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)
        is given.

    Examples
    --------
    >>> dpnp.random.chisquare(2,4)
    array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272]) # random

    """

    if not use_origin_backend(df) and dpnp_queue_is_cpu():
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("chisquare", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("chisquare", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `df`
        # add check for df array like, after adding array-like interface for df param
        if df <= 0:
            checker_throw_value_error("chisquare", "df", df, "positive")
        # TODO:
        # float to int, safe
        return dpnp_chisquare(int(df), size)

    return call_origin(numpy.random.chisquare, df, size)


def choice(a, size=None, replace=True, p=None):
    """
    Generates a random sample from a given 1-D array.

    For full documentation refer to :obj:`numpy.random.choice`.

    Notes
    -----
    The function uses `numpy.random.choice` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.choice, a, size, replace, p)


def dirichlet(alpha, size=None):
    """Dirichlet distribution.

    Draw samples from the Dirichlet distribution.

    For full documentation refer to :obj:`numpy.random.dirichlet`.

    Notes
    -----
    The function uses `numpy.random.dirichlet` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.dirichlet, alpha, size)


def exponential(scale=1.0, size=None):
    """Exponential distribution.

    Draw samples from an exponential distribution.

    Its probability density function is

    .. math:: f(x; \\frac{1}{\\beta}) = \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta}),

    for ``x > 0`` and 0 elsewhere. :math:`\\beta` is the scale parameter,
    which is the inverse of the rate parameter :math:`\\lambda = 1/\\beta`.
    The rate parameter is an alternative, widely used parameterization
    of the exponential distribution [3]_.

    The exponential distribution is a continuous analogue of the
    geometric distribution.  It describes many common situations, such as
    the size of raindrops measured over many rainstorms [1]_, or the time
    between page requests to Wikipedia [2]_.

    Parameters
    ----------
    scale : float
        The scale parameter, :math:`\\beta = 1/\\lambda`. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``scale`` is a scalar.  Otherwise,
        ``np.array(scale).size`` samples are drawn.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized exponential distribution.

    References
    ----------
    .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
           Random Signal Principles", 4th ed, 2001, p. 57.
    .. [2] Wikipedia, "Poisson process",
           https://en.wikipedia.org/wiki/Poisson_process
    .. [3] Wikipedia, "Exponential distribution",
           https://en.wikipedia.org/wiki/Exponential_distribution

    """

    if not use_origin_backend(scale):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("exponential", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("exponential", "type(size)", type(size), int)

        if scale < 0:
            checker_throw_value_error("exponential", "scale", scale, "non-negative")

        return dpnp_exponential(scale, size)

    return call_origin(numpy.random.exponential, scale, size)


def f(dfnum, dfden, size=None):
    """F distribution.

    Draw samples from an F distribution.

    For full documentation refer to :obj:`numpy.random.f`.

    Notes
    -----
    The function uses `numpy.random.f` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.f, dfnum, dfden, size)


def gamma(shape, scale=1.0, size=None):
    """Gamma distribution.

    Draw samples from a Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    `shape` (sometimes designated "k") and `scale` (sometimes designated
    "theta"), where both parameters are > 0.

    Parameters
    ----------
    shape : float or array_like of floats
        The shape of the gamma distribution. Must be non-negative.
    scale : float or array_like of floats, optional
        The scale of the gamma distribution. Must be non-negative.
        Default is equal to 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``shape`` and ``scale`` are both scalars.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized gamma distribution.

    Notes
    -----
    The probability density for the Gamma distribution is

    .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

    where :math:`k` is the shape and :math:`\\theta` the scale,
    and :math:`\\Gamma` is the Gamma function.

    The Gamma distribution is often used to model the times to failure of
    electronic components, and arises naturally in processes for which the
    waiting times between Poisson distributed events are relevant.

    References
    ----------
    .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
           Wolfram Web Resource.
           http://mathworld.wolfram.com/GammaDistribution.html
    .. [2] Wikipedia, "Gamma distribution",
           https://en.wikipedia.org/wiki/Gamma_distribution

    """

    # TODO:
    # array_like of floats for `scale` and `shape`
    if not use_origin_backend(scale) and dpnp_queue_is_cpu():
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("gamma", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("gamma", "type(size)", type(size), int)

        if scale < 0:
            checker_throw_value_error("gamma", "scale", scale, "non-negative")
        if shape < 0:
            checker_throw_value_error("gamma", "shape", shape, "non-negative")

        return dpnp_gamma(shape, scale, size)

    return call_origin(numpy.random.gamma, shape, scale, size)


def geometric(p, size=None):
    """Geometric distribution.

    Draw samples from the geometric distribution.

    Bernoulli trials are experiments with one of two outcomes:
    success or failure (an example of such an experiment is flipping
    a coin).  The geometric distribution models the number of trials
    that must be run in order to achieve success.  It is therefore
    supported on the positive integers, ``k = 1, 2, ...``.

    The probability mass function of the geometric distribution is

    .. math:: f(k) = (1 - p)^{k - 1} p

    where `p` is the probability of success of an individual trial.

    Parameters
    ----------
    p : float
        The probability of success of an individual trial.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``p`` is a scalar.

    Returns
    -------
    out : dparray, int32
        Drawn samples from the parameterized geometric distribution.

    Examples
    --------
    Draw ten thousand values from the geometric distribution,
    with the probability of an individual success equal to 0.35:
    >>> z = dpnp.random.geometric(p=0.35, size=10000)

    """

    if not use_origin_backend(p):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("geometric", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("geometric", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `p` param
        if p > 1 or p <= 0:
            checker_throw_value_error("geometric", "p", p, "in (0, 1]")

        return dpnp_geometric(p, size)

    return call_origin(numpy.random.geometric, p, size)


def gumbel(loc=0.0, scale=1.0, size=None):
    """Gumbel distribution.

    Draw samples from a Gumbel distribution.

    Draw samples from a Gumbel distribution with specified location and
    scale.

    Parameters
    ----------
    loc : float, optional
        The location of the mode of the distribution. Default is 0.
    scale : float, optional
        The scale parameter of the distribution. Default is 1. Must be non-
        negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized Gumbel distribution.

    Examples
    --------
    Draw samples from the distribution:
    >>> mu, beta = 0, 0.1 # location and scale
    >>> s = dpnp.random.gumbel(mu, beta, 1000)

    """

    if not use_origin_backend(loc):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("gumbel", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("gumbel", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `loc` and `scale` params
        if scale < 0:
            checker_throw_value_error("gumbel", "scale", scale, "non-negative")

        return dpnp_gumbel(loc, scale, size)

    return call_origin(numpy.random.gumbel, loc, scale, size)


def hypergeometric(ngood, nbad, nsample, size=None):
    """Hypergeometric distribution.

    Draw samples from a Hypergeometric distribution.

    Samples are drawn from a hypergeometric distribution with specified
    parameters, `ngood` (ways to make a good selection), `nbad` (ways to make
    a bad selection), and `nsample` (number of items sampled, which is less
    than or equal to the sum ``ngood + nbad``).

    Parameters
    ----------
    ngood : int
        Number of ways to make a good selection.  Must be nonnegative.
    nbad : int
        Number of ways to make a bad selection.  Must be nonnegative.
    nsample : int
        Number of items sampled.  Must be at least 1 and at most
        ``ngood + nbad``.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if `ngood`, `nbad`, and `nsample`
        are all scalars.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized hypergeometric distribution. Each
        sample is the number of good items within a randomly selected subset of
        size `nsample` taken from a set of `ngood` good items and `nbad` bad items.

    Notes
    -----
    The probability density for the Hypergeometric distribution is

    .. math:: P(x) = \\frac{\\binom{g}{x}\\binom{b}{n-x}}{\\binom{g+b}{n}},

    where :math:`0 \\le x \\le n` and :math:`n-b \\le x \\le g`

    for P(x) the probability of ``x`` good results in the drawn sample,
    g = `ngood`, b = `nbad`, and n = `nsample`.

    Consider an urn with black and white marbles in it, `ngood` of them
    are black and `nbad` are white. If you draw `nsample` balls without
    replacement, then the hypergeometric distribution describes the
    distribution of black balls in the drawn sample.

    Note that this distribution is very similar to the binomial
    distribution, except that in this case, samples are drawn without
    replacement, whereas in the Binomial case samples are drawn with
    replacement (or the sample space is infinite). As the sample space
    becomes large, this distribution approaches the binomial.

    References
    ----------
    .. [1] Lentner, Marvin, "Elementary Applied Statistics", Bogden
           and Quigley, 1972.
    .. [2] Weisstein, Eric W. "Hypergeometric Distribution." From
           MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/HypergeometricDistribution.html
    .. [3] Wikipedia, "Hypergeometric distribution",
           https://en.wikipedia.org/wiki/Hypergeometric_distribution

    Examples
    --------
    Draw samples from the distribution:
    >>> ngood, nbad, nsamp = 100, 2, 10
    # number of good, number of bad, and number of samples
    >>> s = dpnp.random.hypergeometric(ngood, nbad, nsamp, 1000)

    """

    if not use_origin_backend(ngood) and dpnp_queue_is_cpu():
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("hypergeometric", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("hypergeometric", "type(size)", type(size), int)

        # TODO:
        # array_like of ints for `ngood`, `nbad`, `nsample` param
        if ngood < 0:
            checker_throw_value_error("hypergeometric", "ngood", ngood, "non-negative")
        if nbad < 0:
            checker_throw_value_error("hypergeometric", "nbad", nbad, "non-negative")
        if nsample < 0:
            checker_throw_value_error("hypergeometric", "nsample", nsample, "non-negative")
        if ngood + nbad < nsample:
            checker_throw_value_error("hypergeometric", "nsample", nsample, "ngood + nbad >= nsample")
        if nsample < 1:
            checker_throw_value_error("hypergeometric", "nsample", nsample, ">= 1")

        m = int(ngood)
        l = int(ngood) + int(nbad)
        s = int(nsample)

        return dpnp_hypergeometric(l, s, m, size)

    return call_origin(numpy.random.hypergeometric, ngood, nbad, nsample, size)


def laplace(loc=0.0, scale=1.0, size=None):
    """Laplace distribution.

    Draw samples from the Laplace or double exponential distribution with
    specified location (or mean) and scale (decay).

    The Laplace distribution is similar to the Gaussian/normal distribution,
    but is sharper at the peak and has fatter tails. It represents the
    difference between two independent, identically distributed exponential
    random variables.

    Parameters
    ----------
    loc : float, optional
        The position, :math:`\\mu`, of the distribution peak. Default is 0.
    scale : float, optional
        :math:`\\lambda`, the exponential decay. Default is 1. Must be non-
        negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized Laplace distribution.

    Examples
    --------
    >>> loc, scale = 0., 1.
    >>> s = dpnp.random.laplace(loc, scale, 1000)

    """

    if not use_origin_backend(loc):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("laplace", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("laplace", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `loc` and `scale` params
        if scale < 0:
            checker_throw_value_error("laplace", "scale", scale, "non-negative")

        return dpnp_laplace(loc, scale, size)

    return call_origin(numpy.random.laplace, loc, scale, size)


def logistic(loc=0.0, scale=1.0, size=None):
    """Logistic distribution.

    Draw samples from a logistic distribution.

    For full documentation refer to :obj:`numpy.random.logistic`.

    Notes
    -----
    The function uses `numpy.random.logistic` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.logistic, loc, scale, size)


def lognormal(mean=0.0, sigma=1.0, size=None):
    """Lognormal distribution.

    Draw samples from a log-normal distribution.

    Draw samples from a log-normal distribution with specified mean,
    standard deviation, and array shape.  Note that the mean and standard
    deviation are not the values for the distribution itself, but of the
    underlying normal distribution it is derived from.

    Parameters
    ----------
    mean : float, optional
        Mean value of the underlying normal distribution. Default is 0.
    sigma : float, optional
        Standard deviation of the underlying normal distribution. Must be
        non-negative. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``mean`` and ``sigma`` are both scalars.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized log-normal distribution.

    Notes
    -----
    A variable `x` has a log-normal distribution if `log(x)` is normally
    distributed.  The probability density function for the log-normal
    distribution is:

    .. math:: p(x) = \\frac{1}{\\sigma x \\sqrt{2\\pi}}
                     e^{(-\\frac{(ln(x)-\\mu)^2}{2\\sigma^2})}

    where :math:`\\mu` is the mean and :math:`\\sigma` is the standard
    deviation of the normally distributed logarithm of the variable.
    A log-normal distribution results if a random variable is the *product*
    of a large number of independent, identically-distributed variables in
    the same way that a normal distribution results if the variable is the
    *sum* of a large number of independent, identically-distributed
    variables.

    References
    ----------
    .. [1] Limpert, E., Stahel, W. A., and Abbt, M., "Log-normal
           Distributions across the Sciences: Keys and Clues,"
           BioScience, Vol. 51, No. 5, May, 2001.
           https://stat.ethz.ch/~stahel/lognormal/bioscience.pdf
    .. [2] Reiss, R.D. and Thomas, M., "Statistical Analysis of Extreme
           Values," Basel: Birkhauser Verlag, 2001, pp. 31-32.

    Examples
    --------
    Draw samples from the distribution:
    >>> mu, sigma = 3., 1. # mean and standard deviation
    >>> s = dpnp.random.lognormal(mu, sigma, 1000)

    """

    if not use_origin_backend(mean):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("lognormal", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("lognormal", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `mean` and `sigma` params
        if sigma < 0:
            checker_throw_value_error("lognormal", "sigma", sigma, "non-negative")

        return dpnp_lognormal(mean, sigma, size)

    return call_origin(numpy.random.lognormal, mean, sigma, size)


def logseries(p, size=None):
    """Logseries distribution.

    Draw samples from a logarithmic series distribution.

    For full documentation refer to :obj:`numpy.random.logseries`.

    Notes
    -----
    The function uses `numpy.random.logseries` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.logseries, p, size)


def multinomial(n, pvals, size=None):
    """Multinomial distribution.

    Draw samples from a multinomial distribution.

    The multinomial distribution is a multivariate generalization of the
    binomial distribution.  Take an experiment with one of ``p``
    possible outcomes.  An example of such an experiment is throwing a dice,
    where the outcome can be 1 through 6.  Each sample drawn from the
    distribution represents `n` such experiments.  Its values,
    ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the
    outcome was ``i``.

    Parameters
    ----------
    n : int
        Number of experiments.
    pvals : sequence of floats, length p
        Probabilities of each of the ``p`` different outcomes.  These
        must sum to 1 (however, the last element is always assumed to
        account for the remaining probability, as long as
        ``sum(pvals[:-1]) <= 1)``.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : dparray, int32
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.
        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.

    Examples
    --------
    Throw a dice 20 times:
    >>> dpnp.random.multinomial(20, [1/6.]*6, size=1)
    array([[4, 1, 7, 5, 2, 1]]) # random

    """

    if not use_origin_backend(n) and dpnp_queue_is_cpu():
        if size is None:
            size = (1,)
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("multinomial", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("multinomial", "type(size)", type(size), int)
        else:
            size = (size,)
        pvals_sum = sum(pvals)

        if n < 0:
            checker_throw_value_error("multinomial", "n", n, "non-negative")
        elif n > numpy.iinfo(numpy.int32).max:
            checker_throw_value_error("multinomial", "n", n, "n <= int32 max (2147483647)")
        elif pvals_sum > 1.0:
            checker_throw_value_error("multinomial", "sum(pvals)", pvals_sum, "sum(pvals) <= 1.0")
        elif pvals_sum < 0.0:
            checker_throw_value_error("multinomial", "sum(pvals)", pvals_sum, "sum(pvals) >= 0.0")
        else:
            return dpnp_multinomial(int(n), pvals, size)

    return call_origin(numpy.random.multinomial, n, pvals, size)


def multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8):
    """Multivariate normal distributions.

    Draw random samples from a multivariate normal distribution.

    The multivariate normal, multinormal or Gaussian distribution is a
    generalization of the one-dimensional normal distribution to higher
    dimensions.  Such a distribution is specified by its mean and
    covariance matrix.  These parameters are analogous to the mean
    (average or "center") and variance (standard deviation, or "width,"
    squared) of the one-dimensional normal distribution.

    Parameters
    ----------
    mean : 1-D array_like, of length N
        Mean of the N-dimensional distribution.
    cov : 2-D array_like, of shape (N, N)
        Covariance matrix of the distribution. It must be symmetric and
        positive-semidefinite for proper sampling.
    size : int or tuple of ints, optional
        Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
        generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
        each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
        If no shape is specified, a single (`N`-D) sample is returned.
    check_valid : { 'warn', 'raise', 'ignore' }, optional
        Behavior when the covariance matrix is not positive semidefinite.
        Currently ignored and not used.
    tol : float, optional
        Tolerance when checking the singular values in covariance matrix.
        cov is cast to double before the check. Currently ignored and not used.

    Returns
    -------
    out : dparray
        The drawn samples, of shape *size*, if that was provided.  If not,
        the shape is ``(N,)``.
        In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
        value drawn from the distribution.

    Notes
    -----
    The mean is a coordinate in N-dimensional space, which represents the
    location where samples are most likely to be generated.  This is
    analogous to the peak of the bell curve for the one-dimensional or
    univariate normal distribution.

    Covariance indicates the level to which two variables vary together.
    From the multivariate normal distribution, we draw N-dimensional
    samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
    element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
    The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
    "spread").

    Instead of specifying the full covariance matrix, popular
    approximations include:

      - Spherical covariance (`cov` is a multiple of the identity matrix)
      - Diagonal covariance (`cov` has non-negative elements, and only on
        the diagonal)

    This geometrical property can be seen in two dimensions by plotting
    generated data-points:

    >>> mean = [0, 0]
    >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

    Diagonal covariance means that points are oriented along x or y-axis:

    >>> import matplotlib.pyplot as plt
    >>> x, y = np.random.multivariate_normal(mean, cov, 5000).T
    >>> plt.plot(x, y, 'x')
    >>> plt.axis('equal')
    >>> plt.show()

    Note that the covariance matrix must be positive semidefinite (a.k.a.
    nonnegative-definite). Otherwise, the behavior of this method is
    undefined and backwards compatibility is not guaranteed.

    References
    ----------
    .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
           Processes," 3rd ed., New York: McGraw-Hill, 1991.
    .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
           Classification," 2nd ed., New York: Wiley, 2001.

    Examples
    --------
    >>> mean = (1, 2)
    >>> cov = [[1, 0], [0, 1]]
    >>> x = dpnp.random.multivariate_normal(mean, cov, (3, 3))
    >>> x.shape
    (3, 3, 2)

    """

    if not use_origin_backend(mean) and dpnp_queue_is_cpu():
        mean = numpy.array(mean, dtype=numpy.float64, order='C')
        cov = numpy.array(cov, dtype=numpy.float64, order='C')
        if size is None:
            shape = []
        elif isinstance(size, (int, numpy.integer)):
            shape = [size]
        else:
            shape = size
        if len(mean.shape) != 1:
            raise ValueError("mean must be 1 dimensional")
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
            raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")
        final_shape = list(shape[:])
        final_shape.append(mean.shape[0])

        return dpnp_multivariate_normal(mean, cov, final_shape)

    return call_origin(numpy.random.multivariate_normal, mean, cov, size, check_valid, tol)


def negative_binomial(n, p, size=None):
    """Negative binomial distribution.

    Draw samples from a negative binomial distribution.

    Samples are drawn from a negative binomial distribution with specified
    parameters, `n` successes and `p` probability of success where `n`
    is > 0 and `p` is in the interval [0, 1].

    Parameters
    ----------
    n : float
        Parameter of the distribution, > 0.
    p : float
        Parameter of the distribution, >= 0 and <=1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``n`` and ``p`` are both scalars.

    Returns
    -------
    out : dparray, int32
        Drawn samples from the parameterized negative binomial distribution,
        where each sample is equal to N, the number of failures that
        occurred before a total of n successes was reached.

    Notes
    -----
    The probability mass function of the negative binomial distribution is

    .. math:: P(N;n,p) = \\frac{\\Gamma(N+n)}{N!\\Gamma(n)}p^{n}(1-p)^{N},

    where :math:`n` is the number of successes, :math:`p` is the
    probability of success, :math:`N+n` is the number of trials, and
    :math:`\\Gamma` is the gamma function. When :math:`n` is an integer,
    :math:`\\frac{\\Gamma(N+n)}{N!\\Gamma(n)} = \\binom{N+n-1}{N}`, which is
    the more common form of this term in the the pmf. The negative
    binomial distribution gives the probability of N failures given n
    successes, with a success on the last trial.

    If one throws a die repeatedly until the third time a "1" appears,
    then the probability distribution of the number of non-"1"s that
    appear before the third "1" is a negative binomial distribution.

    References
    ----------
    .. [1] Weisstein, Eric W. "Negative Binomial Distribution." From
           MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/NegativeBinomialDistribution.html
    .. [2] Wikipedia, "Negative binomial distribution",
           https://en.wikipedia.org/wiki/Negative_binomial_distribution

    Examples
    --------
    Draw samples from the distribution:
    A real world example. A company drills wild-cat oil
    exploration wells, each with an estimated probability of
    success of 0.1.  What is the probability of having one success
    for each successive well, that is what is the probability of a
    single success after drilling 5 wells, after 6 wells, etc.?

    >>> s = dpnp.random.negative_binomial(1, 0.1, 100000)
    >>> for i in range(1, 11): # doctest: +SKIP
    ...    probability = sum(s<i) / 100000.
    ...    print(i, "wells drilled, probability of one success =", probability)

    """

    if not use_origin_backend(n) and dpnp_queue_is_cpu():
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("negative_binomial", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("negative_binomial", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `p` and `n` params
        if p > 1 or p < 0:
            checker_throw_value_error("negative_binomial", "p", p, "in [0, 1]")
        if n <= 0:
            checker_throw_value_error("negative_binomial", "n", n, "non-negative")

        return dpnp_negative_binomial(n, p, size)

    return call_origin(numpy.random.negative_binomial, n, p, size)


def normal(loc=0.0, scale=1.0, size=None):
    """Normal distribution.

    Draw random samples from a normal (Gaussian) distribution.

    The probability density function of the normal distribution, first
    derived by De Moivre and 200 years later by both Gauss and Laplace
    independently [2]_, is often called the bell curve because of
    its characteristic shape (see the example below).

    The normal distributions occurs often in nature.  For example, it
    describes the commonly occurring distribution of samples influenced
    by a large number of tiny, random disturbances, each with its own
    unique distribution [2]_.

    Parameters
    ----------
    loc : float
        Mean ("centre") of the distribution.
    scale : float
        Standard deviation (spread or "width") of the distribution. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``loc`` and ``scale`` are both scalars.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized normal distribution.

    Notes
    -----
    The probability density for the Gaussian distribution is

    .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                     e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

    where :math:`\\mu` is the mean and :math:`\\sigma` the standard
    deviation. The square of the standard deviation, :math:`\\sigma^2`,
    is called the variance.

    The function has its peak at the mean, and its "spread" increases with
    the standard deviation (the function reaches 0.607 times its maximum at
    :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that
    normal is more likely to return samples lying close to the mean, rather
    than those far away.

    References
    ----------
    .. [1] Wikipedia, "Normal distribution",
           https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
           Random Variables and Random Signal Principles", 4th ed., 2001,
           pp. 51, 51, 125.

    Examples
    --------
    Draw samples from the distribution:
    >>> mu, sigma = 0, 0.1 # mean and standard deviation
    >>> s = dpnp.random.normal(mu, sigma, 1000)

    """

    if not use_origin_backend(loc):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("normal", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("normal", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `loc` and `scale` params
        if scale < 0:
            checker_throw_value_error("normal", "scale", scale, "non-negative")

        return dpnp_normal(loc, scale, size)

    return call_origin(numpy.random.normal, loc, scale, size)


def noncentral_chisquare(df, nonc, size=None):
    """Noncentral chi-square distribution.

    Draw samples from a noncentral chi-square distribution.

    For full documentation refer to :obj:`numpy.random.noncentral_chisquare`.

    Notes
    -----
    The function uses `numpy.random.noncentral_chisquare` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.noncentral_chisquare, df, nonc, size)


def noncentral_f(dfnum, dfden, nonc, size=None):
    """Noncentral F distribution.

    Draw samples from the noncentral F distribution.

    For full documentation refer to :obj:`numpy.random.noncentral_f`.

    Notes
    -----
    The function uses `numpy.random.noncentral_f` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.noncentral_f, dfnum, dfden, nonc, size)


def pareto(a, size=None):
    """Pareto II or Lomax distribution.

    Draw samples from a Pareto II or Lomax distribution with specified shape.

    For full documentation refer to :obj:`numpy.random.pareto`.

    Notes
    -----
    The function uses `numpy.random.pareto` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.pareto, a, size)


def permutation(x):
    """
    Randomly permute a sequence, or return a permuted range.

    For full documentation refer to :obj:`numpy.random.permutation`.

    Notes
    -----
    The function uses `numpy.random.permutation` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.permutation, x)


def poisson(lam=1.0, size=None):
    """Poisson distribution.

    Draw samples from a Poisson distribution.

    The Poisson distribution is the limit of the binomial distribution
    for large N.

    Parameters
    ----------
    lam : float
        Expectation of interval, must be >= 0. A sequence of expectation
        intervals must be broadcastable over the requested size.
    size : int, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``lam`` is a scalar.

    Returns
    -------
    out : dparray, int32
        Drawn samples from the parameterized Poisson distribution.

    Notes
    -----
    The Poisson distribution

    .. math:: f(k; \\lambda)=\\frac{\\lambda^k e^{-\\lambda}}{k!}

    For events with an expected separation :math:`\\lambda` the Poisson
    distribution :math:`f(k; \\lambda)` describes the probability of
    :math:`k` events occurring within the observed
    interval :math:`\\lambda`.

    References
    ----------
    .. [1] Weisstein, Eric W. "Poisson Distribution."
           From MathWorld--A Wolfram Web Resource.
           http://mathworld.wolfram.com/PoissonDistribution.html
    .. [2] Wikipedia, "Poisson distribution",
           https://en.wikipedia.org/wiki/Poisson_distribution

    Examples
    --------
    Draw samples from the distribution:
    >>> import numpy as np
    >>> s = dpnp.random.poisson(5, 10000)

    """

    if not use_origin_backend(lam):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("poisson", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("poisson", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `lam` param
        if lam < 0:
            checker_throw_value_error("poisson", "lam", lam, "non-negative")

        return dpnp_poisson(lam, size)

    return call_origin(numpy.random.poisson, lam, size)


def power(a, size=None):
    """Power distribution.

    Draws samples in [0, 1] from a power distribution with positive exponent
    a - 1.

    For full documentation refer to :obj:`numpy.random.power`.

    Notes
    -----
    The function uses `numpy.random.power` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.power, a, size)


def rand(d0, *dn):
    """
    Create an array of the given shape and populate it
    with random samples from a uniform distribution over [0, 1).

    Parameters
    ----------
    d0, d1, …, dn : The dimensions of the returned array, must be non-negative.

    Returns
    -------
    out : Random values.

    See Also
    --------
    :obj:`dpnp.random.random`

    """

    if not use_origin_backend(d0):
        dims = tuple([d0, *dn])

        for dim in dims:
            if not isinstance(dim, int):
                checker_throw_value_error("rand", "type(dim)", type(dim), int)
        return dpnp_random(dims)

    return call_origin(numpy.random.rand, d0, *dn)


def randint(low, high=None, size=None, dtype=int):
    """
    randint(low, high=None, size=None, dtype=int)

    Return random integers from `low` (inclusive) to `high` (exclusive).
    Return random integers from the "discrete uniform" distribution of
    the specified dtype in the "half-open" interval [`low`, `high`). If
    `high` is None (the default), then results are from [0, `low`).

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int, optional
        If provided, one above the largest (signed) integer to be drawn
        from the distribution.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result. Byteorder must be native.
        The default value is int.

    Returns
    -------
    out : array of random ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    :obj:`dpnp.random.random_integers` : similar to `randint`, only for the closed
                                         interval [`low`, `high`], and 1 is the
                                         lowest value if `high` is omitted.

    """

    if not use_origin_backend(low):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("randint", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("randint", "type(size)", type(size), int)

        if high is None:
            high = low
            low = 0

        low = int(low)
        high = int(high)

        if (low >= high):
            checker_throw_value_error("randint", "low", low, high)

        _dtype = numpy.dtype(dtype)

        # TODO:
        # supported only int32
        # or just raise error when dtype != numpy.int32
        if _dtype == numpy.int32 or _dtype == numpy.int64:
            _dtype = numpy.int32
        else:
            raise TypeError('Unsupported dtype %r for randint' % dtype)
        return dpnp_uniform(low, high, size, _dtype)

    return call_origin(numpy.random.randint, low, high, size, dtype)


def randn(d0, *dn):
    """
    If positive int_like arguments are provided, randn generates an array of shape (d0, d1, ..., dn),
    filled with random floats sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1.

    Parameters
    ----------
    d0, d1, …, dn : The dimensions of the returned array, must be non-negative.

    Returns
    -------
    out : (d0, d1, ..., dn)-shaped array of floating-point samples from the standard normal distribution.

    See Also
    --------
    :obj:`dpnp.random.standard_normal`
    :obj:`dpnp.random.normal`

    """

    if not use_origin_backend(d0):
        dims = tuple([d0, *dn])

        for dim in dims:
            if not isinstance(dim, int):
                checker_throw_value_error("randn", "type(dim)", type(dim), int)
        return dpnp_randn(dims)

    return call_origin(numpy.random.randn, d0, *dn)


def random(size):
    """
    Return random floats in the half-open interval [0.0, 1.0).
    Alias for random_sample.

    Parameters
    ----------
    size : Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.

    Returns
    -------
    out : Array of random floats of shape size.

    See Also
    --------
    :obj:`dpnp.random.random`

    """

    if not use_origin_backend(size):
        for dim in size:
            if not isinstance(dim, int):
                checker_throw_value_error("random", "type(dim)", type(dim), int)
        return dpnp_random(size)

    return call_origin(numpy.random.random, size)


def random_integers(low, high=None, size=None):
    """
    random_integers(low, high=None, size=None)

    Random integers between `low` and `high`, inclusive.
    Return random integers from the "discrete uniform" distribution in
    the closed interval [`low`, `high`].  If `high` is
    None (the default), then results are from [1, `low`].

    Parameters
    ----------
    low : int
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int, optional
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : array of random ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.

    See Also
    --------
    :obj:`dpnp.random.randint`

    """

    if not use_origin_backend(low):
        if high is None:
            high = low
            low = 1
        return randint(low, int(high) + 1, size=size)

    return call_origin(numpy.random.random_integers, low, high, size)


def random_sample(size):
    """
    Return random floats in the half-open interval [0.0, 1.0).

    Parameters
    ----------
    size : Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.

    Returns
    -------
    out : Array of random floats of shape size.

    See Also
    --------
    :obj:`dpnp.random.random`

    """

    if not use_origin_backend(size):
        for dim in size:
            if not isinstance(dim, int):
                checker_throw_value_error("random_sample", "type(dim)", type(dim), int)
        return dpnp_random(size)

    return call_origin(numpy.random.random_sample, size)


def ranf(size):
    """
    Return random floats in the half-open interval [0.0, 1.0).
    This is an alias of random_sample.

    Parameters
    ----------
    size : Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.

    Returns
    -------
    out : Array of random floats of shape size.

    See Also
    --------
    :obj:`dpnp.random.random`

    """

    if not use_origin_backend(size):
        for dim in size:
            if not isinstance(dim, int):
                checker_throw_value_error("ranf", "type(dim)", type(dim), int)
        return dpnp_random(size)

    return call_origin(numpy.random.ranf, size)


def rayleigh(scale=1.0, size=None):
    """Rayleigh distribution.

    Draw samples from a Rayleigh distribution.

    The :math:`\\chi` and Weibull distributions are generalizations of the
    Rayleigh.

    Parameters
    ----------
    scale : float, optional
        Scale, also equals the mode. Must be non-negative. Default is 1.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``scale`` is a scalar.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized Rayleigh distribution.

    """

    if not use_origin_backend(scale):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("rayleigh", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("rayleigh", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `scale` params
        if scale < 0:
            checker_throw_value_error("rayleigh", "scale", scale, "non-negative")

        return dpnp_rayleigh(scale, size)

    return call_origin(numpy.random.rayleigh, scale, size)


def sample(size):
    """
    Return random floats in the half-open interval [0.0, 1.0).
    This is an alias of random_sample.

    Parameters
    ----------
    size : Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.

    Returns
    -------
    out : Array of random floats of shape size.

    See Also
    --------
    :obj:`dpnp.random.random`

    """

    if not use_origin_backend(size):
        for dim in size:
            if not isinstance(dim, int):
                checker_throw_value_error("sample", "type(dim)", type(dim), int)
        return dpnp_random(size)

    return call_origin(numpy.random.sample, size)


def shuffle(x):
    """
    Modify a sequence in-place by shuffling its contents.

    For full documentation refer to :obj:`numpy.random.shuffle`.

    Notes
    -----
    The function uses `numpy.random.shuffle` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.shuffle, x)


def seed(seed=None):
    """
    Reseed a legacy philox4x32x10 random number generator engine.

    Parameters
    ----------
    seed : {None, int}, optional

    """

    if not use_origin_backend(seed):
        if not isinstance(seed, int):
            pass
        elif seed < 0:
            pass
        else:
            if seed is None:
                seed = 1
            dpnp_srand(seed)

    return call_origin(numpy.random.seed, seed)


def standard_cauchy(size=None):
    """Standard cauchy distribution.

    Draw samples from a standard Cauchy distribution with mode = 0.

    Also known as the Lorentz distribution.

    Parameters
    ----------
    size : int, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.

    Returns
    -------
    samples : dparray
        The drawn samples.

    Examples
    --------
    Draw samples and plot the distribution:
    >>> import matplotlib.pyplot as plt
    >>> s = dpnp.random.standard_cauchy(1000000)
    >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
    >>> plt.hist(s, bins=100)
    >>> plt.show()

    """

    if not use_origin_backend(size):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("standard_cauchy", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("standard_cauchy", "type(size)", type(size), int)

        return dpnp_standard_cauchy(size)

    return call_origin(numpy.random.standard_cauchy, size)


def standard_exponential(size=None):
    """Standard exponential distribution.

    Draw samples from the standard exponential distribution.

    `standard_exponential` is identical to the exponential distribution
    with a scale parameter of 1.

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : dparray
        Drawn samples.

    Examples
    --------
    Output a 3x8000 array:
    >>> n = dpnp.random.standard_exponential((3, 8000))

    """

    if not use_origin_backend(size):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("standard_exponential", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("standard_exponential", "type(size)", type(size), int)

        return dpnp_standard_exponential(size)

    return call_origin(numpy.random.standard_exponential, size)


def standard_gamma(shape, size=None):
    """Standard gamma distribution.

    Draw samples from a standard Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    shape (sometimes designated "k") and scale=1.

    Parameters
    ----------
    shape : float
        Parameter, must be non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``shape`` is a scalar.

    Returns
    -------
    out : dparray
        Drawn samples from the parameterized standard gamma distribution.

    Notes
    -----
    The probability density for the Gamma distribution is

    .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

    where :math:`k` is the shape and :math:`\\theta` the scale,
    and :math:`\\Gamma` is the Gamma function.

    The Gamma distribution is often used to model the times to failure of
    electronic components, and arises naturally in processes for which the
    waiting times between Poisson distributed events are relevant.

    References
    ----------
    .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
           Wolfram Web Resource.
           http://mathworld.wolfram.com/GammaDistribution.html
    .. [2] Wikipedia, "Gamma distribution",
           https://en.wikipedia.org/wiki/Gamma_distribution

    Examples
    --------
    Draw samples from the distribution:
    >>> shape = 2. # mean and width
    >>> s = dpnp.random.standard_gamma(shape, 1000000)

    """

    # TODO:
    # array_like of floats for and `shape`
    if not use_origin_backend(shape) and dpnp_queue_is_cpu():
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("standard_gamma", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("standard_gamma", "type(size)", type(size), int)

        if shape < 0:
            checker_throw_value_error("standard_gamma", "shape", shape, "non-negative")

        return dpnp_standard_gamma(shape, size)

    return call_origin(numpy.random.standard_gamma, shape, size)


def standard_normal(size=None):
    """Standard normal distribution.

    Draw samples from a standard Normal distribution (mean=0, stdev=1).

    Parameters
    ----------
    size : int, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : float or ndarray
        A floating-point array of shape ``size`` of drawn samples, or a
        single sample if ``size`` was not specified.

    """

    if not use_origin_backend(size):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("standard_normal", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("standard_normal", "type(size)", type(size), int)

        return dpnp_standard_normal(size)

    return call_origin(numpy.random.standard_normal, size)


def standard_t(df, size=None):
    """Power distribution.

    Draw samples from a standard Student’s t distribution with df degrees
    of freedom.

    For full documentation refer to :obj:`numpy.random.standard_t`.

    Notes
    -----
    The function uses `numpy.random.standard_t` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.standard_t, df, size)


def triangular(left, mode, right, size=None):
    """Triangular distribution.

    Draw samples from the triangular distribution over the interval
    [left, right].

    For full documentation refer to :obj:`numpy.random.triangular`.

    Notes
    -----
    The function uses `numpy.random.triangular` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.triangular, left, mode, right, size)


def uniform(low=0.0, high=1.0, size=None):
    """
    uniform(low=0.0, high=1.0, size=None)

    Draw samples from a uniform distribution.
    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float
        Upper boundary of the output interval.  All values generated will be
        less than high.  The default value is 1.0.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``low`` and ``high`` are both scalars.

    Returns
    -------
    out : array or scalar
        Drawn samples from the parameterized uniform distribution.

    See Also
    --------
    :obj:`dpnp.random.random` : Floats uniformly distributed over ``[0, 1)``.

    """

    if not use_origin_backend(low):
        if size is None:
            size = 1
        if low == high:
            # TODO:
            # currently dparray.full is not implemented
            # return dpnp.dparray.dparray.full(size, low, dtype=numpy.float64)
            message = "`low` equal to `high`, should return an array, filled with `low` value."
            message += "  Currently not supported. See: numpy.full TODO"
            checker_throw_runtime_error("uniform", message)
        elif low > high:
            low, high = high, low
        return dpnp_uniform(low, high, size, dtype=numpy.float64)

    return call_origin(numpy.random.uniform, low, high, size)


def vonmises(mu, kappa, size=None):
    """von Mises distribution.

    Draw samples from a von Mises distribution.

    For full documentation refer to :obj:`numpy.random.vonmises`.

    Notes
    -----
    The function uses `numpy.random.vonmises` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.vonmises, mu, kappa, size)


def wald(mean, scale, size=None):
    """Wald distribution.

    Draw samples from a Wald, or inverse Gaussian, distribution.

    For full documentation refer to :obj:`numpy.random.wald`.

    Notes
    -----
    The function uses `numpy.random.wald` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.wald, mean, scale, size)


def weibull(a, size=None):
    """

    Draw samples from a Weibull distribution.

    Draw samples from a 1-parameter Weibull distribution with the given
    shape parameter `a`.

    .. math:: X = (-ln(U))^{1/a}

    Here, U is drawn from the uniform distribution over (0,1].
    The more common 2-parameter Weibull, including a scale parameter
    :math:`\\lambda` is just :math:`X = \\lambda(-ln(U))^{1/a}`.

    Parameters
    ----------
    a : float
        Shape parameter of the distribution.  Must be nonnegative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` is a scalar.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized Weibull distribution.

    Examples
    --------
    >>> a = 5. # shape
    >>> s = np.random.weibull(a, 1000)

    """

    if not use_origin_backend(a):
        if size is None:
            size = 1
        elif isinstance(size, tuple):
            for dim in size:
                if not isinstance(dim, int):
                    checker_throw_value_error("weibull", "type(dim)", type(dim), int)
        elif not isinstance(size, int):
            checker_throw_value_error("weibull", "type(size)", type(size), int)

        # TODO:
        # array_like of floats for `a` params
        if a < 0:
            checker_throw_value_error("weibulla", "a", a, "non-negative")

        return dpnp_weibull(a, size)

    return call_origin(numpy.random.weibull, a, size)


def zipf(a, size=None, dtype=int):
    """Zipf distribution.

    Returns an array of samples drawn from the Zipf distribution.

    For full documentation refer to :obj:`numpy.random.zipf`.

    Notes
    -----
    The function uses `numpy.random.zipf` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.zipf, a, size, dtype)
