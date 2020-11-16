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

    .. seealso:: :meth:`numpy.random`

"""


import dpnp
import numpy

from dpnp.dparray import dparray
from dpnp.dpnp_utils import *
from dpnp.random._random import *


__all__ = [
    'beta',
    'binomial',
    'chisquare',
    'exponential',
    'gamma',
    'negative_binomial',
    'rand',
    'ranf',
    'randint',
    'randn',
    'random',
    'random_integers',
    'random_sample',
    'seed',
    'sample',
    'uniform',
    'weibull'
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

    .. note::
        New code should use the ``beta`` method of a ``default_rng()``
        instance instead; please see the :ref:`random-quick-start`.

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
    if not use_origin_backend(a):
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

    if not use_origin_backend(n):
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

    if not use_origin_backend(df):
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

    .. note::
        New code should use the ``exponential`` method of a ``default_rng()``
        instance instead; please see the :ref:`random-quick-start`.

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


def gamma(shape, scale=1.0, size=None):
    """Gamma distribution.

    Draw samples from a Gamma distribution.

    Samples are drawn from a Gamma distribution with specified parameters,
    `shape` (sometimes designated "k") and `scale` (sometimes designated
    "theta"), where both parameters are > 0.

    .. note::
        New code should use the ``gamma`` method of a ``default_rng()``
        instance instead; please see the :ref:`random-quick-start`.

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
    if not use_origin_backend(scale):
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

    if not use_origin_backend(n):
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
    random

    """

    if not use_origin_backend(d0):
        dims = tuple([d0, *dn])

        for dim in dims:
            if not isinstance(dim, int):
                checker_throw_value_error("rand", "type(dim)", type(dim), int)
        return dpnp_random(dims)

    return call_origin(numpy.random.rand, d0, *dn)


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
    random

    """

    if not use_origin_backend(size):
        for dim in size:
            if not isinstance(dim, int):
                checker_throw_value_error("ranf", "type(dim)", type(dim), int)
        return dpnp_random(size)

    return call_origin(numpy.random.ranf, size)


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
    random_integers : similar to `randint`, only for the closed
        interval [`low`, `high`], and 1 is the lowest value if `high` is
        omitted.

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
    standard_normal
    normal

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
    random

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
    randint

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
    random

    """

    if not use_origin_backend(size):
        for dim in size:
            if not isinstance(dim, int):
                checker_throw_value_error("random_sample", "type(dim)", type(dim), int)
        return dpnp_random(size)

    return call_origin(numpy.random.random_sample, size)


def seed(seed=None):
    """
    Reseed a legacy philox4x32x10 random number generator engine

    Parameters
    ----------
    seed : {None, int}, optional

    """
    if not use_origin_backend(seed):
        # TODO:
        # implement seed default value as is in numpy
        if seed is None:
            seed = 1
        elif not isinstance(seed, int):
            checker_throw_value_error("seed", "type(seed)", type(seed), int)
        elif seed < 0:
            checker_throw_value_error("seed", "seed", seed, "non-negative")
        return dpnp_srand(seed)

    return call_origin(numpy.random.seed, seed)


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
    random

    """

    if not use_origin_backend(size):
        for dim in size:
            if not isinstance(dim, int):
                checker_throw_value_error("sample", "type(dim)", type(dim), int)
        return dpnp_random(size)

    return call_origin(numpy.random.sample, size)


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
    random : Floats uniformly distributed over ``[0, 1)``.

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
