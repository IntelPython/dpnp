# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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

import operator

import numpy

import dpnp
from dpnp.dpnp_algo import *
from dpnp.dpnp_utils import *

from .dpnp_algo_random import *
from .dpnp_random_state import RandomState

__all__ = [
    "beta",
    "binomial",
    "bytes",
    "chisquare",
    "choice",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "geometric",
    "gumbel",
    "hypergeometric",
    "laplace",
    "logistic",
    "lognormal",
    "logseries",
    "multinomial",
    "multivariate_normal",
    "negative_binomial",
    "normal",
    "noncentral_chisquare",
    "noncentral_f",
    "pareto",
    "permutation",
    "poisson",
    "power",
    "rand",
    "randint",
    "randn",
    "random",
    "random_integers",
    "random_sample",
    "ranf",
    "rayleigh",
    "sample",
    "shuffle",
    "seed",
    "standard_cauchy",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "triangular",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
    "zipf",
]


def _get_random_state(device=None, sycl_queue=None):
    global _dpnp_random_states

    if not isinstance(_dpnp_random_states, dict):
        _dpnp_random_states = {}
    sycl_queue = dpnp.get_normalized_queue_device(
        device=device, sycl_queue=sycl_queue
    )
    if sycl_queue not in _dpnp_random_states:
        rs = RandomState(device=device, sycl_queue=sycl_queue)
        if sycl_queue == rs.get_sycl_queue():
            _dpnp_random_states[sycl_queue] = rs
        else:
            raise RuntimeError(
                "Normalized SYCL queue {} mismatched with one returned by RandmoState {}".format(
                    sycl_queue, rs.get_sycl_queue()
                )
            )
    return _dpnp_random_states[sycl_queue]


def beta(a, b, size=None):
    """
    Draw samples from a Beta distribution.

    For full documentation refer to :obj:`numpy.random.beta`.

    Limitations
    -----------
    Parameters `a` and `b` are supported as scalar.
    Otherwise, :obj:`numpy.random.beta(a, b, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> a, b = .4, .5  # alpha, beta
    >>> s = dpnp.random.beta(a, b, 1000)

    """

    if not use_origin_backend(a):
        # TODO:
        # array_like of floats for `a`, `b`
        if not dpnp.isscalar(a):
            pass
        elif not dpnp.isscalar(b):
            pass
        elif a <= 0:
            pass
        elif b <= 0:
            pass
        else:
            return dpnp_rng_beta(a, b, size).get_pyobj()

    return call_origin(numpy.random.beta, a, b, size)


def binomial(n, p, size=None):
    """
    Draw samples from a binomial distribution.

    For full documentation refer to :obj:`numpy.random.binomial`.

    Limitations
    -----------
    Output array data type is :obj:`dpnp.int32`.
    Parameters `n` and `p` are supported as scalar.
    Otherwise, :obj:`numpy.random.binomial(n, p, size)` samples are drawn.

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
        # TODO:
        # array_like of floats for `p` param
        if not dpnp.isscalar(n):
            pass
        elif not dpnp.isscalar(p):
            pass
        elif p > 1 or p < 0:
            pass
        elif n < 0:
            pass
        else:
            return dpnp_rng_binomial(int(n), p, size).get_pyobj()

    return call_origin(numpy.random.binomial, n, p, size)


def bytes(length):
    """
    Return random bytes.

    For full documentation refer to :obj:`numpy.random.bytes`.

    Notes
    -----
    The function uses `numpy.random.bytes` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.bytes, length)


def chisquare(df, size=None):
    """
    Draw samples from a chi-square distribution.

    For full documentation refer to :obj:`numpy.random.chisquare`.

    Limitations
    -----------
    Parameter `df` is supported as a scalar.
    Otherwise, :obj:`numpy.random.chisquare(df, size)` samples are drawn.
    Output array data type is default float type.

    Examples
    --------
    >>> dpnp.random.chisquare(2,4)
    array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272]) # random

    """

    if not use_origin_backend(df):
        # TODO:
        # array_like of floats for `df`
        if not dpnp.isscalar(df):
            pass
        elif df <= 0:
            pass
        else:
            # TODO:
            # float to int, safe
            return dpnp_rng_chisquare(int(df), size).get_pyobj()

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
    """
    Draw samples from the Dirichlet distribution.

    For full documentation refer to :obj:`numpy.random.dirichlet`.

    Notes
    -----
    The function uses `numpy.random.dirichlet` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.dirichlet, alpha, size)


def exponential(scale=1.0, size=None):
    """
    Draw samples from an exponential distribution.

    For full documentation refer to :obj:`numpy.random.exponential`.

    Limitations
    -----------
    Parameter `scale` is supported as a scalar.
    Otherwise, :obj:`numpy.random.exponential(scale, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> scale = .5  # alpha
    >>> s = dpnp.random.exponential(scale, 1000)

    """

    if not use_origin_backend(scale):
        # TODO:
        # array_like of floats for `scale`
        if not dpnp.isscalar(scale):
            pass
        elif scale < 0:
            pass
        else:
            return dpnp_rng_exponential(scale, size).get_pyobj()

    return call_origin(numpy.random.exponential, scale, size)


def f(dfnum, dfden, size=None):
    """
    Draw samples from an F distribution.

    For full documentation refer to :obj:`numpy.random.f`.

    Limitations
    -----------
    Parameters `dfnum` and `dfden` are supported as scalar.
    Otherwise, :obj:`numpy.random.f(dfnum, dfden, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    >>> dfnum, dfden = 3., 2.
    >>> s = dpnp.random.f(dfnum, dfden, size)

    """

    if not use_origin_backend(dfnum):
        # TODO:
        # array_like of floats for `dfnum` and `dfden`
        if not dpnp.isscalar(dfnum):
            pass
        elif not dpnp.isscalar(dfden):
            pass
        elif dfnum <= 0:
            pass
        elif dfden <= 0:
            pass
        else:
            return dpnp_rng_f(dfnum, dfden, size).get_pyobj()

    return call_origin(numpy.random.f, dfnum, dfden, size)


def gamma(shape, scale=1.0, size=None):
    """
    Draw samples from a Gamma distribution.

    For full documentation refer to :obj:`numpy.random.gamma`.

    Limitations
    -----------
    Parameters `shape` and `scale` are supported as scalar.
    Otherwise, :obj:`numpy.random.gamma(shape, scale, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> shape, scale = 0, 0.1  # shape and scale
    >>> s = dpnp.random.gamma(shape, scale, 1000)

    """

    if not use_origin_backend(scale):
        # TODO:
        # array_like of floats for `scale` and `shape`
        if not dpnp.isscalar(scale):
            pass
        elif not dpnp.isscalar(shape):
            pass
        elif scale < 0:
            pass
        elif shape < 0:
            pass
        else:
            return dpnp_rng_gamma(shape, scale, size).get_pyobj()

    return call_origin(numpy.random.gamma, shape, scale, size)


def geometric(p, size=None):
    """
    Draw samples from the geometric distribution.

    For full documentation refer to :obj:`numpy.random.geometric`.

    Limitations
    -----------
    Parameter `p` is supported as a scalar.
    Otherwise, :obj:`numpy.random.geometric(p, size)` samples are drawn.
    Output array data type is :obj:`dpnp.int32`.

    Examples
    --------
    Draw ten thousand values from the geometric distribution,
    with the probability of an individual success equal to 0.35:

    >>> z = dpnp.random.geometric(p=0.35, size=10000)

    """

    if not use_origin_backend(p):
        # TODO:
        # array_like of floats for `p` param
        if not dpnp.isscalar(p):
            pass
        elif p > 1 or p <= 0:
            pass
        else:
            return dpnp_rng_geometric(p, size).get_pyobj()

    return call_origin(numpy.random.geometric, p, size)


def gumbel(loc=0.0, scale=1.0, size=None):
    """
    Draw samples from a Gumbel distribution.

    For full documentation refer to :obj:`numpy.random.gumbel`.

    Limitations
    -----------
    Parameters `loc` and `scale` are supported as scalar.
    Otherwise, :obj:`numpy.random.gumbel(loc, scale, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> mu, beta = 0, 0.1 # location and scale
    >>> s = dpnp.random.gumbel(mu, beta, 1000)

    """

    if not use_origin_backend(loc):
        # TODO:
        # array_like of floats for `loc` and `scale` params
        if not dpnp.isscalar(scale):
            pass
        elif not dpnp.isscalar(loc):
            pass
        elif scale < 0:
            pass
        else:
            return dpnp_rng_gumbel(loc, scale, size).get_pyobj()

    return call_origin(numpy.random.gumbel, loc, scale, size)


def hypergeometric(ngood, nbad, nsample, size=None):
    """
    Draw samples from a Hypergeometric distribution.

    For full documentation refer to :obj:`numpy.random.hypergeometric`.

    Limitations
    -----------
    Parameters `ngood`, `nbad` and `nsample` are supported as scalar.
    Otherwise, :obj:`numpy.random.hypergeometric(shape, scale, size)` samples
    are drawn.
    Output array data type is :obj:`dpnp.int32`.

    Examples
    --------
    Draw samples from the distribution:

    >>> ngood, nbad, nsamp = 100, 2, 10
    # number of good, number of bad, and number of samples
    >>> s = dpnp.random.hypergeometric(ngood, nbad, nsamp, 1000)

    """

    if not use_origin_backend(ngood):
        # TODO:
        # array_like of ints for `ngood`, `nbad`, `nsample` param
        if not dpnp.isscalar(ngood):
            pass
        elif not dpnp.isscalar(nbad):
            pass
        elif not dpnp.isscalar(nsample):
            pass
        elif ngood < 0:
            pass
        elif nbad < 0:
            pass
        elif nsample < 0:
            pass
        elif ngood + nbad < nsample:
            pass
        elif nsample < 1:
            pass
        else:
            _m = int(ngood)
            _l = int(ngood) + int(nbad)
            _s = int(nsample)
            return dpnp_rng_hypergeometric(_l, _s, _m, size).get_pyobj()

    return call_origin(numpy.random.hypergeometric, ngood, nbad, nsample, size)


def laplace(loc=0.0, scale=1.0, size=None):
    """
    Draw samples from the Laplace or double exponential distribution with
    specified location (or mean) and scale (decay).

    For full documentation refer to :obj:`numpy.random.laplace`.

    Limitations
    -----------
    Parameters `loc` and `scale` are supported as scalar.
    Otherwise, :obj:`numpy.random.laplace(loc, scale, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    >>> loc, scale = 0., 1.
    >>> s = dpnp.random.laplace(loc, scale, 1000)

    """

    if not use_origin_backend(loc):
        # TODO:
        # array_like of floats for `loc` and `scale`
        if not dpnp.isscalar(loc):
            pass
        elif not dpnp.isscalar(scale):
            pass
        elif scale < 0:
            pass
        else:
            return dpnp_rng_laplace(loc, scale, size).get_pyobj()

    return call_origin(numpy.random.laplace, loc, scale, size)


def logistic(loc=0.0, scale=1.0, size=None):
    """
    Draw samples from a logistic distribution.

    For full documentation refer to :obj:`numpy.random.logistic`.

    Limitations
    -----------
    Parameters `loc` and `scale` are supported as scalar.
    Otherwise, :obj:`numpy.random.logistic(loc, scale, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    >>> loc, scale = 0., 1.
    >>> s = dpnp.random.logistic(loc, scale, 1000)

    """

    if not use_origin_backend(loc):
        # TODO:
        # array_like of floats for `loc` and `scale`
        if not dpnp.isscalar(loc):
            pass
        elif not dpnp.isscalar(scale):
            pass
        elif scale < 0:
            pass
        else:
            result = dpnp_rng_logistic(loc, scale, size).get_pyobj()
            if size is None or size == 1:
                return result[0]
            else:
                return result

    return call_origin(numpy.random.logistic, loc, scale, size)


def lognormal(mean=0.0, sigma=1.0, size=None):
    """
    Draw samples from a log-normal distribution.

    For full documentation refer to :obj:`numpy.random.lognormal`.

    Limitations
    -----------
    Parameters `mean` and `sigma` are supported as scalar.
    Otherwise, :obj:`numpy.random.lognormal(mean, sigma, size)` samples
    are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> mu, sigma = 3., 1. # mean and standard deviation
    >>> s = dpnp.random.lognormal(mu, sigma, 1000)

    """

    if not use_origin_backend(mean):
        # TODO:
        # array_like of floats for `mean` and `sigma` params
        if not dpnp.isscalar(mean):
            pass
        elif not dpnp.isscalar(sigma):
            pass
        elif sigma < 0:
            pass
        else:
            return dpnp_rng_lognormal(mean, sigma, size).get_pyobj()

    return call_origin(numpy.random.lognormal, mean, sigma, size)


def logseries(p, size=None):
    """
    Draw samples from a logarithmic series distribution.

    For full documentation refer to :obj:`numpy.random.logseries`.

    Notes
    -----
    The function uses `numpy.random.logseries` on the backend and will be
    executed on fallback backend.

    """

    return call_origin(numpy.random.logseries, p, size)


def multinomial(n, pvals, size=None):
    """
    Draw samples from a multinomial distribution.

    For full documentation refer to :obj:`numpy.random.multinomial`.

    Limitations
    -----------
    Parameter `n` limited with int32 max. See, `numpy.iinfo(numpy.int32).max`.
    Sum of ``pvals``, `sum(pvals)` should be between (0, 1).
    Otherwise, :obj:`numpy.random.multinomial(n, pvals, size)`
    samples are drawn.

    Examples
    --------
    Throw a dice 20 times:

    >>> s = dpnp.random.multinomial(20, [1/6.]*6, size=1)
    >>> s.shape
    (1, 6)

    """

    if not use_origin_backend(n):
        pvals_sum = sum(pvals)
        pvals_desc = dpnp.get_dpnp_descriptor(dpnp.array(pvals))
        d = len(pvals)
        if n < 0:
            pass
        elif n > dpnp.iinfo(dpnp.int32).max:
            pass
        elif pvals_sum > 1.0:
            pass
        elif pvals_sum < 0.0:
            pass
        else:
            if size is None:
                shape = (d,)
            else:
                try:
                    shape = (operator.index(size), d)
                except Exception:
                    shape = tuple(size) + (d,)

            return dpnp_rng_multinomial(int(n), pvals_desc, shape).get_pyobj()

    return call_origin(numpy.random.multinomial, n, pvals, size)


def multivariate_normal(mean, cov, size=None, check_valid="warn", tol=1e-8):
    """
    Draw random samples from a multivariate normal distribution.

    For full documentation refer to :obj:`numpy.random.multivariate_normal`.

    Limitations
    -----------
    Parameters `check_valid` and `tol` are not supported.
    Otherwise, :obj:`numpy.random.multivariate_normal(mean, cov, size, check_valid, tol)`
    samples are drawn.

    Examples
    --------
    >>> mean = (1, 2)
    >>> cov = [[1, 0], [0, 1]]
    >>> x = dpnp.random.multivariate_normal(mean, cov, (3, 3))
    >>> x.shape
    (3, 3, 2)

    """

    if not use_origin_backend(mean):
        mean_ = dpnp.get_dpnp_descriptor(dpnp.array(mean, dtype=dpnp.float64))
        cov_ = dpnp.get_dpnp_descriptor(dpnp.array(cov, dtype=dpnp.float64))
        if size is None:
            shape = []
        elif isinstance(size, (int, dpnp.integer)):
            shape = [size]
        else:
            shape = size
        if len(mean_.shape) != 1:
            pass
        elif (len(cov_.shape) != 2) or (cov_.shape[0] != cov_.shape[1]):
            pass
        elif mean_.shape[0] != cov_.shape[0]:
            pass
        else:
            final_shape = list(shape[:])
            final_shape.append(mean_.shape[0])
            return dpnp_rng_multivariate_normal(
                mean_, cov_, final_shape
            ).get_pyobj()

    return call_origin(
        numpy.random.multivariate_normal, mean, cov, size, check_valid, tol
    )


def negative_binomial(n, p, size=None):
    """
    Draw samples from a negative binomial distribution.

    For full documentation refer to :obj:`numpy.random.negative_binomial`.

    Limitations
    -----------
    Parameters `n` and `p` are supported as scalar.
    Otherwise, :obj:`numpy.random.negative_binomial(n, p, size)` samples
    are drawn.
    Output array data type is :obj:`dpnp.int32`.

    Examples
    --------
    Draw samples from the distribution:
    A real world example. A company drills wild-cat oil
    exploration wells, each with an estimated probability of
    success of 0.1.  What is the probability of having one success
    for each successive well, that is what is the probability of a
    single success after drilling 5 wells, after 6 wells, etc.?

    >>> s = dpnp.random.negative_binomial(1, 0.1, 100000)
    >>> for i in range(1, 11):
    ...    probability = sum(s<i) / 100000.
    ...    print(i, "wells drilled, probability of one success =", probability)

    """

    if not use_origin_backend(n):
        # TODO:
        # array_like of floats for `p` and `n` params
        if not dpnp.isscalar(n):
            pass
        elif not dpnp.isscalar(p):
            pass
        elif p > 1 or p < 0:
            pass
        elif n <= 0:
            pass
        else:
            return dpnp_rng_negative_binomial(n, p, size).get_pyobj()

    return call_origin(numpy.random.negative_binomial, n, p, size)


def normal(
    loc=0.0,
    scale=1.0,
    size=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw random samples from a normal (Gaussian) distribution.

    For full documentation refer to :obj:`numpy.random.normal`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Drawn samples from the parameterized normal distribution.
        Output array data type is the same as input `dtype`. If `dtype` is ``None`` (the default),
        :obj:`dpnp.float64` type will be used if device supports it, or :obj:`dpnp.float32` otherwise.

    Limitations
    -----------
    Parameters `loc` and `scale` are supported as scalar.
    Otherwise, :obj:`numpy.random.normal(loc, scale, size)` samples are drawn.
    Parameter `dtype` is supported only as :obj:`dpnp.float32`, :obj:`dpnp.float64` or ``None``.

    Examples
    --------
    Draw samples from the distribution:

    >>> mu, sigma = 0, 0.1 # mean and standard deviation
    >>> s = dpnp.random.normal(mu, sigma, 1000)

    """

    rs = _get_random_state(device=device, sycl_queue=sycl_queue)
    return rs.normal(
        loc=loc, scale=scale, size=size, dtype=None, usm_type=usm_type
    )


def noncentral_chisquare(df, nonc, size=None):
    """
    Draw samples from a non-central chi-square distribution.

    For full documentation refer to :obj:`numpy.random.noncentral_chisquare`.

    TODO

    """

    if not use_origin_backend(df):
        # TODO:
        # array_like of floats for `mean` and `scale`
        if not dpnp.isscalar(df):
            pass
        elif not dpnp.isscalar(nonc):
            pass
        elif df <= 0:
            pass
        elif nonc < 0:
            pass
        else:
            return dpnp_rng_noncentral_chisquare(df, nonc, size).get_pyobj()

    return call_origin(numpy.random.noncentral_chisquare, df, nonc, size)


def noncentral_f(dfnum, dfden, nonc, size=None):
    """
    Draw samples from the non-central F distribution.

    For full documentation refer to :obj:`numpy.random.noncentral_f`.

    Notes
    -----
    The function uses `numpy.random.noncentral_f` on the backend and
    will be executed on fallback backend.

    """

    return call_origin(numpy.random.noncentral_f, dfnum, dfden, nonc, size)


def pareto(a, size=None):
    """
    Draw samples from a Pareto II or Lomax distribution with specified shape.

    For full documentation refer to :obj:`numpy.random.pareto`.

    Limitations
    -----------
    Parameter `a` is supported as a scalar.
    Otherwise, :obj:`numpy.random.pareto(a, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> a = .5  # alpha
    >>> s = dpnp.random.pareto(a, 1000)

    """

    if not use_origin_backend(a):
        # TODO:
        # array_like of floats for `a`
        if not dpnp.isscalar(a):
            pass
        elif a <= 0:
            pass
        else:
            return dpnp_rng_pareto(a, size).get_pyobj()

    return call_origin(numpy.random.pareto, a, size)


def permutation(x):
    """
    Randomly permute a sequence, or return a permuted range.

    For full documentation refer to :obj:`numpy.random.permutation`.

    Examples
    --------
    >>> arr = dpnp.random.permutation(10)
    >>> print(arr)
    [3 8 7 9 0 6 1 2 4 5] # random

    >>> arr = dpnp.random.permutation([1, 4, 9, 12, 15])
    >>> print(arr)
    [12  1  4  9 15] # random

    >>> arr = dpnp.arange(9).reshape((3, 3))
    >>> dpnp.random.permutation(arr)
    >>> print(arr)
    [[0 1 2]
     [3 4 5]
     [6 7 8]]  # random

    """
    if not use_origin_backend(x):
        if isinstance(x, (int, dpnp.integer)):
            arr = dpnp.arange(x)
        else:
            arr = dpnp.array(x)
        shuffle(arr)
        return arr

    return call_origin(numpy.random.permutation, x)


def poisson(lam=1.0, size=None):
    """
    Draw samples from a Poisson distribution.

    For full documentation refer to :obj:`numpy.random.poisson`.

    Limitations
    -----------
    Parameter `lam` is supported as a scalar.
    Otherwise, :obj:`numpy.random.poisson(lam, size)` samples are drawn.
    Output array data type is :obj:`dpnp.int32`.

    Examples
    --------
    Draw samples from the distribution:

    >>> import numpy as np
    >>> s = dpnp.random.poisson(5, 10000)

    """

    if not use_origin_backend(lam):
        # TODO:
        # array_like of floats for `lam` param
        if not dpnp.isscalar(lam):
            pass
        elif lam < 0:
            pass
        else:
            return dpnp_rng_poisson(lam, size).get_pyobj()

    return call_origin(numpy.random.poisson, lam, size)


def power(a, size=None):
    """
    Draws samples in [0, 1] from a power distribution with positive
    exponent a - 1.

    For full documentation refer to :obj:`numpy.random.power`.

    Limitations
    -----------
    Parameter `a` is supported as a scalar.
    Otherwise, :obj:`numpy.random.power(a, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> a = .5  # alpha
    >>> s = dpnp.random.power(a, 1000)

    """

    if not use_origin_backend(a):
        # TODO:
        # array_like of floats for `a`
        if not dpnp.isscalar(a):
            pass
        elif a <= 0:
            pass
        else:
            return dpnp_rng_power(a, size).get_pyobj()

    return call_origin(numpy.random.power, a, size)


def rand(d0, *dn, device=None, usm_type="device", sycl_queue=None):
    """
    Random values in a given shape.

    Create an array of the given shape and populate it with random samples
    from a uniform distribution over [0, 1).

    For full documentation refer to :obj:`numpy.random.rand`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Random values in a given shape.
        Output array data type is :obj:`dpnp.float64` if device supports it, or :obj:`dpnp.float32` otherwise.

    Examples
    --------
    >>> s = dpnp.random.rand(3, 2)

    See Also
    --------
    :obj:`dpnp.random.random`
    :obj:`dpnp.random.random_sample`
    :obj:`dpnp.random.uniform`

    """

    rs = _get_random_state(device=device, sycl_queue=sycl_queue)
    return rs.rand(d0, *dn, usm_type=usm_type)


def randint(
    low,
    high=None,
    size=None,
    dtype=int,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return random integers from `low` (inclusive) to `high` (exclusive).

    For full documentation refer to :obj:`numpy.random.randint`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        `size`-shaped array of random integers from the appropriate distribution,
        or a single such random int if `size` is not provided.
        Output array data type is the same as input `dtype`.

    Limitations
    -----------
    Parameters `low` and `high` are supported only as a scalar.
    Parameter `dtype` is supported only as :obj:`dpnp.int32` or ``int``,
    but ``int`` value is considered to be exactly equivalent to :obj:`dpnp.int32`.
    Otherwise, :obj:`numpy.random.RandomState.randint(low, high, size, dtype)` samples are drawn.

    Examples
    --------
    Draw samples from the distribution:

    >>> low, high = 3, 11 # low and high
    >>> s = dpnp.random.randint(low, high, 1000, dtype=dpnp.int32)

    See Also
    --------
    :obj:`dpnp.random.random_integers` : similar to `randint`, only for the closed
                                         interval [`low`, `high`], and 1 is the
                                         lowest value if `high` is omitted.

    """

    rs = _get_random_state(device=device, sycl_queue=sycl_queue)
    return rs.randint(
        low=low, high=high, size=size, dtype=dtype, usm_type=usm_type
    )


def randn(d0, *dn, device=None, usm_type="device", sycl_queue=None):
    """
    Return a sample (or samples) from the "standard normal" distribution.

    For full documentation refer to :obj:`numpy.random.randn`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
        the standard normal distribution, or a single such float if no parameters were supplied.
        Output array data type is :obj:`dpnp.float64` if device supports it, or :obj:`dpnp.float32` otherwise.

    Examples
    --------
    >>> dpnp.random.randn()
    2.1923875335537315  # random

    Two-by-four array of samples from N(3, 6.25):

    >>> s = 3 + 2.5 * dpnp.random.randn(2, 4)

    See Also
    --------
    :obj:`dpnp.random.standard_normal`
    :obj:`dpnp.random.normal`

    """

    rs = _get_random_state(device=device, sycl_queue=sycl_queue)
    return rs.randn(d0, *dn, usm_type=usm_type)


def random(size=None, device=None, usm_type="device", sycl_queue=None):
    """
    Return random floats in the half-open interval [0.0, 1.0).

    Alias for random_sample.

    For full documentation refer to :obj:`numpy.random.random`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of random floats of shape `size` (if ``size=None``, zero dimension array with a single float is returned).
        Output array data type is :obj:`dpnp.float64` if device supports it, or :obj:`dpnp.float32` otherwise.

    Examples
    --------
    >>> s = dpnp.random.random(1000)

    See Also
    --------
    :obj:`dpnp.random.rand`
    :obj:`dpnp.random.random_sample`
    :obj:`dpnp.random.uniform`

    """

    return random_sample(
        size=size, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def random_integers(
    low, high=None, size=None, device=None, usm_type="device", sycl_queue=None
):
    """
    Random integers between `low` and `high`, inclusive.

    For full documentation refer to :obj:`numpy.random.random_integers`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        `size`-shaped array of random integers from the appropriate distribution,
        or a single such random int if `size` is not provided.

    Limitations
    -----------
    Parameters `low` and `high` are supported as scalar.
    Otherwise, :obj:`numpy.random.random_integers(low, high, size)` samples are drawn.

    See Also
    --------
    :obj:`dpnp.random.randint`

    """

    if high is None:
        high = low
        low = 0

    # TODO:
    # array_like of floats for `low` and `high` params
    if not dpnp.isscalar(low):
        pass
    elif not dpnp.isscalar(high):
        pass
    else:
        return randint(
            low,
            int(high) + 1,
            size=size,
            device=device,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )

    return call_origin(numpy.random.random_integers, low, high, size)


def random_sample(size=None, device=None, usm_type="device", sycl_queue=None):
    """
    Return random floats in the half-open interval [0.0, 1.0).

    Results are from the “continuous uniform” distribution over the interval.

    For full documentation refer to :obj:`numpy.random.random_sample`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of random floats of shape `size` (if ``size=None``, zero dimension array with a single float is returned).
        Output array data type is :obj:`dpnp.float64` if device supports it, or :obj:`dpnp.float32` otherwise.

    Examples
    --------
    >>> s = dpnp.random.random_sample((5,))

    See Also
    --------
    :obj:`dpnp.random.rand`
    :obj:`dpnp.random.random`
    :obj:`dpnp.random.uniform`

    """

    rs = _get_random_state(device=device, sycl_queue=sycl_queue)
    return rs.random_sample(size=size, usm_type=usm_type)


def ranf(size=None, device=None, usm_type="device", sycl_queue=None):
    """
    Return random floats in the half-open interval [0.0, 1.0).

    This is an alias of random_sample.

    For full documentation refer to :obj:`numpy.random.ranf`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of random floats of shape `size` (if ``size=None``, zero dimension array with a single float is returned).
        Output array data type is :obj:`dpnp.float64` if device supports it, or :obj:`dpnp.float32` otherwise.

    Examples
    --------
    >>> s = dpnp.random.ranf(1000)

    See Also
    --------
    :obj:`dpnp.random.rand`
    :obj:`dpnp.random.random`
    :obj:`dpnp.random.random_sample`
    :obj:`dpnp.random.uniform`

    """

    return random_sample(
        size=size, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def rayleigh(scale=1.0, size=None):
    """
    Draw samples from a Rayleigh distribution.

    For full documentation refer to :obj:`numpy.random.rayleigh`.

    Limitations
    -----------
    Parameter `scale` is supported as a scalar.
    Otherwise, :obj:`numpy.random.rayleigh(scale, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> import numpy as np
    >>> s = dpnp.random.rayleigh(1.0, 10000)

    """

    if not use_origin_backend(scale):
        # TODO:
        # array_like of floats for `scale` params
        if not dpnp.isscalar(scale):
            pass
        elif scale < 0:
            pass
        else:
            return dpnp_rng_rayleigh(scale, size).get_pyobj()

    return call_origin(numpy.random.rayleigh, scale, size)


def sample(size=None, device=None, usm_type="device", sycl_queue=None):
    """
    Return random floats in the half-open interval [0.0, 1.0).

    This is an alias of random_sample.

    For full documentation refer to :obj:`numpy.random.sample`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Array of random floats of shape `size` (if ``size=None``, zero dimension array with a single float is returned).
        Output array data type is :obj:`dpnp.float64` if device supports it, or :obj:`dpnp.float32` otherwise.

    Examples
    --------
    >>> s = dpnp.random.sample(1000)

    See Also
    --------
    :obj:`dpnp.random.rand`
    :obj:`dpnp.random.random`
    :obj:`dpnp.random.random_sample`
    :obj:`dpnp.random.uniform`

    """

    return random_sample(
        size=size, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def shuffle(x1):
    """
    Modify a sequence in-place by shuffling its contents.

    For full documentation refer to :obj:`numpy.random.shuffle`.

    Limitations
    -----------
    Otherwise, the function will use :obj:`numpy.random.shuffle` on the backend
    and will be executed on fallback backend.

    """

    x1_desc = dpnp.get_dpnp_descriptor(x1, copy_when_strides=False)
    if x1_desc:
        if not dpnp.is_type_supported(x1_desc.dtype):
            pass
        else:
            dpnp_rng_shuffle(x1_desc).get_pyobj()
            return

    call_origin(numpy.random.shuffle, x1, dpnp_inplace=True)
    return


def seed(seed=None, device=None, sycl_queue=None):
    """
    Reseed a legacy MT19937 random number generator engine.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where an array with generated numbers
        will be created. The `device` can be ``None`` (the default), an OneAPI
        filter selector string, an instance of :class:`dpctl.SyclDevice`
        corresponding to a non-partitioned SYCL device, an instance of
        :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for an array with generated numbers.

    Limitations
    -----------
    The `seed` parameter is supported as a scalar or an array of at most three
    integer scalars.

    """

    # update a mt19937 random number for both RandomState and legacy functionality
    global _dpnp_random_states

    sycl_queue = dpnp.get_normalized_queue_device(
        device=device, sycl_queue=sycl_queue
    )
    _dpnp_random_states[sycl_queue] = RandomState(
        seed=seed, sycl_queue=sycl_queue
    )

    if not use_origin_backend(seed):
        # TODO:
        # array_like of ints for `seed`
        if seed is None:
            seed = 1
        if not isinstance(seed, int):
            pass
        elif seed < 0:
            pass
        else:
            # TODO:
            # migrate to a single approach with RandomState class
            dpnp_rng_srand(seed)

    # always reseed numpy engine also
    return call_origin(numpy.random.seed, seed, allow_fallback=True)


def standard_cauchy(size=None):
    """
    Draw samples from a standard Cauchy distribution with mode = 0.

    Also known as the Lorentz distribution.

    Limitations
    -----------
    Output array data type is default float type.

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
        return dpnp_rng_standard_cauchy(size).get_pyobj()

    return call_origin(numpy.random.standard_cauchy, size)


def standard_exponential(size=None):
    """
    Draw samples from the standard exponential distribution.

    `standard_exponential` is identical to the exponential distribution
    with a scale parameter of 1.

    Limitations
    -----------
    Output array data type is default float type.

    Examples
    --------
    Output a 3x8000 array:

    >>> n = dpnp.random.standard_exponential((3, 8000))

    """

    if not use_origin_backend(size):
        return dpnp_rng_standard_exponential(size).get_pyobj()

    return call_origin(numpy.random.standard_exponential, size)


def standard_gamma(shape, size=None):
    """
    Draw samples from a standard Gamma distribution.

    For full documentation refer to :obj:`numpy.random.standard_gamma`.

    Limitations
    -----------
    Parameter `shape` is supported as a scalar.
    Otherwise, :obj:`numpy.random.standard_gamma(shape, size)` samples
    are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> shape = 2.
    >>> s = dpnp.random.standard_gamma(shape, 1000000)

    """

    if not use_origin_backend(shape):
        # TODO:
        # array_like of floats for `shape`
        if not dpnp.isscalar(shape):
            pass
        elif shape < 0:
            pass
        else:
            return dpnp_rng_standard_gamma(shape, size).get_pyobj()

    return call_origin(numpy.random.standard_gamma, shape, size)


def standard_normal(size=None, device=None, usm_type="device", sycl_queue=None):
    """
    Draw samples from a standard Normal distribution ``(mean=0, stdev=1)``.

    For full documentation refer to :obj:`numpy.random.standard_normal`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        A floating-point array of shape `size` of drawn samples, or a
        single sample if `size` was not specified.
        Output array data type is :obj:`dpnp.float64` if device supports it, or :obj:`dpnp.float32` otherwise.

    Examples
    --------
    Draw samples from the distribution:

    >>> s = dpnp.random.standard_normal(1000)

    """

    rs = _get_random_state(device=device, sycl_queue=sycl_queue)
    return rs.standard_normal(size=size, usm_type=usm_type)


def standard_t(df, size=None):
    """
    Draw samples from a standard Student’s t distribution with
    `df` degrees of freedom.

    For full documentation refer to :obj:`numpy.random.standard_t`.

    Limitations
    -----------
    Parameter `df` is supported as a scalar.
    Otherwise, :obj:`numpy.random.standard_t(df, size)` samples
    are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> df = 2.
    >>> s = dpnp.random.standard_t(df, 1000000)

    """

    if not use_origin_backend(df):
        # TODO:
        # array_like of floats for `df`
        if not dpnp.isscalar(df):
            pass
        elif df <= 0:
            pass
        else:
            return dpnp_rng_standard_t(df, size).get_pyobj()

    return call_origin(numpy.random.standard_t, df, size)


def triangular(left, mode, right, size=None):
    """
    Draw samples from the triangular distribution over the interval
    [left, right].

    For full documentation refer to :obj:`numpy.random.triangular`.

    Limitations
    -----------
    Parameters `left`, `mode` and `right` are supported as scalar.
    Otherwise, :obj:`numpy.random.triangular(left, mode, right, size)`
    samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> df = 2.
    >>> s = dpnp.random.triangular(-3, 0, 8, 1000000)

    """

    if not use_origin_backend(left):
        # TODO:
        # array_like of floats for `left`, `mode`, `right`.
        if not dpnp.isscalar(left):
            pass
        elif not dpnp.isscalar(mode):
            pass
        elif not dpnp.isscalar(right):
            pass
        elif left > mode:
            pass
        elif mode > right:
            pass
        elif left == right:
            pass
        else:
            return dpnp_rng_triangular(left, mode, right, size).get_pyobj()

    return call_origin(numpy.random.triangular, left, mode, right, size)


def uniform(
    low=0.0,
    high=1.0,
    size=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high).
    In other words, any value within the given interval is equally likely to be drawn by uniform.

    For full documentation refer to :obj:`numpy.random.uniform`.

    Parameters
    ----------
    device : {None, string, SyclDevice, SyclQueue}, optional
        An array API concept of device where the output array is created.
        The `device` can be ``None`` (the default), an OneAPI filter selector string,
        an instance of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL device,
        an instance of :class:`dpctl.SyclQueue`, or a `Device` object returned by
        :obj:`dpnp.dpnp_array.dpnp_array.device` property.
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

    Returns
    -------
    out : dpnp.ndarray
        Drawn samples from the parameterized uniform distribution.
        Output array data type is the same as input `dtype`. If `dtype` is ``None`` (the default),
        :obj:`dpnp.float64` type will be used if device supports it, or :obj:`dpnp.float32` otherwise.

    Limitations
    -----------
    Parameters `low` and `high` are supported as a scalar. Otherwise,
    :obj:`numpy.random.uniform(low, high, size)` samples are drawn.
    Parameter `dtype` is supported only as :obj:`dpnp.int32`, :obj:`dpnp.float32`, :obj:`dpnp.float64` or ``None``.

    Examples
    --------
    Draw samples from the distribution:

    >>> low, high = 0, 0.1 # low and high
    >>> s = dpnp.random.uniform(low, high, 10000)

    See Also
    --------
    :obj:`dpnp.random.random` : Floats uniformly distributed over ``[0, 1)``.

    """

    rs = _get_random_state(device=device, sycl_queue=sycl_queue)
    return rs.uniform(
        low=low, high=high, size=size, dtype=None, usm_type=usm_type
    )


def vonmises(mu, kappa, size=None):
    """
    Draw samples from a von Mises distribution.

    For full documentation refer to :obj:`numpy.random.vonmises`.

    Limitations
    -----------
    Parameters `mu` and `kappa` are supported as scalar.
    Otherwise, :obj:`numpy.random.vonmises(mu, kappa, size)`
    samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    Draw samples from the distribution:

    >>> mu, kappa = 0.0, 4.0 # mean and dispersion
    >>> s = dpnp.random.vonmises(mu, kappa, 1000)

    """

    if not use_origin_backend(mu):
        # TODO:
        # array_like of floats for `mu`, `kappa`.
        if not dpnp.isscalar(mu):
            pass
        elif not dpnp.isscalar(kappa):
            pass
        elif numpy.isnan(kappa):
            return dpnp.nan
        elif kappa < 0:
            pass
        else:
            return dpnp_rng_vonmises(mu, kappa, size).get_pyobj()

    return call_origin(numpy.random.vonmises, mu, kappa, size)


def wald(mean, scale, size=None):
    """
    Draw samples from a Wald, or inverse Gaussian, distribution.

    For full documentation refer to :obj:`numpy.random.wald`.

    Limitations
    -----------
    Parameters `mean` and `scale` are supported as scalar.
    Otherwise, :obj:`numpy.random.wald(mean, scale, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    >>> loc, scale = 3., 2.
    >>> s = dpnp.random.wald(loc, scale, 1000)

    """

    if not use_origin_backend(mean):
        # TODO:
        # array_like of floats for `mean` and `scale`
        if not dpnp.isscalar(mean):
            pass
        elif not dpnp.isscalar(scale):
            pass
        elif mean <= 0:
            pass
        elif scale <= 0:
            pass
        else:
            return dpnp_rng_wald(mean, scale, size).get_pyobj()

    return call_origin(numpy.random.wald, mean, scale, size)


def weibull(a, size=None):
    """
    Draw samples from a Weibull distribution.

    For full documentation refer to :obj:`numpy.random.weibull`.

    Limitations
    -----------
    Parameter `a` is supported as a scalar.
    Otherwise, :obj:`numpy.random.weibull(a, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    >>> a = 5. # shape
    >>> s = np.random.weibull(a, 1000)

    """

    if not use_origin_backend(a):
        # TODO:
        # array_like of floats for `a` param
        if not dpnp.isscalar(a):
            pass
        elif a < 0:
            pass
        else:
            return dpnp_rng_weibull(a, size).get_pyobj()

    return call_origin(numpy.random.weibull, a, size)


def zipf(a, size=None):
    """
    Returns an array of samples drawn from the Zipf distribution.

    For full documentation refer to :obj:`numpy.random.zipf`.

    Limitations
    -----------
    Parameter `a`` is supported as a scalar.
    Otherwise, :obj:`numpy.zipf.weibull(a, size)` samples are drawn.
    Output array data type is :obj:`dpnp.float64`.

    Examples
    --------
    >>> a = 2. # parameter
    >>> s = np.random.zipf(a, 1000)

    """

    if not use_origin_backend(a):
        # TODO:
        # array_like of floats for `a` param
        if not dpnp.isscalar(a):
            pass
        elif a <= 1:
            pass
        else:
            return dpnp_rng_zipf(a, size).get_pyobj()

    return call_origin(numpy.random.zipf, a, size)


_dpnp_random_states = {}
