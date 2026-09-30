# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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
The legacy random number generation API modelled on
:class:`numpy.random.RandomState`.

A :class:`RandomState` draws from a :class:`dpnp.random.Philox4x3210` bit
generator, and samples every distribution on the device by the algorithms of
:class:`dpnp.random.Generator`, with the parameter checks of NumPy's legacy
API. Neither the values nor the state are those of NumPy's MT19937.

"""

# pylint: disable=protected-access
# pylint: disable=too-many-public-methods

import operator
import warnings

import numpy

import dpnp
import dpnp.libtensor._rng_impl as ri  # pylint: disable=no-name-in-module

from ._bit_generator import Philox4x3210
from ._generator import (
    _CONS_BOUNDED_0_1,
    _CONS_GTE_1,
    _CONS_NON_NEGATIVE,
    _CONS_POSITIVE,
    _INT64_MAX,
    Generator,
    _as_int64,
    _check_constraint,
    _int64,
    _is_array,
    _is_scalar_path,
    _normalize_size,
    _uniform_range,
)

__all__ = ["RandomState"]

# the messages of RandomState.choice
_CHOICE_MSGS = {
    "a_type": "a must be 1-dimensional or an integer",
    "a_int": "a must be greater than 0 unless no samples are taken",
    "a_1d": "a must be 1-dimensional",
    "a_empty": "'a' cannot be empty unless no samples are taken",
    "p_1d": "'p' must be 1-dimensional",
    "p_size": "'a' and 'p' must have same size",
    "p_nan": "probabilities contain NaN",
    "p_neg": "probabilities are not non-negative",
    "p_sum": "probabilities do not sum to 1",
    "too_large": (
        "Cannot take a larger sample than population when 'replace=False'"
    ),
}

_SEED_MAX = 2**32 - 1


def _legacy_seed(seed):
    """
    Check `seed` as NumPy's legacy seeding does, and return it in a form
    :class:`numpy.random.SeedSequence` takes.

    """

    if seed is None:
        return None
    if dpnp.is_supported_array_type(seed):
        seed = dpnp.asnumpy(seed)

    msg = "Seed must be between 0 and 2**32 - 1"
    try:
        if hasattr(seed, "squeeze"):
            seed = seed.squeeze()
        idx = operator.index(seed)
    except TypeError:
        obj = numpy.asarray(seed)
        if obj.size == 0:
            raise ValueError("Seed must be non-empty") from None
        obj = obj.astype(numpy.int64, casting="safe")
        if obj.ndim != 1:
            raise ValueError("Seed array must be 1-d") from None
        if ((obj > _SEED_MAX) | (obj < 0)).any():
            raise ValueError(msg) from None
        return obj
    if idx > _SEED_MAX or idx < 0:
        raise ValueError(msg)
    return idx


class RandomState:
    """
    A container of a pseudo-random number generator on a SYCL device, with
    the legacy API of NumPy.

    For full documentation refer to :obj:`numpy.random.RandomState`.

    Parameters
    ----------
    seed : {None, int, array_like[ints]}, optional
        A random seed to initialize the pseudo-random number generator, as
        for :obj:`numpy.random.RandomState`: an integer in [0, 2**32 - 1] or
        a 1-D array of such integers. With ``None``, fresh entropy is pulled
        from the operating system.

        Default: ``None``.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpnp.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.

        Default: ``None``.

    Limitations
    -----------
    The bit generator is :class:`dpnp.random.Philox4x3210` rather than
    MT19937, so the values drawn differ from NumPy's, and the state is that
    of :class:`dpnp.random.Philox4x3210`. The values for a seed are the same
    on every device, and those of ``dpnp.random.default_rng(seed)`` for the
    distributions whose checks agree.

    Every method returns a :class:`dpnp.ndarray`, zero-dimensional when
    `size` is ``None`` and the parameters are scalars, rather than a Python
    scalar. The floating-point methods take a `dtype`, which defaults to the
    default floating-point type of the device, and every method a
    `usm_type`, the type of USM allocation of the output. Parameters that
    are on the device are not checked, and invalid values give undefined
    results.

    """

    def __init__(self, seed=None, device=None, sycl_queue=None):
        self._sycl_queue = dpnp.get_normalized_queue_device(
            device=device, sycl_queue=sycl_queue
        )
        self._generator = Generator(self._make_bit_generator(seed))

    def _make_bit_generator(self, seed):
        return Philox4x3210(_legacy_seed(seed), sycl_queue=self._sycl_queue)

    def __repr__(self):
        return self.__str__() + f" at 0x{id(self):X}"

    def __str__(self):
        return f"{self.__class__.__name__}(Philox4x3210)"

    def seed(self, seed=None):
        """
        Reseed the generator.

        For full documentation refer to :obj:`numpy.random.RandomState.seed`.

        """
        self._generator.bit_generator = self._make_bit_generator(seed)

    def get_state(self, legacy=True):
        """
        Return the state of the generator.

        For full documentation refer to
        :obj:`numpy.random.RandomState.get_state`.

        Limitations
        -----------
        The state is always the dictionary of
        :attr:`dpnp.random.Philox4x3210.state`, whatever `legacy` is.

        """
        # pylint: disable=unused-argument
        return self._generator.bit_generator.state

    def set_state(self, state):
        """
        Set the state of the generator.

        For full documentation refer to
        :obj:`numpy.random.RandomState.set_state`.

        Limitations
        -----------
        `state` must be a state of :class:`dpnp.random.Philox4x3210`, as
        returned by :meth:`get_state`.

        """
        if not isinstance(state, dict):
            raise ValueError("state must be for a Philox4x3210 PRNG")
        self._generator.bit_generator.state = state

    def get_sycl_queue(self):
        """
        Return the :class:`dpctl.SyclQueue` the generator allocates data on.

        Returns
        -------
        queue : dpctl.SyclQueue
            A SYCL queue used for data allocation.

        """
        return self._sycl_queue

    def get_sycl_device(self):
        """
        Return the :class:`dpctl.SyclDevice` the generator allocates data on.

        Returns
        -------
        device : dpctl.SyclDevice
            A SYCL device used to allocate data on.

        """
        return self._sycl_queue.sycl_device

    # Simple random data

    def rand(self, *args, dtype=None, usm_type="device"):
        """
        Random values in a given shape, from a uniform distribution over
        [0, 1).

        For full documentation refer to :obj:`numpy.random.RandomState.rand`.

        """
        return self.random_sample(args or None, dtype, usm_type=usm_type)

    def randn(self, *args, dtype=None, usm_type="device"):
        """
        Samples of the standard normal distribution in a given shape.

        For full documentation refer to :obj:`numpy.random.RandomState.randn`.

        """
        return self.standard_normal(args or None, dtype, usm_type=usm_type)

    def randint(
        self, low, high=None, size=None, dtype=int, *, usm_type="device"
    ):
        """
        Random integers from `low` (inclusive) to `high` (exclusive).

        For full documentation refer to
        :obj:`numpy.random.RandomState.randint`.

        """
        return self._generator._integers(
            low, high, size, dtype, False, usm_type, "randint"
        )

    def random_integers(self, low, high=None, size=None, *, usm_type="device"):
        """
        Random integers from `low` to `high`, both inclusive.

        For full documentation refer to
        :obj:`numpy.random.RandomState.random_integers`.

        """
        if high is None:
            warnings.warn(
                "This function is deprecated. Please call "
                f"randint(1, {low} + 1) instead",
                DeprecationWarning,
                stacklevel=2,
            )
            high = low
            low = 1
        else:
            warnings.warn(
                "This function is deprecated. Please call "
                f"randint({low}, {high} + 1) instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if dpnp.is_supported_array_type(high):
            high = high + 1
        elif _is_array(high):
            high = numpy.array(high) + 1
        else:
            high = int(high) + 1
        return self.randint(low, high, size, _int64, usm_type=usm_type)

    def random_sample(self, size=None, dtype=None, *, usm_type="device"):
        """
        Random floats in the half-open interval [0.0, 1.0).

        For full documentation refer to
        :obj:`numpy.random.RandomState.random_sample`.

        """
        return self._generator.random(size, dtype, usm_type=usm_type)

    def random(self, size=None, dtype=None, *, usm_type="device"):
        """
        Random floats in the half-open interval [0.0, 1.0).

        For full documentation refer to :obj:`numpy.random.RandomState.random`.

        """
        return self._generator.random(size, dtype, usm_type=usm_type)

    def tomaxint(self, size=None, *, usm_type="device"):
        """
        Random integers from 0 to the largest ``int64``, both inclusive.

        For full documentation refer to
        :obj:`numpy.random.RandomState.tomaxint`.

        """
        return self._generator._integers(
            0, _INT64_MAX, size, _int64, True, usm_type, "tomaxint"
        )

    def choice(self, a, size=None, replace=True, p=None, *, usm_type="device"):
        """
        Generate a random sample from a given 1-D array.

        For full documentation refer to :obj:`numpy.random.RandomState.choice`.

        Limitations
        -----------
        `p` is checked only when it is on the host.

        """
        return self._generator._choice(
            a, size, replace, p, 0, usm_type, _CHOICE_MSGS
        )

    def bytes(self, length):
        """
        Return random bytes.

        For full documentation refer to :obj:`numpy.random.RandomState.bytes`.

        """
        return self._generator.bytes(length)

    # Permutations

    def shuffle(self, x):
        """
        Modify an array in place by shuffling its contents along the first
        axis.

        For full documentation refer to
        :obj:`numpy.random.RandomState.shuffle`.

        Limitations
        -----------
        As in CuPy, `x` must be a :class:`dpnp.ndarray`.

        """
        self._generator.shuffle(x)

    def permutation(self, x):
        """
        Randomly permute a sequence, or return a permuted range.

        For full documentation refer to
        :obj:`numpy.random.RandomState.permutation`.

        """
        if not isinstance(x, (int, numpy.integer)) and numpy.ndim(x) == 0:
            raise IndexError("x must be an integer or at least 1-dimensional")
        return self._generator.permutation(x)

    # Distributions

    def beta(self, a, b, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a Beta distribution.

        For full documentation refer to :obj:`numpy.random.RandomState.beta`.

        """
        return self._generator.beta(a, b, size, dtype, usm_type=usm_type)

    def binomial(self, n, p, size=None, *, usm_type="device"):
        """
        Draw samples from a binomial distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.binomial`.

        """
        return self._generator.binomial(n, p, size, usm_type=usm_type)

    def chisquare(self, df, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a chi-square distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.chisquare`.

        """
        return self._generator.chisquare(df, size, dtype, usm_type=usm_type)

    def dirichlet(self, alpha, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from the Dirichlet distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.dirichlet`.

        """
        if not dpnp.is_supported_array_type(alpha) and numpy.ndim(alpha) == 1:
            if numpy.any(numpy.asarray(alpha, dtype=numpy.float64) <= 0):
                raise ValueError("alpha <= 0")
        return self._generator.dirichlet(alpha, size, dtype, usm_type=usm_type)

    def exponential(
        self, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from an exponential distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.exponential`.

        """
        return self._generator.exponential(
            scale, size, dtype, usm_type=usm_type
        )

    def f(self, dfnum, dfden, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from an F distribution.

        For full documentation refer to :obj:`numpy.random.RandomState.f`.

        """
        return self._generator.f(dfnum, dfden, size, dtype, usm_type=usm_type)

    def gamma(
        self, shape, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a Gamma distribution.

        For full documentation refer to :obj:`numpy.random.RandomState.gamma`.

        """
        return self._generator.gamma(
            shape, scale, size, dtype, usm_type=usm_type
        )

    def geometric(self, p, size=None, *, usm_type="device"):
        """
        Draw samples from the geometric distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.geometric`.

        """
        return self._generator.geometric(p, size, usm_type=usm_type)

    def gumbel(
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a Gumbel distribution.

        For full documentation refer to :obj:`numpy.random.RandomState.gumbel`.

        """
        return self._generator.gumbel(
            loc, scale, size, dtype, usm_type=usm_type
        )

    def hypergeometric(
        self, ngood, nbad, nsample, size=None, *, usm_type="device"
    ):
        """
        Draw samples from a hypergeometric distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.hypergeometric`.

        Limitations
        -----------
        As in :meth:`dpnp.random.Generator.hypergeometric`, `ngood` and
        `nbad` must be less than 10**9.

        """
        params = tuple(_as_int64(x) for x in (ngood, nbad, nsample))
        if not any(dpnp.is_supported_array_type(x) for x in params):
            good, bad, sample = (numpy.asarray(x) for x in params)
            if numpy.any(good + bad < sample):
                raise ValueError("ngood + nbad < nsample")
        scalar = _is_scalar_path(*params)
        for x, name, cons in zip(
            params,
            ("ngood", "nbad", "nsample"),
            (_CONS_NON_NEGATIVE, _CONS_NON_NEGATIVE, _CONS_GTE_1),
        ):
            _check_constraint(x, name, cons, scalar)
        return self._generator.hypergeometric(*params, size, usm_type=usm_type)

    def laplace(
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from the Laplace or double exponential distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.laplace`.

        """
        return self._generator.laplace(
            loc, scale, size, dtype, usm_type=usm_type
        )

    def logistic(
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a logistic distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.logistic`.

        """
        return self._generator.logistic(
            loc, scale, size, dtype, usm_type=usm_type
        )

    def lognormal(
        self, mean=0.0, sigma=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a log-normal distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.lognormal`.

        """
        return self._generator.lognormal(
            mean, sigma, size, dtype, usm_type=usm_type
        )

    def logseries(self, p, size=None, *, usm_type="device"):
        """
        Draw samples from a logarithmic series distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.logseries`.

        """
        return self._generator.logseries(p, size, usm_type=usm_type)

    def multinomial(self, n, pvals, size=None, *, usm_type="device"):
        """
        Draw samples from a multinomial distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.multinomial`.

        """
        t = type(n)
        if _is_array(n) or not (
            hasattr(t, "__int__") or hasattr(t, "__index__")
        ):
            raise TypeError("an integer is required")
        n = int(n)
        if not dpnp.is_supported_array_type(pvals):
            pvals = numpy.asarray(pvals, dtype=numpy.float64)
        ndim = pvals.ndim
        if ndim == 0:
            raise TypeError("pvals must be a 1-d sequence")
        if ndim > 1:
            raise ValueError(
                "setting an array element with a sequence. The requested "
                "array would exceed the maximum number of dimension of 1."
            )
        if len(pvals) == 0:
            size = _normalize_size(size)
            shape = (() if size is None else size) + (0,)
            return dpnp.empty(
                shape,
                dtype=_int64,
                usm_type=usm_type,
                sycl_queue=self._sycl_queue,
            )
        return self._generator.multinomial(n, pvals, size, usm_type=usm_type)

    def multivariate_normal(
        self,
        mean,
        cov,
        size=None,
        check_valid="warn",
        tol=1e-8,
        *,
        dtype=None,
        usm_type="device",
    ):
        """
        Draw random samples from a multivariate normal distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.multivariate_normal`.

        Limitations
        -----------
        The covariance matrix is factorized as in
        :meth:`dpnp.random.Generator.multivariate_normal`.

        """
        return self._generator.multivariate_normal(
            mean,
            cov,
            size,
            check_valid,
            tol,
            method="svd",
            dtype=dtype,
            usm_type=usm_type,
        )

    def negative_binomial(self, n, p, size=None, *, usm_type="device"):
        """
        Draw samples from a negative binomial distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.negative_binomial`.

        Limitations
        -----------
        As in NumPy, the values are undefined when ``(1 - p) / p * n`` is
        too large for the Poisson distribution, as for ``p == 0``.

        """
        scalar = _is_scalar_path(n, p)
        _check_constraint(n, "n", _CONS_POSITIVE, scalar)
        _check_constraint(p, "p", _CONS_BOUNDED_0_1, scalar)
        return self._generator._discrete_sample(
            ri._negative_binomial, (n, p), size, usm_type
        )

    def noncentral_chisquare(
        self, df, nonc, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a noncentral chi-square distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.noncentral_chisquare`.

        """
        return self._generator.noncentral_chisquare(
            df, nonc, size, dtype, usm_type=usm_type
        )

    def noncentral_f(
        self, dfnum, dfden, nonc, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from the noncentral F distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.noncentral_f`.

        """
        return self._generator.noncentral_f(
            dfnum, dfden, nonc, size, dtype, usm_type=usm_type
        )

    def normal(
        self, loc=0.0, scale=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw random samples from a normal (Gaussian) distribution.

        For full documentation refer to :obj:`numpy.random.RandomState.normal`.

        """
        return self._generator.normal(
            loc, scale, size, dtype, usm_type=usm_type
        )

    def pareto(self, a, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a Pareto II or Lomax distribution with specified
        shape.

        For full documentation refer to :obj:`numpy.random.RandomState.pareto`.

        """
        return self._generator.pareto(a, size, dtype, usm_type=usm_type)

    def poisson(self, lam=1.0, size=None, *, usm_type="device"):
        """
        Draw samples from a Poisson distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.poisson`.

        """
        return self._generator.poisson(lam, size, usm_type=usm_type)

    def power(self, a, size=None, dtype=None, *, usm_type="device"):
        """
        Draws samples in [0, 1] from a power distribution with positive
        exponent a - 1.

        For full documentation refer to :obj:`numpy.random.RandomState.power`.

        """
        return self._generator.power(a, size, dtype, usm_type=usm_type)

    def rayleigh(self, scale=1.0, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a Rayleigh distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.rayleigh`.

        """
        return self._generator.rayleigh(scale, size, dtype, usm_type=usm_type)

    def standard_cauchy(self, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a standard Cauchy distribution with mode = 0.

        For full documentation refer to
        :obj:`numpy.random.RandomState.standard_cauchy`.

        """
        return self._generator.standard_cauchy(size, dtype, usm_type=usm_type)

    def standard_exponential(self, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from the standard exponential distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.standard_exponential`.

        """
        return self._generator.standard_exponential(
            size, dtype, usm_type=usm_type
        )

    def standard_gamma(
        self, shape, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a standard Gamma distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.standard_gamma`.

        """
        return self._generator.standard_gamma(
            shape, size, dtype, usm_type=usm_type
        )

    def standard_normal(self, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        For full documentation refer to
        :obj:`numpy.random.RandomState.standard_normal`.

        """
        return self._generator.standard_normal(size, dtype, usm_type=usm_type)

    def standard_t(self, df, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a standard Student's t distribution with `df`
        degrees of freedom.

        For full documentation refer to
        :obj:`numpy.random.RandomState.standard_t`.

        """
        return self._generator.standard_t(df, size, dtype, usm_type=usm_type)

    def triangular(
        self, left, mode, right, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from the triangular distribution over the interval
        [left, right].

        For full documentation refer to
        :obj:`numpy.random.RandomState.triangular`.

        """
        return self._generator.triangular(
            left, mode, right, size, dtype, usm_type=usm_type
        )

    def uniform(
        self, low=0.0, high=1.0, size=None, dtype=None, *, usm_type="device"
    ):
        """
        Draw samples from a uniform distribution over [low, high).

        For full documentation refer to
        :obj:`numpy.random.RandomState.uniform`.

        """
        _uniform_range(low, high, "Range exceeds valid bounds")
        return self._generator._float_sample(
            ri._uniform, (low, high), size, dtype, usm_type, "uniform"
        )

    def vonmises(self, mu, kappa, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a von Mises distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.vonmises`.

        """
        return self._generator.vonmises(
            mu, kappa, size, dtype, usm_type=usm_type
        )

    def wald(self, mean, scale, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a Wald, or inverse Gaussian, distribution.

        For full documentation refer to :obj:`numpy.random.RandomState.wald`.

        """
        return self._generator.wald(mean, scale, size, dtype, usm_type=usm_type)

    def weibull(self, a, size=None, dtype=None, *, usm_type="device"):
        """
        Draw samples from a Weibull distribution.

        For full documentation refer to
        :obj:`numpy.random.RandomState.weibull`.

        """
        return self._generator.weibull(a, size, dtype, usm_type=usm_type)

    def zipf(self, a, size=None, *, usm_type="device"):
        """
        Draw samples from a Zipf distribution.

        For full documentation refer to :obj:`numpy.random.RandomState.zipf`.

        """
        return self._generator.zipf(a, size, usm_type=usm_type)
