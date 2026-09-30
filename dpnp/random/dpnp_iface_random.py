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
The functions of :mod:`dpnp.random` of NumPy's legacy API.

Every function draws from a :class:`dpnp.random.RandomState` of the SYCL
queue given by `device` or `sycl_queue`. Without either, the queue of the
first parameter that is a :class:`dpnp.ndarray` is used, or else the default
queue. There is one state per queue, which :func:`seed` reseeds.

    .. seealso:: :obj:`numpy.random`

"""

# pylint: disable=redefined-builtin
# pylint: disable=too-many-arguments

import threading

import dpnp

from .dpnp_random_state import RandomState

# The states of the queues. The lock guarantees that concurrent callers
# asking for the state of the same queue get the same instance back rather
# than each building one of their own.
_random_states = {}
_random_states_lock = threading.Lock()

_QUEUE_PARAMS = """\
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an
        instance of :class:`dpctl.SyclDevice` corresponding to a
        non-partitioned SYCL device, an instance of :class:`dpctl.SyclQueue`,
        or a :class:`dpnp.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying.

        Default: ``None``."""

_USM_PARAM = """\
    usm_type : {"device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.

        Default: ``"device"``."""

_DTYPE_PARAM = """\
    dtype : {None, float32, float64}, optional
        The floating-point type of the output.

        Default: ``None``, the default floating-point type of the device."""


def _params_doc(func):
    """Fill the placeholders of the parameters common to the functions."""

    func.__doc__ = (
        func.__doc__.replace("    %(queue)s", _QUEUE_PARAMS)
        .replace("    %(usm)s", _USM_PARAM)
        .replace("    %(dtype)s", _DTYPE_PARAM)
    )
    return func


def _get_random_state(device=None, sycl_queue=None, arrays=()):
    """
    The state of the queue of `device` or `sycl_queue`, or else of the first
    array of `arrays`, or else of the default queue.

    """

    if device is None and sycl_queue is None:
        for x in arrays:
            if dpnp.is_supported_array_type(x):
                sycl_queue = x.sycl_queue
                break
    sycl_queue = dpnp.get_normalized_queue_device(
        device=device, sycl_queue=sycl_queue
    )
    with _random_states_lock:
        rs = _random_states.get(sycl_queue)
        if rs is None:
            rs = RandomState(sycl_queue=sycl_queue)
            _random_states[sycl_queue] = rs
        return rs


@_params_doc
def seed(seed=None, *, device=None, sycl_queue=None):
    # pylint: disable=redefined-outer-name
    """
    Reseed the state of a queue.

    For full documentation refer to :obj:`numpy.random.seed`.

    Parameters
    ----------
    %(queue)s

    """
    _get_random_state(device, sycl_queue).seed(seed)


@_params_doc
def get_state(legacy=True, *, device=None, sycl_queue=None):
    """
    Return the state of a queue.

    For full documentation refer to :obj:`numpy.random.get_state`.

    Parameters
    ----------
    %(queue)s

    Limitations
    -----------
    See :meth:`dpnp.random.RandomState.get_state`.

    """
    return _get_random_state(device, sycl_queue).get_state(legacy)


@_params_doc
def set_state(state, *, device=None, sycl_queue=None):
    """
    Set the state of a queue.

    For full documentation refer to :obj:`numpy.random.set_state`.

    Parameters
    ----------
    %(queue)s

    Limitations
    -----------
    See :meth:`dpnp.random.RandomState.set_state`.

    """
    _get_random_state(device, sycl_queue).set_state(state)


@_params_doc
def beta(
    a,
    b,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a Beta distribution.

    For full documentation refer to :obj:`numpy.random.beta`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.beta`

    """
    rs = _get_random_state(device, sycl_queue, (a, b))
    return rs.beta(a, b, size, dtype, usm_type=usm_type)


@_params_doc
def binomial(
    n, p, size=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draw samples from a binomial distribution.

    For full documentation refer to :obj:`numpy.random.binomial`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.binomial`

    """
    rs = _get_random_state(device, sycl_queue, (n, p))
    return rs.binomial(n, p, size, usm_type=usm_type)


@_params_doc
def bytes(length, *, device=None, sycl_queue=None):
    """
    Return random bytes.

    For full documentation refer to :obj:`numpy.random.bytes`.

    Parameters
    ----------
    %(queue)s

    """
    return _get_random_state(device, sycl_queue).bytes(length)


@_params_doc
def chisquare(
    df,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a chi-square distribution.

    For full documentation refer to :obj:`numpy.random.chisquare`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.chisquare`

    """
    rs = _get_random_state(device, sycl_queue, (df,))
    return rs.chisquare(df, size, dtype, usm_type=usm_type)


@_params_doc
def choice(
    a,
    size=None,
    replace=True,
    p=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Generate a random sample from a given 1-D array.

    For full documentation refer to :obj:`numpy.random.choice`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.choice`

    """
    rs = _get_random_state(device, sycl_queue, (a, p))
    return rs.choice(a, size, replace, p, usm_type=usm_type)


@_params_doc
def dirichlet(
    alpha,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from the Dirichlet distribution.

    For full documentation refer to :obj:`numpy.random.dirichlet`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.dirichlet`

    """
    rs = _get_random_state(device, sycl_queue, (alpha,))
    return rs.dirichlet(alpha, size, dtype, usm_type=usm_type)


@_params_doc
def exponential(
    scale=1.0,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from an exponential distribution.

    For full documentation refer to :obj:`numpy.random.exponential`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.exponential`

    """
    rs = _get_random_state(device, sycl_queue, (scale,))
    return rs.exponential(scale, size, dtype, usm_type=usm_type)


@_params_doc
def f(
    dfnum,
    dfden,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from an F distribution.

    For full documentation refer to :obj:`numpy.random.f`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.f`

    """
    rs = _get_random_state(device, sycl_queue, (dfnum, dfden))
    return rs.f(dfnum, dfden, size, dtype, usm_type=usm_type)


@_params_doc
def gamma(
    shape,
    scale=1.0,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a Gamma distribution.

    For full documentation refer to :obj:`numpy.random.gamma`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.gamma`

    """
    rs = _get_random_state(device, sycl_queue, (shape, scale))
    return rs.gamma(shape, scale, size, dtype, usm_type=usm_type)


@_params_doc
def geometric(p, size=None, *, device=None, usm_type="device", sycl_queue=None):
    """
    Draw samples from the geometric distribution.

    For full documentation refer to :obj:`numpy.random.geometric`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.geometric`

    """
    rs = _get_random_state(device, sycl_queue, (p,))
    return rs.geometric(p, size, usm_type=usm_type)


@_params_doc
def gumbel(
    loc=0.0,
    scale=1.0,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a Gumbel distribution.

    For full documentation refer to :obj:`numpy.random.gumbel`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.gumbel`

    """
    rs = _get_random_state(device, sycl_queue, (loc, scale))
    return rs.gumbel(loc, scale, size, dtype, usm_type=usm_type)


@_params_doc
def hypergeometric(
    ngood,
    nbad,
    nsample,
    size=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a hypergeometric distribution.

    For full documentation refer to :obj:`numpy.random.hypergeometric`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.hypergeometric`

    """
    rs = _get_random_state(device, sycl_queue, (ngood, nbad, nsample))
    return rs.hypergeometric(ngood, nbad, nsample, size, usm_type=usm_type)


@_params_doc
def laplace(
    loc=0.0,
    scale=1.0,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from the Laplace or double exponential distribution.

    For full documentation refer to :obj:`numpy.random.laplace`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.laplace`

    """
    rs = _get_random_state(device, sycl_queue, (loc, scale))
    return rs.laplace(loc, scale, size, dtype, usm_type=usm_type)


@_params_doc
def logistic(
    loc=0.0,
    scale=1.0,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a logistic distribution.

    For full documentation refer to :obj:`numpy.random.logistic`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.logistic`

    """
    rs = _get_random_state(device, sycl_queue, (loc, scale))
    return rs.logistic(loc, scale, size, dtype, usm_type=usm_type)


@_params_doc
def lognormal(
    mean=0.0,
    sigma=1.0,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a log-normal distribution.

    For full documentation refer to :obj:`numpy.random.lognormal`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.lognormal`

    """
    rs = _get_random_state(device, sycl_queue, (mean, sigma))
    return rs.lognormal(mean, sigma, size, dtype, usm_type=usm_type)


@_params_doc
def logseries(p, size=None, *, device=None, usm_type="device", sycl_queue=None):
    """
    Draw samples from a logarithmic series distribution.

    For full documentation refer to :obj:`numpy.random.logseries`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.logseries`

    """
    rs = _get_random_state(device, sycl_queue, (p,))
    return rs.logseries(p, size, usm_type=usm_type)


@_params_doc
def multinomial(
    n, pvals, size=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draw samples from a multinomial distribution.

    For full documentation refer to :obj:`numpy.random.multinomial`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.multinomial`

    """
    rs = _get_random_state(device, sycl_queue, (n, pvals))
    return rs.multinomial(n, pvals, size, usm_type=usm_type)


@_params_doc
def multivariate_normal(
    mean,
    cov,
    size=None,
    check_valid="warn",
    tol=1e-8,
    *,
    dtype=None,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw random samples from a multivariate normal distribution.

    For full documentation refer to :obj:`numpy.random.multivariate_normal`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.multivariate_normal`

    """
    rs = _get_random_state(device, sycl_queue, (mean, cov))
    return rs.multivariate_normal(
        mean, cov, size, check_valid, tol, dtype=dtype, usm_type=usm_type
    )


@_params_doc
def negative_binomial(
    n, p, size=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draw samples from a negative binomial distribution.

    For full documentation refer to :obj:`numpy.random.negative_binomial`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.negative_binomial`

    """
    rs = _get_random_state(device, sycl_queue, (n, p))
    return rs.negative_binomial(n, p, size, usm_type=usm_type)


@_params_doc
def noncentral_chisquare(
    df,
    nonc,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a noncentral chi-square distribution.

    For full documentation refer to :obj:`numpy.random.noncentral_chisquare`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.noncentral_chisquare`

    """
    rs = _get_random_state(device, sycl_queue, (df, nonc))
    return rs.noncentral_chisquare(df, nonc, size, dtype, usm_type=usm_type)


@_params_doc
def noncentral_f(
    dfnum,
    dfden,
    nonc,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from the noncentral F distribution.

    For full documentation refer to :obj:`numpy.random.noncentral_f`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.noncentral_f`

    """
    rs = _get_random_state(device, sycl_queue, (dfnum, dfden, nonc))
    return rs.noncentral_f(dfnum, dfden, nonc, size, dtype, usm_type=usm_type)


@_params_doc
def normal(
    loc=0.0,
    scale=1.0,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw random samples from a normal (Gaussian) distribution.

    For full documentation refer to :obj:`numpy.random.normal`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.normal`

    """
    rs = _get_random_state(device, sycl_queue, (loc, scale))
    return rs.normal(loc, scale, size, dtype, usm_type=usm_type)


@_params_doc
def pareto(
    a, size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draw samples from a Pareto II or Lomax distribution with specified
    shape.

    For full documentation refer to :obj:`numpy.random.pareto`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.pareto`

    """
    rs = _get_random_state(device, sycl_queue, (a,))
    return rs.pareto(a, size, dtype, usm_type=usm_type)


@_params_doc
def permutation(x, *, device=None, sycl_queue=None):
    """
    Randomly permute a sequence, or return a permuted range.

    For full documentation refer to :obj:`numpy.random.permutation`.

    Parameters
    ----------
    %(queue)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.permutation`

    """
    return _get_random_state(device, sycl_queue, (x,)).permutation(x)


@_params_doc
def poisson(
    lam=1.0, size=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draw samples from a Poisson distribution.

    For full documentation refer to :obj:`numpy.random.poisson`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.poisson`

    """
    rs = _get_random_state(device, sycl_queue, (lam,))
    return rs.poisson(lam, size, usm_type=usm_type)


@_params_doc
def power(
    a, size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draws samples in [0, 1] from a power distribution with positive
    exponent a - 1.

    For full documentation refer to :obj:`numpy.random.power`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.power`

    """
    rs = _get_random_state(device, sycl_queue, (a,))
    return rs.power(a, size, dtype, usm_type=usm_type)


@_params_doc
def rand(*args, dtype=None, device=None, usm_type="device", sycl_queue=None):
    """
    Random values in a given shape, from a uniform distribution over [0,
    1).

    For full documentation refer to :obj:`numpy.random.rand`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.rand`

    """
    rs = _get_random_state(device, sycl_queue)
    return rs.random_sample(args or None, dtype, usm_type=usm_type)


@_params_doc
def randint(
    low,
    high=None,
    size=None,
    dtype=int,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Return random integers from `low` (inclusive) to `high` (exclusive).

    For full documentation refer to :obj:`numpy.random.randint`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.randint`

    """
    rs = _get_random_state(device, sycl_queue, (low, high))
    return rs.randint(low, high, size, dtype, usm_type=usm_type)


@_params_doc
def randn(*args, dtype=None, device=None, usm_type="device", sycl_queue=None):
    """
    Return a sample (or samples) from the standard normal distribution.

    For full documentation refer to :obj:`numpy.random.randn`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.randn`

    """
    rs = _get_random_state(device, sycl_queue)
    return rs.standard_normal(args or None, dtype, usm_type=usm_type)


@_params_doc
def random(
    size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Return random floats in the half-open interval [0.0, 1.0).

    For full documentation refer to :obj:`numpy.random.random`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.random_sample`

    """
    rs = _get_random_state(device, sycl_queue)
    return rs.random_sample(size, dtype, usm_type=usm_type)


@_params_doc
def random_integers(
    low,
    high=None,
    size=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Random integers between `low` and `high`, inclusive.

    For full documentation refer to :obj:`numpy.random.random_integers`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.random_integers`

    """
    rs = _get_random_state(device, sycl_queue, (low, high))
    with_warning = rs.random_integers
    return with_warning(low, high, size, usm_type=usm_type)


@_params_doc
def random_sample(
    size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Return random floats in the half-open interval [0.0, 1.0).

    For full documentation refer to :obj:`numpy.random.random_sample`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.random_sample`

    """
    rs = _get_random_state(device, sycl_queue)
    return rs.random_sample(size, dtype, usm_type=usm_type)


@_params_doc
def ranf(
    size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Return random floats in the half-open interval [0.0, 1.0). An alias of
    :obj:`dpnp.random.random_sample`.

    For full documentation refer to :obj:`numpy.random.ranf`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.random_sample`

    """
    rs = _get_random_state(device, sycl_queue)
    return rs.random_sample(size, dtype, usm_type=usm_type)


@_params_doc
def rayleigh(
    scale=1.0,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a Rayleigh distribution.

    For full documentation refer to :obj:`numpy.random.rayleigh`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.rayleigh`

    """
    rs = _get_random_state(device, sycl_queue, (scale,))
    return rs.rayleigh(scale, size, dtype, usm_type=usm_type)


@_params_doc
def sample(
    size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Return random floats in the half-open interval [0.0, 1.0). An alias of
    :obj:`dpnp.random.random_sample`.

    For full documentation refer to :obj:`numpy.random.sample`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.random_sample`

    """
    rs = _get_random_state(device, sycl_queue)
    return rs.random_sample(size, dtype, usm_type=usm_type)


def shuffle(x):
    """
    Modify an array in place by shuffling its contents along the first axis.

    For full documentation refer to :obj:`numpy.random.shuffle`.

    The state of the queue of `x` is used.

    See Also
    --------
    :obj:`dpnp.random.RandomState.shuffle`

    """
    _get_random_state(arrays=(x,)).shuffle(x)


@_params_doc
def standard_cauchy(
    size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draw samples from a standard Cauchy distribution with mode = 0.

    For full documentation refer to :obj:`numpy.random.standard_cauchy`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.standard_cauchy`

    """
    rs = _get_random_state(device, sycl_queue, ())
    return rs.standard_cauchy(size, dtype, usm_type=usm_type)


@_params_doc
def standard_exponential(
    size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draw samples from the standard exponential distribution.

    For full documentation refer to :obj:`numpy.random.standard_exponential`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.standard_exponential`

    """
    rs = _get_random_state(device, sycl_queue, ())
    return rs.standard_exponential(size, dtype, usm_type=usm_type)


@_params_doc
def standard_gamma(
    shape,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a standard Gamma distribution.

    For full documentation refer to :obj:`numpy.random.standard_gamma`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.standard_gamma`

    """
    rs = _get_random_state(device, sycl_queue, (shape,))
    return rs.standard_gamma(shape, size, dtype, usm_type=usm_type)


@_params_doc
def standard_normal(
    size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draw samples from a standard Normal distribution (mean=0, stdev=1).

    For full documentation refer to :obj:`numpy.random.standard_normal`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.standard_normal`

    """
    rs = _get_random_state(device, sycl_queue, ())
    return rs.standard_normal(size, dtype, usm_type=usm_type)


@_params_doc
def standard_t(
    df,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a standard Student's t distribution with `df` degrees
    of freedom.

    For full documentation refer to :obj:`numpy.random.standard_t`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.standard_t`

    """
    rs = _get_random_state(device, sycl_queue, (df,))
    return rs.standard_t(df, size, dtype, usm_type=usm_type)


@_params_doc
def triangular(
    left,
    mode,
    right,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from the triangular distribution over the interval [left,
    right].

    For full documentation refer to :obj:`numpy.random.triangular`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.triangular`

    """
    rs = _get_random_state(device, sycl_queue, (left, mode, right))
    return rs.triangular(left, mode, right, size, dtype, usm_type=usm_type)


@_params_doc
def uniform(
    low=0.0,
    high=1.0,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a uniform distribution over [low, high).

    For full documentation refer to :obj:`numpy.random.uniform`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.uniform`

    """
    rs = _get_random_state(device, sycl_queue, (low, high))
    return rs.uniform(low, high, size, dtype, usm_type=usm_type)


@_params_doc
def vonmises(
    mu,
    kappa,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a von Mises distribution.

    For full documentation refer to :obj:`numpy.random.vonmises`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.vonmises`

    """
    rs = _get_random_state(device, sycl_queue, (mu, kappa))
    return rs.vonmises(mu, kappa, size, dtype, usm_type=usm_type)


@_params_doc
def wald(
    mean,
    scale,
    size=None,
    dtype=None,
    *,
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Draw samples from a Wald, or inverse Gaussian, distribution.

    For full documentation refer to :obj:`numpy.random.wald`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.wald`

    """
    rs = _get_random_state(device, sycl_queue, (mean, scale))
    return rs.wald(mean, scale, size, dtype, usm_type=usm_type)


@_params_doc
def weibull(
    a, size=None, dtype=None, *, device=None, usm_type="device", sycl_queue=None
):
    """
    Draw samples from a Weibull distribution.

    For full documentation refer to :obj:`numpy.random.weibull`.

    Parameters
    ----------
    %(dtype)s
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.weibull`

    """
    rs = _get_random_state(device, sycl_queue, (a,))
    return rs.weibull(a, size, dtype, usm_type=usm_type)


@_params_doc
def zipf(a, size=None, *, device=None, usm_type="device", sycl_queue=None):
    """
    Draw samples from a Zipf distribution.

    For full documentation refer to :obj:`numpy.random.zipf`.

    Parameters
    ----------
    %(queue)s
    %(usm)s

    See Also
    --------
    :obj:`dpnp.random.RandomState.zipf`

    """
    rs = _get_random_state(device, sycl_queue, (a,))
    return rs.zipf(a, size, usm_type=usm_type)
