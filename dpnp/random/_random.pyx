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

import numpy
from dpnp.dparray cimport dparray
from dpnp.backend cimport *

import dpnp.config as config
from dpnp.dpnp_utils import use_origin_backend


cpdef dparray dpnp_randn(dims):
    """
    Return a random matrix with data from the "standard normal" distribution.

    `randn` generates a matrix filled with random floats sampled from a
    univariate "normal" (Gaussian) distribution of mean 0 and variance 1.

    """

    cdef dparray result = dparray(dims, dtype=numpy.float64)
    cdef size_t result_size = result.size

    mkl_rng_gaussian[double](result.get_data(), result_size)

    return result


cpdef dparray dpnp_random(dims):
    """
    Create an array of the given shape and populate it
    with random samples from a uniform distribution over [0, 1).

    """

    cdef dparray result = dparray(dims, dtype=numpy.float64)
    cdef size_t result_size = result.size

    mkl_rng_uniform[double](result.get_data(), result_size)

    return result


cpdef dparray dpnp_uniform(long low, long high, size, dtype=numpy.int32):
    """
    Return a random matrix with data from the uniform distribution.

    Generates a matrix filled with random numbers sampled from a
    uniform distribution of the certain left (low) and right (high)
    bounds.

    """

    cdef dparray result = dparray(size, dtype=dtype)
    cdef size_t result_size = result.size

    # TODO:
    # supported dtype int32
    if dtype == numpy.int32:
        mkl_rng_uniform_mt19937[int](result.get_data(), low, high, result_size)
    elif dtype == numpy.float32:
        mkl_rng_uniform_mt19937[float](result.get_data(), low, high, result_size)
    elif dtype == numpy.float64:
        mkl_rng_uniform_mt19937[double](result.get_data(), low, high, result_size)

    return result


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

    if (use_origin_backend(d0)):
        return numpy.random.rand(d0, *dn)

    dims = tuple([d0, *dn])

    for dim in dims:
        if not isinstance(dim, int):
            raise TypeError(f"Intel NumPy random.rand(): Unsupported dim={type(dim)}")

    return dpnp_random(dims)


def randf(size):
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

    if (use_origin_backend(size)):
        return numpy.random.ranf(size)

    for dim in size:
        if not isinstance(dim, int):
            raise TypeError(f"Intel NumPy random.randf(): Unsupported dim={type(dim)}")

    return dpnp_random(size)


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

    if (use_origin_backend(low)):
        return numpy.random.randint(low, high, size, dtype)

    if size is None:
        size = 1
    elif isinstance(size, tuple):
        for dim in size:
            if not isinstance(dim, int):
                raise TypeError(f"Intel NumPy random.sample(): Unsupported dim={type(dim)}")
    elif not isinstance(size, int):
        raise ValueError('Unsupported type %r for `size`' % type(size))

    if high is None:
        high = low
        low = 0

    low = int(low)
    high = int(high)

    if low >= high:
        raise ValueError('low >= high')

    _dtype = numpy.dtype(dtype)

    # TODO:
    # supported only int32
    # or just raise error when dtype != numpy.int32
    if _dtype == numpy.int32 or _dtype == numpy.int64:
        _dtype = numpy.int32
    else:
        raise TypeError('Unsupported dtype %r for randint' % dtype)

    return dpnp_uniform(low, high, size, _dtype)


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

    if (use_origin_backend(d0)):
        return numpy.random.randn(d0, *dn)

    dims = tuple([d0, *dn])

    for dim in dims:
        if not isinstance(dim, int):
            raise TypeError(f"Intel NumPy random.randn(): Unsupported dim={type(dim)}")

    return dpnp_randn(dims)


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

    if (use_origin_backend(size)):
        return numpy.random.random(size)

    for dim in size:
        if not isinstance(dim, int):
            raise TypeError(f"Intel NumPy random.random(): Unsupported dim={type(dim)}")

    return dpnp_random(size)


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

    if (use_origin_backend(low)):
        return numpy.random.random_integers(low, high, size)

    if high is None:
        high = low
        low = 1

    return randint(low, int(high) + 1, size=size)


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

    if (use_origin_backend(size)):
        return numpy.random.random_sample(size)

    for dim in size:
        if not isinstance(dim, int):
            raise TypeError(f"Intel NumPy random.random_sample(): Unsupported dim={type(dim)}")

    return dpnp_random(size)


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

    if (use_origin_backend(size)):
        return numpy.random.sample(size)

    for dim in size:
        if not isinstance(dim, int):
            raise TypeError(f"Intel NumPy random.sample(): Unsupported dim={type(dim)}")

    return dpnp_random(size)


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

    if (use_origin_backend(low)):
        return numpy.random.uniform(low, high, size)

    if size is None:
        size = 1

    if low == high:
        # TODO:
        # currently dparray.full is not implemented
        # return dpnp.dparray.dparray.full(size, low, dtype=numpy.float64)
        raise ValueError('`low` equal to `high`, should return an array, filled with `low` value.'
                         '  Currently not supported. See: numpy.full TODO')
    elif low > high:
        low, high = high, low

    return dpnp_uniform(low, high, size, dtype=numpy.float64)
