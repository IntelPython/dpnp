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

import dpnp.config as config
from dpnp.backend cimport *
from dpnp.dparray cimport dparray
from dpnp.dpnp_utils cimport *


ctypedef void(*fptr_mkl_rng_gaussian_1out_t)(void *, size_t)
ctypedef void(*fptr_mkl_rng_uniform_1out_t)(void *, long, long, size_t, void *)


cdef dparray dpnp_randn(dims):
    """
    Return a random matrix with data from the "standard normal" distribution.

    `randn` generates a matrix filled with random floats sampled from a
    univariate "normal" (Gaussian) distribution of mean 0 and variance 1.

    """

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_GAUSSIAN, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(dims, dtype=result_type)

    cdef fptr_mkl_rng_gaussian_1out_t func = <fptr_mkl_rng_gaussian_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), result.size)

    return result


cdef dparray dpnp_random(dims, void * engine):
    """
    Create an array of the given shape and populate it
    with random samples from a uniform distribution over [0, 1).

    """
    cdef long low = 0
    cdef long high = 1

    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(numpy.float64)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_UNIFORM, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(dims, dtype=result_type)

    cdef fptr_mkl_rng_uniform_1out_t func = <fptr_mkl_rng_uniform_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), low, high, result.size, engine)

    return result


cdef dparray dpnp_uniform(long low, long high, size, void * engine, dtype=numpy.int32):
    """
    Return a random matrix with data from the uniform distribution.

    Generates a matrix filled with random numbers sampled from a
    uniform distribution of the certain left (low) and right (high)
    bounds.

    """
    # convert string type names (dparray.dtype) to C enum DPNPFuncType
    cdef DPNPFuncType param1_type = dpnp_dtype_to_DPNPFuncType(dtype)

    # get the FPTR data structure
    cdef DPNPFuncData kernel_data = get_dpnp_function_ptr(DPNP_FN_UNIFORM, param1_type, param1_type)

    result_type = dpnp_DPNPFuncType_to_dtype( < size_t > kernel_data.return_type)
    # ceate result array with type given by FPTR data
    cdef dparray result = dparray(size, dtype=result_type)

    cdef fptr_mkl_rng_uniform_1out_t func = <fptr_mkl_rng_uniform_1out_t > kernel_data.ptr
    # call FPTR function
    func(result.get_data(), low, high, result.size, engine)

    return result

cdef class RandomState:
    """
    TODO:
    description
    """
    cdef size_t seed_
    cdef void * rng_engine

    def __init__(self, seed=None):
        # TODO:
        #self.bit_generator = None
        #self.seed = 1
        self.seed_ = 1
        self.rng_engine = rng_engine_init(self.seed_)

    def __repr__(self):
        return self.__str__() + ' at 0x{:X}'.format(id(self))

    def __str__(self):
        _str = self.__class__.__name__
        return _str

#    # Pickling support:
#    def __getstate__(self):
#        return self.get_state(legacy=False)
#
#    def __setstate__(self, state):
#        self.set_state(state)
#
#    def __reduce__(self):
#        state = self.get_state(legacy=False)
#
#    cdef _reset_gauss(self):
#        self._aug_state.has_gauss = 0
#        self._aug_state.gauss = 0.0

    def seed(self, seed=None):
        """
        seed(self, seed=None)
        Reseed a legacy MT19937 BitGenerator
        Notes
        -----
        This is a convenience, legacy function.
        The best practice is to **not** reseed a BitGenerator, rather to
        recreate a new one. This method is here for legacy reasons.
        """
        pass

    def get_state(self, legacy=True):
        """
        get_state()
        Return a tuple representing the internal state of the generator.
        """
        pass

    def set_state(self, state):
        """
        set_state(state)
        Set the internal state of the generator from a tuple.
        """
        pass

    def rand(self, d0, *dn):
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
                checker_throw_value_error("randint", "type(dim)", type(dim), int)

        return dpnp_random(dims, self.rng_engine)

    def randf(self, size):
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
                checker_throw_value_error("randint", "type(dim)", type(dim), int)

        return dpnp_random(size, self.rng_engine)

    def randint(self, low, high=None, size=None, dtype=int):
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
            checker_throw_type_error("randint", dtype)

        return dpnp_uniform(low, high, size, self.rng_engine, _dtype)

    def randn(self, d0, *dn):
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
                checker_throw_value_error("randint", "type(dim)", type(dim), int)

        return dpnp_randn(dims)

    def random(self, size):
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
                checker_throw_value_error("randint", "type(dim)", type(dim), int)

        return dpnp_random(size, self.rng_engine)

    def random_integers(self, low, high=None, size=None):
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

        return self.randint(low, int(high) + 1, size=size)

    def random_sample(self, size):
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
                checker_throw_value_error("randint", "type(dim)", type(dim), int)

        return dpnp_random(size, self.rng_engine)


_rand = RandomState()

rand = _rand.rand
# TODO:
# update randf to f or randf
randf = _rand.randf
randint = _rand.randint
randn = _rand.randn
random = _rand.random
random_integers = _rand.random_integers
random_sample = _rand.random_sample
# TODO
# seed = _rand.seed
# set_state = _rand.set_state


# TODO:
# update ~~ + __init__

#def sample(size):
#    """
#    Return random floats in the half-open interval [0.0, 1.0).
#    This is an alias of random_sample.
#    Parameters
#    ----------
#    size : Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn.
#    Returns
#    -------
#    out : Array of random floats of shape size.
#    See Also
#    --------
#    random
#    """
#
#    if (use_origin_backend(size)):
#        return numpy.random.sample(size)
#
#    for dim in size:
#        if not isinstance(dim, int):
#            checker_throw_value_error("randint", "type(dim)", type(dim), int)
#
#    return dpnp_random(size, rng_engine)
#
#
#def uniform(low=0.0, high=1.0, size=None):
#    """
#    uniform(low=0.0, high=1.0, size=None)
#    Draw samples from a uniform distribution.
#    Samples are uniformly distributed over the half-open interval
#    ``[low, high)`` (includes low, but excludes high).  In other words,
#    any value within the given interval is equally likely to be drawn
#    by `uniform`.
#    Parameters
#    ----------
#    low : float, optional
#        Lower boundary of the output interval.  All values generated will be
#        greater than or equal to low.  The default value is 0.
#    high : float
#        Upper boundary of the output interval.  All values generated will be
#        less than high.  The default value is 1.0.
#    size : int or tuple of ints, optional
#        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
#        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
#        a single value is returned if ``low`` and ``high`` are both scalars.
#    Returns
#    -------
#    out : array or scalar
#        Drawn samples from the parameterized uniform distribution.
#    See Also
#    --------
#    random : Floats uniformly distributed over ``[0, 1)``.
#    """
#
#    if (use_origin_backend(low)):
#        return numpy.random.uniform(low, high, size)
#
#    if size is None:
#        size = 1
#
#    if low == high:
#        # TODO:
#        # currently dparray.full is not implemented
#        # return dpnp.dparray.dparray.full(size, low, dtype=numpy.float64)
#        message = "`low` equal to `high`, should return an array, filled with `low` value."
#        message += "  Currently not supported. See: numpy.full TODO"
#        checker_throw_runtime_error("uniform", message)
#    elif low > high:
#        low, high = high, low
#
#    return dpnp_uniform(low, high, size, rng_engine, dtype=numpy.float64)
