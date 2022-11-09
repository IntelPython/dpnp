# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2022, Intel Corporation
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
Module Intel NumPy RandomState

Set of functions to implement NumPy random module API

    .. seealso:: :obj:`numpy.random.RandomState`

"""


import numpy
import dpctl.utils as dpu

import dpnp
from dpnp.dpnp_utils.dpnp_algo_utils import (
    call_origin,
    use_origin_backend
)
from dpnp.random.dpnp_algo_random import MT19937


__all__ = [
    'RandomState'
]


class RandomState:
    """
    A container for the Mersenne Twister pseudo-random number generator.

    For full documentation refer to :obj:`numpy.random.RandomState`.
    """

    def __init__(self, seed=None, device=None, sycl_queue=None):
        self._seed = 1 if seed is None else seed
        self._sycl_queue = dpnp.get_normalized_queue_device(device=device, sycl_queue=sycl_queue)

        self._def_float_type = dpnp.float32
        if self._sycl_queue.get_sycl_device().has_aspect_fp64:
            self._def_float_type = dpnp.float64

        self._random_state = MT19937(self._seed, self._sycl_queue)
        self._fallback_random_state = call_origin(numpy.random.RandomState, seed)


    def get_state(self):
        """
        Return an internal state of the generator.

        For full documentation refer to :obj:`numpy.random.RandomState.get_state`.
        """
        return self._random_state


    def get_sycl_queue(self):
        """
        Return a sycl queue used from the container.
        """
        return self._sycl_queue


    def normal(self, loc=0.0, scale=1.0, size=None, dtype=None, usm_type="device"):
        """
        Draw random samples from a normal (Gaussian) distribution.

        For full documentation refer to :obj:`numpy.random.RandomState.normal`.

        Limitations
        -----------
        Parameters ``loc`` and ``scale`` are supported as scalar.
        Otherwise, :obj:`numpy.random.RandomState.normal(loc, scale, size)` samples are drawn.

        Parameter ``dtype`` is supported only for :obj:`dpnp.float32`, :obj:`dpnp.float64` or `None`.
        If ``dtype`` is None (default), :obj:`dpnp.float64` type will be used if device supports it
        or :obj:`dpnp.float32` otherwise.
        Output array data type is the same as ``dtype``.

        Examples
        --------
        Draw samples from the distribution:
        >>> s = dpnp.random.RandomState().normal(loc=3.7, scale=2.5, size=(2, 4))
        >>> print(s)
        [[ 1.58997253 -0.84288406  2.33836967  4.16394577]
         [ 4.40882036  5.39295758  6.48927254  6.74921661]]

        See Also
        --------
        :obj:`dpnp.random.RandomState.randn`
        :obj:`dpnp.random.RandomState.standard_normal`

        """

        if not use_origin_backend():
            if not dpnp.isscalar(loc):
                pass
            elif not dpnp.isscalar(scale):
                pass
            else:
                min_double = numpy.finfo('double').min
                max_double = numpy.finfo('double').max
                if (loc >= max_double or loc <= min_double) and dpnp.isfinite(loc):
                    raise OverflowError(f"Range of loc={loc} exceeds valid bounds")

                if (scale >= max_double) and dpnp.isfinite(scale):
                    raise OverflowError(f"Range of scale={scale} exceeds valid bounds")
                # # scale = -0.0 is cosidered as negative
                elif scale < 0 or scale == 0 and numpy.signbit(scale):
                    raise ValueError(f"scale={scale}, but must be non-negative.")

                if dtype is None:
                    dtype = self._def_float_type
                elif not dtype in (dpnp.float32, dpnp.float64):
                    raise TypeError(f"dtype={dtype} is unsupported.")

                dpu.validate_usm_type(usm_type=usm_type, allow_none=False)
                return self._random_state.normal(loc=loc,
                                                 scale=scale,
                                                 size=size,
                                                 dtype=dtype,
                                                 usm_type=usm_type).get_pyobj()

        return call_origin(self._fallback_random_state.normal,
                           loc=loc, scale=scale, size=size, sycl_queue=self._sycl_queue)


    def rand(self, *args, usm_type="device"):
        """
        Draw random values in a given shape.

        Create an array of the given shape and populate it with random samples
        from a uniform distribution over [0, 1).

        For full documentation refer to :obj:`numpy.random.RandomState.rand`.

        Limitations
        -----------
        Output array data type is :obj:`dpnp.float64`.

        Examples
        --------
        >>> s = dpnp.random.RandomState().rand(5, 2)
        >>> print(s)
        [[0.13436424 0.56920387]
         [0.84743374 0.80226506]
         [0.76377462 0.06310682]
         [0.25506903 0.1179187 ]
         [0.49543509 0.76096244]]

        See Also
        --------
        :obj:`dpnp.random.RandomState.random_sample`
        :obj:`dpnp.random.RandomState.uniform`

        """

        if len(args) == 0:
            return self.random_sample(usm_type=usm_type)
        else:
            return self.random_sample(size=args, usm_type=usm_type)


    def randint(self, low, high=None, size=None, dtype=int, usm_type="device"):
        """
        Draw random integers from low (inclusive) to high (exclusive).

        Return random integers from the “discrete uniform” distribution of the specified type
        in the “half-open” interval [low, high).

        For full documentation refer to :obj:`numpy.random.RandomState.randint`.

        Limitations
        -----------
        Parameters ``low`` and ``high`` are supported only as scalar.
        Parameter ``dtype`` is supported only as `int`.
        Otherwise, :obj:`numpy.random.randint(low, high, size, dtype)` samples are drawn.

        Examples
        --------
        >>> s = dpnp.random.RandomState().randint(2, size=10)
        >>> print(s)
        [0 1 1 1 1 0 0 0 0 1]

        See Also
        --------
        :obj:`dpnp.random.RandomState.random_integers` : similar to `randint`, only for the closed
            interval [`low`, `high`], and 1 is the lowest value if `high` is omitted.

        """

        if not use_origin_backend(low):
            if not dpnp.isscalar(low):
                pass
            elif not dpnp.isscalar(high):
                pass
            else:
                _dtype = dpnp.int32 if dtype is int else dpnp.dtype(dtype)
                if _dtype != dpnp.int32:
                    pass
                else:
                    if high is None:
                        high = low
                        low = 0

                    min_int = numpy.iinfo('int32').min
                    max_int = numpy.iinfo('int32').max
                    if not dpnp.isfinite(low) or low > max_int or low < min_int:
                        raise OverflowError(f"Range of low={low} exceeds valid bounds")
                    elif not dpnp.isfinite(high) or high > max_int or high < min_int:
                        raise OverflowError(f"Range of high={high} exceeds valid bounds")

                    low = int(low)
                    high = int(high)
                    if low >= high:
                        raise ValueError(f"low={low} >= high={high}")

                    return self.uniform(low=low,
                                        high=high,
                                        size=size,
                                        dtype=_dtype,
                                        usm_type=usm_type)

        return call_origin(self._fallback_random_state.randint,
                           low=low, high=high, size=size, dtype=dtype, sycl_queue=self._sycl_queue)


    def randn(self, *args, usm_type="device"):
        """
        Return a sample (or samples) from the "standard normal" distribution.

        For full documentation refer to :obj:`numpy.random.RandomState.randn`.

        Limitations
        -----------
        Output array data type is :obj:`dpnp.float64` if device supports it
        or :obj:`dpnp.float32` otherwise.

        Examples
        --------
        >>> s = dpnp.random.RandomState().randn()
        >>> print(s)
        -0.84401099

        Two-by-four array of samples from the normal distribution with
        mean 3 and standard deviation 2.5:

        >>> s = dpnp.random.RandomState().randn(2, 4)
        >>> print(s)
        [[ 0.88997253 -1.54288406  1.63836967  3.46394577]
         [ 3.70882036  4.69295758  5.78927254  6.04921661]]

        See Also
        --------
        :obj:`dpnp.random.normal`
        :obj:`dpnp.random.standard_normal`

        """

        if len(args) == 0:
            return self.standard_normal(usm_type=usm_type)
        return self.standard_normal(size=args,
                                    usm_type=usm_type)


    def random_sample(self, size=None, usm_type="device"):
        """
        Draw random floats in the half-open interval [0.0, 1.0).

        Results are from the “continuous uniform” distribution over the interval.

        For full documentation refer to :obj:`numpy.random.RandomState.random_sample`.

        Limitations
        -----------
        Output array data type is :obj:`dpnp.float64` if device supports it
        or :obj:`dpnp.float32` otherwise.

        Examples
        --------
        >>> s = dpnp.random.RandomState().random_sample(size=(4,))
        >>> print(s)
        [0.13436424 0.56920387 0.84743374 0.80226506]

        See Also
        --------
        :obj:`dpnp.random.RandomState.rand`
        :obj:`dpnp.random.RandomState.uniform`

        """

        return self.uniform(low=0.0,
                            high=1.0,
                            size=size,
                            dtype=None,
                            usm_type=usm_type)


    def standard_normal(self, size=None, usm_type="device"):
        """
        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        For full documentation refer to :obj:`numpy.random.RandomState.standard_normal`.

        Limitations
        -----------
        Output array data type is :obj:`dpnp.float64` if device supports it
        or :obj:`dpnp.float32` otherwise.

        Examples
        --------
        Draw samples from the distribution:
        >>> s = dpnp.random.RandomState().standard_normal(size=(3, 5))
        >>> print(s)
        [[-0.84401099 -1.81715362 -0.54465213  0.18557831  0.28352814]
         [ 0.67718303  1.11570901  1.21968665 -1.18236388  0.08156915]
         [ 0.21941987 -1.24544512  0.63522211 -0.673174    0.        ]]

        See Also
        --------
        :obj:`dpnp.random.RandomState.normal`
        :obj:`dpnp.random.RandomState.randn`

        """

        return self.normal(loc=0.0,
                           scale=1.0,
                           size=size,
                           dtype=None,
                           usm_type=usm_type)


    def uniform(self, low=0.0, high=1.0, size=None, dtype=None, usm_type="device"):
        """
        Draw samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high).
        In other words, any value within the given interval is equally likely to be drawn by uniform.

        For full documentation refer to :obj:`numpy.random.RandomState.uniform`.

        Limitations
        -----------
        Parameters ``low`` and ``high`` are supported as scalar.
        Otherwise, :obj:`numpy.random.uniform(low, high, size)` samples are drawn.
        Parameter ``dtype`` is supported only for :obj:`dpnp.int32`, :obj:`dpnp.float32`, :obj:`dpnp.float64` or `None`.
        If ``dtype`` is None (default), :obj:`dpnp.float64` type will be used if device supports it
        or :obj:`dpnp.float32` otherwise.
        Output array data type is the same as ``dtype``.

        Examples
        --------
        Draw samples from the distribution:
        >>> low, high = 1.23, 10.54    # low and high
        >>> s = dpnp.random.RandomState().uniform(low, high, 5)
        >>> print(s)
        [2.48093112 6.52928804 9.1196081  8.6990877  8.34074171]

        See Also
        --------
        :obj:`dpnp.random.RandomState.randint` : Discrete uniform distribution, yielding integers.
        :obj:`dpnp.random.RandomState.random_integers` : Discrete uniform distribution over the closed interval ``[low, high]``.
        :obj:`dpnp.random.RandomState.random_sample` : Floats uniformly distributed over ``[0, 1)``.
        :obj:`dpnp.random.RandomState.random` : Alias for :obj:`dpnp.random.RandomState.random_sample`.
        :obj:`dpnp.random.RandomState.rand` : Convenience function that accepts dimensions as input, e.g.,
            ``rand(2, 2)`` would generate a 2-by-2 array of floats, uniformly distributed over ``[0, 1)``.

        """

        if not use_origin_backend():
            if not dpnp.isscalar(low):
                pass
            elif not dpnp.isscalar(high):
                pass
            else:
                min_double = numpy.finfo('double').min
                max_double = numpy.finfo('double').max
                if not dpnp.isfinite(low) or low >= max_double or low <= min_double:
                    raise OverflowError(f"Range of low={low} exceeds valid bounds")
                elif not dpnp.isfinite(high) or high >= max_double or high <= min_double:
                    raise OverflowError(f"Range of high={high} exceeds valid bounds")

                if low > high:
                    low, high = high, low

                if dtype is None:
                    dtype = self._def_float_type
                elif not dtype in (dpnp.int32, dpnp.float32, dpnp.float64):
                    raise TypeError(f"dtype={dtype} is unsupported.")

                dpu.validate_usm_type(usm_type, allow_none=False)
                return self._random_state.uniform(low=low,
                                                  high=high,
                                                  size=size,
                                                  dtype=dtype,
                                                  usm_type=usm_type).get_pyobj()

        return call_origin(self._fallback_random_state.uniform,
                           low=low, high=high, size=size, sycl_queue=self._sycl_queue)
