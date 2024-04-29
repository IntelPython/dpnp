# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2024, Intel Corporation
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
Interface of histogram-related DPNP functions

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

import operator
import warnings

import dpctl.utils as dpu
import numpy

import dpnp

__all__ = [
    "histogram",
]

# range is a keyword argument to many functions, so save the builtin so they can
# use it.
_range = range


def _ravel_check_a_and_weights(a, weights):
    """Check input `a` and `weights` arrays, and ravel both."""

    # ensure that `a` array has supported type
    dpnp.check_supported_arrays_type(a)
    usm_type = a.usm_type

    # ensure that the array is a "subtractable" dtype
    if a.dtype == dpnp.bool:
        warnings.warn(
            f"Converting input from {a.dtype} to {numpy.uint8} "
            "for compatibility.",
            RuntimeWarning,
            stacklevel=3,
        )
        a = a.astype(numpy.uint8)

    if weights is not None:
        # check that `weights` array has supported type
        dpnp.check_supported_arrays_type(weights)
        usm_type = dpu.get_coerced_usm_type([usm_type, weights.usm_type])

        # check that arrays have the same allocation queue
        if dpu.get_execution_queue([a.sycl_queue, weights.sycl_queue]) is None:
            raise ValueError(
                "a and weights must be allocated on the same SYCL queue"
            )

        if weights.shape != a.shape:
            raise ValueError("weights should have the same shape as a.")
        weights = weights.ravel()
    a = a.ravel()
    return a, weights, usm_type


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument.

    """

    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError("max must be larger than min in range parameter.")

        if not (numpy.isfinite(first_edge) and numpy.isfinite(last_edge)):
            raise ValueError(
                f"supplied range of [{first_edge}, {last_edge}] is not finite"
            )

    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1

    else:
        first_edge, last_edge = a.min(), a.max()
        if not (dpnp.isfinite(first_edge) and dpnp.isfinite(last_edge)):
            raise ValueError(
                f"autodetected range of [{first_edge}, {last_edge}] "
                "is not finite"
            )

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _get_bin_edges(a, bins, range, usm_type):
    """Computes the bins used internally by `histogram`."""

    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None
    sycl_queue = a.sycl_queue

    if isinstance(bins, str):
        # TODO: implement support of string bins
        raise NotImplementedError("only integer and array bins are implemented")

    if numpy.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError as e:
            raise TypeError("`bins` must be an integer or an array") from e
        if n_equal_bins < 1:
            raise ValueError("`bins` must be positive, when an integer")

        first_edge, last_edge = _get_outer_edges(a, range)

    elif numpy.ndim(bins) == 1:
        if dpnp.is_supported_array_type(bins):
            if dpu.get_execution_queue([a.sycl_queue, bins.sycl_queue]) is None:
                raise ValueError(
                    "a and bins must be allocated on the same SYCL queue"
                )

            bin_edges = bins
        else:
            bin_edges = dpnp.asarray(
                bins, sycl_queue=sycl_queue, usm_type=usm_type
            )

        if dpnp.any(bin_edges[:-1] > bin_edges[1:]):
            raise ValueError(
                "`bins` must increase monotonically, when an array"
            )

    else:
        raise ValueError("`bins` must be 1d, when an array")

    if n_equal_bins is not None:
        # numpy's gh-10322 means that type resolution rules are dependent on
        # array shapes. To avoid this causing problems, we pick a type now and
        # stick with it throughout.
        bin_type = dpnp.result_type(first_edge, last_edge, a)
        if dpnp.issubdtype(bin_type, dpnp.integer):
            bin_type = dpnp.result_type(
                bin_type, dpnp.default_float_type(sycl_queue=sycl_queue), a
            )

        # bin edges must be computed
        bin_edges = dpnp.linspace(
            first_edge,
            last_edge,
            n_equal_bins + 1,
            endpoint=True,
            dtype=bin_type,
            sycl_queue=sycl_queue,
            usm_type=usm_type,
        )
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    return bin_edges, None


def _search_sorted_inclusive(a, v):
    """
    Like :obj:`dpnp.searchsorted`, but where the last item in `v` is placed
    on the right.
    In the context of a histogram, this makes the last bin edge inclusive

    """

    return dpnp.concatenate(
        (a.searchsorted(v[:-1], "left"), a.searchsorted(v[-1:], "right"))
    )


def histogram(a, bins=10, range=None, density=None, weights=None):
    """
    Compute the histogram of a data set.

    For full documentation refer to :obj:`numpy.histogram`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input data. The histogram is computed over the flattened array.
    bins : {int, dpnp.ndarray, usm_ndarray, sequence of scalars}, optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (``10``, by default).
        If `bins` is a sequence, it defines a monotonically increasing array
        of bin edges, including the rightmost edge, allowing for non-uniform
        bin widths.
        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by :obj:`dpnp.histogram_bin_edges`.
    range : {2-tuple of float}, optional
        The lower and upper range of the bins. If not provided, range is simply
        ``(a.min(), a.max())``. Values outside the range are ignored. The first
        element of the range must be less than or equal to the second. `range`
        affects the automatic bin computation as well. While bin width is
        computed to be optimal based on the actual data within `range`, the bin
        count will fill the entire range including portions containing no data.
    weights : {dpnp.ndarray, usm_ndarray}, optional
        An array of weights, of the same shape as `a`. Each value in `a` only
        contributes its associated weight towards the bin count (instead of 1).
        If `density` is ``True``, the weights are normalized, so that the
        integral of the density over the range remains ``1``.
        Please note that the ``dtype`` of `weights` will also become the
        ``dtype`` of the returned accumulator (`hist`), so it must be large
        enough to hold accumulated values as well.
    density : {bool}, optional
        If ``False``, the result will contain the number of samples in each bin.
        If ``True``, the result is the value of the probability *density*
        function at the bin, normalized such that the *integral* over the range
        is ``1``. Note that the sum of the histogram values will not be equal
        to ``1`` unless bins of unity width are chosen; it is not a probability
        *mass* function.

    Returns
    -------
    hist : {dpnp.ndarray}
        The values of the histogram. See `density` and `weights` for a
        description of the possible semantics. If `weights` are given,
        ``hist.dtype`` will be taken from `weights`.
    bin_edges : {dpnp.ndarray of floating data type}
        Return the bin edges ``(length(hist) + 1)``.

    See Also
    --------
    :obj:`dpnp.histogramdd` : Compute the multidimensional histogram.
    :obj:`dpnp.bincount` : Count number of occurrences of each value in array
                           of non-negative integers.
    :obj:`dpnp.searchsorted` : Find indices where elements should be inserted
                               to maintain order.
    :obj:`dpnp.digitize` : Return the indices of the bins to which each value
                           in input array belongs.
    :obj:`dpnp.histogram_bin_edges` : Return only the edges of the bins used
                                      by the obj:`dpnp.histogram` function.

    Examples
    --------
    >>> import dpnp as np
    >>> np.histogram(np.array([1, 2, 1]), bins=[0, 1, 2, 3])
    (array([0, 2, 1]), array([0, 1, 2, 3]))
    >>> np.histogram(np.arange(4), bins=np.arange(5), density=True)
    (array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4]))
    >>> np.histogram(np.array([[1, 2, 1], [1, 0, 1]]), bins=[0, 1, 2, 3])
    (array([1, 4, 1]), array([0, 1, 2, 3]))

    >>> a = np.arange(5)
    >>> hist, bin_edges = np.histogram(a, density=True)
    >>> hist
    array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])
    >>> hist.sum()
    array(2.5)
    >>> np.sum(hist * np.diff(bin_edges))
    array(1.)

    """

    a, weights, usm_type = _ravel_check_a_and_weights(a, weights)

    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, usm_type)

    # Histogram is an integer or a float array depending on the weights.
    if weights is None:
        ntype = dpnp.dtype(dpnp.intp)
    else:
        ntype = weights.dtype

    # The fast path uses bincount, but that only works for certain types
    # of weight
    # simple_weights = (
    #     weights is None or
    #     dpnp.can_cast(weights.dtype, dpnp.double) or
    #     dpnp.can_cast(weights.dtype, complex)
    # )
    # TODO: implement a fast path
    simple_weights = False

    if uniform_bins is not None and simple_weights:
        # TODO: implement fast algorithm for equal bins
        pass
    else:
        # Compute via cumulative histogram
        if weights is None:
            sa = dpnp.sort(a)
            cum_n = _search_sorted_inclusive(sa, bin_edges)
        else:
            zero = dpnp.zeros(
                1, dtype=ntype, sycl_queue=a.sycl_queue, usm_type=usm_type
            )
            sorting_index = dpnp.argsort(a)
            sa = a[sorting_index]
            sw = weights[sorting_index]
            cw = dpnp.concatenate((zero, sw.cumsum(dtype=ntype)))
            bin_index = _search_sorted_inclusive(sa, bin_edges)
            cum_n = cw[bin_index]

        n = dpnp.diff(cum_n)

    if density:
        db = dpnp.diff(bin_edges).astype(dpnp.default_float_type())
        return n / db / n.sum(), bin_edges

    return n, bin_edges
