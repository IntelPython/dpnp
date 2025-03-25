# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2024-2025, Intel Corporation
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
from collections.abc import Iterable

import dpctl.utils as dpu
import numpy

import dpnp

# pylint: disable=no-name-in-module
import dpnp.backend.extensions.statistics._statistics_impl as statistics_ext
from dpnp.dpnp_utils.dpnp_utils_common import (
    result_type_for_device,
    to_supported_dtypes,
)

# pylint: disable=no-name-in-module
from .dpnp_utils import get_usm_allocations

__all__ = [
    "bincount",
    "digitize",
    "histogram",
    "histogram_bin_edges",
    "histogram2d",
    "histogramdd",
]

# range is a keyword argument to many functions, so save the builtin so they can
# use it.
_range = range


def _align_dtypes(a_dtype, bins_dtype, ntype, supported_types, device):
    a_bin_dtype = result_type_for_device([a_dtype, bins_dtype], device)

    # histogram implementation doesn't support uint64 as histogram type
    # we can use int64 instead. Result would be correct even in case of overflow
    if ntype == numpy.uint64:
        ntype = dpnp.int64

    return to_supported_dtypes([a_bin_dtype, ntype], supported_types, device)


def _ravel_check_a_and_weights(a, weights):
    """
    Check input `a` and `weights` arrays, and ravel both.
    The returned array have :class:`dpnp.ndarray` type always.

    """

    # ensure that `a` array has supported type
    dpnp.check_supported_arrays_type(a)
    usm_type = a.usm_type

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
        weights = dpnp.ravel(weights)

    a = dpnp.ravel(a)
    return a, weights, usm_type


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument.

    """

    def _is_finite(a):
        if dpnp.is_supported_array_type(a):
            return dpnp.isfinite(a)

        return numpy.isfinite(a)

    if range is not None:
        if len(range) != 2:
            raise ValueError("range argument must consist of 2 elements.")

        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError("max must be larger than min in range parameter.")

        if not (_is_finite(first_edge) and _is_finite(last_edge)):
            raise ValueError(
                f"supplied range of [{first_edge}, {last_edge}] is not finite"
            )

    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1

    else:
        first_edge, last_edge = a.min(), a.max()
        if not (_is_finite(first_edge) and _is_finite(last_edge)):
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
        # pylint: disable=possibly-used-before-assignment
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


def _bincount_validate(x, weights, minlength):
    dpnp.check_supported_arrays_type(x)
    if x.ndim > 1:
        raise ValueError("object too deep for desired array")

    if x.ndim < 1:
        raise ValueError("object of too small depth for desired array")

    if not dpnp.issubdtype(x.dtype, dpnp.integer) and not dpnp.issubdtype(
        x.dtype, dpnp.bool
    ):
        raise TypeError("x must be an integer array")

    if weights is not None:
        dpnp.check_supported_arrays_type(weights)
        if x.shape != weights.shape:
            raise ValueError("The weights and x don't have the same length.")

        if not (
            dpnp.issubdtype(weights.dtype, dpnp.integer)
            or dpnp.issubdtype(weights.dtype, dpnp.floating)
            or dpnp.issubdtype(weights.dtype, dpnp.bool)
        ):
            raise ValueError(
                f"Weights must be integer or float. Got {weights.dtype}"
            )

    if minlength is None:
        raise TypeError("use 0 instead of None for minlength")

    minlength = int(minlength)
    if minlength < 0:
        raise ValueError("minlength must be non-negative")


def _bincount_run_native(
    x_casted, weights_casted, minlength, n_dtype, usm_type
):
    queue = x_casted.sycl_queue

    max_v = dpnp.max(x_casted)
    min_v = dpnp.min(x_casted)

    if min_v < 0:
        raise ValueError("x argument must have no negative arguments")

    size = max(int(max_v) + 1, minlength)

    # bincount implementation uses atomics, but atomics doesn't work with
    # host usm memory
    n_usm_type = "device" if usm_type == "host" else usm_type

    # bincount implementation requires output array to be filled with zeros
    n_casted = dpnp.zeros(
        size, dtype=n_dtype, usm_type=n_usm_type, sycl_queue=queue
    )

    _manager = dpu.SequentialOrderManager[queue]

    x_usm = dpnp.get_usm_ndarray(x_casted)
    weights_usm = (
        dpnp.get_usm_ndarray(weights_casted)
        if weights_casted is not None
        else None
    )
    n_usm = dpnp.get_usm_ndarray(n_casted)

    mem_ev, bc_ev = statistics_ext.bincount(
        x_usm,
        min_v.item(),
        max_v.item(),
        weights_usm,
        n_usm,
        depends=_manager.submitted_events,
    )

    _manager.add_event_pair(mem_ev, bc_ev)

    return n_casted


def bincount(x, weights=None, minlength=0):
    """
    bincount(x, /, weights=None, minlength=0)

    Count number of occurrences of each value in array of non-negative ints.

    For full documentation refer to :obj:`numpy.bincount`.

    Warning
    -------
    This function synchronizes in order to calculate binning edges.
    This may harm performance in some applications.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input 1-dimensional array with non-negative integer values.
    weights : {None, dpnp.ndarray, usm_ndarray}, optional
        Weights, array of the same shape as `x`.

        Default: ``None``
    minlength : int, optional
        A minimum number of bins for the output array.

        Default: ``0``

    Returns
    -------
    out : dpnp.ndarray of ints
        The result of binning the input array.
        The length of `out` is equal to ``dpnp.max(x) + 1``.

    See Also
    --------
    :obj:`dpnp.histogram` : Compute the histogram of a data set.
    :obj:`dpnp.digitize` : Return the indices of the bins to which each value
    :obj:`dpnp.unique` : Find the unique elements of an array.

    Examples
    --------
    >>> import dpnp as np
    >>> np.bincount(np.arange(5))
    array([1, 1, 1, 1, 1])
    >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
    array([1, 3, 1, 1, 0, 0, 0, 1])

    >>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
    >>> np.bincount(x).size == np.amax(x) + 1
    array(True)

    The input array needs to be of integer dtype, otherwise a
    TypeError is raised:

    >>> np.bincount(np.arange(5, dtype=np.float32))
    Traceback (most recent call last):
      ...
    TypeError: x must be an integer array

    A possible use of :obj:`dpnp.bincount` is to perform sums over
    variable-size chunks of an array, using the `weights` keyword.

    >>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6], dtype=np.float32) # weights
    >>> x = np.array([0, 1, 1, 2, 2, 2])
    >>> np.bincount(x, weights=w)
    array([0.3, 0.7, 1.1], dtype=float32)

    """

    _bincount_validate(x, weights, minlength)

    x, weights, usm_type = _ravel_check_a_and_weights(x, weights)

    queue = x.sycl_queue
    device = queue.sycl_device

    if weights is None:
        ntype = dpnp.dtype(dpnp.intp)
    else:
        # unlike in case of histogram result type is integer if no weights
        # provided and float if weights are provided even if weights are integer
        ntype = dpnp.default_float_type(sycl_queue=queue)

    weights_casted = None

    supported_types = statistics_ext.bincount_dtypes()
    x_casted_dtype, ntype_casted = _align_dtypes(
        x.dtype, x.dtype, ntype, supported_types, device
    )

    if x_casted_dtype is None or ntype_casted is None:  # pragma: no cover
        raise ValueError(
            f"Input types ({x.dtype}, {ntype}) are not supported, "
            "and the inputs could not be coerced to any supported types"
        )

    x_casted = dpnp.asarray(x, dtype=x_casted_dtype, order="C")

    if weights is not None:
        weights_casted = dpnp.asarray(weights, dtype=ntype_casted, order="C")

    n_casted = _bincount_run_native(
        x_casted, weights_casted, minlength, ntype_casted, usm_type
    )

    return dpnp.asarray(n_casted, dtype=ntype, usm_type=usm_type)


def digitize(x, bins, right=False):
    """
    Return the indices of the bins to which each value in input array belongs.

    For full documentation refer to :obj:`numpy.digitize`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array to be binned.
    bins : {dpnp.ndarray, usm_ndarray}
        Array of bins. It has to be 1-dimensional and monotonic
        increasing or decreasing.
    right : bool, optional
        Indicates whether the intervals include the right or the left bin edge.

        Default: ``False``.

    Returns
    -------
    indices : dpnp.ndarray
        Array of indices with the same shape as `x`.

    Notes
    -----
    This will not raise an exception when the input array is
    not monotonic.

    See Also
    --------
    :obj:`dpnp.bincount` : Count number of occurrences of each value in array
                           of non-negative integers.
    :obj:`dpnp.histogram` : Compute the histogram of a data set.
    :obj:`dpnp.unique` : Find the unique elements of an array.
    :obj:`dpnp.searchsorted` : Find indices where elements should be inserted
                               to maintain order.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> inds = np.digitize(x, bins)
    >>> inds
    array([1, 4, 3, 2])
    >>> for n in range(x.size):
    ...     print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])
    ...
    0. <= 0.2 < 1.
    4. <= 6.4 < 10.
    2.5 <= 3. < 4.
    1. <= 1.6 < 2.5

    >>> x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
    >>> bins = np.array([0, 5, 10, 15, 20])
    >>> np.digitize(x, bins, right=True)
    array([1, 2, 3, 4, 4])
    >>> np.digitize(x, bins, right=False)
    array([1, 3, 3, 4, 5])

    """

    dpnp.check_supported_arrays_type(x, bins)

    if dpnp.issubdtype(x.dtype, dpnp.complexfloating):
        raise TypeError("x may not be complex")

    if bins.ndim > 1:
        raise ValueError("object too deep for desired array")
    if bins.ndim < 1:
        raise ValueError("object of too small depth for desired array")

    # This is backwards because the arguments below are swapped
    side = "left" if right else "right"

    # Check if bins are monotonically increasing.
    # If bins is empty, the array is considered to be increasing.
    # If all bins are NaN, the array is considered to be decreasing.
    if bins.size == 0:
        bins_increasing = True
    else:
        bins_increasing = bins[0] <= bins[-1] or (
            not dpnp.isnan(bins[0]) and dpnp.isnan(bins[-1])
        )

    if bins_increasing:
        # Use dpnp.searchsorted directly if bins are increasing
        return dpnp.searchsorted(bins, x, side=side)

    # Reverse bins and adjust indices if bins are decreasing
    return bins.size - dpnp.searchsorted(bins[::-1], x, side=side)


def histogram(a, bins=10, range=None, density=None, weights=None):
    """
    Compute the histogram of a data set.

    For full documentation refer to :obj:`numpy.histogram`.

    Warning
    -------
    This function may synchronize in order to check a monotonically increasing
    array of bin edges. This may harm performance in some applications.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input data. The histogram is computed over the flattened array.
    bins : {int, dpnp.ndarray, usm_ndarray, sequence of scalars}, optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range.
        If `bins` is a sequence, it defines a monotonically increasing array
        of bin edges, including the rightmost edge, allowing for non-uniform
        bin widths.

        Default: ``10``.
    range : {None, 2-tuple of float}, optional
        The lower and upper range of the bins. If not provided, range is simply
        ``(a.min(), a.max())``. Values outside the range are ignored. The first
        element of the range must be less than or equal to the second. `range`
        affects the automatic bin computation as well. While bin width is
        computed to be optimal based on the actual data within `range`, the bin
        count will fill the entire range including portions containing no data.

        Default: ``None``.
    density : {None, bool}, optional
        If ``False`` or ``None``, the result will contain the number of samples
        in each bin. If ``True``, the result is the value of the probability
        *density* function at the bin, normalized such that the *integral* over
        the range is ``1``. Note that the sum of the histogram values will not
        be equal to ``1`` unless bins of unity width are chosen; it is not
        a probability *mass* function.

        Default: ``None``.
    weights : {None, dpnp.ndarray, usm_ndarray}, optional
        An array of weights, of the same shape as `a`. Each value in `a` only
        contributes its associated weight towards the bin count (instead of 1).
        If `density` is ``True``, the weights are normalized, so that the
        integral of the density over the range remains ``1``.
        Please note that the ``dtype`` of `weights` will also become the
        ``dtype`` of the returned accumulator (`hist`), so it must be large
        enough to hold accumulated values as well.

        Default: ``None``.

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

    bin_edges, _ = _get_bin_edges(a, bins, range, usm_type)

    # Histogram is an integer or a float array depending on the weights.
    if weights is None:
        ntype = dpnp.dtype(dpnp.intp)
    else:
        ntype = weights.dtype

    queue = a.sycl_queue
    device = queue.sycl_device

    supported_types = statistics_ext.histogram_dtypes()
    a_bin_dtype, hist_dtype = _align_dtypes(
        a.dtype, bin_edges.dtype, ntype, supported_types, device
    )

    if a_bin_dtype is None or hist_dtype is None:  # pragma: no cover
        raise ValueError(
            f"Input types ({a.dtype}, {bin_edges.dtype}, {ntype}) "
            "are not supported, and the inputs could not be coerced to any "
            "supported types"
        )

    a_casted = dpnp.asarray(a, dtype=a_bin_dtype, order="C")
    bin_edges_casted = dpnp.asarray(bin_edges, dtype=a_bin_dtype, order="C")
    weights_casted = (
        dpnp.asarray(weights, dtype=hist_dtype, order="C")
        if weights is not None
        else None
    )

    # histogram implementation uses atomics, but atomics doesn't work with
    # host usm memory
    n_usm_type = "device" if usm_type == "host" else usm_type

    # histogram implementation requires output array to be filled with zeros
    n_casted = dpnp.zeros(
        bin_edges.size - 1,
        dtype=hist_dtype,
        sycl_queue=a.sycl_queue,
        usm_type=n_usm_type,
    )

    _manager = dpu.SequentialOrderManager[queue]

    a_usm = dpnp.get_usm_ndarray(a_casted)
    bins_usm = dpnp.get_usm_ndarray(bin_edges_casted)
    weights_usm = (
        dpnp.get_usm_ndarray(weights_casted)
        if weights_casted is not None
        else None
    )
    n_usm = dpnp.get_usm_ndarray(n_casted)

    mem_ev, ht_ev = statistics_ext.histogram(
        a_usm,
        bins_usm,
        weights_usm,
        n_usm,
        depends=_manager.submitted_events,
    )
    _manager.add_event_pair(mem_ev, ht_ev)

    n = dpnp.asarray(n_casted, dtype=ntype, usm_type=usm_type)

    if density:
        db = dpnp.astype(
            dpnp.diff(bin_edges), dpnp.default_float_type(sycl_queue=queue)
        )
        return n / db / dpnp.sum(n), bin_edges

    return n, bin_edges


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """
    Function to calculate only the edges of the bins used by the
    :obj:`dpnp.histogram` function.

    For full documentation refer to :obj:`numpy.histogram_bin_edges`.

    Warning
    -------
    This function may synchronize in order to check a monotonically increasing
    array of bin edges. This may harm performance in some applications.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input data. The histogram is computed over the flattened array.
    bins : {int, dpnp.ndarray, usm_ndarray, sequence of scalars}, optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range.
        If `bins` is a sequence, it defines the bin edges, including the
        rightmost edge, allowing for non-uniform bin widths.

        Default: ``10``.
    range : {None, 2-tuple of float}, optional
        The lower and upper range of the bins. If not provided, range is simply
        ``(a.min(), a.max())``. Values outside the range are ignored. The first
        element of the range must be less than or equal to the second. `range`
        affects the automatic bin computation as well. While bin width is
        computed to be optimal based on the actual data within `range`, the bin
        count will fill the entire range including portions containing no data.

        Default: ``None``.
    weights : {None, dpnp.ndarray, usm_ndarray}, optional
        An array of weights, of the same shape as `a`. Each value in `a` only
        contributes its associated weight towards the bin count (instead of 1).
        This is currently not used by any of the bin estimators, but may be in
        the future.

        Default: ``None``.

    Returns
    -------
    bin_edges : {dpnp.ndarray of floating data type}
        The edges to pass into :obj:`dpnp.histogram`.

    See Also
    --------
    :obj:`dpnp.histogram` : Compute the histogram of a data set.

    Examples
    --------
    >>> import dpnp as np
    >>> arr = np.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
    >>> np.histogram_bin_edges(arr, bins=2)
    array([0. , 2.5, 5. ])

    For consistency with histogram, an array of pre-computed bins is
    passed through unmodified:

    >>> np.histogram_bin_edges(arr, [1, 2])
    array([1, 2])

    This function allows one set of bins to be computed, and reused across
    multiple histograms:

    >>> shared_bins = np.histogram_bin_edges(arr, bins=5)
    >>> shared_bins
    array([0., 1., 2., 3., 4., 5.])

    >>> gid = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    >>> hist_0, _ = np.histogram(arr[gid == 0], bins=shared_bins)
    >>> hist_1, _ = np.histogram(arr[gid == 1], bins=shared_bins)

    >>> hist_0, hist_1
    (array([1, 1, 0, 1, 0]), array([2, 0, 1, 1, 2]))

    Which gives more easily comparable results than using separate bins for
    each histogram:

    >>> hist_0, bins_0 = np.histogram(arr[gid == 0], bins=3)
    >>> hist_1, bins_1 = np.histogram(arr[gid == 1], bins=4)
    >>> hist_0, hist_1
    (array([1, 1, 1]), array([2, 1, 1, 2]))
    >>> bins_0, bins_1
    (array([0., 1., 2., 3.]), array([0.  , 1.25, 2.5 , 3.75, 5.  ]))

    """

    a, weights, usm_type = _ravel_check_a_and_weights(a, weights)
    bin_edges, _ = _get_bin_edges(a, bins, range, usm_type)
    return bin_edges


def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    """
    Compute the bi-dimensional histogram of two data samples.

    For full documentation refer to :obj:`numpy.histogram2d`.

    Warning
    -------
    This function may synchronize in order to check a monotonically increasing
    array of bin edges. This may harm performance in some applications.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray} of shape (N,)
        An array containing the `x` coordinates of the points to be
        histogrammed.
    y : {dpnp.ndarray, usm_ndarray} of shape (N,)
        An array containing the `y` coordinates of the points to be
        histogrammed.
    bins : {int, dpnp.ndarray, usm_ndarray, [int, int], [array, array], \
        [int, array], [array, int]}, optional

        The bins specification:

        * If int, the number of bins for the two dimensions (nx=ny=bins).
        * If array, the bin edges for the two dimensions
          (x_edges=y_edges=bins).
        * If [int, int], the number of bins in each dimension
          (nx, ny = bins).
        * If [array, array], the bin edges in each dimension
          (x_edges, y_edges = bins).
        * A combination [int, array] or [array, int], where int
          is the number of bins and array is the bin edges.

        Default: ``10``.
    range : {None, dpnp.ndarray, usm_ndarray} of shape (2, 2), optional
        The leftmost and rightmost edges of the bins along each dimension
        If ``None`` the ranges are
        ``[[x.min(), x.max()], [y.min(), y.max()]]``. All values outside
        of this range will be considered outliers and not tallied in the
        histogram.

        Default: ``None``.
    density : {None, bool}, optional
        If ``False`` or ``None``, the default, returns the number of
        samples in each bin.
        If ``True``, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_volume``.

        Default: ``None``.
    weights : {None, dpnp.ndarray, usm_ndarray} of shape (N,), optional
        An array of values ``w_i`` weighing each sample ``(x_i, y_i)``.
        Weights are normalized to ``1`` if `density` is ``True``.
        If `density` is ``False``, the values of the returned histogram
        are equal to the sum of the weights belonging to the samples
        falling into each bin.
        If ``None`` all samples are assigned a weight of ``1``.

        Default: ``None``.
    Returns
    -------
    H : dpnp.ndarray of shape (nx, ny)
        The bi-dimensional histogram of samples `x` and `y`. Values in `x`
        are histogrammed along the first dimension and values in `y` are
        histogrammed along the second dimension.
    xedges : dpnp.ndarray of shape (nx+1,)
        The bin edges along the first dimension.
    yedges : dpnp.ndarray of shape (ny+1,)
        The bin edges along the second dimension.

    See Also
    --------
    :obj:`dpnp.histogram` : 1D histogram
    :obj:`dpnp.histogramdd` : Multidimensional histogram

    Notes
    -----
    When `density` is ``True``, then the returned histogram is the sample
    density, defined such that the sum over bins of the product
    ``bin_value * bin_area`` is 1.

    Please note that the histogram does not follow the Cartesian convention
    where `x` values are on the abscissa and `y` values on the ordinate
    axis. Rather, `x` is histogrammed along the first dimension of the
    array (vertical), and `y` along the second dimension of the array
    (horizontal). This ensures compatibility with `histogramdd`.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.random.randn(20).astype("float32")
    >>> y = np.random.randn(20).astype("float32")
    >>> hist, edges_x, edges_y = np.histogram2d(x, y, bins=(4, 3))
    >>> hist.shape
    (4, 3)
    >>> hist
    array([[1., 2., 0.],
           [0., 3., 1.],
           [1., 4., 1.],
           [1., 3., 3.]], dtype=float32)
    >>> edges_x.shape
    (5,)
    >>> edges_x
    array([-1.7516936 , -0.96109843, -0.17050326,  0.62009203,  1.4106871 ],
          dtype=float32)
    >>> edges_y.shape
    (4,)
    >>> edges_y
    array([-2.6604428 , -0.94615364,  0.76813555,  2.4824247 ], dtype=float32)

    Please note, that resulting values of histogram and edges may vary.

    """

    dpnp.check_supported_arrays_type(x, y)
    if weights is not None:
        dpnp.check_supported_arrays_type(weights)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(
            f"x and y must be 1-dimensional arrays."
            f"Got {x.ndim} and {y.ndim} respectively"
        )

    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length."
            f"Got {len(x)} and {len(y)} respectively"
        )

    usm_type, exec_q = get_usm_allocations([x, y, bins, range, weights])
    device = exec_q.sycl_device

    sample_dtype = result_type_for_device([x.dtype, y.dtype], device)

    # Unlike histogramdd histogram2d accepts 1d bins and
    # apply it to both dimensions
    # at the same moment two elements bins should be interpreted as
    # number of bins in each dimension and array-like bins with one element
    # is not allowed
    if isinstance(bins, Iterable) and len(bins) > 2:
        bins = [bins] * 2

    bins = _histdd_normalize_bins(bins, 2)
    bins_dtypes = [sample_dtype]
    bins_dtypes += [b.dtype for b in bins if hasattr(b, "dtype")]

    bins_dtype = result_type_for_device(bins_dtypes, device)
    hist_dtype = _histdd_hist_dtype(exec_q, weights)

    supported_types = statistics_ext.histogramdd_dtypes()

    sample_dtype, _ = _align_dtypes(
        sample_dtype, bins_dtype, hist_dtype, supported_types, device
    )

    sample = dpnp.empty_like(
        x, shape=x.shape + (2,), dtype=sample_dtype, usm_type=usm_type
    )
    sample[:, 0] = x
    sample[:, 1] = y

    hist, edges = histogramdd(
        sample, bins=bins, range=range, density=density, weights=weights
    )
    return hist, edges[0], edges[1]


def _histdd_validate_bins(bins):
    for i, b in enumerate(bins):
        if numpy.ndim(b) == 0:
            if b < 1:
                raise ValueError(
                    f"'bins[{i}' must be positive, when an integer"
                )
        elif numpy.ndim(b) == 1:
            # will check for monotonicity later
            pass
        else:
            raise ValueError(
                f"'bins[{i}]' must be either scalar or 1d array-like,"
                + f" but it is {type(b)}"
            )


def _histdd_normalize_bins(bins, ndims):
    if not isinstance(bins, Iterable):
        if not dpnp.issubdtype(type(bins), dpnp.integer):
            raise ValueError("'bins' must be an integer, when a scalar")

        bins = [bins] * ndims

    if len(bins) != ndims:
        raise ValueError(
            f"The dimension of bins ({len(bins)}) must be equal"
            + f" to the dimension of the sample ({ndims})."
        )

    _histdd_validate_bins(bins)

    return bins


def _histdd_normalize_range(range, ndims):
    if range is None:
        range = [None] * ndims

    if len(range) != ndims:
        raise ValueError(
            f"range argument length ({len(range)}) must match"
            + f" number of dimensions ({ndims})"
        )

    return range


def _histdd_make_edges(sample, bins, range, usm_type):
    bedges_list = []
    for i, (r, _bins) in enumerate(zip(range, bins)):
        bedges, _ = _get_bin_edges(sample[:, i], _bins, r, usm_type)
        bedges_list.append(bedges)

    return bedges_list


def _histdd_flatten_binedges(bedges_list, edges_count_list, dtype):
    total_edges_size = numpy.sum(edges_count_list)

    bin_edges_flat = dpnp.empty_like(
        bedges_list[0], shape=total_edges_size, dtype=dtype
    )

    offset = numpy.pad(numpy.cumsum(edges_count_list), (1, 0))
    bin_edges_view_list = []
    for start, end, bedges in zip(offset[:-1], offset[1:], bedges_list):
        edges_slice = bin_edges_flat[start:end]
        bin_edges_view_list.append(edges_slice)
        edges_slice[:] = bedges

    return bin_edges_flat, bin_edges_view_list


def _histdd_run_native(
    sample, weights, hist_dtype, bin_edges, edges_count_list, usm_type
):
    queue = sample.sycl_queue

    hist_shape = [ec - 1 for ec in edges_count_list]
    bin_edges_count = dpnp.asarray(
        edges_count_list, dtype=dpnp.int64, sycl_queue=queue
    )

    n_usm_type = "device" if usm_type == "host" else usm_type
    n = dpnp.zeros(
        shape=hist_shape,
        dtype=hist_dtype,
        sycl_queue=queue,
        usm_type=n_usm_type,
    )

    sample_usm = dpnp.get_usm_ndarray(sample)
    weights_usm = dpnp.get_usm_ndarray(weights) if weights is not None else None
    edges_usm = dpnp.get_usm_ndarray(bin_edges)
    edges_count_usm = dpnp.get_usm_ndarray(bin_edges_count)
    n_usm = dpnp.get_usm_ndarray(n)

    _manager = dpu.SequentialOrderManager[queue]

    mem_ev, hdd_ev = statistics_ext.histogramdd(
        sample_usm,
        edges_usm,
        edges_count_usm,
        weights_usm,
        n_usm,
        depends=_manager.submitted_events,
    )

    _manager.add_event_pair(mem_ev, hdd_ev)

    return n


def _histdd_hist_dtype(queue, weights):
    hist_dtype = dpnp.default_float_type(sycl_queue=queue)
    device = queue.sycl_device

    if weights is not None:
        # hist_dtype is either float or complex, so it is ok
        # to calculate it as result type between default_float and
        # weights.dtype
        hist_dtype = result_type_for_device([hist_dtype, weights.dtype], device)

    return hist_dtype


def _histdd_sample_dtype(queue, sample, bin_edges_list):
    device = queue.sycl_device

    dtypes_ = [bin_edges.dtype for bin_edges in bin_edges_list]
    dtypes_.append(sample.dtype)

    return result_type_for_device(dtypes_, device)


def _histdd_supported_dtypes(sample, bin_edges_list, weights):
    queue = sample.sycl_queue
    device = queue.sycl_device

    hist_dtype = _histdd_hist_dtype(queue, weights)
    sample_dtype = _histdd_sample_dtype(queue, sample, bin_edges_list)

    supported_types = statistics_ext.histogramdd_dtypes()

    # passing sample_dtype twice as we already
    # aligned sample_dtype and bins dtypes
    sample_dtype, hist_dtype = _align_dtypes(
        sample_dtype, sample_dtype, hist_dtype, supported_types, device
    )

    return sample_dtype, hist_dtype


def _histdd_extract_arrays(sample, weights, bins):
    all_arrays = [sample]
    if weights is not None:
        all_arrays.append(weights)

    if isinstance(bins, Iterable):
        all_arrays.extend([b for b in bins if dpnp.is_supported_array_type(b)])

    return all_arrays


def histogramdd(sample, bins=10, range=None, density=None, weights=None):
    """
    Compute the multidimensional histogram of some data.

    For full documentation refer to :obj:`numpy.histogramdd`.

    Warning
    -------
    This function may synchronize in order to check a monotonically increasing
    array of bin edges. This may harm performance in some applications.

    Parameters
    ----------
    sample : {dpnp.ndarray, usm_ndarray}
        Input (N, D)-shaped array to be histogrammed.
    bins : {sequence, int}, optional
        The bin specification:

        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

        Default: ``10``.
    range : {None, sequence}, optional
        A sequence of length D, each an optional (lower, upper) tuple giving
        the outer bin edges to be used if the edges are not given explicitly in
        `bins`.
        An entry of ``None`` in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        ``None`` is equivalent to passing a tuple of D ``None`` values.

        Default: ``None``.
    density : {None, bool}, optional
        If ``False`` or ``None``, the default, returns the number of
        samples in each bin.
        If ``True``, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_volume``.

        Default: ``None``.
    weights : {None, dpnp.ndarray, usm_ndarray}, optional
        An (N,)-shaped array of values `w_i` weighing each sample
        `(x_i, y_i, z_i, ...)`.
        Weights are normalized to ``1`` if density is ``True``.
        If density is ``False``, the values of the returned histogram
        are equal to the sum of the weights belonging to the samples
        falling into each bin.
        If ``None`` all samples are assigned a weight of ``1``.

        Default: ``None``.

    Returns
    -------
    H : dpnp.ndarray
        The multidimensional histogram of sample x. See density and weights
        for the different possible semantics.
    edges : list of {dpnp.ndarray or usm_ndarray}
        A list of D arrays describing the bin edges for each dimension.

    See Also
    --------
    :obj:`dpnp.histogram`: 1-D histogram
    :obj:`dpnp.histogram2d`: 2-D histogram

    Examples
    --------
    >>> import dpnp as np
    >>> r = np.random.normal(size=(100, 3))
    >>> H, edges = np.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)

    """

    dpnp.check_supported_arrays_type(sample)
    if weights is not None:
        dpnp.check_supported_arrays_type(weights)

    if sample.ndim < 2:
        sample = dpnp.reshape(sample, (sample.size, 1))
    elif sample.ndim > 2:
        raise ValueError("sample must have no more than 2 dimensions")

    ndim = sample.shape[1]

    _arrays = _histdd_extract_arrays(sample, weights, bins)
    usm_type, queue = get_usm_allocations(_arrays)

    bins = _histdd_normalize_bins(bins, ndim)
    range = _histdd_normalize_range(range, ndim)

    bin_edges_list = _histdd_make_edges(sample, bins, range, usm_type)
    sample_dtype, hist_dtype = _histdd_supported_dtypes(
        sample, bin_edges_list, weights
    )

    edges_count_list = [bin_edges.size for bin_edges in bin_edges_list]
    bin_edges_flat, bin_edges_view_list = _histdd_flatten_binedges(
        bin_edges_list, edges_count_list, sample_dtype
    )

    sample_ = dpnp.asarray(sample, dtype=sample_dtype, order="C")
    weights_ = (
        dpnp.asarray(weights, dtype=hist_dtype, order="C")
        if weights is not None
        else None
    )
    n = _histdd_run_native(
        sample_,
        weights_,
        hist_dtype,
        bin_edges_flat,
        edges_count_list,
        usm_type,
    )

    expexted_hist_dtype = _histdd_hist_dtype(queue, weights)
    n = dpnp.asarray(n, dtype=expexted_hist_dtype, usm_type=usm_type)

    if density:
        # calculate the probability density function
        s = n.sum()
        for i in _range(ndim):
            diff = dpnp.diff(bin_edges_view_list[i])
            shape = [1] * ndim
            shape[i] = diff.size
            n = n / dpnp.reshape(diff, shape=shape)
        n /= s

    for i, b in enumerate(bins):
        if dpnp.is_supported_array_type(b):
            bin_edges_view_list[i] = b

    return n, bin_edges_view_list
