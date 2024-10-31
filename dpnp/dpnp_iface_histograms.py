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
import dpnp.backend.extensions.sycl_ext._sycl_ext_impl as sycl_ext  # pylint: disable=C0301,E0611

__all__ = [
    "digitize",
    "histogram",
    "histogram_bin_edges",
]

# range is a keyword argument to many functions, so save the builtin so they can
# use it.
_range = range


def _ravel_check_a_and_weights(a, weights):
    """
    Check input `a` and `weights` arrays, and ravel both.
    The returned array have :class:`dpnp.ndarray` type always.

    """

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
        a = dpnp.astype(a, numpy.uint8)

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


def _search_sorted_inclusive(a, v):
    """
    Like :obj:`dpnp.searchsorted`, but where the last item in `v` is placed
    on the right.
    In the context of a histogram, this makes the last bin edge inclusive

    """

    return dpnp.concatenate(
        (a.searchsorted(v[:-1], "left"), a.searchsorted(v[-1:], "right"))
    )


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


def _find_supported_dtype(dt, supported):
    if dt in supported:
        return dt

    for t in supported:
        if dpnp.can_cast(dt, t):
            return t

    return None


def _result_type(dtype1, dtype2, has_fp64):
    rt = dpnp.result_type(dtype1, dtype2)
    if rt == dpnp.float64 and not has_fp64:
        return dpnp.float32

    if rt == dpnp.complex128 and not has_fp64:
        return dpnp.complex64

    return rt


def _align_dtypes(a_dtype, bins_dtype, ntype, has_fp64):
    a_bin_dtype = _result_type(a_dtype, bins_dtype, has_fp64)

    supported_types = (dpnp.float32, dpnp.int64, dpnp.uint64, dpnp.complex64)
    if has_fp64:
        supported_types += (dpnp.float64, dpnp.complex128)

    float_types = [dpnp.float32, dpnp.float64]
    complex_types = [dpnp.complex64, dpnp.complex128]

    a_bin_dtype = _find_supported_dtype(a_bin_dtype, supported_types)
    hist_dtype = _find_supported_dtype(ntype, supported_types)
    if hist_dtype == dpnp.uint64:
        hist_dtype = dpnp.int64

    if (a_bin_dtype in float_types and hist_dtype in float_types) or (
        a_bin_dtype in complex_types and hist_dtype in complex_types
    ):
        common_type = _result_type(a_bin_dtype, hist_dtype, has_fp64)
        a_bin_dtype = common_type
        hist_dtype = common_type

    if (a_bin_dtype in float_types and hist_dtype in complex_types) or (
        a_bin_dtype in complex_types and hist_dtype in float_types
    ):
        if a_bin_dtype == dpnp.complex128:
            hist_dtype = dpnp.float64
        elif a_bin_dtype == dpnp.float64:
            hist_dtype = dpnp.complex128

        if hist_dtype == dpnp.complex128:
            a_bin_dtype = dpnp.float64
        elif hist_dtype == dpnp.float64:
            a_bin_dtype = dpnp.complex128

    return a_bin_dtype, hist_dtype


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
    has_fp64 = queue.sycl_device.has_aspect_fp64

    a_bin_dtype, hist_dtype = _align_dtypes(
        a.dtype, bin_edges.dtype, ntype, has_fp64
    )

    if a_bin_dtype is None or hist_dtype is None:
        raise ValueError(
            f"function '{histogram}' does not support input types "
            f"({a.dtype}, {bin_edges.dtype}, {ntype}), "
            "and the inputs could not be coerced to any "
            "supported types"
        )

    a_casted = a.astype(a_bin_dtype, order="C", copy=False)
    bin_edges_casted = bin_edges.astype(a_bin_dtype, order="C", copy=False)
    weights_casted = (
        weights.astype(hist_dtype, order="C", copy=False)
        if weights is not None
        else None
    )

    n_usm_type = "device" if usm_type == "host" else usm_type
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

    ht_ev, mem_ev = sycl_ext.histogram(
        a_usm,
        bins_usm,
        weights_usm,
        n_usm,
        depends=_manager.submitted_events,
    )
    _manager.add_event_pair(ht_ev, mem_ev)

    if usm_type != n_usm_type:
        n = dpnp.asarray(n_casted, dtype=ntype, usm_type=usm_type)
    else:
        n = n_casted.astype(ntype, copy=False)

    if density:
        # pylint: disable=possibly-used-before-assignment
        db = dpnp.diff(bin_edges).astype(dpnp.default_float_type())
        return n / db / n.sum(), bin_edges

    return n, bin_edges


def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """
    Function to calculate only the edges of the bins used by the
    :obj:`dpnp.histogram` function.

    For full documentation refer to :obj:`numpy.histogram_bin_edges`.

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
