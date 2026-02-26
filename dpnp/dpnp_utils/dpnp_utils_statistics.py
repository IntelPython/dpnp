# *****************************************************************************
# Copyright (c) 2023, Intel Corporation
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

import warnings

import dpctl
import dpctl.tensor as dpt
import dpctl.utils as dpu
from dpctl.tensor._numpy_helper import normalize_axis_tuple
from dpctl.utils import ExecutionPlacementError

import dpnp
import dpnp.backend.extensions.statistics._statistics_impl as statistics_ext
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils.dpnp_utils_common import to_supported_dtypes

__all__ = ["dpnp_cov", "dpnp_median"]


def _calc_median(a, axis, out=None):
    """Compute the median of an array along a specified axis."""

    indexer = [slice(None)] * a.ndim
    index, remainder = divmod(a.shape[axis], 2)
    if remainder == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index + 1)
    else:
        indexer[axis] = slice(index - 1, index + 1)

    # Use `mean` in odd and even case to coerce data type and use `out` array
    res = dpnp.mean(a[tuple(indexer)], axis=axis, out=out)
    nan_mask = dpnp.isnan(a).any(axis=axis)
    if nan_mask.any():
        res[nan_mask] = dpnp.nan

    return res


def _calc_nanmedian(a, out=None):
    """Compute the median of an array along a specified axis, ignoring NaNs."""
    mask = dpnp.isnan(a)
    valid_counts = dpnp.sum(~mask, axis=-1)
    if out is None:
        res = dpnp.empty_like(valid_counts, dtype=a.dtype)
    else:
        dpnp.check_supported_arrays_type(out)
        exec_q = dpctl.utils.get_execution_queue((a.sycl_queue, out.sycl_queue))
        if exec_q is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )
        if out.shape != valid_counts.shape:
            raise ValueError(
                f"Output array of shape {valid_counts.shape} is needed, got {out.shape}."
            )
        res = out

    left = (valid_counts - 1) // 2
    right = valid_counts // 2

    left_data = dpnp.take_along_axis(a, left[..., None], axis=-1)
    right_data = dpnp.take_along_axis(a, right[..., None], axis=-1)
    res = dpnp.where(
        valid_counts[..., None] > 0, (left_data + right_data) / 2.0, dpnp.nan
    )

    if mask.all(axis=-1).any():
        warnings.warn("All-NaN slice encountered", RuntimeWarning, stacklevel=6)

    return dpnp.squeeze(res)


def _flatten_array_along_axes(a, axes_to_flatten, overwrite_input):
    """Flatten an array along a specific set of axes."""

    a_ndim = a.ndim
    axes_to_keep = tuple(
        axis for axis in range(a_ndim) if axis not in axes_to_flatten
    )

    # Move the axes_to_flatten to the end
    destination = list(range(len(axes_to_keep), a_ndim))
    a_moved = dpnp.moveaxis(a, axes_to_flatten, destination)
    new_shape = tuple(a.shape[axis] for axis in axes_to_keep) + (-1,)
    a_flatten = a_moved.reshape(new_shape)

    # Note that the output of a_flatten is not necessarily a view of the input
    # since `reshape` is used here. If this is the case, we can safely use
    # overwrite_input=True in calculating median
    if (
        dpnp.get_usm_ndarray(a)._pointer
        == dpnp.get_usm_ndarray(a_flatten)._pointer
    ):
        overwrite_input = overwrite_input
    else:
        overwrite_input = True

    return a_flatten, overwrite_input


def dpnp_cov(
    m, y=None, rowvar=True, ddof=1, dtype=None, fweights=None, aweights=None
):
    """
    Estimate a covariance matrix based on passed data.

    The implementation is done through existing dpnp functions.

    """

    # need to create a copy of input, since it will be modified in-place
    x = dpnp.array(m, ndmin=2, dtype=dtype)
    if not rowvar and m.ndim != 1:
        x = x.T

    if x.shape[0] == 0:
        return dpnp.empty_like(
            x, shape=(0, 0), dtype=dpnp.default_float_type(m.sycl_queue)
        )

    if y is not None:
        y_ndim = y.ndim
        y = dpnp.array(y, copy=None, ndmin=2, dtype=dtype)
        if not rowvar and y_ndim != 1:
            y = y.T
        x = dpnp.concatenate((x, y), axis=0)

    # get the product of frequencies and weights
    w = None
    if fweights is not None:
        if fweights.shape[0] != x.shape[1]:
            raise ValueError("incompatible numbers of samples and fweights")

        w = fweights

    if aweights is not None:
        if aweights.shape[0] != x.shape[1]:
            raise ValueError("incompatible numbers of samples and aweights")

        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = dpnp.average(x, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]

    # determine the normalization
    if w is None:
        fact = x.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * dpnp.sum(w * aweights) / w_sum

    if fact <= 0:
        warnings.warn(
            "Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2
        )
        fact = 0.0

    x -= avg[:, None]
    if w is None:
        x_t = x.T
    else:
        x_t = (x * w).T

    c = dpnp.dot(x, x_t.conj()) / fact
    return c.squeeze()


def native_median(a, ignore_nan):
    a = dpnp.reshape(a, a.size)
    device = a.sycl_device

    result_dtype = dpnp.default_float_type()
    if dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        result_dtype = a.dtype

    if a.size == 0:
        return dpnp.array(dpnp.nan, ndmin=1, dtype=result_dtype)
    elif a.size == 1:
        return dpnp.array(a[0], ndmin=1, dtype=result_dtype)

    supported_types = statistics_ext.kth_element_dtypes()
    supported_dtype = to_supported_dtypes(a.dtype, supported_types, device)

    if supported_dtype is None:  # pragma: no cover
        raise ValueError(
            f"function does not support input type "
            f"{a.dtype.name}, "
            "and the input could not be coerced to any "
            f"supported type. List of supported types: "
            f"{[st.name for st in supported_types]}"
        )

    a_casted = dpnp.asarray(a, dtype=supported_dtype, order="C")

    partitioned = dpnp.empty_like(a_casted)

    a_usm = dpnp.get_usm_ndarray(a_casted)
    partitioned_usm = dpnp.get_usm_ndarray(partitioned)

    _manager = dpu.SequentialOrderManager[a.sycl_queue]

    result = dpnp.empty_like(a, dtype=result_dtype, shape=1)

    nans = 0
    if ignore_nan:
        nans = dpnp.isnan(a_usm).sum()
    k = (a.shape[0] - 1 - nans) // 2

    found, buff_offset, elems_offset, num_elems, nan_count = (
        statistics_ext.kth_element(
            a_usm,
            partitioned_usm,
            k,
            depends=_manager.submitted_events,
        )
    )

    if not ignore_nan and nan_count > 0:
        return dpnp.array(dpnp.nan, ndmin=1, dtype=result_dtype)

    if found:
        if a.shape[0] % 2 == 0:
            # even number of elements
            result[0] = (partitioned[0] + partitioned[1]) / 2
        else:
            result[0] = partitioned[0]
    else:
        partitioned[buff_offset : buff_offset + num_elems].sort()
        kth_idx = buff_offset + k - elems_offset
        if a.shape[0] % 2 == 0:
            # even number of elements
            result[0] = (partitioned[kth_idx] + partitioned[kth_idx + 1]) / 2
        else:
            result[0] = partitioned[kth_idx]

    return result


def dpnp_median(
    a,
    axis=None,
    out=None,
    overwrite_input=False,
    keepdims=False,
    ignore_nan=False,
):
    """Compute the median of an array along a specified axis."""

    if axis is None or a.ndim == 1:
        result = native_median(a, ignore_nan)
        if not keepdims:
            return result[0]

        return result.reshape((1,) * a.ndim)

    a_ndim = a.ndim
    a_shape = a.shape
    _axis = range(a_ndim) if axis is None else axis
    _axis = normalize_axis_tuple(_axis, a_ndim)

    if len(_axis) == 1:
        if ignore_nan:
            a = dpnp.moveaxis(a, _axis[0], -1)
            axis = -1
        else:
            axis = _axis[0]
    else:
        # Need to flatten `a` if `_axis` is a sequence of axes
        # since `dpnp.sort` only accepts integer for `axis` kwarg
        if axis is None:
            a = dpnp.ravel(a)
        else:
            a, overwrite_input = _flatten_array_along_axes(
                a, _axis, overwrite_input
            )
        axis = -1

    if overwrite_input:
        if isinstance(a, dpt.usm_ndarray):
            # dpnp.ndarray.sort only works with dpnp_array
            a = dpnp_array._create_from_usm_ndarray(a)
        a.sort(axis=axis)
        a_sorted = a
    else:
        a_sorted = dpnp.sort(a, axis=axis)

    if ignore_nan:
        # sorting puts NaNs at the end
        assert axis == -1
        res = _calc_nanmedian(a_sorted, out=out)
    else:
        # We can't pass keepdims and use it in dpnp.mean and dpnp.any
        # because of the reshape hack that might have been used in
        # `_flatten_array_along_axes`.
        res = _calc_median(a_sorted, axis=axis, out=out)

    if keepdims:
        res_shape = list(a_shape)
        for i in _axis:
            res_shape[i] = 1
        res = res.reshape(tuple(res_shape))

    return dpnp.get_result_array(res, out)
