# *****************************************************************************
# Copyright (c) 2023-2024, Intel Corporation
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

import warnings

import dpctl
import dpctl.tensor as dpt
from dpctl.tensor._numpy_helper import normalize_axis_tuple
from dpctl.utils import ExecutionPlacementError

import dpnp
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations, map_dtype_to_device

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


def _calc_nanmedian(a, axis, out=None):
    """Compute the median of an array along a specified axis, ignoring NaNs."""
    mask = dpnp.isnan(a)
    valid_counts = dpnp.sum(~mask, axis=axis)
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

    # Iterate over all indices of the output shape
    for idx in dpnp.ndindex(res.shape):
        current_valid_counts = valid_counts[idx]

        if current_valid_counts > 0:
            # Extract the corresponding slice from the last axis of `a`
            data = a[idx][:current_valid_counts]
            left = (current_valid_counts - 1) // 2
            right = current_valid_counts // 2

            if left == right:
                res[idx] = data[left]
            else:
                res[idx] = (data[left] + data[right]) / 2.0
        else:
            warnings.warn(
                "All-NaN slice encountered", RuntimeWarning, stacklevel=6
            )
            res[idx] = dpnp.nan

    return res


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


def dpnp_cov(m, y=None, rowvar=True, dtype=None):
    """
    dpnp_cov(m, y=None, rowvar=True, dtype=None)

    Estimate a covariance matrix based on passed data.
    No support for given weights is provided now.

    The implementation is done through existing dpnp and dpctl methods
    instead of separate function call of dpnp backend.

    """

    def _get_2dmin_array(x, dtype):
        """
        Transform an input array to a form required for building a covariance matrix.

        If applicable, it reshapes the input array to have 2 dimensions or greater.
        If applicable, it transposes the input array when 'rowvar' is False.
        It casts to another dtype, if the input array differs from requested one.

        """
        if x.ndim == 0:
            x = x.reshape((1, 1))
        elif x.ndim == 1:
            x = x[dpnp.newaxis, :]

        if not rowvar and x.shape[0] != 1:
            x = x.T

        if x.dtype != dtype:
            x = dpnp.astype(x, dtype)
        return x

    # input arrays must follow CFD paradigm
    _, queue = get_usm_allocations((m,) if y is None else (m, y))

    # calculate a type of result array if not passed explicitly
    if dtype is None:
        dtypes = [m.dtype, dpnp.default_float_type(sycl_queue=queue)]
        if y is not None:
            dtypes.append(y.dtype)
        dtype = dpnp.result_type(*dtypes)
        # TODO: remove when dpctl.result_type() is returned dtype based on fp64
        dtype = map_dtype_to_device(dtype, queue.sycl_device)

    X = _get_2dmin_array(m, dtype)
    if y is not None:
        y = _get_2dmin_array(y, dtype)

        X = dpnp.concatenate((X, y), axis=0)

    avg = X.mean(axis=1)

    fact = X.shape[1] - 1
    X -= avg[:, None]

    c = dpnp.dot(X, X.T.conj())
    c *= 1 / fact if fact != 0 else dpnp.nan

    return dpnp.squeeze(c)


def dpnp_median(
    a,
    axis=None,
    out=None,
    overwrite_input=False,
    keepdims=False,
    ignore_nan=False,
):
    """Compute the median of an array along a specified axis."""

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
        res = _calc_nanmedian(a_sorted, axis=axis, out=out)
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
