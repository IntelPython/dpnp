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

import math
import warnings

import dpctl.utils as dpu
import numpy

import dpnp

# pylint: disable=no-name-in-module
import dpnp.libtensor._statistics_impl as statistics_ext
import dpnp.libtensor._tensor_impl as dti
import dpnp.libtensor._tensor_reductions_impl as tri
import dpnp.tensor as dpt
from dpnp.dpnp_array import dpnp_array
from dpnp.exceptions import ExecutionPlacementError
from dpnp.tensor._numpy_helper import normalize_axis_tuple

__all__ = ["dpnp_cov", "dpnp_median", "dpnp_minmax"]


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


def _all_nan_usm_type(sycl_device):
    """
    USM type of the all-NaN flag of `_calc_median_by_select`, which the
    kernel sets with an atomic store.

    """

    # GPU Max faults on an atomic store to host USM
    if sycl_device.has_aspect_usm_atomic_host_allocations:
        return "host"
    return "device"


def _calc_median_by_select(a, ignore_nan, out=None):
    """
    Compute the median of an array along its last axis by radix select,
    ignoring NaNs if `ignore_nan`.

    """

    usm_a = dpnp.get_usm_ndarray(a)
    exec_q = usm_a.sycl_queue
    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if dpt.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )
        if out.shape != usm_a.shape[:-1]:
            raise ValueError(
                f"Output array of shape {usm_a.shape[:-1]} is needed, "
                f"got {out.shape}."
            )

    _manager = dpu.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events

    # the medians are selected along the last axis of a C-contiguous array
    if not usm_a.flags.c_contiguous:
        usm_tmp = dpt.empty_like(usm_a, order="C")
        ht_ev, copy_ev = dti._copy_usm_ndarray_into_usm_ndarray(
            src=usm_a, dst=usm_tmp, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        usm_a = usm_tmp
        dep_evs = [copy_ev]

    # the medians of floating data types keep their type, as with numpy
    if dpnp.issubdtype(usm_a.dtype, dpnp.floating):
        res_dt = usm_a.dtype
    else:
        res_dt = dpnp.default_float_type(sycl_queue=exec_q)

    usm_res = dpt.empty(
        usm_a.shape[:-1],
        dtype=res_dt,
        usm_type=usm_a.usm_type,
        order="C",
        sycl_queue=exec_q,
    )
    # set to whether a row has NaNs only, the one case warned about
    usm_all_nan = None
    if ignore_nan:
        usm_all_nan = dpt.empty(
            1,
            dtype=dpt.uint32,
            usm_type=_all_nan_usm_type(exec_q.sycl_device),
            sycl_queue=exec_q,
        )

    ht_ev, median_ev = statistics_ext._median(
        usm_a,
        ignore_nan,
        usm_res,
        sycl_queue=exec_q,
        all_nan=usm_all_nan,
        depends=dep_evs,
    )
    _manager.add_event_pair(ht_ev, median_ev)

    if ignore_nan:
        if usm_all_nan.usm_type == "host":
            # read in place, without the queue wait and copy of asnumpy
            median_ev.wait()
            all_nan = numpy.frombuffer(usm_all_nan.usm_data, dtype=numpy.uint32)
        else:
            all_nan = dpt.asnumpy(usm_all_nan)
        if all_nan[0]:
            warnings.warn(
                "All-NaN slice encountered", RuntimeWarning, stacklevel=4
            )

    res = dpnp_array._create_from_usm_ndarray(usm_res)
    if out is None:
        return res
    out[...] = res
    return out


def _calc_nanmedian(a, out=None):
    """Compute the median of an array along a specified axis, ignoring NaNs."""
    mask = dpnp.isnan(a)
    valid_counts = dpnp.sum(~mask, axis=-1)
    if out is None:
        res = dpnp.empty_like(valid_counts, dtype=a.dtype)
    else:
        dpnp.check_supported_arrays_type(out)
        exec_q = dpt.get_execution_queue((a.sycl_queue, out.sycl_queue))
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

    # only drop the reduced axis, keep size-1 dimensions that are not reduced
    return dpnp.squeeze(res, axis=-1)


def _flatten_array_along_axes(a, axes_to_flatten, overwrite_input):
    """Flatten an array along a specific set of axes."""

    a_ndim = a.ndim
    axes_to_keep = tuple(
        axis for axis in range(a_ndim) if axis not in axes_to_flatten
    )

    # Move the axes_to_flatten to the end
    destination = list(range(len(axes_to_keep), a_ndim))
    a_moved = dpnp.moveaxis(a, axes_to_flatten, destination)
    # Compute the merged length explicitly instead of letting `reshape` infer
    # it with -1, since -1 is ambiguous when a kept axis has size 0
    merged = math.prod(a.shape[axis] for axis in axes_to_flatten)
    new_shape = tuple(a.shape[axis] for axis in axes_to_keep) + (merged,)
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


def _calc_median_by_sort(a, axis, out, overwrite_input, ignore_nan):
    """Compute the median of an array along a specified axis by sorting."""

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
        return _calc_nanmedian(a_sorted, out=out)

    # We can't pass keepdims and use it in dpnp.mean and dpnp.any
    # because of the reshape hack that might have been used in
    # `_flatten_array_along_axes`.
    return _calc_median(a_sorted, axis=axis, out=out)


def _calc_median_of_empty(a, shape):
    """Return the medians of empty array `a`, all NaN, in `shape`."""

    if dpnp.issubdtype(a.dtype, dpnp.inexact):
        res_dt = a.dtype
    else:
        res_dt = dpnp.default_float_type(sycl_queue=a.sycl_queue)
    # numpy's mean of an empty complex slice is NaN in both parts
    if dpnp.issubdtype(res_dt, dpnp.complexfloating):
        fill_value = complex(dpnp.nan, dpnp.nan)
    else:
        fill_value = dpnp.nan
    return dpnp.full(
        shape,
        fill_value,
        dtype=res_dt,
        usm_type=a.usm_type,
        sycl_queue=a.sycl_queue,
    )


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

    # the medians of real data types are selected rather than sorted
    use_select = not dpnp.issubdtype(a.dtype, dpnp.complexfloating)
    if not dpnp.issubdtype(a.dtype, dpnp.inexact):
        # there are no NaNs to ignore
        ignore_nan = False

    if len(_axis) == 1:
        if ignore_nan or use_select:
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

    res_out = out
    if keepdims:
        res_shape = list(a_shape)
        for i in _axis:
            res_shape[i] = 1
        res_shape = tuple(res_shape)

        if out is not None:
            dpnp.check_supported_arrays_type(out)
            if out.shape != res_shape:
                raise ValueError(
                    f"Output array of shape {res_shape} is needed, "
                    f"got {out.shape}."
                )
            # the medians go to a view of `out` without the reduced axes
            res_out = out[
                tuple(0 if i in _axis else slice(None) for i in range(a_ndim))
            ]

    if a.size == 0:
        res = _calc_median_of_empty(
            a, tuple(n for i, n in enumerate(a_shape) if i not in _axis)
        )
    elif use_select:
        assert axis == -1
        res = _calc_median_by_select(a, ignore_nan, out=res_out)
    else:
        res = _calc_median_by_sort(
            a, axis, res_out, overwrite_input, ignore_nan
        )

    if out is None:
        if keepdims:
            res = res.reshape(res_shape)
        return dpnp.get_result_array(res)

    # the medians are cast to `out` as with numpy
    dpnp.get_result_array(res, res_out, casting="unsafe")
    return dpnp.get_result_array(out)


def dpnp_minmax(x, /, *, axis=None, keepdims=False, out=None):
    """
    Calculates the minimum and the maximum of the input array ``x`` in a
    single pass.

    Args:
        x (usm_ndarray):
            input array.
        axis (Optional[int, Tuple[int, ...]]):
            axis or axes along which the extrema must be computed.
            If ``None``, they are computed over the entire array.
            Default: ``None``.
        keepdims (Optional[bool]):
            if ``True``, the reduced axes (dimensions) are included in the
            results as singleton dimensions. Default: ``False``.
        out (Optional[Tuple[Optional[usm_ndarray], Optional[usm_ndarray]]]):
            the arrays into which the minima and the maxima are written.
            Each array must have the expected shape and the data type of
            ``x``. A ``None`` entry means that a new array is returned.
            Default: ``None``.

    Returns:
        Tuple[usm_ndarray, usm_ndarray]:
            the minima and the maxima of ``x``, with its data type.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpnp.tensor.usm_ndarray, got {type(x)}")

    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
        perm = list(axis)
        x_tmp = x
    else:
        if not isinstance(axis, (tuple, list)):
            axis = (axis,)
        axis = normalize_axis_tuple(axis, nd, "axis")
        perm = [i for i in range(nd) if i not in axis] + list(axis)
        x_tmp = dpt.permute_dims(x, perm)
    red_nd = len(axis)
    if any([x_tmp.shape[i] == 0 for i in range(-red_nd, 0)]):
        raise ValueError("reduction cannot be performed over zero-size axes")
    res_shape = x_tmp.shape[: nd - red_nd]
    exec_q = x.sycl_queue
    res_dt = x.dtype
    res_usm_type = x.usm_type

    if keepdims:
        final_res_shape = tuple(
            x.shape[d] if d not in axis else 1 for d in range(nd)
        )
    else:
        final_res_shape = res_shape

    if out is None:
        out = (None, None)
    orig_out = list(out)
    out = list(out)
    for i, res in enumerate(out):
        if res is None:
            out[i] = dpt.empty(
                res_shape,
                dtype=res_dt,
                usm_type=res_usm_type,
                sycl_queue=exec_q,
            )
            continue

        if not isinstance(res, dpt.usm_ndarray):
            raise TypeError(
                f"output array must be of usm_ndarray type, got {type(res)}"
            )
        if not res.flags.writable:
            raise ValueError("provided `out` array is read-only")
        if not res.shape == final_res_shape:
            raise ValueError(
                "The shape of input and output arrays are inconsistent. "
                f"Expected output shape is {final_res_shape}, got {res.shape}"
            )
        if res_dt != res.dtype:
            raise ValueError(
                f"Output array of type {res_dt} is needed, got {res.dtype}"
            )
        if dpt.get_execution_queue((exec_q, res.sycl_queue)) is None:
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )
        if keepdims:
            res = dpt.squeeze(res, axis=axis)
            orig_out[i] = res
        if dti._array_overlap(x, res):
            res = dpt.empty_like(res)
        out[i] = res

    _manager = dpu.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    if red_nd == 0:
        red_evs = []
        for res in out:
            ht_e_cpy, cpy_e = dti._copy_usm_ndarray_into_usm_ndarray(
                src=x_tmp, dst=res, sycl_queue=exec_q, depends=dep_evs
            )
            _manager.add_event_pair(ht_e_cpy, cpy_e)
            red_evs.append(cpy_e)
    else:
        hev, red_ev = tri._minmax_over_axis(
            src=x_tmp,
            trailing_dims_to_reduce=red_nd,
            dst_min=out[0],
            dst_max=out[1],
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(hev, red_ev)
        red_evs = [red_ev]

    for i, res in enumerate(out):
        if not (orig_out[i] is None or orig_out[i] is res):
            ht_e_cpy2, cpy2_e = dti._copy_usm_ndarray_into_usm_ndarray(
                src=res, dst=orig_out[i], sycl_queue=exec_q, depends=red_evs
            )
            _manager.add_event_pair(ht_e_cpy2, cpy2_e)
            out[i] = orig_out[i]

    if keepdims:
        res_shape = res_shape + (1,) * red_nd
        inv_perm = sorted(range(nd), key=lambda d: perm[d])
        out = [
            dpt.permute_dims(dpt.reshape(res, res_shape), inv_perm)
            for res in out
        ]
    return tuple(out)
