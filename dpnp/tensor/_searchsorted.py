# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

from typing import Literal

import dpctl.utils as du

import dpnp.tensor as dpt

from ._compute_follows_data import (
    ExecutionPlacementError,
    get_coerced_usm_type,
    get_execution_queue,
)
from ._copy_utils import _empty_like_orderK
from ._ctors import empty_like
from ._scalar_utils import _get_dtype, _get_queue_usm_type, _validate_dtype
from ._tensor_impl import _copy_usm_ndarray_into_usm_ndarray as ti_copy
from ._tensor_impl import _take as ti_take
from ._tensor_impl import (
    default_device_index_type as ti_default_device_index_type,
)
from ._tensor_sorting_impl import _searchsorted_left, _searchsorted_right
from ._type_utils import (
    _resolve_weak_types_all_py_ints,
    _to_device_supported_dtype,
    isdtype,
)
from ._usmarray import usm_ndarray


def searchsorted(
    x1: usm_ndarray,
    x2: usm_ndarray | int | float | complex | bool,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: usm_ndarray | None = None,
) -> usm_ndarray:
    """searchsorted(x1, x2, side='left', sorter=None)

    Finds the indices into `x1` such that, if the corresponding elements
    in `x2` were inserted before the indices, the order of `x1`, when sorted
    in ascending order, would be preserved.

    Args:
        x1 (usm_ndarray):
            input array. Must be a one-dimensional array. If `sorter` is
            `None`, must be sorted in ascending order; otherwise, `sorter` must
            be an array of indices that sort `x1` in ascending order.
        x2 (usm_ndarray | int | float | complex | bool):
            search value or values.
        side (Literal["left", "right]):
            argument controlling which index is returned if a value lands
            exactly on an edge. If `x2` is an array of rank `N` where
            `v = x2[n, m, ..., j]`, the element `ret[n, m, ..., j]` in the
            return array `ret` contains the position `i` such that
            if `side="left"`, it is the first index such that
            `x1[i-1] < v <= x1[i]`, `0` if `v <= x1[0]`, and `x1.size`
            if `v > x1[-1]`;
            and if `side="right"`, it is the first position `i` such that
            `x1[i-1] <= v < x1[i]`, `0` if `v < x1[0]`, and `x1.size`
            if `v >= x1[-1]`. Default: `"left"`.
        sorter (Optional[usm_ndarray]):
            array of indices that sort `x1` in ascending order. The array must
            have the same shape as `x1` and have an integral data type.
            Out of bound index values of `sorter` array are treated using
            `"wrap"` mode documented in :py:func:`dpnp.tensor.take`.
            Default: `None`.
    """
    if not isinstance(x1, usm_ndarray):
        raise TypeError(f"Expected dpnp.tensor.usm_ndarray, got {type(x1)}")
    if sorter is not None and not isinstance(sorter, usm_ndarray):
        raise TypeError(f"Expected dpnp.tensor.usm_ndarray, got {type(sorter)}")

    if side not in ["left", "right"]:
        raise ValueError(
            "Unrecognized value of 'side' keyword argument. "
            "Expected either 'left' or 'right'"
        )

    q1, x1_usm_type = x1.sycl_queue, x1.usm_type
    q2, x2_usm_type = _get_queue_usm_type(x2)
    q3 = sorter.sycl_queue if sorter is not None else None
    q = get_execution_queue(tuple(q for q in (q1, q2, q3) if q is not None))
    if q is None:
        raise ExecutionPlacementError(
            "Execution placement can not be unambiguously "
            "inferred from input arguments."
        )

    res_usm_type = get_coerced_usm_type(
        tuple(
            ut
            for ut in (
                x1_usm_type,
                x2_usm_type,
            )
            if ut is not None
        )
    )
    dpt.validate_usm_type(res_usm_type, allow_none=False)
    sycl_dev = q.sycl_device

    if x1.ndim != 1:
        raise ValueError("First argument array must be one-dimensional")

    x1_dt = x1.dtype
    x2_dt = _get_dtype(x2, sycl_dev)
    if not _validate_dtype(x2_dt):
        raise ValueError(
            "dpt.searchsorted search value argument has "
            f"unsupported data type {x2_dt}"
        )

    _manager = du.SequentialOrderManager[q]
    dep_evs = _manager.submitted_events
    x1_deps = dep_evs
    if sorter is not None:
        if not isdtype(sorter.dtype, "integral"):
            raise ValueError(
                f"Sorter array must have integral data type, got {sorter.dtype}"
            )
        if x1.shape != sorter.shape:
            raise ValueError(
                "Sorter array must be one-dimension with the same "
                "shape as the first argument array"
            )
        res = empty_like(x1)
        ind = (sorter,)
        axis = 0
        wrap_out_of_bound_indices_mode = 0
        ht_ev, ev = ti_take(
            x1,
            ind,
            res,
            axis,
            axis + 1,
            wrap_out_of_bound_indices_mode,
            sycl_queue=q,
            depends=dep_evs,
        )
        x1 = res
        x1_deps = [ev]
        _manager.add_event_pair(ht_ev, ev)

    dt1, dt2 = _resolve_weak_types_all_py_ints(x1_dt, x2_dt, sycl_dev)
    dt = _to_device_supported_dtype(dpt.result_type(dt1, dt2), sycl_dev)

    if x1_dt != dt:
        x1_buf = _empty_like_orderK(x1, dt)
        # get the submitted events again to ensure the copy waits take call
        ht_ev, ev = ti_copy(src=x1, dst=x1_buf, sycl_queue=q, depends=x1_deps)
        _manager.add_event_pair(ht_ev, ev)
        x1 = x1_buf

    if not isinstance(x2, usm_ndarray):
        x2 = dpt.asarray(x2, dtype=dt, usm_type=res_usm_type, sycl_queue=q)
    elif x2_dt != dt:
        x2_buf = _empty_like_orderK(x2, dt)
        ht_ev, ev = ti_copy(src=x2, dst=x2_buf, sycl_queue=q, depends=dep_evs)
        _manager.add_event_pair(ht_ev, ev)
        x2 = x2_buf

    index_dt = ti_default_device_index_type(q)

    dst = _empty_like_orderK(x2, index_dt, usm_type=res_usm_type)

    dep_evs = _manager.submitted_events
    if side == "left":
        ht_ev, s_ev = _searchsorted_left(
            hay=x1,
            needles=x2,
            positions=dst,
            sycl_queue=q,
            depends=dep_evs,
        )
    else:
        ht_ev, s_ev = _searchsorted_right(
            hay=x1, needles=x2, positions=dst, sycl_queue=q, depends=dep_evs
        )
    _manager.add_event_pair(ht_ev, s_ev)
    return dst
