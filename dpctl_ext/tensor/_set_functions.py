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

from typing import NamedTuple

import dpctl.tensor as dpt
import dpctl.utils as du
from dpctl.tensor._tensor_elementwise_impl import _not_equal, _subtract

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt_ext

from ._tensor_impl import (
    _copy_usm_ndarray_into_usm_ndarray,
    _extract,
    _full_usm_ndarray,
    _linspace_step,
    default_device_index_type,
    mask_positions,
)
from ._tensor_sorting_impl import (
    _sort_ascending,
)


class UniqueCountsResult(NamedTuple):
    values: dpt.usm_ndarray
    counts: dpt.usm_ndarray


def unique_values(x: dpt.usm_ndarray) -> dpt.usm_ndarray:
    """unique_values(x)

    Returns the unique elements of an input array `x`.

    Args:
        x (usm_ndarray):
            input array. Inputs with more than one dimension are flattened.
    Returns:
        usm_ndarray
            an array containing the set of unique elements in `x`. The
            returned array has the same data type as `x`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    array_api_dev = x.device
    exec_q = array_api_dev.sycl_queue
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt_ext.reshape(x, (x.size,), order="C")
    if fx.size == 0:
        return fx
    s = dpt_ext.empty_like(fx, order="C")
    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    if fx.flags.c_contiguous:
        ht_ev, sort_ev = _sort_ascending(
            src=fx,
            trailing_dims_to_sort=1,
            dst=s,
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_ev, sort_ev)
    else:
        tmp = dpt_ext.empty_like(fx, order="C")
        ht_ev, copy_ev = _copy_usm_ndarray_into_usm_ndarray(
            src=fx, dst=tmp, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        ht_ev, sort_ev = _sort_ascending(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=s,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        _manager.add_event_pair(ht_ev, sort_ev)
    unique_mask = dpt_ext.empty(fx.shape, dtype="?", sycl_queue=exec_q)
    ht_ev, uneq_ev = _not_equal(
        src1=s[:-1],
        src2=s[1:],
        dst=unique_mask[1:],
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    _manager.add_event_pair(ht_ev, uneq_ev)
    # writing into new allocation, no dependencies
    ht_ev, one_ev = _full_usm_ndarray(
        fill_value=True, dst=unique_mask[0], sycl_queue=exec_q
    )
    _manager.add_event_pair(ht_ev, one_ev)
    cumsum = dpt_ext.empty(s.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[one_ev, uneq_ev]
    )
    if n_uniques == fx.size:
        return s
    unique_vals = dpt_ext.empty(
        n_uniques, dtype=x.dtype, usm_type=x.usm_type, sycl_queue=exec_q
    )
    ht_ev, ex_e = _extract(
        src=s,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_vals,
        sycl_queue=exec_q,
    )
    _manager.add_event_pair(ht_ev, ex_e)
    return unique_vals


def unique_counts(x: dpt.usm_ndarray) -> UniqueCountsResult:
    """unique_counts(x)

    Returns the unique elements of an input array `x` and the corresponding
    counts for each unique element in `x`.

    Args:
        x (usm_ndarray):
            input array. Inputs with more than one dimension are flattened.
    Returns:
        tuple[usm_ndarray, usm_ndarray]
            a namedtuple `(values, counts)` whose

            * first element is the field name `values` and is an array
               containing the unique elements of `x`. This array has the
               same data type as `x`.
            * second element has the field name `counts` and is an array
              containing the number of times each unique element occurs in `x`.
              This array has the same shape as `values` and has the default
              array index data type.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    array_api_dev = x.device
    exec_q = array_api_dev.sycl_queue
    x_usm_type = x.usm_type
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt_ext.reshape(x, (x.size,), order="C")
    ind_dt = default_device_index_type(exec_q)
    if fx.size == 0:
        return UniqueCountsResult(fx, dpt_ext.empty_like(fx, dtype=ind_dt))
    s = dpt_ext.empty_like(fx, order="C")

    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    if fx.flags.c_contiguous:
        ht_ev, sort_ev = _sort_ascending(
            src=fx,
            trailing_dims_to_sort=1,
            dst=s,
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_ev, sort_ev)
    else:
        tmp = dpt_ext.empty_like(fx, order="C")
        ht_ev, copy_ev = _copy_usm_ndarray_into_usm_ndarray(
            src=fx, dst=tmp, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        ht_ev, sort_ev = _sort_ascending(
            src=tmp,
            dst=s,
            trailing_dims_to_sort=1,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        _manager.add_event_pair(ht_ev, sort_ev)
    unique_mask = dpt_ext.empty(s.shape, dtype="?", sycl_queue=exec_q)
    ht_ev, uneq_ev = _not_equal(
        src1=s[:-1],
        src2=s[1:],
        dst=unique_mask[1:],
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    _manager.add_event_pair(ht_ev, uneq_ev)
    # no dependency, since we write into new allocation
    ht_ev, one_ev = _full_usm_ndarray(
        fill_value=True, dst=unique_mask[0], sycl_queue=exec_q
    )
    _manager.add_event_pair(ht_ev, one_ev)
    cumsum = dpt_ext.empty(
        unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q
    )
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[one_ev, uneq_ev]
    )
    if n_uniques == fx.size:
        return UniqueCountsResult(
            s,
            dpt_ext.ones(
                n_uniques, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
            ),
        )
    unique_vals = dpt_ext.empty(
        n_uniques, dtype=x.dtype, usm_type=x_usm_type, sycl_queue=exec_q
    )
    # populate unique values
    ht_ev, ex_e = _extract(
        src=s,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_vals,
        sycl_queue=exec_q,
    )
    _manager.add_event_pair(ht_ev, ex_e)
    unique_counts = dpt_ext.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
    )
    idx = dpt_ext.empty(x.size, dtype=ind_dt, sycl_queue=exec_q)
    # writing into new allocation, no dependency
    ht_ev, id_ev = _linspace_step(start=0, dt=1, dst=idx, sycl_queue=exec_q)
    _manager.add_event_pair(ht_ev, id_ev)
    ht_ev, extr_ev = _extract(
        src=idx,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_counts[:-1],
        sycl_queue=exec_q,
        depends=[id_ev],
    )
    _manager.add_event_pair(ht_ev, extr_ev)
    # no dependency, writing into disjoint segmenent of new allocation
    ht_ev, set_ev = _full_usm_ndarray(
        x.size, dst=unique_counts[-1], sycl_queue=exec_q
    )
    _manager.add_event_pair(ht_ev, set_ev)
    _counts = dpt_ext.empty_like(unique_counts[1:])
    ht_ev, sub_ev = _subtract(
        src1=unique_counts[1:],
        src2=unique_counts[:-1],
        dst=_counts,
        sycl_queue=exec_q,
        depends=[set_ev, extr_ev],
    )
    _manager.add_event_pair(ht_ev, sub_ev)
    return UniqueCountsResult(unique_vals, _counts)
