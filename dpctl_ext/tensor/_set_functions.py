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

from typing import NamedTuple, Optional, Union

import dpctl.utils as du

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt
from dpctl_ext.tensor._tensor_elementwise_impl import _not_equal, _subtract

from ._copy_utils import _empty_like_orderK
from ._scalar_utils import (
    _get_dtype,
    _get_queue_usm_type,
    _get_shape,
    _validate_dtype,
)
from ._tensor_impl import (
    _copy_usm_ndarray_into_usm_ndarray,
    _extract,
    _full_usm_ndarray,
    _linspace_step,
    _take,
    default_device_index_type,
    mask_positions,
)
from ._tensor_sorting_impl import (
    _argsort_ascending,
    _isin,
    _searchsorted_left,
    _sort_ascending,
)
from ._type_utils import (
    _resolve_weak_types_all_py_ints,
    _to_device_supported_dtype,
)

__all__ = [
    "isin",
    "unique_values",
    "unique_counts",
    "unique_inverse",
    "unique_all",
    "UniqueAllResult",
    "UniqueCountsResult",
    "UniqueInverseResult",
]


class UniqueAllResult(NamedTuple):
    values: dpt.usm_ndarray
    indices: dpt.usm_ndarray
    inverse_indices: dpt.usm_ndarray
    counts: dpt.usm_ndarray


class UniqueCountsResult(NamedTuple):
    values: dpt.usm_ndarray
    counts: dpt.usm_ndarray


class UniqueInverseResult(NamedTuple):
    values: dpt.usm_ndarray
    inverse_indices: dpt.usm_ndarray


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
        fx = dpt.reshape(x, (x.size,), order="C")
    if fx.size == 0:
        return fx
    s = dpt.empty_like(fx, order="C")
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
        tmp = dpt.empty_like(fx, order="C")
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
    unique_mask = dpt.empty(fx.shape, dtype="?", sycl_queue=exec_q)
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
    cumsum = dpt.empty(s.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[one_ev, uneq_ev]
    )
    if n_uniques == fx.size:
        return s
    unique_vals = dpt.empty(
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
        fx = dpt.reshape(x, (x.size,), order="C")
    ind_dt = default_device_index_type(exec_q)
    if fx.size == 0:
        return UniqueCountsResult(fx, dpt.empty_like(fx, dtype=ind_dt))
    s = dpt.empty_like(fx, order="C")

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
        tmp = dpt.empty_like(fx, order="C")
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
    unique_mask = dpt.empty(s.shape, dtype="?", sycl_queue=exec_q)
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
    cumsum = dpt.empty(unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[one_ev, uneq_ev]
    )
    if n_uniques == fx.size:
        return UniqueCountsResult(
            s,
            dpt.ones(
                n_uniques, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
            ),
        )
    unique_vals = dpt.empty(
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
    unique_counts = dpt.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
    )
    idx = dpt.empty(x.size, dtype=ind_dt, sycl_queue=exec_q)
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
    _counts = dpt.empty_like(unique_counts[1:])
    ht_ev, sub_ev = _subtract(
        src1=unique_counts[1:],
        src2=unique_counts[:-1],
        dst=_counts,
        sycl_queue=exec_q,
        depends=[set_ev, extr_ev],
    )
    _manager.add_event_pair(ht_ev, sub_ev)
    return UniqueCountsResult(unique_vals, _counts)


def unique_inverse(x):
    """unique_inverse

    Returns the unique elements of an input array x and the indices from the
    set of unique elements that reconstruct `x`.

    Args:
        x (usm_ndarray):
            input array. Inputs with more than one dimension are flattened.
    Returns:
        tuple[usm_ndarray, usm_ndarray]
            a namedtuple `(values, inverse_indices)` whose

            * first element has the field name `values` and is an array
              containing the unique elements of `x`. The array has the same
              data type as `x`.
            * second element has the field name `inverse_indices` and is an
              array containing the indices of values that reconstruct `x`.
              The array has the same shape as `x` and has the default array
              index data type.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    array_api_dev = x.device
    exec_q = array_api_dev.sycl_queue
    x_usm_type = x.usm_type
    ind_dt = default_device_index_type(exec_q)
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt.reshape(x, (x.size,), order="C")
    sorting_ids = dpt.empty_like(fx, dtype=ind_dt, order="C")
    unsorting_ids = dpt.empty_like(sorting_ids, dtype=ind_dt, order="C")
    if fx.size == 0:
        return UniqueInverseResult(fx, dpt.reshape(unsorting_ids, x.shape))

    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    if fx.flags.c_contiguous:
        ht_ev, sort_ev = _argsort_ascending(
            src=fx,
            trailing_dims_to_sort=1,
            dst=sorting_ids,
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_ev, sort_ev)
    else:
        tmp = dpt.empty_like(fx, order="C")
        ht_ev, copy_ev = _copy_usm_ndarray_into_usm_ndarray(
            src=fx, dst=tmp, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        ht_ev, sort_ev = _argsort_ascending(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=sorting_ids,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        _manager.add_event_pair(ht_ev, sort_ev)
    ht_ev, argsort_ev = _argsort_ascending(
        src=sorting_ids,
        trailing_dims_to_sort=1,
        dst=unsorting_ids,
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    _manager.add_event_pair(ht_ev, argsort_ev)
    s = dpt.empty_like(fx)
    # s = fx[sorting_ids]
    ht_ev, take_ev = _take(
        src=fx,
        ind=(sorting_ids,),
        dst=s,
        axis_start=0,
        mode=0,
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    _manager.add_event_pair(ht_ev, take_ev)
    unique_mask = dpt.empty(fx.shape, dtype="?", sycl_queue=exec_q)
    ht_ev, uneq_ev = _not_equal(
        src1=s[:-1],
        src2=s[1:],
        dst=unique_mask[1:],
        sycl_queue=exec_q,
        depends=[take_ev],
    )
    _manager.add_event_pair(ht_ev, uneq_ev)
    # no dependency
    ht_ev, one_ev = _full_usm_ndarray(
        fill_value=True, dst=unique_mask[0], sycl_queue=exec_q
    )
    _manager.add_event_pair(ht_ev, one_ev)
    cumsum = dpt.empty(unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[uneq_ev, one_ev]
    )
    if n_uniques == fx.size:
        return UniqueInverseResult(s, dpt.reshape(unsorting_ids, x.shape))
    unique_vals = dpt.empty(
        n_uniques, dtype=x.dtype, usm_type=x_usm_type, sycl_queue=exec_q
    )
    ht_ev, uv_ev = _extract(
        src=s,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_vals,
        sycl_queue=exec_q,
    )
    _manager.add_event_pair(ht_ev, uv_ev)
    cum_unique_counts = dpt.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
    )
    idx = dpt.empty(x.size, dtype=ind_dt, sycl_queue=exec_q)
    ht_ev, id_ev = _linspace_step(start=0, dt=1, dst=idx, sycl_queue=exec_q)
    _manager.add_event_pair(ht_ev, id_ev)
    ht_ev, extr_ev = _extract(
        src=idx,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=cum_unique_counts[:-1],
        sycl_queue=exec_q,
        depends=[id_ev],
    )
    _manager.add_event_pair(ht_ev, extr_ev)
    ht_ev, set_ev = _full_usm_ndarray(
        x.size, dst=cum_unique_counts[-1], sycl_queue=exec_q
    )
    _manager.add_event_pair(ht_ev, set_ev)
    _counts = dpt.empty_like(cum_unique_counts[1:])
    ht_ev, sub_ev = _subtract(
        src1=cum_unique_counts[1:],
        src2=cum_unique_counts[:-1],
        dst=_counts,
        sycl_queue=exec_q,
        depends=[set_ev, extr_ev],
    )
    _manager.add_event_pair(ht_ev, sub_ev)

    inv = dpt.empty_like(x, dtype=ind_dt, order="C")
    ht_ev, ssl_ev = _searchsorted_left(
        hay=unique_vals,
        needles=x,
        positions=inv,
        sycl_queue=exec_q,
        depends=[
            uv_ev,
        ],
    )
    _manager.add_event_pair(ht_ev, ssl_ev)

    return UniqueInverseResult(unique_vals, inv)


def unique_all(x: dpt.usm_ndarray) -> UniqueAllResult:
    """unique_all(x)

    Returns the unique elements of an input array `x`, the first occurring
    indices for each unique element in `x`, the indices from the set of unique
    elements that reconstruct `x`, and the corresponding counts for each
    unique element in `x`.

    Args:
        x (usm_ndarray):
            input array. Inputs with more than one dimension are flattened.
    Returns:
        tuple[usm_ndarray, usm_ndarray, usm_ndarray, usm_ndarray]
            a namedtuple `(values, indices, inverse_indices, counts)` whose

            * first element has the field name `values` and is an array
              containing the unique elements of `x`. The array has the same
              data type as `x`.
            * second element has the field name `indices` and is an array
              the indices (of first occurrences) of `x` that result in
              `values`. The array has the same shape as `values` and has the
              default array index data type.
            * third element has the field name `inverse_indices` and is an
              array containing the indices of values that reconstruct `x`.
              The array has the same shape as `x` and has the default array
              index data type.
            * fourth element has the field name `counts` and is an array
              containing the number of times each unique element occurs in `x`.
              This array has the same shape as `values` and has the default
              array index data type.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    array_api_dev = x.device
    exec_q = array_api_dev.sycl_queue
    x_usm_type = x.usm_type
    ind_dt = default_device_index_type(exec_q)
    if x.ndim == 1:
        fx = x
    else:
        fx = dpt.reshape(x, (x.size,), order="C")
    sorting_ids = dpt.empty_like(fx, dtype=ind_dt, order="C")
    unsorting_ids = dpt.empty_like(sorting_ids, dtype=ind_dt, order="C")
    if fx.size == 0:
        # original array contains no data
        # so it can be safely returned as values
        return UniqueAllResult(
            fx,
            sorting_ids,
            dpt.reshape(unsorting_ids, x.shape),
            dpt.empty_like(fx, dtype=ind_dt),
        )
    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    if fx.flags.c_contiguous:
        ht_ev, sort_ev = _argsort_ascending(
            src=fx,
            trailing_dims_to_sort=1,
            dst=sorting_ids,
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_ev, sort_ev)
    else:
        tmp = dpt.empty_like(fx, order="C")
        ht_ev, copy_ev = _copy_usm_ndarray_into_usm_ndarray(
            src=fx, dst=tmp, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, copy_ev)
        ht_ev, sort_ev = _argsort_ascending(
            src=tmp,
            trailing_dims_to_sort=1,
            dst=sorting_ids,
            sycl_queue=exec_q,
            depends=[copy_ev],
        )
        _manager.add_event_pair(ht_ev, sort_ev)
    ht_ev, args_ev = _argsort_ascending(
        src=sorting_ids,
        trailing_dims_to_sort=1,
        dst=unsorting_ids,
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    _manager.add_event_pair(ht_ev, args_ev)
    s = dpt.empty_like(fx)
    # s = fx[sorting_ids]
    ht_ev, take_ev = _take(
        src=fx,
        ind=(sorting_ids,),
        dst=s,
        axis_start=0,
        mode=0,
        sycl_queue=exec_q,
        depends=[sort_ev],
    )
    _manager.add_event_pair(ht_ev, take_ev)
    unique_mask = dpt.empty(fx.shape, dtype="?", sycl_queue=exec_q)
    ht_ev, uneq_ev = _not_equal(
        src1=s[:-1],
        src2=s[1:],
        dst=unique_mask[1:],
        sycl_queue=exec_q,
        depends=[take_ev],
    )
    _manager.add_event_pair(ht_ev, uneq_ev)
    ht_ev, one_ev = _full_usm_ndarray(
        fill_value=True, dst=unique_mask[0], sycl_queue=exec_q
    )
    _manager.add_event_pair(ht_ev, one_ev)
    cumsum = dpt.empty(unique_mask.shape, dtype=dpt.int64, sycl_queue=exec_q)
    # synchronizing call
    n_uniques = mask_positions(
        unique_mask, cumsum, sycl_queue=exec_q, depends=[uneq_ev, one_ev]
    )
    if n_uniques == fx.size:
        _counts = dpt.ones(
            n_uniques, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
        )
        return UniqueAllResult(
            s,
            sorting_ids,
            dpt.reshape(unsorting_ids, x.shape),
            _counts,
        )
    unique_vals = dpt.empty(
        n_uniques, dtype=x.dtype, usm_type=x_usm_type, sycl_queue=exec_q
    )
    ht_ev, uv_ev = _extract(
        src=s,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=unique_vals,
        sycl_queue=exec_q,
    )
    _manager.add_event_pair(ht_ev, uv_ev)
    cum_unique_counts = dpt.empty(
        n_uniques + 1, dtype=ind_dt, usm_type=x_usm_type, sycl_queue=exec_q
    )
    idx = dpt.empty(x.size, dtype=ind_dt, sycl_queue=exec_q)
    ht_ev, id_ev = _linspace_step(start=0, dt=1, dst=idx, sycl_queue=exec_q)
    _manager.add_event_pair(ht_ev, id_ev)
    ht_ev, extr_ev = _extract(
        src=idx,
        cumsum=cumsum,
        axis_start=0,
        axis_end=1,
        dst=cum_unique_counts[:-1],
        sycl_queue=exec_q,
        depends=[id_ev],
    )
    _manager.add_event_pair(ht_ev, extr_ev)
    ht_ev, set_ev = _full_usm_ndarray(
        x.size, dst=cum_unique_counts[-1], sycl_queue=exec_q
    )
    _manager.add_event_pair(ht_ev, set_ev)
    _counts = dpt.empty_like(cum_unique_counts[1:])
    ht_ev, sub_ev = _subtract(
        src1=cum_unique_counts[1:],
        src2=cum_unique_counts[:-1],
        dst=_counts,
        sycl_queue=exec_q,
        depends=[set_ev, extr_ev],
    )
    _manager.add_event_pair(ht_ev, sub_ev)

    inv = dpt.empty_like(x, dtype=ind_dt, order="C")
    ht_ev, ssl_ev = _searchsorted_left(
        hay=unique_vals,
        needles=x,
        positions=inv,
        sycl_queue=exec_q,
        depends=[
            uv_ev,
        ],
    )
    _manager.add_event_pair(ht_ev, ssl_ev)
    return UniqueAllResult(
        unique_vals,
        sorting_ids[cum_unique_counts[:-1]],
        inv,
        _counts,
    )


def isin(
    x: Union[dpt.usm_ndarray, int, float, complex, bool],
    test_elements: Union[dpt.usm_ndarray, int, float, complex, bool],
    /,
    *,
    invert: Optional[bool] = False,
) -> dpt.usm_ndarray:
    """isin(x, test_elements, /, *, invert=False)

    Tests `x in test_elements` for each element of `x`. Returns a boolean array
    with the same shape as `x` that is `True` where the element is in
    `test_elements`, `False` otherwise.

    Args:
        x (Union[usm_ndarray, bool, int, float, complex]):
            input element or elements.
        test_elements (Union[usm_ndarray, bool, int, float, complex]):
            elements against which to test each value of `x`.
        invert (Optional[bool]):
            if `True`, the output results are inverted, i.e., are equivalent to
            testing `x not in test_elements` for each element of `x`.
            Default: `False`.

    Returns:
        usm_ndarray:
            an array of the inclusion test results. The returned array has a
            boolean data type and the same shape as `x`.
    """
    q1, x_usm_type = _get_queue_usm_type(x)
    q2, test_usm_type = _get_queue_usm_type(test_elements)
    if q1 is None and q2 is None:
        raise du.ExecutionPlacementError(
            "Execution placement can not be unambiguously inferred "
            "from input arguments. "
            "One of the arguments must represent USM allocation and "
            "expose `__sycl_usm_array_interface__` property"
        )
    if q1 is None:
        exec_q = q2
        res_usm_type = test_usm_type
    elif q2 is None:
        exec_q = q1
        res_usm_type = x_usm_type
    else:
        exec_q = du.get_execution_queue((q1, q2))
        if exec_q is None:
            raise du.ExecutionPlacementError(
                "Execution placement can not be unambiguously inferred "
                "from input arguments."
            )
        res_usm_type = du.get_coerced_usm_type(
            (
                x_usm_type,
                test_usm_type,
            )
        )
    du.validate_usm_type(res_usm_type, allow_none=False)
    sycl_dev = exec_q.sycl_device

    if not isinstance(invert, bool):
        raise TypeError(
            "`invert` keyword argument must be of boolean type, "
            f"got {type(invert)}"
        )

    x_dt = _get_dtype(x, sycl_dev)
    test_dt = _get_dtype(test_elements, sycl_dev)
    if not all(_validate_dtype(dt) for dt in (x_dt, test_dt)):
        raise ValueError("Operands have unsupported data types")

    x_sh = _get_shape(x)
    if isinstance(test_elements, dpt.usm_ndarray) and test_elements.size == 0:
        if invert:
            return dpt.ones(
                x_sh, dtype=dpt.bool, usm_type=res_usm_type, sycl_queue=exec_q
            )
        else:
            return dpt.zeros(
                x_sh, dtype=dpt.bool, usm_type=res_usm_type, sycl_queue=exec_q
            )

    dt1, dt2 = _resolve_weak_types_all_py_ints(x_dt, test_dt, sycl_dev)
    dt = _to_device_supported_dtype(dpt.result_type(dt1, dt2), sycl_dev)

    if not isinstance(x, dpt.usm_ndarray):
        x_arr = dpt.asarray(
            x, dtype=dt1, usm_type=res_usm_type, sycl_queue=exec_q
        )
    else:
        x_arr = x

    if not isinstance(test_elements, dpt.usm_ndarray):
        test_arr = dpt.asarray(
            test_elements, dtype=dt2, usm_type=res_usm_type, sycl_queue=exec_q
        )
    else:
        test_arr = test_elements

    _manager = du.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events

    if x_dt != dt:
        x_buf = _empty_like_orderK(x_arr, dt, res_usm_type, exec_q)
        ht_ev, ev = _copy_usm_ndarray_into_usm_ndarray(
            src=x_arr, dst=x_buf, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, ev)
    else:
        x_buf = x_arr

    if test_dt != dt:
        # copy into C-contiguous memory, because the array will be flattened
        test_buf = dpt.empty_like(
            test_arr, dtype=dt, order="C", usm_type=res_usm_type
        )
        ht_ev, ev = _copy_usm_ndarray_into_usm_ndarray(
            src=test_arr, dst=test_buf, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(ht_ev, ev)
    else:
        test_buf = test_arr

    test_buf = dpt.reshape(test_buf, -1)
    test_buf = dpt.sort(test_buf)

    dst = dpt.empty_like(
        x_buf, dtype=dpt.bool, usm_type=res_usm_type, order="C"
    )

    dep_evs = _manager.submitted_events
    ht_ev, s_ev = _isin(
        needles=x_buf,
        hay=test_buf,
        dst=dst,
        sycl_queue=exec_q,
        invert=invert,
        depends=dep_evs,
    )
    _manager.add_event_pair(ht_ev, s_ev)
    return dst
