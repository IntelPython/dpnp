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

import operator

import dpctl
import dpctl.tensor as dpt
import dpctl.utils

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt_ext
import dpctl_ext.tensor._tensor_impl as ti

from ._copy_utils import (
    _extract_impl,
    _nonzero_impl,
    _put_multi_index,
    _take_multi_index,
)
from ._numpy_helper import normalize_axis_index


def _get_indexing_mode(name):
    modes = {"wrap": 0, "clip": 1}
    try:
        return modes[name]
    except KeyError:
        raise ValueError(
            "`mode` must be `wrap` or `clip`." "Got `{}`.".format(name)
        )


def _range(sh_i, i, nd, q, usm_t, dt):
    ind = dpt.arange(sh_i, dtype=dt, usm_type=usm_t, sycl_queue=q)
    ind.shape = tuple(sh_i if i == j else 1 for j in range(nd))
    return ind


def extract(condition, arr):
    """extract(condition, arr)

    Returns the elements of an array that satisfies the condition.

    If ``condition`` is boolean ``dpctl.tensor.extract`` is
    equivalent to ``arr[condition]``.

    Note that ``dpctl.tensor.place`` does the opposite of
    ``dpctl.tensor.extract``.

    Args:
       conditions (usm_ndarray):
            An array whose non-zero or ``True`` entries indicate the element
            of ``arr`` to extract.

       arr (usm_ndarray):
            Input array of the same size as ``condition``.

    Returns:
        usm_ndarray:
            Rank 1 array of values from ``arr`` where ``condition`` is
            ``True``.
    """
    if not isinstance(condition, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(condition)}"
        )
    if not isinstance(arr, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(arr)}"
        )
    exec_q = dpctl.utils.get_execution_queue(
        (
            condition.sycl_queue,
            arr.sycl_queue,
        )
    )
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    if condition.shape != arr.shape:
        raise ValueError("Arrays are not of the same size")
    return _extract_impl(arr, condition)


def nonzero(arr):
    """nonzero(arr)

    Return the indices of non-zero elements.

    Returns a tuple of usm_ndarrays, one for each dimension
    of ``arr``, containing the indices of the non-zero elements
    in that dimension. The values of ``arr`` are always tested in
    row-major, C-style order.

    Args:
        arr (usm_ndarray):
            Input array, which has non-zero array rank.

    Returns:
        Tuple[usm_ndarray, ...]:
            Indices of non-zero array elements.
    """
    if not isinstance(arr, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(arr)}"
        )
    if arr.ndim == 0:
        raise ValueError("Array of positive rank is expected")
    return _nonzero_impl(arr)


def place(arr, mask, vals):
    """place(arr, mask, vals)

    Change elements of an array based on conditional and input values.

    If ``mask`` is boolean ``dpctl.tensor.place`` is
    equivalent to ``arr[condition] = vals``.

    Args:
        arr (usm_ndarray):
            Array to put data into.
        mask (usm_ndarray):
            Boolean mask array. Must have the same size as ``arr``.
        vals (usm_ndarray, sequence):
            Values to put into ``arr``. Only the first N elements are
            used, where N is the number of True values in ``mask``. If
            ``vals`` is smaller than N, it will be repeated, and if
            elements of ``arr`` are to be masked, this sequence must be
            non-empty. Array ``vals`` must be one dimensional.
    """
    if not isinstance(arr, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(arr)}"
        )
    if not isinstance(mask, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(mask)}"
        )
    if not isinstance(vals, dpt.usm_ndarray):
        raise TypeError(
            "Expecting dpctl.tensor.usm_ndarray type, " f"got {type(vals)}"
        )
    exec_q = dpctl.utils.get_execution_queue(
        (
            arr.sycl_queue,
            mask.sycl_queue,
            vals.sycl_queue,
        )
    )
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    if arr.shape != mask.shape or vals.ndim != 1:
        raise ValueError("Array sizes are not as required")
    cumsum = dpt.empty(mask.size, dtype="i8", sycl_queue=exec_q)
    _manager = dpctl.utils.SequentialOrderManager[exec_q]
    deps_ev = _manager.submitted_events
    nz_count = ti.mask_positions(
        mask, cumsum, sycl_queue=exec_q, depends=deps_ev
    )
    if nz_count == 0:
        return
    if vals.size == 0:
        raise ValueError("Cannot insert from an empty array!")
    if vals.dtype == arr.dtype:
        rhs = vals
    else:
        rhs = dpt.astype(vals, arr.dtype)
    hev, pl_ev = ti._place(
        dst=arr,
        cumsum=cumsum,
        axis_start=0,
        axis_end=mask.ndim,
        rhs=rhs,
        sycl_queue=exec_q,
    )
    _manager.add_event_pair(hev, pl_ev)


def put(x, indices, vals, /, *, axis=None, mode="wrap"):
    """put(x, indices, vals, axis=None, mode="wrap")

    Puts values into an array along a given axis at given indices.

    Args:
        x (usm_ndarray):
            The array the values will be put into.
        indices (usm_ndarray):
            One-dimensional array of indices.
        vals (usm_ndarray):
            Array of values to be put into ``x``.
            Must be broadcastable to the result shape
            ``x.shape[:axis] + indices.shape + x.shape[axis+1:]``.
        axis (int, optional):
            The axis along which the values will be placed.
            If ``x`` is one-dimensional, this argument is optional.
            Default: ``None``.
        mode (str, optional):
            How out-of-bounds indices will be handled. Possible values
            are:

            - ``"wrap"``: clamps indices to (``-n <= i < n``), then wraps
              negative indices.
            - ``"clip"``: clips indices to (``0 <= i < n``).

            Default: ``"wrap"``.

    .. note::

        If input array ``indices`` contains duplicates, a race condition
        occurs, and the value written into corresponding positions in ``x``
        may vary from run to run. Preserving sequential semantics in handing
        the duplicates to achieve deterministic behavior requires additional
        work, e.g.

        :Example:

            .. code-block:: python

                from dpctl import tensor as dpt

                def put_vec_duplicates(vec, ind, vals):
                    "Put values into vec, handling possible duplicates in ind"
                    assert vec.ndim, ind.ndim, vals.ndim == 1, 1, 1

                    # find positions of last occurrences of each
                    # unique index
                    ind_flipped = dpt.flip(ind)
                    ind_uniq = dpt.unique_all(ind_flipped).indices
                    has_dups = len(ind) != len(ind_uniq)

                    if has_dups:
                        ind_uniq = dpt.subtract(vec.size - 1, ind_uniq)
                        ind = dpt.take(ind, ind_uniq)
                        vals = dpt.take(vals, ind_uniq)

                    dpt.put(vec, ind, vals)

                n = 512
                ind = dpt.concat((dpt.arange(n), dpt.arange(n, -1, step=-1)))
                x = dpt.zeros(ind.size, dtype="int32")
                vals = dpt.arange(ind.size, dtype=x.dtype)

                # Values corresponding to last positions of
                # duplicate indices are written into the vector x
                put_vec_duplicates(x, ind, vals)

                parts = (vals[-1:-n-2:-1], dpt.zeros(n, dtype=x.dtype))
                expected = dpt.concat(parts)
                assert dpt.all(x == expected)
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected instance of `dpt.usm_ndarray`, got `{}`.".format(type(x))
        )
    if not isinstance(indices, dpt.usm_ndarray):
        raise TypeError(
            "`indices` expected `dpt.usm_ndarray`, got `{}`.".format(
                type(indices)
            )
        )
    if isinstance(vals, dpt.usm_ndarray):
        queues_ = [x.sycl_queue, indices.sycl_queue, vals.sycl_queue]
        usm_types_ = [x.usm_type, indices.usm_type, vals.usm_type]
    else:
        queues_ = [x.sycl_queue, indices.sycl_queue]
        usm_types_ = [x.usm_type, indices.usm_type]
    if indices.ndim != 1:
        raise ValueError(
            "`indices` expected a 1D array, got `{}`".format(indices.ndim)
        )
    if indices.dtype.kind not in "ui":
        raise IndexError(
            "`indices` expected integer data type, got `{}`".format(
                indices.dtype
            )
        )
    exec_q = dpctl.utils.get_execution_queue(queues_)
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    vals_usm_type = dpctl.utils.get_coerced_usm_type(usm_types_)

    mode = _get_indexing_mode(mode)

    x_ndim = x.ndim
    if axis is None:
        if x_ndim > 1:
            raise ValueError(
                "`axis` cannot be `None` for array of dimension `{}`".format(
                    x_ndim
                )
            )
        axis = 0

    if x_ndim > 0:
        axis = normalize_axis_index(operator.index(axis), x_ndim)
        x_sh = x.shape
        if x_sh[axis] == 0 and indices.size != 0:
            raise IndexError("cannot take non-empty indices from an empty axis")
        val_shape = x.shape[:axis] + indices.shape + x.shape[axis + 1 :]
    else:
        if axis != 0:
            raise ValueError("`axis` must be 0 for an array of dimension 0.")
        val_shape = indices.shape

    if not isinstance(vals, dpt.usm_ndarray):
        vals = dpt.asarray(
            vals, dtype=x.dtype, usm_type=vals_usm_type, sycl_queue=exec_q
        )
    # choose to throw here for consistency with `place`
    if vals.size == 0:
        raise ValueError(
            "cannot put into non-empty indices along an empty axis"
        )
    if vals.dtype == x.dtype:
        rhs = vals
    else:
        rhs = dpt_ext.astype(vals, x.dtype)
    rhs = dpt.broadcast_to(rhs, val_shape)

    _manager = dpctl.utils.SequentialOrderManager[exec_q]
    deps_ev = _manager.submitted_events
    hev, put_ev = ti._put(
        x, (indices,), rhs, axis, mode, sycl_queue=exec_q, depends=deps_ev
    )
    _manager.add_event_pair(hev, put_ev)


def put_along_axis(x, indices, vals, /, *, axis=-1, mode="wrap"):
    """
    Puts elements into an array at the one-dimensional indices specified by
    ``indices`` along a provided ``axis``.

    Args:
        x (usm_ndarray):
            input array. Must be compatible with ``indices``, except for the
            axis (dimension) specified by ``axis``.
        indices (usm_ndarray):
            array indices. Must have the same rank (i.e., number of dimensions)
            as ``x``.
        vals (usm_ndarray):
            Array of values to be put into ``x``.
            Must be broadcastable to the shape of ``indices``.
        axis: int
            axis along which to select values. If ``axis`` is negative, the
            function determines the axis along which to select values by
            counting from the last dimension. Default: ``-1``.
        mode (str, optional):
            How out-of-bounds indices will be handled. Possible values
            are:

            - ``"wrap"``: clamps indices to (``-n <= i < n``), then wraps
              negative indices.
            - ``"clip"``: clips indices to (``0 <= i < n``).

            Default: ``"wrap"``.

    .. note::

        If input array ``indices`` contains duplicates, a race condition
        occurs, and the value written into corresponding positions in ``x``
        may vary from run to run. Preserving sequential semantics in handing
        the duplicates to achieve deterministic behavior requires additional
        work.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    if not isinstance(indices, dpt.usm_ndarray):
        raise TypeError(
            f"Expected dpctl.tensor.usm_ndarray, got {type(indices)}"
        )
    x_nd = x.ndim
    if x_nd != indices.ndim:
        raise ValueError(
            "Number of dimensions in the first and the second "
            "argument arrays must be equal"
        )
    pp = normalize_axis_index(operator.index(axis), x_nd)
    if isinstance(vals, dpt.usm_ndarray):
        queues_ = [x.sycl_queue, indices.sycl_queue, vals.sycl_queue]
        usm_types_ = [x.usm_type, indices.usm_type, vals.usm_type]
    else:
        queues_ = [x.sycl_queue, indices.sycl_queue]
        usm_types_ = [x.usm_type, indices.usm_type]
    exec_q = dpctl.utils.get_execution_queue(queues_)
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError(
            "Execution placement can not be unambiguously inferred "
            "from input arguments. "
        )
    out_usm_type = dpctl.utils.get_coerced_usm_type(usm_types_)
    mode_i = _get_indexing_mode(mode)
    indexes_dt = (
        dpt.uint64
        if indices.dtype == dpt.uint64
        else ti.default_device_index_type(exec_q.sycl_device)
    )
    _ind = tuple(
        (
            indices
            if i == pp
            else _range(x.shape[i], i, x_nd, exec_q, out_usm_type, indexes_dt)
        )
        for i in range(x_nd)
    )
    return _put_multi_index(x, _ind, 0, vals, mode=mode_i)


def take(x, indices, /, *, axis=None, out=None, mode="wrap"):
    """take(x, indices, axis=None, out=None, mode="wrap")

    Takes elements from an array along a given axis at given indices.

    Args:
        x (usm_ndarray):
            The array that elements will be taken from.
        indices (usm_ndarray):
            One-dimensional array of indices.
        axis (int, optional):
            The axis along which the values will be selected.
            If ``x`` is one-dimensional, this argument is optional.
            Default: ``None``.
        out (Optional[usm_ndarray]):
            Output array to populate. Array must have the correct
            shape and the expected data type.
        mode (str, optional):
            How out-of-bounds indices will be handled. Possible values
            are:

            - ``"wrap"``: clamps indices to (``-n <= i < n``), then wraps
              negative indices.
            - ``"clip"``: clips indices to (``0 <= i < n``).

            Default: ``"wrap"``.

    Returns:
       usm_ndarray:
          Array with shape
          ``x.shape[:axis] + indices.shape + x.shape[axis + 1:]``
          filled with elements from ``x``.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            "Expected instance of `dpt.usm_ndarray`, got `{}`.".format(type(x))
        )

    if not isinstance(indices, dpt.usm_ndarray):
        raise TypeError(
            "`indices` expected `dpt.usm_ndarray`, got `{}`.".format(
                type(indices)
            )
        )
    if indices.dtype.kind not in "ui":
        raise IndexError(
            "`indices` expected integer data type, got `{}`".format(
                indices.dtype
            )
        )
    if indices.ndim != 1:
        raise ValueError(
            "`indices` expected a 1D array, got `{}`".format(indices.ndim)
        )
    exec_q = dpctl.utils.get_execution_queue([x.sycl_queue, indices.sycl_queue])
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError
    res_usm_type = dpctl.utils.get_coerced_usm_type(
        [x.usm_type, indices.usm_type]
    )

    mode = _get_indexing_mode(mode)

    x_ndim = x.ndim
    if axis is None:
        if x_ndim > 1:
            raise ValueError(
                "`axis` cannot be `None` for array of dimension `{}`".format(
                    x_ndim
                )
            )
        axis = 0

    if x_ndim > 0:
        axis = normalize_axis_index(operator.index(axis), x_ndim)
        x_sh = x.shape
        if x_sh[axis] == 0 and indices.size != 0:
            raise IndexError("cannot take non-empty indices from an empty axis")
        res_shape = x.shape[:axis] + indices.shape + x.shape[axis + 1 :]
    else:
        if axis != 0:
            raise ValueError("`axis` must be 0 for an array of dimension 0.")
        res_shape = indices.shape

    dt = x.dtype

    orig_out = out
    if out is not None:
        if not isinstance(out, dpt.usm_ndarray):
            raise TypeError(
                f"output array must be of usm_ndarray type, got {type(out)}"
            )
        if not out.flags.writable:
            raise ValueError("provided `out` array is read-only")

        if out.shape != res_shape:
            raise ValueError(
                "The shape of input and output arrays are inconsistent. "
                f"Expected output shape is {res_shape}, got {out.shape}"
            )
        if dt != out.dtype:
            raise ValueError(
                f"Output array of type {dt} is needed, got {out.dtype}"
            )
        if dpctl.utils.get_execution_queue((exec_q, out.sycl_queue)) is None:
            raise dpctl.utils.ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )
        if ti._array_overlap(x, out):
            out = dpt.empty_like(out)
    else:
        out = dpt.empty(
            res_shape, dtype=dt, usm_type=res_usm_type, sycl_queue=exec_q
        )

    _manager = dpctl.utils.SequentialOrderManager[exec_q]
    deps_ev = _manager.submitted_events
    hev, take_ev = ti._take(
        x, (indices,), out, axis, mode, sycl_queue=exec_q, depends=deps_ev
    )
    _manager.add_event_pair(hev, take_ev)

    if not (orig_out is None or out is orig_out):
        # Copy the out data from temporary buffer to original memory
        ht_e_cpy, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=out, dst=orig_out, sycl_queue=exec_q, depends=[take_ev]
        )
        _manager.add_event_pair(ht_e_cpy, cpy_ev)
        out = orig_out

    return out


def take_along_axis(x, indices, /, *, axis=-1, mode="wrap"):
    """
    Returns elements from an array at the one-dimensional indices specified
    by ``indices`` along a provided ``axis``.

    Args:
        x (usm_ndarray):
            input array. Must be compatible with ``indices``, except for the
            axis (dimension) specified by ``axis``.
        indices (usm_ndarray):
            array indices. Must have the same rank (i.e., number of dimensions)
            as ``x``.
        axis: int
            axis along which to select values. If ``axis`` is negative, the
            function determines the axis along which to select values by
            counting from the last dimension. Default: ``-1``.
        mode (str, optional):
            How out-of-bounds indices will be handled. Possible values
            are:

            - ``"wrap"``: clamps indices to (``-n <= i < n``), then wraps
              negative indices.
            - ``"clip"``: clips indices to (``0 <= i < n``).

            Default: ``"wrap"``.

    Returns:
        usm_ndarray:
            an array having the same data type as ``x``. The returned array has
            the same rank (i.e., number of dimensions) as ``x`` and a shape
            determined according to broadcasting rules, except for the axis
            (dimension) specified by ``axis`` whose size must equal the size
            of the corresponding axis (dimension) in ``indices``.

    Note:
        Treatment of the out-of-bound indices in ``indices`` array is controlled
        by the value of ``mode`` keyword.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected dpctl.tensor.usm_ndarray, got {type(x)}")
    if not isinstance(indices, dpt.usm_ndarray):
        raise TypeError(
            f"Expected dpctl.tensor.usm_ndarray, got {type(indices)}"
        )
    x_nd = x.ndim
    if x_nd != indices.ndim:
        raise ValueError(
            "Number of dimensions in the first and the second "
            "argument arrays must be equal"
        )
    pp = normalize_axis_index(operator.index(axis), x_nd)
    out_usm_type = dpctl.utils.get_coerced_usm_type(
        (x.usm_type, indices.usm_type)
    )
    exec_q = dpctl.utils.get_execution_queue((x.sycl_queue, indices.sycl_queue))
    if exec_q is None:
        raise dpctl.utils.ExecutionPlacementError(
            "Execution placement can not be unambiguously inferred "
            "from input arguments. "
        )
    mode_i = _get_indexing_mode(mode)
    indexes_dt = (
        dpt.uint64
        if indices.dtype == dpt.uint64
        else ti.default_device_index_type(exec_q.sycl_device)
    )
    _ind = tuple(
        (
            indices
            if i == pp
            else _range(x.shape[i], i, x_nd, exec_q, out_usm_type, indexes_dt)
        )
        for i in range(x_nd)
    )
    return _take_multi_index(x, _ind, 0, mode=mode_i)
