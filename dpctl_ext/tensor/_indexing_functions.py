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

import dpctl_ext.tensor._tensor_impl as ti

from ._numpy_helper import normalize_axis_index


def _get_indexing_mode(name):
    modes = {"wrap": 0, "clip": 1}
    try:
        return modes[name]
    except KeyError:
        raise ValueError(
            "`mode` must be `wrap` or `clip`." "Got `{}`.".format(name)
        )


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
        rhs = dpt.astype(vals, x.dtype)
    rhs = dpt.broadcast_to(rhs, val_shape)

    _manager = dpctl.utils.SequentialOrderManager[exec_q]
    deps_ev = _manager.submitted_events
    hev, put_ev = ti._put(
        x, (indices,), rhs, axis, mode, sycl_queue=exec_q, depends=deps_ev
    )
    _manager.add_event_pair(hev, put_ev)


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
