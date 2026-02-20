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

import itertools
import operator

import dpctl
import dpctl.tensor as dpt
import dpctl.utils as dputils
import numpy as np

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor._tensor_impl as ti

from ._numpy_helper import normalize_axis_index, normalize_axis_tuple

__doc__ = (
    "Implementation module for array manipulation "
    "functions in :module:`dpctl.tensor`"
)


def _broadcast_shape_impl(shapes):
    if len(set(shapes)) == 1:
        return shapes[0]
    mutable_shapes = False
    nds = [len(s) for s in shapes]
    biggest = max(nds)
    sh_len = len(shapes)
    for i in range(sh_len):
        diff = biggest - nds[i]
        if diff > 0:
            ty = type(shapes[i])
            shapes[i] = ty(
                itertools.chain(itertools.repeat(1, diff), shapes[i])
            )
    common_shape = []
    for axis in range(biggest):
        lengths = [s[axis] for s in shapes]
        unique = set(lengths + [1])
        if len(unique) > 2:
            raise ValueError(
                "Shape mismatch: two or more arrays have "
                f"incompatible dimensions on axis ({axis},)"
            )
        elif len(unique) == 2:
            unique.remove(1)
            new_length = unique.pop()
            common_shape.append(new_length)
            for i in range(sh_len):
                if shapes[i][axis] == 1:
                    if not mutable_shapes:
                        shapes = [list(s) for s in shapes]
                        mutable_shapes = True
                    shapes[i][axis] = new_length
        else:
            common_shape.append(1)

    return tuple(common_shape)


def repeat(x, repeats, /, *, axis=None):
    """repeat(x, repeats, axis=None)

    Repeat elements of an array on a per-element basis.

    Args:
        x (usm_ndarray): input array

        repeats (Union[int, Sequence[int, ...], usm_ndarray]):
            The number of repetitions for each element.

            `repeats` must be broadcast-compatible with `N` where `N` is
            `prod(x.shape)` if `axis` is `None` and `x.shape[axis]`
            otherwise.

            If `repeats` is an array, it must have an integer data type.
            Otherwise, `repeats` must be a Python integer or sequence of
            Python integers (i.e., a tuple, list, or range).

        axis (Optional[int]):
            The axis along which to repeat values. If `axis` is `None`, the
            function repeats elements of the flattened array. Default: `None`.

    Returns:
        usm_ndarray:
            output array with repeated elements.

            If `axis` is `None`, the returned array is one-dimensional,
            otherwise, it has the same shape as `x`, except for the axis along
            which elements were repeated.

            The returned array will have the same data type as `x`.
            The returned array will be located on the same device as `x` and
            have the same USM allocation type as `x`.

    Raises:
        AxisError: if `axis` value is invalid.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(x)}.")

    x_ndim = x.ndim
    x_shape = x.shape
    if axis is not None:
        axis = normalize_axis_index(operator.index(axis), x_ndim)
        axis_size = x_shape[axis]
    else:
        axis_size = x.size

    scalar = False
    if isinstance(repeats, int):
        if repeats < 0:
            raise ValueError("`repeats` must be a positive integer")
        usm_type = x.usm_type
        exec_q = x.sycl_queue
        scalar = True
    elif isinstance(repeats, dpt.usm_ndarray):
        if repeats.ndim > 1:
            raise ValueError(
                "`repeats` array must be 0- or 1-dimensional, got "
                f"{repeats.ndim}"
            )
        exec_q = dpctl.utils.get_execution_queue(
            (x.sycl_queue, repeats.sycl_queue)
        )
        if exec_q is None:
            raise dputils.ExecutionPlacementError(
                "Execution placement can not be unambiguously inferred "
                "from input arguments."
            )
        usm_type = dpctl.utils.get_coerced_usm_type(
            (
                x.usm_type,
                repeats.usm_type,
            )
        )
        dpctl.utils.validate_usm_type(usm_type, allow_none=False)
        if not dpt.can_cast(repeats.dtype, dpt.int64, casting="same_kind"):
            raise TypeError(
                f"'repeats' data type {repeats.dtype} cannot be cast to "
                "'int64' according to the casting rule ''safe.''"
            )
        if repeats.size == 1:
            scalar = True
            # bring the single element to the host
            if repeats.ndim == 0:
                repeats = int(repeats)
            else:
                # Get the single element explicitly
                # since non-0D arrays can not be converted to scalars
                repeats = int(repeats[0])
            if repeats < 0:
                raise ValueError("`repeats` elements must be positive")
        else:
            if repeats.size != axis_size:
                raise ValueError(
                    "'repeats' array must be broadcastable to the size of "
                    "the repeated axis"
                )
            if not dpt.all(repeats >= 0):
                raise ValueError("'repeats' elements must be positive")

    elif isinstance(repeats, (tuple, list, range)):
        usm_type = x.usm_type
        exec_q = x.sycl_queue

        len_reps = len(repeats)
        if len_reps == 1:
            repeats = repeats[0]
            if repeats < 0:
                raise ValueError("`repeats` elements must be positive")
            scalar = True
        else:
            if len_reps != axis_size:
                raise ValueError(
                    "`repeats` sequence must have the same length as the "
                    "repeated axis"
                )
            repeats = dpt.asarray(
                repeats, dtype=dpt.int64, usm_type=usm_type, sycl_queue=exec_q
            )
            if not dpt.all(repeats >= 0):
                raise ValueError("`repeats` elements must be positive")
    else:
        raise TypeError(
            "Expected int, sequence, or `usm_ndarray` for second argument,"
            f"got {type(repeats)}"
        )

    _manager = dputils.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    if scalar:
        res_axis_size = repeats * axis_size
        if axis is not None:
            res_shape = x_shape[:axis] + (res_axis_size,) + x_shape[axis + 1 :]
        else:
            res_shape = (res_axis_size,)
        res = dpt.empty(
            res_shape, dtype=x.dtype, usm_type=usm_type, sycl_queue=exec_q
        )
        if res_axis_size > 0:
            ht_rep_ev, rep_ev = ti._repeat_by_scalar(
                src=x,
                dst=res,
                reps=repeats,
                axis=axis,
                sycl_queue=exec_q,
                depends=dep_evs,
            )
            _manager.add_event_pair(ht_rep_ev, rep_ev)
    else:
        if repeats.dtype != dpt.int64:
            rep_buf = dpt.empty(
                repeats.shape,
                dtype=dpt.int64,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=repeats, dst=rep_buf, sycl_queue=exec_q, depends=dep_evs
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)
            cumsum = dpt.empty(
                (axis_size,),
                dtype=dpt.int64,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
            # _cumsum_1d synchronizes so `depends` ends here safely
            res_axis_size = ti._cumsum_1d(
                rep_buf, cumsum, sycl_queue=exec_q, depends=[copy_ev]
            )
            if axis is not None:
                res_shape = (
                    x_shape[:axis] + (res_axis_size,) + x_shape[axis + 1 :]
                )
            else:
                res_shape = (res_axis_size,)
            res = dpt.empty(
                res_shape,
                dtype=x.dtype,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
            if res_axis_size > 0:
                ht_rep_ev, rep_ev = ti._repeat_by_sequence(
                    src=x,
                    dst=res,
                    reps=rep_buf,
                    cumsum=cumsum,
                    axis=axis,
                    sycl_queue=exec_q,
                )
                _manager.add_event_pair(ht_rep_ev, rep_ev)
        else:
            cumsum = dpt.empty(
                (axis_size,),
                dtype=dpt.int64,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
            res_axis_size = ti._cumsum_1d(
                repeats, cumsum, sycl_queue=exec_q, depends=dep_evs
            )
            if axis is not None:
                res_shape = (
                    x_shape[:axis] + (res_axis_size,) + x_shape[axis + 1 :]
                )
            else:
                res_shape = (res_axis_size,)
            res = dpt.empty(
                res_shape,
                dtype=x.dtype,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
            if res_axis_size > 0:
                ht_rep_ev, rep_ev = ti._repeat_by_sequence(
                    src=x,
                    dst=res,
                    reps=repeats,
                    cumsum=cumsum,
                    axis=axis,
                    sycl_queue=exec_q,
                )
                _manager.add_event_pair(ht_rep_ev, rep_ev)
    return res


def roll(x, /, shift, *, axis=None):
    """
    roll(x, shift, axis)

    Rolls array elements along a specified axis.
    Array elements that roll beyond the last position are re-introduced
    at the first position. Array elements that roll beyond the first position
    are re-introduced at the last position.

    Args:
        x (usm_ndarray): input array
        shift (Union[int, Tuple[int,...]]): number of places by which the
            elements are shifted. If `shift` is a tuple, then `axis` must be a
            tuple of the same size, and each of the given axes must be shifted
            by the corresponding element in `shift`. If `shift` is an `int`
            and `axis` a tuple, then the same `shift` must be used for all
            specified axes. If a `shift` is positive, then array elements is
            shifted positively (toward larger indices) along the dimension of
            `axis`.
            If a `shift` is negative, then array elements must be shifted
            negatively (toward smaller indices) along the dimension of `axis`.
        axis (Optional[Union[int, Tuple[int,...]]]): axis (or axes) along which
            elements to shift. If `axis` is `None`, the array is
            flattened, shifted, and then restored to its original shape.
            Default: `None`.

    Returns:
        usm_ndarray:
            An array having the same `dtype`, `usm_type` and
            `device` attributes as `x` and whose elements are shifted relative
            to `x`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(x)}.")
    exec_q = x.sycl_queue
    _manager = dputils.SequentialOrderManager[exec_q]
    if axis is None:
        shift = operator.index(shift)
        res = dpt.empty(
            x.shape, dtype=x.dtype, usm_type=x.usm_type, sycl_queue=exec_q
        )
        sz = operator.index(x.size)
        shift = (shift % sz) if sz > 0 else 0
        dep_evs = _manager.submitted_events
        hev, roll_ev = ti._copy_usm_ndarray_for_roll_1d(
            src=x,
            dst=res,
            shift=shift,
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(hev, roll_ev)
        return res
    axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
    broadcasted = np.broadcast(shift, axis)
    if broadcasted.ndim > 1:
        raise ValueError("'shift' and 'axis' should be scalars or 1D sequences")
    shifts = [
        0,
    ] * x.ndim
    shape = x.shape
    for sh, ax in broadcasted:
        n_i = operator.index(shape[ax])
        shifted = shifts[ax] + operator.index(sh)
        shifts[ax] = (shifted % n_i) if n_i > 0 else 0
    res = dpt.empty(
        x.shape, dtype=x.dtype, usm_type=x.usm_type, sycl_queue=exec_q
    )
    dep_evs = _manager.submitted_events
    ht_e, roll_ev = ti._copy_usm_ndarray_for_roll_nd(
        src=x, dst=res, shifts=shifts, sycl_queue=exec_q, depends=dep_evs
    )
    _manager.add_event_pair(ht_e, roll_ev)
    return res
