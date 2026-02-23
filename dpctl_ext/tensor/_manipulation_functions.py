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
import dpctl_ext.tensor as dpt_ext
import dpctl_ext.tensor._tensor_impl as ti

from ._numpy_helper import normalize_axis_index, normalize_axis_tuple
from ._type_utils import _supported_dtype, _to_device_supported_dtype

__doc__ = (
    "Implementation module for array manipulation "
    "functions in :module:`dpctl.tensor`"
)


def _arrays_validation(arrays, check_ndim=True):
    n = len(arrays)
    if n == 0:
        raise TypeError("Missing 1 required positional argument: 'arrays'.")

    if not isinstance(arrays, (list, tuple)):
        raise TypeError(f"Expected tuple or list type, got {type(arrays)}.")

    for X in arrays:
        if not isinstance(X, dpt.usm_ndarray):
            raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    exec_q = dputils.get_execution_queue([X.sycl_queue for X in arrays])
    if exec_q is None:
        raise ValueError("All the input arrays must have same sycl queue.")

    res_usm_type = dputils.get_coerced_usm_type([X.usm_type for X in arrays])
    if res_usm_type is None:
        raise ValueError("All the input arrays must have usm_type.")

    X0 = arrays[0]
    _supported_dtype(Xi.dtype for Xi in arrays)

    res_dtype = X0.dtype
    dev = exec_q.sycl_device
    for i in range(1, n):
        res_dtype = np.promote_types(res_dtype, arrays[i])
        res_dtype = _to_device_supported_dtype(res_dtype, dev)

    if check_ndim:
        for i in range(1, n):
            if X0.ndim != arrays[i].ndim:
                raise ValueError(
                    "All the input arrays must have same number of dimensions, "
                    f"but the array at index 0 has {X0.ndim} dimension(s) and "
                    f"the array at index {i} has {arrays[i].ndim} dimension(s)."
                )
    return res_dtype, res_usm_type, exec_q


def _broadcast_shapes(*args):
    """
    Broadcast the input shapes into a single shape;
    returns tuple broadcasted shape.
    """
    array_shapes = [array.shape for array in args]
    return _broadcast_shape_impl(array_shapes)


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


def _broadcast_strides(X_shape, X_strides, res_ndim):
    """
    Broadcasts strides to match the given dimensions;
    returns tuple type strides.
    """
    out_strides = [0] * res_ndim
    X_shape_len = len(X_shape)
    str_dim = -X_shape_len
    for i in range(X_shape_len):
        shape_value = X_shape[i]
        if not shape_value == 1:
            out_strides[str_dim] = X_strides[i]
        str_dim += 1

    return tuple(out_strides)


def _check_same_shapes(X0_shape, axis, n, arrays):
    for i in range(1, n):
        Xi_shape = arrays[i].shape
        for j, X0j in enumerate(X0_shape):
            if X0j != Xi_shape[j] and j != axis:
                raise ValueError(
                    "All the input array dimensions for the concatenation "
                    f"axis must match exactly, but along dimension {j}, the "
                    f"array at index 0 has size {X0j} and the array "
                    f"at index {i} has size {Xi_shape[j]}."
                )


def _concat_axis_None(arrays):
    """Implementation of concat(arrays, axis=None)."""
    res_dtype, res_usm_type, exec_q = _arrays_validation(
        arrays, check_ndim=False
    )
    res_shape = 0
    for array in arrays:
        res_shape += array.size
    res = dpt_ext.empty(
        res_shape, dtype=res_dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )

    fill_start = 0
    _manager = dputils.SequentialOrderManager[exec_q]
    deps = _manager.submitted_events
    for array in arrays:
        fill_end = fill_start + array.size
        if array.flags.c_contiguous:
            hev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=dpt_ext.reshape(array, -1),
                dst=res[fill_start:fill_end],
                sycl_queue=exec_q,
                depends=deps,
            )
            _manager.add_event_pair(hev, cpy_ev)
        else:
            src_ = array
            # _copy_usm_ndarray_for_reshape requires src and dst to have
            # the same data type
            if not array.dtype == res_dtype:
                src2_ = dpt_ext.empty_like(src_, dtype=res_dtype)
                ht_copy_ev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                    src=src_, dst=src2_, sycl_queue=exec_q, depends=deps
                )
                _manager.add_event_pair(ht_copy_ev, cpy_ev)
                hev, reshape_copy_ev = ti._copy_usm_ndarray_for_reshape(
                    src=src2_,
                    dst=res[fill_start:fill_end],
                    sycl_queue=exec_q,
                    depends=[cpy_ev],
                )
                _manager.add_event_pair(hev, reshape_copy_ev)
            else:
                hev, cpy_ev = ti._copy_usm_ndarray_for_reshape(
                    src=src_,
                    dst=res[fill_start:fill_end],
                    sycl_queue=exec_q,
                    depends=deps,
                )
                _manager.add_event_pair(hev, cpy_ev)
        fill_start = fill_end

    return res


def broadcast_arrays(*args):
    """broadcast_arrays(*arrays)

    Broadcasts one or more :class:`dpctl.tensor.usm_ndarrays` against
    one another.

    Args:
        arrays (usm_ndarray): an arbitrary number of arrays to be
            broadcasted.

    Returns:
        List[usm_ndarray]:
            A list of broadcasted arrays. Each array
            must have the same shape. Each array must have the same `dtype`,
            `device` and `usm_type` attributes as its corresponding input
            array.
    """
    if len(args) == 0:
        raise ValueError("`broadcast_arrays` requires at least one argument")
    for X in args:
        if not isinstance(X, dpt.usm_ndarray):
            raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    shape = _broadcast_shapes(*args)

    if all(X.shape == shape for X in args):
        return args

    return [broadcast_to(X, shape) for X in args]


def broadcast_to(X, /, shape):
    """broadcast_to(x, shape)

    Broadcast an array to a new `shape`; returns the broadcasted
    :class:`dpctl.tensor.usm_ndarray` as a view.

    Args:
        x (usm_ndarray): input array
        shape (Tuple[int,...]): array shape. The `shape` must be
            compatible with `x` according to broadcasting rules.

    Returns:
        usm_ndarray:
            An array with the specified `shape`.
            The output array is a view of the input array, and
            hence has the same data type, USM allocation type and
            device attributes.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    # Use numpy.broadcast_to to check the validity of the input
    # parameter 'shape'. Raise ValueError if 'X' is not compatible
    # with 'shape' according to NumPy's broadcasting rules.
    new_array = np.broadcast_to(
        np.broadcast_to(np.empty(tuple(), dtype="u1"), X.shape), shape
    )
    new_sts = _broadcast_strides(X.shape, X.strides, new_array.ndim)
    return dpt.usm_ndarray(
        shape=new_array.shape,
        dtype=X.dtype,
        buffer=X,
        strides=new_sts,
        offset=X._element_offset,
    )


def concat(arrays, /, *, axis=0):
    """concat(arrays, axis)

    Joins a sequence of arrays along an existing axis.

    Args:
        arrays (Union[List[usm_ndarray, Tuple[usm_ndarray,...]]]):
            input arrays to join. The arrays must have the same shape,
            except in the dimension specified by `axis`.
        axis (Optional[int]): axis along which the arrays will be joined.
            If `axis` is `None`, arrays must be flattened before
            concatenation. If `axis` is negative, it is understood as
            being counted from the last dimension. Default: `0`.

    Returns:
        usm_ndarray:
            An output array containing the concatenated
            values. The output array data type is determined by Type
            Promotion Rules of array API.

    All input arrays must have the same device attribute. The output array
    is allocated on that same device, and data movement operations are
    scheduled on a queue underlying the device. The USM allocation type
    of the output array is determined by USM allocation type promotion
    rules.
    """
    if axis is None:
        return _concat_axis_None(arrays)

    res_dtype, res_usm_type, exec_q = _arrays_validation(arrays)
    n = len(arrays)
    X0 = arrays[0]

    axis = normalize_axis_index(axis, X0.ndim)
    X0_shape = X0.shape
    _check_same_shapes(X0_shape, axis, n, arrays)

    res_shape_axis = 0
    for X in arrays:
        res_shape_axis = res_shape_axis + X.shape[axis]

    res_shape = tuple(
        X0_shape[i] if i != axis else res_shape_axis for i in range(X0.ndim)
    )

    res = dpt_ext.empty(
        res_shape, dtype=res_dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )

    _manager = dputils.SequentialOrderManager[exec_q]
    deps = _manager.submitted_events
    fill_start = 0
    for i in range(n):
        fill_end = fill_start + arrays[i].shape[axis]
        c_shapes_copy = tuple(
            np.s_[fill_start:fill_end] if j == axis else np.s_[:]
            for j in range(X0.ndim)
        )
        hev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arrays[i],
            dst=res[c_shapes_copy],
            sycl_queue=exec_q,
            depends=deps,
        )
        _manager.add_event_pair(hev, cpy_ev)
        fill_start = fill_end

    return res


def expand_dims(X, /, *, axis=0):
    """expand_dims(x, axis)

    Expands the shape of an array by inserting a new axis (dimension)
    of size one at the position specified by axis.

    Args:
        x (usm_ndarray):
            input array
        axis (Union[int, Tuple[int]]):
            axis position in the expanded axes (zero-based). If `x` has rank
            (i.e, number of dimensions) `N`, a valid `axis` must reside
            in the closed-interval `[-N-1, N]`. If provided a negative
            `axis`, the `axis` position at which to insert a singleton
            dimension is computed as `N + axis + 1`. Hence, if
            provided `-1`, the resolved axis position is `N` (i.e.,
            a singleton dimension must be appended to the input array `x`).
            If provided `-N-1`, the resolved axis position is `0` (i.e., a
            singleton dimension is prepended to the input array `x`).

    Returns:
        usm_ndarray:
            Returns a view, if possible, and a copy otherwise with the number
            of dimensions increased.
            The expanded array has the same data type as the input array `x`.
            The expanded array is located on the same device as the input
            array, and has the same USM allocation type.

    Raises:
        IndexError: if `axis` value is invalid.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    if type(axis) not in (tuple, list):
        axis = (axis,)

    out_ndim = len(axis) + X.ndim
    axis = normalize_axis_tuple(axis, out_ndim)

    shape_it = iter(X.shape)
    shape = tuple(1 if ax in axis else next(shape_it) for ax in range(out_ndim))

    return dpt_ext.reshape(X, shape)


def flip(X, /, *, axis=None):
    """flip(x, axis)

    Reverses the order of elements in an array `x` along the given `axis`.
    The shape of the array is preserved, but the elements are reordered.

    Args:
        x (usm_ndarray): input array.
        axis (Optional[Union[int, Tuple[int,...]]]): axis (or axes) along
            which to flip.
            If `axis` is `None`, all input array axes are flipped.
            If `axis` is negative, the flipped axis is counted from the
            last dimension. If provided more than one axis, only the specified
            axes are flipped. Default: `None`.

    Returns:
        usm_ndarray:
            A view of `x` with the entries of `axis` reversed.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")
    X_ndim = X.ndim
    if axis is None:
        indexer = (np.s_[::-1],) * X_ndim
    else:
        axis = normalize_axis_tuple(axis, X_ndim)
        indexer = tuple(
            np.s_[::-1] if i in axis else np.s_[:] for i in range(X.ndim)
        )
    return X[indexer]


def moveaxis(X, source, destination, /):
    """moveaxis(x, source, destination)

    Moves axes of an array to new positions.

    Args:
        x (usm_ndarray): input array

        source (int or a sequence of int):
            Original positions of the axes to move.
            These must be unique. If `x` has rank (i.e., number of
            dimensions) `N`, a valid `axis` must be in the
            half-open interval `[-N, N)`.

        destination (int or a sequence of int):
            Destination positions for each of the original axes.
            These must also be unique. If `x` has rank
            (i.e., number of dimensions) `N`, a valid `axis` must be
            in the half-open interval `[-N, N)`.

    Returns:
        usm_ndarray:
            Array with moved axes.
            The returned array must has the same data type as `x`,
            is created on the same device as `x` and has the same
            USM allocation type as `x`.

    Raises:
        AxisError: if `axis` value is invalid.
        ValueError: if `src` and `dst` have not equal number of elements.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    source = normalize_axis_tuple(source, X.ndim, "source")
    destination = normalize_axis_tuple(destination, X.ndim, "destination")

    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` arguments must have "
            "the same number of elements"
        )

    ind = [n for n in range(X.ndim) if n not in source]

    for src, dst in sorted(zip(destination, source)):
        ind.insert(src, dst)

    return dpt_ext.permute_dims(X, tuple(ind))


def permute_dims(X, /, axes):
    """permute_dims(x, axes)

    Permute the axes (dimensions) of an array; returns the permuted
    array as a view.

    Args:
        x (usm_ndarray): input array.
        axes (Tuple[int, ...]): tuple containing permutation of
           `(0,1,...,N-1)` where `N` is the number of axes (dimensions)
           of `x`.
    Returns:
        usm_ndarray:
            An array with permuted axes.
            The returned array must has the same data type as `x`,
            is created on the same device as `x` and has the same USM allocation
            type as `x`.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")
    axes = normalize_axis_tuple(axes, X.ndim, "axes")
    if not X.ndim == len(axes):
        raise ValueError(
            "The length of the passed axes does not match "
            "to the number of usm_ndarray dimensions."
        )
    newstrides = tuple(X.strides[i] for i in axes)
    newshape = tuple(X.shape[i] for i in axes)
    return dpt.usm_ndarray(
        shape=newshape,
        dtype=X.dtype,
        buffer=X,
        strides=newstrides,
        offset=X._element_offset,
    )


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
        if not dpt_ext.can_cast(repeats.dtype, dpt.int64, casting="same_kind"):
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
            repeats = dpt_ext.asarray(
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
        res = dpt_ext.empty(
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
            rep_buf = dpt_ext.empty(
                repeats.shape,
                dtype=dpt.int64,
                usm_type=usm_type,
                sycl_queue=exec_q,
            )
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=repeats, dst=rep_buf, sycl_queue=exec_q, depends=dep_evs
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)
            cumsum = dpt_ext.empty(
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
            res = dpt_ext.empty(
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
            cumsum = dpt_ext.empty(
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
            res = dpt_ext.empty(
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
        res = dpt_ext.empty(
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
    res = dpt_ext.empty(
        x.shape, dtype=x.dtype, usm_type=x.usm_type, sycl_queue=exec_q
    )
    dep_evs = _manager.submitted_events
    ht_e, roll_ev = ti._copy_usm_ndarray_for_roll_nd(
        src=x, dst=res, shifts=shifts, sycl_queue=exec_q, depends=dep_evs
    )
    _manager.add_event_pair(ht_e, roll_ev)
    return res


def squeeze(X, /, axis=None):
    """squeeze(x, axis)

    Removes singleton dimensions (axes) from array `x`.

    Args:
        x (usm_ndarray): input array
        axis (Union[int, Tuple[int,...]]): axis (or axes) to squeeze.

    Returns:
        usm_ndarray:
            Output array is a view, if possible,
            and a copy otherwise, but with all or a subset of the
            dimensions of length 1 removed. Output has the same data
            type as the input, is allocated on the same device as the
            input and has the same USM allocation type as the input
            array `x`.

    Raises:
        ValueError: if the specified axis has a size greater than one.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")
    X_shape = X.shape
    if axis is not None:
        axis = normalize_axis_tuple(axis, X.ndim if X.ndim != 0 else X.ndim + 1)
        new_shape = []
        for i, x in enumerate(X_shape):
            if i not in axis:
                new_shape.append(x)
            else:
                if x != 1:
                    raise ValueError(
                        "Cannot select an axis to squeeze out "
                        "which has size not equal to one."
                    )
        new_shape = tuple(new_shape)
    else:
        new_shape = tuple(axis for axis in X_shape if axis != 1)
    if new_shape == X.shape:
        return X
    else:
        return dpt_ext.reshape(X, new_shape)


def stack(arrays, /, *, axis=0):
    """
    stack(arrays, axis)

    Joins a sequence of arrays along a new axis.

    Args:
        arrays (Union[List[usm_ndarray], Tuple[usm_ndarray,...]]):
            input arrays to join. Each array must have the same shape.
        axis (int): axis along which the arrays will be joined. Providing
            an `axis` specified the index of the new axis in the dimensions
            of the output array. A valid axis must be on the interval
            `[-N, N)`, where `N` is the rank (number of dimensions) of `x`.
            Default: `0`.

    Returns:
        usm_ndarray:
            An output array having rank `N+1`, where `N` is
            the rank (number of dimensions) of `x`. If the input arrays have
            different data types, array API Type Promotion Rules apply.

    Raises:
        ValueError: if not all input arrays have the same shape
        IndexError: if provided an `axis` outside of the required interval.
    """
    res_dtype, res_usm_type, exec_q = _arrays_validation(arrays)

    n = len(arrays)
    X0 = arrays[0]
    res_ndim = X0.ndim + 1
    axis = normalize_axis_index(axis, res_ndim)
    X0_shape = X0.shape

    for i in range(1, n):
        if X0_shape != arrays[i].shape:
            raise ValueError("All input arrays must have the same shape")

    res_shape = tuple(
        X0_shape[i - 1 * (i >= axis)] if i != axis else n
        for i in range(res_ndim)
    )

    res = dpt_ext.empty(
        res_shape, dtype=res_dtype, usm_type=res_usm_type, sycl_queue=exec_q
    )

    _manager = dputils.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events
    for i in range(n):
        c_shapes_copy = tuple(
            i if j == axis else np.s_[:] for j in range(res_ndim)
        )
        _dst = res[c_shapes_copy]
        hev, cpy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=arrays[i], dst=_dst, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(hev, cpy_ev)

    return res


def swapaxes(X, axis1, axis2):
    """swapaxes(x, axis1, axis2)

    Interchanges two axes of an array.

    Args:
        x (usm_ndarray): input array

        axis1 (int): First axis.
            If `x` has rank (i.e., number of dimensions) `N`,
            a valid `axis` must be in the half-open interval `[-N, N)`.

        axis2 (int): Second axis.
            If `x` has rank (i.e., number of dimensions) `N`,
            a valid `axis` must be in the half-open interval `[-N, N)`.

    Returns:
        usm_ndarray:
            Array with swapped axes.
            The returned array must has the same data type as `x`,
            is created on the same device as `x` and has the same USM
            allocation type as `x`.

    Raises:
        AxisError: if `axis` value is invalid.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(X)}.")

    axis1 = normalize_axis_index(axis1, X.ndim, "axis1")
    axis2 = normalize_axis_index(axis2, X.ndim, "axis2")

    ind = list(range(0, X.ndim))
    ind[axis1] = axis2
    ind[axis2] = axis1
    return dpt_ext.permute_dims(X, tuple(ind))


def tile(x, repetitions, /):
    """tile(x, repetitions)

    Repeat an input array `x` along each axis a number of times given by
    `repetitions`.

    For `N` = len(`repetitions`) and `M` = len(`x.shape`):

        * If `M < N`, `x` will have `N - M` new axes prepended to its shape
        * If `M > N`, `repetitions` will have `M - N` ones prepended to it

    Args:
        x (usm_ndarray): input array

        repetitions (Union[int, Tuple[int, ...]]):
            The number of repetitions along each dimension of `x`.

    Returns:
        usm_ndarray:
            tiled output array.

            The returned array will have rank `max(M, N)`. If `S` is the
            shape of `x` after prepending dimensions and `R` is
            `repetitions` after prepending ones, then the shape of the
            result will be `S[i] * R[i]` for each dimension `i`.

            The returned array will have the same data type as `x`.
            The returned array will be located on the same device as `x` and
            have the same USM allocation type as `x`.
    """
    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(f"Expected usm_ndarray type, got {type(x)}.")

    if not isinstance(repetitions, tuple):
        if isinstance(repetitions, int):
            repetitions = (repetitions,)
        else:
            raise TypeError(
                f"Expected tuple or integer type, got {type(repetitions)}."
            )

    rep_dims = len(repetitions)
    x_dims = x.ndim
    if rep_dims < x_dims:
        repetitions = (x_dims - rep_dims) * (1,) + repetitions
    elif x_dims < rep_dims:
        x = dpt_ext.reshape(x, (rep_dims - x_dims) * (1,) + x.shape)
    res_shape = tuple(map(lambda sh, rep: sh * rep, x.shape, repetitions))
    # case of empty input
    if x.size == 0:
        return dpt_ext.empty(
            res_shape,
            dtype=x.dtype,
            usm_type=x.usm_type,
            sycl_queue=x.sycl_queue,
        )
    in_sh = x.shape
    if res_shape == in_sh:
        return dpt_ext.copy(x)
    expanded_sh = []
    broadcast_sh = []
    out_sz = 1
    for i in range(len(res_shape)):
        out_sz *= res_shape[i]
        reps, sh = repetitions[i], in_sh[i]
        if reps == 1:
            # dimension will be unchanged
            broadcast_sh.append(sh)
            expanded_sh.append(sh)
        elif sh == 1:
            # dimension will be broadcast
            broadcast_sh.append(reps)
            expanded_sh.append(sh)
        else:
            broadcast_sh.extend([reps, sh])
            expanded_sh.extend([1, sh])
    exec_q = x.sycl_queue
    xdt = x.dtype
    xut = x.usm_type
    res = dpt_ext.empty((out_sz,), dtype=xdt, usm_type=xut, sycl_queue=exec_q)
    # no need to copy data for empty output
    if out_sz > 0:
        x = dpt_ext.broadcast_to(
            # this reshape should never copy
            dpt_ext.reshape(x, expanded_sh),
            broadcast_sh,
        )
        # copy broadcast input into flat array
        _manager = dputils.SequentialOrderManager[exec_q]
        dep_evs = _manager.submitted_events
        hev, cp_ev = ti._copy_usm_ndarray_for_reshape(
            src=x, dst=res, sycl_queue=exec_q, depends=dep_evs
        )
        _manager.add_event_pair(hev, cp_ev)
    return dpt_ext.reshape(res, res_shape)
