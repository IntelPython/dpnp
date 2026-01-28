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

import numbers
from operator import index
from cpython.buffer cimport PyObject_CheckBuffer
from numpy import ndarray


cdef bint _is_buffer(object o):
    return PyObject_CheckBuffer(o)


cdef Py_ssize_t _slice_len(
    Py_ssize_t sl_start,
    Py_ssize_t sl_stop,
    Py_ssize_t sl_step
):
    """
    Compute len(range(sl_start, sl_stop, sl_step))
    """
    if sl_start == sl_stop:
        return 0
    if sl_step > 0:
        if sl_start > sl_stop:
            return 0
        # 1 + argmax k such htat sl_start + sl_step*k < sl_stop
        return 1 + ((sl_stop - sl_start - 1) // sl_step)
    else:
        if sl_start < sl_stop:
            return 0
        return 1 + ((sl_stop - sl_start + 1) // sl_step)


cdef bint _is_integral(object x) except *:
    """Gives True if x is an integral slice spec"""
    if isinstance(x, (ndarray, usm_ndarray)):
        if x.ndim > 0:
            return False
        if x.dtype.kind not in "ui":
            return False
        return True
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if _is_buffer(x):
        mbuf = memoryview(x)
        if mbuf.ndim == 0:
            f = mbuf.format
            return f in "bBhHiIlLqQ"
        else:
            return False
    if callable(getattr(x, "__index__", None)):
        try:
            index(x)
        except (TypeError, ValueError):
            return False
        return True
    return False


cdef bint _is_boolean(object x) except *:
    """Gives True if x is an integral slice spec"""
    if isinstance(x, (ndarray, usm_ndarray)):
        if x.ndim > 0:
            return False
        if x.dtype.kind not in "b":
            return False
        return True
    if isinstance(x, bool):
        return True
    if isinstance(x, (int, float, complex)):
        return False
    if _is_buffer(x):
        mbuf = memoryview(x)
        if mbuf.ndim == 0:
            f = mbuf.format
            return f in "?"
        else:
            return False
    if callable(getattr(x, "__bool__", None)):
        try:
            x.__bool__()
        except (TypeError, ValueError):
            return False
        return True
    return False


def _basic_slice_meta(ind, shape : tuple, strides : tuple, offset : int):
    """
    Give basic slicing index `ind` and array layout information produce
    a 5-tuple (resulting_shape, resulting_strides, resulting_offset,
       advanced_ind, resulting_advanced_ind_pos)
    used to construct a view into underlying array over which advanced
    indexing, if any, is to be performed.

    Raises IndexError for invalid index `ind`.
    """
    _no_advanced_ind = tuple()
    _no_advanced_pos = -1
    if ind is Ellipsis:
        return (shape, strides, offset, _no_advanced_ind, _no_advanced_pos)
    elif ind is None:
        return (
            (1,) + shape,
            (0,) + strides,
            offset,
            _no_advanced_ind,
            _no_advanced_pos,
        )
    elif isinstance(ind, slice):
        sl_start, sl_stop, sl_step = ind.indices(shape[0])
        sh0 = _slice_len(sl_start, sl_stop, sl_step)
        str0 = sl_step * strides[0]
        new_strides = (
            strides if (sl_step == 1 or sh0 == 0) else (str0,) + strides[1:]
        )
        new_shape = (sh0, ) + shape[1:]
        is_empty = any(sh_i == 0 for sh_i in new_shape)
        new_offset = offset if is_empty else offset + sl_start * strides[0]
        return (
            new_shape,
            new_strides,
            new_offset,
            _no_advanced_ind,
            _no_advanced_pos,
        )
    elif _is_boolean(ind):
        if ind:
            return (
                (1,) + shape,
                (0,) + strides,
                offset,
                _no_advanced_ind,
                _no_advanced_pos,
            )
        else:
            return (
                (0,) + shape,
                (0,) + strides,
                offset,
                _no_advanced_ind,
                _no_advanced_pos,
            )
    elif _is_integral(ind):
        ind = index(ind)
        new_shape = shape[1:]
        new_strides = strides[1:]
        is_empty = any(sh_i == 0 for sh_i in new_shape)
        if 0 <= ind < shape[0]:
            new_offset = offset if is_empty else offset + ind * strides[0]
            return (
                new_shape,
                new_strides,
                new_offset,
                _no_advanced_ind,
                _no_advanced_pos,
            )
        elif -shape[0] <= ind < 0:
            new_offset = (
                offset if is_empty else offset + (shape[0] + ind) * strides[0]
            )
            return (
                new_shape,
                new_strides,
                new_offset,
                _no_advanced_ind,
                _no_advanced_pos,
            )
        else:
            raise IndexError(
                "Index {0} is out of range for axes 0 with "
                "size {1}".format(ind, shape[0]))
    elif isinstance(ind, (ndarray, usm_ndarray)):
        return (shape, strides, offset, (ind,), 0)
    elif isinstance(ind, tuple):
        axes_referenced = 0
        ellipses_count = 0
        newaxis_count = 0
        explicit_index = 0
        seen_arrays_yet = False
        array_streak_started = False
        array_streak_interrupted = False
        for i in ind:
            if i is None:
                newaxis_count += 1
                if array_streak_started:
                    array_streak_interrupted = True
            elif i is Ellipsis:
                ellipses_count += 1
                if array_streak_started:
                    array_streak_interrupted = True
            elif isinstance(i, slice):
                axes_referenced += 1
                if array_streak_started:
                    array_streak_interrupted = True
            elif _is_boolean(i):
                newaxis_count += 1
                if array_streak_started:
                    array_streak_interrupted = True
            elif _is_integral(i):
                axes_referenced += 1
                if not array_streak_started and array_streak_interrupted:
                    explicit_index += 1
            elif isinstance(i, (ndarray, usm_ndarray)):
                if not seen_arrays_yet:
                    seen_arrays_yet = True
                    array_streak_started = True
                    array_streak_interrupted = False
                if array_streak_interrupted:
                    raise IndexError(
                        "Advanced indexing array specs may not be "
                        "separated by basic slicing specs."
                    )
                dt_k = i.dtype.kind
                if dt_k == "b" and i.ndim > 0:
                    axes_referenced += i.ndim
                elif dt_k in "ui" and i.ndim > 0:
                    axes_referenced += 1
                else:
                    raise IndexError(
                        "arrays used as indices must be of integer "
                        "(or boolean) type"
                    )
            else:
                raise IndexError(
                    "Only integers, slices (`:`), ellipsis (`...`), "
                    "dpctl.tensor.newaxis (`None`) and integer and "
                    "boolean arrays are valid indices."
                )
        if ellipses_count > 1:
            raise IndexError(
                "an index can only have a single ellipsis ('...')")
        if axes_referenced > len(shape):
            raise IndexError(
                "too many indices for an array, array is "
                "{0}-dimensional, but {1} were indexed".format(
                    len(shape), axes_referenced))
        if ellipses_count:
            ellipses_count = len(shape) - axes_referenced
        new_shape_len = (newaxis_count + ellipses_count
                         + axes_referenced - explicit_index)
        new_shape = list()
        new_strides = list()
        new_advanced_ind = list()
        k = 0
        new_advanced_start_pos = -1
        advanced_start_pos_set = False
        new_offset = offset
        is_empty = False
        array_streak = False
        for i in range(len(ind)):
            ind_i = ind[i]
            if (ind_i is Ellipsis):
                k_new = k + ellipses_count
                new_shape.extend(shape[k:k_new])
                new_strides.extend(strides[k:k_new])
                if any(dim == 0 for dim in shape[k:k_new]):
                    is_empty = True
                    new_offset = offset
                k = k_new
                if array_streak:
                    array_streak = False
            elif ind_i is None:
                new_shape.append(1)
                new_strides.append(0)
                if array_streak:
                    array_streak = False
            elif isinstance(ind_i, slice):
                k_new = k + 1
                sl_start, sl_stop, sl_step = ind_i.indices(shape[k])
                sh_i = _slice_len(sl_start, sl_stop, sl_step)
                str_i = (1 if sh_i == 0 else sl_step) * strides[k]
                new_shape.append(sh_i)
                new_strides.append(str_i)
                if sh_i > 0 and not is_empty:
                    new_offset = new_offset + sl_start * strides[k]
                if sh_i == 0:
                    is_empty = True
                    new_offset = offset
                k = k_new
                if array_streak:
                    array_streak = False
            elif _is_boolean(ind_i):
                new_shape.append(1 if ind_i else 0)
                new_strides.append(0)
                if array_streak:
                    array_streak = False
            elif _is_integral(ind_i):
                if array_streak:
                    if not isinstance(ind_i, (ndarray, usm_ndarray)):
                        ind_i = index(ind_i)
                        # integer will be converted to an array,
                        # still raise if OOB
                        if not (
                            0 <= ind_i < shape[k] or -shape[k] <= ind_i < 0
                        ):
                            raise IndexError(
                                "Index {0} is out of range for axes "
                                "{1} with size {2}".format(ind_i, k, shape[k])
                            )
                    new_advanced_ind.append(ind_i)
                    k_new = k + 1
                    new_shape.extend(shape[k:k_new])
                    new_strides.extend(strides[k:k_new])
                    k = k_new
                else:
                    ind_i = index(ind_i)
                    if 0 <= ind_i < shape[k]:
                        k_new = k + 1
                        if not is_empty:
                            new_offset = new_offset + ind_i * strides[k]
                        k = k_new
                    elif -shape[k] <= ind_i < 0:
                        k_new = k + 1
                        if not is_empty:
                            new_offset = (
                                new_offset + (shape[k] + ind_i) * strides[k]
                            )
                        k = k_new
                    else:
                        raise IndexError(
                            "Index {0} is out of range for axes "
                            "{1} with size {2}".format(ind_i, k, shape[k])
                        )
            elif isinstance(ind_i, (ndarray, usm_ndarray)):
                if not array_streak:
                    array_streak = True
                if not advanced_start_pos_set:
                    new_advanced_start_pos = len(new_shape)
                    advanced_start_pos_set = True
                new_advanced_ind.append(ind_i)
                dt_k = ind_i.dtype.kind
                if dt_k == "b":
                    k_new = k + ind_i.ndim
                else:
                    k_new = k + 1
                new_shape.extend(shape[k:k_new])
                new_strides.extend(strides[k:k_new])
                k = k_new
        new_shape.extend(shape[k:])
        new_strides.extend(strides[k:])
        new_shape_len += len(shape) - k
        return (
            tuple(new_shape),
            tuple(new_strides),
            new_offset,
            tuple(new_advanced_ind),
            new_advanced_start_pos
        )
    else:
        raise IndexError(
            "Only integers, slices (`:`), ellipsis (`...`), "
            "dpctl.tensor.newaxis (`None`) and integer and "
            "boolean arrays are valid indices."
        )
