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

import dpctl.tensor as dpt
import dpctl.utils as dputils
import numpy as np

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor._tensor_impl as ti

from ._numpy_helper import normalize_axis_tuple

__doc__ = (
    "Implementation module for array manipulation "
    "functions in :module:`dpctl.tensor`"
)


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
