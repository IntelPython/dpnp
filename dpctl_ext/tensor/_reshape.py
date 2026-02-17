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
import dpctl.utils
import numpy as np
from dpctl.tensor._tensor_impl import (
    _copy_usm_ndarray_for_reshape,
    _ravel_multi_index,
    _unravel_index,
)

__doc__ = "Implementation module for :func:`dpctl.tensor.reshape`."


def _make_unit_indexes(shape):
    """
    Construct a diagonal matrix with with one on the diagonal
    except if the corresponding element of shape is 1.
    """
    nd = len(shape)
    mi = np.zeros((nd, nd), dtype="u4")
    for i, dim in enumerate(shape):
        mi[i, i] = 1 if dim > 1 else 0
    return mi


def ti_unravel_index(flat_index, shape, order="C"):
    return _unravel_index(flat_index, shape, order)


def ti_ravel_multi_index(multi_index, shape, order="C"):
    return _ravel_multi_index(multi_index, shape, order)


def reshaped_strides(old_sh, old_sts, new_sh, order="C"):
    """
    When reshaping array with `old_sh` shape and `old_sts` strides
    into the new shape `new_sh`, returns the new stride if the reshape
    can be a view, otherwise returns `None`.
    """
    eye_new_mi = _make_unit_indexes(new_sh)
    new_sts = [
        sum(
            st_i * ind_i
            for st_i, ind_i in zip(
                old_sts, ti_unravel_index(flat_index, old_sh, order=order)
            )
        )
        for flat_index in [
            ti_ravel_multi_index(unitvec, new_sh, order=order)
            for unitvec in eye_new_mi
        ]
    ]
    eye_old_mi = _make_unit_indexes(old_sh)
    check_sts = [
        sum(
            st_i * ind_i
            for st_i, ind_i in zip(
                new_sts, ti_unravel_index(flat_index, new_sh, order=order)
            )
        )
        for flat_index in [
            ti_ravel_multi_index(unitvec, old_sh, order=order)
            for unitvec in eye_old_mi
        ]
    ]
    valid = all(
        check_st == old_st or old_dim == 1
        for check_st, old_st, old_dim in zip(check_sts, old_sts, old_sh)
    )
    return new_sts if valid else None


def reshape(X, /, shape, *, order="C", copy=None):
    """reshape(x, shape, order="C")

    Reshapes array ``x`` into new shape.

    Args:
        x (usm_ndarray):
            input array
        shape (Tuple[int]):
            the desired shape of the resulting array.
        order ("C", "F", optional):
            memory layout of the resulting array
            if a copy is found to be necessary. Supported
            choices are ``"C"`` for C-contiguous, or row-major layout;
            and ``"F"`` for F-contiguous, or column-major layout.

    Returns:
        usm_ndarray:
            Reshaped array is a view, if possible,
            and a copy otherwise with memory layout as indicated
            by ``order`` keyword.
    """
    if not isinstance(X, dpt.usm_ndarray):
        raise TypeError
    if not isinstance(shape, (list, tuple)):
        shape = (shape,)
    if order in "cfCF":
        order = order.upper()
    else:
        raise ValueError(
            f"Keyword 'order' not recognized. Expecting 'C' or 'F', got {order}"
        )
    if copy not in (True, False, None):
        raise ValueError(
            f"Keyword 'copy' not recognized. Expecting True, False, "
            f"or None, got {copy}"
        )
    shape = [operator.index(d) for d in shape]
    negative_ones_count = 0
    for nshi in shape:
        if nshi == -1:
            negative_ones_count = negative_ones_count + 1
        if (nshi < -1) or negative_ones_count > 1:
            raise ValueError(
                "Target shape should have at most 1 negative "
                "value which can only be -1"
            )
    if negative_ones_count:
        sz = -np.prod(shape)
        if sz == 0:
            raise ValueError(
                f"Can not reshape array of size {X.size} into "
                f"shape {tuple(i for i in shape if i >= 0)}"
            )
        v = X.size // sz
        shape = [v if d == -1 else d for d in shape]
    if X.size != np.prod(shape):
        raise ValueError(f"Can not reshape into {shape}")
    if X.size:
        newsts = reshaped_strides(X.shape, X.strides, shape, order=order)
    else:
        newsts = (1,) * len(shape)
    copy_required = newsts is None
    if copy_required and (copy is False):
        raise ValueError(
            "Reshaping the array requires a copy, but no copying was "
            "requested by using copy=False"
        )
    copy_q = X.sycl_queue
    if copy_required or (copy is True):
        # must perform a copy
        copy_q = X.sycl_queue
        flat_res = dpt.usm_ndarray(
            (X.size,),
            dtype=X.dtype,
            buffer=X.usm_type,
            buffer_ctor_kwargs={"queue": copy_q},
        )
        _manager = dpctl.utils.SequentialOrderManager[copy_q]
        dep_evs = _manager.submitted_events
        if order == "C":
            hev, r_e = _copy_usm_ndarray_for_reshape(
                src=X, dst=flat_res, sycl_queue=copy_q, depends=dep_evs
            )
        else:
            X_t = dpt.permute_dims(X, range(X.ndim - 1, -1, -1))
            hev, r_e = _copy_usm_ndarray_for_reshape(
                src=X_t, dst=flat_res, sycl_queue=copy_q, depends=dep_evs
            )
        _manager.add_event_pair(hev, r_e)
        return dpt.usm_ndarray(
            tuple(shape), dtype=X.dtype, buffer=flat_res, order=order
        )
    # can form a view
    if (len(shape) == X.ndim) and all(
        s1 == s2 for s1, s2 in zip(shape, X.shape)
    ):
        return X
    return dpt.usm_ndarray(
        shape,
        dtype=X.dtype,
        buffer=X,
        strides=tuple(newsts),
        offset=X._element_offset,
    )
