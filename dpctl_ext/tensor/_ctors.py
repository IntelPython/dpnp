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

from numbers import Number

import dpctl
import dpctl.tensor as dpt
import dpctl.utils
import numpy as np
from dpctl.tensor._data_types import _get_dtype
from dpctl.tensor._device import normalize_queue_device

import dpctl_ext.tensor._tensor_impl as ti


def _cast_fill_val(fill_val, dt):
    """
    Casts the Python scalar `fill_val` to another Python type coercible to the
    requested data type `dt`, if necessary.
    """
    val_type = type(fill_val)
    if val_type in [float, complex] and np.issubdtype(dt, np.integer):
        return int(fill_val.real)
    elif val_type is complex and np.issubdtype(dt, np.floating):
        return fill_val.real
    elif val_type is int and np.issubdtype(dt, np.integer):
        return _to_scalar(fill_val, dt)
    else:
        return fill_val


def _to_scalar(obj, sc_ty):
    """A way to convert object to NumPy scalar type.
    Raises OverflowError if obj can not be represented
    using the requested scalar type.
    """
    zd_arr = np.asarray(obj, dtype=sc_ty)
    return zd_arr[()]


def _validate_fill_value(fill_val):
    """Validates that `fill_val` is a numeric or boolean scalar."""
    # TODO: verify if `np.True_` and `np.False_` should be instances of
    # Number in NumPy, like other NumPy scalars and like Python bools
    # check for `np.bool_` separately as NumPy<2 has no `np.bool`
    if not isinstance(fill_val, Number) and not isinstance(fill_val, np.bool_):
        raise TypeError(
            f"array cannot be filled with scalar of type {type(fill_val)}"
        )


def full(
    shape,
    fill_value,
    *,
    dtype=None,
    order="C",
    device=None,
    usm_type=None,
    sycl_queue=None,
):
    """
    Returns a new :class:`dpctl.tensor.usm_ndarray` having a specified
    shape and filled with `fill_value`.

    Args:
        shape (tuple):
            Dimensions of the array to be created.
        fill_value (int,float,complex,usm_ndarray):
            fill value
        dtype (optional): data type of the array. Can be typestring,
            a :class:`numpy.dtype` object, :mod:`numpy` char string,
            or a NumPy scalar type. Default: ``None``
        order ("C", or "F"):
            memory layout for the array. Default: ``"C"``
        device (optional): array API concept of device where the output array
            is created. ``device`` can be ``None``, a oneAPI filter selector
            string, an instance of :class:`dpctl.SyclDevice` corresponding to
            a non-partitioned SYCL device, an instance of
            :class:`dpctl.SyclQueue`, or a :class:`dpctl.tensor.Device` object
            returned by :attr:`dpctl.tensor.usm_ndarray.device`.
            Default: ``None``
        usm_type (``"device"``, ``"shared"``, ``"host"``, optional):
            The type of SYCL USM allocation for the output array.
            Default: ``"device"``
        sycl_queue (:class:`dpctl.SyclQueue`, optional):
            The SYCL queue to use
            for output array allocation and copying. ``sycl_queue`` and
            ``device`` are complementary arguments, i.e. use one or another.
            If both are specified, a :exc:`TypeError` is raised unless both
            imply the same underlying SYCL queue to be used. If both are
            ``None``, a cached queue targeting default-selected device is
            used for allocation and population. Default: ``None``

    Returns:
        usm_ndarray:
            New array initialized with given value.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=True)

    if isinstance(fill_value, (dpt.usm_ndarray, np.ndarray, tuple, list)):
        if (
            isinstance(fill_value, dpt.usm_ndarray)
            and sycl_queue is None
            and device is None
        ):
            sycl_queue = fill_value.sycl_queue
        else:
            sycl_queue = normalize_queue_device(
                sycl_queue=sycl_queue, device=device
            )
        X = dpt.asarray(
            fill_value,
            dtype=dtype,
            order=order,
            usm_type=usm_type,
            sycl_queue=sycl_queue,
        )
        return dpt.copy(dpt.broadcast_to(X, shape), order=order)
    else:
        _validate_fill_value(fill_value)

    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    usm_type = usm_type if usm_type is not None else "device"
    dtype = _get_dtype(dtype, sycl_queue, ref_type=type(fill_value))
    res = dpt.usm_ndarray(
        shape,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    fill_value = _cast_fill_val(fill_value, dtype)

    _manager = dpctl.utils.SequentialOrderManager[sycl_queue]
    # populating new allocation, no dependent events
    hev, full_ev = ti._full_usm_ndarray(fill_value, res, sycl_queue)
    _manager.add_event_pair(hev, full_ev)
    return res
