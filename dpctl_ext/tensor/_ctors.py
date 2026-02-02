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

import dpctl
import dpctl.utils

# TODO: revert to `import dpctl.tensor as dpt`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor as dpt

# TODO: revert to `from dpctl.tensor._data_types import ...`
from dpctl_ext.tensor._data_types import _get_dtype

# TODO: revert to `from dpctl.tensor._device import  ...`
from dpctl_ext.tensor._device import normalize_queue_device

__doc__ = "Implementation of creation functions in :module:`dpctl.tensor`"


def _ensure_native_dtype_device_support(dtype, dev) -> None:
    """Check that dtype is natively supported by device.

    Arg:
        dtype:
            Elemental data-type
        dev (:class:`dpctl.SyclDevice`):
            The device about which the query is being made.
    Returns:
        None
    Raise:
        ValueError:
            if device does not natively support this `dtype`.
    """
    if dtype in [dpt.float64, dpt.complex128] and not dev.has_aspect_fp64:
        raise ValueError(
            f"Device {dev.name} does not provide native support "
            "for double-precision floating point type."
        )
    if (
        dtype
        in [
            dpt.float16,
        ]
        and not dev.has_aspect_fp16
    ):
        raise ValueError(
            f"Device {dev.name} does not provide native support "
            "for half-precision floating point type."
        )


def empty(
    shape,
    *,
    dtype=None,
    order="C",
    device=None,
    usm_type="device",
    sycl_queue=None,
):
    """
    Creates :class:`dpctl.tensor.usm_ndarray` from uninitialized
    USM allocation.

    Args:
        shape (Tuple[int], int):
            Dimensions of the array to be created.
        dtype (optional):
            data type of the array. Can be typestring,
            a :class:`numpy.dtype` object, :mod:`numpy` char string,
            or a NumPy scalar type. The ``None`` value creates an
            array of floating point data type. Default: ``None``
        order (``"C"``, or ``F"``):
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
            Created empty array.
    """
    if not isinstance(order, str) or len(order) == 0 or order[0] not in "CcFf":
        raise ValueError(
            "Unrecognized order keyword value, expecting 'F' or 'C'."
        )
    order = order[0].upper()
    dpctl.utils.validate_usm_type(usm_type, allow_none=False)
    sycl_queue = normalize_queue_device(sycl_queue=sycl_queue, device=device)
    dtype = _get_dtype(dtype, sycl_queue)
    _ensure_native_dtype_device_support(dtype, sycl_queue.sycl_device)
    res = dpt.usm_ndarray(
        shape,
        dtype=dtype,
        buffer=usm_type,
        order=order,
        buffer_ctor_kwargs={"queue": sycl_queue},
    )
    return res
