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

import dpctl.memory as dpm
import dpctl.tensor as dpt
import numpy as np
from dpctl.tensor._usmarray import _is_object_with_buffer_protocol as _is_buffer

from ._type_utils import (
    WeakBooleanType,
    WeakComplexType,
    WeakFloatingType,
    WeakIntegralType,
    _to_device_supported_dtype,
)


def _get_queue_usm_type(o):
    """Return SYCL device where object `o` allocated memory, or None."""
    if isinstance(o, dpt.usm_ndarray):
        return o.sycl_queue, o.usm_type
    elif hasattr(o, "__sycl_usm_array_interface__"):
        try:
            m = dpm.as_usm_memory(o)
            return m.sycl_queue, m.get_usm_type()
        except Exception:
            return None, None
    return None, None


def _get_dtype(o, dev):
    if isinstance(o, dpt.usm_ndarray):
        return o.dtype
    if hasattr(o, "__sycl_usm_array_interface__"):
        return dpt.asarray(o).dtype
    if _is_buffer(o):
        host_dt = np.array(o).dtype
        dev_dt = _to_device_supported_dtype(host_dt, dev)
        return dev_dt
    if hasattr(o, "dtype"):
        dev_dt = _to_device_supported_dtype(o.dtype, dev)
        return dev_dt
    if isinstance(o, bool):
        return WeakBooleanType(o)
    if isinstance(o, int):
        return WeakIntegralType(o)
    if isinstance(o, float):
        return WeakFloatingType(o)
    if isinstance(o, complex):
        return WeakComplexType(o)
    return np.object_


def _validate_dtype(dt) -> bool:
    return isinstance(
        dt,
        (WeakBooleanType, WeakIntegralType, WeakFloatingType, WeakComplexType),
    ) or (
        isinstance(dt, dpt.dtype)
        and dt
        in [
            dpt.bool,
            dpt.int8,
            dpt.uint8,
            dpt.int16,
            dpt.uint16,
            dpt.int32,
            dpt.uint32,
            dpt.int64,
            dpt.uint64,
            dpt.float16,
            dpt.float32,
            dpt.float64,
            dpt.complex64,
            dpt.complex128,
        ]
    )


def _get_shape(o):
    if isinstance(o, dpt.usm_ndarray):
        return o.shape
    if _is_buffer(o):
        return memoryview(o).shape
    if isinstance(o, numbers.Number):
        return ()
    return getattr(o, "shape", tuple())


__all__ = [
    "_get_dtype",
    "_get_queue_usm_type",
    "_get_shape",
    "_validate_dtype",
]
