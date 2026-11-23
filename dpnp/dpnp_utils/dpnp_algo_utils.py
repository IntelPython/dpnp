# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
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


"""Helpers to place the result of a computation on its inputs' queue."""

import dpctl

import dpnp
from dpnp.tensor import get_coerced_usm_type, get_execution_queue

__all__ = [
    "get_usm_allocations",
    "map_dtype_to_device",
]


def _get_coerced_usm_type(objects):
    types_in_use = [obj.usm_type for obj in objects if hasattr(obj, "usm_type")]
    if len(types_in_use) == 0:
        return None
    if len(types_in_use) == 1:
        return types_in_use[0]

    common_usm_type = get_coerced_usm_type(types_in_use)
    if common_usm_type is None:
        raise ValueError("Input arrays must have coerced USM types")
    return common_usm_type


def _get_common_allocation_queue(objects):
    queues_in_use = [
        obj.sycl_queue for obj in objects if hasattr(obj, "sycl_queue")
    ]
    if len(queues_in_use) == 0:
        return None
    if len(queues_in_use) == 1:
        return queues_in_use[0]

    common_queue = get_execution_queue(queues_in_use)
    if common_queue is None:
        raise ValueError(
            "Input arrays must be allocated on the same SYCL queue"
        )
    return common_queue


def get_usm_allocations(objects):
    """
    Given a list of objects returns a tuple of USM type
    and SYCL queue which can be used for a memory
    allocation and to follow compute follows data
    paradigm, or returns `(None, None)` if the default
    USM type and SYCL queue can be used.
    An exception will be raised, if the paradigm is
    broken for the given list of objects.

    """

    if not isinstance(objects, (list, tuple)):
        raise TypeError(f"Expected a list or a tuple, got {type(objects)}")

    if len(objects) == 0:
        return (None, None)
    return (
        _get_coerced_usm_type(objects),
        _get_common_allocation_queue(objects),
    )


def map_dtype_to_device(dtype, device):
    """Map an input ``dtype`` with type ``device`` may use."""

    dtype = dpnp.dtype(dtype)
    if not hasattr(dtype, "char"):
        raise TypeError(f"Invalid type of input dtype={dtype}")
    if not isinstance(device, dpctl.SyclDevice):
        raise TypeError(f"Invalid type of input device={device}")

    dtc = dtype.char
    if dtc == "?" or dpnp.issubdtype(dtype, dpnp.integer):
        # bool or integer type
        return dtype

    if dpnp.issubdtype(dtype, dpnp.floating):
        if dtc == "f":
            # float32 type
            return dtype
        if dtc == "d" and device.has_aspect_fp64:
            return dtype
        if dtc == "e" and device.has_aspect_fp16:
            return dtype
        # float32 is default floating type
        return dpnp.dtype("f4")

    if dpnp.issubdtype(dtype, dpnp.complexfloating):
        if dtc == "F":
            # complex64 type
            return dtype
        if dtc == "D" and device.has_aspect_fp64:
            return dtype
        # complex64 is default complex type
        return dpnp.dtype("c8")
    raise RuntimeError(f"Unrecognized type of input dtype={dtype}")
