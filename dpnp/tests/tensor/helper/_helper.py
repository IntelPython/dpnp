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
import pytest


def has_gpu(backend="opencl"):
    return bool(dpctl.get_num_devices(backend=backend, device_type="gpu"))


def has_cpu(backend="opencl"):
    return bool(dpctl.get_num_devices(backend=backend, device_type="cpu"))


def has_sycl_platforms():
    return bool(len(dpctl.get_platforms()))


def create_invalid_capsule():
    """Creates an invalid capsule for the purpose of testing dpctl
    constructors that accept capsules.
    """
    import ctypes

    ctor = ctypes.pythonapi.PyCapsule_New
    ctor.restype = ctypes.py_object
    ctor.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return ctor(id(ctor), b"invalid", 0)


def get_queue_or_skip(args=()):
    try:
        q = dpctl.SyclQueue(*args)
    except dpctl.SyclQueueCreationError:
        pytest.skip(f"Queue could not be created from {args}")
    return q


def skip_if_dtype_not_supported(dt, q_or_dev):
    import dpnp.tensor as dpt

    dt = dpt.dtype(dt)
    if type(q_or_dev) is dpctl.SyclQueue:
        dev = q_or_dev.sycl_device
    elif type(q_or_dev) is dpctl.SyclDevice:
        dev = q_or_dev
    else:
        raise TypeError(
            "Expected dpctl.SyclQueue or dpctl.SyclDevice, "
            f"got {type(q_or_dev)}"
        )
    dev_has_dp = dev.has_aspect_fp64
    if dev_has_dp is False and dt in [dpt.float64, dpt.complex128]:
        pytest.skip(
            f"{dev.name} does not support double precision floating point types"
        )
    dev_has_hp = dev.has_aspect_fp16
    if dev_has_hp is False and dt in [
        dpt.float16,
    ]:
        pytest.skip(
            f"{dev.name} does not support half precision floating point type"
        )
