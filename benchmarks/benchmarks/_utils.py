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

"""Shared helpers and parameter axes for the dpnp ASV benchmarks."""

import dpctl
import numpy
from asv_runner.benchmarks.mark import SkipNotImplemented

import dpnp

# Keyed by name, so ASV's tables stay readable.
_EXECUTORS = {"dpnp": dpnp, "numpy": numpy}
_EXECUTOR_NAMES = list(_EXECUTORS)

# axes shared across multiple files
_SIZES_1D = [2**16, 2**20, 2**24]
# float only, where an integer input would time a promotion, not the kernel
_FLOAT_DTYPES = ["float64", "float32"]

_DEFAULT_QUEUE = None


def default_queue():
    """Return a queue on dpnp's default device, created on first use."""
    global _DEFAULT_QUEUE

    if _DEFAULT_QUEUE is None:
        _DEFAULT_QUEUE = dpctl.SyclQueue()
    return _DEFAULT_QUEUE


def make_synchronizer(executor):
    """Return a callable blocking until ``executor``'s work has finished."""
    if executor != "dpnp":
        return lambda result: None

    def sync(result):
        # Some results are tuples, e.g. linalg.svd.
        for array in result if isinstance(result, tuple) else (result,):
            dpnp.synchronize_array_data(array)

    return sync


def skip_unsupported_dtype(q, dtype):
    """Skip the benchmark if the device does not support the given dtype."""
    dtype = dpnp.dtype(dtype)
    device = q.sycl_device
    if (
        dtype in (dpnp.float64, dpnp.complex128) and not device.has_aspect_fp64
    ) or (dtype == dpnp.float16 and not device.has_aspect_fp16):
        raise SkipNotImplemented(
            f"Skipping benchmark for {dtype.name} on this device"
            + " as it is not supported."
        )
