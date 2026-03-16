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

# TODO: revert to `import dpctl.tensor...`
# when dpnp fully migrates dpctl/tensor
import dpctl_ext.tensor._type_utils as tu

_integral_dtypes = [
    "i1",
    "u1",
    "i2",
    "u2",
    "i4",
    "u4",
    "i8",
    "u8",
]
_real_fp_dtypes = ["f2", "f4", "f8"]
_complex_fp_dtypes = [
    "c8",
    "c16",
]
_real_value_dtypes = _integral_dtypes + _real_fp_dtypes
_no_complex_dtypes = [
    "b1",
] + _real_value_dtypes
_all_dtypes = _no_complex_dtypes + _complex_fp_dtypes

_usm_types = ["device", "shared", "host"]


def _map_to_device_dtype(dt, dev):
    return tu._to_device_supported_dtype(dt, dev)


def _compare_dtypes(dt, ref_dt, sycl_queue=None):
    assert isinstance(sycl_queue, dpctl.SyclQueue)
    dev = sycl_queue.sycl_device
    expected_dt = _map_to_device_dtype(ref_dt, dev)
    return dt == expected_dt


__all__ = [
    "_no_complex_dtypes",
    "_all_dtypes",
    "_usm_types",
    "_map_to_device_dtype",
    "_compare_dtypes",
]
