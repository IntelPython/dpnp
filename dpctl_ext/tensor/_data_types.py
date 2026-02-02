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


# TODO: migrate dpctl.tensor._tensor_impl into dpnp
from dpctl.tensor._tensor_impl import (
    default_device_bool_type as ti_default_device_bool_type,
)
from dpctl.tensor._tensor_impl import (
    default_device_complex_type as ti_default_device_complex_type,
)
from dpctl.tensor._tensor_impl import (
    default_device_fp_type as ti_default_device_fp_type,
)
from dpctl.tensor._tensor_impl import (
    default_device_int_type as ti_default_device_int_type,
)
from numpy import bool_ as np_bool_
from numpy import complexfloating as np_complexfloating
from numpy import dtype
from numpy import floating as np_floating
from numpy import integer as np_integer
from numpy import issubdtype as np_issubdtype

bool = dtype("bool")
int8 = dtype("int8")
int16 = dtype("int16")
int32 = dtype("int32")
int64 = dtype("int64")
uint8 = dtype("uint8")
uint16 = dtype("uint16")
uint32 = dtype("uint32")
uint64 = dtype("uint64")
float16 = dtype("float16")
float32 = dtype("float32")
float64 = dtype("float64")
complex64 = dtype("complex64")
complex128 = dtype("complex128")


def _get_dtype(inp_dt, sycl_obj, ref_type=None):
    """
    Type inference utility to construct data type
    object with defaults based on reference type.

    _get_dtype is used by dpctl.tensor.asarray
    to infer data type of the output array from the
    input sequence.
    """
    if inp_dt is None:
        if ref_type in [None, float] or np_issubdtype(ref_type, np_floating):
            fp_dt = ti_default_device_fp_type(sycl_obj)
            return dtype(fp_dt)
        if ref_type in [bool, np_bool_]:
            bool_dt = ti_default_device_bool_type(sycl_obj)
            return dtype(bool_dt)
        if ref_type is int or np_issubdtype(ref_type, np_integer):
            int_dt = ti_default_device_int_type(sycl_obj)
            return dtype(int_dt)
        if ref_type is complex or np_issubdtype(ref_type, np_complexfloating):
            cfp_dt = ti_default_device_complex_type(sycl_obj)
            return dtype(cfp_dt)
        raise TypeError(f"Reference type {ref_type} not recognized.")
    return dtype(inp_dt)


__all__ = [
    "dtype",
    "_get_dtype",
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
