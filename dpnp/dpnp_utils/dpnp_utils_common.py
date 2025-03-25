# *****************************************************************************
# Copyright (c) 2023-2025, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
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


from collections.abc import Iterable

from dpctl.tensor._type_utils import _can_cast

import dpnp
from dpnp.dpnp_utils import map_dtype_to_device

__all__ = ["result_type_for_device", "to_supported_dtypes"]


def result_type_for_device(dtypes, device):
    """Works as dpnp.result_type, but taking into account the device capabilities."""

    rt = dpnp.result_type(*dtypes)
    return map_dtype_to_device(rt, device)


def to_supported_dtypes(dtypes, supported_types, device):
    """
    Convert input dtypes to the supported ones based on the device capabilities.
    Return types from the supported_types list that are compatible with the input dtypes.
    If no compatible types are found, return None.
    """

    has_fp64 = device.has_aspect_fp64
    has_fp16 = device.has_aspect_fp16

    def is_castable(dtype, stype):
        return _can_cast(dtype, stype, has_fp16, has_fp64)

    if not isinstance(supported_types, Iterable):
        supported_types = (supported_types,)  # pragma: no cover

    if isinstance(dtypes, Iterable):
        sdtypes_elem = supported_types[0]
        if not isinstance(sdtypes_elem, Iterable):
            raise ValueError(  # pragma: no cover
                "Input and supported types must have the same length"
            )

        typ = type(sdtypes_elem)
        dtypes = typ(dtypes)

    if dtypes in supported_types:
        return dtypes

    for stypes in supported_types:
        if not isinstance(dtypes, Iterable):
            if isinstance(stypes, Iterable):  # pragma: no cover
                raise ValueError(
                    "Input and supported types must have the same length"
                )

            if is_castable(dtypes, stypes):
                return stypes
        else:
            if not isinstance(stypes, Iterable) or len(dtypes) != len(
                stypes
            ):  # pragma: no cover
                raise ValueError(
                    "Input and supported types must have the same length"
                )

            if all(
                is_castable(dtype, stype)
                for dtype, stype in zip(dtypes, stypes)
            ):
                return stypes

    if not isinstance(dtypes, Iterable):  # pragma: no cover
        return None  # pragma: no cover

    return (None,) * len(dtypes)  # pragma: no cover
