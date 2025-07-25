# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2025, Intel Corporation
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

import dpctl.tensor as dpt
from dpctl.memory import MemoryUSMDevice as DPCTLMemoryUSMDevice
from dpctl.memory import MemoryUSMHost as DPCTLMemoryUSMHost
from dpctl.memory import MemoryUSMShared as DPCTLMemoryUSMShared


def _add_ptr_property(cls):
    _storage_attr = "_ptr"

    @property
    def ptr(self):
        """
        Returns USM pointer to the start of array (element with zero
        multi-index) encoded as integer.

        """

        return getattr(self, _storage_attr, None)

    @ptr.setter
    def ptr(self, value):
        setattr(self, _storage_attr, value)

    cls.ptr = ptr
    return cls


@_add_ptr_property
class MemoryUSMDevice(DPCTLMemoryUSMDevice):
    pass


@_add_ptr_property
class MemoryUSMHost(DPCTLMemoryUSMHost):
    pass


@_add_ptr_property
class MemoryUSMShared(DPCTLMemoryUSMShared):
    pass


def create_data(x):
    """
    Create an instance of :class:`.MemoryUSMDevice`, :class:`.MemoryUSMHost`,
    or :class:`.MemoryUSMShared` class depending on the type of USM allocation.

    Parameters
    ----------
    x : usm_ndarray
        Input array of :class:`dpctl.tensor.usm_ndarray` type.

    Returns
    -------
    out : {MemoryUSMDevice, MemoryUSMHost, MemoryUSMShared}
        A data object with a reference on USM memory.

    """

    dispatch = {
        DPCTLMemoryUSMDevice: MemoryUSMDevice,
        DPCTLMemoryUSMHost: MemoryUSMHost,
        DPCTLMemoryUSMShared: MemoryUSMShared,
    }

    if not isinstance(x, dpt.usm_ndarray):
        raise TypeError(
            f"An array must be any of supported type, but got {type(x)}"
        )
    usm_data = x.usm_data

    if isinstance(usm_data, tuple(dispatch.values())):
        return usm_data

    cls = dispatch.get(type(usm_data), None)
    if cls:
        data = cls(usm_data)
        # `ptr` is expecting to point at the start of the array's data,
        # while `usm_data._pointer` is a pointer at the start of memory buffer
        data.ptr = x._pointer
        return data
    raise TypeError(f"Expected USM memory, but got {type(usm_data)}")
