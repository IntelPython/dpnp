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

"""ctypes mirrors of the DLPack structs, to build tensors in tests"""

import ctypes

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
_PyCapsule_New.restype = ctypes.py_object

kDLInt = 0
kDLUInt = 1
kDLFloat = 2
kDLBfloat = 4
kDLComplex = 5
kDLBool = 6


class DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    pass


class DLManagedTensorVersioned(ctypes.Structure):
    pass


DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))
DLManagedTensorVersionedDeleter = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(DLManagedTensorVersioned)
)

DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", DLManagedTensorDeleter),
]

DLManagedTensorVersioned._fields_ = [
    ("version", DLPackVersion),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", DLManagedTensorVersionedDeleter),
    ("flags", ctypes.c_uint64),
    ("dl_tensor", DLTensor),
]


def make_dl_tensor(
    shape, dl_dtype, device, data=None, strides=None, byte_offset=0
):
    """Builds a DLTensor; the shape and strides arrays are kept alive on
    the returned object"""
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)
    t = DLTensor()
    t.data = data
    t.device = DLDevice(*device)
    t.ndim = ndim
    t.dtype = DLDataType(*dl_dtype)
    t.shape = shape_arr if ndim else None
    t._keep_alive = [shape_arr]
    if strides is not None:
        strides_arr = (ctypes.c_int64 * ndim)(*strides)
        t.strides = strides_arr
        t._keep_alive.append(strides_arr)
    t.byte_offset = byte_offset
    return t


def make_managed_tensor(
    dl_tensor, versioned=True, version=(1, 3), flags=0, deleter=None
):
    """Builds a DLManagedTensorVersioned, or a legacy DLManagedTensor,
    around ``dl_tensor``, with an optional Python ``deleter(tensor_ptr)``"""
    if versioned:
        mt = DLManagedTensorVersioned()
        mt.version = DLPackVersion(*version)
        mt.flags = flags
        deleter_type = DLManagedTensorVersionedDeleter
    else:
        mt = DLManagedTensor()
        deleter_type = DLManagedTensorDeleter
    mt.dl_tensor = dl_tensor
    if deleter is not None:
        mt.deleter = deleter_type(deleter)
    mt._keep_alive = dl_tensor._keep_alive
    return mt


def make_dlpack_capsule(managed_tensor):
    """Wraps ``managed_tensor`` in an unconsumed DLPack capsule without a
    destructor, as a producer that frees nothing on its own would"""
    name = (
        b"dltensor_versioned"
        if isinstance(managed_tensor, DLManagedTensorVersioned)
        else b"dltensor"
    )
    return _PyCapsule_New(ctypes.addressof(managed_tensor), name, None)
