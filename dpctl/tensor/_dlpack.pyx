//*****************************************************************************
// Copyright (c) 2026, Intel Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of the copyright holder nor the names of its contributors
//   may be used to endorse or promote products derived from this software
//   without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//*****************************************************************************

# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True

cdef extern from "numpy/npy_no_deprecated_api.h":
    pass

cimport cpython
cimport dpctl as c_dpctl
cimport dpctl.memory as c_dpmem
from dpctl._backend cimport (
    DPCTLDevice_Delete,
    DPCTLDevice_GetParentDevice,
    DPCTLSyclDeviceRef,
    DPCTLSyclUSMRef,
)
from dpctl._sycl_queue_manager cimport get_device_cached_queue
from libc cimport stdlib
from libc.stdint cimport int64_t, uint8_t, uint16_t, uint32_t, uint64_t
from numpy cimport ndarray

from ._usmarray cimport (
    USM_ARRAY_C_CONTIGUOUS,
    USM_ARRAY_F_CONTIGUOUS,
    USM_ARRAY_WRITABLE,
    usm_ndarray,
)

import ctypes

import dpctl
import dpctl.memory as dpmem
import numpy as np

from ._device import Device


cdef extern from "dlpack/dlpack.h" nogil:
    cdef int DLPACK_MAJOR_VERSION

    cdef int DLPACK_MINOR_VERSION

    cdef int DLPACK_FLAG_BITMASK_READ_ONLY

    cdef int DLPACK_FLAG_BITMASK_IS_COPIED

    ctypedef struct DLPackVersion:
        uint32_t major
        uint32_t minor

    cdef enum DLDeviceType:
        kDLCPU
        kDLCUDA
        kDLCUDAHost
        kDLCUDAManaged
        kDLROCM
        kDLROCMHost
        kDLOpenCL
        kDLVulkan
        kDLMetal
        kDLVPI
        kDLOneAPI
        kDLWebGPU
        kDLHexagon
        kDLMAIA
        kDLTrn

    ctypedef struct DLDevice:
        DLDeviceType device_type
        int device_id

    cdef enum DLDataTypeCode:
        kDLInt
        kDLUInt
        kDLFloat
        kDLBfloat
        kDLComplex
        kDLBool
        kDLFloat8_e3m4
        kDLFloat8_e4m3
        kDLFloat8_e4m3b11fnuz
        kDLFloat8_e4m3fn
        kDLFloat8_e4m3fnuz
        kDLFloat8_e5m2
        kDLFloat8_e5m2fnuz
        kDLFloat8_e8m0fnu
        kDLFloat6_e2m3fn
        kDLFloat6_e3m2fn
        kDLFloat4_e2m1fn

    ctypedef struct DLDataType:
        uint8_t code
        uint8_t bits
        uint16_t lanes

    ctypedef struct DLTensor:
        void *data
        DLDevice device
        int ndim
        DLDataType dtype
        int64_t *shape
        int64_t *strides
        uint64_t byte_offset

    ctypedef struct DLManagedTensor:
        DLTensor dl_tensor
        void *manager_ctx
        void (*deleter)(DLManagedTensor *)  # noqa: E211

    ctypedef struct DLManagedTensorVersioned:
        DLPackVersion version
        void *manager_ctx
        void (*deleter)(DLManagedTensorVersioned *)  # noqa: E211
        uint64_t flags
        DLTensor dl_tensor


def get_build_dlpack_version():
    """
    Returns a tuple of integers representing the `major` and `minor`
    version of DLPack :module:`dpctl.tensor` was built with.
    This tuple can be passed as the `max_version` argument to
    `__dlpack__` to guarantee module:`dpctl.tensor` can properly
    consume capsule.

    Returns:
        Tuple[int, int]
            A tuple of integers representing the `major` and `minor`
            version of DLPack used to build :module:`dpctl.tensor`.
    """
    return (DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION)


cdef void _pycapsule_deleter(object dlt_capsule) noexcept:
    cdef DLManagedTensor *dlm_tensor = NULL
    if cpython.PyCapsule_IsValid(dlt_capsule, "dltensor"):
        dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
            dlt_capsule, "dltensor")
        dlm_tensor.deleter(dlm_tensor)


cdef void _managed_tensor_deleter(
    DLManagedTensor *dlm_tensor
) noexcept with gil:
    if dlm_tensor is not NULL:
        # we only delete shape, because we make single allocation to
        # accommodate both shape and strides if strides are needed
        stdlib.free(dlm_tensor.dl_tensor.shape)
        cpython.Py_DECREF(<object>dlm_tensor.manager_ctx)
        dlm_tensor.manager_ctx = NULL
        stdlib.free(dlm_tensor)


cdef void _pycapsule_versioned_deleter(object dlt_capsule) noexcept:
    cdef DLManagedTensorVersioned *dlmv_tensor = NULL
    if cpython.PyCapsule_IsValid(dlt_capsule, "dltensor_versioned"):
        dlmv_tensor = <DLManagedTensorVersioned*>cpython.PyCapsule_GetPointer(
            dlt_capsule, "dltensor_versioned")
        dlmv_tensor.deleter(dlmv_tensor)


cdef void _managed_tensor_versioned_deleter(
    DLManagedTensorVersioned *dlmv_tensor
) noexcept with gil:
    if dlmv_tensor is not NULL:
        # we only delete shape, because we make single allocation to
        # accommodate both shape and strides if strides are needed
        stdlib.free(dlmv_tensor.dl_tensor.shape)
        cpython.Py_DECREF(<object>dlmv_tensor.manager_ctx)
        dlmv_tensor.manager_ctx = NULL
        stdlib.free(dlmv_tensor)


cdef object _get_default_context(c_dpctl.SyclDevice dev):
    try:
        default_context = dev.sycl_platform.default_context
    except RuntimeError:
        # RT does not support default_context
        default_context = None

    return default_context

cdef int get_array_dlpack_device_id(
    usm_ndarray usm_ary
) except -1:
    """Finds ordinal number of the parent of device where array
    was allocated.
    """
    cdef c_dpctl.SyclQueue ary_sycl_queue
    cdef c_dpctl.SyclDevice ary_sycl_device
    cdef DPCTLSyclDeviceRef pDRef = NULL
    cdef int device_id = -1

    ary_sycl_queue = usm_ary.get_sycl_queue()
    ary_sycl_device = ary_sycl_queue.get_sycl_device()

    default_context = _get_default_context(ary_sycl_device)
    if default_context is None:
        # check that ary_sycl_device is a non-partitioned device
        pDRef = DPCTLDevice_GetParentDevice(ary_sycl_device.get_device_ref())
        if pDRef is not NULL:
            DPCTLDevice_Delete(pDRef)
            raise DLPackCreationError(
                "to_dlpack_capsule: DLPack can only export arrays allocated "
                "on non-partitioned SYCL devices on platforms where "
                "default_context oneAPI extension is not supported."
            )
    else:
        if not usm_ary.sycl_context == default_context:
            raise DLPackCreationError(
                "to_dlpack_capsule: DLPack can only export arrays based on USM "
                "allocations bound to a default platform SYCL context"
            )
    device_id = ary_sycl_device.get_device_id()

    if device_id < 0:
        raise DLPackCreationError(
            "get_array_dlpack_device_id: failed to determine device_id"
        )

    return device_id


cpdef to_dlpack_capsule(usm_ndarray usm_ary):
    """
    to_dlpack_capsule(usm_ary)

    Constructs named Python capsule object referencing
    instance of ``DLManagedTensor`` from
    :class:`dpctl.tensor.usm_ndarray` instance.

    Args:
        usm_ary: An instance of :class:`dpctl.tensor.usm_ndarray`
    Returns:
        A new capsule with name ``"dltensor"`` that contains
        a pointer to ``DLManagedTensor`` struct.
    Raises:
        DLPackCreationError: when array can be represented as
            DLPack tensor. This may happen when array was allocated
            on a partitioned sycl device, or its USM allocation is
            not bound to the platform default SYCL context.
        MemoryError: when host allocation to needed for ``DLManagedTensor``
            did not succeed.
        ValueError: when array elements data type could not be represented
            in ``DLManagedTensor``.
    """
    cdef DLManagedTensor *dlm_tensor = NULL
    cdef DLTensor *dl_tensor = NULL
    cdef int nd = usm_ary.get_ndim()
    cdef char *data_ptr = usm_ary.get_data()
    cdef Py_ssize_t *shape_ptr = NULL
    cdef Py_ssize_t *strides_ptr = NULL
    cdef int64_t *shape_strides_ptr = NULL
    cdef int i = 0
    cdef int device_id = -1
    cdef int flags = 0
    cdef Py_ssize_t element_offset = 0
    cdef Py_ssize_t byte_offset = 0
    cdef Py_ssize_t si = 1

    ary_base = usm_ary.get_base()

    device_id = get_array_dlpack_device_id(usm_ary)

    dlm_tensor = <DLManagedTensor *> stdlib.malloc(
        sizeof(DLManagedTensor))
    if dlm_tensor is NULL:
        raise MemoryError(
            "to_dlpack_capsule: Could not allocate memory for DLManagedTensor"
        )
    if nd > 0:
        shape_strides_ptr = <int64_t *>stdlib.malloc((sizeof(int64_t) * 2) * nd)
        if shape_strides_ptr is NULL:
            stdlib.free(dlm_tensor)
            raise MemoryError(
                "to_dlpack_capsule: Could not allocate memory for shape/strides"
            )
        shape_ptr = usm_ary.get_shape()
        for i in range(nd):
            shape_strides_ptr[i] = shape_ptr[i]
        strides_ptr = usm_ary.get_strides()
        flags = usm_ary.flags_
        if strides_ptr:
            for i in range(nd):
                shape_strides_ptr[nd + i] = strides_ptr[i]
        else:
            if flags & USM_ARRAY_C_CONTIGUOUS:
                si = 1
                for i in range(nd - 1, -1, -1):
                    shape_strides_ptr[nd + i] = si
                    si = si * shape_ptr[i]
            elif flags & USM_ARRAY_F_CONTIGUOUS:
                si = 1
                for i in range(0, nd):
                    shape_strides_ptr[nd + i] = si
                    si = si * shape_ptr[i]
            else:
                stdlib.free(shape_strides_ptr)
                stdlib.free(dlm_tensor)
                raise BufferError(
                    "to_dlpack_capsule: Invalid array encountered "
                    "when building strides"
                )

            strides_ptr = <Py_ssize_t *>&shape_strides_ptr[nd]

    ary_dt = usm_ary.dtype
    ary_dtk = ary_dt.kind
    element_offset = usm_ary.get_offset()
    byte_offset = element_offset * (<Py_ssize_t>ary_dt.itemsize)

    dl_tensor = &dlm_tensor.dl_tensor
    dl_tensor.data = <void*>(data_ptr - byte_offset)
    dl_tensor.ndim = nd
    dl_tensor.byte_offset = <uint64_t>byte_offset
    dl_tensor.shape = &shape_strides_ptr[0] if nd > 0 else NULL
    dl_tensor.strides = &shape_strides_ptr[nd] if nd > 0 else NULL
    dl_tensor.device.device_type = kDLOneAPI
    dl_tensor.device.device_id = device_id
    dl_tensor.dtype.lanes = <uint16_t>1
    dl_tensor.dtype.bits = <uint8_t>(ary_dt.itemsize * 8)
    if (ary_dtk == "b"):
        dl_tensor.dtype.code = <uint8_t>kDLBool
    elif (ary_dtk == "u"):
        dl_tensor.dtype.code = <uint8_t>kDLUInt
    elif (ary_dtk == "i"):
        dl_tensor.dtype.code = <uint8_t>kDLInt
    elif (ary_dtk == "f"):
        dl_tensor.dtype.code = <uint8_t>kDLFloat
    elif (ary_dtk == "c"):
        dl_tensor.dtype.code = <uint8_t>kDLComplex
    else:
        stdlib.free(shape_strides_ptr)
        stdlib.free(dlm_tensor)
        raise ValueError("Unrecognized array data type")

    dlm_tensor.manager_ctx = <void*>ary_base
    cpython.Py_INCREF(ary_base)
    dlm_tensor.deleter = _managed_tensor_deleter

    return cpython.PyCapsule_New(dlm_tensor, "dltensor", _pycapsule_deleter)


cpdef to_dlpack_versioned_capsule(usm_ndarray usm_ary, bint copied):
    """
    to_dlpack_versioned_capsule(usm_ary, copied)

    Constructs named Python capsule object referencing
    instance of ``DLManagedTensorVersioned`` from
    :class:`dpctl.tensor.usm_ndarray` instance.

    Args:
        usm_ary: An instance of :class:`dpctl.tensor.usm_ndarray`
        copied: A bint representing whether the data was previously
            copied in order to set the flags with the is-copied
            bitmask.
    Returns:
        A new capsule with name ``"dltensor_versioned"`` that
        contains a pointer to ``DLManagedTensorVersioned`` struct.
    Raises:
        DLPackCreationError: when array can be represented as
            DLPack tensor. This may happen when array was allocated
            on a partitioned sycl device, or its USM allocation is
            not bound to the platform default SYCL context.
        MemoryError: when host allocation to needed for
            ``DLManagedTensorVersioned`` did not succeed.
        ValueError: when array elements data type could not be represented
            in ``DLManagedTensorVersioned``.
    """
    cdef DLManagedTensorVersioned *dlmv_tensor = NULL
    cdef DLTensor *dl_tensor = NULL
    cdef uint32_t dlmv_flags = 0
    cdef int nd = usm_ary.get_ndim()
    cdef char *data_ptr = usm_ary.get_data()
    cdef Py_ssize_t *shape_ptr = NULL
    cdef Py_ssize_t *strides_ptr = NULL
    cdef int64_t *shape_strides_ptr = NULL
    cdef int i = 0
    cdef int device_id = -1
    cdef int flags = 0
    cdef Py_ssize_t element_offset = 0
    cdef Py_ssize_t byte_offset = 0
    cdef Py_ssize_t si = 1

    ary_base = usm_ary.get_base()

    # Find ordinal number of the parent device
    device_id = get_array_dlpack_device_id(usm_ary)

    dlmv_tensor = <DLManagedTensorVersioned *> stdlib.malloc(
        sizeof(DLManagedTensorVersioned))
    if dlmv_tensor is NULL:
        raise MemoryError(
            "to_dlpack_versioned_capsule: Could not allocate memory "
            "for DLManagedTensorVersioned"
        )
    if nd > 0:
        shape_strides_ptr = <int64_t *>stdlib.malloc((sizeof(int64_t) * 2) * nd)
        if shape_strides_ptr is NULL:
            stdlib.free(dlmv_tensor)
            raise MemoryError(
                "to_dlpack_versioned_capsule: Could not allocate memory "
                "for shape/strides"
            )
        # this can be a separate function for handling shapes and strides
        shape_ptr = usm_ary.get_shape()
        for i in range(nd):
            shape_strides_ptr[i] = shape_ptr[i]
        strides_ptr = usm_ary.get_strides()
        flags = usm_ary.flags_
        if strides_ptr:
            for i in range(nd):
                shape_strides_ptr[nd + i] = strides_ptr[i]
        else:
            if flags & USM_ARRAY_C_CONTIGUOUS:
                si = 1
                for i in range(nd - 1, -1, -1):
                    shape_strides_ptr[nd + i] = si
                    si = si * shape_ptr[i]
            elif flags & USM_ARRAY_F_CONTIGUOUS:
                si = 1
                for i in range(0, nd):
                    shape_strides_ptr[nd + i] = si
                    si = si * shape_ptr[i]
            else:
                stdlib.free(shape_strides_ptr)
                stdlib.free(dlmv_tensor)
                raise BufferError(
                    "to_dlpack_versioned_capsule: Invalid array encountered "
                    "when building strides"
                )

            strides_ptr = <Py_ssize_t *>&shape_strides_ptr[nd]

    # this can all be a function for building the dl_tensor
    # object (separate from dlm/dlmv)
    ary_dt = usm_ary.dtype
    ary_dtk = ary_dt.kind
    element_offset = usm_ary.get_offset()
    byte_offset = element_offset * (<Py_ssize_t>ary_dt.itemsize)

    dl_tensor = &dlmv_tensor.dl_tensor
    dl_tensor.data = <void*>(data_ptr - byte_offset)
    dl_tensor.ndim = nd
    dl_tensor.byte_offset = <uint64_t>byte_offset
    dl_tensor.shape = &shape_strides_ptr[0] if nd > 0 else NULL
    dl_tensor.strides = &shape_strides_ptr[nd] if nd > 0 else NULL
    dl_tensor.device.device_type = kDLOneAPI
    dl_tensor.device.device_id = device_id
    dl_tensor.dtype.lanes = <uint16_t>1
    dl_tensor.dtype.bits = <uint8_t>(ary_dt.itemsize * 8)
    if (ary_dtk == "b"):
        dl_tensor.dtype.code = <uint8_t>kDLBool
    elif (ary_dtk == "u"):
        dl_tensor.dtype.code = <uint8_t>kDLUInt
    elif (ary_dtk == "i"):
        dl_tensor.dtype.code = <uint8_t>kDLInt
    elif (ary_dtk == "f"):
        dl_tensor.dtype.code = <uint8_t>kDLFloat
    elif (ary_dtk == "c"):
        dl_tensor.dtype.code = <uint8_t>kDLComplex
    else:
        stdlib.free(shape_strides_ptr)
        stdlib.free(dlmv_tensor)
        raise ValueError("Unrecognized array data type")

    # set flags down here
    if copied:
        dlmv_flags |= DLPACK_FLAG_BITMASK_IS_COPIED
    if not (flags & USM_ARRAY_WRITABLE):
        dlmv_flags |= DLPACK_FLAG_BITMASK_READ_ONLY
    dlmv_tensor.flags = dlmv_flags

    dlmv_tensor.version.major = DLPACK_MAJOR_VERSION
    dlmv_tensor.version.minor = DLPACK_MINOR_VERSION

    dlmv_tensor.manager_ctx = <void*>ary_base
    cpython.Py_INCREF(ary_base)
    dlmv_tensor.deleter = _managed_tensor_versioned_deleter

    return cpython.PyCapsule_New(
        dlmv_tensor, "dltensor_versioned", _pycapsule_versioned_deleter
    )


cpdef numpy_to_dlpack_versioned_capsule(ndarray npy_ary, bint copied):
    """
    to_dlpack_versioned_capsule(npy_ary, copied)

    Constructs named Python capsule object referencing
    instance of ``DLManagedTensorVersioned`` from
    :class:`numpy.ndarray` instance.

    Args:
        npy_ary: An instance of :class:`numpy.ndarray`
        copied: A bint representing whether the data was previously
            copied in order to set the flags with the is-copied
            bitmask.
    Returns:
        A new capsule with name ``"dltensor_versioned"`` that
        contains a pointer to ``DLManagedTensorVersioned`` struct.
    Raises:
        DLPackCreationError: when array can be represented as
            DLPack tensor.
        MemoryError: when host allocation to needed for
            ``DLManagedTensorVersioned`` did not succeed.
        ValueError: when array elements data type could not be represented
            in ``DLManagedTensorVersioned``.
    """
    cdef DLManagedTensorVersioned *dlmv_tensor = NULL
    cdef DLTensor *dl_tensor = NULL
    cdef uint32_t dlmv_flags = 0
    cdef int nd = npy_ary.ndim
    cdef int64_t *shape_strides_ptr = NULL
    cdef int i = 0
    cdef Py_ssize_t byte_offset = 0
    cdef int itemsize = npy_ary.itemsize

    dlmv_tensor = <DLManagedTensorVersioned *> stdlib.malloc(
        sizeof(DLManagedTensorVersioned))
    if dlmv_tensor is NULL:
        raise MemoryError(
            "numpy_to_dlpack_versioned_capsule: Could not allocate memory "
            "for DLManagedTensorVersioned"
        )

    shape = npy_ary.ctypes.shape_as(ctypes.c_int64)
    strides = npy_ary.ctypes.strides_as(ctypes.c_int64)
    if nd > 0:
        if npy_ary.size != 1:
            for i in range(nd):
                if shape[i] != 1 and strides[i] % itemsize != 0:
                    stdlib.free(dlmv_tensor)
                    raise BufferError(
                        "numpy_to_dlpack_versioned_capsule: DLPack cannot "
                        "encode an array if strides are not a multiple of "
                        "itemsize"
                    )
        shape_strides_ptr = <int64_t *>stdlib.malloc((sizeof(int64_t) * 2) * nd)
        if shape_strides_ptr is NULL:
            stdlib.free(dlmv_tensor)
            raise MemoryError(
                "numpy_to_dlpack_versioned_capsule: Could not allocate memory "
                "for shape/strides"
            )
        for i in range(nd):
            shape_strides_ptr[i] = shape[i]
            shape_strides_ptr[nd + i] = strides[i] // itemsize

    writable_flag = npy_ary.flags["W"]

    ary_dt = npy_ary.dtype
    ary_dtk = ary_dt.kind

    dl_tensor = &dlmv_tensor.dl_tensor
    dl_tensor.data = <void *> npy_ary.data
    dl_tensor.ndim = nd
    dl_tensor.byte_offset = <uint64_t>byte_offset
    dl_tensor.shape = &shape_strides_ptr[0] if nd > 0 else NULL
    dl_tensor.strides = &shape_strides_ptr[nd] if nd > 0 else NULL
    dl_tensor.device.device_type = kDLCPU
    dl_tensor.device.device_id = 0
    dl_tensor.dtype.lanes = <uint16_t>1
    dl_tensor.dtype.bits = <uint8_t>(ary_dt.itemsize * 8)
    if (ary_dtk == "b"):
        dl_tensor.dtype.code = <uint8_t>kDLBool
    elif (ary_dtk == "u"):
        dl_tensor.dtype.code = <uint8_t>kDLUInt
    elif (ary_dtk == "i"):
        dl_tensor.dtype.code = <uint8_t>kDLInt
    elif (ary_dtk == "f" and ary_dt.itemsize <= 8):
        dl_tensor.dtype.code = <uint8_t>kDLFloat
    elif (ary_dtk == "c" and ary_dt.itemsize <= 16):
        dl_tensor.dtype.code = <uint8_t>kDLComplex
    else:
        stdlib.free(shape_strides_ptr)
        stdlib.free(dlmv_tensor)
        raise ValueError("Unrecognized array data type")

    # set flags down here
    if copied:
        dlmv_flags |= DLPACK_FLAG_BITMASK_IS_COPIED
    if not writable_flag:
        dlmv_flags |= DLPACK_FLAG_BITMASK_READ_ONLY
    dlmv_tensor.flags = dlmv_flags

    dlmv_tensor.version.major = DLPACK_MAJOR_VERSION
    dlmv_tensor.version.minor = DLPACK_MINOR_VERSION

    dlmv_tensor.manager_ctx = <void*>npy_ary
    cpython.Py_INCREF(npy_ary)
    dlmv_tensor.deleter = _managed_tensor_versioned_deleter

    return cpython.PyCapsule_New(
        dlmv_tensor, "dltensor_versioned", _pycapsule_versioned_deleter
    )


cdef class _DLManagedTensorOwner:
    """
    Helper class managing the lifetime of the DLManagedTensor struct
    transferred from a 'dlpack' capsule.
    """
    cdef DLManagedTensor * dlm_tensor

    def __cinit__(self):
        self.dlm_tensor = NULL

    def __dealloc__(self):
        if self.dlm_tensor:
            self.dlm_tensor.deleter(self.dlm_tensor)
            self.dlm_tensor = NULL

    @staticmethod
    cdef _DLManagedTensorOwner _create(DLManagedTensor *dlm_tensor_src):
        cdef _DLManagedTensorOwner res
        res = _DLManagedTensorOwner.__new__(_DLManagedTensorOwner)
        res.dlm_tensor = dlm_tensor_src
        return res


cdef class _DLManagedTensorVersionedOwner:
    """
    Helper class managing the lifetime of the DLManagedTensorVersioned
    struct transferred from a 'dlpack_versioned' capsule.
    """
    cdef DLManagedTensorVersioned * dlmv_tensor

    def __cinit__(self):
        self.dlmv_tensor = NULL

    def __dealloc__(self):
        if self.dlmv_tensor:
            self.dlmv_tensor.deleter(self.dlmv_tensor)
            self.dlmv_tensor = NULL

    @staticmethod
    cdef _DLManagedTensorVersionedOwner _create(
        DLManagedTensorVersioned *dlmv_tensor_src
    ):
        cdef _DLManagedTensorVersionedOwner res
        res = _DLManagedTensorVersionedOwner.__new__(
            _DLManagedTensorVersionedOwner
        )
        res.dlmv_tensor = dlmv_tensor_src
        return res


cdef dict _numpy_array_interface_from_dl_tensor(DLTensor *dlt, bint ro_flag):
    """Constructs a NumPy `__array_interface__` dictionary from a DLTensor."""
    cdef int itemsize = 0

    if dlt.dtype.lanes != 1:
        raise BufferError(
            "Can not import DLPack tensor with lanes != 1"
        )
    itemsize = dlt.dtype.bits // 8
    shape = list()
    if (dlt.strides is NULL):
        strides = None
        for dim in range(dlt.ndim):
            shape.append(dlt.shape[dim])
    else:
        strides = list()
        for dim in range(dlt.ndim):
            shape.append(dlt.shape[dim])
            # convert to byte-strides
            strides.append(dlt.strides[dim] * itemsize)
        strides = tuple(strides)
    shape = tuple(shape)
    if (dlt.dtype.code == kDLUInt):
        ary_dt = "u" + str(itemsize)
    elif (dlt.dtype.code == kDLInt):
        ary_dt = "i" + str(itemsize)
    elif (dlt.dtype.code == kDLFloat):
        ary_dt = "f" + str(itemsize)
    elif (dlt.dtype.code == kDLComplex):
        ary_dt = "c" + str(itemsize)
    elif (dlt.dtype.code == kDLBool):
        ary_dt = "b" + str(itemsize)
    else:
        raise BufferError(
            "Can not import DLPack tensor with type code {}.".format(
                <object>dlt.dtype.code
            )
        )
    typestr = "|" + ary_dt
    return dict(
        version=3,
        shape=shape,
        strides=strides,
        data=(<size_t> dlt.data, True if ro_flag else False),
        offset=dlt.byte_offset,
        typestr=typestr,
    )


class _numpy_array_interface_wrapper:
    """
    Class that wraps a Python capsule and dictionary for consumption by NumPy.

    Implementation taken from
    https://github.com/dmlc/dlpack/blob/main/apps/numpy_dlpack/dlpack/to_numpy.py

    Args:
        array_interface:
            A dictionary describing the underlying memory. Formatted
            to match `numpy.ndarray.__array_interface__`.

        pycapsule:
            A Python capsule wrapping the dlpack tensor that will be
            converted to numpy.
    """

    def __init__(self, array_interface, memory_owner) -> None:
        self.__array_interface__ = array_interface
        self._memory_owner = memory_owner


cdef bint _is_kdlcpu_device(DLDevice *dev):
    "Check if DLTensor.DLDevice denotes (kDLCPU, 0)"
    return (dev[0].device_type == kDLCPU) and (dev[0].device_id == 0)


cpdef object from_dlpack_capsule(object py_caps):
    """
    from_dlpack_capsule(py_caps)

    Reconstructs instance of :class:`dpctl.tensor.usm_ndarray` from
    named Python capsule object referencing instance of ``DLManagedTensor``
    without copy. The instance forms a view in the memory of the tensor.

    Args:
        caps:
            Python capsule with name ``"dltensor"`` expected to reference
            an instance of ``DLManagedTensor`` struct.
    Returns:
        Instance of :class:`dpctl.tensor.usm_ndarray` with a view into
        memory of the tensor. Capsule is renamed to ``"used_dltensor"``
        upon success.
    Raises:
        TypeError:
            if argument is not a ``"dltensor"`` capsule.
        ValueError:
            if argument is ``"used_dltensor"`` capsule
        BufferError:
            if the USM pointer is not bound to the reconstructed
            sycl context, or the DLPack's device_type is not supported
            by :mod:`dpctl`.
    """
    cdef DLManagedTensorVersioned *dlmv_tensor = NULL
    cdef DLManagedTensor *dlm_tensor = NULL
    cdef DLTensor *dl_tensor = NULL
    cdef int versioned = 0
    cdef int readonly = 0
    cdef bytes usm_type
    cdef size_t sz = 1
    cdef size_t alloc_sz = 1
    cdef int i
    cdef int device_id = -1
    cdef int element_bytesize = 0
    cdef Py_ssize_t offset_min = 0
    cdef Py_ssize_t offset_max = 0
    cdef char *mem_ptr = NULL
    cdef Py_ssize_t mem_ptr_delta = 0
    cdef Py_ssize_t element_offset = 0
    cdef int64_t stride_i = -1
    cdef int64_t shape_i = -1

    if cpython.PyCapsule_IsValid(py_caps, "dltensor"):
        dlm_tensor = <DLManagedTensor*>cpython.PyCapsule_GetPointer(
                py_caps, "dltensor")
        dl_tensor = &dlm_tensor.dl_tensor
    elif cpython.PyCapsule_IsValid(py_caps, "dltensor_versioned"):
        dlmv_tensor = <DLManagedTensorVersioned*>cpython.PyCapsule_GetPointer(
                py_caps, "dltensor_versioned")
        if dlmv_tensor.version.major > DLPACK_MAJOR_VERSION:
            raise BufferError(
                "Can not import DLPack tensor with major version "
                f"greater than {DLPACK_MAJOR_VERSION}"
            )
        versioned = 1
        readonly = (dlmv_tensor.flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0
        dl_tensor = &dlmv_tensor.dl_tensor
    elif (
        cpython.PyCapsule_IsValid(py_caps, "used_dltensor")
        or cpython.PyCapsule_IsValid(py_caps, "used_dltensor_versioned")
    ):
        raise ValueError(
            "A DLPack tensor object can not be consumed multiple times"
        )
    else:
        raise TypeError(
            "`from_dlpack_capsule` expects a Python 'dltensor' capsule"
        )

    # Verify that we can work with this device
    if dl_tensor.device.device_type == kDLOneAPI:
        device_id = dl_tensor.device.device_id
        root_device = dpctl.SyclDevice(str(<int>device_id))
        try:
            default_context = root_device.sycl_platform.default_context
        except RuntimeError:
            default_context = get_device_cached_queue(root_device).sycl_context
        if dl_tensor.data is NULL:
            usm_type = b"device"
            q = get_device_cached_queue((default_context, root_device,))
        else:
            usm_type = c_dpmem._Memory.get_pointer_type(
                <DPCTLSyclUSMRef> dl_tensor.data,
                <c_dpctl.SyclContext>default_context)
            if usm_type == b"unknown":
                raise BufferError(
                    "Data pointer in DLPack is not bound to default sycl "
                    f"context of device '{device_id}', translated to "
                    f"{root_device.filter_string}"
                )
            alloc_device = c_dpmem._Memory.get_pointer_device(
                <DPCTLSyclUSMRef> dl_tensor.data,
                <c_dpctl.SyclContext>default_context
            )
            q = get_device_cached_queue((default_context, alloc_device,))
        if dl_tensor.dtype.bits % 8:
            raise BufferError(
                "Can not import DLPack tensor whose element's "
                "bitsize is not a multiple of 8"
            )
        if dl_tensor.dtype.lanes != 1:
            raise BufferError(
                "Can not import DLPack tensor with lanes != 1"
            )
        if dl_tensor.ndim > 0:
            offset_min = 0
            offset_max = 0
            for i in range(dl_tensor.ndim):
                stride_i = dl_tensor.strides[i]
                shape_i = dl_tensor.shape[i]
                if shape_i > 1:
                    shape_i -= 1
                    if stride_i > 0:
                        offset_max = offset_max + stride_i * shape_i
                    else:
                        offset_min = offset_min + stride_i * shape_i
            sz = offset_max - offset_min + 1
        if sz == 0:
            sz = 1

        element_bytesize = (dl_tensor.dtype.bits // 8)
        sz = sz * element_bytesize
        element_offset = dl_tensor.byte_offset // element_bytesize

        # transfer ownership
        if not versioned:
            dlm_holder = _DLManagedTensorOwner._create(dlm_tensor)
            cpython.PyCapsule_SetName(py_caps, "used_dltensor")
        else:
            dlmv_holder = _DLManagedTensorVersionedOwner._create(dlmv_tensor)
            cpython.PyCapsule_SetName(py_caps, "used_dltensor_versioned")

        if dl_tensor.data is NULL:
            usm_mem = dpmem.MemoryUSMDevice(sz, q)
        else:
            mem_ptr_delta = dl_tensor.byte_offset - (
                element_offset * element_bytesize
            )
            mem_ptr = <char *>dl_tensor.data
            alloc_sz = dl_tensor.byte_offset + <uint64_t>(
                (offset_max + 1) * element_bytesize)
            tmp = c_dpmem._Memory.create_from_usm_pointer_size_qref(
                <DPCTLSyclUSMRef> mem_ptr,
                max(alloc_sz, <uint64_t>element_bytesize),
                (<c_dpctl.SyclQueue>q).get_queue_ref(),
                memory_owner=dlmv_holder if versioned else dlm_holder
            )
            if mem_ptr_delta == 0:
                usm_mem = tmp
            else:
                alloc_sz = dl_tensor.byte_offset + <uint64_t>(
                    (offset_max * element_bytesize + mem_ptr_delta))
                usm_mem = c_dpmem._Memory.create_from_usm_pointer_size_qref(
                    <DPCTLSyclUSMRef> (
                        mem_ptr + (element_bytesize - mem_ptr_delta)
                    ),
                    max(alloc_sz, <uint64_t>element_bytesize),
                    (<c_dpctl.SyclQueue>q).get_queue_ref(),
                    memory_owner=tmp
                )

        py_shape = list()
        if (dl_tensor.shape is not NULL):
            for i in range(dl_tensor.ndim):
                py_shape.append(dl_tensor.shape[i])
        if (dl_tensor.strides is not NULL):
            py_strides = list()
            for i in range(dl_tensor.ndim):
                py_strides.append(dl_tensor.strides[i])
        else:
            py_strides = None
        if (dl_tensor.dtype.code == kDLUInt):
            ary_dt = np.dtype("u" + str(element_bytesize))
        elif (dl_tensor.dtype.code == kDLInt):
            ary_dt = np.dtype("i" + str(element_bytesize))
        elif (dl_tensor.dtype.code == kDLFloat):
            ary_dt = np.dtype("f" + str(element_bytesize))
        elif (dl_tensor.dtype.code == kDLComplex):
            ary_dt = np.dtype("c" + str(element_bytesize))
        elif (dl_tensor.dtype.code == kDLBool):
            ary_dt = np.dtype("?")
        else:
            raise BufferError(
                "Can not import DLPack tensor with type code {}.".format(
                    <object>dl_tensor.dtype.code
                )
            )
        res_ary = usm_ndarray(
            py_shape,
            dtype=ary_dt,
            buffer=usm_mem,
            strides=py_strides,
            offset=element_offset
        )
        if readonly:
            res_ary.flags_ = (res_ary.flags_ & ~USM_ARRAY_WRITABLE)
        return res_ary
    elif _is_kdlcpu_device(&dl_tensor.device):
        ary_iface = _numpy_array_interface_from_dl_tensor(dl_tensor, readonly)
        if not versioned:
            dlm_holder = _DLManagedTensorOwner._create(dlm_tensor)
            cpython.PyCapsule_SetName(py_caps, "used_dltensor")
            return np.ctypeslib.as_array(
                _numpy_array_interface_wrapper(ary_iface, dlm_holder)
            )
        else:
            dlmv_holder = _DLManagedTensorVersionedOwner._create(dlmv_tensor)
            cpython.PyCapsule_SetName(py_caps, "used_dltensor_versioned")
            return np.ctypeslib.as_array(
                _numpy_array_interface_wrapper(ary_iface, dlmv_holder)
            )
    else:
        raise BufferError(
            "The DLPack tensor resides on unsupported device."
        )

cdef usm_ndarray _to_usm_ary_from_host_blob(object host_blob, dev : Device):
    q = dev.sycl_queue
    np_ary = np.asarray(host_blob)
    dt = np_ary.dtype
    if dt.char in "dD" and q.sycl_device.has_aspect_fp64 is False:
        Xusm_dtype = (
            "float32" if dt.char == "d" else "complex64"
        )
    else:
        Xusm_dtype = dt
    usm_mem = dpmem.MemoryUSMDevice(np_ary.nbytes, queue=q)
    usm_ary = usm_ndarray(np_ary.shape, dtype=Xusm_dtype, buffer=usm_mem)
    usm_mem.copy_from_host(np.reshape(np_ary.view(dtype="u1"), -1))
    return usm_ary


# only cdef to make it private
cdef object _create_device(object device, object dl_device):
    if isinstance(device, Device):
        return device
    elif isinstance(device, dpctl.SyclDevice):
        return Device.create_device(device)
    else:
        root_device = dpctl.SyclDevice(str(<int>dl_device[1]))
        return Device.create_device(root_device)


def from_dlpack(x, /, *, device=None, copy=None):
    """from_dlpack(x, /, *, device=None, copy=None)

    Constructs :class:`dpctl.tensor.usm_ndarray` or :class:`numpy.ndarray`
    instance from a Python object ``x`` that implements ``__dlpack__`` protocol.

    Args:
        x (object):
            A Python object representing an array that supports
            ``__dlpack__`` protocol.
        device (
            Optional[str, :class:`dpctl.SyclDevice`,
            :class:`dpctl.SyclQueue`,
            :class:`dpctl.tensor.Device`,
            tuple([:class:`enum.IntEnum`, int])])
        ):
            Device where the output array is to be placed. ``device`` keyword
            values can be:

            * ``None``
                The data remains on the same device.
            * oneAPI filter selector string
                SYCL device selected by :ref:`filter selector string
                <filter_selector_string>`.
            * :class:`dpctl.SyclDevice`
                explicit SYCL device that must correspond to
                a non-partitioned SYCL device.
            * :class:`dpctl.SyclQueue`
                implies SYCL device targeted by the SYCL queue.
            * :class:`dpctl.tensor.Device`
                implies SYCL device `device.sycl_queue`. The `Device` object
                is obtained via :attr:`dpctl.tensor.usm_ndarray.device`.
            * ``(device_type, device_id)``
               2-tuple matching the format of the output of the
               ``__dlpack_device__`` method: an integer enumerator representing
               the device type followed by an integer representing the index of
               the device. The only supported :class:`dpctl.tensor.DLDeviceType`
               device types are ``"kDLCPU"`` and ``"kDLOneAPI"``.

            Default: ``None``.

        copy (bool, optional)
            Boolean indicating whether or not to copy the input.

            * If ``copy`` is ``True``, the input will always be
              copied.
            * If ``False``, a ``BufferError`` will be raised if a
              copy is deemed necessary.
            * If ``None``, a copy will be made only if deemed
              necessary, otherwise, the existing memory buffer will
              be reused.

            Default: ``None``.

    Returns:
        Alternative[usm_ndarray, numpy.ndarray]:
            An array containing the data in ``x``. When ``copy`` is
            ``None`` or ``False``, this may be a view into the original
            memory.

            The type of the returned object
            depends on where the data backing up input object ``x`` resides.
            If it resides in a USM allocation on a SYCL device, the
            type :class:`dpctl.tensor.usm_ndarray` is returned, otherwise if it
            resides on ``"kDLCPU"`` device the type is :class:`numpy.ndarray`,
            and otherwise an exception is raised.

            .. note::

                If the return type is :class:`dpctl.tensor.usm_ndarray`, the
                associated SYCL queue is derived from the ``device`` keyword.
                When ``device`` keyword value has type :class:`dpctl.SyclQueue`,
                the explicit queue instance is used, when ``device`` keyword
                value has type :class:`dpctl.tensor.Device`, the
                ``device.sycl_queue`` is used. In all other cases, the cached
                SYCL queue corresponding to the implied SYCL device is used.

    Raises:
        TypeError:
            if ``x`` does not implement ``__dlpack__`` method
        ValueError:
            if data of the input object resides on an unsupported device

    See https://dmlc.github.io/dlpack/latest/ for more details.

    :Example:

        .. code-block:: python

            import dpctl
            import dpctl.tensor as dpt

            class Container:
                "Helper class implementing `__dlpack__` protocol"
                def __init__(self, array):
                    self._array = array

                def __dlpack__(self, stream=None):
                    return self._array.__dlpack__(stream=stream)

                def __dlpack_device__(self):
                    return self._array.__dlpack_device__()

            C = Container(dpt.linspace(0, 100, num=20, dtype="int16"))
            # create usm_ndarray view
            X = dpt.from_dlpack(C)
            # migrate content of the container to device of type kDLCPU
            Y = dpt.from_dlpack(C, device=(dpt.DLDeviceType.kDLCPU, 0))

    """
    dlpack_attr = getattr(x, "__dlpack__", None)
    dlpack_dev_attr = getattr(x, "__dlpack_device__", None)
    if not callable(dlpack_attr) or not callable(dlpack_dev_attr):
        raise TypeError(
            f"The argument of type {type(x)} does not implement "
            "`__dlpack__` and `__dlpack_device__` methods."
        )
    # device is converted to a dlpack_device if necessary
    dl_device = None
    if device:
        if isinstance(device, tuple):
            dl_device = device
            if len(dl_device) != 2:
                raise ValueError(
                    "Argument `device` specified as a tuple must have length 2"
                )
        else:
            if not isinstance(device, dpctl.SyclDevice):
                device = Device.create_device(device)
                d = device.sycl_device
            else:
                d = device
            dl_device = (device_OneAPI, d.get_device_id())
    if dl_device is not None:
        if (dl_device[0] not in [device_OneAPI, device_CPU]):
            raise ValueError(
                f"Argument `device`={device} is not supported."
            )
    got_type_error = False
    got_buffer_error = False
    got_other_error = False
    saved_exception = None
    # First DLPack version supporting dl_device, and copy
    requested_ver = (1, 0)
    cpu_dev = (device_CPU, 0)
    try:
        # setting max_version to minimal version that supports
        # dl_device/copy keywords
        dlpack_capsule = dlpack_attr(
            max_version=requested_ver,
            dl_device=dl_device,
            copy=copy
        )
    except TypeError:
        # exporter does not support max_version keyword
        got_type_error = True
    except (BufferError, NotImplementedError, ValueError) as e:
        # Either dl_device, or copy cannot be satisfied
        got_buffer_error = True
        saved_exception = e
    except Exception as e:
        got_other_error = True
        saved_exception = e
    else:
        # execution did not raise exceptions
        return from_dlpack_capsule(dlpack_capsule)
    finally:
        if got_type_error:
            # max_version/dl_device, copy keywords are not supported
            # by __dlpack__
            x_dldev = dlpack_dev_attr()
            if (dl_device is None) or (dl_device == x_dldev):
                dlpack_capsule = dlpack_attr()
                return from_dlpack_capsule(dlpack_capsule)
            # must copy via host
            if copy is False:
                raise BufferError(
                    "Importing data via DLPack requires copying, but "
                    "copy=False was provided"
                )
            # when max_version/dl_device/copy are not supported
            # we can only support importing to OneAPI devices
            # from host, or from another oneAPI device
            is_supported_x_dldev = (
                x_dldev == cpu_dev or
                (x_dldev[0] == device_OneAPI)
            )
            is_supported_dl_device = (
                dl_device == cpu_dev or
                dl_device[0] == device_OneAPI
            )
            if is_supported_x_dldev and is_supported_dl_device:
                dlpack_capsule = dlpack_attr()
                blob = from_dlpack_capsule(dlpack_capsule)
            else:
                raise BufferError(
                    f"Can not import to requested device {dl_device}"
                )
            dev = _create_device(device, dl_device)
            if x_dldev == cpu_dev and dl_device == cpu_dev:
                # both source and destination are CPU
                return blob
            elif x_dldev == cpu_dev:
                # source is CPU, destination is oneAPI
                return _to_usm_ary_from_host_blob(blob, dev)
            elif dl_device == cpu_dev:
                # source is oneAPI, destination is CPU
                cpu_caps = blob.__dlpack__(
                    max_version=get_build_dlpack_version(),
                    dl_device=cpu_dev
                )
                return from_dlpack_capsule(cpu_caps)
            else:
                import dpctl.tensor as dpt
                return dpt.asarray(blob, device=dev)
        elif got_buffer_error:
            # we are here, because dlpack_attr could not deal with requested
            # dl_device, or copying was required
            if copy is False:
                raise BufferError(
                    "Importing data via DLPack requires copying, but "
                    "copy=False was provided"
                )
            if dl_device is None:
                raise saved_exception
            # must copy via host
            if dl_device[0] != device_OneAPI:
                raise BufferError(
                    f"Can not import to requested device {dl_device}"
                )
            x_dldev = dlpack_dev_attr()
            if x_dldev == cpu_dev:
                dlpack_capsule = dlpack_attr()
                host_blob = from_dlpack_capsule(dlpack_capsule)
            else:
                dlpack_capsule = dlpack_attr(
                    max_version=requested_ver,
                    dl_device=cpu_dev,
                    copy=copy
                )
                host_blob = from_dlpack_capsule(dlpack_capsule)
            dev = _create_device(device, dl_device)
            return _to_usm_ary_from_host_blob(host_blob, dev)
        elif got_other_error:
            raise saved_exception
