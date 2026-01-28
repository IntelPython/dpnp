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
from dpctl._sycl_device cimport SyclDevice
from numpy cimport ndarray

from ._usmarray cimport usm_ndarray


cdef extern from "dlpack/dlpack.h" nogil:
    int device_CPU "kDLCPU"
    int device_CUDA "kDLCUDA"
    int device_CUDAHost "kDLCUDAHost"
    int device_CUDAManaged "kDLCUDAManaged"
    int device_DLROCM "kDLROCM"
    int device_ROCMHost "kDLROCMHost"
    int device_OpenCL "kDLOpenCL"
    int device_Vulkan "kDLVulkan"
    int device_Metal "kDLMetal"
    int device_VPI "kDLVPI"
    int device_OneAPI "kDLOneAPI"
    int device_WebGPU "kDLWebGPU"
    int device_Hexagon "kDLHexagon"
    int device_MAIA "kDLMAIA"
    int device_Trn "kDLTrn"

cpdef object to_dlpack_capsule(usm_ndarray array) except +
cpdef object to_dlpack_versioned_capsule(
    usm_ndarray array, bint copied
) except +
cpdef object numpy_to_dlpack_versioned_capsule(
    ndarray array, bint copied
) except +
cpdef object from_dlpack_capsule(object dltensor) except +

cdef class DLPackCreationError(Exception):
    """
    A DLPackCreateError exception is raised when constructing
    DLPack capsule from `usm_ndarray` based on a USM allocation
    on a partitioned SYCL device.
    """
    pass
