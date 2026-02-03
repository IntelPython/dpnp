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

# distutils: language = c++
# cython: language_level=3

cimport dpctl


cdef public api int USM_ARRAY_C_CONTIGUOUS
cdef public api int USM_ARRAY_F_CONTIGUOUS
cdef public api int USM_ARRAY_WRITABLE

cdef public api int UAR_BOOL
cdef public api int UAR_BYTE
cdef public api int UAR_UBYTE
cdef public api int UAR_SHORT
cdef public api int UAR_USHORT
cdef public api int UAR_INT
cdef public api int UAR_UINT
cdef public api int UAR_LONG
cdef public api int UAR_ULONG
cdef public api int UAR_LONGLONG
cdef public api int UAR_ULONGLONG
cdef public api int UAR_FLOAT
cdef public api int UAR_DOUBLE
cdef public api int UAR_CFLOAT
cdef public api int UAR_CDOUBLE
cdef public api int UAR_TYPE_SENTINEL
cdef public api int UAR_HALF


cdef api class usm_ndarray [object PyUSMArrayObject, type PyUSMArrayType]:
    # data fields
    cdef char* data_
    cdef int nd_
    cdef Py_ssize_t *shape_
    cdef Py_ssize_t *strides_
    cdef int typenum_
    cdef int flags_
    cdef object base_
    cdef object array_namespace_
    # make usm_ndarray weak-referenceable
    cdef object __weakref__

    cdef void _reset(usm_ndarray self)
    cdef void _cleanup(usm_ndarray self)
    cdef Py_ssize_t get_offset(usm_ndarray self) except *

    cdef char* get_data(self)
    cdef int get_ndim(self)
    cdef Py_ssize_t * get_shape(self)
    cdef Py_ssize_t * get_strides(self)
    cdef int get_typenum(self)
    cdef int get_itemsize(self)
    cdef int get_flags(self)
    cdef object get_base(self)
    cdef dpctl.DPCTLSyclQueueRef get_queue_ref(self) except *
    cdef dpctl.SyclQueue get_sycl_queue(self)

    cdef _set_writable_flag(self, int)

    cdef __cythonbufferdefaults__ = {"mode": "strided"}
