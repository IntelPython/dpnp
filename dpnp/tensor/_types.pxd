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

cdef int UAR_BOOL
cdef int UAR_BYTE
cdef int UAR_UBYTE
cdef int UAR_SHORT
cdef int UAR_USHORT
cdef int UAR_INT
cdef int UAR_UINT
cdef int UAR_LONG
cdef int UAR_ULONG
cdef int UAR_LONGLONG
cdef int UAR_ULONGLONG
cdef int UAR_FLOAT
cdef int UAR_DOUBLE
cdef int UAR_CFLOAT
cdef int UAR_CDOUBLE
cdef int UAR_TYPE_SENTINEL
cdef int UAR_HALF

cdef int type_bytesize(int typenum)

cdef str _make_typestr(int typenum)

cdef int typenum_from_format(str s)

cdef int descr_to_typenum(object dtype)

cdef int dtype_to_typenum(dtype)
