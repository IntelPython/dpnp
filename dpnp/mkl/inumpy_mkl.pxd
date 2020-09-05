# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2020, Intel Corporation
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

cdef extern from "<stdint.h>":
    ctypedef signed int int64_t


cdef extern from "<complex>" namespace "std":
    cdef cppclass complex[T]:
        complex()


cdef extern from "<CL/sycl/ctl.hpp>" namespace "cl::sycl":
    cdef cppclass vector_class[T]:
        vector_class()


cdef extern from "<CL/sycl.hpp>" namespace "cl::sycl":
    cdef cppclass event:
        event()


cdef extern from "<CL/sycl/queue.hpp>" namespace "cl::sycl":
    cdef cppclass queue:
        queue()


cdef extern from "<mkl_sycl_types.hpp>" namespace "mkl":
    cdef cppclass transpose:
        transpose()

cdef extern from "<mkl_blas_sycl_usm.hpp>" namespace "mkl::blas":
    cdef event gemm(queue & queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    int64_t k, float alpha, const float * a, int64_t lda,
                    const float * b, int64_t ldb, float beta, float * c,
                    int64_t ldc, vector_class[event] & dependencies)

    cdef event gemm(queue & queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    int64_t k, double alpha, const double * a, int64_t lda,
                    const double * b, int64_t ldb, double beta, double * c,
                    int64_t ldc, vector_class[event] & dependencies)

    cdef event gemm(queue & queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    int64_t k, complex[float] alpha, complex[float] * a, int64_t lda,
                    complex[float] * b, int64_t ldb, complex[float] beta, complex[float] * c,
                    int64_t ldc, vector_class[event] & dependencies)

    cdef event gemm(queue & queue, transpose transa, transpose transb, int64_t m, int64_t n,
                    int64_t k, complex[double] alpha, complex[double] * a, int64_t lda,
                    complex[double] * b, int64_t ldb, complex[double] beta, complex[double] * c,
                    int64_t ldc, vector_class[event] & dependencies)
