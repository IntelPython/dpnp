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

cdef int ERROR_MALLOC
cdef int ERROR_INTERNAL
cdef int ERROR_INCORRECT_ORDER
cdef int ERROR_UNEXPECTED_STRIDES

cdef Py_ssize_t shape_to_elem_count(int nd, Py_ssize_t *shape_arr)

cdef int _from_input_shape_strides(
    int nd, object shape, object strides, int itemsize, char order,
    Py_ssize_t **shape_ptr, Py_ssize_t **strides_ptr,
    Py_ssize_t *nelems, Py_ssize_t *min_disp, Py_ssize_t *max_disp,
    int *contig
)

cdef object _make_int_tuple(int nd, const Py_ssize_t *ary)

cdef object _make_reversed_int_tuple(int nd, const Py_ssize_t *ary)

cdef object _c_contig_strides(int nd, Py_ssize_t *shape)

cdef object _f_contig_strides(int nd, Py_ssize_t *shape)

cdef object _swap_last_two(tuple t)
