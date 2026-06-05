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

import use_dpnp_array as uda

import dpnp.tensor as dpt


def test_ndim():
    arr = dpt.usm_ndarray((3, 4), dtype="i4")
    assert uda.get_ndim(arr) == 2
    arr = dpt.usm_ndarray(10, dtype="f4")
    assert uda.get_ndim(arr) == 1


def test_shape():
    arr = dpt.usm_ndarray((5, 7, 3), dtype="f8")
    assert uda.get_shape(arr) == [5, 7, 3]


def test_size():
    arr = dpt.usm_ndarray((4, 5), dtype="i4")
    assert uda.get_size(arr) == 20


def test_elemsize():
    arr_f4 = dpt.usm_ndarray(10, dtype="f4")
    assert uda.get_elemsize(arr_f4) == 4

    arr_f8 = dpt.usm_ndarray(10, dtype="f8")
    assert uda.get_elemsize(arr_f8) == 8


def test_c_contiguous():
    arr = dpt.usm_ndarray((3, 4), dtype="f4", order="C")
    assert uda.is_c_contiguous(arr) is True


def test_f_contiguous():
    arr = dpt.usm_ndarray((3, 4), dtype="f4", order="F")
    assert uda.is_f_contiguous(arr) is True


def test_writable():
    arr = dpt.usm_ndarray(10, dtype="i4")
    assert uda.is_writable(arr) is True


def test_typenum():
    arr_f4 = dpt.usm_ndarray(5, dtype="f4")
    arr_f8 = dpt.usm_ndarray(5, dtype="f8")
    assert uda.get_typenum(arr_f4) != uda.get_typenum(arr_f8)
