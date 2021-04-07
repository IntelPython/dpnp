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

"""Module Backend (Sorting part)

This module contains interface functions between C backend layer
and the rest of the library

"""


import numpy
from dpnp.dpnp_utils cimport *


__all__ += [
    "dpnp_argsort",
    "dpnp_sort"
]


cpdef dparray dpnp_argsort(dparray in_array1):
    return call_fptr_1in_1out(DPNP_FN_ARGSORT, in_array1, in_array1.shape)


cpdef dparray dpnp_sort(dparray in_array1):
    return call_fptr_1in_1out(DPNP_FN_SORT, in_array1, in_array1.shape)
