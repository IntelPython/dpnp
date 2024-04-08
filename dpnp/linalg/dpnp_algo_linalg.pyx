# cython: language_level=3
# cython: linetrace=True
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
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

"""Module Backend

This module contains interface functions between C backend layer
and the rest of the library

"""

import numpy

from dpnp.dpnp_algo cimport *

import dpnp
import dpnp.dpnp_utils as utils_py

cimport numpy

cimport dpnp.dpnp_utils as utils

__all__ = [
    "dpnp_cond",
]


cpdef object dpnp_cond(object input, object p):
    if p in ('f', 'fro'):
        # TODO: change order='K' when support is implemented
        input = dpnp.ravel(input, order='C')
        sqnorm = dpnp.dot(input, input)
        res = dpnp.sqrt(sqnorm)
        ret = dpnp.array([res])
    elif p == dpnp.inf:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=1)
        ret = dpnp.max(dpnp_sum_val)
    elif p == -dpnp.inf:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=1)
        ret = dpnp.min(dpnp_sum_val)
    elif p == 1:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=0)
        ret = dpnp.max(dpnp_sum_val)
    elif p == -1:
        dpnp_sum_val = dpnp.sum(dpnp.abs(input), axis=0)
        ret = dpnp.min(dpnp_sum_val)
    else:
        ret = dpnp.array([input.item(0)])
    return ret
