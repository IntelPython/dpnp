# cython: language_level=3
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2023, Intel Corporation
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

"""Module Backend (Binary operations)

This module contains interface functions between C backend layer
and the rest of the library

"""

# NO IMPORTs here. All imports must be placed into main "dpnp_algo.pyx" file

__all__ += [
    "dpnp_bitwise_and",
    "dpnp_bitwise_or",
    "dpnp_bitwise_xor",
    "dpnp_invert",
    "dpnp_left_shift",
    "dpnp_right_shift",
]


cpdef utils.dpnp_descriptor dpnp_bitwise_and(utils.dpnp_descriptor x1_obj,
                                             utils.dpnp_descriptor x2_obj,
                                             object dtype=None,
                                             utils.dpnp_descriptor out=None,
                                             object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_BITWISE_AND_EXT, x1_obj, x2_obj, dtype=dtype, out=out, where=where)


cpdef utils.dpnp_descriptor dpnp_bitwise_or(utils.dpnp_descriptor x1_obj,
                                            utils.dpnp_descriptor x2_obj,
                                            object dtype=None,
                                            utils.dpnp_descriptor out=None,
                                            object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_BITWISE_OR_EXT, x1_obj, x2_obj, dtype=dtype, out=out, where=where)


cpdef utils.dpnp_descriptor dpnp_bitwise_xor(utils.dpnp_descriptor x1_obj,
                                             utils.dpnp_descriptor x2_obj,
                                             object dtype=None,
                                             utils.dpnp_descriptor out=None,
                                             object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_BITWISE_XOR_EXT, x1_obj, x2_obj, dtype=dtype, out=out, where=where)


cpdef utils.dpnp_descriptor dpnp_invert(utils.dpnp_descriptor arr, utils.dpnp_descriptor out=None):
    return call_fptr_1in_1out(DPNP_FN_INVERT_EXT, arr, arr.shape, out=out, func_name="invert")


cpdef utils.dpnp_descriptor dpnp_left_shift(utils.dpnp_descriptor x1_obj,
                                            utils.dpnp_descriptor x2_obj,
                                            object dtype=None,
                                            utils.dpnp_descriptor out=None,
                                            object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_LEFT_SHIFT_EXT, x1_obj, x2_obj, dtype=dtype, out=out, where=where)

cpdef utils.dpnp_descriptor dpnp_right_shift(utils.dpnp_descriptor x1_obj,
                                             utils.dpnp_descriptor x2_obj,
                                             object dtype=None,
                                             utils.dpnp_descriptor out=None,
                                             object where=True):
    return call_fptr_2in_1out_strides(DPNP_FN_RIGHT_SHIFT_EXT, x1_obj, x2_obj, dtype=dtype, out=out, where=where)
