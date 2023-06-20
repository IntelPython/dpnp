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

"""
Interface between DPNP Algo (Cython) layer and Numba.

Notes
-----
This module represents DPNP Algo interface to Numba JIT

The code idea gathered from numba-scipy and needs to be syncronized with
https://github.com/numba/numba-scipy/blob/master/numba_scipy/special/
"""

import ctypes

import numba
from numba.extending import get_cython_function_address as nba_addr


name_to_numba_signatures = {"cos": [(numba.types.float64)]}

name_and_types_to_pointer = {
    ("cos", numba.types.float64): ctypes.CFUNCTYPE(
        ctypes.c_double, ctypes.c_double
    )(nba_addr("dpnp.dpnp_algo", "dpnp_cos"))
}


def choose_kernel(name, all_signatures):
    def choice_function(*args):
        for signature in all_signatures:
            if args == signature:
                f = name_and_types_to_pointer[(name, *signature)]
                return lambda *args: f(*args)

    return choice_function


def add_overloads():
    for name, all_signatures in name_to_numba_signatures.items():
        sc_function = getattr(sc, name)
        print(f"sc_function={sc_function}")
        numba.extending.overload(sc_function)(
            choose_kernel(name, all_signatures)
        )
