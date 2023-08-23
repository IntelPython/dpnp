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

"""Example BS.

This example shows simple usage of the DPNP
to calculate black scholes algorithm

"""

import dpnp as np

SIZE = 2**8
DTYPE = np.default_float_type()

SEED = 7777777
PL, PH = 10.0, 50.0
SL, SH = 10.0, 50.0
TL, TH = 1.0, 2.0
RISK_FREE = 0.1
VOLATILITY = 0.2


def black_scholes(price, strike, t, rate, vol, call, put):
    mr = -rate
    sig_sig_two = vol * vol * 2

    P = price
    S = strike
    T = t

    a = np.log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = 1.0 / np.sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * np.erf(w1)
    d2 = 0.5 + 0.5 * np.erf(w2)

    Se = np.exp(b) * S

    r = P * d1 - Se * d2
    call[:] = r  # temporary `r` is necessary for faster `put` computation
    put[:] = r - P + Se


np.random.seed(SEED)
price = np.random.uniform(PL, PH, SIZE)
strike = np.random.uniform(SL, SH, SIZE)
t = np.random.uniform(TL, TH, SIZE)

call = np.zeros(SIZE, dtype=DTYPE)
put = -np.ones(SIZE, dtype=DTYPE)

black_scholes(price, strike, t, RISK_FREE, VOLATILITY, call, put)
print(call[:10])
print(put[:10])
