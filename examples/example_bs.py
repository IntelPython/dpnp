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

"""Example BS.

This example shows simple usage of the DPNP
to calculate black scholes algorithm

"""

try:
    import dpnp as np
except ImportError:
    import sys
    import os

    root_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(root_dir)

    import dpnp as np


SIZE = 2 ** 20
SHAPE = (SIZE,)
DTYPE = np.float64

SEED = 7777777
SL, SH = 10.0, 50.0
KL, KH = 10.0, 50.0
TL, TH = 1.0, 2.0
RISK_FREE = 0.1
VOLATILITY = 0.2


def black_scholes_put(S, K, T, sigmas, r_sigma_sigma_2, nrs, sqrt2, ones, twos):
    d1 = (np.log(S/K) + r_sigma_sigma_2*T) / (sigmas*np.sqrt(T))
    d2 = d1 - sigmas * np.sqrt(T)

    cdf_d1 = (ones + np.erf(d1 / sqrt2)) / twos
    cdf_d2 = (ones + np.erf(d2 / sqrt2)) / twos

    bs_call = S*cdf_d1 - K*np.exp(nrs*T)*cdf_d2

    return K*np.exp(nrs*T) - S + bs_call


np.random.seed(SEED)
S = np.random.uniform(SL, SH, SIZE)
K = np.random.uniform(KL, KH, SIZE)
T = np.random.uniform(TL, TH, SIZE)

r, sigma = RISK_FREE, VOLATILITY

sigmas = np.full(SHAPE, sigma, dtype=DTYPE)
r_sigma_sigma_2 = np.full(SHAPE, r + sigma*sigma/2., dtype=DTYPE)
nrs = np.full(SHAPE, -r, dtype=DTYPE)

sqrt2 = np.full(SHAPE, np.sqrt(2), dtype=DTYPE)
ones = np.full(SHAPE, 1, dtype=DTYPE)
twos = np.full(SHAPE, 2, dtype=DTYPE)

bs_put = black_scholes_put(S, K, T, sigmas, r_sigma_sigma_2, nrs, sqrt2, ones, twos)
print(bs_put[:10])
