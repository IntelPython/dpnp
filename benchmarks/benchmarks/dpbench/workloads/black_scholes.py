# *****************************************************************************
# Copyright (c) 2020, Intel Corporation
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

"""Black-Scholes formula workload.

The dpnp implementation and the data initialization are copied verbatim from
dpBench (https://github.com/IntelPython/dpbench), and the metadata below
mirrors ``dpbench/configs/bench_info/black_scholes.toml``.
"""

import dpnp as np

# --- dpBench benchmark metadata (see black_scholes.toml) --------------------

NAME = "black_scholes"
PRECISION = "double"

# Arguments passed to the kernel, in order.
INPUT_ARGS = [
    "nopt",
    "price",
    "strike",
    "t",
    "rate",
    "volatility",
    "call",
    "put",
]
# Arguments that are arrays and therefore copied to the device.
ARRAY_ARGS = ["price", "strike", "t", "call", "put"]
# Arguments that the kernel writes into.
OUTPUT_ARGS = ["call", "put"]

# Arguments passed to ``initialize`` and the values it returns, in order.
INIT_INPUT_ARGS = ["nopt", "seed", "types_dict"]
INIT_OUTPUT_ARGS = [
    "price",
    "strike",
    "t",
    "rate",
    "volatility",
    "call",
    "put",
]

# Data-size presets, copied verbatim from dpBench.
PRESETS = {
    "S": {"nopt": 524288, "seed": 777777},
    "M16Gb": {"nopt": 67108864, "seed": 777777},
    "M": {"nopt": 134217728, "seed": 777777},
    "L": {"nopt": 268435456, "seed": 777777},
}
# Presets actually exercised by ASV. Larger presets require several GiB of
# device memory; add them here to benchmark bigger problem sizes.
ASV_PRESETS = ["S"]


def initialize(nopt, seed, types_dict):
    import numpy as np
    import numpy.random as default_rng

    dtype: np.dtype = types_dict["float"]
    S0L = dtype.type(10.0)
    S0H = dtype.type(50.0)
    XL = dtype.type(10.0)
    XH = dtype.type(50.0)
    TL = dtype.type(1.0)
    TH = dtype.type(2.0)
    RISK_FREE = dtype.type(0.1)
    VOLATILITY = dtype.type(0.2)

    default_rng.seed(seed)
    price = default_rng.uniform(S0L, S0H, nopt).astype(dtype)
    strike = default_rng.uniform(XL, XH, nopt).astype(dtype)
    t = default_rng.uniform(TL, TH, nopt).astype(dtype)
    rate = RISK_FREE
    volatility = VOLATILITY
    call = np.zeros(nopt, dtype=dtype)
    put = -np.ones(nopt, dtype=dtype)

    return (price, strike, t, rate, volatility, call, put)


def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    mr = -rate
    sig_sig_two = volatility * volatility * 2

    P = price
    S = strike
    T = t

    a = np.log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = np.true_divide(1.0, np.sqrt(z))

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * np.scipy.special.erf(w1)
    d2 = 0.5 + 0.5 * np.scipy.special.erf(w2)

    Se = np.exp(b) * S

    call[:] = P * d1 - Se * d2
    put[:] = call - P + Se

    np.synchronize_array_data(put)
