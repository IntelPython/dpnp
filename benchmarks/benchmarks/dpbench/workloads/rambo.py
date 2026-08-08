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

"""Rambo workload.

The dpnp implementation and the data initialization are copied verbatim from
dpBench (https://github.com/IntelPython/dpbench), and the metadata below
mirrors ``dpbench/configs/bench_info/rambo.toml``.
"""

import dpnp as np

# --- dpBench benchmark metadata (see rambo.toml) ----------------------------

NAME = "rambo"
PRECISION = "double"

INPUT_ARGS = ["nevts", "nout", "C1", "F1", "Q1", "output"]
ARRAY_ARGS = ["C1", "F1", "Q1", "output"]
OUTPUT_ARGS = ["output"]

INIT_INPUT_ARGS = ["nevts", "nout", "types_dict"]
INIT_OUTPUT_ARGS = ["C1", "F1", "Q1", "output"]

PRESETS = {
    "S": {"nevts": 32768, "nout": 4},
    "M16Gb": {"nevts": 16777216, "nout": 4},
    "M": {"nevts": 8388608, "nout": 4},
    "L": {"nevts": 16777216, "nout": 4},
}
ASV_PRESETS = ["S"]


def initialize(nevts, nout, types_dict):
    import numpy as np

    dtype = types_dict["float"]

    C1 = np.empty((nevts, nout), dtype=dtype)
    F1 = np.empty((nevts, nout), dtype=dtype)
    Q1 = np.empty((nevts, nout), dtype=dtype)

    np.random.seed(777)
    for i in range(nevts):
        for j in range(nout):
            C1[i, j] = np.random.rand()
            F1[i, j] = np.random.rand()
            Q1[i, j] = np.random.rand() * np.random.rand()

    return (C1, F1, Q1, np.empty((nevts, nout, 4), dtype))


def rambo(nevts, nout, C1, F1, Q1, output):
    C = 2.0 * C1 - 1.0
    S = np.sqrt(1 - np.square(C))
    F = 2.0 * np.pi * F1
    Q = -np.log(Q1)

    output[:, :, 0] = Q
    output[:, :, 1] = Q * S * np.sin(F)
    output[:, :, 2] = Q * S * np.cos(F)
    output[:, :, 3] = Q * C

    np.synchronize_array_data(output)
