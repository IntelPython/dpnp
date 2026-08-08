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

"""Pairwise-distance workload.

The dpnp implementation and the data initialization are copied verbatim from
dpBench (https://github.com/IntelPython/dpbench), and the metadata below
mirrors ``dpbench/configs/bench_info/pairwise_distance.toml``.
"""

import dpnp as np

# --- dpBench benchmark metadata (see pairwise_distance.toml) ----------------

NAME = "pairwise_distance"
PRECISION = "double"

INPUT_ARGS = ["X1", "X2", "D"]
ARRAY_ARGS = ["X1", "X2", "D"]
OUTPUT_ARGS = ["D"]

INIT_INPUT_ARGS = ["npoints", "dims", "seed", "types_dict"]
INIT_OUTPUT_ARGS = ["X1", "X2", "D"]

PRESETS = {
    "S": {"npoints": 1024, "dims": 3, "seed": 7777777},
    "M16Gb": {"npoints": 21846, "dims": 3, "seed": 7777777},
    "M": {"npoints": 32768, "dims": 3, "seed": 7777777},
    "L": {"npoints": 44032, "dims": 3, "seed": 7777777},
}
ASV_PRESETS = ["S"]


def initialize(npoints, dims, seed, types_dict):
    import numpy as np
    import numpy.random as default_rng

    dtype = types_dict["float"]

    default_rng.seed(seed)

    return (
        default_rng.random((npoints, dims)).astype(dtype),
        default_rng.random((npoints, dims)).astype(dtype),
        np.empty((npoints, npoints), dtype),
    )


def pairwise_distance(X1, X2, D):
    x1 = np.sum(np.square(X1), axis=1, dtype=X1.dtype)
    x2 = np.sum(np.square(X2), axis=1, dtype=X2.dtype)
    np.dot(X1, X2.T, D)
    D *= -2
    x3 = x1.reshape(x1.size, 1)
    np.add(D, x3, D)
    np.add(D, x2, D)
    np.sqrt(D, D)

    np.synchronize_array_data(D)
