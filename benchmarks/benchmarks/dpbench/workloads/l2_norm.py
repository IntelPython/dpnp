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

"""L2-norm workload.

The dpnp implementation and the data initialization are copied verbatim from
dpBench (https://github.com/IntelPython/dpbench), and the metadata below
mirrors ``dpbench/configs/bench_info/l2_norm.toml``.
"""

import dpnp as np

# --- dpBench benchmark metadata (see l2_norm.toml) --------------------------

NAME = "l2_norm"
PRECISION = "double"

INPUT_ARGS = ["a", "d"]
ARRAY_ARGS = ["a", "d"]
OUTPUT_ARGS = ["d"]

INIT_INPUT_ARGS = ["npoints", "dims", "seed", "types_dict"]
INIT_OUTPUT_ARGS = ["a", "d"]

PRESETS = {
    "S": {"npoints": 32768, "dims": 3, "seed": 777777},
    "M16Gb": {"npoints": 134217728, "dims": 3, "seed": 777777},
    "M": {"npoints": 268435456, "dims": 3, "seed": 777777},
    "L": {"npoints": 536870912, "dims": 3, "seed": 777777},
}
ASV_PRESETS = ["S"]


def initialize(npoints, dims, seed, types_dict):
    import numpy as np
    import numpy.random as default_rng

    dtype = types_dict["float"]

    default_rng.seed(seed)

    return (
        default_rng.random((npoints, dims)).astype(dtype),
        np.zeros(npoints).astype(dtype),
    )


def l2_norm(a, d):
    sq = np.square(a)
    sum = sq.sum(axis=1, dtype=sq.dtype)
    d[:] = np.sqrt(sum)

    np.synchronize_array_data(d)
