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

"""GPairs (galaxy pair counting) workload.

The dpnp implementation and the data initialization are copied verbatim from
dpBench (https://github.com/IntelPython/dpbench), and the metadata below
mirrors ``dpbench/configs/bench_info/gpairs.toml``.
"""

import numpy

import dpnp as np

# --- dpBench benchmark metadata (see gpairs.toml) ---------------------------

NAME = "gpairs"
PRECISION = "double"

INPUT_ARGS = [
    "nopt",
    "nbins",
    "x1",
    "y1",
    "z1",
    "w1",
    "x2",
    "y2",
    "z2",
    "w2",
    "rbins",
    "results",
]
ARRAY_ARGS = [
    "x1",
    "y1",
    "z1",
    "w1",
    "x2",
    "y2",
    "z2",
    "w2",
    "rbins",
    "results",
]
OUTPUT_ARGS = ["results"]

INIT_INPUT_ARGS = ["nopt", "seed", "nbins", "rmax", "rmin", "types_dict"]
INIT_OUTPUT_ARGS = [
    "x1",
    "y1",
    "z1",
    "w1",
    "x2",
    "y2",
    "z2",
    "w2",
    "rbins",
    "results",
]

PRESETS = {
    "S": {"nopt": 128, "seed": 1234, "nbins": 20, "rmax": 50, "rmin": 0.1},
    "M16Gb": {
        "nopt": 4096,
        "seed": 1234,
        "nbins": 20,
        "rmax": 50,
        "rmin": 0.1,
    },
    "M": {"nopt": 8192, "seed": 1234, "nbins": 20, "rmax": 50, "rmin": 0.1},
    "L": {
        "nopt": 524288,
        "seed": 1234,
        "nbins": 20,
        "rmax": 50,
        "rmin": 0.1,
    },
}
ASV_PRESETS = ["S"]


def _generate_rbins(dtype, nbins, rmax, rmin):
    rbins = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), nbins).astype(
        dtype
    )

    return (rbins**2).astype(dtype)


def initialize(nopt, seed, nbins, rmax, rmin, types_dict):
    import numpy.random as default_rng

    default_rng.seed(seed)
    dtype = types_dict["float"]
    x1 = numpy.random.randn(nopt).astype(dtype)
    y1 = numpy.random.randn(nopt).astype(dtype)
    z1 = numpy.random.randn(nopt).astype(dtype)
    w1 = numpy.random.rand(nopt).astype(dtype)
    w1 = w1 / numpy.sum(w1)

    x2 = numpy.random.randn(nopt).astype(dtype)
    y2 = numpy.random.randn(nopt).astype(dtype)
    z2 = numpy.random.randn(nopt).astype(dtype)
    w2 = numpy.random.rand(nopt).astype(dtype)
    w2 = w2 / numpy.sum(w2)

    rbins = _generate_rbins(dtype=dtype, rmin=rmin, rmax=rmax, nbins=nbins)
    results = numpy.zeros_like(rbins).astype(dtype)
    return (x1, y1, z1, w1, x2, y2, z2, w2, rbins, results)


def _gpairs_impl(x1, y1, z1, w1, x2, y2, z2, w2, rbins):
    dm = (
        np.square(x2 - x1[:, None])
        + np.square(y2 - y1[:, None])
        + np.square(z2 - z1[:, None])
    )
    return np.array(
        [
            np.outer(w1, w2)[dm <= rbins[k]].sum(dtype=np.result_type(w1, w2))
            for k in range(len(rbins))
        ],
        device=x1.device,
    )


def gpairs(nopt, nbins, x1, y1, z1, w1, x2, y2, z2, w2, rbins, results):
    results[:] = _gpairs_impl(x1, y1, z1, w1, x2, y2, z2, w2, rbins)

    np.synchronize_array_data(results)
