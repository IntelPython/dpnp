# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
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

The dpnp implementation, the NumPy reference and the data initialization are
copied from dpBench (https://github.com/IntelPython/dpbench), and the metadata
below mirrors ``dpbench/configs/bench_info/rambo.toml``.
"""

import dpnp as np

# --- dpBench benchmark metadata (see rambo.toml) ----------------------------

NAME = "rambo"
# See the note on ``PRECISION`` in ``black_scholes.py``.
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


def peak_elements(params):
    """Estimated peak number of float elements held on the device.

    The ``(nevts, nout, 4)`` output, the three ``(nevts, nout)`` inputs and the
    ~6 same-shaped temporaries the kernel materializes (``C``, ``S``, ``F``,
    ``Q``, and the ``sin``/``cos`` results).
    """
    return 13 * params["nevts"] * params["nout"]


def initialize(nevts, nout, types_dict):
    import numpy

    dtype = types_dict["float"]

    # dpBench draws these element-by-element in a Python loop; drawing the
    # whole block at once consumes the same RNG stream in the same order (so
    # the data is bit-identical) but is orders of magnitude faster, which
    # matters because ASV re-runs ``setup`` for every benchmark round.
    numpy.random.seed(777)
    draws = numpy.random.rand(nevts, nout, 4)

    C1 = draws[..., 0].astype(dtype)
    F1 = draws[..., 1].astype(dtype)
    Q1 = (draws[..., 2] * draws[..., 3]).astype(dtype)

    return (C1, F1, Q1, numpy.empty((nevts, nout, 4), dtype))


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


def reference(nevts, nout, C1, F1, Q1, output):
    """NumPy reference, copied from dpBench's ``rambo_numpy.py``."""
    import numpy

    C = 2.0 * C1 - 1.0
    S = numpy.sqrt(1 - numpy.square(C))
    F = 2.0 * numpy.pi * F1
    Q = -numpy.log(Q1)

    output[:, :, 0] = Q
    output[:, :, 1] = Q * S * numpy.sin(F)
    output[:, :, 2] = Q * S * numpy.cos(F)
    output[:, :, 3] = Q * C
