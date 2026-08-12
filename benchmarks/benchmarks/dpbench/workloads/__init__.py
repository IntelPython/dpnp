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

"""dpnp workloads vendored from dpBench.

Each module exposes a uniform interface consumed by ``_dpbench_runner``:

* ``NAME`` -- workload name; also the name of the kernel function;
* ``PRECISION`` -- the precision dpBench's config requests for this workload;
* ``INPUT_ARGS`` / ``ARRAY_ARGS`` / ``OUTPUT_ARGS`` -- kernel argument metadata;
* ``INIT_INPUT_ARGS`` / ``INIT_OUTPUT_ARGS`` -- ``initialize`` argument metadata;
* ``PRESETS`` -- all dpBench data-size presets (S, M16Gb, M, L);
* ``peak_elements(params)`` -- estimated peak device element count for a preset,
  used to pick the presets that fit into the device's memory;
* ``initialize(...)`` -- host data generator;
* ``<NAME>(...)`` -- the dpnp kernel;
* ``reference(...)`` -- the NumPy kernel the dpnp results are validated against.
"""

from . import black_scholes, gpairs, l2_norm, pairwise_distance, rambo

# All vendored workloads, in a stable order.
WORKLOADS = [
    black_scholes,
    l2_norm,
    pairwise_distance,
    rambo,
    gpairs,
]

__all__ = [
    "WORKLOADS",
    "black_scholes",
    "l2_norm",
    "pairwise_distance",
    "rambo",
    "gpairs",
]
