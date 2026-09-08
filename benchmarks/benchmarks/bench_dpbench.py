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

"""Benchmarks for whole dpnp workloads derived from dpBench.

See ``dpbench/README.md`` for where the workloads come from.
"""

from asv_runner.benchmarks.mark import SkipNotImplemented

from ._utils import default_queue, skip_unsupported_dtype
from .dpbench import _dpbench_runner as runner
from .dpbench.workloads import (
    black_scholes,
    gpairs,
    l2_norm,
    pairwise_distance,
    rambo,
)

# Static, so the matrix is identical on every machine.
_PRESETS = ["S", "M16Gb", "M", "L"]
_PRECISIONS = list(runner.PRECISIONS)


class _Workload:
    """Shared setup for one dpBench-derived workload.

    Defines no ``time_*``, so ASV does not discover it as a benchmark.
    """

    WORKLOAD = None
    params = [_PRESETS, _PRECISIONS]
    param_names = ["preset", "precision"]

    def setup(self, preset, precision):
        queue = default_queue()
        skip_unsupported_dtype(queue, runner.float_dtype(precision))

        if preset not in self.WORKLOAD.PRESETS:
            raise SkipNotImplemented(
                f"{self.WORKLOAD.NAME} has no {preset} preset."
            )

        device = queue.sycl_device
        if not runner.preset_fits(self.WORKLOAD, preset, device, precision):
            raise SkipNotImplemented(
                f"Skipping the {preset} preset as its estimated peak footprint"
                " does not fit this device's memory."
            )

        self._runner = runner.WorkloadRunner(self.WORKLOAD, preset, precision)
        self._runner.setup()

        # Cheapest preset only; the numerics do not depend on its size.
        if preset == runner.presets_by_size(self.WORKLOAD)[0]:
            self._runner.validate()


# ---------------------------------------------------------------------------
# Black-Scholes formula (finance)
# ---------------------------------------------------------------------------


class BlackScholes(_Workload):
    """European option pricing over an array of options."""

    WORKLOAD = black_scholes

    def time_black_scholes(self, preset, precision):
        self._runner.run()


# ---------------------------------------------------------------------------
# L2 norm (distance compute)
# ---------------------------------------------------------------------------


class L2Norm(_Workload):
    """Row-wise Euclidean norm of an (npoints, dims) point cloud."""

    WORKLOAD = l2_norm

    def time_l2_norm(self, preset, precision):
        self._runner.run()


# ---------------------------------------------------------------------------
# Pairwise distance (distance compute)
# ---------------------------------------------------------------------------


class PairwiseDistance(_Workload):
    """Full (npoints, npoints) Euclidean distance matrix via GEMM."""

    WORKLOAD = pairwise_distance

    def time_pairwise_distance(self, preset, precision):
        self._runner.run()


# ---------------------------------------------------------------------------
# Rambo (particle physics)
# ---------------------------------------------------------------------------


class Rambo(_Workload):
    """Phase-space four-momenta generation for collision events."""

    WORKLOAD = rambo

    def time_rambo(self, preset, precision):
        self._runner.run()


# ---------------------------------------------------------------------------
# Galaxy pairs (astrophysics)
# ---------------------------------------------------------------------------


class Gpairs(_Workload):
    """Weighted galaxy-pair counts binned by separation radius."""

    WORKLOAD = gpairs

    def time_gpairs(self, preset, precision):
        self._runner.run()
