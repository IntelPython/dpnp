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

"""ASV benchmarks for dpnp workloads vendored from dpBench.

The workloads (kernels + data initialization) and their data-size presets are
copied from dpBench (https://github.com/IntelPython/dpbench); see
``benchmarks/benchmarks/dpbench``.

Each vendored kernel ends with ``dpnp.synchronize_array_data`` on its output,
so a single call blocks until the device work has finished. The ``time_*``
methods below simply invoke the workload once and let ASV wall-clock-time it
(handling repeats, samples and statistics natively) -- the same end-to-end
quantity dpBench itself measures, and the same plain ``time_*`` style used by
the mkl_fft ASV benchmarks.

A separate benchmark class is generated for each workload -- e.g.
``BlackScholes.time_black_scholes`` -- and parametrized by the data-size preset.
"""

import dpctl

from . import benchmark_utils as bench_utils
from .dpbench import _dpbench_runner as runner
from .dpbench.workloads import WORKLOADS

# Default-device queue, used only to query device capabilities (e.g. fp64
# support) so unsupported-precision workloads can be skipped. This is the
# device dpnp allocates on by default.
DEVICE_QUEUE = dpctl.SyclQueue()


def _camel_case(name):
    """``black_scholes`` -> ``BlackScholes``, ``l2_norm`` -> ``L2Norm``."""
    return "".join(part.capitalize() for part in name.split("_"))


def _make_benchmark_class(workload):
    """Build an ASV benchmark class for a single dpBench workload."""

    class WorkloadBenchmark:
        # The per-benchmark timeout is governed by ``default_benchmark_timeout``
        # in ``asv.conf.json``; larger presets on a busy device can take a
        # while.

        params = list(workload.ASV_PRESETS)
        param_names = ["preset"]

        def setup(self, preset):
            # Skip on devices that do not support the workload's precision
            # (e.g. no fp64), mirroring the dpctl ASV benchmarks.
            float_dtype = runner.build_types_dict(workload.PRECISION)["float"]
            bench_utils.skip_unsupported_dtype(DEVICE_QUEUE, float_dtype)

            self._runner = runner.WorkloadRunner(workload, preset)
            self._runner.setup()

        def time_workload(self, preset):
            self._runner.run()

    # Name things so ASV displays e.g. ``BlackScholes.time_black_scholes``.
    WorkloadBenchmark.__name__ = _camel_case(workload.NAME)
    WorkloadBenchmark.__qualname__ = WorkloadBenchmark.__name__

    time_method = WorkloadBenchmark.time_workload
    time_method.__name__ = f"time_{workload.NAME}"
    setattr(WorkloadBenchmark, time_method.__name__, time_method)
    del WorkloadBenchmark.time_workload

    return WorkloadBenchmark


def _generate_benchmark_classes():
    """Create and register a benchmark class for every vendored workload."""
    for workload in WORKLOADS:
        cls = _make_benchmark_class(workload)
        # Register the class at module scope so ASV can discover it.
        globals()[cls.__name__] = cls


_generate_benchmark_classes()
