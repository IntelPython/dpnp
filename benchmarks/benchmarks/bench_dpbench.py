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
``BlackScholes.time_black_scholes`` -- parametrized by the data-size preset and
the floating-point precision. The presets are chosen per device so that only
problem sizes fitting into device memory are benchmarked, and a precision the
device does not support (typically fp64 on an iGPU) is skipped rather than
failing the run.

``setup`` also validates the dpnp results against the workload's NumPy
reference, so a numerically wrong kernel fails the benchmark instead of being
timed. Validation happens outside the timed region and therefore does not
affect the reported numbers, but it is limited to the cheapest preset: the
reference runs on the host, and at the larger presets it costs far more than
the benchmark it guards (measured at ~70 s for ``pairwise_distance`` at
``M16Gb``) while checking numerics that do not depend on the problem size.
"""

import dpctl

from . import benchmark_utils as bench_utils
from .dpbench import _dpbench_runner as runner
from .dpbench.workloads import WORKLOADS

# Default-device queue, used to query device capabilities (fp64 support, memory
# size) so the parameter matrix can be tailored to the device. This is the
# device dpnp allocates on by default.
DEVICE_QUEUE = dpctl.SyclQueue()
DEVICE = DEVICE_QUEUE.sycl_device


def _camel_case(name):
    """``black_scholes`` -> ``BlackScholes``, ``l2_norm`` -> ``L2Norm``."""
    return "".join(part.capitalize() for part in name.split("_"))


def _make_benchmark_class(workload):
    """Build an ASV benchmark class for a single dpBench workload."""

    class WorkloadBenchmark:
        # The per-benchmark timeout is governed by ``default_benchmark_timeout``
        # in ``asv.conf.json``; larger presets on a busy device can take a
        # while.

        params = [
            runner.select_presets(workload, DEVICE),
            list(runner.PRECISIONS),
        ]
        param_names = ["preset", "precision"]

        # Preset the results are validated against; see the module docstring.
        _validated_preset = runner.presets_by_size(workload)[0]

        def setup(self, preset, precision):
            # Skip precisions the device does not support (e.g. fp64 on many
            # iGPUs), mirroring the dpctl ASV benchmarks.
            bench_utils.skip_unsupported_dtype(
                DEVICE_QUEUE, runner.float_dtype(precision)
            )

            self._runner = runner.WorkloadRunner(workload, preset, precision)
            self._runner.setup()
            if preset == self._validated_preset:
                self._runner.validate()

        def time_workload(self, preset, precision):
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
