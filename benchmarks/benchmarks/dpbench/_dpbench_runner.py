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

"""Minimal re-implementation of dpBench's benchmark execution model for ASV.

dpBench (https://github.com/IntelPython/dpbench) drives its benchmarks through
a fairly heavy runner that spawns a sub-process per framework, resolves TOML
configuration, validates results against a reference and persists timings to a
database. None of that machinery is importable in a lightweight ASV
environment (it pulls in ``numba_dpex``, ``sqlalchemy``, ``alembic`` and more),
so this module re-implements just the parts that matter for benchmarking:

* data initialization -- the host (NumPy) input data is produced exactly the
  way dpBench produces it, using each workload's ``initialize`` function and a
  precision-driven ``types_dict`` (see ``dpbench.infrastructure.benchmark``);
* host-to-device transfer -- array arguments are copied to the device with the
  same ``dpnp.asarray`` logic dpBench's ``DpnpFramework.copy_to_func`` uses;
* execution -- the dpnp implementation is invoked and blocks on device
  completion (each vendored kernel ends with ``dpnp.synchronize_array_data``),
  matching how dpBench itself times the workload.
"""

import numpy

import dpnp

# Precision -> dtype mapping, copied from dpBench's
# ``dpbench/configs/precision_dtypes.toml``.
PRECISION_DTYPES = {
    "int": {"single": "i4", "double": "i8"},
    "float": {"single": "f4", "double": "f8"},
}


def build_types_dict(precision):
    """Build the ``types_dict`` passed to a workload's ``initialize``.

    Mirrors ``Benchmark._get_types_dict`` in dpBench.
    """
    return {
        kind: numpy.dtype(precision_strings[precision])
        for kind, precision_strings in PRECISION_DTYPES.items()
    }


def initialize_host_data(workload, preset):
    """Produce the host (NumPy) input data for ``workload`` at ``preset``.

    Mirrors ``Benchmark.initialize_input_data`` /
    ``_initialize_input_data_from_init`` in dpBench.
    """
    if preset not in workload.PRESETS:
        raise NotImplementedError(
            f"{workload.NAME} doesn't have a {preset} preset."
        )

    # Preset parameters (scalars such as ``nopt``, ``seed``, ``nbins``, ...).
    data = dict(workload.PRESETS[preset])

    # The precision-driven types dictionary, if the workload's ``initialize``
    # consumes one.
    if "types_dict" in workload.INIT_INPUT_ARGS:
        data["types_dict"] = build_types_dict(workload.PRECISION)

    # Call ``initialize`` and store its outputs under the configured names.
    init_kwargs = {arg: data[arg] for arg in workload.INIT_INPUT_ARGS}
    initialized = workload.initialize(**init_kwargs)

    if isinstance(initialized, tuple):
        for name, value in zip(workload.INIT_OUTPUT_ARGS, initialized):
            data[name] = value
    elif len(workload.INIT_OUTPUT_ARGS) == 1:
        data[workload.INIT_OUTPUT_ARGS[0]] = initialized
    else:
        raise ValueError("Unsupported initialize output")

    return data


def _copy_to_device(ref_array):
    """Copy a host array to the (default) device.

    Mirrors ``DpnpFramework.copy_to_func`` in dpBench.
    """
    if ref_array.flags["C_CONTIGUOUS"]:
        order = "C"
    elif ref_array.flags["F_CONTIGUOUS"]:
        order = "F"
    else:
        order = "K"
    return dpnp.asarray(
        ref_array,
        dtype=ref_array.dtype,
        order=order,
    )


def set_input_args(workload, host_data):
    """Build the kernel keyword arguments, copying array args to the device.

    Mirrors ``_set_input_args`` in dpBench.
    """
    inputs = {}
    for arg in workload.INPUT_ARGS:
        if arg in workload.ARRAY_ARGS:
            inputs[arg] = _copy_to_device(host_data[arg])
        else:
            inputs[arg] = host_data[arg]
    return inputs


class WorkloadRunner:
    """Sets up and runs a single dpBench workload for one preset.

    Each vendored kernel ends with ``dpnp.synchronize_array_data`` on its
    output, so a single :meth:`run` call blocks until the device work has
    completed. ASV wall-clock-times the ``time_*`` method that calls
    :meth:`run`, and thus captures the end-to-end (host dispatch + device)
    execution time of the workload -- the same quantity dpBench measures.
    """

    def __init__(self, workload, preset):
        self.workload = workload
        self.preset = preset

        self.fn = getattr(workload, workload.NAME)
        self.kwargs = None

    def setup(self):
        """Initialize host data, transfer it to the device and warm up."""
        host_data = initialize_host_data(self.workload, self.preset)
        inputs = set_input_args(self.workload, host_data)
        self.kwargs = {arg: inputs[arg] for arg in self.workload.INPUT_ARGS}

        # Warmup (equivalent to dpBench's warmup step in ``_exec``).
        self.run()

    def run(self):
        """Execute the kernel once, blocking on device completion."""
        self.fn(**self.kwargs)
