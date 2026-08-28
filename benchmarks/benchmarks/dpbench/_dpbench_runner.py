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
  matching how dpBench itself times the workload;
* validation -- the dpnp results are compared elementwise against the
  workload's NumPy reference implementation.
"""

import numpy
from numpy.testing import assert_allclose

import dpnp

# Precision -> dtype mapping, copied from dpBench's
# ``dpbench/configs/precision_dtypes.toml``.
PRECISION_DTYPES = {
    "int": {"single": "i4", "double": "i8"},
    "float": {"single": "f4", "double": "f8"},
}

# Precisions ASV benchmarks each workload at. dpBench's configs request
# ``double`` throughout, but not every device supports fp64 (many iGPUs do
# not), so ``single`` is benchmarked as well and the unsupported one is skipped
# per device -- that way an fp64-less device still produces results instead of
# reporting nothing.
PRECISIONS = ["single", "double"]

# Fraction of the device's global memory a benchmark's estimated peak
# footprint is allowed to occupy. Kept well below 1.0 because the estimates
# below only count the obvious buffers, the device is usually shared with a
# display server, and dpnp's own allocator caches freed blocks.
_MEMORY_BUDGET_FRACTION = 0.25

# atol matters as much as rtol: some outputs pass through zero.
_VALIDATION_TOL = {
    "single": {"rtol": 1e-3, "atol": 1e-4},
    "double": {"rtol": 1e-6, "atol": 1e-9},
}


def build_types_dict(precision):
    """Build the ``types_dict`` passed to a workload's ``initialize``.

    Mirrors ``Benchmark._get_types_dict`` in dpBench.
    """
    return {
        kind: numpy.dtype(precision_strings[precision])
        for kind, precision_strings in PRECISION_DTYPES.items()
    }


def float_dtype(precision):
    """Return the floating-point dtype used at ``precision``."""
    return build_types_dict(precision)["float"]


def preset_fits(workload, preset, device, precision):
    """Whether ``preset``'s estimated peak footprint fits ``device``'s memory."""
    # The cheapest preset always runs, so an undersized device fails loudly.
    if preset == presets_by_size(workload)[0]:
        return True

    itemsize = float_dtype(precision).itemsize
    budget = _MEMORY_BUDGET_FRACTION * device.global_mem_size
    peak = workload.peak_elements(workload.PRESETS[preset])
    return peak * itemsize <= budget


def presets_by_size(workload):
    """Return the workload's preset names ordered cheapest-first.

    dpBench's preset names are not ordered by size (``M16Gb`` is smaller than
    ``M``), so sort explicitly rather than relying on the declaration order.
    """
    return sorted(
        workload.PRESETS,
        key=lambda name: workload.peak_elements(workload.PRESETS[name]),
    )


def initialize_host_data(workload, preset, precision):
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
        data["types_dict"] = build_types_dict(precision)

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


def validate(expected, actual, precision):
    """Check that ``actual`` matches ``expected`` closely enough."""
    tol = _VALIDATION_TOL[precision]
    for name, ref in expected.items():
        assert_allclose(
            actual[name],
            ref,
            equal_nan=True,
            err_msg=f"Validation failed for {name!r}",
            **tol,
        )


class WorkloadRunner:
    """Sets up and runs a single dpBench workload for one preset.

    Each vendored kernel ends with ``dpnp.synchronize_array_data`` on its
    output, so a single :meth:`run` call blocks until the device work has
    completed. ASV wall-clock-times the ``time_*`` method that calls
    :meth:`run`, and thus captures the end-to-end (host dispatch + device)
    execution time of the workload -- the same quantity dpBench measures.
    """

    def __init__(self, workload, preset, precision="double"):
        self.workload = workload
        self.preset = preset
        self.precision = precision

        self.fn = getattr(workload, workload.NAME)
        self.kwargs = None

    def setup(self):
        """Initialize host data, transfer it to the device and warm up."""
        # The host data is deliberately not retained: once the array arguments
        # have been copied to the device it would just pin a second, host-side
        # copy of the whole problem (several GiB at the larger presets).
        host_data = initialize_host_data(
            self.workload, self.preset, self.precision
        )
        inputs = set_input_args(self.workload, host_data)
        self.kwargs = {arg: inputs[arg] for arg in self.workload.INPUT_ARGS}

        # Warmup (equivalent to dpBench's warmup step in ``_exec``).
        self.run()

    def run(self):
        """Execute the kernel once, blocking on device completion."""
        self.fn(**self.kwargs)

    def validate(self):
        """Compare the dpnp results against the NumPy reference.

        Compares every ``OUTPUT_ARGS`` entry against the NumPy reference run
        on freshly initialized host data.
        """
        expected = {
            arg: value
            for arg, value in self._reference_outputs().items()
            if arg in self.workload.OUTPUT_ARGS
        }
        actual = {
            arg: dpnp.asnumpy(self.kwargs[arg])
            for arg in self.workload.OUTPUT_ARGS
        }
        validate(expected, actual, self.precision)

    def _reference_outputs(self):
        """Run the NumPy reference on freshly initialized host data."""
        # setup() does not retain the host data, so re-initialize it here.
        host_data = initialize_host_data(
            self.workload, self.preset, self.precision
        )
        kwargs = {arg: host_data[arg] for arg in self.workload.INPUT_ARGS}
        self.workload.reference(**kwargs)
        return kwargs
