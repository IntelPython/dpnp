# *****************************************************************************
# Copyright (c) 2016, Intel Corporation
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

import gc
import os
import sys
import warnings

import dpctl
import dpctl.utils as dpu
import numpy
import pytest

from . import config as dtype_config

if numpy.lib.NumpyVersion(numpy.__version__) >= "2.0.0b1":
    from numpy.exceptions import ComplexWarning
else:
    from numpy import ComplexWarning

import dpnp

from .helper import get_dev_id
from .infra_warning_utils import register_infra_warnings_plugin_if_enabled

skip_mark = pytest.mark.skip(reason="Skipping test.")


def get_excluded_tests(test_exclude_file):
    excluded_tests = []
    if os.path.exists(test_exclude_file):
        with open(test_exclude_file) as skip_names_file:
            excluded_tests = skip_names_file.readlines()
    # Remove whitespace and filter out empty lines
    return [line.strip() for line in excluded_tests if line.strip()]


# Normalize the nodeid to a relative path starting
# from the "tests/" directory
def normalize_test_name(nodeid):
    nodeid = nodeid.replace("\n", "").strip()

    # case if run pytest from dpnp folder
    if nodeid.startswith("tests/"):
        return nodeid

    # case if run pytest --pyargs
    if "/tests/" in nodeid:
        nodeid = nodeid.split("tests/", 1)[-1]
        # Add the "tests/" prefix to ensure the nodeid matches
        # the paths in the skipped tests files.
        normalized_nodeid = "tests/" + nodeid
    # case if run pytest from tests folder
    else:
        normalized_nodeid = "tests/" + nodeid

    return normalized_nodeid


def format_memory_size(size_bytes):
    """Format memory size in human-readable format."""
    if size_bytes is None:
        return "N/A"

    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.2f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} bytes"


def get_device_memory_info(device=None):
    """
    Safely retrieve device memory information.

    Returns a dict describing the device and its memory, or ``None`` if the
    information cannot be retrieved.

    The ``free_memory`` key is only populated when the Level-Zero driver
    reports it, which requires the environment variable ``ZES_ENABLE_SYSMAN``
    to be set to ``1`` before the driver is initialized (see ``run_test.sh``).
    """
    try:
        if device is None:
            device = dpctl.select_default_device()

        info = {
            "device_name": device.name,
            "device_type": str(device.device_type),
            "backend": str(device.backend),
            "global_mem_size": device.global_mem_size,
            "max_mem_alloc_size": device.max_mem_alloc_size,
            "local_mem_size": device.local_mem_size,
            "free_memory": None,
        }

        # `free_memory` is exposed through the Intel-specific device info dict
        # (not as a SyclDevice attribute) and only when ZES_ENABLE_SYSMAN=1.
        try:
            intel_info = dpu.intel_device_info(device)
            info["free_memory"] = intel_info.get("free_memory")
        except Exception:
            pass

        return info
    except Exception as e:
        warnings.warn(f"Failed to get device memory info: {e}")
        return None


def get_queue_event_stats():
    """
    Collect the number of outstanding events tracked by the sequential order
    manager, per SYCL queue.

    A large or steadily growing ``host_task`` count indicates arrays are kept
    alive by not-yet-retired host tasks (see ``keep_args_alive``), i.e. memory
    is held on the queue rather than leaked by the Python garbage collector.
    Returns a list of per-queue dicts (possibly empty).
    """
    stats = []
    try:
        # `_map` is a ContextVar holding a {SyclQueue: _SequentialOrderManager}.
        queue_map = dpu.SequentialOrderManager._map.get()
        for queue, order_manager in queue_map.items():
            stats.append(
                {
                    "device": queue.sycl_device.name,
                    "submitted_events": order_manager.num_submitted_events,
                    "host_task_events": order_manager.num_host_task_events,
                }
            )
    except Exception as e:
        warnings.warn(f"Failed to get queue event stats: {e}")
    return stats


def format_diagnostics(prefix, nodeid):
    """
    Build a multi-line diagnostics string covering both OOM hypotheses:
      * events/host tasks held on the SYCL queue, and
      * Python GC state (uncollected objects).
    Intended for logging around a test to investigate
    ``UR_RESULT_ERROR_OUT_OF_RESOURCES`` failures.
    """
    lines = [f"[{prefix}] {nodeid}"]

    mem_info = get_device_memory_info()
    if mem_info:
        lines.append(
            "  Device memory: "
            f"free={format_memory_size(mem_info['free_memory'])}, "
            f"global={format_memory_size(mem_info['global_mem_size'])}, "
            f"max_alloc={format_memory_size(mem_info['max_mem_alloc_size'])}"
        )

    queue_stats = get_queue_event_stats()
    if queue_stats:
        for st in queue_stats:
            lines.append(
                "  Queue events: "
                f"submitted={st['submitted_events']}, "
                f"host_task={st['host_task_events']} "
                f"(device={st['device']})"
            )
    else:
        lines.append("  Queue events: none tracked")

    # gc.get_count() -> (gen0, gen1, gen2) allocations since last collection.
    gc_count = gc.get_count()
    lines.append(
        f"  GC: tracked_objects={len(gc.get_objects())}, "
        f"gen_counts={gc_count}"
    )

    return "\n".join(lines)


def pytest_configure(config):
    # By default, tests marked as slow will be deselected.
    # To run all tests, use -m "slow or not slow".
    # To run only slow tests, use -m "slow".
    # Equivalent to addopts = -m "not slow"
    if not config.getoption("markexpr"):
        config.option.markexpr = "not slow"
    # Equivalent to addopts = --tb=short
    if not config.getoption("tbstyle"):
        config.option.tbstyle = "short"
    # Equivalent to addopts = --strict-markers
    if not config.getoption("strict_markers"):
        config.option.strict_markers = True

    # Equivalent to norecursedirs = tests_perf
    config.addinivalue_line("norecursedirs", "tests_perf")

    # Equivalent to norecursedirs = tests/tensor (conditional)
    if dtype_config.skip_tensor_tests:
        config.addinivalue_line("norecursedirs", "tests/tensor")

    # Register pytest markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "multi_gpu: marks tests that require a specified number of GPUs",
    )

    # NumPy arccosh
    # Undefined behavior depends on the backend:
    # NumPy with OpenBLAS for np.array[1.0] does not raise a warning
    # while numpy with OneMKL raises RuntimeWarning
    config.addinivalue_line(
        "filterwarnings",
        "ignore:invalid value encountered in arccosh:RuntimeWarning",
    )

    register_infra_warnings_plugin_if_enabled(config)


def pytest_collection_modifyitems(config, items):
    test_path = os.path.split(__file__)[0]
    excluded_tests = []
    # global skip file
    test_exclude_file = os.path.join(test_path, "skipped_tests.tbl")

    # global skip file, where gpu device is not supported
    test_exclude_file_gpu = os.path.join(test_path, "skipped_tests_gpu.tbl")

    # global skip file, where gpu device with no fp64 support
    test_exclude_file_gpu_no_fp64 = os.path.join(
        test_path, "skipped_tests_gpu_no_fp64.tbl"
    )

    # global skip file for cuda backend
    test_exclude_file_cuda = os.path.join(test_path, "skipped_tests_cuda.tbl")

    dev = dpctl.select_default_device()
    is_cpu = dev.is_cpu
    is_gpu = dev.is_gpu
    support_fp64 = dev.has_aspect_fp64
    is_cuda = dpnp.is_cuda_backend(dev)

    print("")
    print(
        f"DPNP Test scope includes all integer dtypes: {bool(dtype_config.all_int_types)}"
    )
    print(f"DPNP current device ID: 0x{get_dev_id(dev):04X}")
    print(f"DPNP current device is CPU: {is_cpu}")
    print(f"DPNP current device is GPU: {is_gpu}")
    print(f"DPNP current device supports fp64: {support_fp64}")
    print(f"DPNP current device is GPU with cuda backend: {is_cuda}")
    print(f"DPNP version: {dpnp.__version__}, location: {dpnp}")
    print(f"NumPy version: {numpy.__version__}, location: {numpy}")
    print(f"Python version: {sys.version}")

    # Log device memory information at start up. `free_memory` requires
    # ZES_ENABLE_SYSMAN=1 (set by run_test.sh) and a driver that reports it.
    mem_info = get_device_memory_info(dev)
    if mem_info:
        print("")
        print("Device Memory Information:")
        print(f"  Device: {mem_info['device_name']}")
        print(f"  Backend: {mem_info['backend']}")
        print(
            f"  Global Memory Size: {format_memory_size(mem_info['global_mem_size'])}"
        )
        print(
            f"  Max Allocation Size: {format_memory_size(mem_info['max_mem_alloc_size'])}"
        )
        print(
            f"  Local Memory Size: {format_memory_size(mem_info['local_mem_size'])}"
        )
        print(f"  Free Memory: {format_memory_size(mem_info['free_memory'])}")
        if mem_info["free_memory"] is None:
            print(
                "    (free memory not reported; set ZES_ENABLE_SYSMAN=1 "
                "before launching to enable it)"
            )

    print("")
    if is_gpu or os.getenv("DPNP_QUEUE_GPU") == "1":
        excluded_tests.extend(get_excluded_tests(test_exclude_file_gpu))
        if not support_fp64:
            excluded_tests.extend(
                get_excluded_tests(test_exclude_file_gpu_no_fp64)
            )
        if is_cuda:
            excluded_tests.extend(get_excluded_tests(test_exclude_file_cuda))
    else:
        excluded_tests.extend(get_excluded_tests(test_exclude_file))

    for item in items:
        test_name = normalize_test_name(item.nodeid)

        for item_tbl in excluded_tests:
            # exact match of the test name with items from excluded_list
            if test_name == item_tbl:
                item.add_marker(skip_mark)

    # Handle the exclusion of tests marked as "slow"
    selected_marker = config.getoption("markexpr")
    if "not slow" in selected_marker:
        skip_slow = pytest.mark.skip(reason="Skipping slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture
def allow_fall_back_on_numpy(monkeypatch):
    monkeypatch.setattr(
        dpnp.config, "__DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK__", 0
    )


@pytest.fixture
def suppress_complex_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComplexWarning)
        yield


@pytest.fixture
def suppress_divide_numpy_warnings():
    # divide: treatment for division by zero (infinite result obtained from finite numbers)
    old_settings = numpy.seterr(divide="ignore")
    yield
    numpy.seterr(**old_settings)  # reset to default


@pytest.fixture
def suppress_invalid_numpy_warnings():
    # invalid: treatment for invalid floating-point operation
    # (result is not an expressible number, typically indicates that a NaN was produced)
    old_settings = numpy.seterr(invalid="ignore")
    yield
    numpy.seterr(**old_settings)  # reset to default


@pytest.fixture
def suppress_dof_numpy_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")
        yield


@pytest.fixture
def suppress_mean_empty_slice_numpy_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice")
        yield


@pytest.fixture
def suppress_overflow_encountered_in_cast_numpy_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"overflow encountered in cast")
        yield


@pytest.fixture
def suppress_divide_invalid_numpy_warnings(
    suppress_divide_numpy_warnings, suppress_invalid_numpy_warnings
):
    yield


# Substrings that identify an out-of-resources / out-of-memory device failure,
# e.g. `RuntimeError: ... UR_RESULT_ERROR_OUT_OF_RESOURCES` on low-memory
# devices.
_OOM_MARKERS = (
    "OUT_OF_RESOURCES",
    "OUT_OF_HOST_MEMORY",
    "OUT_OF_DEVICE_MEMORY",
    "OUT_OF_MEMORY",
)


def _is_oom_failure(excinfo):
    """Return True if the raised exception looks like a device OOM error."""
    if excinfo is None:
        return False
    if issubclass(excinfo.type, MemoryError):
        return True
    text = str(excinfo.value).upper()
    return any(marker in text for marker in _OOM_MARKERS)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Dump device-memory and queue-event diagnostics when a test fails with an
    out-of-memory / out-of-resources device error.

    Only OOM-type failures are annotated (ordinary assertion failures are left
    untouched), so the two hypotheses -- events held on the SYCL queue vs.
    objects not garbage collected -- can be checked from the failure output.
    """
    outcome = yield
    report = outcome.get_result()

    if (
        report.when == "call"
        and report.failed
        and _is_oom_failure(call.excinfo)
    ):
        try:
            diagnostics = format_diagnostics(
                "OOM DIAGNOSTICS ON FAILURE", item.nodeid
            )
            report.sections.append(("Device memory diagnostics", diagnostics))
        except Exception as e:  # never let diagnostics mask the real failure
            report.sections.append(
                ("Device memory diagnostics", f"Failed to collect: {e}")
            )


# Per-file memory-trend logging.
#
# An intermittent OOM (seen once in several runs) is an accumulation signature:
# memory that creeps up over the session and only occasionally crosses the
# limit. Sampling once per test file (instead of per test) is cheap -- a few
# hundred samples over a run -- yet dense enough to read as a trend: watch
# whether free memory drifts down file-over-file (accumulation, i.e. events
# held on the queue or objects not collected) or stays flat until one file
# spikes (that file's footprint / external pressure).
_last_logged_file = None


def pytest_runtest_logstart(nodeid, location):
    """Log a memory overview the first time each test file is entered."""
    global _last_logged_file
    # `location` is (filename, lineno, testname); group by filename.
    test_file = location[0]
    if test_file == _last_logged_file:
        return
    _last_logged_file = test_file

    # No terminal writer here; a plain print is captured and shown with -s or
    # on failure, which is sufficient for an offline trend read.
    print("")
    print(format_diagnostics("MEMORY AT FILE START", test_file))
