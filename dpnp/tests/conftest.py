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

import os
import sys
import warnings

import dpctl
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


def get_device_memory_info(device=None):
    """
    Safely retrieve device memory information.

    Returns dict with keys: 'global_mem_size', 'max_mem_alloc_size', 'local_mem_size'
    or None if information cannot be retrieved.
    """
    try:
        if device is None:
            device = dpctl.select_default_device()

        return {
            "global_mem_size": device.global_mem_size,
            "max_mem_alloc_size": device.max_mem_alloc_size,
            "local_mem_size": device.local_mem_size,
            "device_name": device.name,
            "device_type": str(device.device_type),
            "backend": str(device.backend),
        }
    except Exception as e:
        warnings.warn(f"Failed to get device memory info: {e}")
        return None


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

    # Get device memory information
    mem_info = get_device_memory_info(dev)

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

    # Log device memory information
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


# Memory logging hooks
# Set DPNP_TEST_LOG_MEMORY=1 to enable per-test memory logging
def pytest_runtest_setup(item):
    """Log memory info before each test if enabled."""
    if os.getenv("DPNP_TEST_LOG_MEMORY") == "1":
        mem_info = get_device_memory_info()
        if mem_info:
            # Get the pytest terminal writer to bypass capture
            tw = item.config.get_terminal_writer()
            tw.line()
            tw.write(
                f"[MEMORY BEFORE] {item.nodeid}: "
                f"Global={format_memory_size(mem_info['global_mem_size'])}, "
                f"MaxAlloc={format_memory_size(mem_info['max_mem_alloc_size'])}, "
                f"Local={format_memory_size(mem_info['local_mem_size'])}"
            )
            tw.line()


def pytest_runtest_teardown(item):
    """Log memory info after each test if enabled."""
    if os.getenv("DPNP_TEST_LOG_MEMORY") == "1":
        mem_info = get_device_memory_info()
        if mem_info:
            tw = item.config.get_terminal_writer()
            tw.write(
                f"[MEMORY AFTER] {item.nodeid}: "
                f"Global={format_memory_size(mem_info['global_mem_size'])}, "
                f"MaxAlloc={format_memory_size(mem_info['max_mem_alloc_size'])}, "
                f"Local={format_memory_size(mem_info['local_mem_size'])}"
            )
            tw.line()


def pytest_runtest_makereport(item, call):
    """
    Enhanced error reporting that handles device failures gracefully.

    This hook catches device errors during test reporting phase and
    provides meaningful error messages instead of crashing.
    """
    # Only intercept if we're in the call phase and there was an exception
    if call.when == "call" and call.excinfo is not None:
        # Check if this is a device-related error
        exc_type = call.excinfo.type
        exc_value = call.excinfo.value

        # Log device state if there's a device-related error
        if (
            "sycl" in str(exc_type).lower()
            or "device" in str(exc_value).lower()
        ):
            try:
                mem_info = get_device_memory_info()
                if mem_info:
                    print(
                        f"\n[DEVICE ERROR] Test: {item.nodeid}"
                        f"\n  Device: {mem_info['device_name']}"
                        f"\n  Backend: {mem_info['backend']}"
                        f"\n  Global Memory: {format_memory_size(mem_info['global_mem_size'])}"
                    )
            except Exception as e:
                print(
                    f"\n[DEVICE ERROR] Failed to retrieve device info during error: {e}"
                )
