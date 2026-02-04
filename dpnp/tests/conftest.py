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
import json
import warnings

import dpctl
import numpy
import pytest

from collections import Counter

from . import config as dtype_config

if numpy.lib.NumpyVersion(numpy.__version__) >= "2.0.0b1":
    from numpy.exceptions import ComplexWarning
else:
    from numpy import ComplexWarning

import dpnp

from .helper import get_dev_id

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

DPNP_INFRA_WARNINGS_ENABLE = os.getenv("DPNP_INFRA_WARNINGS_ENABLE", "0") == "1"
DPNP_INFRA_WARNINGS_DIRECTORY = os.getenv("DPNP_INFRA_WARNINGS_DIRECTORY", None)

def _origin_from_filename(filename: str) -> str:
    file = (filename or "").replace("\\", "/")
    if "/dpnp/" in file or file.startswith("dpnp/"):
        return "dpnp"
    elif "/numpy/" in file or file.startswith("numpy/"):
        return "numpy"
    elif "/dpctl/" in file or file.startswith("dpctl/"):
        return "dpctl"
    else:
        return "third_party"

_warn_counts = Counter()
_warn_events = {}
_warn_totals = Counter()
_warn_env = {}
_warn_events_fp = None
_warn_events_file = None
_warn_print_events = os.getenv("DPNP_INFRA_WARNINGS_PRINT", "0") == "1"


def _json_dumps_one_line(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))

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

    if DPNP_INFRA_WARNINGS_ENABLE:
        try:
            numpy_version = numpy.__version__
            numpy_path = getattr(numpy, "__file__", "unknown")
            dpnp_version = dpnp.__version__
            dpnp_path = getattr(dpnp, "__file__", "unknown")
            dpctl_version = dpctl.__version__
            dpctl_path = getattr(dpctl, "__file__", "unknown")
        except Exception:
            numpy_version = "unknown"
            dpnp_version = "unknown"
            numpy_path = "unknown"
            dpnp_path = "unknown"
            dpctl_version = "unknown"
            dpctl_path = "unknown"

        global _warn_events_fp
        global _warn_events_file
        if DPNP_INFRA_WARNINGS_DIRECTORY:
            os.makedirs(DPNP_INFRA_WARNINGS_DIRECTORY, exist_ok=True)
            # Optional override for file name.
            events_file_override = os.getenv("DPNP_INFRA_WARNINGS_EVENTS_FILE")
            if events_file_override:
                _warn_events_file = events_file_override
            else:
                _warn_events_file = os.path.join(
                    DPNP_INFRA_WARNINGS_DIRECTORY,
                    "dpnp_infra_warnings_events.jsonl",
                )
            # Line-buffered for streaming CI artifacts without holding everything in memory.
            _warn_events_fp = open(
                _warn_events_file,
                "w",
                encoding="utf-8",
                buffering=1,
                newline="\n",
            )

        _warn_env.update({
            "numpy_version": numpy_version,
            "numpy_path": numpy_path,
            "dpnp_version": dpnp_version,
            "dpnp_path": dpnp_path,
            "dpctl_version": dpctl_version,
            "dpctl_path": dpctl_path,
            "job": os.getenv("JOB_NAME", "unknown"),
            "build_number": os.getenv("BUILD_NUMBER", "unknown"),
            "git_sha": os.getenv("GIT_COMMIT", "unknown"),
            "events_file": _warn_events_file,
        })


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


def pytest_warning_recorded(warning_message, when, nodeid, location):
    if not DPNP_INFRA_WARNINGS_ENABLE:
        return

    category = getattr(
        getattr(warning_message, "category", None),
        "__name__",
        str(getattr(warning_message, "category", "Warning")),
    )
    msg = str(getattr(warning_message, "message", warning_message))

    filename = getattr(warning_message, "filename", None) or (
        location[0] if location and len(location) > 0 else None
    )
    lineno = getattr(warning_message, "lineno", None) or (
        location[1] if location and len(location) > 1 else None
    )
    func = location[2] if location and len(location) > 2 else None

    origin = _origin_from_filename(filename or "")
    key = f"{category}||{origin}||{msg}"
    _warn_counts[key] += 1
    _warn_totals[f"category::{category}"] += 1
    _warn_totals[f"origin::{origin}"] += 1
    _warn_totals[f"phase::{when}"] += 1

    if key not in _warn_events:
        _warn_events[key] = {
            "category": category,
            "origin": origin,
            "when": when,
            "nodeid": nodeid,
            "filename": filename,
            "lineno": lineno,
            "function": func,
            "message": msg,
        }

    event = {
        "when": when,
        "nodeid": nodeid,
        "category": category,
        "origin": origin,
        "message": msg,
        "filename": filename,
        "lineno": lineno,
        "function": func,
    }

    # Write every warning event to artifact (JSONL).
    if _warn_events_fp is not None:
        try:
            _warn_events_fp.write(_json_dumps_one_line(event) + "\n")
        except Exception:
            # Never break the test run due to artifact IO.
            pass

    # Optionally print every warning event to console (one-line JSON).
    if _warn_print_events:
        try:
            sys.stderr.write("DPNP_WARNING_EVENT " + _json_dumps_one_line(event) + "\n")
            sys.stderr.flush()
        except Exception:
            pass


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not DPNP_INFRA_WARNINGS_ENABLE:
        return

    summary = {
        "schema_version": "1.0",
        "exit_status": exitstatus,
        "environment": dict(_warn_env),
        "total_warning_events": sum(_warn_counts.values()),
        "unique_warning_types": len(_warn_counts),
        "totals": dict(_warn_totals),
        "top_unique_warnings": [
            dict(_warn_events[k], count=c)
            for k, c in _warn_counts.most_common(50)
            if k in _warn_events
        ],
    }

    if DPNP_INFRA_WARNINGS_DIRECTORY:
        os.makedirs(DPNP_INFRA_WARNINGS_DIRECTORY, exist_ok=True)
        output_file = os.path.join(
            DPNP_INFRA_WARNINGS_DIRECTORY, "dpnp_infra_warnings_summary.json"
        )
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)
            terminalreporter.write_line(
                f"DPNP infrastructure warnings summary written to: {output_file}"
            )
        except Exception as e:
            terminalreporter.write_line(
                f"Failed to write DPNP infrastructure warnings summary to: {output_file}. Error: {e}"
            )

    global _warn_events_fp
    if _warn_events_fp is not None:
        try:
            _warn_events_fp.close()
        except Exception:
            pass
        _warn_events_fp = None

    terminalreporter.write_line("DPNP_WARNINGS_SUMMARY_BEGIN")
    terminalreporter.write_line(_json_dumps_one_line(summary))
    terminalreporter.write_line("DPNP_WARNINGS_SUMMARY_END")


def pytest_unconfigure(config):
    global _warn_events_fp
    if _warn_events_fp is not None:
        try:
            _warn_events_fp.close()
        except Exception:
            pass
        _warn_events_fp = None
