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

"""
Tests dpnp's installed package.

Runs with one built package installed in a clean environment, as ci/hooks.py
sets up, from an empty working directory, so the tests import the installed
dpnp rather than the source tree. GH_CI_DEVICE=gpu requires a GPU. The test
requirements come from ci/config.yml. Writes JUnit reports to $GH_CI_RESULTS.

It runs in a scrubbed environment, so it sets those the SYCL runtime, the OpenCL
loader and the test suite need itself.
"""

import contextlib
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import sysconfig
import tempfile

if os.name == "nt":
    import winreg

PACKAGE = "dpnp"
# pytest's exit status when it collects nothing
NO_TESTS = 5
# Tests now and then fail or hang, and a rerun passes
MAIN_ATTEMPTS, TENSOR_ATTEMPTS = 2, 3
# A tensor test stuck at 99%: https://github.com/IntelPython/dpctl/issues/1954
TENSOR_TEST_TIMEOUT = 120
MAIN_TIMEOUT = 60 * 60
TENSOR_TIMEOUT = 20 * 60
# The Python version whose main suite runs every integer type
ALL_INT_TYPES_PYTHON = "3.14"
PROBE = """
import json
import dpctl

devices = dpctl.get_devices()
try:
    default = "gpu" if dpctl.select_default_device().is_gpu else "cpu"
except dpctl.SyclDeviceCreationError:
    default = "none"
print(json.dumps({
    "cpu": any(device.is_cpu for device in devices),
    "gpu": any(device.is_gpu for device in devices),
    "default": default,
}))
"""
# What the PyPI wheel of the runtime has for the prefix, which conda replaces on
# install
CONDA_PLACEHOLDER = "/opt/anaconda1anaconda2anaconda3"
VENDORS_KEY = r"SOFTWARE\Khronos\OpenCL\Vendors"


def running_python() -> str:
    """This interpreter's version as artifact names spell it, such as 3.14t."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    threaded = sysconfig.get_config_var("Py_GIL_DISABLED")
    return version + ("t" if threaded else "")


def merged_icds(scratch: pathlib.Path) -> str | None:
    """
    The directory of OpenCL ICD files for OCL_ICD_VENDORS, if there are any.

    It holds the environment's CPU runtime, with the path of its library, and
    the system's, such as a GPU driver's, so the loader sees both.
    """
    prefix = pathlib.Path(sys.prefix)
    system = pathlib.Path("/etc/OpenCL/vendors")
    merged = scratch / "vendors"
    merged.mkdir()
    for directory in (prefix / "etc" / "OpenCL" / "vendors", system):
        for icd in sorted(directory.glob("*.icd")):
            library = (
                icd.read_text(encoding="utf-8")
                .strip()
                .replace(CONDA_PLACEHOLDER, str(prefix))
            )
            local = prefix / "lib" / library
            if directory != system and local.is_file():
                library = str(local)
            if not (merged / icd.name).exists():
                (merged / icd.name).write_text(library + "\n", encoding="utf-8")
    return str(merged) if any(merged.iterdir()) else None


def runtime_environment(scratch: pathlib.Path) -> dict[str, str]:
    """The variables the SYCL runtime and the OpenCL loader need."""
    env = {"SYCL_CACHE_PERSISTENT": "1"}
    if os.name != "nt" and (icds := merged_icds(scratch)):
        env["OCL_ICD_VENDORS"] = icds
    return env


@contextlib.contextmanager
def registered_cpu_driver():
    """
    Registers the environment's OpenCL CPU runtime; that needs an administrator.

    The Windows loader finds drivers only in the registry, and the runtime
    does not register itself. The entry goes when the tests end. A GPU run
    or a machine with drivers registered already changes nothing.
    """
    bin_dir = pathlib.Path(sys.prefix, "Library", "bin")
    driver = next(bin_dir.rglob("intelocl64.dll"), None)
    if os.name != "nt" or os.environ.get("GH_CI_DEVICE") == "gpu":
        yield
    elif driver is None:
        print(f"no OpenCL CPU runtime under {bin_dir}")
        yield
    else:
        with winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, VENDORS_KEY) as key:
            if winreg.QueryInfoKey(key)[1]:
                print("OpenCL drivers are already registered; adding none")
                yield
                return
            winreg.SetValueEx(key, str(driver), 0, winreg.REG_DWORD, 0)
            try:
                yield
            finally:
                winreg.DeleteValue(key, str(driver))


def probe_devices(env: dict[str, str]) -> dict:
    """Lists the SYCL devices, and tells which kinds there are."""
    subprocess.run([sys.executable, "-I", "-m", "dpctl", "-f"], env=env)
    probe = subprocess.run(
        [sys.executable, "-I", "-c", PROBE],
        env=env,
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        print(probe.stderr)
        return {"cpu": False, "gpu": False, "default": "none"}
    return json.loads(probe.stdout.splitlines()[-1])


def device_problem(devices: dict) -> str | None:
    """Why the tests cannot say anything on these devices, if they cannot."""
    if not (devices["cpu"] or devices["gpu"]):
        return "there is no SYCL device"
    if os.environ.get("GH_CI_DEVICE") == "gpu" and not devices["gpu"]:
        return "GH_CI_DEVICE asks for a GPU, and there is none"
    return None


def pytest_command(
    module: str,
    report: pathlib.Path,
    suite: str,
    parallel: bool,
    extra: list[str],
) -> list[str]:
    """The pytest command for one module of the installed package."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-ra",
        "-p",
        "no:cacheprovider",
        f"--junitxml={report}",
        "-o",
        f"junit_suite_name={suite}",
        *extra,
        "--pyargs",
        module,
    ]
    return command + (["-n", "auto"] if parallel else [])


def run_suite(
    command: list[str], env: dict[str, str], timeout: int, attempts: int
) -> int | None:
    """
    Runs a pytest command until it passes or collects nothing, up to attempts
    times, killing it and its workers if it takes too long.

    Returns the last exit status, or None if the last attempt hung.
    """
    code = None
    for attempt in range(1, attempts + 1):
        with subprocess.Popen(
            command, env=env, start_new_session=os.name != "nt"
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                code = None
                print(f"hung for {timeout} s, attempt {attempt}", flush=True)
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        check=False,
                    )
                else:
                    os.killpg(process.pid, signal.SIGKILL)
        if code in (0, NO_TESTS):
            return code
        if code is not None:
            print(
                f"failed with exit status {code}, attempt {attempt}", flush=True
            )
    return code


def main() -> int:
    """Runs the installed package's test suites."""
    expected = os.environ["GH_CI_PYTHON"]
    if running_python() != expected:
        print(f"running Python {running_python()}, expected {expected}")
        return 1
    with tempfile.TemporaryDirectory() as scratch, registered_cpu_driver():
        env = dict(os.environ, **runtime_environment(pathlib.Path(scratch)))
        devices = probe_devices(env)
        print(f"devices: {devices}", flush=True)
        if problem := device_problem(devices):
            print(f"error: {problem}")
            trace = dict(env, SYCL_UR_TRACE="2", OCL_ICD_DEBUG="7")
            print(f"OpenCL ICDs: {env.get('OCL_ICD_VENDORS')}")
            subprocess.run(
                [sys.executable, "-I", "-m", "dpctl", "-f"], env=trace
            )
            return 1
        # Workers would share one GPU, and its memory.
        workers = (
            devices["default"] == "cpu"
            and importlib.util.find_spec("xdist") is not None
        )
        # The suite name says which platform and Python ran.
        cell = f"{os.environ.get('GH_CI_PLATFORM', 'linux')}.py{expected}"
        results = pathlib.Path(os.environ["GH_CI_RESULTS"])
        main_env = dict(env, SKIP_TENSOR_TESTS="1")
        # Every integer type in parallel exhausts a runner's memory.
        main_workers = workers and expected != ALL_INT_TYPES_PYTHON
        if expected == ALL_INT_TYPES_PYTHON:
            main_env["DPNP_TEST_ALL_INT_TYPES"] = "1"
        # The tensor tests need a lot of memory, so they run on their own.
        tensor_env = {
            name: value
            for name, value in env.items()
            if name != "SKIP_TENSOR_TESTS"
        }
        status = 0
        tensor_extra = [
            f"--timeout={TENSOR_TEST_TIMEOUT}",
            "--timeout-method=thread",
        ]
        for module, suite_env, name, parallel, timeout, attempts, extra in (
            (
                f"{PACKAGE}.tests",
                main_env,
                PACKAGE,
                main_workers,
                MAIN_TIMEOUT,
                MAIN_ATTEMPTS,
                [],
            ),
            (
                f"{PACKAGE}.tests.tensor",
                tensor_env,
                f"{PACKAGE}-tensor",
                workers,
                TENSOR_TIMEOUT,
                TENSOR_ATTEMPTS,
                tensor_extra,
            ),
        ):
            command = pytest_command(
                module,
                results / f"{name}.xml",
                f"{name} {cell}",
                parallel,
                extra,
            )
            print("+ " + " ".join(command), flush=True)
            code = run_suite(command, suite_env, timeout, attempts)
            if code is None:
                status = 1
                continue
            if code == NO_TESTS:
                # Say why nothing was collected.
                subprocess.run(
                    [sys.executable, "-I", "-c", f"import {module}"],
                    env=suite_env,
                )
            status |= code
    return 1 if status else 0


if __name__ == "__main__":
    sys.exit(main())
