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
Installs Intel's icx and the oneAPI libraries for dpnp's wheel builds.

cibuildwheel runs this before the builds, as pyproject.toml says. It fetches
micromamba, checked against its published sha256, and creates PREFIX with
conda-forge's oneMKL, oneDPL and oneTBB development packages the wheels build
against. On Windows PREFIX also gets conda-forge's dpcpp_win-64, the compiler
the conda recipe builds with. On Linux, in the manylinux container, icpx comes
from Intel's yum repository and is linked at COMPILER_LINK, and conda-forge's
libstdc++ moves from PREFIX to STDCXX: the build runs with it, as dpctl's
import needs it, but links the image's gcc-toolset one, whose symbol versions
manylinux_2_28 allows. It then prints the compiler's and oneMKL's versions
under "Installed SDK versions:", where ci/build.py reads them for
[tool.gh-ci.pin-compatible].

Usage:
  python ci/install_icx.py PREFIX
"""

# The manylinux container runs this with its Python 3.9
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

LINUX = sys.platform.startswith("linux")
# The compiler: Intel's yum package version on Linux, conda-forge's on Windows
LINUX_COMPILER = "2026.2"
WINDOWS_COMPILER = "dpcpp_win-64=2026.2.0"
LIBRARIES = ["mkl-devel-dpcpp>=2026.0,<2027", "onedpl-devel", "tbb-devel"]
# The micromamba build per platform, and the sha256 its release publishes
MICROMAMBA_RELEASE = (
    "https://github.com/mamba-org/micromamba-releases/releases/download/"
    "2.10.0-0/"
)
MICROMAMBA = {
    "linux": (
        "micromamba-linux-64",
        "701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd",
    ),
    "win32": (
        "micromamba-win-64",
        "f8b4284bd4eb1134122eaf93b3ea6c7be567e596de5829e1a6f478e34e41974a",
    ),
}
# What the printed versions are named, per conda package
VERSIONS = {"dpcpp_win-64": "dpcpp", "mkl-devel-dpcpp": "mkl-devel-dpcpp"}

# Linux only; [tool.cibuildwheel.linux.environment] names both paths
COMPILER_LINK = Path("/opt/icx")
STDCXX = Path("/opt/stdcxx")
INTEL_KEY = (
    "https://yum.repos.intel.com/intel-gpg-keys/"
    "GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB"
)
ONEAPI_REPO = f"""\
[oneAPI]
name=Intel oneAPI
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey={INTEL_KEY}
"""
LINUX_PACKAGE = f"intel-oneapi-compiler-dpcpp-cpp-{LINUX_COMPILER}"


def run(*command: str) -> int:
    """Runs command, echoed first, and returns its exit code."""
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


def fail(message: str) -> int:
    """Reports message and returns the exit code for it."""
    print(f"install_icx: {message}", file=sys.stderr)
    return 1


def fetch_micromamba(directory: Path) -> Path | None:
    """Downloads micromamba into directory, if its sha256 matches."""
    name, sha256 = MICROMAMBA["linux" if LINUX else "win32"]
    micromamba = directory / ("micromamba" if LINUX else "micromamba.exe")
    url = MICROMAMBA_RELEASE + name
    with urllib.request.urlopen(url, timeout=300) as response:
        micromamba.write_bytes(response.read())
    digest = hashlib.sha256(micromamba.read_bytes()).hexdigest()
    if digest != sha256:
        fail(f"micromamba has sha256 {digest}, not {sha256}")
        return None
    micromamba.chmod(0o755)
    return micromamba


def install_linux_compiler() -> int:
    """Installs icpx from Intel's yum repository and links it."""
    if code := run("rpm", "--import", INTEL_KEY):
        return code
    Path("/etc/yum.repos.d/oneAPI.repo").write_text(
        ONEAPI_REPO, encoding="utf-8"
    )
    if code := run("dnf", "install", "-y", LINUX_PACKAGE):
        return code
    COMPILER_LINK.symlink_to(Path("/opt/intel/oneapi/compiler", LINUX_COMPILER))
    return 0


def move_stdcxx(prefix: Path) -> None:
    """Moves conda-forge's libstdc++ out of prefix, where -lstdc++ finds it."""
    STDCXX.mkdir()
    for library in (prefix / "lib").glob("libstdc++.so.6*"):
        shutil.move(str(library), str(STDCXX / library.name))
    (prefix / "lib" / "libstdc++.so").unlink()


def print_versions(prefix: Path) -> int:
    """Prints the compiler's and oneMKL's versions for ci/build.py."""
    found = {}
    for record in (prefix / "conda-meta").glob("*.json"):
        meta = json.loads(record.read_text(encoding="utf-8"))
        if meta.get("name") in VERSIONS:
            found[VERSIONS[meta["name"]]] = meta["version"]
    if LINUX:
        rpm = subprocess.run(
            ["rpm", "-q", "--qf", "%{VERSION}", LINUX_PACKAGE],
            capture_output=True,
            text=True,
            check=False,
        )
        if not rpm.returncode:
            found["dpcpp"] = rpm.stdout.strip()
    if missing := sorted(set(VERSIONS.values()) - set(found)):
        return fail(f"no version of {', '.join(missing)}")
    print("Installed SDK versions:")
    for name, version in sorted(found.items()):
        print(f"- {name}=={version}", flush=True)
    return 0


def main() -> int:
    """Creates PREFIX with the libraries, and installs the compiler."""
    prefix = Path(sys.argv[1])
    if LINUX and (code := install_linux_compiler()):
        return code
    packages = LIBRARIES if LINUX else [WINDOWS_COMPILER, *LIBRARIES]
    with tempfile.TemporaryDirectory() as scratch:
        micromamba = fetch_micromamba(Path(scratch))
        if micromamba is None:
            return 1
        root = str(Path(scratch) / "root")
        command = ["create", "-y", "-r", root, "-p", str(prefix)]
        channels = ["--override-channels", "-c", "conda-forge"]
        if code := run(str(micromamba), *command, *channels, *packages):
            return code
    if LINUX:
        move_stdcxx(prefix)
    return print_versions(prefix)


if __name__ == "__main__":
    sys.exit(main())
