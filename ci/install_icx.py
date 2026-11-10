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
Installs Intel's icx and the oneAPI libraries for dpnp's Windows wheel builds.

cibuildwheel runs this before the builds, as pyproject.toml says. It fetches
micromamba, checked against its published sha256, and creates PREFIX with
conda-forge's dpcpp_win-64, the compiler the conda recipe builds with, and the
oneMKL, oneDPL and oneTBB development packages it builds against.

Usage:
  python ci/install_icx.py PREFIX
"""

import hashlib
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MICROMAMBA = (
    "https://github.com/mamba-org/micromamba-releases/releases/download/"
    "2.10.0-0/micromamba-win-64"
)
MICROMAMBA_SHA256 = (
    "f8b4284bd4eb1134122eaf93b3ea6c7be567e596de5829e1a6f478e34e41974a"
)
PACKAGES = [
    "dpcpp_win-64=2026.2.0",
    "mkl-devel-dpcpp>=2026.0,<2027",
    "onedpl-devel",
    "tbb-devel",
]


def main() -> int:
    """Creates PREFIX with the compiler and the libraries."""
    prefix = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as scratch:
        micromamba = Path(scratch) / "micromamba.exe"
        with urllib.request.urlopen(MICROMAMBA, timeout=300) as response:
            micromamba.write_bytes(response.read())
        digest = hashlib.sha256(micromamba.read_bytes()).hexdigest()
        if digest != MICROMAMBA_SHA256:
            print(
                f"install_icx: micromamba has sha256 {digest}, "
                f"not {MICROMAMBA_SHA256}",
                file=sys.stderr,
            )
            return 1
        command = [
            str(micromamba),
            "create",
            "-y",
            "-r",
            str(Path(scratch) / "root"),
            "-p",
            str(prefix),
            "--override-channels",
            "-c",
            "conda-forge",
            *PACKAGES,
        ]
        print("+ " + " ".join(command), flush=True)
        return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
