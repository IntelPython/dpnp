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
Benchmarks the installed package.

Runs the asv suite in benchmarks/ against the package installed in this
environment, from an empty working directory, without a build of its own, and
records the results under the commit being tested, in $GH_CI_RESULTS. A pull
request samples each benchmark once, enough to show that every one still runs;
a tag measures them properly. A benchmark that fails fails the run; a slower one
does not, as nothing compares the timings with a baseline yet.
"""

import json
import os
import pathlib
import subprocess
import sys

# asv's own dependency, which reads the comments in its configuration.
import json5

ASV_CONFIG = pathlib.Path("benchmarks", "asv.conf.json")


def asv_config(source: pathlib.Path, results: pathlib.Path) -> pathlib.Path:
    """Writes this run's asv configuration into the working directory.

    It is the repository's own, with absolute paths, since asv reads them from
    where the configuration is, and with results under GH_CI_RESULTS. It lists
    one branch, HEAD: the checkout is the commit under test, detached, and asv
    cannot resolve the repository's branches there.
    """
    path = source / ASV_CONFIG
    config = json5.loads(path.read_text(encoding="utf-8"))
    work = pathlib.Path.cwd()
    config.update(
        repo=str((path.parent / config.get("repo", "..")).resolve()),
        benchmark_dir=str(
            (path.parent / config.get("benchmark_dir", "benchmarks")).resolve()
        ),
        branches=["HEAD"],
        results_dir=str(results / "asv"),
        html_dir=str(work / "html"),
        env_dir=str(work / "env"),
    )
    written = work / "asv.conf.json"
    written.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return written


def main() -> int:
    """Runs the asv suite against the installed package."""
    config = asv_config(
        pathlib.Path(os.environ["GH_CI_SOURCE"]),
        pathlib.Path(os.environ["GH_CI_RESULTS"]),
    )
    asv = [sys.executable, "-m", "asv"]
    # asv keeps its machine description in ~/.asv-machine.json; this one stays
    # with the run rather than with the machine's user.
    home = pathlib.Path.cwd() / "home"
    home.mkdir(exist_ok=True)
    env = dict(os.environ, HOME=str(home))
    described = subprocess.run(
        asv + ["machine", "--yes", "--config", str(config)], env=env
    )
    if described.returncode:
        return described.returncode
    run = [
        "run",
        "--config",
        str(config),
        f"--environment=existing:{sys.executable}",
        f"--set-commit-hash={os.environ['GH_CI_COMMIT']}",
        "--no-pull",
        "--show-stderr",
        "--launch-method=spawn",
    ]
    if os.environ.get("GH_CI_EVENT") != "tag":
        run.append("--quick")
    # asv exits 2 when a benchmark failed.
    return subprocess.run(asv + run, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
