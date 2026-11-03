"""
Intel Python CI (gh-ci) benchmark of the installed package.

Intel's Jenkins runs this because the repository has a benchmarks/ directory:
once per pull request or tag, with the conda package of the newest regular
Linux Python installed in a clean conda-forge environment with the benchmark
requirements from ci/config.yml, on a bare-metal benchmarking machine, from an
empty working directory.

It runs the asv suite in benchmarks/ against the installed package, without a
build of its own, and records the results under the commit being tested, in
$GH_CI_RESULTS, which Jenkins keeps with the build. A pull request samples each
benchmark once, enough to show that every one still runs; a tag measures them
properly. A benchmark that fails fails the cell; a slower one does not, as
nothing compares the timings with a baseline yet.
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
