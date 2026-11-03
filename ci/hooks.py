"""
Runs ci/test.py against one built artifact, the way Intel Python CI does.

For the GitHub Actions test job.

Each Python version's artifact is installed into a fresh conda environment from
a local channel plus conda-forge, pinned to its exact build, with the test
requirements ci/config.yml lists, and the test runs with that environment's
Python from an empty working directory, on Linux or Windows. A wheel's
environment has only Python from conda-forge: the wheel and the test
requirements install with pip, as in the wheel's users' environments. The
interpreter spec and the requirements are the ones Jenkins uses. It gets the
same variables, and nothing else from the runner, as it does in Jenkins, so a
test that passes here passes there. Jenkins runs ci/test.py again on its own
machines, GPUs included; the scans are Jenkins' own and never run here.

Usage, from any Python 3.10 or newer; it runs itself again with the conda base
install's Python, from $CONDA, which setup-miniconda sets, or the conda on PATH:
  python ci/hooks.py --python 3.14 --artifacts DIR --work DIR [--kind wheel]

Needs conda, conda-package-handling and ruamel.yaml, which every conda base has,
and installs nothing into it.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matrix

# Dependencies of conda itself; YAML is read with PyYAML, else ruamel.yaml
try:
    from conda_package_handling import api as CPH_API
except ModuleNotFoundError:
    CPH_API = None
try:
    import yaml as PYYAML
except ModuleNotFoundError:
    PYYAML = None
try:
    from ruamel.yaml import YAML as RUAMEL_YAML
except ModuleNotFoundError:
    RUAMEL_YAML = None

ROOT = matrix.ROOT
IS_WINDOWS = os.name == "nt"
if IS_WINDOWS:
    PLATFORM, HOST_SUBDIR = "windows", "win-64"
elif sys.platform == "darwin":
    PLATFORM, HOST_SUBDIR = "macos", "osx-64"
else:
    PLATFORM, HOST_SUBDIR = "linux", "linux-64"
CONDA_SUFFIXES = (".conda", ".tar.bz2")
# All the test sees of the runner's environment, so no tokens reach it
KEEP = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "TERM",
    "TMPDIR",
    "TZ",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CONDA_PKGS_DIRS",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)


class HookError(Exception):
    """Raised when the artifact cannot be installed or ci/test.py is missing."""


def run(command: list[str], **options: object) -> None:
    """Echoes and runs a command, never through a shell."""
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True, **options)


def conda(*args: str) -> None:
    """Runs conda from the base environment running this script."""
    run([sys.executable, "-m", "conda", *args])


def empty_repodata(subdir: str) -> dict:
    """The repodata.json of a channel subdir with no packages."""
    return {
        "info": {"subdir": subdir},
        "packages": {},
        "packages.conda": {},
        "repodata_version": 1,
    }


def package_record(package: Path) -> tuple[str, dict]:
    """A conda package's subdir and repodata record, from its index.json."""
    if CPH_API is None:
        raise HookError(
            "run this with a conda base Python, "
            "which has conda-package-handling"
        )
    with tempfile.TemporaryDirectory() as scratch:
        CPH_API.extract(str(package), scratch, components="info")
        record = json.loads(
            (Path(scratch) / "info" / "index.json").read_text(encoding="utf-8")
        )
    data = package.read_bytes()
    record.update(
        {
            "md5": hashlib.md5(data, usedforsecurity=False).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
    )
    subdir = str(record.get("subdir") or "noarch")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9_]+)?", subdir):
        raise HookError(
            f"{package.name} declares an unusable subdir {subdir!r}"
        )
    return subdir, record


def stage_channel(packages: list[Path], channel: Path) -> None:
    """
    Lays the packages out as a local channel, by the subdir each one declares,
    and writes its repodata.json the way Jenkins does, without conda-index.
    """
    repodata = {
        subdir: empty_repodata(subdir) for subdir in ("noarch", HOST_SUBDIR)
    }
    for package in packages:
        subdir, record = package_record(package)
        (channel / subdir).mkdir(parents=True, exist_ok=True)
        shutil.copy2(package, channel / subdir / package.name)
        section = (
            "packages.conda" if package.name.endswith(".conda") else "packages"
        )
        repodata.setdefault(subdir, empty_repodata(subdir))[section][
            package.name
        ] = record
    for subdir, data in repodata.items():
        (channel / subdir).mkdir(parents=True, exist_ok=True)
        (channel / subdir / "repodata.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )


def test_python(version: str) -> str:
    """The interpreter spec for the test environment, as Jenkins spells it."""
    minor, threaded = matrix.version_key(version)
    # The .* matters: conda reads a bare 3.11 as exactly 3.11.0
    if minor < matrix.FREE_THREADED_MINOR:
        return f"3.{minor}.*"
    # conda-forge marks the ABI from 3.13, so 3.14 can't resolve to 3.14t
    return f'3.{minor}.* *_cp3{minor}{"t" if threaded else ""}'


def test_requirements() -> tuple[list[str], list[str], list[str] | None]:
    """
    The conda-forge and pip packages ci/config.yml installs next to the
    artifact, and a wheel's PyPI requirements, None when it gives none.
    """
    text = (ROOT / "ci" / "config.yml").read_text(encoding="utf-8")
    if PYYAML is not None:
        config = PYYAML.safe_load(text)
    elif RUAMEL_YAML is not None:
        config = RUAMEL_YAML(typ="safe", pure=True).load(text)
    else:
        raise HookError("reading ci/config.yml needs PyYAML or ruamel.yaml")
    test = (config or {}).get("test") or {}
    lists = []
    for key in ("requires", "pip_requires", "wheel_requires"):
        values = test.get(key)
        if values is not None and (
            not isinstance(values, list)
            or not all(isinstance(v, str) for v in values)
        ):
            raise HookError(
                f"ci/config.yml: test.{key} must be a list of package specs"
            )
        lists.append(values)
    return lists[0] or [], lists[1] or [], lists[2]


def exact_spec(package: Path) -> str:
    """name=version=build from a conda package file name."""
    stem = package.name
    for suffix in CONDA_SUFFIXES:
        stem = stem.removesuffix(suffix)
    name, version, build = stem.rsplit("-", 2)
    return f"{name}={version}={build}"


def env_python(env_dir: Path) -> Path:
    """The interpreter of a conda environment."""
    return env_dir / "python.exe" if IS_WINDOWS else env_dir / "bin" / "python"


def env_path(env_dir: Path) -> list[Path]:
    """
    The directories conda's activation would put first on PATH for an
    environment.
    """
    if not IS_WINDOWS:
        return [env_dir / "bin"]
    library = env_dir / "Library"
    return [
        env_dir,
        library / "mingw-w64" / "bin",
        library / "usr" / "bin",
        library / "bin",
        env_dir / "Scripts",
        env_dir / "bin",
    ]


def test_env(
    env_dir: Path, artifacts: Path, results: Path, args: argparse.Namespace
) -> dict:
    """The variables Jenkins gives ci/test.py, and no runner tokens."""
    env = {key: value for key, value in os.environ.items() if key in KEEP}
    env["PATH"] = os.pathsep.join(
        [*map(str, env_path(env_dir)), env.get("PATH", "")]
    )
    env.update(
        {
            "CONDA_PREFIX": str(env_dir),
            "GH_CI_ARTIFACTS": str(artifacts),
            "GH_CI_SOURCE": str(ROOT),
            "GH_CI_RESULTS": str(results),
            "GH_CI_KIND": args.kind,
            "GH_CI_PYTHON": args.python,
            "GH_CI_PLATFORM": PLATFORM,
            "GH_CI_EVENT": args.event,
            "GH_CI_COMMIT": args.commit,
        }
    )
    return env


def summarize(lines: list[str]) -> None:
    """Adds the results to the job summary page, in GitHub Actions."""
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as page:
            page.write("\n".join(lines) + "\n")


def install(args: argparse.Namespace) -> Path:
    """
    Creates the test environment with the artifact, pinned to its exact build;
    returns it.
    """
    system, selector = matrix.detect(
        ROOT, "wheel" if args.kind == "wheel" else ""
    )
    declared = {
        python.version: python for python in matrix.declared(system, selector)
    }
    if args.python not in declared:
        raise HookError(
            f"Python {args.python} is not declared; "
            f'declared: {", ".join(declared)}'
        )
    suffixes = (".whl",) if args.kind == "wheel" else CONDA_SUFFIXES
    packages = sorted(
        path
        for path in args.artifacts.rglob("*")
        if path.name.endswith(suffixes)
    )
    if not packages:
        raise HookError(f"{args.artifacts} holds no {args.kind} package")
    requires, pip_requires, wheel_requires = test_requirements()
    env_dir = args.work / "env"
    pip = [str(env_python(env_dir)), "-m", "pip", "install", "--no-input"]
    python = f"python {test_python(args.python)}"
    if args.kind == "wheel":
        conda_create(env_dir, python, "pip")
        wheel_specs = requires if wheel_requires is None else wheel_requires
        run([*pip, *map(str, packages), *wheel_specs])
    else:
        channel = args.work / "channel"
        stage_channel(packages, channel)
        conda_create(
            env_dir,
            python,
            *requires,
            *map(exact_spec, packages),
            channels=(channel.as_uri(),),
        )
    if pip_requires:
        run([*pip, *pip_requires])
    return env_dir


def conda_create(
    env_dir: Path, *specs: str, channels: tuple[str, ...] = ()
) -> None:
    """Creates a conda environment from conda-forge and channels."""
    conda(
        "create",
        "-y",
        "-q",
        "-p",
        str(env_dir),
        "--override-channels",
        *(arg for channel in channels for arg in ("-c", channel)),
        "-c",
        "conda-forge",
        *specs,
    )


def run_test(env_dir: Path, args: argparse.Namespace) -> int:
    """Runs ci/test.py with the environment's Python; returns its status."""
    script = ROOT / "ci" / "test.py"
    if not script.is_file():
        raise HookError("ci/test.py is required and missing")
    results, scratch = (
        args.work / "results" / "test",
        args.work / "scratch" / "test",
    )
    for directory in (results, scratch):
        directory.mkdir(parents=True, exist_ok=True)
    command = [str(env_python(env_dir)), str(script)]
    print(f"+ {shlex.join(command)}", flush=True)
    status = subprocess.run(
        command,
        cwd=scratch,
        check=False,
        env=test_env(env_dir, args.artifacts, results, args),
    ).returncode
    result = "passed" if status == 0 else f"failed, exit {status}"
    summarize(
        [
            f"### {matrix.PLATFORMS[PLATFORM]} Python {args.python}"
            f" {args.kind}",
            "",
            "| Test | Result |",
            "|---|---|",
            f"| ci/test.py | {result} |",
        ]
    )
    return status


def conda_base_python() -> Path:
    """The conda base install's Python: from $CONDA, else the conda on PATH."""
    root = os.environ.get("CONDA", "")
    if not root:
        found = shutil.which("conda")
        if not found:
            raise HookError(
                "no conda base install: set CONDA or put conda on PATH"
            )
        # conda lives in bin, Scripts or condabin, one level below the install.
        root = str(Path(found).resolve().parent.parent)
    python = env_python(Path(root))
    if not python.is_file():
        raise HookError(
            f"{root} is not a conda base install; it has no {python.name}"
        )
    return python


def main(argv: list[str] | None = None) -> int:
    """Installs the artifact, then runs ci/test.py against it."""
    if CPH_API is None:
        # Re-run with base's Python, which has conda-package-handling
        command = [
            str(conda_base_python()),
            str(Path(__file__).resolve()),
            *(sys.argv[1:] if argv is None else argv),
        ]
        return subprocess.run(command, check=False).returncode
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--python", required=True, help="a declared version, such as 3.14t"
    )
    parser.add_argument(
        "--artifacts", required=True, type=Path, help="the downloaded artifact"
    )
    parser.add_argument(
        "--work", required=True, type=Path, help="an empty scratch directory"
    )
    parser.add_argument(
        "--event", default="push", choices=("pr", "tag", "push")
    )
    parser.add_argument("--commit", default="")
    parser.add_argument(
        "--kind",
        default="conda",
        choices=("conda", "wheel"),
        help="what the artifact holds",
    )
    args = parser.parse_args(argv)
    args.artifacts, args.work = args.artifacts.resolve(), args.work.resolve()
    if args.work.exists() and (
        not args.work.is_dir() or any(args.work.iterdir())
    ):
        raise HookError(f"--work {args.work} must be an empty directory")
    return run_test(install(args), args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        HookError,
        matrix.DeclarationError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"hooks: {error}", file=sys.stderr)
        sys.exit(1)
