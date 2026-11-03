"""
The platforms and Python versions this repository builds.

Both are read from the repository's own build files.

Every build system already has a standard place for both, so each is written
once, where the build reads it, and the GitHub Actions matrix and Intel Python
CI (gh-ci) both follow it:

  rattler-build, selected by recipe.yaml
    Python versions  python: in variants.yaml next to the recipe
    platforms        build: skip: in the recipe
  conda-build, selected by meta.yaml
    Python versions  python: in conda_build_config.yaml next to the recipe
    platforms        skip: true  # [selector] lines in the recipe
  wheel, selected by pyproject.toml
    Python versions  the Programming Language :: Python :: 3.N classifiers;
                     any Free Threading classifier adds 3.Nt for each from 3.14
    platforms        the Operating System :: classifiers

A recipe builds on every platform gh-ci builds, Linux, Windows and macOS on
Intel, unless a skip that depends on the platform alone rules one out, such as
`skip: [win]` or `skip: true  # [osx]`; a skip that also depends on something
else, such as `win and py < 311` or `osx and arm64`, skips some builds, not the
platform. A pyproject.toml without Operating System classifiers, or with OS
Independent, builds on all three. Of the declared platforms, the matrix holds
those GitHub Actions has runners for, BUILT_PLATFORMS. macOS builds and tests in
GitHub Actions only: Jenkins tests, scans and releases Linux and Windows.

meson-python, scikit-build-core and every other PEP 517 backend declare these in
pyproject.toml too, so the wheel row covers them. Another convention, such as
pixi.toml, is one more reader in READERS and PLATFORM_READERS and, if it is a
new build system, one more entry in SYSTEMS.

Usage:
  python ci/matrix.py [--system SYSTEM] [--event EVENT] [--github-output NAME]

Prints the build matrix as JSON for strategy.matrix: {"include": [{"platform":
"linux", "name": "Linux", "python": "3.14"}, ...]}, every platform with every
Python version. For a pull_request event only the oldest and the newest regular
versions are listed, which is all gh-ci tests on a pull request. --github-output
also writes NAME=<matrix> to $GITHUB_OUTPUT, for later jobs. Standard library
only; reading pyproject.toml needs Python 3.11 or newer for tomllib.
"""

import argparse
import ast
import json
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Only the wheel reader needs it, and only it needs Python 3.11 or newer.
try:
    import tomllib as TOMLLIB
except ModuleNotFoundError:
    TOMLLIB = None

ROOT = Path(__file__).resolve().parent.parent

# Build system -> files that select it, in the order they are looked for. Where
# a repository still has conda-recipe-cf/ next to conda-recipe/, the conda-forge
# one comes first: gh-ci tests from conda-forge alone.
SYSTEMS = {
    "rattler-build": ("recipe.yaml", "recipe/recipe.yaml"),
    "conda-build": (
        "conda-recipe-cf/meta.yaml",
        "meta.yaml",
        "conda-recipe/meta.yaml",
        "recipe/meta.yaml",
    ),
    "wheel": ("pyproject.toml",),
}

# The platforms gh-ci builds and tests, and how artifact names spell them.
PLATFORMS = {"linux": "Linux", "windows": "Windows", "macos": "macOS"}
# The platforms the workflow can build on: GitHub-hosted runners have all
# three. A declared platform missing here is reported and left out.
BUILT_PLATFORMS = ("linux", "windows", "macos")
# What a platform-only selector means on each platform, for both conda-build and
# rattler-build.
SELECTORS = {
    "linux": {
        "linux": True,
        "linux64": True,
        "unix": True,
        "win": False,
        "win64": False,
        "win32": False,
        "osx": False,
        "osx64": False,
        "x86_64": True,
        "x86": False,
        "aarch64": False,
        "arm64": False,
        "ppc64le": False,
        "s390x": False,
    },
    "windows": {
        "linux": False,
        "linux64": False,
        "unix": False,
        "win": True,
        "win64": True,
        "win32": False,
        "osx": False,
        "osx64": False,
        "x86_64": True,
        "x86": False,
        "aarch64": False,
        "arm64": False,
        "ppc64le": False,
        "s390x": False,
    },
    # Intel Macs only, as MKL has no Apple silicon build.
    "macos": {
        "linux": False,
        "linux64": False,
        "unix": True,
        "win": False,
        "win64": False,
        "win32": False,
        "osx": True,
        "osx64": True,
        "x86_64": True,
        "x86": False,
        "aarch64": False,
        "arm64": False,
        "ppc64le": False,
        "s390x": False,
    },
}
CONDA_SKIP_RE = re.compile(r"\s*skip\s*:\s*(?:true|True|yes)\s*#\s*\[(.+)\]\s*")
OS_CLASSIFIERS = {
    "Operating System :: OS Independent": tuple(PLATFORMS),
    "Operating System :: POSIX": ("linux",),
    "Operating System :: Unix": ("linux",),
    "Operating System :: POSIX :: Linux": ("linux",),
    "Operating System :: Microsoft :: Windows": ("windows",),
    "Operating System :: MacOS": ("macos",),
}

PULL_REQUEST_EVENTS = ("pull_request",)
OLDEST_MINOR = 10
FREE_THREADED_MINOR = 13
# A Free Threading classifier adds 3.Nt from 3.14, the first supported one,
# which is also the oldest cibuildwheel builds.
CLASSIFIED_FREE_THREADED_MINOR = 14

# A conda python variant: 3.12, 3.12.*, 3.12.* *_cpython, 3.13.* *_cp313 or
# 3.14.* *_cp314t.
CONDA_SPEC_RE = re.compile(
    r"3\.(\d+)(?:\.\*)?(?:\s+\*_(?:cpython|cp3(\d+)(t?)))?"
)
CLASSIFIER_RE = re.compile(r"Programming Language :: Python :: 3\.(\d+)")
FREE_THREADING_CLASSIFIER = "Programming Language :: Python :: Free Threading"
LOWER_BOUND_RE = re.compile(r">=\s*3\.(\d+)")


class DeclarationError(Exception):
    """
    Raised when the build files declare no Python versions, or ones this cannot
    read.
    """


@dataclass(frozen=True)
class Python:
    """One declared Python version."""

    version: str  # as artifact names and gh-ci spell it: 3.14 or 3.14t
    spec: str  # as the build system spells it, passed back to it unchanged


def version_key(version: str) -> tuple[int, bool]:
    """(minor, free-threaded); also the sort order."""
    return int(version.rstrip("t").split(".")[1]), version.endswith("t")


def checked(minor: int, threaded: bool, where: str) -> str:
    """The version string, once it is one gh-ci can test."""
    if minor < OLDEST_MINOR or (threaded and minor < FREE_THREADED_MINOR):
        raise DeclarationError(
            f'{where}: 3.{minor}{"t" if threaded else ""} is not supported; '
            f"the oldest is 3.{OLDEST_MINOR}, free-threaded from "
            f"3.{FREE_THREADED_MINOR}"
        )
    return f'3.{minor}{"t" if threaded else ""}'


def _strip_comment(line: str, where: str) -> str:
    """
    The line without a trailing comment; conda-build selectors are refused, not
    guessed at.
    """
    in_quote = ""
    for index, char in enumerate(line):
        if char in "\"'" and in_quote in ("", char):
            in_quote = "" if in_quote else char
        elif (
            char == "#"
            and not in_quote
            and (index == 0 or line[index - 1].isspace())
        ):
            if line[index:].lstrip("# ").startswith("["):
                raise DeclarationError(
                    f"{where}: selectors in the python list are not supported"
                )
            return line[:index].rstrip()
    return line.rstrip()


def _unquote(value: str, where: str) -> str:
    """A plain or quoted YAML scalar."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if not value or value[0] in "{[&*!|>" or ":" in value:
        raise DeclarationError(
            f"{where}: write each python entry as a plain value, not {value!r}"
        )
    return value


def yaml_python_list(text: str, where: str) -> list[str]:
    """
    The top-level python: list of a variants file, without a YAML library.

    Accepts a block list or a one-line [a, b] list of plain or quoted values.
    Anything richer, such as if/then selectors or templating, is refused so the
    matrix is never guessed.
    """
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if not re.match(r"python\s*:", line):
            continue
        rest = _strip_comment(line.split(":", 1)[1], where).strip()
        if rest.startswith("[") and rest.endswith("]"):
            return [
                _unquote(item, where)
                for item in rest[1:-1].split(",")
                if item.strip()
            ]
        if rest:
            raise DeclarationError(f"{where}: python must be a list")
        values = []
        for item in lines[number + 1 :]:
            # Checked before the comment, as the next key may have a selector
            if item[:1] not in ("", " ", "\t", "-", "#"):
                break  # the next top-level key
            entry = _strip_comment(item, where).lstrip()
            if not entry:
                continue
            if not entry.startswith("- "):
                raise DeclarationError(
                    f"{where}: python must be a plain list of versions"
                )
            values.append(_unquote(entry[2:], where))
        return values
    raise DeclarationError(f"{where} has no top-level python: list")


def from_conda_variants(path: Path) -> list[Python]:
    """The python: list of a variants.yaml or a conda_build_config.yaml."""
    where = (
        path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ROOT)
        else str(path)
    )
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as ex:
        raise DeclarationError(
            f"{where} is missing; list the Python versions to build there"
        ) from ex
    pythons = []
    for spec in yaml_python_list(text, where):
        match = CONDA_SPEC_RE.fullmatch(spec)
        if not match or (match[2] and match[2] != match[1]):
            raise DeclarationError(
                f"{where}: {spec!r} is not a CPython variant such as "
                '"3.13.* *_cp313" or "3.14.* *_cp314t"'
            )
        pythons.append(
            Python(checked(int(match[1]), bool(match[3]), where), spec)
        )
    return pythons


def from_pyproject(path: Path) -> list[Python]:
    """
    The 3.N classifiers in pyproject.toml, plus 3.Nt from 3.14 if free-threading
    is declared.
    """
    if TOMLLIB is None:
        raise DeclarationError(
            "reading pyproject.toml needs Python 3.11 or newer"
        )
    project = TOMLLIB.loads(path.read_text(encoding="utf-8")).get("project", {})
    classifiers = project.get("classifiers", [])
    minors = sorted(
        {
            int(match[1])
            for classifier in classifiers
            if (match := CLASSIFIER_RE.fullmatch(classifier))
        }
    )
    if not minors:
        raise DeclarationError(
            "pyproject.toml lists no Programming Language :: Python :: 3.N "
            "classifiers"
        )
    bound = LOWER_BOUND_RE.search(project.get("requires-python", ""))
    if bound and minors[0] < int(bound[1]):
        raise DeclarationError(
            f"pyproject.toml classifies 3.{minors[0]}, below requires-python "
            f'{project["requires-python"]}'
        )
    threaded = any(
        classifier.startswith(FREE_THREADING_CLASSIFIER)
        for classifier in classifiers
    )
    versions = [checked(minor, False, "pyproject.toml") for minor in minors]
    versions += [
        f"3.{minor}t"
        for minor in minors
        if threaded and minor >= CLASSIFIED_FREE_THREADED_MINOR
    ]
    return [Python(version, version) for version in versions]


def skips_platform(expression: str, platform: str) -> bool | None:
    """
    Whether a skip expression rules out a whole platform: True or False when it
    depends on the platform alone, None when it also depends on anything else,
    such as the Python version.
    """
    try:
        tree = ast.parse(
            expression.strip().removeprefix("${{").removesuffix("}}").strip(),
            mode="eval",
        ).body
    except SyntaxError:
        return None

    def value(node: ast.AST) -> bool:
        if isinstance(node, ast.BoolOp):
            values = [value(item) for item in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not value(node.operand)
        if isinstance(node, ast.Name) and node.id in SELECTORS[platform]:
            return SELECTORS[platform][node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        raise ValueError(ast.dump(node))

    try:
        return value(tree)
    except ValueError:
        return None


def platforms_without(skips: list[str], where: str) -> list[str]:
    """Every platform gh-ci builds, but those a platform-only skip rules out."""
    kept = [
        platform
        for platform in PLATFORMS
        if not any(skips_platform(skip, platform) for skip in skips)
    ]
    if not kept:
        raise DeclarationError(
            f"{where} skips every platform gh-ci builds: "
            + ", ".join(PLATFORMS.values())
        )
    return kept


def yaml_build_skips(text: str, where: str) -> list[str]:
    """
    The skip: entries directly under a rattler-build recipe's top-level build:,
    as a list of plain or quoted values, a one-line [a, b] list, or one value;
    none when there is no skip.
    """
    lines = text.splitlines()
    starts = [
        number
        for number, line in enumerate(lines)
        if re.match(r"build\s*:\s*(#.*)?$", line)
    ]
    if not starts:
        return []
    child = ""
    for number in range(starts[0] + 1, len(lines)):
        line = (
            _strip_comment(lines[number], where)
            if "#" in lines[number]
            else lines[number]
        )
        if not line.strip():
            continue
        if not line[0].isspace():
            return []  # the next top-level key
        indent = line[: len(line) - len(line.lstrip())]
        child = child or indent
        match = (
            re.match(r"skip\s*:(.*)", line.lstrip())
            if indent == child
            else None
        )
        if not match:
            continue
        rest = match[1].strip()
        if rest.startswith("[") and rest.endswith("]"):
            return [
                _unquote(item, where)
                for item in rest[1:-1].split(",")
                if item.strip()
            ]
        if rest:
            return [_unquote(rest, where)]
        values = []
        for item in lines[number + 1 :]:
            entry = item.strip()
            if not entry or entry.startswith("#"):
                continue
            if not entry.startswith("- "):
                break
            values.append(_unquote(entry[2:].split(" #")[0], where))
        return values
    return []


def from_rattler_recipe(recipe: Path) -> list[str]:
    """The platforms a rattler-build recipe builds on, from its build: skip:."""
    where = (
        recipe.relative_to(ROOT).as_posix()
        if recipe.is_relative_to(ROOT)
        else str(recipe)
    )
    return platforms_without(
        yaml_build_skips(recipe.read_text(encoding="utf-8"), where), where
    )


def from_conda_recipe(recipe: Path) -> list[str]:
    """
    The platforms a conda-build meta.yaml builds on, from its skip: true  #
    [selector] lines.
    """
    where = (
        recipe.relative_to(ROOT).as_posix()
        if recipe.is_relative_to(ROOT)
        else str(recipe)
    )
    skips = [
        match[1]
        for line in recipe.read_text(encoding="utf-8").splitlines()
        if (match := CONDA_SKIP_RE.fullmatch(line))
    ]
    return platforms_without(skips, where)


def from_os_classifiers(path: Path) -> list[str]:
    """
    The platforms pyproject.toml's Operating System classifiers name; all of
    them when it names none.
    """
    if TOMLLIB is None:
        raise DeclarationError(
            "reading pyproject.toml needs Python 3.11 or newer"
        )
    classifiers = (
        TOMLLIB.loads(path.read_text(encoding="utf-8"))
        .get("project", {})
        .get("classifiers", [])
    )
    named = [
        classifier
        for classifier in classifiers
        if classifier.startswith("Operating System ::")
    ]
    if not named:
        return list(PLATFORMS)
    found = {
        platform
        for classifier in named
        for prefix, platforms in OS_CLASSIFIERS.items()
        if classifier == prefix or classifier.startswith(prefix + " ::")
        for platform in platforms
    }
    if not found:
        raise DeclarationError(
            "pyproject.toml names no Operating System gh-ci builds: "
            + ", ".join(PLATFORMS.values())
        )
    return [platform for platform in PLATFORMS if platform in found]


# Build system -> where it declares Python versions, given the file that
# selected it.
READERS: dict[str, Callable[[Path], list[Python]]] = {
    "rattler-build": lambda recipe: from_conda_variants(
        recipe.parent / "variants.yaml"
    ),
    "conda-build": lambda recipe: from_conda_variants(
        recipe.parent / "conda_build_config.yaml"
    ),
    "wheel": from_pyproject,
}

# Build system -> where it declares the platforms it builds on, given the file
# that selected it.
PLATFORM_READERS: dict[str, Callable[[Path], list[str]]] = {
    "rattler-build": from_rattler_recipe,
    "conda-build": from_conda_recipe,
    "wheel": from_os_classifiers,
}


def detect(root: Path, system: str = "") -> tuple[str, Path]:
    """The build system and the file that selects it."""
    for name, candidates in SYSTEMS.items():
        if system and name != system:
            continue
        for candidate in candidates:
            if (root / candidate).is_file():
                return name, root / candidate
    wanted = system or " or ".join(SYSTEMS)
    raise DeclarationError(
        f"no {wanted} input found; looked for "
        + ", ".join(c for name, cs in SYSTEMS.items() for c in cs)
    )


def declared(system: str, selector: Path) -> list[Python]:
    """The build system's Python versions, sorted, each listed once."""
    pythons = READERS[system](selector)
    versions = [python.version for python in pythons]
    if not pythons:
        raise DeclarationError(f"{system} declares no Python versions")
    if len(set(versions)) != len(versions):
        raise DeclarationError(
            f"{system} declares a Python version twice: {versions}"
        )
    return sorted(pythons, key=lambda python: version_key(python.version))


def declared_platforms(system: str, selector: Path) -> list[str]:
    """The platforms the build system builds on, in PLATFORMS order."""
    return PLATFORM_READERS[system](selector)


def for_event(pythons: list[Python], event: str) -> list[Python]:
    """All of them, or for a pull request the oldest and newest regular ones."""
    if event not in PULL_REQUEST_EVENTS:
        return pythons
    regular = [python for python in pythons if not python.version.endswith("t")]
    regular = regular or pythons
    return list(dict.fromkeys(regular[:1] + regular[-1:]))


def build_matrix(system: str, selector: Path, event: str) -> dict:
    """
    Every declared platform Actions can build on, with every Python version the
    event builds.
    """
    pythons = for_event(declared(system, selector), event)
    platforms = declared_platforms(system, selector)
    waiting = [
        PLATFORMS[platform]
        for platform in platforms
        if platform not in BUILT_PLATFORMS
    ]
    if waiting:
        print(
            f'matrix: {", ".join(waiting)} declared but not built: '
            "no runners for it yet",
            file=sys.stderr,
        )
    built = [platform for platform in platforms if platform in BUILT_PLATFORMS]
    if not built:
        raise DeclarationError(
            "the build files declare only platforms there are no runners "
            "for yet: " + ", ".join(waiting)
        )
    return {
        "include": [
            {
                "platform": platform,
                "name": PLATFORMS[platform],
                "python": python.version,
            }
            for platform in built
            for python in pythons
        ]
    }


def main(argv: list[str] | None = None) -> int:
    """Prints the matrix to build; writes it for GitHub Actions if asked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--system",
        default="",
        choices=("", *SYSTEMS),
        help="build system to read instead of the detected one",
    )
    parser.add_argument(
        "--event", default="", help="GitHub event name, such as pull_request"
    )
    parser.add_argument(
        "--github-output",
        default="",
        metavar="NAME",
        help="also write NAME=<matrix> to $GITHUB_OUTPUT",
    )
    args = parser.parse_args(argv)
    if args.github_output:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", args.github_output):
            raise DeclarationError(
                f"{args.github_output!r} is not an output name"
            )
        if not os.environ.get("GITHUB_OUTPUT"):
            raise DeclarationError(
                "--github-output needs $GITHUB_OUTPUT, which Actions sets"
            )
    system, selector = detect(ROOT, args.system)
    matrix = build_matrix(system, selector, args.event)
    cells = ", ".join(
        f'{cell["name"]} {cell["python"]}' for cell in matrix["include"]
    )
    print(
        f"matrix: {system} from {selector.relative_to(ROOT)}: {cells}",
        file=sys.stderr,
    )
    text = json.dumps(matrix)
    print(text)
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"{args.github_output}={text}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DeclarationError as error:
        print(f"matrix: {error}", file=sys.stderr)
        sys.exit(1)
