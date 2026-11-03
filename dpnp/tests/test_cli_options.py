import os
import subprocess
import sys

import dpnp


def test_includes():
    res = subprocess.run(
        [sys.executable, "-m", "dpnp", "--includes"],
        capture_output=True,
    )
    assert res.returncode == 0
    assert res.stdout
    flags = res.stdout.decode("utf-8")
    res = subprocess.run(
        [sys.executable, "-m", "dpnp", "--include-dir"],
        capture_output=True,
    )
    assert res.returncode == 0
    assert res.stdout
    include_dir = res.stdout.decode("utf-8")
    assert flags == "-I " + include_dir


def test_tensor_includes():
    res = subprocess.run(
        [sys.executable, "-m", "dpnp", "--tensor-includes"],
        capture_output=True,
    )
    assert res.returncode == 0
    assert res.stdout
    flags = res.stdout.decode("utf-8")
    res = subprocess.run(
        [sys.executable, "-m", "dpnp", "--tensor-include-dir"],
        capture_output=True,
    )
    assert res.returncode == 0
    assert res.stdout
    dir = res.stdout.decode("utf-8")
    assert flags == "-I " + dir
    assert os.path.isdir(os.path.join(dir.strip(), "kernels"))


def test_get_include():
    res = subprocess.run(
        [sys.executable, "-m", "dpnp", "--tensor-include-dir"],
        capture_output=True,
    )
    assert res.returncode == 0
    tensor_include_dir = res.stdout.decode("utf-8").strip()
    include_dir = dpnp.get_include()
    assert os.path.realpath(include_dir) == os.path.realpath(tensor_include_dir)
    assert os.path.isdir(os.path.join(include_dir, "kernels"))


def test_cmakedir():
    res = subprocess.run(
        [sys.executable, "-m", "dpnp", "--cmakedir"], capture_output=True
    )
    assert res.returncode == 0
    assert res.stdout
    cmake_dir = res.stdout.decode("utf-8").strip()
    assert os.path.exists(os.path.join(cmake_dir, "dpnp-config.cmake"))
