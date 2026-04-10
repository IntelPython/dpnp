import subprocess
import sys


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
