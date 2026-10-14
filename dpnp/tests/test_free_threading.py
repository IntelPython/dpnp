import os
import subprocess
import sys
import sysconfig

import pytest

# Prints the first dpnp module whose import re-enables the GIL,
# "deps" if a dependency already re-enabled it, or "ok".
_IMPORT_ALL = """
import importlib, importlib.machinery, pathlib, sys
import dpctl, numpy
if sys._is_gil_enabled():
    print("deps")
    sys.exit()
import dpnp
if sys._is_gil_enabled():
    print("dpnp")
    sys.exit()
root = pathlib.Path(dpnp.__file__).parent
suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
for path in sorted(root.rglob("*" + suffix)):
    rel = path.relative_to(root.parent)
    name = ".".join(rel.parts[:-1] + (path.name[: -len(suffix)],))
    importlib.import_module(name)
    if sys._is_gil_enabled():
        print(name)
        sys.exit()
print("ok")
"""


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="requires a free-threaded Python build",
)
def test_extensions_keep_gil_disabled():
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    res = subprocess.run(
        [sys.executable, "-c", _IMPORT_ALL],
        capture_output=True,
        text=True,
        env=env,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.strip()
    if out == "deps":
        pytest.skip("a dependency re-enabled the GIL")
    assert out == "ok", f"importing {out} re-enabled the GIL"
