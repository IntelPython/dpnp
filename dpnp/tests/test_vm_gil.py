"""oneMKL VM calls in the VM extension must release the GIL.

On a CPU device oneMKL VM waits on the dependent events and computes on the
calling thread. On an in-order queue those events are ordered after the host
tasks dpnp submits to keep arguments alive, and these host tasks acquire the
GIL, so a VM call holding the GIL never returns.
"""

import subprocess
import sys

import dpctl
import pytest

import dpnp.libtensor._vm_impl as vmi

# Prints "novm" if oneMKL VM is not used for the subtraction, or "ok".
_VM_AFTER_HOST_TASKS = """
import dpctl, dpnp
import dpnp.libtensor._vm_impl as vmi
q = dpctl.SyclQueue("cpu", property="in_order")
xp = dpnp.linspace(10.0, 0.0, 5, sycl_queue=q)
a, b = xp[-1:], xp[:1]
out = dpnp.empty_like(a)
if not vmi._mkl_sub_to_call(q, a.get_array(), b.get_array(), out.get_array()):
    print("novm")
    raise SystemExit
for _ in range(100):
    xs = xp[dpnp.argsort(xp)]  # submits host tasks keeping arguments alive
    xs[-1:] - xs[:1]
q.wait()
print("ok")
"""

_TIMEOUT = 120  # seconds, a run takes about one


def _cpu_device_with_fp64():
    for dev in dpctl.get_devices(device_type="cpu"):
        if dev.has_aspect_fp64:
            return dev
    return None


@pytest.mark.skipif(not vmi._is_available(), reason="requires oneMKL VM")
@pytest.mark.skipif(
    _cpu_device_with_fp64() is None,
    reason="requires a CPU device with fp64 support",
)
def test_in_order_queue_after_host_tasks():
    try:
        res = subprocess.run(
            [sys.executable, "-c", _VM_AFTER_HOST_TASKS],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"no progress in {_TIMEOUT}s: a oneMKL VM call holds the GIL "
            "while waiting on events behind host tasks. It needs "
            "py::gil_scoped_release."
        )
    assert res.returncode == 0, res.stderr
    out = res.stdout.strip()
    if out == "novm":
        pytest.skip("oneMKL VM is not used for the subtraction")
    assert out == "ok", out
