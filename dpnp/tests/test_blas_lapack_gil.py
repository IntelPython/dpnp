"""Blocking oneMKL calls in the BLAS/LAPACK extensions must release the GIL.

Progress of a competing thread is compared against ``SyclQueue.wait()``, which
is ``nogil`` and therefore the best rate achievable on the machine.
"""

import sys
import threading
import time

import dpctl
import pytest

import dpnp

from .helper import has_support_aspect64, is_gpu_device

# Smaller sizes stop discriminating: the calls either stay asynchronous or
# block too briefly for a stable measurement.
_SIZE = 2048

_MIN_RATIO = 0.10

_BACKLOG = 2  # queued matmuls, so the measured call has something to wait on

_TRIALS = 5  # samples averaged per measurement


class _Ticker:
    """Counts how often a competing Python thread gets scheduled."""

    def __enter__(self):
        self.ticks = 0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop:
            self.ticks += 1
            time.sleep(0)

    def __exit__(self, *exc):
        self._stop = True
        self._thread.join(timeout=5)
        return False


@pytest.mark.skipif(not has_support_aspect64(), reason="requires fp64 support")
class TestBlockingCallsReleaseGil:
    @pytest.fixture(autouse=True)
    def _switch_interval(self):
        # Stop CPython handing the GIL over on its own timer.
        previous = sys.getswitchinterval()
        sys.setswitchinterval(0.001)
        yield
        sys.setswitchinterval(previous)

    @pytest.fixture
    def mats(self):
        spd = dpnp.eye(_SIZE, dtype="f8") * float(_SIZE)
        return spd, dpnp.eye(_SIZE, dtype="f8") + 0.1

    def _assert_releases_gil(self, name, spd, fn):
        queue = spd.sycl_queue

        def rate(ticker, func):
            total_ticks = 0
            total_ms = 0.0
            for _ in range(_TRIALS):
                for _ in range(_BACKLOG):
                    dpnp.matmul(spd, spd)
                ticker.ticks = 0
                start = time.perf_counter()
                func()
                total_ms += 1000 * (time.perf_counter() - start)
                total_ticks += ticker.ticks
                queue.wait()
            return total_ticks / max(total_ms, 1e-3)

        fn()  # warm up JIT
        queue.wait()

        with _Ticker() as ticker:
            measured = rate(ticker, fn)
            reference = rate(ticker, queue.wait)

        assert reference > 0, "reference measurement produced no ticks"
        ratio = measured / reference
        assert ratio >= _MIN_RATIO, (
            f"{name} holds the GIL while blocking: {measured:.2f} ticks/ms vs "
            f"{reference:.2f} for nogil queue.wait() (ratio {ratio:.3f}, need "
            f">= {_MIN_RATIO}). The oneMKL call needs py::gil_scoped_release."
        )

    @pytest.mark.slow
    def test_potrf(self, mats):
        spd, _ = mats
        self._assert_releases_gil(
            "potrf", spd, lambda: dpnp.linalg.cholesky(spd)
        )

    @pytest.mark.slow
    def test_getrf(self, mats):
        spd, gen = mats
        self._assert_releases_gil("getrf", spd, lambda: dpnp.linalg.det(gen))

    @pytest.mark.slow
    def test_syevd(self, mats):
        spd, _ = mats
        self._assert_releases_gil("syevd", spd, lambda: dpnp.linalg.eigh(spd))

    @pytest.mark.slow
    def test_gesv(self, mats):
        spd, gen = mats
        rhs = dpnp.ones(_SIZE, dtype="f8")
        self._assert_releases_gil(
            "gesv", spd, lambda: dpnp.linalg.solve(gen, rhs)
        )


# GPU only: on a CPU device oneMKL already saturates every core, so two threads
# contend for the same hardware and the ratio stays ~1.0 either way.
@pytest.mark.slow
@pytest.mark.skipif(not is_gpu_device(), reason="requires a GPU device")
@pytest.mark.skipif(not has_support_aspect64(), reason="requires fp64 support")
def test_multithreaded_linalg_overlaps():
    """Two threads on two queues must run oneMKL work concurrently."""
    max_ratio = 0.85

    device = dpctl.select_default_device()
    queues = [dpctl.SyclQueue(device, property="in_order") for _ in range(2)]
    mats = [
        dpnp.eye(_SIZE, dtype="f8", sycl_queue=q) * float(_SIZE) for q in queues
    ]

    def work(idx):
        for _ in range(2):
            dpnp.linalg.eigh(mats[idx])
        queues[idx].wait()

    for idx in range(2):  # warm up JIT
        work(idx)

    previous = sys.getswitchinterval()
    sys.setswitchinterval(0.001)
    try:
        start = time.perf_counter()
        work(0)
        work(1)
        serial = time.perf_counter() - start

        start = time.perf_counter()
        threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        parallel = time.perf_counter() - start
    finally:
        sys.setswitchinterval(previous)

    ratio = parallel / serial
    assert ratio <= max_ratio, (
        f"two threads running eigh did not overlap: {parallel * 1000:.1f} ms "
        f"parallel vs {serial * 1000:.1f} ms serial (ratio {ratio:.3f}, need "
        f"<= {max_ratio}). The oneMKL call is holding the GIL."
    )
