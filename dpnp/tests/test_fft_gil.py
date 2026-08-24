"""Blocking oneMKL calls in the FFT extension must release the GIL.

Progress of a competing thread is compared against ``SyclQueue.wait()``, which
is ``nogil`` and therefore the best rate achievable on the machine.
"""

import sys
import threading
import time

import pytest

import dpnp

from .helper import has_support_aspect64

# Smaller sizes stop discriminating: the calls either stay asynchronous or
# block too briefly for a stable measurement.
_BATCH = 512
_SIZE = 4096

_MIN_RATIO = 0.10

_BACKLOG = 2  # queued transforms, so the measured call has something to wait on

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
class TestFftReleasesGil:
    @pytest.fixture(autouse=True)
    def _switch_interval(self):
        # Stop CPython handing the GIL over on its own timer.
        previous = sys.getswitchinterval()
        sys.setswitchinterval(0.001)
        yield
        sys.setswitchinterval(previous)

    def _assert_releases_gil(self, name, a, fn):
        queue = a.sycl_queue

        def rate(ticker, func):
            total_ticks = 0
            total_ms = 0.0
            for _ in range(_TRIALS):
                for _ in range(_BACKLOG):
                    dpnp.fft.fft(a)
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
    def test_fft_out_of_place(self):
        # a complex input is passed to oneMKL as is, so the transform is
        # computed out-of-place
        a = dpnp.ones((_BATCH, _SIZE), dtype="c16")
        self._assert_releases_gil(
            "compute_fft_out_of_place", a, lambda: dpnp.fft.fft(a)
        )

    @pytest.mark.slow
    def test_fft_in_place(self):
        # a real input is copied to a complex array first, which allows the
        # transform to be computed in-place
        a = dpnp.ones((_BATCH, _SIZE), dtype="f8")
        self._assert_releases_gil(
            "compute_fft_in_place", a, lambda: dpnp.fft.fft(a)
        )
