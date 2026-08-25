"""Blocking oneMKL calls in the FFT extension must release the GIL.

Progress of a competing Python thread is measured while the FFT call blocks,
and compared against ``time.sleep()`` of the same duration, which is ``nogil``
and therefore the best rate achievable on the machine.
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

# A call that holds the GIL measures around 0.09 of the reference rate, one
# that releases it around 0.3. The threshold sits between the two.
_MIN_RATIO = 0.18

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
        sys.setswitchinterval(0.0005)
        yield
        sys.setswitchinterval(previous)

    def _assert_releases_gil(self, name, a, fn):
        queue = a.sycl_queue

        def measure(ticker, func):
            total_ticks = 0
            total_s = 0.0
            for _ in range(_TRIALS):
                for _ in range(_BACKLOG):
                    dpnp.fft.fft(a)
                ticker.ticks = 0
                start = time.perf_counter()
                func()
                total_s += time.perf_counter() - start
                total_ticks += ticker.ticks
                queue.wait()
            # ticks per millisecond, and the average duration of a single call
            return total_ticks / max(1000 * total_s, 1e-3), total_s / _TRIALS

        fn()  # warm up JIT
        queue.wait()

        with _Ticker() as ticker:
            measured, duration = measure(ticker, fn)
            # time.sleep() is nogil, so blocking in it for the same amount of
            # time gives the best tick rate the machine can produce
            reference, _ = measure(ticker, lambda: time.sleep(duration))

        assert reference > 0, "reference measurement produced no ticks"
        ratio = measured / reference
        assert ratio >= _MIN_RATIO, (
            f"{name} holds the GIL while blocking: {measured:.2f} ticks/ms vs "
            f"{reference:.2f} for a nogil sleep of the same duration (ratio "
            f"{ratio:.3f}, need >= {_MIN_RATIO}). The oneMKL call needs "
            f"py::gil_scoped_release."
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
