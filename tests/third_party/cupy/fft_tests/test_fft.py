import functools
import unittest

import numpy as np
import pytest

import dpnp as cupy
from tests.helper import has_support_aspect64
from tests.third_party.cupy import testing


@pytest.fixture
def skip_forward_backward(request):
    if request.instance.norm in ("backward", "forward"):
        if not (np.lib.NumpyVersion(np.__version__) >= "1.20.0"):
            pytest.skip("forward/backward is supported by NumPy 1.20+")


@pytest.mark.usefixtures("skip_forward_backward")
@testing.parameterize(
    *testing.product(
        {
            "n": [None, 0, 5, 10, 15],
            "shape": [(0,), (10, 0), (10,), (10, 10)],
            "norm": [None, "backward", "ortho", "forward", ""],
        }
    )
)
class TestFft:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft(a, n=self.n, norm=self.norm)

        # np.fft.fft always returns np.complex128
        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    # NumPy 1.17.0 and 1.17.1 raises ZeroDivisonError due to a bug
    @testing.with_requires("numpy!=1.17.0")
    @testing.with_requires("numpy!=1.17.1")
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    *testing.product(
        {
            "shape": [(0, 10), (10, 0, 10), (10, 10), (10, 5, 10)],
            "data_order": ["F", "C"],
            "axis": [0, 1, -1],
        }
    )
)
class TestFftOrder:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-6,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_fft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.data_order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.fft(a, axis=self.axis)

        # np.fft.fft always returns np.complex128
        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_ifft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        if self.data_order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.ifft(a, axis=self.axis)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {"shape": (3, 4), "s": None, "axes": None, "norm": None},
    {"shape": (3, 4), "s": (1, None), "axes": None, "norm": None},
    {"shape": (3, 4), "s": (1, 5), "axes": None, "norm": None},
    {"shape": (3, 4), "s": None, "axes": (-2, -1), "norm": None},
    {"shape": (3, 4), "s": None, "axes": (-1, -2), "norm": None},
    {"shape": (3, 4), "s": None, "axes": (0,), "norm": None},
    {"shape": (3, 4), "s": None, "axes": None, "norm": "ortho"},
    {"shape": (3, 4), "s": None, "axes": (), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": None, "norm": None},
    {"shape": (2, 3, 4), "s": (1, 4, None), "axes": None, "norm": None},
    {"shape": (2, 3, 4), "s": (1, 4, 10), "axes": None, "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (-3, -2, -1), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (-1, -2, -3), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (0, 1), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": None, "norm": "ortho"},
    {"shape": (2, 3, 4), "s": None, "axes": (), "norm": None},
    {"shape": (2, 3, 4), "s": (2, 3), "axes": (0, 1, 2), "norm": "ortho"},
    {"shape": (2, 3, 4, 5), "s": None, "axes": None, "norm": None},
    {"shape": (0, 5), "s": None, "axes": None, "norm": None},
    {"shape": (2, 0, 5), "s": None, "axes": None, "norm": None},
    {"shape": (0, 0, 5), "s": None, "axes": None, "norm": None},
    {"shape": (3, 4), "s": (0, 5), "axes": None, "norm": None},
    {"shape": (3, 4), "s": (1, 0), "axes": None, "norm": None},
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestFft2(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=False,
    )
    def test_fft2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fft2(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=False,
    )
    def test_ifft2(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifft2(a, s=self.s, axes=self.axes, norm=self.norm)

        return out


@testing.parameterize(
    {"shape": (3, 4), "s": None, "axes": None, "norm": None},
    {"shape": (3, 4), "s": (1, None), "axes": None, "norm": None},
    {"shape": (3, 4), "s": (1, 5), "axes": None, "norm": None},
    {"shape": (3, 4), "s": None, "axes": (-2, -1), "norm": None},
    {"shape": (3, 4), "s": None, "axes": (-1, -2), "norm": None},
    {"shape": (3, 4), "s": None, "axes": [-1, -2], "norm": None},
    {"shape": (3, 4), "s": None, "axes": (0,), "norm": None},
    {"shape": (3, 4), "s": None, "axes": (), "norm": None},
    {"shape": (3, 4), "s": None, "axes": None, "norm": "ortho"},
    {"shape": (2, 3, 4), "s": None, "axes": None, "norm": None},
    {"shape": (2, 3, 4), "s": (1, 4, None), "axes": None, "norm": None},
    {"shape": (2, 3, 4), "s": (1, 4, 10), "axes": None, "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (-3, -2, -1), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (-1, -2, -3), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (-1, -3), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (0, 1), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": None, "norm": "ortho"},
    {"shape": (2, 3, 4), "s": None, "axes": (), "norm": "ortho"},
    {"shape": (2, 3, 4), "s": (2, 3), "axes": (0, 1, 2), "norm": "ortho"},
    {"shape": (2, 3, 4), "s": (4, 3, 2), "axes": (2, 0, 1), "norm": "ortho"},
    {"shape": (2, 3, 4, 5), "s": None, "axes": None, "norm": None},
    {"shape": (0, 5), "s": None, "axes": None, "norm": None},
    {"shape": (2, 0, 5), "s": None, "axes": None, "norm": None},
    {"shape": (0, 0, 5), "s": None, "axes": None, "norm": None},
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestFftn(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=False,
    )
    def test_fftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=False,
    )
    def test_ifftn(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)

        return out


@pytest.mark.usefixtures("skip_forward_backward")
@testing.parameterize(
    *testing.product(
        {
            "n": [None, 5, 10, 15],
            "shape": [(10,), (10, 10)],
            "norm": [None, "backward", "ortho", "forward", ""],
        }
    )
)
class TestRfft:
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_rfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.rfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=2e-6,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_irfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.irfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@testing.parameterize(
    {"shape": (3, 4), "s": None, "axes": (), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (), "norm": None},
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestRfft2EmptyAxes:
    @testing.for_all_dtypes(no_complex=True)
    def test_rfft2(self, dtype):
        for xp in (np, cupy):
            a = testing.shaped_random(self.shape, xp, dtype)
            with pytest.raises(IndexError):
                xp.fft.rfft2(a, s=self.s, axes=self.axes, norm=self.norm)

    @testing.for_all_dtypes()
    def test_irfft2(self, dtype):
        for xp in (np, cupy):
            a = testing.shaped_random(self.shape, xp, dtype)
            with pytest.raises(IndexError):
                xp.fft.irfft2(a, s=self.s, axes=self.axes, norm=self.norm)


@testing.parameterize(
    {"shape": (3, 4), "s": None, "axes": (), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (), "norm": None},
)
@pytest.mark.usefixtures("allow_fall_back_on_numpy")
class TestRfftnEmptyAxes:
    @testing.for_all_dtypes(no_complex=True)
    def test_rfftn(self, dtype):
        for xp in (np, cupy):
            a = testing.shaped_random(self.shape, xp, dtype)
            with pytest.raises(IndexError):
                xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)

    @testing.for_all_dtypes()
    def test_irfftn(self, dtype):
        for xp in (np, cupy):
            a = testing.shaped_random(self.shape, xp, dtype)
            with pytest.raises(IndexError):
                xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)


@pytest.mark.usefixtures("skip_forward_backward")
@testing.parameterize(
    *testing.product(
        {
            "n": [None, 5, 10, 15],
            "shape": [(10,), (10, 10)],
            "norm": [None, "backward", "ortho", "forward", ""],
        }
    )
)
class TestHfft:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=2e-6,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_hfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.hfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out

    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_ihfft(self, xp, dtype):
        a = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ihfft(a, n=self.n, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@testing.parameterize(
    {"n": 1, "d": 1},
    {"n": 10, "d": 0.5},
    {"n": 100, "d": 2},
)
class TestFftfreq:
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        type_check=has_support_aspect64(),
    )
    def test_fftfreq(self, xp):
        out = xp.fft.fftfreq(self.n, self.d)

        return out

    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        type_check=has_support_aspect64(),
    )
    def test_rfftfreq(self, xp):
        out = xp.fft.rfftfreq(self.n, self.d)

        return out


@testing.parameterize(
    {"shape": (5,), "axes": None},
    {"shape": (5,), "axes": 0},
    {"shape": (10,), "axes": None},
    {"shape": (10,), "axes": 0},
    {"shape": (10, 10), "axes": None},
    {"shape": (10, 10), "axes": 0},
    {"shape": (10, 10), "axes": (0, 1)},
)
class TestFftshift:
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        type_check=has_support_aspect64(),
    )
    def test_fftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.fftshift(x, self.axes)

        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        type_check=has_support_aspect64(),
    )
    def test_ifftshift(self, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        out = xp.fft.ifftshift(x, self.axes)

        return out
