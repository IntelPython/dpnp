import functools

import numpy as np
import pytest

import dpnp as cupy
from dpnp.tests.helper import has_support_aspect64, is_cuda_device
from dpnp.tests.third_party.cupy import testing


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


@pytest.mark.usefixtures("skip_forward_backward")
@testing.parameterize(
    *(
        testing.product_dict(
            [
                # some of the following cases are modified, since in NumPy 2.0.0
                # `s` must contain only integer `s`, not None values, and
                # If `s` is not None, `axes` must not be None either.
                {"shape": (3, 4), "s": None, "axes": None},
                {"shape": (3, 4), "s": (1, 4), "axes": (0, 1)},
                {"shape": (3, 4), "s": (1, 5), "axes": (0, 1)},
                {"shape": (3, 4), "s": None, "axes": (-2, -1)},
                {"shape": (3, 4), "s": None, "axes": (-1, -2)},
                # {"shape": (3, 4), "s": None, "axes": (0,)}, # mkl_fft gh-109
                # {"shape": (3, 4), "s": None, "axes": ()}, # mkl_fft gh-108
                {"shape": (2, 3, 4), "s": None, "axes": None},
                {"shape": (2, 3, 4), "s": (1, 4, 4), "axes": (0, 1, 2)},
                {"shape": (2, 3, 4), "s": (1, 4, 10), "axes": (0, 1, 2)},
                {"shape": (2, 3, 4), "s": None, "axes": (-3, -2, -1)},
                {"shape": (2, 3, 4), "s": None, "axes": (-1, -2, -3)},
                # {"shape": (2, 3, 4), "s": None, "axes": (0, 1)}, # mkl_fft gh-109
                # {"shape": (2, 3, 4), "s": None, "axes": ()}, # mkl_fft gh-108
                # {"shape": (2, 3, 4), "s": (2, 3), "axes": (0, 1, 2)}, # mkl_fft gh-109
                {"shape": (2, 3, 4, 5), "s": None, "axes": None},
                # {"shape": (0, 5), "s": None, "axes": None}, # mkl_fft gh-110
                # {"shape": (2, 0, 5), "s": None, "axes": None}, # mkl_fft gh-110
                # {"shape": (0, 0, 5), "s": None, "axes": None}, # mkl_fft gh-110
                {"shape": (3, 4), "s": (0, 5), "axes": (0, 1)},
                {"shape": (3, 4), "s": (1, 0), "axes": (0, 1)},
            ],
            testing.product(
                {"norm": [None, "backward", "ortho", "forward", ""]}
            ),
        )
    )
)
class TestFft2:
    @testing.for_orders("CF")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_fft2(self, xp, dtype, order):
        if is_cuda_device() and self.shape == (2, 3, 4, 5):
            pytest.skip("SAT-7587")
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.fft2(a, s=self.s, axes=self.axes, norm=self.norm)

        if self.axes is not None and not self.axes:
            assert out is a
            return out

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_orders("CF")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_ifft2(self, xp, dtype, order):
        if is_cuda_device() and self.shape == (2, 3, 4, 5):
            pytest.skip("SAT-7587")
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.ifft2(a, s=self.s, axes=self.axes, norm=self.norm)

        if self.axes is not None and not self.axes:
            assert out is a
            return out

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out


@pytest.mark.usefixtures("skip_forward_backward")
@testing.parameterize(
    *(
        testing.product_dict(
            [
                # some of the following cases are modified, since in NumPy 2.0.0
                # `s` must contain only integer `s`, not None values, and
                # If `s` is not None, `axes` must not be None either.
                {"shape": (3, 4), "s": None, "axes": None},
                {"shape": (3, 4), "s": (1, 4), "axes": (0, 1)},
                {"shape": (3, 4), "s": (1, 5), "axes": (0, 1)},
                {"shape": (3, 4), "s": None, "axes": (-2, -1)},
                {"shape": (3, 4), "s": None, "axes": (-1, -2)},
                {"shape": (3, 4), "s": None, "axes": [-1, -2]},
                # {"shape": (3, 4), "s": None, "axes": (0,)}, # mkl_fft gh-109
                # {"shape": (3, 4), "s": None, "axes": ()}, # mkl_fft gh-108
                {"shape": (2, 3, 4), "s": None, "axes": None},
                {"shape": (2, 3, 4), "s": (1, 4, 4), "axes": (0, 1, 2)},
                {"shape": (2, 3, 4), "s": (1, 4, 10), "axes": (0, 1, 2)},
                {"shape": (2, 3, 4), "s": None, "axes": (-3, -2, -1)},
                {"shape": (2, 3, 4), "s": None, "axes": (-1, -2, -3)},
                # {"shape": (2, 3, 4), "s": None, "axes": (-1, -3)}, # mkl_fft gh-109
                # {"shape": (2, 3, 4), "s": None, "axes": (0, 1)}, # mkl_fft gh-109
                # {"shape": (2, 3, 4), "s": None, "axes": ()}, # mkl_fft gh-108
                # {"shape": (2, 3, 4), "s": (2, 3), "axes": (0, 1, 2)}, # mkl_fft gh-109
                {"shape": (2, 3, 4), "s": (4, 3, 2), "axes": (2, 0, 1)},
                {"shape": (2, 3, 4, 5), "s": None, "axes": None},
                # {"shape": (0, 5), "s": None, "axes": None}, # mkl_fft gh-110
                # {"shape": (2, 0, 5), "s": None, "axes": None}, # mkl_fft gh-110
                # {"shape": (0, 0, 5), "s": None, "axes": None}, # mkl_fft gh-110
            ],
            testing.product(
                {"norm": [None, "backward", "ortho", "forward", ""]}
            ),
        )
    )
)
class TestFftn:
    @testing.for_orders("CF")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_fftn(self, xp, dtype, order):
        if is_cuda_device() and self.shape == (2, 3, 4, 5):
            pytest.skip("SAT-7587")
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.fftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if self.axes is not None and not self.axes:
            assert out is a
            return out

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_orders("CF")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_ifftn(self, xp, dtype, order):
        if is_cuda_device() and self.shape == (2, 3, 4, 5):
            pytest.skip("SAT-7587")
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.ifftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if self.axes is not None and not self.axes:
            assert out is a
            return out

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

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


@pytest.mark.usefixtures("skip_forward_backward")
@testing.parameterize(
    *(
        testing.product_dict(
            [
                # some of the following cases are modified, since in NumPy 2.0.0
                # `s` must contain only integer `s`, not None values, and
                # If `s` is not None, `axes` must not be None either.
                {"shape": (3, 4), "s": None, "axes": None},
                {"shape": (3, 4), "s": (1, 4), "axes": (0, 1)},
                {"shape": (3, 4), "s": (1, 5), "axes": (0, 1)},
                {"shape": (3, 4), "s": None, "axes": (-2, -1)},
                {"shape": (3, 4), "s": None, "axes": (-1, -2)},
                {"shape": (3, 4), "s": None, "axes": (0,)},
                # {"shape": (2, 3, 4), "s": None, "axes": None}, # mkl_fft gh-116
                # {"shape": (2, 3, 4), "s": (1, 4, 4), "axes": (0, 1, 2)}, # mkl_fft gh-115
                # {"shape": (2, 3, 4), "s": (1, 4, 10), "axes": (0, 1, 2)}, # mkl_fft gh-115
                # {"shape": (2, 3, 4), "s": None, "axes": (-3, -2, -1)}, # mkl_fft gh-116
                # {"shape": (2, 3, 4), "s": None, "axes": (-1, -2, -3)}, # mkl_fft gh-116
                {"shape": (2, 3, 4), "s": None, "axes": (0, 1)},
                {"shape": (2, 3, 4), "s": (2, 3), "axes": (0, 1, 2)},
                # {"shape": (2, 3, 4, 5), "s": None, "axes": None}, # mkl_fft gh-109 and gh-116
            ],
            testing.product(
                {"norm": [None, "backward", "ortho", "forward", ""]}
            ),
        )
    )
)
class TestRfft2:
    @testing.for_orders("CF")
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_rfft2(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.rfft2(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_orders("CF")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_irfft2(self, xp, dtype, order):
        if self.s is None and self.axes in [None, (-2, -1)]:
            pytest.skip("Input is not Hermitian Symmetric")
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.irfft2(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@testing.parameterize(
    {"shape": (3, 4), "s": None, "axes": (), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (), "norm": None},
)
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


@pytest.mark.usefixtures("skip_forward_backward")
@testing.parameterize(
    *(
        testing.product_dict(
            [
                # some of the following cases are modified, since in NumPy 2.0.0
                # `s` must contain only integer `s`, not None values, and
                # If `s` is not None, `axes` must not be None either.
                {"shape": (3, 4), "s": None, "axes": None},
                {"shape": (3, 4), "s": (1, 4), "axes": (0, 1)},
                {"shape": (3, 4), "s": (1, 5), "axes": (0, 1)},
                {"shape": (3, 4), "s": None, "axes": (-2, -1)},
                {"shape": (3, 4), "s": None, "axes": (-1, -2)},
                {"shape": (3, 4), "s": None, "axes": (0,)},
                # {"shape": (2, 3, 4), "s": None, "axes": None}, # mkl_fft gh-116
                # {"shape": (2, 3, 4), "s": (1, 4, 4), "axes": (0, 1, 2)}, # mkl_fft gh-115
                # {"shape": (2, 3, 4), "s": (1, 4, 10), "axes": (0, 1, 2)}, # mkl_fft gh-115
                # {"shape": (2, 3, 4), "s": None, "axes": (-3, -2, -1)}, # mkl_fft gh-116
                # {"shape": (2, 3, 4), "s": None, "axes": (-1, -2, -3)}, # mkl_fft gh-116
                {"shape": (2, 3, 4), "s": None, "axes": (0, 1)},
                {"shape": (2, 3, 4), "s": (2, 3), "axes": (0, 1, 2)},
                # {"shape": (2, 3, 4, 5), "s": None, "axes": None}, # mkl_fft gh-109 and gh-116
            ],
            testing.product(
                {"norm": [None, "backward", "ortho", "forward", ""]}
            ),
        )
    )
)
class TestRfftn:
    @testing.for_orders("CF")
    @testing.for_all_dtypes(no_complex=True)
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_rfftn(self, xp, dtype, order):
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.rfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.complex64)

        return out

    @testing.for_orders("CF")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4,
        atol=1e-7,
        accept_error=ValueError,
        contiguous_check=False,
        type_check=has_support_aspect64(),
    )
    def test_irfftn(self, xp, dtype, order):
        if self.s is None and self.axes in [None, (-2, -1)]:
            pytest.skip("Input is not Hermitian Symmetric")
        a = testing.shaped_random(self.shape, xp, dtype)
        if order == "F":
            a = xp.asfortranarray(a)
        out = xp.fft.irfftn(a, s=self.s, axes=self.axes, norm=self.norm)

        if xp is np and dtype in [np.float16, np.float32, np.complex64]:
            out = out.astype(np.float32)

        return out


@testing.parameterize(
    {"shape": (3, 4), "s": None, "axes": (), "norm": None},
    {"shape": (2, 3, 4), "s": None, "axes": (), "norm": None},
)
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
