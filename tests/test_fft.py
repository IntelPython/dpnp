import numpy
import pytest
from numpy.testing import assert_raises

import dpnp

from .helper import assert_dtype_allclose, get_all_dtypes, get_complex_dtypes


class TestFft:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
    @pytest.mark.parametrize(
        "shape", [(64,), (8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)]
    )
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    def test_fft_ndim(self, dtype, shape, norm):
        np_data = numpy.arange(64, dtype=dtype).reshape(shape)
        dpnp_data = dpnp.arange(64, dtype=dtype).reshape(shape)

        np_res = numpy.fft.fft(np_data, norm=norm)
        dpnp_res = dpnp.fft.fft(dpnp_data, norm=norm)
        assert_dtype_allclose(dpnp_res, np_res)

        np_res = numpy.fft.ifft(np_data, norm=norm)
        dpnp_res = dpnp.fft.ifft(dpnp_data, norm=norm)
        assert_dtype_allclose(dpnp_res, np_res)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    def test_fft_1D(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11, dtype=dtype)
        a = dpnp.sin(x)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.fft(a, n=n, norm=norm)
        expected = numpy.fft.fft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected)

        iresult = dpnp.fft.ifft(result, n=n, norm=norm)
        iexpected = numpy.fft.ifft(expected, n=n, norm=norm)
        assert_dtype_allclose(iresult, iexpected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
    def test_fft_1D_complex(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        a = dpnp.asarray(a, dtype=dtype)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.fft(a, n=n, norm=norm)
        expected = numpy.fft.fft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected)

        iresult = dpnp.fft.ifft(result, n=n, norm=norm)
        iexpected = numpy.fft.ifft(expected, n=n, norm=norm)
        assert_dtype_allclose(iresult, iexpected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 1, 0])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft_1D_on_2D_array(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(12, dtype=dtype).reshape(3, 4, order=order)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.fft(a, n=n, axis=axis, norm=norm)
        expected = numpy.fft.fft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

        iresult = dpnp.fft.ifft(result, n=n, axis=axis, norm=norm)
        iexpected = numpy.fft.ifft(expected, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(iresult, iexpected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft_1D_on_3D_array(self, dtype, n, axis, norm, order):
        x1 = numpy.random.uniform(-10, 10, 24)
        x2 = numpy.random.uniform(-10, 10, 24)
        a_np = numpy.array(x1 + 1j * x2, dtype=dtype).reshape(
            2, 3, 4, order=order
        )
        a = dpnp.asarray(a_np)

        result = dpnp.fft.fft(a, n=n, axis=axis, norm=norm)
        expected = numpy.fft.fft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

        iresult = dpnp.fft.ifft(result, n=n, axis=axis, norm=norm)
        iexpected = numpy.fft.ifft(expected, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(iresult, iexpected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_fft_error(self, xp):
        # 0-D input
        a = xp.array(3)
        assert_raises(ValueError, xp.fft.fft, a)

        # n is not int
        a = xp.ones((4, 3))
        if xp == dpnp:
            # dpnp and vanilla NumPy return TypeError
            # Intel® NumPy returns SystemError for Python 3.10 and 3.11
            # and no error for Python 3.9
            assert_raises(TypeError, xp.fft.fft, a, n=5.0)

        # Invalid number of FFT point for incorrect n value
        assert_raises(ValueError, xp.fft.fft, a, n=-5)

        # invalid norm
        assert_raises(ValueError, xp.fft.fft, a, norm="square")

        # Invalid number of FFT point for empty arrays
        a = xp.ones((5, 0, 4))
        assert_raises(ValueError, xp.fft.fft, a, axis=1)


class TestRfft:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "shape", [(64,), (8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)]
    )
    def test_fft_rfft(self, dtype, shape):
        np_data = numpy.arange(64, dtype=dtype).reshape(shape)
        dpnp_data = dpnp.arange(64, dtype=dtype).reshape(shape)

        np_res = numpy.fft.rfft(np_data)
        dpnp_res = dpnp.fft.rfft(dpnp_data)

        assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)

    @pytest.mark.parametrize(
        "func_name",
        [
            "rfft",
        ],
    )
    def test_fft_invalid_dtype(self, func_name):
        a = dpnp.array([True, False, True])
        dpnp_func = getattr(dpnp.fft, func_name)
        with pytest.raises(TypeError):
            dpnp_func(a)
