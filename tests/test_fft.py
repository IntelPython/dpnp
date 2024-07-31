import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_raises

import dpnp
from dpnp.dpnp_utils import map_dtype_to_device

from .helper import (
    assert_dtype_allclose,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_dtypes,
)


# TODO: `assert_dtype_allclose` calls in this file have `check_only_type_kind=True`
# since stock NumPy is currently used in public CI for code coverege which
# always returns complex128/float64 for FFT functions, but Intel® NumPy and
# dpnp return complex64/float32 if input is complex64/float32
class TestFft:
    def setup_method(self):
        numpy.random.seed(42)

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
        assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)

        np_res = numpy.fft.ifft(np_data, norm=norm)
        dpnp_res = dpnp.fft.ifft(dpnp_data, norm=norm)
        assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    def test_fft_1D(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11, dtype=dtype)
        a = dpnp.sin(x)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.fft(a, n=n, norm=norm)
        expected = numpy.fft.fft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifft(result, n=n, norm=norm)
        iexpected = numpy.fft.ifft(expected, n=n, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

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
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifft(result, n=n, norm=norm)
        iexpected = numpy.fft.ifft(expected, n=n, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

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
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifft(result, n=n, axis=axis, norm=norm)
        iexpected = numpy.fft.ifft(expected, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

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
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifft(result, n=n, axis=axis, norm=norm)
        iexpected = numpy.fft.ifft(expected, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("n", [None, 5, 20])
    def test_fft_usm_ndarray(self, n):
        x = dpt.linspace(-1, 1, 11)
        a = dpt.sin(x) + 1j * dpt.cos(x)
        a_usm = dpt.asarray(a, dtype=dpt.complex64)
        a_np = dpt.asnumpy(a_usm)
        out_shape = (n,) if n is not None else a_usm.shape
        out = dpt.empty(out_shape, dtype=a_usm.dtype)

        result = dpnp.fft.fft(a_usm, n=n, out=out)
        assert out is result.get_array()
        expected = numpy.fft.fft(a_np, n=n)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        # in-place
        if n is None:
            result = dpnp.fft.fft(a_usm, n=n, out=a_usm)
            assert a_usm is result.get_array()
            assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
    def test_fft_1D_out(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        a = dpnp.asarray(a, dtype=dtype)
        a_np = dpnp.asnumpy(a)
        out_shape = (n,) if n is not None else a.shape
        out = dpnp.empty(out_shape, dtype=a.dtype)

        result = dpnp.fft.fft(a, n=n, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.fft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifft(result, n=n, norm=norm, out=out)
        assert out is iresult
        iexpected = numpy.fft.ifft(expected, n=n, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_fft_inplace_out(self, axis):
        # Test some weirder in-place combinations
        y = dpnp.random.rand(20, 20) + 1j * dpnp.random.rand(20, 20)
        y_np = y.asnumpy()
        # Fully in-place.
        y1 = y.copy()
        expected1 = numpy.fft.fft(y1.asnumpy(), axis=axis)
        result1 = dpnp.fft.fft(y1, axis=axis, out=y1)
        assert result1 is y1
        assert_dtype_allclose(result1, expected1)

        # In-place of part of the array; rest should be unchanged.
        y2 = y.copy()
        out2 = y2[:10] if axis == 0 else y2[:, :10]
        expected2 = numpy.fft.fft(y2.asnumpy(), n=10, axis=axis)
        result2 = dpnp.fft.fft(y2, n=10, axis=axis, out=out2)
        assert result2 is out2
        assert_dtype_allclose(out2, expected2)
        assert_dtype_allclose(result2, expected2)
        if axis == 0:
            assert_dtype_allclose(y2[10:], y_np[10:])
        else:
            assert_dtype_allclose(y2[:, 10:], y_np[:, 10:])

        # In-place of another part of the array.
        y3 = y.copy()
        y3_sel = y3[5:] if axis == 0 else y3[:, 5:]
        out3 = y3[5:15] if axis == 0 else y3[:, 5:15]
        expected3 = numpy.fft.fft(y3_sel.asnumpy(), n=10, axis=axis)
        result3 = dpnp.fft.fft(y3_sel, n=10, axis=axis, out=out3)
        assert result3 is out3
        assert_dtype_allclose(result3, expected3)
        if axis == 0:
            assert_dtype_allclose(y3[:5], y_np[:5])
            assert_dtype_allclose(y3[15:], y_np[15:])
        else:
            assert_dtype_allclose(y3[:, :5], y_np[:, :5])
            assert_dtype_allclose(y3[:, 15:], y_np[:, 15:])

        # In-place with n > nin; rest should be unchanged.
        # for this case, out-of-place FFT is called with a temporary
        # buffer for output array in FFT call
        y4 = y.copy()
        y4_sel = y4[:10] if axis == 0 else y4[:, :10]
        out4 = y4[:15] if axis == 0 else y4[:, :15]
        expected4 = numpy.fft.fft(y4_sel.asnumpy(), n=15, axis=axis)
        result4 = dpnp.fft.fft(y4_sel, n=15, axis=axis, out=out4)
        assert result4 is out4
        assert_dtype_allclose(result4, expected4)
        if axis == 0:
            assert_dtype_allclose(y4[15:], y_np[15:])
        else:
            assert_dtype_allclose(y4[:, 15:], y_np[:, 15:])

        # Overwrite in a transpose.
        # for this case, out-of-place FFT is called with a temporary
        # buffer for output array in FFT call
        y5 = y.copy()
        out5 = y5.T
        result5 = dpnp.fft.fft(y5, axis=axis, out=out5)
        assert result5 is out5
        assert_dtype_allclose(y5, expected1.T)
        assert_dtype_allclose(result5, expected1)

        # Reverse strides.
        # for this case, out-of-place FFT is called with a temporary
        # buffer for output array in FFT call
        y6 = y.copy()
        out6 = y6[::-1] if axis == 0 else y6[:, ::-1]
        result6 = dpnp.fft.fft(y6, axis=axis, out=out6)
        assert result6 is out6
        assert_dtype_allclose(result6, expected1)
        if axis == 0:
            assert_dtype_allclose(y6, expected1[::-1])
        else:
            assert_dtype_allclose(y6, expected1[:, ::-1])

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 0])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft_1D_on_2D_array_out(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(12, dtype=dtype).reshape(3, 4, order=order)
        a = dpnp.asarray(a_np)
        out_shape = list(a.shape)
        if n is not None:
            out_shape[axis] = n
        out_shape = tuple(out_shape)
        out = dpnp.empty(out_shape, dtype=a.dtype)

        result = dpnp.fft.fft(a, n=n, axis=axis, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.fft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifft(result, n=n, axis=axis, norm=norm, out=out)
        assert out is iresult
        iexpected = numpy.fft.ifft(expected, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("stride", [-1, -3, 2, 5])
    def test_fft_strided_1D(self, stride):
        x1 = numpy.random.uniform(-10, 10, 20)
        x2 = numpy.random.uniform(-10, 10, 20)
        A_np = numpy.array(x1 + 1j * x2, dtype=numpy.complex64)
        A = dpnp.asarray(A_np)
        a_np = A_np[::stride]
        a = A[::stride]

        result = dpnp.fft.fft(a)
        expected = numpy.fft.fft(a_np)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("stride_x", [-1, -3, 2, 3])
    @pytest.mark.parametrize("stride_y", [-1, -3, 2, 3])
    def test_fft_strided_2D(self, stride_x, stride_y):
        x1 = numpy.random.uniform(-10, 10, 120)
        x2 = numpy.random.uniform(-10, 10, 120)
        a_np = numpy.array(x1 + 1j * x2, dtype=numpy.complex64).reshape(12, 10)
        a = dpnp.asarray(a_np)
        a_np = a_np[::stride_x, ::stride_y]
        a = a[::stride_x, ::stride_y]

        result = dpnp.fft.fft(a)
        expected = numpy.fft.fft(a_np)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    def test_fft_empty_array(self):
        a_np = numpy.empty((10, 0, 4), dtype=numpy.complex64)
        a = dpnp.array(a_np)

        result = dpnp.fft.fft(a, axis=0)
        expected = numpy.fft.fft(a_np, axis=0)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        result = dpnp.fft.fft(a, axis=1, n=2)
        expected = numpy.fft.fft(a_np, axis=1, n=2)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_fft_error(self, xp):
        # 0-D input
        a = xp.array(3)
        # dpnp and Intel® NumPy return ValueError
        # stock NumPy returns IndexError
        assert_raises((ValueError, IndexError), xp.fft.fft, a)

        # n is not int
        a = xp.ones((4, 3))
        if xp == dpnp:
            # dpnp and stock NumPy return TypeError
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

    def test_fft_validate_out(self):
        # Inconsistent sycl_queue
        a = dpnp.ones((10,), dtype=dpnp.complex64, sycl_queue=dpctl.SyclQueue())
        out = dpnp.empty((10,), sycl_queue=dpctl.SyclQueue())
        assert_raises(ExecutionPlacementError, dpnp.fft.fft, a, out=out)

        # Invalid shape
        a = dpnp.ones((10,), dtype=dpnp.complex64)
        out = dpnp.empty((11,), dtype=dpnp.complex64)
        assert_raises(ValueError, dpnp.fft.fft, a, out=out)

        # Invalid dtype for c2c or r2c FFT
        a = dpnp.ones((10,), dtype=dpnp.complex64)
        out = dpnp.empty((10,), dtype=dpnp.float32)
        assert_raises(TypeError, dpnp.fft.fft, a, out=out)


class TestFftfreq:
    @pytest.mark.parametrize("func", ["fftfreq", "rfftfreq"])
    @pytest.mark.parametrize("n", [10, 20])
    @pytest.mark.parametrize("d", [0.5, 2])
    def test_fftfreq(self, func, n, d):
        expected = getattr(dpnp.fft, func)(n, d)
        result = getattr(numpy.fft, func)(n, d)
        assert_dtype_allclose(expected, result)

    @pytest.mark.parametrize("func", ["fftfreq", "rfftfreq"])
    def test_error(self, func):
        # n should be an integer
        assert_raises(ValueError, getattr(dpnp.fft, func), 10.0)

        # d should be an scalar
        assert_raises(ValueError, getattr(dpnp.fft, func), 10, (2,))


class TestFftshift:
    @pytest.mark.parametrize("func", ["fftshift", "ifftshift"])
    @pytest.mark.parametrize("axes", [None, 1, (0, 1)])
    def test_fftshift(self, func, axes):
        x = dpnp.arange(12).reshape(3, 4)
        x_np = x.asnumpy()
        expected = getattr(dpnp.fft, func)(x, axes=axes)
        result = getattr(numpy.fft, func)(x_np, axes=axes)
        assert_dtype_allclose(expected, result)


class TestHfft:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    def test_hfft_1D(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11, dtype=dtype)
        a = dpnp.sin(x)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.hfft(a, n=n, norm=norm)
        expected = numpy.fft.hfft(a_np, n=n, norm=norm)
        # check_only_type_kind=True since numpy always returns float64
        # but dpnp return float32 if input is float32
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 18])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
    def test_hfft_1D_complex(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        # input should be Hermitian
        a[0].imag = 0
        if n in [None, 18]:
            f_ny = -1 if n is None else n // 2  # Nyquist mode
            a[f_ny].imag = 0
            a[f_ny:] = 0  # no data needed after Nyquist mode
        a = dpnp.asarray(a, dtype=dtype)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.hfft(a, n=n, norm=norm)
        expected = numpy.fft.hfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    def test_ihfft_1D(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11, dtype=dtype)
        a = dpnp.sin(x)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.ihfft(a, n=n, norm=norm)
        expected = numpy.fft.ihfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    def test_ihfft_bool(self, n, norm):
        a = dpnp.ones(11, dtype=dpnp.bool)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.ihfft(a, n=n, norm=norm)
        expected = numpy.fft.ihfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)


class TestIrfft:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    def test_fft_1D(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11, dtype=dtype)
        a = dpnp.sin(x)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.irfft(a, n=n, norm=norm)
        expected = numpy.fft.irfft(a_np, n=n, norm=norm)
        # check_only_type_kind=True since Intel® NumPy always returns float64
        # but dpnp return float32 if input is float32
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 18])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
    def test_fft_1D_complex(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        # input should be Hermitian
        a[0].imag = 0
        if n in [None, 18]:
            f_ny = -1 if n is None else n // 2  # Nyquist mode
            a[f_ny].imag = 0
            a[f_ny:] = 0  # no data needed after Nyquist mode
        a = dpnp.asarray(a, dtype=dtype)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.irfft(a, n=n, norm=norm)
        expected = numpy.fft.irfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 1, 0])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft_1D_on_2D_array(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(12, dtype=dtype).reshape(3, 4, order=order)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.irfft(a, n=n, axis=axis, norm=norm)
        expected = numpy.fft.irfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft_1D_on_3D_array(self, dtype, n, axis, norm, order):
        x1 = numpy.random.uniform(-10, 10, 120)
        x2 = numpy.random.uniform(-10, 10, 120)
        a_np = numpy.array(x1 + 1j * x2, dtype=dtype).reshape(
            4, 5, 6, order=order
        )
        # each 1-D array of input should be Hermitian
        if axis == 0:
            a_np[0].imag = 0
            if n is None:
                # for axis=0 and n=8, Nyquist mode is not present
                f_ny = -1  # Nyquist mode
                a_np[-1].imag = 0
        elif axis == 1:
            a_np[:, 0, :].imag = 0
            if n in [None, 8]:
                f_ny = -1  # Nyquist mode
                a_np[:, f_ny, :].imag = 0
                a_np[:, f_ny:, :] = 0  # no data needed after Nyquist mode
        elif axis == 2:
            a_np[..., 0].imag = 0
            if n in [None, 8]:
                f_ny = -1 if n is None else n // 2  # Nyquist mode
                a_np[..., f_ny].imag = 0
                a_np[..., f_ny:] = 0  # no data needed after Nyquist mode

        a = dpnp.asarray(a_np)

        result = dpnp.fft.irfft(a, n=n, axis=axis, norm=norm)
        expected = numpy.fft.irfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(
            result, expected, check_only_type_kind=True, factor=16
        )

    @pytest.mark.parametrize("n", [None, 5, 18])
    def test_fft_usm_ndarray(self, n):
        x = dpt.linspace(-1, 1, 11)
        a = dpt.sin(x) + 1j * dpt.cos(x)
        # input should be Hermitian
        a[0] = dpt.sin(x[0])
        if n in [None, 18]:
            f_ny = -1 if n is None else n // 2  # Nyquist mode
            a[f_ny] = dpt.sin(x[f_ny])
            a[f_ny:] = 0  # no data needed after Nyquist mode
        a_usm = dpt.asarray(a, dtype=dpt.complex64)
        a_np = dpt.asnumpy(a_usm)
        out_shape = n if n is not None else 2 * (a_usm.shape[0] - 1)
        out = dpt.empty(out_shape, dtype=a_usm.real.dtype)

        result = dpnp.fft.irfft(a_usm, n=n, out=out)
        assert out is result.get_array()
        expected = numpy.fft.irfft(a_np, n=n)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 18])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
    def test_fft_1D_out(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        # input should be Hermitian
        a[0].imag = 0
        if n in [None, 18]:
            f_ny = -1 if n is None else n // 2  # Nyquist mode
            a[f_ny].imag = 0
            a[f_ny:] = 0  # no data needed after Nyquist mode
        a = dpnp.asarray(a, dtype=dtype)
        a_np = dpnp.asnumpy(a)

        out_shape = n if n is not None else 2 * (a.shape[0] - 1)
        out = dpnp.empty(out_shape, dtype=a.real.dtype)

        result = dpnp.fft.irfft(a, n=n, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.irfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 0])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft_1D_on_2D_array_out(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(12, dtype=dtype).reshape(3, 4, order=order)
        a = dpnp.asarray(a_np)

        out_shape = list(a.shape)
        out_shape[axis] = 2 * (a.shape[axis] - 1) if n is None else n
        out_shape = tuple(out_shape)
        out = dpnp.empty(out_shape, dtype=a.real.dtype)

        result = dpnp.fft.irfft(a, n=n, axis=axis, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.irfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    def test_fft_validate_out(self):
        # Invalid dtype for c2r FFT
        a = dpnp.ones((10,), dtype=dpnp.complex64)
        out = dpnp.empty((18,), dtype=dpnp.complex64)
        assert_raises(TypeError, dpnp.fft.irfft, a, out=out)


class TestRfft:
    def setup_method(self):
        numpy.random.seed(42)

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
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    def test_fft_1D(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11, dtype=dtype)
        a = dpnp.sin(x)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.rfft(a, n=n, norm=norm)
        expected = numpy.fft.rfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    def test_fft_bool(self, n, norm):
        a = dpnp.ones(11, dtype=dpnp.bool)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.rfft(a, n=n, norm=norm)
        expected = numpy.fft.rfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 1, 0])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft_1D_on_2D_array(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(12, dtype=dtype).reshape(3, 4, order=order)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.rfft(a, n=n, axis=axis, norm=norm)
        expected = numpy.fft.rfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft_1D_on_3D_array(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(24, dtype=dtype).reshape(2, 3, 4, order=order)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.rfft(a, n=n, axis=axis, norm=norm)
        expected = numpy.fft.rfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("n", [None, 5, 20])
    def test_fft_usm_ndarray(self, n):
        x = dpt.linspace(-1, 1, 11)
        a_usm = dpt.asarray(dpt.sin(x))
        a_np = dpt.asnumpy(a_usm)
        out_shape = a_usm.shape[0] // 2 + 1 if n is None else n // 2
        out_dtype = map_dtype_to_device(dpnp.complex128, a_usm.sycl_device)
        out = dpt.empty(out_shape, dtype=out_dtype)

        result = dpnp.fft.rfft(a_usm, n=n, out=out)
        assert out is result.get_array()
        expected = numpy.fft.rfft(a_np, n=n)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
    def test_fft_1D_out(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        a = dpnp.asarray(a, dtype=dtype)
        a_np = dpnp.asnumpy(a)

        out_shape = a.shape[0] // 2 + 1 if n is None else n // 2
        out_dtype = dpnp.complex64 if dtype == dpnp.float32 else dpnp.complex128
        out = dpnp.empty(out_shape, dtype=out_dtype)

        result = dpnp.fft.rfft(a, n=n, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.rfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 0])
    @pytest.mark.parametrize("norm", [None, "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft_1D_on_2D_array_out(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(12, dtype=dtype).reshape(3, 4, order=order)
        a = dpnp.asarray(a_np)

        out_shape = list(a.shape)
        out_shape[axis] = a.shape[axis] // 2 + 1 if n is None else n // 2
        out_shape = tuple(out_shape)
        out_dtype = dpnp.complex64 if dtype == dpnp.float32 else dpnp.complex128
        out = dpnp.empty(out_shape, dtype=out_dtype)

        result = dpnp.fft.rfft(a, n=n, axis=axis, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.rfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_fft_error(self, xp):
        a = xp.ones((4, 3), dtype=xp.complex64)
        # invalid dtype of input array for r2c FFT
        if xp == dpnp:
            # stock NumPy-1.26 ignores imaginary part
            # Intel® NumPy, dpnp, stock NumPy-2.0 return TypeError
            assert_raises(TypeError, xp.fft.rfft, a)

    def test_fft_validate_out(self):
        # Invalid shape for r2c FFT
        a = dpnp.ones((10,), dtype=dpnp.float32)
        out = dpnp.empty((10,), dtype=dpnp.complex64)
        assert_raises(ValueError, dpnp.fft.rfft, a, out=out)
