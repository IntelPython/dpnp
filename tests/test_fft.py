import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_raises

import dpnp

from .helper import assert_dtype_allclose, get_all_dtypes, get_complex_dtypes


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

    @pytest.mark.parametrize("n", [None, 5, 20])
    def test_fft_usm_ndarray(self, n):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        a_usm = dpt.asarray(a, dtype=dpnp.complex64)
        a_np = dpnp.asnumpy(a_usm)
        out_shape = (n,) if n is not None else a.shape
        out = dpt.empty(out_shape, dtype=a.dtype)

        result = dpnp.fft.fft(a_usm, n=n, out=out)
        assert out is result.get_array()
        expected = numpy.fft.fft(a_np, n=n)
        assert_dtype_allclose(result, expected)

        # in-place
        if n is None:
            result = dpnp.fft.fft(a_usm, n=n, out=a_usm)
            assert a_usm is result.get_array()
            assert_dtype_allclose(result, expected)

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
        assert_dtype_allclose(result, expected)

        iresult = dpnp.fft.ifft(result, n=n, norm=norm, out=out)
        assert out is iresult
        iexpected = numpy.fft.ifft(expected, n=n, norm=norm)
        assert_dtype_allclose(iresult, iexpected)

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
        assert_dtype_allclose(result, expected)

        iresult = dpnp.fft.ifft(result, n=n, axis=axis, norm=norm, out=out)
        assert out is iresult
        iexpected = numpy.fft.ifft(expected, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(iresult, iexpected)

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
        assert_dtype_allclose(result, expected)

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
        assert_dtype_allclose(result, expected)

    def test_fft_empty_array(self):
        a_np = numpy.empty((10, 0, 4), dtype=numpy.complex64)
        a = dpnp.array(a_np)

        result = dpnp.fft.fft(a, axis=0)
        expected = numpy.fft.fft(a_np, axis=0)
        assert_dtype_allclose(result, expected)

        result = dpnp.fft.fft(a, axis=1, n=2)
        expected = numpy.fft.fft(a_np, axis=1, n=2)
        assert_dtype_allclose(result, expected)

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

    def test_fft_validate_out(self):
        # Inconsistent sycl_queue
        a = dpnp.ones((10,), dtype=dpnp.complex64, sycl_queue=dpctl.SyclQueue())
        out = dpnp.empty((10,), sycl_queue=dpctl.SyclQueue())
        assert_raises(ExecutionPlacementError, dpnp.fft.fft, a, out=out)

        # Invalid shape
        a = dpnp.ones((10,), dtype=dpnp.complex64)
        out = dpnp.empty((11,), dtype=dpnp.complex64)
        assert_raises(ValueError, dpnp.fft.fft, a, out=out)

        # Invalid dtype
        a = dpnp.ones((10,), dtype=dpnp.complex64)
        out = dpnp.empty((11,), dtype=dpnp.float32)
        assert_raises(ValueError, dpnp.fft.fft, a, out=out)


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
