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
    generate_random_numpy_array,
    get_all_dtypes,
    get_complex_dtypes,
    get_float_dtypes,
)


def _make_array_Hermitian(a, n):
    """
    This function makes necessary changes of the input array of
    `dpnp.fft.irfft` and `dpnp.fft.hfft` functions to make sure the
    given array is Hermitian.

    """

    a[0] = a[0].real
    if n in [None, 18]:
        # f_ny is Nyquist mode (n//2+1 mode) which is n//2 element
        f_ny = -1 if n is None else n // 2
        a[f_ny] = a[f_ny].real
        a[f_ny:] = 0  # no data needed after Nyquist mode

    return a


# TODO: `assert_dtype_allclose` calls in this file have `check_only_type_kind=True`
# since stock NumPy is currently used in public CI for code coverege which
# always returns complex128/float64 for FFT functions, but Intel® NumPy and
# dpnp return complex64/float32 if input is complex64/float32
class TestFft:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.parametrize(
        "shape", [(64,), (8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)]
    )
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_fft_ndim(self, dtype, shape, norm):
        np_data = numpy.arange(64, dtype=dtype).reshape(shape)
        dpnp_data = dpnp.arange(64, dtype=dtype).reshape(shape)

        np_res = numpy.fft.fft(np_data, norm=norm)
        dpnp_res = dpnp.fft.fft(dpnp_data, norm=norm)
        assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)

        np_res = numpy.fft.ifft(np_data, norm=norm)
        dpnp_res = dpnp.fft.ifft(dpnp_data, norm=norm)
        assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_fft_1D(self, dtype, n, norm):
        x = generate_random_numpy_array(11, dtype, low=-1, high=1)
        a_np = numpy.sin(x)  # a.dtype is float16 if x.dtype is bool
        a = dpnp.array(a_np)

        factor = 140 if dtype == dpnp.bool else 8
        result = dpnp.fft.fft(a, n=n, norm=norm)
        expected = numpy.fft.fft(a_np, n=n, norm=norm)
        assert_dtype_allclose(
            result, expected, factor=factor, check_only_type_kind=True
        )

        iresult = dpnp.fft.ifft(result, n=n, norm=norm)
        iexpected = numpy.fft.ifft(expected, n=n, norm=norm)
        assert_dtype_allclose(
            iresult, iexpected, factor=factor, check_only_type_kind=True
        )

    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_fft_1D_bool(self, norm):
        a = dpnp.linspace(-1, 1, 11, dtype=dpnp.bool)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.fft(a, norm=norm)
        expected = numpy.fft.fft(a_np, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifft(result, norm=norm)
        iexpected = numpy.fft.ifft(expected, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 1, 0])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
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
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
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
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
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
        y_np = numpy.random.rand(20, 20) + 1j * numpy.random.rand(20, 20)
        y = dpnp.asarray(y_np)
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
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
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

        # returns empty array, a.size=0
        result = dpnp.fft.fft(a, axis=0)
        expected = numpy.fft.fft(a_np, axis=0)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        # calculates FFT, a.size become non-zero because of n=2
        result = dpnp.fft.fft(a, axis=1, n=2)
        expected = numpy.fft.fft(a_np, axis=1, n=2)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_fft_error(self, xp):
        # 0-D input
        a = xp.array(3)
        # dpnp and Intel® NumPy raise ValueError
        # stock NumPy raises IndexError
        assert_raises((ValueError, IndexError), xp.fft.fft, a)

        # n is not int
        a = xp.ones((4, 3))
        if xp == dpnp:
            # dpnp and stock NumPy raise TypeError
            # Intel® NumPy raises SystemError for Python 3.10 and 3.11
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

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    def test_negative_stride(self, dtype):
        a = dpnp.arange(10, dtype=dtype)
        result = dpnp.fft.fft(a[::-1])
        expected = numpy.fft.fft(a.asnumpy()[::-1])

        assert_dtype_allclose(result, expected, check_only_type_kind=True)


class TestFft2:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axes", [(0, 1), (1, 2), (0, 2), (2, 1), (2, 0)])
    @pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fft2(self, dtype, axes, norm, order):
        a_np = generate_random_numpy_array((2, 3, 4), dtype, order)
        a = dpnp.array(a_np)

        result = dpnp.fft.fft2(a, axes=axes, norm=norm)
        expected = numpy.fft.fft2(a_np, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifft2(result, axes=axes, norm=norm)
        iexpected = numpy.fft.ifft2(expected, axes=axes, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("s", [None, (3, 3), (10, 10), (3, 10)])
    def test_fft2_s(self, s):
        a_np = generate_random_numpy_array((6, 8), dtype=numpy.complex64)
        a = dpnp.array(a_np)

        result = dpnp.fft.fft2(a, s=s)
        expected = numpy.fft.fft2(a_np, s=s)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifft2(result, s=s)
        iexpected = numpy.fft.ifft2(expected, s=s)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_fft_error(self, xp):
        # 0-D input
        a = xp.ones(())
        assert_raises(IndexError, xp.fft.fft2, a)


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


class TestFftn:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    @pytest.mark.parametrize(
        "axes", [None, (0, 1, 2), (-1, -4, -2), (-2, -4, -1, -3)]
    )
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_fftn(self, dtype, axes, norm, order):
        a_np = generate_random_numpy_array((2, 3, 4, 5), dtype, order)
        a = dpnp.array(a_np)

        result = dpnp.fft.fftn(a, axes=axes, norm=norm)
        expected = numpy.fft.fftn(a_np, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifftn(result, axes=axes, norm=norm)
        iexpected = numpy.fft.ifftn(expected, axes=axes, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize(
        "axes", [(2, 0, 2, 0), (0, 1, 1), (2, 0, 1, 3, 2, 1)]
    )
    def test_fftn_repeated_axes(self, axes):
        a_np = generate_random_numpy_array((2, 3, 4, 5), dtype=numpy.complex64)
        a = dpnp.array(a_np)

        result = dpnp.fft.fftn(a, axes=axes)
        # Intel® NumPy ignores repeated axes, handle it one by one
        expected = a_np
        for ii in axes:
            expected = numpy.fft.fft(expected, axis=ii)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifftn(result, axes=axes)
        iexpected = expected
        for ii in axes:
            iexpected = numpy.fft.ifft(iexpected, axis=ii)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("axes", [(2, 3, 3, 2), (0, 0, 3, 3)])
    @pytest.mark.parametrize("s", [(5, 4, 3, 3), (7, 8, 10, 9)])
    def test_fftn_repeated_axes_with_s(self, axes, s):
        a_np = generate_random_numpy_array((2, 3, 4, 5), dtype=numpy.complex64)
        a = dpnp.array(a_np)

        result = dpnp.fft.fftn(a, s=s, axes=axes)
        # Intel® NumPy ignores repeated axes, handle it one by one
        expected = a_np
        for jj, ii in zip(s[::-1], axes[::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.ifftn(result, s=s, axes=axes)
        iexpected = expected
        for jj, ii in zip(s[::-1], axes[::-1]):
            iexpected = numpy.fft.ifft(iexpected, n=jj, axis=ii)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("axes", [(0, 1, 2, 3), (1, 2, 1, 2), (2, 2, 2, 3)])
    @pytest.mark.parametrize("s", [(2, 3, 4, 5), (5, 4, 7, 8), (2, 5, 1, 2)])
    def test_fftn_out(self, axes, s):
        a_np = generate_random_numpy_array((2, 3, 4, 5), dtype=numpy.complex64)
        a = dpnp.array(a_np)

        out_shape = list(a.shape)
        for s_i, axis in zip(s[::-1], axes[::-1]):
            out_shape[axis] = s_i
        out = dpnp.empty(out_shape, dtype=a.dtype)
        result = dpnp.fft.fftn(a, out=out, s=s, axes=axes)
        assert out is result
        # Intel® NumPy ignores repeated axes, handle it one by one
        expected = a_np
        for jj, ii in zip(s[::-1], axes[::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        out = dpnp.empty(out_shape, dtype=a.dtype)
        iresult = dpnp.fft.ifftn(result, out=out, s=s, axes=axes)
        assert out is iresult
        iexpected = expected
        for jj, ii in zip(s[::-1], axes[::-1]):
            iexpected = numpy.fft.ifft(iexpected, n=jj, axis=ii)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    def test_negative_s(self):
        a_np = generate_random_numpy_array((3, 4, 5), dtype=numpy.complex64)
        a = dpnp.array(a_np)

        # For dpnp and stock NumPy 2.0, if s is -1, the whole input is used
        # (no padding or trimming).
        result = dpnp.fft.fftn(a, s=(-1, -1), axes=(0, 2))
        expected = numpy.fft.fftn(a_np, s=(3, 5), axes=(0, 2))
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    def test_fftn_empty_array(self):
        a_np = numpy.empty((10, 0, 4), dtype=numpy.complex64)
        a = dpnp.array(a_np)

        result = dpnp.fft.fftn(a, axes=(0, 2))
        expected = numpy.fft.fftn(a_np, axes=(0, 2))
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        result = dpnp.fft.fftn(a, axes=(0, 1, 2), s=(5, 2, 4))
        expected = numpy.fft.fftn(a_np, axes=(0, 1, 2), s=(5, 2, 4))
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_fftn_0D(self, dtype):
        a = dpnp.array(3, dtype=dtype)  # 0-D input

        # axes is None
        # For 0-D array, stock Numpy and dpnp return input array
        # while Intel® NumPy return a complex zero
        result = dpnp.fft.fftn(a)
        expected = a.asnumpy()
        assert_dtype_allclose(result, expected)

        # axes=()
        # For 0-D array with axes=(), stock Numpy and dpnp return input array
        # Intel® NumPy does not support empty axes and raises an Error
        result = dpnp.fft.fftn(a, axes=())
        expected = a.asnumpy()
        assert_dtype_allclose(result, expected)

        # axes=(0,)
        # For 0-D array with non-empty axes, stock Numpy and dpnp raise
        # IndexError, while Intel® NumPy raises ZeroDivisionError
        assert_raises(IndexError, dpnp.fft.fftn, a, axes=(0,))

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_fftn_empty_axes(self, dtype):
        a = dpnp.ones((2, 3, 4), dtype=dtype)

        # For axes=(), stock Numpy and dpnp return input array
        # Intel® NumPy does not support empty axes and raises an Error
        result = dpnp.fft.fftn(a, axes=())
        expected = a.asnumpy()
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_fft_error(self, xp):
        # s is not int
        a = xp.ones((4, 3))
        # dpnp and stock NumPy raise TypeError
        # Intel® NumPy raises ValueError
        assert_raises(
            (TypeError, ValueError), xp.fft.fftn, a, s=(5.0,), axes=(0,)
        )

        # s is not a sequence
        assert_raises(TypeError, xp.fft.fftn, a, s=5, axes=(0,))

        # Invalid number of FFT point, invalid s value
        assert_raises(ValueError, xp.fft.fftn, a, s=(-5,), axes=(0,))

        # axes should be given if s is not None
        # dpnp raises ValueError
        # stock NumPy will raise an Error in future versions
        # Intel® NumPy raises TypeError for a different reason:
        # when given, axes and shape arguments have to be of the same length
        if xp == dpnp:
            assert_raises(ValueError, xp.fft.fftn, a, s=(5,))

        # axes and s should have the same length
        assert_raises(ValueError, xp.fft.fftn, a, s=(5, 5), axes=(0,))


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

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("n", [None, 5, 18])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_hfft_1D(self, dtype, n, norm):
        x = generate_random_numpy_array(11, dtype, low=-1, high=1)
        a_np = numpy.sin(x)
        if numpy.issubdtype(dtype, numpy.complexfloating):
            a_np = _make_array_Hermitian(a_np, n)
        a = dpnp.array(a_np)

        result = dpnp.fft.hfft(a, n=n, norm=norm)
        expected = numpy.fft.hfft(a_np, n=n, norm=norm)
        # check_only_type_kind=True since numpy always returns float64
        # but dpnp return float32 if input is float32
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_ihfft_1D(self, dtype, n, norm):
        x = generate_random_numpy_array(11, dtype, low=-1, high=1)
        a_np = numpy.sin(x)
        a = dpnp.array(a_np)

        result = dpnp.fft.ihfft(a, n=n, norm=norm)
        expected = numpy.fft.ihfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_ihfft_bool(self, n, norm):
        a = dpnp.ones(11, dtype=dpnp.bool)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.ihfft(a, n=n, norm=norm)
        expected = numpy.fft.ihfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    def test_ihfft_error(self):
        a = dpnp.ones(11)
        # incorrect norm
        assert_raises(ValueError, dpnp.fft.ihfft, a, norm="backwards")


class TestIrfft:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("n", [None, 5, 18])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_irfft_1D(self, dtype, n, norm):
        x = generate_random_numpy_array(11, dtype, low=-1, high=1)
        a_np = numpy.sin(x)
        if numpy.issubdtype(dtype, numpy.complexfloating):
            a_np = _make_array_Hermitian(a_np, n)
        a = dpnp.array(a_np)

        result = dpnp.fft.irfft(a, n=n, norm=norm)
        expected = numpy.fft.irfft(a_np, n=n, norm=norm)
        # check_only_type_kind=True since Intel® NumPy always returns float64
        # but dpnp return float32 if input is float32
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 1, 0])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_irfft_1D_on_2D_array(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(12, dtype=dtype).reshape(3, 4, order=order)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.irfft(a, n=n, axis=axis, norm=norm)
        expected = numpy.fft.irfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_irfft_1D_on_3D_array(self, dtype, n, axis, norm, order):
        a_np = generate_random_numpy_array((4, 5, 6), dtype, order)
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
    def test_irfft_usm_ndarray(self, n):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        a = _make_array_Hermitian(a, n)
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
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_irfft_1D_out(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        a = _make_array_Hermitian(a, n)
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
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_irfft_1D_on_2D_array_out(self, dtype, n, axis, norm, order):
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

    def test_irfft_validate_out(self):
        # Invalid dtype for c2r FFT
        a = dpnp.ones((10,), dtype=dpnp.complex64)
        out = dpnp.empty((18,), dtype=dpnp.complex64)
        assert_raises(TypeError, dpnp.fft.irfft, a, out=out)


class TestRfft:
    def setup_method(self):
        numpy.random.seed(42)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "shape", [(64,), (8, 8), (4, 16), (4, 4, 4), (2, 4, 4, 2)]
    )
    def test_rfft(self, dtype, shape):
        np_data = numpy.arange(64, dtype=dtype).reshape(shape)
        dpnp_data = dpnp.arange(64, dtype=dtype).reshape(shape)

        np_res = numpy.fft.rfft(np_data)
        dpnp_res = dpnp.fft.rfft(dpnp_data)

        assert_dtype_allclose(dpnp_res, np_res, check_only_type_kind=True)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_rfft_1D(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11, dtype=dtype)
        a = dpnp.sin(x)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.rfft(a, n=n, norm=norm)
        expected = numpy.fft.rfft(a_np, n=n, norm=norm)
        factor = 120 if dtype in [dpnp.int8, dpnp.uint8] else 8
        assert_dtype_allclose(
            result, expected, factor=factor, check_only_type_kind=True
        )

    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_rfft_bool(self, n, norm):
        a = dpnp.ones(11, dtype=dpnp.bool)
        a_np = dpnp.asnumpy(a)

        result = dpnp.fft.rfft(a, n=n, norm=norm)
        expected = numpy.fft.rfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 1, 0])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_rfft_1D_on_2D_array(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(12, dtype=dtype).reshape(3, 4, order=order)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.rfft(a, n=n, axis=axis, norm=norm)
        expected = numpy.fft.rfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_rfft_1D_on_3D_array(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(24, dtype=dtype).reshape(2, 3, 4, order=order)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.rfft(a, n=n, axis=axis, norm=norm)
        expected = numpy.fft.rfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("n", [None, 5, 20])
    def test_rfft_usm_ndarray(self, n):
        x = dpt.linspace(-1, 1, 11)
        a_usm = dpt.asarray(dpt.sin(x))
        a_np = dpt.asnumpy(a_usm)
        out_shape = a_usm.shape[0] // 2 + 1 if n is None else n // 2 + 1
        out_dtype = map_dtype_to_device(dpnp.complex128, a_usm.sycl_device)
        out = dpt.empty(out_shape, dtype=out_dtype)

        result = dpnp.fft.rfft(a_usm, n=n, out=out)
        assert out is result.get_array()
        expected = numpy.fft.rfft(a_np, n=n)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_rfft_1D_out(self, dtype, n, norm):
        x = dpnp.linspace(-1, 1, 11)
        a = dpnp.sin(x) + 1j * dpnp.cos(x)
        a = dpnp.asarray(a, dtype=dtype)
        a_np = dpnp.asnumpy(a)

        out_shape = a.shape[0] // 2 + 1 if n is None else n // 2 + 1
        out_dtype = dpnp.complex64 if dtype == dpnp.float32 else dpnp.complex128
        out = dpnp.empty(out_shape, dtype=out_dtype)

        result = dpnp.fft.rfft(a, n=n, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.rfft(a_np, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 0])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_rfft_1D_on_2D_array_out(self, dtype, n, axis, norm, order):
        a_np = numpy.arange(12, dtype=dtype).reshape(3, 4, order=order)
        a = dpnp.asarray(a_np)

        out_shape = list(a.shape)
        out_shape[axis] = a.shape[axis] // 2 + 1 if n is None else n // 2 + 1
        out_shape = tuple(out_shape)
        out_dtype = dpnp.complex64 if dtype == dpnp.float32 else dpnp.complex128
        out = dpnp.empty(out_shape, dtype=out_dtype)

        result = dpnp.fft.rfft(a, n=n, axis=axis, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.rfft(a_np, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_rfft_error(self, xp):
        a = xp.ones((4, 3), dtype=xp.complex64)
        # invalid dtype of input array for r2c FFT
        if xp == dpnp:
            # stock NumPy-1.26 ignores imaginary part
            # Intel® NumPy, dpnp, stock NumPy-2.0 raise TypeError
            assert_raises(TypeError, xp.fft.rfft, a)

    def test_fft_validate_out(self):
        # Invalid shape for r2c FFT
        a = dpnp.ones((10,), dtype=dpnp.float32)
        out = dpnp.empty((10,), dtype=dpnp.complex64)
        assert_raises(ValueError, dpnp.fft.rfft, a, out=out)


class TestRfft2:
    def setup_method(self):
        numpy.random.seed(42)

    # TODO: add other axes when mkl_fft gh-119 is addressed
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("axes", [(0, 1)])  # (1, 2),(0, 2),(2, 1),(2, 0)
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_rfft2(self, dtype, axes, norm, order):
        a_np = generate_random_numpy_array((2, 3, 4), dtype, order)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.rfft2(a, axes=axes, norm=norm)
        expected = numpy.fft.rfft2(a_np, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        s = (a.shape[axes[0]], a.shape[axes[1]])
        result = dpnp.fft.irfft2(result, s=s, axes=axes, norm=norm)
        expected = numpy.fft.irfft2(expected, s=s, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_irfft2(self, dtype):
        # x is Hermitian symmetric
        x = numpy.array([[0, 1, 2], [5, 4, 6], [5, 7, 6]])
        a_np = numpy.array(x, dtype=dtype)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.irfft2(a)
        expected = numpy.fft.irfft2(a_np)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("s", [None, (3, 3), (10, 10), (3, 10)])
    def test_rfft2_s(self, s):
        a_np = generate_random_numpy_array((6, 8))
        a = dpnp.array(a_np)

        result = dpnp.fft.rfft2(a, s=s)
        expected = numpy.fft.rfft2(a_np, s=s)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        result = dpnp.fft.irfft2(result, s=s)
        expected = numpy.fft.irfft2(expected, s=s)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_rfft2_error(self, xp):
        a = xp.ones((2, 3))
        # empty axes
        assert_raises(IndexError, xp.fft.rfft2, a, axes=())

        a = xp.ones((2, 3), dtype=xp.complex64)
        # Input array must be real
        # Stock NumPy 2.0 raises TypeError
        # while stock NumPy 1.26 ignores imaginary part
        if xp == dpnp:
            assert_raises(TypeError, xp.fft.rfft2, a)


class TestRfftn:
    def setup_method(self):
        numpy.random.seed(42)

    # TODO: add additional axes when mkl_fft gh-119 is addressed
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "axes", [(0, 1, 2), (-2, -4, -1, -3)]  # (-1, -4, -2)
    )
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_rfftn(self, dtype, axes, norm, order):
        a_np = generate_random_numpy_array((2, 3, 4, 5), dtype, order)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.rfftn(a, axes=axes, norm=norm)
        expected = numpy.fft.rfftn(a_np, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        s = []
        for axis in axes:
            s.append(a.shape[axis])
        iresult = dpnp.fft.irfftn(result, s=s, axes=axes, norm=norm)
        iexpected = numpy.fft.irfftn(expected, s=s, axes=axes, norm=norm)
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize(
        "axes", [(2, 0, 2, 0), (0, 1, 1), (2, 0, 1, 3, 2, 1)]
    )
    def test_rfftn_repeated_axes(self, axes):
        a_np = generate_random_numpy_array((2, 3, 4, 5))
        a = dpnp.array(a_np)

        result = dpnp.fft.rfftn(a, axes=axes)
        # Intel® NumPy ignores repeated axes, handle it one by one
        expected = numpy.fft.rfft(a_np, axis=axes[-1])
        # need to pass shape for c2c FFT since expected and a_np
        # do not have the same shape after calling rfft
        shape = []
        for axis in axes:
            shape.append(a_np.shape[axis])
        for jj, ii in zip(shape[-2::-1], axes[-2::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.irfftn(result, axes=axes)
        iexpected = expected
        for ii in axes[-2::-1]:
            iexpected = numpy.fft.ifft(iexpected, axis=ii)
        iexpected = numpy.fft.irfft(iexpected, axis=axes[-1])
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("axes", [(2, 3, 3, 2), (0, 0, 3, 3)])
    @pytest.mark.parametrize("s", [(5, 4, 3, 3), (7, 8, 10, 9)])
    def test_rfftn_repeated_axes_with_s(self, axes, s):
        a_np = generate_random_numpy_array((2, 3, 4, 5), dtype=numpy.float32)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.rfftn(a, s=s, axes=axes)
        # Intel® NumPy ignores repeated axes, handle it one by one
        expected = numpy.fft.rfft(a_np, n=s[-1], axis=axes[-1])
        for jj, ii in zip(s[-2::-1], axes[-2::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        iresult = dpnp.fft.irfftn(result, s=s, axes=axes)
        iexpected = expected
        for jj, ii in zip(s[-2::-1], axes[-2::-1]):
            iexpected = numpy.fft.ifft(iexpected, n=jj, axis=ii)
        iexpected = numpy.fft.irfft(iexpected, n=s[-1], axis=axes[-1])
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    @pytest.mark.parametrize("axes", [(0, 1, 2, 3), (1, 2, 1, 2), (2, 2, 2, 3)])
    @pytest.mark.parametrize("s", [(2, 3, 4, 5), (5, 6, 7, 9), (2, 5, 1, 2)])
    def test_rfftn_out(self, axes, s):
        x = numpy.random.uniform(-10, 10, 120)
        a_np = numpy.array(x, dtype=numpy.float32).reshape(2, 3, 4, 5)
        a = dpnp.asarray(a_np)

        out_shape = list(a.shape)
        out_shape[axes[-1]] = s[-1] // 2 + 1
        for s_i, axis in zip(s[-2::-1], axes[-2::-1]):
            out_shape[axis] = s_i
        out = dpnp.empty(out_shape, dtype=numpy.complex64)

        result = dpnp.fft.rfftn(a, out=out, s=s, axes=axes)
        assert out is result
        # Intel® NumPy ignores repeated axes, handle it one by one
        expected = numpy.fft.rfft(a_np, n=s[-1], axis=axes[-1])
        for jj, ii in zip(s[-2::-1], axes[-2::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        out_shape = list(a.shape)
        for s_i, axis in zip(s[-2::-1], axes[-2::-1]):
            out_shape[axis] = s_i
        out_shape[axes[-1]] = s[-1]
        out = dpnp.empty(out_shape, dtype=numpy.float32)

        iresult = dpnp.fft.irfftn(result, out=out, s=s, axes=axes)
        assert out is iresult

        iexpected = expected
        for jj, ii in zip(s[-2::-1], axes[-2::-1]):
            iexpected = numpy.fft.ifft(iexpected, n=jj, axis=ii)
        iexpected = numpy.fft.irfft(iexpected, n=s[-1], axis=axes[-1])
        assert_dtype_allclose(iresult, iexpected, check_only_type_kind=True)

    def test_rfftn_1d_array(self):
        x = numpy.random.uniform(-10, 10, 20)
        a_np = numpy.array(x, dtype=numpy.float32)
        a = dpnp.asarray(a_np)

        result = dpnp.fft.rfftn(a)
        expected = numpy.fft.rfftn(a_np)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

        result = dpnp.fft.irfftn(a)
        expected = numpy.fft.irfftn(a_np)
        assert_dtype_allclose(result, expected, check_only_type_kind=True)
