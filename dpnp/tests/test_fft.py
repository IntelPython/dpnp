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
    has_support_aspect16,
    numpy_version,
)
from .third_party.cupy import testing


class TestFft:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_basic(self, dtype, n, norm):
        a = generate_random_numpy_array(11, dtype)
        ia = dpnp.array(a)

        result = dpnp.fft.fft(ia, n=n, norm=norm)
        expected = numpy.fft.fft(a, n=n, norm=norm)
        flag = True if numpy_version() < "2.0.0" else False
        assert_dtype_allclose(result, expected, check_only_type_kind=flag)

        # inverse FFT
        result = dpnp.fft.ifft(result, n=n, norm=norm)
        expected = numpy.fft.ifft(expected, n=n, norm=norm)
        assert_dtype_allclose(result, expected, check_only_type_kind=flag)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 1, 0])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_2d_array(self, dtype, n, axis, norm, order):
        a = generate_random_numpy_array((3, 4), dtype=dtype, order=order)
        ia = dpnp.array(a)

        result = dpnp.fft.fft(ia, n=n, axis=axis, norm=norm)
        expected = numpy.fft.fft(a, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.ifft(result, n=n, axis=axis, norm=norm)
        expected = numpy.fft.ifft(expected, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_3d_array(self, dtype, n, axis, norm, order):
        a = generate_random_numpy_array((2, 3, 4), dtype=dtype, order=order)
        ia = dpnp.array(a)

        result = dpnp.fft.fft(ia, n=n, axis=axis, norm=norm)
        expected = numpy.fft.fft(a, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.ifft(result, n=n, axis=axis, norm=norm)
        expected = numpy.fft.ifft(expected, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("n", [None, 5, 20])
    def test_usm_ndarray(self, n):
        a = generate_random_numpy_array(11, dtype=numpy.complex64)
        a_usm = dpt.asarray(a)

        expected = numpy.fft.fft(a, n=n)
        out = dpt.empty(expected.shape, dtype=a_usm.dtype)
        result = dpnp.fft.fft(a_usm, n=n, out=out)
        assert out is result.get_array()
        assert_dtype_allclose(result, expected)

        # in-place
        if n is None:
            result = dpnp.fft.fft(a_usm, n=n, out=a_usm)
            assert a_usm is result.get_array()
            assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_out(self, dtype, n, norm):
        a = generate_random_numpy_array(11, dtype=dtype)
        ia = dpnp.array(a)

        # FFT
        expected = numpy.fft.fft(a, n=n, norm=norm)
        out = dpnp.empty(expected.shape, dtype=a.dtype)
        result = dpnp.fft.fft(ia, n=n, norm=norm, out=out)
        assert out is result
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.ifft(result, n=n, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.ifft(expected, n=n, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_inplace_out(self, axis):
        # Test some weirder in-place combinations
        y_np = generate_random_numpy_array((20, 20), dtype=numpy.complex64)
        y = dpnp.array(y_np)
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
    def test_2d_array_out(self, dtype, n, axis, norm, order):
        a = generate_random_numpy_array((3, 4), dtype=dtype, order=order)
        ia = dpnp.array(a)

        expected = numpy.fft.fft(a, n=n, axis=axis, norm=norm)
        out = dpnp.empty(expected.shape, dtype=a.dtype)

        result = dpnp.fft.fft(ia, n=n, axis=axis, norm=norm, out=out)
        assert out is result
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.ifft(result, n=n, axis=axis, norm=norm, out=out)
        assert out is result
        expected = numpy.fft.ifft(expected, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("stride", [-1, -3, 2, 5])
    def test_strided_1d(self, stride):
        a = generate_random_numpy_array(20, dtype=numpy.complex64)
        ia = dpnp.array(a)
        a = a[::stride]
        ia = ia[::stride]

        result = dpnp.fft.fft(ia)
        expected = numpy.fft.fft(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("stride_x", [-1, -3, 2, 3])
    @pytest.mark.parametrize("stride_y", [-1, -3, 2, 3])
    def test_strided_2d(self, stride_x, stride_y):
        a = generate_random_numpy_array((12, 10), dtype=numpy.complex64)
        ia = dpnp.array(a)
        a = a[::stride_x, ::stride_y]
        ia = ia[::stride_x, ::stride_y]

        result = dpnp.fft.fft(ia)
        expected = numpy.fft.fft(a)
        assert_dtype_allclose(result, expected)

    def test_empty_array(self):
        a = numpy.empty((10, 0, 4), dtype=numpy.complex64)
        ia = dpnp.array(a)

        # returns empty array, a.size=0
        result = dpnp.fft.fft(ia, axis=0)
        expected = numpy.fft.fft(a, axis=0)
        assert_dtype_allclose(result, expected)

        # calculates FFT, a.size become non-zero because of n=2
        result = dpnp.fft.fft(ia, axis=1, n=2)
        expected = numpy.fft.fft(a, axis=1, n=2)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        # 0-D input
        a = xp.array(3)
        # dpnp and Intel NumPy raise ValueError
        # stock NumPy raises IndexError
        assert_raises((ValueError, IndexError), xp.fft.fft, a)

        # n is not int
        a = xp.ones((4, 3))
        if xp == dpnp:
            # dpnp and stock NumPy raise TypeError
            # Intel NumPy raises SystemError for Python 3.10 and 3.11
            # and no error for Python 3.9
            assert_raises(TypeError, xp.fft.fft, a, n=5.0)

        # Invalid number of FFT point for incorrect n value
        assert_raises(ValueError, xp.fft.fft, a, n=-5)

        # invalid norm
        assert_raises(ValueError, xp.fft.fft, a, norm="square")

        # Invalid number of FFT point for empty arrays
        a = xp.ones((5, 0, 4))
        assert_raises(ValueError, xp.fft.fft, a, axis=1)

    def test_validate_out(self):
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

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("stride", [2, 3, -1, -3])
    def test_strided(self, dtype, stride):
        a = generate_random_numpy_array(20, dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.fft.fft(ia[::stride])
        expected = numpy.fft.fft(a[::stride])
        assert_dtype_allclose(result, expected)


class TestFft2:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axes", [(0, 1), (1, 2), (0, 2), (2, 1), (2, 0)])
    @pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_basic(self, dtype, axes, norm, order):
        a = generate_random_numpy_array((2, 3, 4), dtype, order)
        ia = dpnp.array(a)

        result = dpnp.fft.fft2(ia, axes=axes, norm=norm)
        expected = numpy.fft.fft2(a, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.ifft2(result, axes=axes, norm=norm)
        expected = numpy.fft.ifft2(expected, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("s", [None, (3, 3), (10, 10), (3, 10)])
    def test_s(self, s):
        a = generate_random_numpy_array((6, 8), dtype=numpy.complex64)
        ia = dpnp.array(a)

        result = dpnp.fft.fft2(ia, s=s)
        expected = numpy.fft.fft2(a, s=s)
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.ifft2(result, s=s)
        expected = numpy.fft.ifft2(expected, s=s)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        # 0-D input
        a = xp.ones(())
        assert_raises(IndexError, xp.fft.fft2, a)


@pytest.mark.parametrize("func", ["fftfreq", "rfftfreq"])
class TestFftfreq:
    @pytest.mark.parametrize("n", [10, 20])
    @pytest.mark.parametrize("d", [0.5, 2])
    def test_basic(self, func, n, d):
        result = getattr(dpnp.fft, func)(n, d)
        expected = getattr(numpy.fft, func)(n, d)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dt", [None] + get_float_dtypes())
    def test_dtype(self, func, dt):
        n = 15
        result = getattr(dpnp.fft, func)(n, dtype=dt)
        expected = getattr(numpy.fft, func)(n).astype(dt)
        assert_dtype_allclose(result, expected)

    def test_error(self, func):
        func = getattr(dpnp.fft, func)
        # n must be an integer
        assert_raises(ValueError, func, 10.0)

        # d must be an scalar
        assert_raises(ValueError, func, 10, (2,))

        # dtype must be None or a real-valued floating-point dtype
        # which is passed as a keyword argument only
        assert_raises(TypeError, func, 10, 2, None)
        assert_raises(ValueError, func, 10, 2, dtype=dpnp.intp)
        assert_raises(ValueError, func, 10, 2, dtype=dpnp.complex64)


class TestFftn:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize(
        "axes", [None, (0, 1, 2), (-1, -4, -2), (-2, -4, -1, -3)]
    )
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_basic(self, dtype, axes, norm, order):
        a = generate_random_numpy_array((2, 3, 4, 5), dtype, order)
        ia = dpnp.array(a)

        result = dpnp.fft.fftn(ia, axes=axes, norm=norm)
        expected = numpy.fft.fftn(a, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.ifftn(result, axes=axes, norm=norm)
        expected = numpy.fft.ifftn(expected, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes", [(2, 0, 2, 0), (0, 1, 1), (2, 0, 1, 3, 2, 1)]
    )
    def test_repeated_axes(self, axes):
        a = generate_random_numpy_array((2, 3, 4, 5), dtype=numpy.complex64)
        ia = dpnp.array(a)

        result = dpnp.fft.fftn(ia, axes=axes)
        # Intel NumPy ignores repeated axes (mkl_fft-gh-104), handle it one by one
        expected = a
        for ii in axes:
            expected = numpy.fft.fft(expected, axis=ii)
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.ifftn(result, axes=axes)
        for ii in axes:
            expected = numpy.fft.ifft(expected, axis=ii)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axes", [(2, 3, 3, 2), (0, 0, 3, 3)])
    @pytest.mark.parametrize("s", [(5, 4, 3, 3), (7, 8, 10, 9)])
    def test_repeated_axes_with_s(self, axes, s):
        a = generate_random_numpy_array((2, 3, 4, 5), dtype=numpy.complex64)
        ia = dpnp.array(a)

        result = dpnp.fft.fftn(ia, s=s, axes=axes)
        # Intel NumPy ignores repeated axes (mkl_fft-gh-104), handle it one by one
        expected = a
        for jj, ii in zip(s[::-1], axes[::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.ifftn(result, s=s, axes=axes)
        for jj, ii in zip(s[::-1], axes[::-1]):
            expected = numpy.fft.ifft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axes", [(0, 1, 2, 3), (1, 2, 1, 2), (2, 2, 2, 3)])
    @pytest.mark.parametrize("s", [(2, 3, 4, 5), (5, 4, 7, 8), (2, 5, 1, 2)])
    def test_out(self, axes, s):
        a = generate_random_numpy_array((2, 3, 4, 5), dtype=numpy.complex64)
        ia = dpnp.array(a)

        # Intel NumPy ignores repeated axes (mkl_fft-gh-104), handle it one by one
        expected = a
        for jj, ii in zip(s[::-1], axes[::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        out = dpnp.empty(expected.shape, dtype=a.dtype)

        result = dpnp.fft.fftn(ia, out=out, s=s, axes=axes)
        assert out is result
        assert_dtype_allclose(result, expected)

        # inverse FFT
        out = dpnp.empty(expected.shape, dtype=a.dtype)
        result = dpnp.fft.ifftn(result, out=out, s=s, axes=axes)
        assert out is result
        for jj, ii in zip(s[::-1], axes[::-1]):
            expected = numpy.fft.ifft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected)

    def test_negative_s(self):
        a = generate_random_numpy_array((3, 4, 5), dtype=numpy.complex64)
        ia = dpnp.array(a)

        # For dpnp and stock NumPy 2.0, if s is -1, the whole input is used
        # (no padding or trimming).
        result = dpnp.fft.fftn(ia, s=(-1, -1), axes=(0, 2))
        expected = numpy.fft.fftn(a, s=(3, 5), axes=(0, 2))
        assert_dtype_allclose(result, expected)

    def test_empty_array(self):
        a = numpy.empty((10, 0, 4), dtype=numpy.complex64)
        ia = dpnp.array(a)

        result = dpnp.fft.fftn(ia, axes=(0, 2))
        expected = numpy.fft.fftn(a, axes=(0, 2))
        assert_dtype_allclose(result, expected)

        result = dpnp.fft.fftn(ia, axes=(0, 1, 2), s=(5, 2, 4))
        expected = numpy.fft.fftn(a, axes=(0, 1, 2), s=(5, 2, 4))
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_0D(self, dtype):
        a = dpnp.array(3, dtype=dtype)  # 0-D input

        # axes is None
        # For 0-D array, stock Numpy and dpnp return input array
        # while Intel NumPy return a complex zero
        result = dpnp.fft.fftn(a)
        expected = a.asnumpy()
        assert_dtype_allclose(result, expected)

        # axes=()
        # For 0-D array with axes=(), stock Numpy and dpnp return input array
        # Intel NumPy does not support empty axes and raises an Error
        result = dpnp.fft.fftn(a, axes=())
        expected = a.asnumpy()
        assert_dtype_allclose(result, expected)

        # axes=(0,)
        # For 0-D array with non-empty axes, stock Numpy and dpnp raise
        # IndexError, while Intel NumPy raises ZeroDivisionError
        assert_raises(IndexError, dpnp.fft.fftn, a, axes=(0,))

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_empty_axes(self, dtype):
        a = dpnp.ones((2, 3, 4), dtype=dtype)

        # For axes=(), stock Numpy and dpnp return input array
        # Intel NumPy does not support empty axes and raises an Error
        result = dpnp.fft.fftn(a, axes=())
        expected = a.asnumpy()
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        # s is not int
        a = xp.ones((4, 3))
        # dpnp and stock NumPy raise TypeError
        # Intel NumPy raises ValueError
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
        # Intel NumPy raises TypeError for a different reason:
        # when given, axes and shape arguments have to be of the same length
        if xp == dpnp:
            assert_raises(ValueError, xp.fft.fftn, a, s=(5,))

        # axes and s should have the same length
        assert_raises(ValueError, xp.fft.fftn, a, s=(5, 5), axes=(0,))


class TestFftshift:
    @pytest.mark.parametrize("func", ["fftshift", "ifftshift"])
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("axes", [None, 1, (0, 1)])
    def test_basic(self, func, dtype, axes):
        a = generate_random_numpy_array((3, 4), dtype=dtype)
        ia = dpnp.array(a)

        expected = getattr(dpnp.fft, func)(ia, axes=axes)
        result = getattr(numpy.fft, func)(a, axes=axes)
        assert_dtype_allclose(expected, result)


class TestHfft:
    # TODO: include boolean dtype when mkl_fft-gh-180 is merged
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_bool=True)
    )
    @pytest.mark.parametrize("n", [None, 5, 18])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_basic(self, dtype, n, norm):
        a = generate_random_numpy_array(11, dtype)
        ia = dpnp.array(a)

        result = dpnp.fft.hfft(ia, n=n, norm=norm)
        expected = numpy.fft.hfft(a, n=n, norm=norm)
        # check_only_type_kind=True since NumPy always returns float64
        # but dpnp return float32 if input is float32
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_inverse(self, dtype, n, norm):
        a = generate_random_numpy_array(11, dtype)
        ia = dpnp.array(a)

        result = dpnp.fft.ihfft(ia, n=n, norm=norm)
        expected = numpy.fft.ihfft(a, n=n, norm=norm)
        flag = True if numpy_version() < "2.0.0" else False
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    def test_error(self):
        a = dpnp.ones(11)
        # incorrect norm
        assert_raises(ValueError, dpnp.fft.hfft, a, norm="backwards")

    @testing.with_requires("numpy>=2.0.0")
    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_complex_error(self, dtype):
        a = generate_random_numpy_array(11, dtype)
        ia = dpnp.array(a)
        assert_raises(TypeError, dpnp.fft.ihfft, ia)
        assert_raises(TypeError, numpy.fft.ihfft, a)


class TestIrfft:
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    @pytest.mark.parametrize("n", [None, 5, 18])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_basic(self, dtype, n, norm):
        a = generate_random_numpy_array(11)
        ia = dpnp.array(a)

        result = dpnp.fft.irfft(ia, n=n, norm=norm)
        expected = numpy.fft.irfft(a, n=n, norm=norm)
        # check_only_type_kind=True since NumPy always returns float64
        # but dpnp return float32 if input is float32
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 1, 0])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_2d_array(self, dtype, n, axis, norm, order):
        a = generate_random_numpy_array((3, 4), dtype=dtype, order=order)
        ia = dpnp.array(a)

        result = dpnp.fft.irfft(ia, n=n, axis=axis, norm=norm)
        expected = numpy.fft.irfft(a, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_3d_array(self, dtype, n, axis, norm, order):
        a = generate_random_numpy_array((4, 5, 6), dtype, order)
        ia = dpnp.array(a)

        result = dpnp.fft.irfft(ia, n=n, axis=axis, norm=norm)
        expected = numpy.fft.irfft(a, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected, factor=16)

    @pytest.mark.parametrize("n", [None, 5, 17, 18])
    def test_usm_ndarray(self, n):
        a = generate_random_numpy_array(11, dtype=numpy.complex64)
        a_usm = dpt.asarray(a)

        expected = numpy.fft.irfft(a, n=n)
        out = dpt.empty(expected.shape, dtype=a_usm.real.dtype)
        result = dpnp.fft.irfft(a_usm, n=n, out=out)
        assert out is result.get_array()
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 18])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_out(self, dtype, n, norm):
        a = generate_random_numpy_array(11, dtype=dtype)
        ia = dpnp.array(a)

        expected = numpy.fft.irfft(a, n=n, norm=norm)
        out = dpnp.empty(expected.shape, dtype=a.real.dtype)
        result = dpnp.fft.irfft(ia, n=n, norm=norm, out=out)
        assert out is result
        assert_dtype_allclose(result, expected, factor=24)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 0])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_2d_array_out(self, dtype, n, axis, norm, order):
        a = generate_random_numpy_array((3, 4), dtype=dtype, order=order)
        ia = dpnp.array(a)

        expected = numpy.fft.irfft(a, n=n, axis=axis, norm=norm)
        out = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = dpnp.fft.irfft(ia, n=n, axis=axis, norm=norm, out=out)
        assert out is result
        assert_dtype_allclose(result, expected)

    def test_validate_out(self):
        # Invalid dtype for c2r FFT
        a = dpnp.ones((10,), dtype=dpnp.complex64)
        out = dpnp.empty((18,), dtype=dpnp.complex64)
        assert_raises(TypeError, dpnp.fft.irfft, a, out=out)


class TestRfft:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_basic(self, dtype, n, norm):
        a = generate_random_numpy_array(11, dtype, low=-1, high=1)
        ia = dpnp.array(a)

        result = dpnp.fft.rfft(ia, n=n, norm=norm)
        expected = numpy.fft.rfft(a, n=n, norm=norm)
        factor = 120 if dtype in [dpnp.int8, dpnp.uint8] else 8
        assert_dtype_allclose(result, expected, factor=factor)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 1, 0])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_2d_array(self, dtype, n, axis, norm, order):
        a = generate_random_numpy_array((3, 4), dtype=dtype, order=order)
        ia = dpnp.array(a)

        result = dpnp.fft.rfft(ia, n=n, axis=axis, norm=norm)
        expected = numpy.fft.rfft(a, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_3d_array(self, dtype, n, axis, norm, order):
        a = generate_random_numpy_array((2, 3, 4), dtype=dtype, order=order)
        ia = dpnp.array(a)

        result = dpnp.fft.rfft(ia, n=n, axis=axis, norm=norm)
        expected = numpy.fft.rfft(a, n=n, axis=axis, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("n", [None, 5, 20])
    def test_usm_ndarray(self, n):
        a = generate_random_numpy_array(11)
        a_usm = dpt.asarray(a)

        expected = numpy.fft.rfft(a, n=n)
        out_dt = map_dtype_to_device(dpnp.complex128, a_usm.sycl_device)
        out = dpt.empty(expected.shape, dtype=out_dt)

        result = dpnp.fft.rfft(a_usm, n=n, out=out)
        assert out is result.get_array()
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 20])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    def test_out(self, dtype, n, norm):
        a = generate_random_numpy_array(11, dtype=dtype)
        ia = dpnp.array(a)

        expected = numpy.fft.rfft(a, n=n, norm=norm)
        out = dpnp.empty(expected.shape, dtype=expected.dtype)
        result = dpnp.fft.rfft(ia, n=n, norm=norm, out=out)
        assert out is result
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize("n", [None, 5, 8])
    @pytest.mark.parametrize("axis", [-1, 0])
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_2d_array_out(self, dtype, n, axis, norm, order):
        a = generate_random_numpy_array((3, 4), dtype=dtype, order=order)
        ia = dpnp.array(a)

        expected = numpy.fft.rfft(a, n=n, axis=axis, norm=norm)
        out = dpnp.empty(expected.shape, dtype=expected.dtype)

        result = dpnp.fft.rfft(ia, n=n, axis=axis, norm=norm, out=out)
        assert out is result
        assert_dtype_allclose(result, expected)

    @pytest.mark.skipif(not has_support_aspect16(), reason="no fp16 support")
    def test_float16(self):
        a = numpy.arange(10, dtype=numpy.float16)
        ia = dpnp.array(a)

        expected = numpy.fft.rfft(a)
        result = dpnp.fft.rfft(ia)
        # check_only_type_kind=True since Intel NumPy returns complex128
        assert_dtype_allclose(result, expected, check_only_type_kind=True)

    @testing.with_requires("numpy>=2.0.0")
    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
        a = xp.ones((4, 3), dtype=xp.complex64)
        # invalid dtype of input array for r2c FFT
        assert_raises(TypeError, xp.fft.rfft, a)

    def test_validate_out(self):
        # Invalid shape for r2c FFT
        a = dpnp.ones((10,), dtype=dpnp.float32)
        out = dpnp.empty((10,), dtype=dpnp.complex64)
        assert_raises(ValueError, dpnp.fft.rfft, a, out=out)


class TestRfft2:
    # TODO: add other axes when mkl_fft gh-119 is addressed
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize("axes", [(0, 1)])  # (1, 2),(0, 2),(2, 1),(2, 0)
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_basic(self, dtype, axes, norm, order):
        a = generate_random_numpy_array((2, 3, 4), dtype, order)
        ia = dpnp.array(a)

        result = dpnp.fft.rfft2(ia, axes=axes, norm=norm)
        expected = numpy.fft.rfft2(a, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected)

        s = (a.shape[axes[0]], a.shape[axes[1]])
        result = dpnp.fft.irfft2(result, s=s, axes=axes, norm=norm)
        expected = numpy.fft.irfft2(expected, s=s, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_inverse(self, dtype):
        # x is Hermitian symmetric
        x = numpy.array([[0, 1, 2], [5, 4, 6], [5, 7, 6]])
        a = numpy.array(x, dtype=dtype)
        ia = dpnp.array(a)

        result = dpnp.fft.irfft2(ia)
        expected = numpy.fft.irfft2(a)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("s", [None, (3, 3), (10, 10), (3, 10)])
    def test_s(self, s):
        a = generate_random_numpy_array((6, 8))
        ia = dpnp.array(a)

        result = dpnp.fft.rfft2(ia, s=s)
        expected = numpy.fft.rfft2(a, s=s)
        assert_dtype_allclose(result, expected)

        result = dpnp.fft.irfft2(result, s=s)
        expected = numpy.fft.irfft2(expected, s=s)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("xp", [numpy, dpnp])
    def test_error(self, xp):
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
    # TODO: add additional axes when mkl_fft gh-119 is addressed
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_complex=True)
    )
    @pytest.mark.parametrize(
        "axes", [(0, 1, 2), (-2, -4, -1, -3)]  # (-1, -4, -2)
    )
    @pytest.mark.parametrize("norm", [None, "backward", "forward", "ortho"])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_basic(self, dtype, axes, norm, order):
        a = generate_random_numpy_array((2, 3, 4, 5), dtype, order)
        ia = dpnp.array(a)

        result = dpnp.fft.rfftn(ia, axes=axes, norm=norm)
        expected = numpy.fft.rfftn(a, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected)

        # inverse FFT
        s = []
        for axis in axes:
            s.append(a.shape[axis])
        result = dpnp.fft.irfftn(result, s=s, axes=axes, norm=norm)
        expected = numpy.fft.irfftn(expected, s=s, axes=axes, norm=norm)
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize(
        "axes", [(2, 0, 2, 0), (0, 1, 1), (2, 0, 1, 3, 2, 1)]
    )
    def test_repeated_axes(self, axes):
        a = generate_random_numpy_array((2, 3, 4, 5))
        ia = dpnp.array(a)

        result = dpnp.fft.rfftn(ia, axes=axes)
        # Intel NumPy ignores repeated axes (mkl_fft-gh-104), handle it one by one
        expected = numpy.fft.rfft(a, axis=axes[-1])
        # need to pass shape for c2c FFT since expected and a
        # do not have the same shape after calling rfft
        shape = []
        for axis in axes:
            shape.append(a.shape[axis])
        for jj, ii in zip(shape[-2::-1], axes[-2::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected)

        # inverse FFT
        result = dpnp.fft.irfftn(result, axes=axes)
        for ii in axes[-2::-1]:
            expected = numpy.fft.ifft(expected, axis=ii)
        expected = numpy.fft.irfft(expected, axis=axes[-1])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axes", [(2, 3, 3, 2), (0, 0, 3, 3)])
    @pytest.mark.parametrize("s", [(5, 4, 3, 3), (7, 8, 10, 9)])
    def test_repeated_axes_with_s(self, axes, s):
        a = generate_random_numpy_array((2, 3, 4, 5), dtype=numpy.float32)
        ia = dpnp.array(a)

        result = dpnp.fft.rfftn(ia, s=s, axes=axes)
        # Intel NumPy ignores repeated axes (mkl_fft-gh-104), handle it one by one
        expected = numpy.fft.rfft(a, n=s[-1], axis=axes[-1])
        for jj, ii in zip(s[-2::-1], axes[-2::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        assert_dtype_allclose(result, expected)

        result = dpnp.fft.irfftn(result, s=s, axes=axes)
        for jj, ii in zip(s[-2::-1], axes[-2::-1]):
            expected = numpy.fft.ifft(expected, n=jj, axis=ii)
        expected = numpy.fft.irfft(expected, n=s[-1], axis=axes[-1])
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("axes", [(0, 1, 2, 3), (1, 2, 1, 2), (2, 2, 2, 3)])
    @pytest.mark.parametrize("s", [(2, 3, 4, 5), (5, 6, 7, 9), (2, 5, 1, 2)])
    def test_out(self, axes, s):
        a = generate_random_numpy_array((2, 3, 4, 5), dtype=numpy.float32)
        ia = dpnp.array(a)

        # Intel NumPy ignores repeated axes (mkl_fft-gh-104), handle it one by one
        expected = numpy.fft.rfft(a, n=s[-1], axis=axes[-1])
        for jj, ii in zip(s[-2::-1], axes[-2::-1]):
            expected = numpy.fft.fft(expected, n=jj, axis=ii)
        out = dpnp.empty(expected.shape, dtype=numpy.complex64)

        result = dpnp.fft.rfftn(ia, out=out, s=s, axes=axes)
        assert out is result
        assert_dtype_allclose(result, expected)

        # inverse FFT
        for jj, ii in zip(s[-2::-1], axes[-2::-1]):
            expected = numpy.fft.ifft(expected, n=jj, axis=ii)
        expected = numpy.fft.irfft(expected, n=s[-1], axis=axes[-1])
        out = dpnp.empty(expected.shape, dtype=numpy.float32)

        result = dpnp.fft.irfftn(result, out=out, s=s, axes=axes)
        assert out is result
        assert_dtype_allclose(result, expected)

    def test_1d_array(self):
        a = generate_random_numpy_array(20, dtype=numpy.float32)
        ia = dpnp.array(a)

        result = dpnp.fft.rfftn(ia)
        expected = numpy.fft.rfftn(a)
        assert_dtype_allclose(result, expected)

        result = dpnp.fft.irfftn(ia)
        expected = numpy.fft.irfftn(a)
        # TODO: change to the commented line when mkl_fft-gh-180 is merged
        flag = True
        # flag = True if numpy_version() < "2.0.0" else False
        assert_dtype_allclose(result, expected, check_only_type_kind=flag)
