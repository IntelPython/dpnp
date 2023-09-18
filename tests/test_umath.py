import numpy
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)

import dpnp

from .helper import (
    get_all_dtypes,
    get_complex_dtypes,
    get_float_dtypes,
    has_support_aspect16,
    has_support_aspect64,
)

# full list of umaths
umaths = [i for i in dir(numpy) if isinstance(getattr(numpy, i), numpy.ufunc)]

types = {
    "d": numpy.float64,
    "f": numpy.float32,
    "l": numpy.int64,
    "i": numpy.int32,
}

supported_types = "fli"
if has_support_aspect64():
    supported_types += "d"


def check_types(args_str):
    for s in args_str:
        if s not in supported_types:
            return False
    return True


def shaped_arange(shape, xp=numpy, dtype=numpy.float32):
    size = 1
    for i in shape:
        size = size * i
    array_data = numpy.arange(1, size + 1, 1).tolist()
    return xp.reshape(xp.array(array_data, dtype=dtype), shape)


def get_args(args_str, sh, xp=numpy):
    args = []
    for s in args_str:
        args.append(shaped_arange(shape=sh, xp=xp, dtype=types[s]))
    return tuple(args)


test_cases = []
for umath in umaths:
    np_umath = getattr(numpy, umath)

    for type_ in np_umath.types:
        args_str = type_[: type_.find("->")]
        if check_types(args_str):
            val_ = (umath, args_str)
            if val_ not in test_cases:
                test_cases.append(val_)


def get_id(val):
    return val.__str__()


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.usefixtures("suppress_divide_invalid_numpy_warnings")
@pytest.mark.parametrize("test_cases", test_cases, ids=get_id)
def test_umaths(test_cases):
    umath, args_str = test_cases
    if umath == "matmul":
        sh = (4, 4)
    elif umath == "power":
        sh = (2, 3)
    else:
        sh = (3, 4)

    args = get_args(args_str, sh, xp=numpy)
    iargs = get_args(args_str, sh, xp=dpnp)

    # original
    expected = getattr(numpy, umath)(*args)

    # DPNP
    result = getattr(dpnp, umath)(*iargs)

    assert_allclose(result, expected, rtol=1e-6)


class TestSin:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_sin(self, dtype):
        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_out_dtype = dpnp.float32
        if has_support_aspect64() and dtype != dpnp.float32:
            dp_out_dtype = dpnp.float64

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.sin(dp_array, out=dp_out)

        # original
        expected = numpy.sin(np_array, out=np_out)

        precision = numpy.finfo(dtype=result.dtype).precision
        assert_array_almost_equal(expected, result.asnumpy(), decimal=precision)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_sin_complex(self, dtype):
        np_array = numpy.arange(10, 20, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.complex128)

        # DPNP
        dp_out_dtype = dpnp.complex64
        if has_support_aspect64() and dtype != dpnp.complex64:
            dp_out_dtype = dpnp.complex128

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.sin(dp_array, out=dp_out)

        # original
        expected = numpy.sin(np_array, out=np_out)

        precision = numpy.finfo(dtype=result.dtype).precision
        assert_array_almost_equal(expected, result.asnumpy(), decimal=precision)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.skipif(
        not has_support_aspect16(), reason="No fp16 support by device"
    )
    def test_sin_bool(self):
        np_array = numpy.arange(2, dtype=numpy.bool_)
        np_out = numpy.empty(2, dtype=numpy.float16)

        # DPNP
        dp_array = dpnp.array(np_array, dtype=np_array.dtype)
        dp_out = dpnp.array(np_out, dtype=np_out.dtype)
        result = dpnp.sin(dp_array, out=dp_out)

        # original
        expected = numpy.sin(np_array, out=np_out)
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.sin(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.sin(dp_array, out=dp_out)


class TestSinh:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_sinh(self, dtype):
        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_out_dtype = dpnp.float32
        if has_support_aspect64() and dtype != dpnp.float32:
            dp_out_dtype = dpnp.float64

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.sinh(dp_array, out=dp_out)

        # original
        expected = numpy.sinh(np_array, out=np_out)

        tol = numpy.finfo(dtype=result.dtype).resolution
        assert_allclose(expected, result.asnumpy(), rtol=tol)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_sinh_complex(self, dtype):
        np_array = numpy.arange(10, 20, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.complex128)

        # DPNP
        dp_out_dtype = dpnp.complex64
        if has_support_aspect64() and dtype != dpnp.complex64:
            dp_out_dtype = dpnp.complex128

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.sinh(dp_array, out=dp_out)

        # original
        expected = numpy.sinh(np_array, out=np_out)

        tol = numpy.finfo(dtype=result.dtype).resolution
        assert_allclose(expected, result.asnumpy(), rtol=tol)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.skipif(
        not has_support_aspect16(), reason="No fp16 support by device"
    )
    def test_sinh_bool(self):
        np_array = numpy.arange(2, dtype=numpy.bool_)
        np_out = numpy.empty(2, dtype=numpy.float16)

        # DPNP
        dp_array = dpnp.array(np_array, dtype=np_array.dtype)
        dp_out = dpnp.array(np_out, dtype=np_out.dtype)
        result = dpnp.sinh(dp_array, out=dp_out)

        # original
        expected = numpy.sinh(np_array, out=np_out)
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.sinh(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.sinh(dp_array, out=dp_out)


class TestCos:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_cos(self, dtype):
        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_out_dtype = dpnp.float32
        if has_support_aspect64() and dtype != dpnp.float32:
            dp_out_dtype = dpnp.float64

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.cos(dp_array, out=dp_out)

        # original
        expected = numpy.cos(np_array, out=np_out)

        precision = numpy.finfo(dtype=result.dtype).precision
        assert_array_almost_equal(expected, result.asnumpy(), decimal=precision)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_cos_complex(self, dtype):
        np_array = numpy.arange(10, 20, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.complex128)

        # DPNP
        dp_out_dtype = dpnp.complex64
        if has_support_aspect64() and dtype != dpnp.complex64:
            dp_out_dtype = dpnp.complex128

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.cos(dp_array, out=dp_out)

        # original
        expected = numpy.cos(np_array, out=np_out)

        precision = numpy.finfo(dtype=result.dtype).precision
        assert_array_almost_equal(expected, result.asnumpy(), decimal=precision)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.skipif(
        not has_support_aspect16(), reason="No fp16 support by device"
    )
    def test_cos_bool(self):
        np_array = numpy.arange(2, dtype=numpy.bool_)
        np_out = numpy.empty(2, dtype=numpy.float16)

        # DPNP
        dp_array = dpnp.array(np_array, dtype=np_array.dtype)
        dp_out = dpnp.array(np_out, dtype=np_out.dtype)
        result = dpnp.cos(dp_array, out=dp_out)

        # original
        expected = numpy.cos(np_array, out=np_out)
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.cos(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.cos(dp_array, out=dp_out)


class TestCosh:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_cosh(self, dtype):
        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_out_dtype = dpnp.float32
        if has_support_aspect64() and dtype != dpnp.float32:
            dp_out_dtype = dpnp.float64

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.cosh(dp_array, out=dp_out)

        # original
        expected = numpy.cosh(np_array, out=np_out)

        tol = numpy.finfo(dtype=result.dtype).resolution
        assert_allclose(expected, result.asnumpy(), rtol=tol)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_cosh_complex(self, dtype):
        np_array = numpy.arange(10, 20, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.complex128)

        # DPNP
        dp_out_dtype = dpnp.complex64
        if has_support_aspect64() and dtype != dpnp.complex64:
            dp_out_dtype = dpnp.complex128

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.cosh(dp_array, out=dp_out)

        # original
        expected = numpy.cosh(np_array, out=np_out)

        tol = numpy.finfo(dtype=result.dtype).resolution
        assert_allclose(expected, result.asnumpy(), rtol=tol)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.skipif(
        not has_support_aspect16(), reason="No fp16 support by device"
    )
    def test_cosh_bool(self):
        np_array = numpy.arange(2, dtype=numpy.bool_)
        np_out = numpy.empty(2, dtype=numpy.float16)

        # DPNP
        dp_array = dpnp.array(np_array, dtype=np_array.dtype)
        dp_out = dpnp.array(np_out, dtype=np_out.dtype)
        result = dpnp.cosh(dp_array, out=dp_out)

        # original
        expected = numpy.cosh(np_array, out=np_out)
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.cosh(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.cosh(dp_array, out=dp_out)


class TestsLog:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_log(self, dtype):
        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_out_dtype = dpnp.float32
        if has_support_aspect64() and dtype != dpnp.float32:
            dp_out_dtype = dpnp.float64

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.log(dp_array, out=dp_out)

        # original
        expected = numpy.log(np_array, out=np_out)

        precision = numpy.finfo(dtype=result.dtype).precision
        assert_array_almost_equal(expected, result.asnumpy(), decimal=precision)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_log_complex(self, dtype):
        np_array = numpy.arange(10, 20, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.complex128)

        # DPNP
        dp_out_dtype = dpnp.complex64
        if has_support_aspect64() and dtype != dpnp.complex64:
            dp_out_dtype = dpnp.complex128

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.log(dp_array, out=dp_out)

        # original
        expected = numpy.log(np_array, out=np_out)

        precision = numpy.finfo(dtype=result.dtype).precision
        assert_array_almost_equal(expected, result.asnumpy(), decimal=precision)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.skipif(
        not has_support_aspect16(), reason="No fp16 support by device"
    )
    def test_log_bool(self):
        np_array = numpy.arange(2, dtype=numpy.bool_)
        np_out = numpy.empty(2, dtype=numpy.float16)

        # DPNP
        dp_array = dpnp.array(np_array, dtype=np_array.dtype)
        dp_out = dpnp.array(np_out, dtype=np_out.dtype)
        result = dpnp.log(dp_array, out=dp_out)

        # original
        expected = numpy.log(np_array, out=np_out)
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "dtype",
        [numpy.float32, numpy.int64, numpy.int32],
        ids=["numpy.float32", "numpy.int64", "numpy.int32"],
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.complex64)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.log(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10)
        dp_out = dpnp.empty(shape, dtype=dp_array.dtype)

        with pytest.raises(ValueError):
            dpnp.log(dp_array, out=dp_out)


class TestExp:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_exp(self, dtype):
        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_out_dtype = dpnp.float32
        if has_support_aspect64() and dtype != dpnp.float32:
            dp_out_dtype = dpnp.float64

        dp_array = dpnp.array(np_array, dtype=dp_out_dtype)
        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        result = dpnp.exp(dp_array, out=dp_out)

        # original
        expected = numpy.exp(np_array, out=np_out)

        tol = numpy.finfo(dtype=result.dtype).resolution
        assert_allclose(expected, result.asnumpy(), rtol=tol)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.exp(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.exp(dp_array, out=dp_out)


class TestArccos:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    def test_arccos(self, dtype):
        array_data = numpy.arange(-9, 10, 2) / 10
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.arccos(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.arccos(np_array, out=out)

        tol = numpy.finfo(dtype=result.dtype).resolution
        assert_allclose(expected, result.asnumpy(), rtol=tol)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arccos(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arccos(dp_array, out=dp_out)


class TestArccosh:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    def test_arccosh(self, dtype):
        array_data = numpy.arange(2, 12)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.arccosh(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.arccosh(np_array, out=out)

        tol = numpy.finfo(dtype=result.dtype).resolution
        assert_allclose(expected, result.asnumpy(), rtol=tol)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arccosh(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arccosh(dp_array, out=dp_out)


class TestArcsin:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    def test_arcsin(self, dtype):
        array_data = numpy.arange(-9, 10, 2) / 10
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.arcsin(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.arcsin(np_array, out=out)

        tol = numpy.finfo(dtype=result.dtype).resolution
        assert_allclose(expected, result.asnumpy(), rtol=tol)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arcsin(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arcsin(dp_array, out=dp_out)


class TestArcsinh:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.usefixtures("suppress_invalid_numpy_warnings")
    def test_arcsinh(self, dtype):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.arcsinh(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.arcsinh(np_array, out=out)

        tol = numpy.finfo(dtype=result.dtype).resolution
        assert_allclose(expected, result.asnumpy(), rtol=tol)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arcsinh(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arcsinh(dp_array, out=dp_out)


class TestArctan:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_arctan(self, dtype):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.arctan(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.arctan(np_array, out=out)

        tol = numpy.finfo(dtype).resolution
        assert_allclose(expected, result, tol)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arctan(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arctan(dp_array, out=dp_out)


class TestArctanh:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_arctanh(self, dtype):
        array_data = numpy.arange(-9, 10, 2) / 10
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.arctanh(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.arctanh(np_array, out=out)

        tol = numpy.finfo(dtype).resolution
        assert_allclose(expected, result, tol)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.arctanh(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arctanh(dp_array, out=dp_out)


class TestTan:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_tan(self, dtype):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.tan(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.tan(np_array, out=out)

        tol = numpy.finfo(dtype).resolution
        assert_allclose(expected, result, rtol=tol)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_complex=True, no_none=True)[:-1]
    )
    def test_invalid_dtype(self, dtype):
        dpnp_dtype = get_all_dtypes(no_complex=True, no_none=True)[-1]
        dp_array = dpnp.arange(10, dtype=dpnp_dtype)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.tan(dp_array, out=dp_out)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.tan(dp_array, out=dp_out)


class TestArctan2:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_arctan2(self, dtype):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.arctan2(dp_array, dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.arctan2(np_array, np_array, out=out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True)
    )
    def test_out_dtypes(self, dtype):
        if has_support_aspect64() and dtype != numpy.float32:
            dtype_out = numpy.float64
        else:
            dtype_out = numpy.float32
        size = 2 if dtype == dpnp.bool else 10

        np_array = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=dtype_out)
        expected = numpy.arctan2(np_array, np_array, out=np_out)

        dp_array = dpnp.arange(size, dtype=dtype)
        dp_out = dpnp.empty(size, dtype=dtype_out)
        result = dpnp.arctan2(dp_array, dp_array, out=dp_out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape, dtype):
        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(shape, dtype=dtype)

        with pytest.raises(ValueError):
            dpnp.arctan2(dp_array, dp_array, out=dp_out)


class TestSqrt:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_complex=True)
    )
    def test_sqrt_int_float(self, dtype):
        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_out_dtype = dpnp.float32
        if has_support_aspect64() and dtype != dpnp.float32:
            dp_out_dtype = dpnp.float64

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.sqrt(dp_array, out=dp_out)

        # original
        expected = numpy.sqrt(np_array, out=np_out)
        assert_allclose(expected, result)

    @pytest.mark.parametrize("dtype", get_complex_dtypes())
    def test_sqrt_complex(self, dtype):
        np_array = numpy.arange(10, 20, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.complex128)

        # DPNP
        dp_out_dtype = dpnp.complex64
        if has_support_aspect64() and dtype != dpnp.complex64:
            dp_out_dtype = dpnp.complex128

        dp_out = dpnp.array(np_out, dtype=dp_out_dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.sqrt(dp_array, out=dp_out)

        # original
        expected = numpy.sqrt(np_array, out=np_out)
        assert_allclose(expected, result)

    @pytest.mark.usefixtures("suppress_divide_numpy_warnings")
    @pytest.mark.skipif(
        not has_support_aspect16(), reason="No fp16 support by device"
    )
    def test_sqrt_bool(self):
        np_array = numpy.arange(2, dtype=numpy.bool_)
        np_out = numpy.empty(2, dtype=numpy.float16)

        # DPNP
        dp_array = dpnp.array(np_array, dtype=np_array.dtype)
        dp_out = dpnp.array(np_out, dtype=np_out.dtype)
        result = dpnp.sqrt(dp_array, out=dp_out)

        # original
        expected = numpy.sqrt(np_array, out=np_out)
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "dtype", [numpy.int64, numpy.int32], ids=["numpy.int64", "numpy.int32"]
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.arange(10, dtype=dpnp.float32)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.sqrt(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10, dtype=dpnp.float32)
        dp_out = dpnp.empty(shape, dtype=dpnp.float32)

        with pytest.raises(ValueError):
            dpnp.sqrt(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        numpy.testing.assert_raises(TypeError, dpnp.sqrt, a, out)
        numpy.testing.assert_raises(TypeError, numpy.sqrt, a.asnumpy(), out)


class TestSquare:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_square(self, dtype):
        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_out = dpnp.array(np_out, dtype=dtype)
        dp_array = dpnp.array(np_array, dtype=dtype)
        result = dpnp.square(dp_array, out=dp_out)

        # original
        expected = numpy.square(np_array, out=np_out)
        assert_allclose(expected, result)

    def test_square_bool(self):
        np_array = numpy.arange(2, dtype=numpy.bool_)
        np_out = numpy.empty(2, dtype=numpy.int8)

        # DPNP
        dp_array = dpnp.array(np_array, dtype=np_array.dtype)
        dp_out = dpnp.array(np_out, dtype=np_out.dtype)
        result = dpnp.square(dp_array, out=dp_out)

        # original
        expected = numpy.square(np_array, out=np_out)
        assert_allclose(expected, result)

    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_bool=True, no_none=True)
    )
    def test_invalid_dtype(self, dtype):
        dp_array = dpnp.ones(10, dtype=dpnp.bool)
        dp_out = dpnp.empty(10, dtype=dtype)

        with pytest.raises(TypeError):
            dpnp.square(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "shape", [(0,), (15,), (2, 2)], ids=["(0,)", "(15, )", "(2,2)"]
    )
    def test_invalid_shape(self, shape):
        dp_array = dpnp.arange(10, dtype=dpnp.float32)
        dp_out = dpnp.empty(shape, dtype=dpnp.float32)

        with pytest.raises(ValueError):
            dpnp.square(dp_array, out=dp_out)

    @pytest.mark.parametrize(
        "out",
        [4, (), [], (3, 7), [2, 4]],
        ids=["4", "()", "[]", "(3, 7)", "[2, 4]"],
    )
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        numpy.testing.assert_raises(TypeError, dpnp.square, a, out)
        numpy.testing.assert_raises(TypeError, numpy.square, a.asnumpy(), out)
