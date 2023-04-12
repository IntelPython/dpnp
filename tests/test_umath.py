import pytest
from .helper import (
    get_all_dtypes,
    get_float_dtypes
)

import numpy
from numpy.testing import (
    assert_allclose,
    assert_array_equal
)

import dpnp

# full list of umaths
umaths = [i for i in dir(numpy) if isinstance(getattr(numpy, i), numpy.ufunc)]
# print(umaths)
umaths = ['equal']
# trigonometric
umaths.extend(['arccos', 'arcsin', 'arctan', 'cos', 'deg2rad', 'degrees',
               'rad2deg', 'radians', 'sin', 'tan', 'arctan2', 'hypot'])
# 'unwrap'

types = {
    'd': numpy.float64,
    'f': numpy.float32,
    'l': numpy.int64,
    'i': numpy.int32,
}

supported_types = 'dfli'


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


def get_args(args_str, xp=numpy):
    args = []
    for s in args_str:
        args.append(shaped_arange(shape=(3, 4), xp=xp, dtype=types[s]))
    return tuple(args)


test_cases = []
for umath in umaths:
    np_umath = getattr(numpy, umath)
    _types = np_umath.types
    for type in _types:
        args_str = type[:type.find('->')]
        if check_types(args_str):
            test_cases.append((umath, args_str))


def get_id(val):
    return val.__str__()


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize('test_cases', test_cases, ids=get_id)
def test_umaths(test_cases):
    umath, args_str = test_cases
    args = get_args(args_str, xp=numpy)
    iargs = get_args(args_str, xp=dpnp)

    # original
    expected = getattr(numpy, umath)(*args)

    # DPNP
    result = getattr(dpnp, umath)(*iargs)

    assert_allclose(result, expected, rtol=1e-6)


class TestSin:

    def test_sin_ordinary(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.sin(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.sin(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
    @pytest.mark.parametrize("dtype_out", get_float_dtypes())
    def test_out_dtype(self, dtype, dtype_out):

        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=dtype_out)
        expected = numpy.sin(np_array, np_out)

        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(10, dtype=dtype_out)
        result = dpnp.sin(dp_array, dp_out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("shape",
                             [(0,), (15, ), (2, 2)],
                             ids=['(0,)', '(15, )', '(2,2)'])
    def test_invalid_shape(self, shape):

        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.sin(dp_array, out=dp_out)


class TestCos:

    def test_cos(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.cos(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.cos(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
    @pytest.mark.parametrize("dtype_out", get_float_dtypes())
    def test_out_dtype(self, dtype, dtype_out):

        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=dtype_out)
        expected = numpy.cos(np_array, np_out)

        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(10, dtype=dtype_out)
        result = dpnp.cos(dp_array, dp_out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("shape",
                             [(0,), (15, ), (2, 2)],
                             ids=['(0,)', '(15, )', '(2,2)'])
    def test_invalid_shape(self, shape):

        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.cos(dp_array, out=dp_out)


class TestLog:

    def test_log(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.log(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.log(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
    @pytest.mark.parametrize("dtype_out", get_float_dtypes())
    def test_out_dtype(self, dtype, dtype_out):

        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=dtype_out)
        expected = numpy.log(np_array, np_out)

        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(10, dtype=dtype_out)
        result = dpnp.log(dp_array, dp_out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("shape",
                             [(0,), (15, ), (2, 2)],
                             ids=['(0,)', '(15, )', '(2,2)'])
    def test_invalid_shape(self, shape):

        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.log(dp_array, out=dp_out)


class TestExp:

    def test_exp(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.exp(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.exp(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
    @pytest.mark.parametrize("dtype_out", get_float_dtypes())
    def test_out_dtype(self, dtype, dtype_out):

        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=dtype_out)
        expected = numpy.exp(np_array, np_out)

        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(10, dtype=dtype_out)
        result = dpnp.exp(dp_array, dp_out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("shape",
                             [(0,), (15, ), (2, 2)],
                             ids=['(0,)', '(15, )', '(2,2)'])
    def test_invalid_shape(self, shape):

        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.exp(dp_array, out=dp_out)


class TestArcsin:

    def test_arcsin(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.arcsin(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.arcsin(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
    @pytest.mark.parametrize("dtype_out", get_float_dtypes())
    def test_out_dtype(self, dtype, dtype_out):

        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=dtype_out)
        expected = numpy.arcsin(np_array, np_out)

        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(10, dtype=dtype_out)
        result = dpnp.arcsin(dp_array, dp_out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("shape",
                             [(0,), (15, ), (2, 2)],
                             ids=['(0,)', '(15, )', '(2,2)'])
    def test_invalid_shape(self, shape):

        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.arcsin(dp_array, out=dp_out)


class TestArctan:

    def test_arctan(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.arctan(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.arctan(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
    @pytest.mark.parametrize("dtype_out", get_float_dtypes())
    def test_out_dtype(self, dtype, dtype_out):

        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=dtype_out)
        expected = numpy.arctan(np_array, np_out)

        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(10, dtype=dtype_out)
        result = dpnp.arctan(dp_array, dp_out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("shape",
                             [(0,), (15, ), (2, 2)],
                             ids=['(0,)', '(15, )', '(2,2)'])
    def test_invalid_shape(self, shape):

        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.arctan(dp_array, out=dp_out)


class TestTan:

    def test_tan(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.tan(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.tan(np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
    @pytest.mark.parametrize("dtype_out", get_float_dtypes())
    def test_out_dtype(self, dtype, dtype_out):

        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=dtype_out)
        expected = numpy.tan(np_array, np_out)

        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(10, dtype=dtype_out)
        result = dpnp.tan(dp_array, dp_out)

        assert_allclose(expected, result, rtol=1e-06)

    @pytest.mark.parametrize("shape",
                             [(0,), (15, ), (2, 2)],
                             ids=['(0,)', '(15, )', '(2,2)'])
    def test_invalid_shape(self, shape):

        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.tan(dp_array, out=dp_out)


class TestArctan2:
    def test_arctan2(self):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=numpy.float64)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dpnp.float64)
        dp_out = dpnp.array(out, dtype=dpnp.float64)
        result = dpnp.arctan2(dp_array, dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=numpy.float64)
        expected = numpy.arctan2(np_array, np_array, out=out)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
    def test_out_dtypes(self, dtype):
        size = 2 if dtype == dpnp.bool else 10

        np_array = numpy.arange(size, dtype=dtype)
        np_out = numpy.empty(size, dtype=numpy.complex64)
        expected = numpy.arctan2(np_array, np_array, out=np_out)

        dp_array = dpnp.arange(size, dtype=dtype)
        dp_out = dpnp.empty(size, dtype=dpnp.complex64)
        result = dpnp.arctan2(dp_array, dp_array, out=dp_out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("shape",
                             [(0,), (15, ), (2, 2)],
                             ids=['(0,)', '(15, )', '(2,2)'])
    def test_invalid_shape(self, shape):

        dp_array = dpnp.arange(10, dtype=dpnp.float64)
        dp_out = dpnp.empty(shape, dtype=dpnp.float64)

        with pytest.raises(ValueError):
            dpnp.arctan2(dp_array, dp_array, out=dp_out)


class TestSqrt:
    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_sqrt_ordinary(self, dtype):
        array_data = numpy.arange(10)
        out = numpy.empty(10, dtype=dtype)

        # DPNP
        dp_array = dpnp.array(array_data, dtype=dtype)
        dp_out = dpnp.array(out, dtype=dtype)
        result = dpnp.sqrt(dp_array, out=dp_out)

        # original
        np_array = numpy.array(array_data, dtype=dtype)
        expected = numpy.sqrt(np_array, out=out)

        numpy.testing.assert_allclose(expected, result)
        numpy.testing.assert_allclose(out, dp_out)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True, no_none=True))
    def test_out_dtype(self, dtype):

        np_array = numpy.arange(10, dtype=dtype)
        np_out = numpy.empty(10, dtype=numpy.float32)
        expected = numpy.sqrt(np_array, np_out)

        dp_array = dpnp.arange(10, dtype=dtype)
        dp_out = dpnp.empty(10, dtype=dpnp.float32)
        result = dpnp.sqrt(dp_array, dp_out)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("dtype", get_float_dtypes())
    def test_out_equal_array(self, dtype):

        np_array = numpy.arange(10, dtype=dtype)
        expected = numpy.sqrt(np_array, np_array)

        dp_array = dpnp.arange(10, dtype=dtype)
        result = dpnp.sqrt(dp_array, dp_array)

        assert_allclose(expected, result)

    @pytest.mark.parametrize("shape",
                             [(0,), (15, ), (2, 2)],
                             ids=['(0,)', '(15, )', '(2,2)'])
    def test_invalid_shape(self, shape):

        dp_array = dpnp.arange(10, dtype=dpnp.float32)
        dp_out = dpnp.empty(shape, dtype=dpnp.float32)

        with pytest.raises(ValueError):
            dpnp.sqrt(dp_array, out=dp_out)

    @pytest.mark.parametrize("out",
                             [4, (), [], (3, 7), [2, 4]],
                             ids=['4', '()', '[]', '(3, 7)', '[2, 4]'])
    def test_invalid_out(self, out):
        a = dpnp.arange(10)

        numpy.testing.assert_raises(TypeError, dpnp.sqrt, a, out)
        numpy.testing.assert_raises(TypeError, numpy.sqrt, a.asnumpy(), out)
