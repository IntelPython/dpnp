import pytest
from .helper import get_all_dtypes

import dpnp
import dpctl.tensor as dpt

import numpy
from numpy.testing import (
    assert_array_equal
)


@pytest.mark.parametrize("res_dtype", get_all_dtypes())
@pytest.mark.parametrize("arr_dtype", get_all_dtypes())
@pytest.mark.parametrize("arr",
                         [[-2, -1, 0, 1, 2], [[-2, -1], [1, 2]], []],
                         ids=['[-2, -1, 0, 1, 2]', '[[-2, -1], [1, 2]]', '[]'])
def test_astype(arr, arr_dtype, res_dtype):
    numpy_array = numpy.array(arr, dtype=arr_dtype)
    dpnp_array = dpnp.array(numpy_array)
    expected = numpy_array.astype(res_dtype)
    result = dpnp_array.astype(res_dtype)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("arr_dtype", get_all_dtypes())
@pytest.mark.parametrize("arr",
                         [[-2, -1, 0, 1, 2], [[-2, -1], [1, 2]], []],
                         ids=['[-2, -1, 0, 1, 2]', '[[-2, -1], [1, 2]]', '[]'])
def test_flatten(arr, arr_dtype):
    numpy_array = numpy.array(arr, dtype=arr_dtype)
    dpnp_array = dpnp.array(arr, dtype=arr_dtype)
    expected = numpy_array.flatten()
    result = dpnp_array.flatten()
    assert_array_equal(expected, result)


@pytest.mark.parametrize("shape",
                         [(), 0, (0,), (2), (5, 2), (5, 0, 2), (5, 3, 2)],
                         ids=['()', '0', '(0,)', '(2)', '(5, 2)', '(5, 0, 2)', '(5, 3, 2)'])
@pytest.mark.parametrize("order",
                         ["C", "F"],
                         ids=['C', 'F'])
def test_flags(shape, order):
    usm_array = dpt.usm_ndarray(shape, order=order)
    numpy_array = numpy.ndarray(shape, order=order)
    dpnp_array = dpnp.ndarray(shape, order=order)
    assert usm_array.flags == dpnp_array.flags
    assert numpy_array.flags.c_contiguous == dpnp_array.flags.c_contiguous
    assert numpy_array.flags.f_contiguous == dpnp_array.flags.f_contiguous


@pytest.mark.parametrize("dtype",
                         [numpy.complex64, numpy.float32, numpy.int64, numpy.int32, numpy.bool_],
                         ids=['complex64', 'float32', 'int64', 'int32', 'bool'])
@pytest.mark.parametrize("strides",
                         [(1, 4) , (4, 1)],
                         ids=['(1, 4)', '(4, 1)'])
@pytest.mark.parametrize("order",
                         ["C", "F"],
                         ids=['C', 'F'])
def test_flags_strides(dtype, order, strides):
    itemsize = numpy.dtype(dtype).itemsize
    numpy_strides = tuple([el * itemsize for el in strides])
    usm_array = dpt.usm_ndarray((4, 4), dtype=dtype, order=order, strides=strides)
    numpy_array = numpy.ndarray((4, 4), dtype=dtype, order=order, strides=numpy_strides)
    dpnp_array = dpnp.ndarray((4, 4), dtype=dtype, order=order, strides=strides)
    assert usm_array.flags == dpnp_array.flags
    assert numpy_array.flags.c_contiguous == dpnp_array.flags.c_contiguous
    assert numpy_array.flags.f_contiguous == dpnp_array.flags.f_contiguous


@pytest.mark.parametrize("func", [bool, float, int, complex])
@pytest.mark.parametrize("shape", [tuple(), (1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False, no_complex=True))
def test_scalar_type_casting(func, shape, dtype):
    numpy_array = numpy.full(shape, 5, dtype=dtype)
    dpnp_array = dpnp.full(shape, 5, dtype=dtype)
    assert func(numpy_array) == func(dpnp_array)


@pytest.mark.parametrize("method", ["__bool__", "__float__", "__int__", "__complex__"])
@pytest.mark.parametrize("shape", [tuple(), (1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_float16=False, no_complex=True, no_none=True))
def test_scalar_type_casting_by_method(method, shape, dtype):
    numpy_array = numpy.full(shape, 4.7, dtype=dtype)
    dpnp_array = dpnp.full(shape, 4.7, dtype=dtype)
    assert getattr(numpy_array, method)() == getattr(dpnp_array, method)()


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize("index_dtype", [dpnp.int32, dpnp.int64])
def test_array_as_index(shape, index_dtype):
    ind_arr = dpnp.ones(shape, dtype=index_dtype)
    a = numpy.arange(ind_arr.size + 1)
    assert a[tuple(ind_arr)] == a[1]
