import dpctl.tensor as dpt
import numpy
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import dpnp

from .helper import (
    get_all_dtypes,
    get_complex_dtypes,
    get_float_dtypes,
    has_support_aspect64,
)


@pytest.mark.usefixtures("suppress_complex_warning")
@pytest.mark.parametrize("res_dtype", get_all_dtypes())
@pytest.mark.parametrize("arr_dtype", get_all_dtypes())
@pytest.mark.parametrize(
    "arr",
    [[-2, -1, 0, 1, 2], [[-2, -1], [1, 2]], []],
    ids=["[-2, -1, 0, 1, 2]", "[[-2, -1], [1, 2]]", "[]"],
)
def test_astype(arr, arr_dtype, res_dtype):
    numpy_array = numpy.array(arr, dtype=arr_dtype)
    dpnp_array = dpnp.array(numpy_array)
    expected = numpy_array.astype(res_dtype)
    result = dpnp_array.astype(res_dtype)
    assert_allclose(expected, result)


def test_astype_subok_error():
    x = dpnp.ones((4))
    with pytest.raises(NotImplementedError):
        x.astype("i4", subok=False)


@pytest.mark.parametrize("arr_dtype", get_all_dtypes())
@pytest.mark.parametrize(
    "arr",
    [[-2, -1, 0, 1, 2], [[-2, -1], [1, 2]], []],
    ids=["[-2, -1, 0, 1, 2]", "[[-2, -1], [1, 2]]", "[]"],
)
def test_flatten(arr, arr_dtype):
    numpy_array = numpy.array(arr, dtype=arr_dtype)
    dpnp_array = dpnp.array(arr, dtype=arr_dtype)
    expected = numpy_array.flatten()
    result = dpnp_array.flatten()
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "shape",
    [(), 0, (0,), (2), (5, 2), (5, 0, 2), (5, 3, 2)],
    ids=["()", "0", "(0,)", "(2)", "(5, 2)", "(5, 0, 2)", "(5, 3, 2)"],
)
@pytest.mark.parametrize("order", ["C", "F"], ids=["C", "F"])
def test_flags(shape, order):
    usm_array = dpt.usm_ndarray(shape, order=order)
    numpy_array = numpy.ndarray(shape, order=order)
    dpnp_array = dpnp.ndarray(shape, order=order)
    assert usm_array.flags == dpnp_array.flags
    assert numpy_array.flags.c_contiguous == dpnp_array.flags.c_contiguous
    assert numpy_array.flags.f_contiguous == dpnp_array.flags.f_contiguous


@pytest.mark.parametrize(
    "dtype",
    [numpy.complex64, numpy.float32, numpy.int64, numpy.int32, numpy.bool_],
    ids=["complex64", "float32", "int64", "int32", "bool"],
)
@pytest.mark.parametrize("strides", [(1, 4), (4, 1)], ids=["(1, 4)", "(4, 1)"])
@pytest.mark.parametrize("order", ["C", "F"], ids=["C", "F"])
def test_flags_strides(dtype, order, strides):
    itemsize = numpy.dtype(dtype).itemsize
    numpy_strides = tuple([el * itemsize for el in strides])
    usm_array = dpt.usm_ndarray(
        (4, 4), dtype=dtype, order=order, strides=strides
    )
    numpy_array = numpy.ndarray(
        (4, 4), dtype=dtype, order=order, strides=numpy_strides
    )
    dpnp_array = dpnp.ndarray((4, 4), dtype=dtype, order=order, strides=strides)
    assert usm_array.flags == dpnp_array.flags
    assert numpy_array.flags.c_contiguous == dpnp_array.flags.c_contiguous
    assert numpy_array.flags.f_contiguous == dpnp_array.flags.f_contiguous


def test_print_dpnp_int():
    result = repr(dpnp.array([1, 0, 2, -3, -1, 2, 21, -9], dtype="i4"))
    expected = "array([ 1,  0,  2, -3, -1,  2, 21, -9], dtype=int32)"
    assert result == expected

    result = str(dpnp.array([1, 0, 2, -3, -1, 2, 21, -9], dtype="i4"))
    expected = "[ 1  0  2 -3 -1  2 21 -9]"
    assert result == expected
    # int32
    result = repr(dpnp.array([1, -1, 21], dtype=dpnp.int32))
    expected = "array([ 1, -1, 21], dtype=int32)"
    assert result == expected

    result = str(dpnp.array([1, -1, 21], dtype=dpnp.int32))
    expected = "[ 1 -1 21]"
    assert result == expected
    # uint8
    result = repr(dpnp.array([1, 0, 3], dtype=numpy.uint8))
    expected = "array([1, 0, 3], dtype=uint8)"
    assert result == expected

    result = str(dpnp.array([1, 0, 3], dtype=numpy.uint8))
    expected = "[1 0 3]"
    assert result == expected


@pytest.mark.parametrize("dtype", get_float_dtypes())
def test_print_dpnp_float(dtype):
    result = repr(dpnp.array([1, -1, 21], dtype=dtype))
    expected = "array([ 1., -1., 21.])"
    if dtype is dpnp.float32:
        expected = expected[:-1] + ", dtype=float32)"

    result = str(dpnp.array([1, -1, 21], dtype=dtype))
    expected = "[ 1. -1. 21.]"
    assert result == expected


@pytest.mark.parametrize("dtype", get_complex_dtypes())
def test_print_dpnp_complex(dtype):
    result = repr(dpnp.array([1, -1, 21], dtype=dtype))
    expected = "array([ 1.+0.j, -1.+0.j, 21.+0.j])"
    if dtype is dpnp.complex64:
        expected = expected[:-1] + ", dtype=complex64)"
    assert result == expected

    result = str(dpnp.array([1, -1, 21], dtype=dtype))
    expected = "[ 1.+0.j -1.+0.j 21.+0.j]"
    assert result == expected


def test_print_dpnp_boolean():
    result = repr(dpnp.array([1, 0, 3], dtype=bool))
    expected = "array([ True, False,  True])"
    assert result == expected

    result = str(dpnp.array([1, 0, 3], dtype=bool))
    expected = "[ True False  True]"
    assert result == expected


@pytest.mark.parametrize(
    "character", [dpnp.nan, dpnp.inf], ids=["dpnp.nan", "dpnp.inf"]
)
def test_print_dpnp_special_character(character):
    result = repr(dpnp.array([1.0, 0.0, character, 3.0]))
    expected = f"array([ 1.,  0., {character},  3.])"
    if not has_support_aspect64():
        expected = expected[:-1] + ", dtype=float32)"
    assert result == expected

    result = str(dpnp.array([1.0, 0.0, character, 3.0]))
    expected = f"[ 1.  0. {character}  3.]"
    assert result == expected


def test_print_dpnp_1d():
    dtype = dpnp.default_float_type()
    result = repr(dpnp.arange(10000, dtype=dtype))
    expected = "array([0.000e+00, 1.000e+00, 2.000e+00, ..., 9.997e+03, 9.998e+03,\n       9.999e+03])"
    if not has_support_aspect64():
        expected = expected[:-1] + ", dtype=float32)"
    assert result == expected

    result = str(dpnp.arange(10000, dtype=dtype))
    expected = (
        "[0.000e+00 1.000e+00 2.000e+00 ... 9.997e+03 9.998e+03 9.999e+03]"
    )
    assert result == expected


def test_print_dpnp_2d():
    dtype = dpnp.default_float_type()
    result = repr(dpnp.array([[1, 2], [3, 4]], dtype=dtype))
    expected = "array([[1., 2.],\n       [3., 4.]])"
    if not has_support_aspect64():
        expected = expected[:-1] + ", dtype=float32)"
    assert result == expected

    result = str(dpnp.array([[1, 2], [3, 4]]))
    expected = "[[1 2]\n [3 4]]"
    assert result == expected


def test_print_dpnp_zero_shape():
    result = repr(dpnp.empty(shape=(0, 0)))
    if has_support_aspect64():
        expected = "array([])"
    else:
        expected = "array([], dtype=float32)"
    assert result == expected

    result = str(dpnp.empty(shape=(0, 0)))
    expected = "[]"
    assert result == expected


@pytest.mark.parametrize("func", [bool, float, int, complex])
@pytest.mark.parametrize("shape", [tuple(), (1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_float16=False, no_complex=True)
)
def test_scalar_type_casting(func, shape, dtype):
    numpy_array = numpy.full(shape, 5, dtype=dtype)
    dpnp_array = dpnp.full(shape, 5, dtype=dtype)
    assert func(numpy_array) == func(dpnp_array)


@pytest.mark.parametrize(
    "method", ["__bool__", "__float__", "__int__", "__complex__"]
)
@pytest.mark.parametrize("shape", [tuple(), (1,), (1, 1), (1, 1, 1)])
@pytest.mark.parametrize(
    "dtype", get_all_dtypes(no_float16=False, no_complex=True, no_none=True)
)
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


def test_ravel():
    a = dpnp.ones((2, 2))
    b = a.ravel()
    a[0, 0] = 5
    assert_array_equal(a.ravel(), b)


def test_repeat():
    numpy_array = numpy.arange(4).repeat(3)
    dpnp_array = dpnp.arange(4).repeat(3)
    assert_array_equal(numpy_array, dpnp_array)
