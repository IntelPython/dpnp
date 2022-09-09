import numpy
import pytest

import dpnp


@pytest.mark.parametrize(
    "res_dtype",
    [
        numpy.float64,
        numpy.float32,
        numpy.int64,
        numpy.int32,
        numpy.bool,
        numpy.bool_,
        numpy.complex,
    ],
    ids=["float64", "float32", "int64", "int32", "bool", "bool_", "complex"],
)
@pytest.mark.parametrize(
    "arr_dtype",
    [
        numpy.float64,
        numpy.float32,
        numpy.int64,
        numpy.int32,
        numpy.bool,
        numpy.bool_,
        numpy.complex,
    ],
    ids=["float64", "float32", "int64", "int32", "bool", "bool_", "complex"],
)
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
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "arr_dtype",
    [
        numpy.float64,
        numpy.float32,
        numpy.int64,
        numpy.int32,
        numpy.bool,
        numpy.bool_,
        numpy.complex,
    ],
    ids=["float64", "float32", "int64", "int32", "bool", "bool_", "complex"],
)
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
    numpy.testing.assert_array_equal(expected, result)
