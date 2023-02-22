import pytest
from .helper import get_all_dtypes
import numpy
import dpnp


dtypes_list = get_all_dtypes(no_none=True, no_complex=True)
testdata = []
testdata += [([True, False, True], dtype) for dtype in dtypes_list]
testdata += [
    ([1, -1, 0], dtype)
    for dtype in dtypes_list
    if not numpy.issubdtype(dtype, numpy.bool_)
]
testdata += [
    ([0.1, 0.0, -0.1], dtype)
    for dtype in dtypes_list
    if numpy.issubdtype(dtype, numpy.floating)
]
testdata += [
    ([1j, -1j, 1 - 2j], dtype)
    for dtype in ["complex128"]
    if numpy.complex128 in get_all_dtypes(no_none=True, no_bool=True)
]


@pytest.mark.parametrize("in_obj,out_dtype", testdata)
def test_copyto_dtype(in_obj, out_dtype):
    ndarr = numpy.array(in_obj)
    expected = numpy.empty(ndarr.size, dtype=out_dtype)
    numpy.copyto(expected, ndarr)

    dparr = dpnp.array(in_obj)
    result = dpnp.empty(dparr.size, dtype=out_dtype)
    dpnp.copyto(result, dparr)

    numpy.testing.assert_array_equal(result, expected)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("arr",
                         [[], [1, 2, 3, 4], [[1, 2], [3, 4]], [[[1], [2]], [[3], [4]]]],
                         ids=['[]', '[1, 2, 3, 4]', '[[1, 2], [3, 4]]', '[[[1], [2]], [[3], [4]]]'])
def test_repeat(arr):
    a = numpy.array(arr)
    dpnp_a = dpnp.array(arr)
    expected = numpy.repeat(a, 2)
    result = dpnp.repeat(dpnp_a, 2)
    numpy.testing.assert_array_equal(expected, result)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("array",
                         [[1, 2, 3],
                          [1, 2, 2, 1, 2, 4],
                          [2, 2, 2, 2],
                          []],
                         ids=['[1, 2, 3]',
                              '[1, 2, 2, 1, 2, 4]',
                              '[2, 2, 2, 2]',
                              '[]'])
def test_unique(array):
    np_a = numpy.array(array)
    dpnp_a = dpnp.array(array)

    expected = numpy.unique(np_a)
    result = dpnp.unique(dpnp_a)
    numpy.testing.assert_array_equal(expected, result)
