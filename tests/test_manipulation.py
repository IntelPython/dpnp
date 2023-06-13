import pytest

import numpy
from numpy.testing import (
    assert_array_equal
)

import dpnp


testdata = []
testdata += [([True, False, True], dtype) for dtype in ['float32', 'float64', 'int32', 'int64', 'bool']]
testdata += [([1, -1, 0], dtype) for dtype in ['float32', 'float64', 'int32', 'int64']]
testdata += [([0.1, 0.0, -0.1], dtype) for dtype in ['float32', 'float64']]
testdata += [([1j, -1j, 1 - 2j], dtype) for dtype in ['complex128']]


@pytest.mark.parametrize("in_obj,out_dtype", testdata)
def test_copyto_dtype(in_obj, out_dtype):
    ndarr = numpy.array(in_obj)
    expected = numpy.empty(ndarr.size, dtype=out_dtype)
    numpy.copyto(expected, ndarr)

    dparr = dpnp.array(in_obj)
    result = dpnp.empty(dparr.size, dtype=out_dtype)
    dpnp.copyto(result, dparr)

    assert_array_equal(result, expected)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize("arr",
                         [[], [1, 2, 3, 4], [[1, 2], [3, 4]], [[[1], [2]], [[3], [4]]]],
                         ids=['[]', '[1, 2, 3, 4]', '[[1, 2], [3, 4]]', '[[[1], [2]], [[3], [4]]]'])
def test_repeat(arr):
    a = numpy.array(arr)
    dpnp_a = dpnp.array(arr)
    expected = numpy.repeat(a, 2)
    result = dpnp.repeat(dpnp_a, 2)
    assert_array_equal(expected, result)


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
    assert_array_equal(expected, result)


class TestTranspose:
    @pytest.mark.parametrize("axes", [(0, 1), (1, 0)])
    def test_2d_with_axes(self, axes):
        na = numpy.array([[1, 2], [3, 4]])
        da = dpnp.array(na)

        expected = numpy.transpose(na, axes)
        result = dpnp.transpose(da, axes)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("axes", [(1, 0, 2), ((1, 0, 2),)])
    def test_3d_with_packed_axes(self, axes):
        na = numpy.ones((1, 2, 3))
        da = dpnp.array(na)

        expected = na.transpose(*axes)
        result = da.transpose(*axes)
        assert_array_equal(expected, result)

    @pytest.mark.parametrize("shape", [(10,), (2, 4), (5, 3, 7), (3, 8, 4, 1)])
    def test_none_axes(self, shape):
        na = numpy.ones(shape)
        da = dpnp.ones(shape)

        assert_array_equal(na.transpose(), da.transpose())
        assert_array_equal(na.transpose(None), da.transpose(None))
