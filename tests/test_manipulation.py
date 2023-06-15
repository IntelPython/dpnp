import pytest
import numpy
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
