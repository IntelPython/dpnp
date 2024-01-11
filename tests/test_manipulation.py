import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp

from .helper import (
    get_all_dtypes,
    get_complex_dtypes,
    get_float_dtypes,
    has_support_aspect64,
)

testdata = []
testdata += [
    ([True, False, True], dtype)
    for dtype in get_all_dtypes(no_none=True, no_complex=True)
]
testdata += [
    ([1, -1, 0], dtype)
    for dtype in get_all_dtypes(no_none=True, no_bool=True, no_complex=True)
]
testdata += [([0.1, 0.0, -0.1], dtype) for dtype in get_float_dtypes()]
testdata += [([1j, -1j, 1 - 2j], dtype) for dtype in get_complex_dtypes()]


@pytest.mark.parametrize("in_obj, out_dtype", testdata)
def test_copyto_dtype(in_obj, out_dtype):
    ndarr = numpy.array(in_obj)
    expected = numpy.empty(ndarr.size, dtype=out_dtype)
    numpy.copyto(expected, ndarr)

    dparr = dpnp.array(in_obj)
    result = dpnp.empty(dparr.size, dtype=out_dtype)
    dpnp.copyto(result, dparr)

    assert_array_equal(result, expected)


@pytest.mark.parametrize("dst", [7, numpy.ones(10), (2, 7), [5], range(3)])
def test_copyto_dst_raises(dst):
    a = dpnp.array(4)
    with pytest.raises(
        TypeError,
        match="Destination array must be any of supported type, but got",
    ):
        dpnp.copyto(dst, a)


@pytest.mark.parametrize("where", [numpy.ones(10), (2, 7), [5], range(3)])
def test_copyto_where_raises(where):
    a = dpnp.empty((2, 3))
    b = dpnp.arange(6).reshape((2, 3))

    with pytest.raises(
        TypeError, match="`where` array must be any of supported type, but got"
    ):
        dpnp.copyto(a, b, where=where)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize(
    "arr",
    [[], [1, 2, 3, 4], [[1, 2], [3, 4]], [[[1], [2]], [[3], [4]]]],
    ids=["[]", "[1, 2, 3, 4]", "[[1, 2], [3, 4]]", "[[[1], [2]], [[3], [4]]]"],
)
def test_repeat(arr):
    a = numpy.array(arr)
    dpnp_a = dpnp.array(arr)
    expected = numpy.repeat(a, 2)
    result = dpnp.repeat(dpnp_a, 2)
    assert_array_equal(expected, result)


# TODO: Temporary skipping the test, until Internal CI is updated with
# recent changed in dpctl regarding dpt.result_type function
@pytest.mark.skip("Temporary skipping the test")
def test_result_type():
    X = [dpnp.ones((2), dtype=dpnp.int64), dpnp.int32, "float32"]
    X_np = [numpy.ones((2), dtype=numpy.int64), numpy.int32, "float32"]

    if has_support_aspect64():
        assert dpnp.result_type(*X) == numpy.result_type(*X_np)
    else:
        assert dpnp.result_type(*X) == dpnp.default_float_type(X[0].device)


def test_result_type_only_dtypes():
    X = [dpnp.int64, dpnp.int32, dpnp.bool, dpnp.float32]
    X_np = [numpy.int64, numpy.int32, numpy.bool_, numpy.float32]

    assert dpnp.result_type(*X) == numpy.result_type(*X_np)


def test_result_type_only_arrays():
    X = [dpnp.ones((2), dtype=dpnp.int64), dpnp.ones((7, 4), dtype=dpnp.int32)]
    X_np = [
        numpy.ones((2), dtype=numpy.int64),
        numpy.ones((7, 4), dtype=numpy.int32),
    ]

    assert dpnp.result_type(*X) == numpy.result_type(*X_np)


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
@pytest.mark.parametrize(
    "array",
    [[1, 2, 3], [1, 2, 2, 1, 2, 4], [2, 2, 2, 2], []],
    ids=["[1, 2, 3]", "[1, 2, 2, 1, 2, 4]", "[2, 2, 2, 2]", "[]"],
)
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
