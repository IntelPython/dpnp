from itertools import permutations

import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_float_dtypes,
)


@pytest.mark.usefixtures("suppress_complex_warning")
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1, 2, 3),
        (1, 0, 2),
        (10,),
        (3, 3, 3),
        (5, 5),
        (0, 6),
        (10, 1),
        (1, 10),
        (35, 40),
        (40, 35),
    ],
)
@pytest.mark.parametrize("dtype_in", get_all_dtypes(no_none=True))
@pytest.mark.parametrize("dtype_out", get_all_dtypes())
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("order", ["C", "F"])
def test_sum(shape, dtype_in, dtype_out, transpose, keepdims, order):
    a_np = generate_random_numpy_array(shape, dtype_in, order)
    a = dpnp.asarray(a_np)

    if transpose:
        a_np = a_np.T
        a = a.T

    axes_range = list(numpy.arange(len(shape)))
    axes = [None]
    axes += axes_range
    axes += permutations(axes_range, 2)
    axes.append(tuple(axes_range))

    for axis in axes:
        dpnp_res = a.sum(axis=axis, dtype=dtype_out, keepdims=keepdims)
        numpy_res = a_np.sum(axis=axis, dtype=dtype_out, keepdims=keepdims)
        assert_dtype_allclose(dpnp_res, numpy_res, factor=16)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True, no_bool=True))
def test_sum_empty_out(dtype):
    a = dpnp.empty((1, 2, 0, 4), dtype=dtype)
    out = dpnp.ones((), dtype=dtype)
    res = a.sum(out=out)
    assert out is res
    assert_array_equal(out, numpy.array(0, dtype=dtype))


@pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
@pytest.mark.parametrize("axis", [None, 0, 1, 2, 3])
def test_sum_empty(dtype, axis):
    a = numpy.empty((1, 2, 0, 4), dtype=dtype)
    numpy_res = a.sum(axis=axis)
    dpnp_res = dpnp.array(a).sum(axis=axis)
    assert_array_equal(numpy_res, dpnp_res)


@pytest.mark.parametrize("dtype", get_float_dtypes())
def test_sum_float(dtype):
    a = numpy.array(
        [
            [[-2.0, 3.0], [9.1, 0.2]],
            [[-2.0, 5.0], [-2, -1.2]],
            [[1.0, -2.0], [5.0, -1.1]],
        ],
        dtype=dtype,
    )
    ia = dpnp.array(a)

    for axis in range(len(a)):
        result = dpnp.sum(ia, axis=axis)
        expected = numpy.sum(a, axis=axis)
        assert_dtype_allclose(result, expected)


def test_sum_int():
    a = numpy.array([1, 0, 2, -3, -1, 2, 21, -9])
    ia = dpnp.array(a)

    result = dpnp.sum(ia)
    expected = numpy.sum(a)
    assert_array_equal(expected, result)


def test_sum_axis():
    a = numpy.array(
        [
            [[-2.0, 3.0], [9.1, 0.2]],
            [[-2.0, 5.0], [-2, -1.2]],
            [[1.0, -2.0], [5.0, -1.1]],
        ],
        dtype="f4",
    )
    ia = dpnp.array(a)

    result = dpnp.sum(ia, axis=1)
    expected = numpy.sum(a, axis=1)
    assert_array_equal(expected, result)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True))
@pytest.mark.parametrize("axis", [0, 1, (0, 1)])
def test_sum_out(dtype, axis):
    a = dpnp.arange(2 * 4, dtype=dtype).reshape(2, 4)
    a_np = dpnp.asnumpy(a)

    expected = numpy.sum(a_np, axis=axis)
    res = dpnp.empty(expected.shape, dtype=dtype)
    a.sum(axis=axis, out=res)
    # NumPy result is without out keyword, dtype may differ
    assert_array_equal(res, expected, strict=False)


@pytest.mark.usefixtures("suppress_complex_warning")
@pytest.mark.parametrize("arr_dt", get_all_dtypes(no_none=True))
@pytest.mark.parametrize("out_dt", get_all_dtypes(no_none=True))
@pytest.mark.parametrize("dtype", get_all_dtypes())
def test_sum_out_dtype(arr_dt, out_dt, dtype):
    a = numpy.arange(10, 20).reshape((2, 5)).astype(dtype=arr_dt)
    out = numpy.zeros_like(a, shape=(2,), dtype=out_dt)

    ia = dpnp.array(a)
    iout = dpnp.array(out)

    result = dpnp.sum(ia, out=iout, dtype=dtype, axis=1)
    expected = numpy.sum(a, out=out, dtype=dtype, axis=1)
    assert_array_equal(expected, result)
    assert result is iout


def test_sum_NotImplemented():
    ia = dpnp.arange(5)
    with pytest.raises(NotImplementedError):
        dpnp.sum(ia, where=False)

    with pytest.raises(NotImplementedError):
        dpnp.sum(ia, initial=1)
