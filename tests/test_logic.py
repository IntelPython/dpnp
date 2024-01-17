import numpy
import pytest
from numpy.testing import assert_allclose, assert_equal

import dpnp

from .helper import (
    get_all_dtypes,
    get_float_complex_dtypes,
    has_support_aspect64,
)


@pytest.mark.parametrize("type", get_all_dtypes())
@pytest.mark.parametrize(
    "shape",
    [(0,), (4,), (2, 3), (2, 2, 2)],
    ids=["(0,)", "(4,)", "(2,3)", "(2,2,2)"],
)
def test_all(type, shape):
    size = 1
    for i in range(len(shape)):
        size *= shape[i]

    for i in range(2**size):
        t = i

        a = numpy.empty(size, dtype=type)

        for j in range(size):
            a[j] = 0 if t % 2 == 0 else j + 1
            t = t >> 1

        a = a.reshape(shape)

        ia = dpnp.array(a)

        np_res = numpy.all(a)
        dpnp_res = dpnp.all(ia)
        assert_allclose(dpnp_res, np_res)

        np_res = a.all()
        dpnp_res = ia.all()
        assert_allclose(dpnp_res, np_res)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
def test_allclose(dtype):
    a = numpy.random.rand(10)
    b = a + numpy.random.rand(10) * 1e-8

    dpnp_a = dpnp.array(a, dtype=dtype)
    dpnp_b = dpnp.array(b, dtype=dtype)

    np_res = numpy.allclose(a, b)
    dpnp_res = dpnp.allclose(dpnp_a, dpnp_b)
    assert_allclose(dpnp_res, np_res)

    a[0] = numpy.inf

    dpnp_a = dpnp.array(a)

    np_res = numpy.allclose(a, b)
    dpnp_res = dpnp.allclose(dpnp_a, dpnp_b)
    assert_allclose(dpnp_res, np_res)


class TestAllClose:
    @pytest.mark.parametrize("val", [1.0, 3, numpy.inf, -numpy.inf, numpy.nan])
    def test_input_0d(self, val):
        dp_arr = dpnp.array(val)
        np_arr = numpy.array(val)

        # array & scalar
        dp_res = dpnp.allclose(dp_arr, val)
        np_res = numpy.allclose(np_arr, val)
        assert_allclose(dp_res, np_res)

        # scalar & array
        dp_res = dpnp.allclose(val, dp_arr)
        np_res = numpy.allclose(val, np_arr)
        assert_allclose(dp_res, np_res)

        # two arrays
        dp_res = dpnp.allclose(dp_arr, dp_arr)
        np_res = numpy.allclose(np_arr, np_arr)
        assert_allclose(dp_res, np_res)

    @pytest.mark.parametrize("sh_a", [(10,), (10, 10)])
    @pytest.mark.parametrize("sh_b", [(1, 10), (1, 10, 1)])
    def test_broadcast(self, sh_a, sh_b):
        dp_a = dpnp.ones(sh_a)
        dp_b = dpnp.ones(sh_b)

        np_a = numpy.ones(sh_a)
        np_b = numpy.ones(sh_b)

        dp_res = dpnp.allclose(dp_a, dp_b)
        np_res = numpy.allclose(np_a, np_b)
        assert_allclose(dp_res, np_res)

    def test_input_as_scalars(self):
        with pytest.raises(NotImplementedError):
            dpnp.allclose(1.0, 1.0)

    @pytest.mark.parametrize("val", [[1.0], (-3, 7), numpy.arange(5)])
    def test_wrong_input_arrays(self, val):
        with pytest.raises(NotImplementedError):
            dpnp.allclose(val, val)

    @pytest.mark.parametrize(
        "tol", [[0.001], (1.0e-6,), dpnp.array(1.0e-3), numpy.array([1.0e-5])]
    )
    def test_wrong_tols(self, tol):
        a = dpnp.ones(10)
        b = dpnp.ones(10)

        for kw in [{"rtol": tol}, {"atol": tol}, {"rtol": tol, "atol": tol}]:
            with pytest.raises(
                TypeError, match=r"An argument .* must be a scalar, but got"
            ):
                dpnp.allclose(a, b, **kw)


@pytest.mark.parametrize("type", get_all_dtypes())
@pytest.mark.parametrize(
    "shape",
    [(0,), (4,), (2, 3), (2, 2, 2)],
    ids=["(0,)", "(4,)", "(2,3)", "(2,2,2)"],
)
def test_any(type, shape):
    size = 1
    for i in range(len(shape)):
        size *= shape[i]

    for i in range(2**size):
        t = i

        a = numpy.empty(size, dtype=type)

        for j in range(size):
            a[j] = 0 if t % 2 == 0 else j + 1
            t = t >> 1

        a = a.reshape(shape)

        ia = dpnp.array(a)

        np_res = numpy.any(a)
        dpnp_res = dpnp.any(ia)
        assert_allclose(dpnp_res, np_res)

        np_res = a.any()
        dpnp_res = ia.any()
        assert_allclose(dpnp_res, np_res)


def test_equal():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a)):
        np_res = a == i
        dpnp_res = ia == i
        assert_equal(dpnp_res, np_res)


def test_greater():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a) + 1):
        np_res = a > i
        dpnp_res = ia > i
        assert_equal(dpnp_res, np_res)


def test_greater_equal():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a) + 1):
        np_res = a >= i
        dpnp_res = ia >= i
        assert_equal(dpnp_res, np_res)


def test_less():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a) + 1):
        np_res = a < i
        dpnp_res = ia < i
        assert_equal(dpnp_res, np_res)


def test_less_equal():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a) + 1):
        np_res = a <= i
        dpnp_res = ia <= i
        assert_equal(dpnp_res, np_res)


def test_not_equal():
    a = numpy.array([1, 2, 3, 4, 5, 6, 7, 8])
    ia = dpnp.array(a)
    for i in range(len(a)):
        np_res = a != i
        dpnp_res = ia != i
        assert_equal(dpnp_res, np_res)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
@pytest.mark.parametrize(
    "op",
    ["logical_and", "logical_or", "logical_xor"],
    ids=["logical_and", "logical_or", "logical_xor"],
)
def test_logic_comparison(op, dtype):
    a = numpy.array([0, 0, 3, 2], dtype=dtype)
    b = numpy.array([0, 4, 0, 2], dtype=dtype)

    # x1 OP x2
    np_res = getattr(numpy, op)(a, b)
    dpnp_res = getattr(dpnp, op)(dpnp.array(a), dpnp.array(b))
    assert_equal(dpnp_res, np_res)

    # x2 OP x1
    np_res = getattr(numpy, op)(b, a)
    dpnp_res = getattr(dpnp, op)(dpnp.array(b), dpnp.array(a))
    assert_equal(dpnp_res, np_res)

    # numpy.tile(x1, (10,)) OP numpy.tile(x2, (10,))
    a, b = numpy.tile(a, (10,)), numpy.tile(b, (10,))
    np_res = getattr(numpy, op)(a, b)
    dpnp_res = getattr(dpnp, op)(dpnp.array(a), dpnp.array(b))
    assert_equal(dpnp_res, np_res)

    # numpy.tile(x2, (10, 2)) OP numpy.tile(x1, (10, 2))
    a, b = numpy.tile(a, (10, 1)), numpy.tile(b, (10, 1))
    np_res = getattr(numpy, op)(b, a)
    dpnp_res = getattr(dpnp, op)(dpnp.array(b), dpnp.array(a))
    assert_equal(dpnp_res, np_res)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
def test_logical_not(dtype):
    a = dpnp.array([0, 4, 0, 2], dtype=dtype)

    np_res = numpy.logical_not(a.asnumpy())
    dpnp_res = dpnp.logical_not(a)
    assert_equal(dpnp_res, np_res)

    dp_out = dpnp.empty(np_res.shape, dtype=dpnp.bool)
    dpnp_res = dpnp.logical_not(a, out=dp_out)
    assert dpnp_res is dp_out
    assert_equal(dpnp_res, np_res)


@pytest.mark.parametrize(
    "op",
    [
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "not_equal",
    ],
    ids=[
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "not_equal",
    ],
)
@pytest.mark.parametrize(
    "x1",
    [
        [3, 4, 5, 6],
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[1, 2, 5, 6], [3, 4, 7, 8], [1, 2, 7, 8]],
    ],
    ids=[
        "[3, 4, 5, 6]",
        "[[1, 2, 3, 4], [5, 6, 7, 8]]",
        "[[1, 2, 5, 6], [3, 4, 7, 8], [1, 2, 7, 8]]",
    ],
)
@pytest.mark.parametrize("x2", [5, [1, 2, 5, 6]], ids=["5", "[1, 2, 5, 6]"])
@pytest.mark.parametrize("dtype", get_all_dtypes(no_complex=True))
def test_elemwise_comparison(op, x1, x2, dtype):
    create_func = (
        lambda xp, a: xp.asarray(a, dtype=dtype)
        if not numpy.isscalar(a)
        else numpy.dtype(dtype=dtype).type(a)
    )

    np_x1, np_x2 = create_func(numpy, x1), create_func(numpy, x2)
    dp_x1, dp_x2 = create_func(dpnp, np_x1), create_func(dpnp, np_x2)

    # x1 OP x2
    np_res = getattr(numpy, op)(np_x1, np_x2)
    dpnp_res = getattr(dpnp, op)(dp_x1, dp_x2)
    assert_equal(dpnp_res, np_res)

    # x2 OP x1
    np_res = getattr(numpy, op)(np_x2, np_x1)
    dpnp_res = getattr(dpnp, op)(dp_x2, dp_x1)
    assert_equal(dpnp_res, np_res)

    # x1[::-1] OP x2
    np_res = getattr(numpy, op)(np_x1[::-1], np_x2)
    dpnp_res = getattr(dpnp, op)(dp_x1[::-1], dp_x2)
    assert_equal(dpnp_res, np_res)

    # out keyword
    np_res = getattr(numpy, op)(np_x1, np_x2)
    dp_out = dpnp.empty(np_res.shape, dtype=dpnp.bool)
    dpnp_res = getattr(dpnp, op)(dp_x1, dp_x2, out=dp_out)
    assert dp_out is dpnp_res
    assert_equal(dpnp_res, np_res)


@pytest.mark.parametrize(
    "op",
    [
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "not_equal",
    ],
    ids=[
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "not_equal",
    ],
)
@pytest.mark.parametrize(
    "sh1", [[10], [8, 4], [4, 1, 2]], ids=["(10,)", "(8, 4)", "(4, 1, 2)"]
)
@pytest.mark.parametrize(
    "sh2", [[12], [4, 8], [1, 8, 6]], ids=["(12,)", "(4, 8)", "(1, 8, 6)"]
)
def test_comparison_no_broadcast_with_shapes(op, sh1, sh2):
    x1, x2 = dpnp.random.randn(*sh1), dpnp.random.randn(*sh2)

    # x1 OP x2
    with pytest.raises(ValueError):
        getattr(dpnp, op)(x1, x2)
        getattr(numpy, op)(x1.asnumpy(), x2.asnumpy())


@pytest.mark.parametrize(
    "op", ["isfinite", "isinf", "isnan"], ids=["isfinite", "isinf", "isnan"]
)
@pytest.mark.parametrize(
    "data",
    [
        [dpnp.inf, -1, 0, 1, dpnp.nan],
        [[dpnp.inf, dpnp.nan], [dpnp.nan, 0], [1, dpnp.inf]],
    ],
    ids=[
        "[dpnp.inf, -1, 0, 1, dpnp.nan]",
        "[[dpnp.inf, dpnp.nan], [dpnp.nan, 0], [1, dpnp.inf]]",
    ],
)
@pytest.mark.parametrize("dtype", get_float_complex_dtypes())
def test_finite(op, data, dtype):
    x = dpnp.asarray(data, dtype=dtype)
    np_res = getattr(numpy, op)(x.asnumpy())
    dpnp_res = getattr(dpnp, op)(x)
    assert_equal(dpnp_res, np_res)

    dp_out = dpnp.empty(np_res.shape, dtype=dpnp.bool)
    dpnp_res = getattr(dpnp, op)(x, out=dp_out)
    assert dp_out is dpnp_res
    assert_equal(dpnp_res, np_res)
