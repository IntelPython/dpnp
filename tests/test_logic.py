import numpy
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises

import dpnp

from .helper import (
    get_all_dtypes,
    get_float_complex_dtypes,
    get_float_dtypes,
)


class TestAllAny:
    @pytest.mark.parametrize("func", ["all", "any"])
    @pytest.mark.parametrize("dtype", get_all_dtypes())
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_all_any(self, func, dtype, axis, keepdims):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=dtype)
        np_array = dpnp.asnumpy(dp_array)

        expected = getattr(numpy, func)(np_array, axis=axis, keepdims=keepdims)
        result = getattr(dpnp, func)(dp_array, axis=axis, keepdims=keepdims)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("func", ["all", "any"])
    @pytest.mark.parametrize("a_dtype", get_all_dtypes())
    @pytest.mark.parametrize("out_dtype", get_all_dtypes())
    def test_all_any_out(self, func, a_dtype, out_dtype):
        dp_array = dpnp.array([[0, 1, 2], [3, 4, 0]], dtype=a_dtype)
        np_array = dpnp.asnumpy(dp_array)

        expected = getattr(numpy, func)(np_array)
        out = dpnp.empty(expected.shape, dtype=out_dtype)
        result = getattr(dpnp, func)(dp_array, out=out)
        assert out is result
        assert_allclose(result, expected)

    @pytest.mark.parametrize("func", ["all", "any"])
    @pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
    @pytest.mark.parametrize("shape", [(2, 3), (2, 0), (0, 3)])
    def test_all_any_empty(self, func, axis, shape):
        dp_array = dpnp.empty(shape, dtype=dpnp.int64)
        np_array = dpnp.asnumpy(dp_array)

        result = getattr(dpnp, func)(dp_array, axis=axis)
        expected = getattr(numpy, func)(np_array, axis=axis)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("func", ["all", "any"])
    def test_all_any_scalar(self, func):
        dp_array = dpnp.array(0)
        np_array = dpnp.asnumpy(dp_array)

        result = getattr(dp_array, func)()
        expected = getattr(np_array, func)()
        assert_allclose(result, expected)

    @pytest.mark.parametrize("func", ["all", "any"])
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("keepdims", [True, False])
    def test_all_any_nan_inf(self, func, axis, keepdims):
        dp_array = dpnp.array([[dpnp.nan, 1, 2], [dpnp.inf, -dpnp.inf, 0]])
        np_array = dpnp.asnumpy(dp_array)

        expected = getattr(numpy, func)(np_array, axis=axis, keepdims=keepdims)
        result = getattr(dpnp, func)(dp_array, axis=axis, keepdims=keepdims)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("func", ["all", "any"])
    def test_all_any_error(self, func):
        def check_raises(func_name, exception, *args, **kwargs):
            assert_raises(
                exception, lambda: getattr(dpnp, func_name)(*args, **kwargs)
            )

        a = dpnp.arange(5)
        # unsupported where parameter
        check_raises(func, NotImplementedError, a, where=False)
        # unsupported type
        check_raises(func, TypeError, dpnp.asnumpy(a))
        check_raises(func, TypeError, [0, 1, 2, 3])


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
        with pytest.raises(TypeError):
            dpnp.allclose(1.0, 1.0)

    @pytest.mark.parametrize("val", [[1.0], (-3, 7), numpy.arange(5)])
    def test_wrong_input_arrays(self, val):
        with pytest.raises(TypeError):
            dpnp.allclose(val, val)

    @pytest.mark.parametrize("tol", [[0.001], (1.0e-6,), numpy.array([1.0e-5])])
    def test_wrong_tols(self, tol):
        a = dpnp.ones(10)
        b = dpnp.ones(10)

        for kw in [{"rtol": tol}, {"atol": tol}, {"rtol": tol, "atol": tol}]:
            with pytest.raises(TypeError):
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
    create_func = lambda xp, a: (
        xp.asarray(a, dtype=dtype)
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


@pytest.mark.parametrize("func", ["isneginf", "isposinf"])
@pytest.mark.parametrize(
    "data",
    [
        [dpnp.inf, -1, 0, 1, dpnp.nan, -dpnp.inf],
        [[dpnp.inf, dpnp.nan], [dpnp.nan, 0], [1, -dpnp.inf]],
    ],
    ids=[
        "1D array",
        "2D array",
    ],
)
@pytest.mark.parametrize("dtype", get_float_dtypes())
def test_infinity_sign(func, data, dtype):
    x = dpnp.asarray(data, dtype=dtype)
    np_res = getattr(numpy, func)(x.asnumpy())
    dpnp_res = getattr(dpnp, func)(x)
    assert_equal(dpnp_res, np_res)

    dp_out = dpnp.empty(np_res.shape, dtype=dpnp.bool)
    dpnp_res = getattr(dpnp, func)(x, out=dp_out)
    assert dp_out is dpnp_res
    assert_equal(dpnp_res, np_res)


@pytest.mark.parametrize("func", ["isneginf", "isposinf"])
def test_infinity_sign_errors(func):
    data = [dpnp.inf, 0, -dpnp.inf]

    # unsupported data type
    x = dpnp.asarray(data, dtype="c8")
    x_np = dpnp.asnumpy(x)
    assert_raises(TypeError, getattr(dpnp, func), x)
    assert_raises(TypeError, getattr(numpy, func), x_np)

    # unsupported type
    assert_raises(TypeError, getattr(dpnp, func), data)
    assert_raises(TypeError, getattr(dpnp, func), x_np)

    # unsupported `out` data type
    x = dpnp.asarray(data, dtype=dpnp.default_float_type())
    out = dpnp.empty_like(x, dtype="int32")
    with pytest.raises(ValueError):
        getattr(dpnp, func)(x, out=out)


@pytest.mark.parametrize("dtype", get_all_dtypes(no_bool=True, no_complex=True))
@pytest.mark.parametrize(
    "rtol", [1e-05, dpnp.array(1e-05), dpnp.full(10, 1e-05)]
)
@pytest.mark.parametrize(
    "atol", [1e-08, dpnp.array(1e-08), dpnp.full(10, 1e-08)]
)
def test_isclose(dtype, rtol, atol):
    a = numpy.random.rand(10)
    b = a + numpy.random.rand(10) * 1e-8

    dpnp_a = dpnp.array(a, dtype=dtype)
    dpnp_b = dpnp.array(b, dtype=dtype)

    np_res = numpy.isclose(a, b, 1e-05, 1e-08)
    dpnp_res = dpnp.isclose(dpnp_a, dpnp_b, rtol, atol)
    assert_allclose(dpnp_res, np_res)


def _test_array_equal_parametrizations():
    """
    we pre-create arrays as we sometime want to pass the same instance
    and sometime not. Passing the same instances may not mean the array are
    equal, especially when containing None
    """
    # those are 0-d arrays, it used to be a special case
    # where (e0 == e0).all() would raise
    e0 = dpnp.array(0, dtype="i4")
    e1 = dpnp.array(1, dtype="f4")
    # x,y, nan_equal, expected_result
    yield (e0, e0.copy(), None, True)
    yield (e0, e0.copy(), False, True)
    yield (e0, e0.copy(), True, True)

    #
    yield (e1, e1.copy(), None, True)
    yield (e1, e1.copy(), False, True)
    yield (e1, e1.copy(), True, True)

    # Non-nanable – those cannot hold nans
    a12 = dpnp.array([1, 2])
    a12b = a12.copy()
    a123 = dpnp.array([1, 2, 3])
    a13 = dpnp.array([1, 3])
    a34 = dpnp.array([3, 4])

    yield (a12, a12b, None, True)
    yield (a12, a12, None, True)
    yield (a12, a123, None, False)
    yield (a12, a34, None, False)
    yield (a12, a13, None, False)

    # Non-float dtype - equal_nan should have no effect,
    yield (a123, a123, None, True)
    yield (a123, a123, False, True)
    yield (a123, a123, True, True)
    yield (a123, a123.copy(), None, True)
    yield (a123, a123.copy(), False, True)
    yield (a123, a123.copy(), True, True)
    yield (a123.astype("f4"), a123.astype("f4"), None, True)
    yield (a123.astype("f4"), a123.astype("f4"), False, True)
    yield (a123.astype("f4"), a123.astype("f4"), True, True)

    # these can hold None
    b1 = dpnp.array([1, 2, dpnp.nan])
    b2 = dpnp.array([1, dpnp.nan, 2])
    b3 = dpnp.array([1, 2, dpnp.inf])
    b4 = dpnp.array(dpnp.nan)

    # instances are the same
    yield (b1, b1, None, False)
    yield (b1, b1, False, False)
    yield (b1, b1, True, True)

    # equal but not same instance
    yield (b1, b1.copy(), None, False)
    yield (b1, b1.copy(), False, False)
    yield (b1, b1.copy(), True, True)

    # same once stripped of Nan
    yield (b1, b2, None, False)
    yield (b1, b2, False, False)
    yield (b1, b2, True, False)

    # nan's not conflated with inf's
    yield (b1, b3, None, False)
    yield (b1, b3, False, False)
    yield (b1, b3, True, False)

    # all Nan
    yield (b4, b4, None, False)
    yield (b4, b4, False, False)
    yield (b4, b4, True, True)
    yield (b4, b4.copy(), None, False)
    yield (b4, b4.copy(), False, False)
    yield (b4, b4.copy(), True, True)

    # Multi-dimensional array
    md1 = dpnp.array([[0, 1], [dpnp.nan, 1]])

    yield (md1, md1, None, False)
    yield (md1, md1, False, False)
    yield (md1, md1, True, True)
    yield (md1, md1.copy(), None, False)
    yield (md1, md1.copy(), False, False)
    yield (md1, md1.copy(), True, True)
    # both complexes are nan+nan.j but the same instance
    cplx1, cplx2 = [dpnp.array([dpnp.nan + dpnp.nan * 1j])] * 2

    # only real or img are nan.
    cplx3 = dpnp.array(1 + dpnp.nan * 1j)
    cplx4 = dpnp.array(dpnp.nan + 1j)

    # Complex values
    yield (cplx1, cplx2, None, False)
    yield (cplx1, cplx2, False, False)
    yield (cplx1, cplx2, True, True)

    # Complex values, 1+nan, nan+1j
    yield (cplx3, cplx4, None, False)
    yield (cplx3, cplx4, False, False)
    yield (cplx3, cplx4, True, True)


class TestArrayComparisons:
    @pytest.mark.parametrize(
        "bx,by,equal_nan,expected", _test_array_equal_parametrizations()
    )
    def test_array_equal_equal_nan(self, bx, by, equal_nan, expected):
        if equal_nan is None:
            res = dpnp.array_equal(bx, by)
        else:
            res = dpnp.array_equal(bx, by, equal_nan=equal_nan)
        assert_equal(res, dpnp.array(expected))

    def test_array_equiv(self):
        res = dpnp.array_equiv(dpnp.array([1, 2]), dpnp.array([1, 2]))
        assert res
        res = dpnp.array_equiv(dpnp.array([1, 2]), dpnp.array([1, 2, 3]))
        assert not res
        res = dpnp.array_equiv(dpnp.array([1, 2]), dpnp.array([3, 4]))
        assert not res
        res = dpnp.array_equiv(dpnp.array([1, 2]), dpnp.array([1, 3]))
        assert not res

        res = dpnp.array_equiv(dpnp.array([1, 1]), dpnp.array([1]))
        assert res
        res = dpnp.array_equiv(dpnp.array([1, 1]), dpnp.array([[1], [1]]))
        assert res
        res = dpnp.array_equiv(dpnp.array([1, 2]), dpnp.array([2]))
        assert not res
        res = dpnp.array_equiv(dpnp.array([1, 2]), dpnp.array([[1], [2]]))
        assert not res
        res = dpnp.array_equiv(
            dpnp.array([1, 2]), dpnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        )
        assert not res
