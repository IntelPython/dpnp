import numpy
import pytest
from numpy.testing import (
    assert_array_equal,
    assert_equal,
    assert_raises,
    assert_raises_regex,
)

import dpnp

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_unsigned_dtypes,
)


class TestApplyAlongAxis:
    def test_tuple_func1d(self):
        def sample_1d(x):
            return x[1], x[0]

        a = numpy.array([[1, 2], [3, 4]])
        ia = dpnp.array(a)

        # 2d insertion along first axis
        expected = numpy.apply_along_axis(sample_1d, 1, a)
        result = dpnp.apply_along_axis(sample_1d, 1, ia)
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("stride", [-1, 2, -3])
    def test_stride(self, stride):
        a = numpy.ones((20, 10), dtype="f")
        ia = dpnp.array(a)

        expected = numpy.apply_along_axis(len, 0, a[::stride, ::stride])
        result = dpnp.apply_along_axis(len, 0, ia[::stride, ::stride])
        assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes())
    def test_args(self, dtype):
        a = numpy.ones((20, 10))
        ia = dpnp.array(a)

        # kwargs
        expected = numpy.apply_along_axis(
            numpy.mean, 0, a, dtype=dtype, keepdims=True
        )
        result = dpnp.apply_along_axis(
            dpnp.mean, 0, ia, dtype=dtype, keepdims=True
        )
        assert_array_equal(result, expected)

        # positional args: axis, dtype, out, keepdims
        result = dpnp.apply_along_axis(dpnp.mean, 0, ia, 0, dtype, None, True)
        assert_array_equal(result, expected)


class TestApplyOverAxes:
    @pytest.mark.parametrize("func", ["sum", "cumsum"])
    @pytest.mark.parametrize("axes", [1, [0, 2], (-1, -2)])
    def test_basic(self, func, axes):
        a = numpy.arange(24).reshape(2, 3, 4)
        ia = dpnp.array(a)

        expected = numpy.apply_over_axes(getattr(numpy, func), a, axes)
        result = dpnp.apply_over_axes(getattr(dpnp, func), ia, axes)
        assert_array_equal(result, expected)

    def test_custom_func(self):
        def custom_func(x, axis):
            return dpnp.expand_dims(dpnp.expand_dims(x, axis), axis)

        ia = dpnp.arange(24).reshape(2, 3, 4)
        assert_raises(ValueError, dpnp.apply_over_axes, custom_func, ia, 1)


class TestPiecewise:
    @pytest.mark.parametrize(
        "dtype", get_all_dtypes(no_none=True, no_unsigned=True)
    )
    @pytest.mark.parametrize("funclist", [[True, False], [-1, 1], [-1.5, 1.5]])
    def test_basic(self, dtype, funclist):
        low = 0 if dpnp.issubdtype(dtype, dpnp.unsignedinteger) else -10
        a = generate_random_numpy_array(10, dtype=dtype, low=low)
        ia = dpnp.array(a)

        expected = numpy.piecewise(a, [a < 0, a >= 0], funclist)
        result = dpnp.piecewise(ia, [ia < 0, ia >= 0], funclist)
        assert a.dtype == result.dtype
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_unsigned_dtypes())
    @pytest.mark.parametrize("funclist", [[True, False], [1, 2], [1.5, 4.5]])
    def test_unsigned(self, dtype, funclist):
        a = generate_random_numpy_array(10, dtype=dtype, low=0)
        ia = dpnp.array(a)

        expected = numpy.piecewise(a, [a < 0, a >= 0], funclist)
        result = dpnp.piecewise(ia, [ia < 0, ia >= 0], funclist)
        assert a.dtype == result.dtype
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic_complex(self, dtype):
        a = generate_random_numpy_array(10, dtype=dtype)
        ia = dpnp.array(a)
        funclist = [-1.5 - 1j * 1.5, 1.5 + 1j * 1.5]

        if numpy.issubdtype(dtype, numpy.complexfloating) or dtype == dpnp.bool:
            expected = numpy.piecewise(a, [a < 0, a >= 0], funclist)
            result = dpnp.piecewise(ia, [ia < 0, ia >= 0], funclist)
            assert a.dtype == result.dtype
            assert_dtype_allclose(result, expected)
        else:
            # If dtype is not complex, piecewise should raise an error
            pytest.raises(
                TypeError, numpy.piecewise, a, [a < 0, a >= 0], funclist
            )
            pytest.raises(
                TypeError, dpnp.piecewise, ia, [ia < 0, ia >= 0], funclist
            )

    def test_simple(self):
        a = numpy.array([0, 0])
        ia = dpnp.array(a)
        # Condition is single bool list
        expected = numpy.piecewise(a, [True, False], [1])
        result = dpnp.piecewise(ia, [True, False], [1])
        assert_array_equal(result, expected)

        # List of conditions: single bool list
        expected = numpy.piecewise(a, [[True, False]], [1])
        result = dpnp.piecewise(ia, [[True, False]], [1])
        assert_array_equal(result, expected)

        # Conditions is single bool array
        expected = numpy.piecewise(a, [numpy.array([True, False])], [1])
        result = dpnp.piecewise(ia, dpnp.array([True, False]), [1])
        assert_array_equal(result, expected)

        # Condition is single int array
        expected = numpy.piecewise(a, [numpy.array([1, 0])], [1])
        result = dpnp.piecewise(ia, dpnp.array([1, 0]), [1])
        assert_array_equal(result, expected)

        # List of conditions: int array
        expected = numpy.piecewise(a, [numpy.array([1, 0])], [1])
        result = dpnp.piecewise(ia, [dpnp.array([1, 0])], [1])
        assert_array_equal(result, expected)

        # List of conditions: single bool tuple
        expected = numpy.piecewise(a, ([True, False], [False, True]), [1, -4])
        result = dpnp.piecewise(ia, ([True, False], [False, True]), [1, -4])
        assert_array_equal(result, expected)

        # Condition is single bool tuple
        expected = numpy.piecewise(a, (True, False), [1])
        result = dpnp.piecewise(ia, (True, False), [1])
        assert_array_equal(result, expected)

    def test_two_conditions(self):
        a = numpy.array([1, 2])
        ia = dpnp.array(a)
        cond = numpy.array([True, False])
        icond = dpnp.array(cond)
        expected = numpy.piecewise(a, [cond, cond], [3, 4])
        result = dpnp.piecewise(ia, [icond, icond], [3, 4])
        assert_array_equal(result, expected)

    def test_default(self):
        a = numpy.array([1, 2])
        ia = dpnp.array(a)
        # No value specified for x[1], should be 0
        expected = numpy.piecewise(a, [True, False], [2])
        result = dpnp.piecewise(ia, [True, False], [2])
        assert_array_equal(result, expected)

        # Should set x[1] to 3
        expected = numpy.piecewise(a, [True, False], [2, 3])
        result = dpnp.piecewise(ia, [True, False], [2, 3])
        assert_array_equal(result, expected)

    def test_0d(self):
        a = numpy.array(3)
        ia = dpnp.array(a)

        expected = numpy.piecewise(a, a > 3, [4, 0])
        result = dpnp.piecewise(ia, ia > 3, [4, 0])
        assert_array_equal(result, expected)

        a = numpy.array(5)
        ia = dpnp.array(a)
        expected = numpy.piecewise(a, [True, False], [1, 0])
        result = dpnp.piecewise(ia, [True, False], [1, 0])
        assert_array_equal(result, expected)

        expected = numpy.piecewise(a, [False, False, True], [1, 2, 3])
        result = dpnp.piecewise(ia, [False, False, True], [1, 2, 3])
        assert_array_equal(result, expected)

    def test_0d_comparison(self):
        a = numpy.array(3)
        ia = dpnp.array(a)
        expected = numpy.piecewise(a, [a > 3, a <= 3], [4, 0])
        result = dpnp.piecewise(ia, [ia > 3, ia <= 3], [4, 0])
        assert_array_equal(result, expected)

        a = numpy.array(4)
        ia = dpnp.array(a)
        expected = numpy.piecewise(
            a, [a <= 3, (a > 3) * (a <= 5), a > 5], [1, 2, 3]
        )
        result = dpnp.piecewise(
            ia, [ia <= 3, (ia > 3) * (ia <= 5), ia > 5], [1, 2, 3]
        )
        assert_array_equal(result, expected)

        assert_raises_regex(
            ValueError,
            "2 or 3 functions are expected",
            dpnp.piecewise,
            ia,
            [ia <= 3, ia > 3],
            [1],
        )
        assert_raises_regex(
            ValueError,
            "2 or 3 functions are expected",
            dpnp.piecewise,
            ia,
            [ia <= 3, ia > 3],
            [1, 1, 1, 1],
        )

    def test_0d_0d_condition(self):
        a = numpy.array(3)
        ia = dpnp.array(a)
        c = numpy.array(a > 3)
        ic = dpnp.array(ia > 3)

        expected = numpy.piecewise(a, [c], [1, 2])
        result = dpnp.piecewise(ia, [ic], [1, 2])
        assert_equal(result, expected)

    def test_multidimensional_extrafunc(self):
        a = numpy.array([[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]])
        ia = dpnp.array(a)

        expected = numpy.piecewise(a, [a < 0, a >= 2], [-1, 1, 3])
        result = dpnp.piecewise(ia, [ia < 0, ia >= 2], [-1, 1, 3])
        assert_array_equal(result, expected)

    def test_error_dpnp(self):
        ia = dpnp.array([0, 0])
        # values cannot be a callable function
        assert_raises_regex(
            NotImplementedError,
            "Callable functions are not supported currently",
            dpnp.piecewise,
            ia,
            [dpnp.array([True, False])],
            [lambda x: -1],
        )

        # default value cannot be a callable function
        assert_raises_regex(
            NotImplementedError,
            "Callable functions are not supported currently",
            dpnp.piecewise,
            ia,
            [dpnp.array([True, False])],
            [-1, lambda x: 1],
        )

        # funclist is not array-like
        assert_raises_regex(
            TypeError,
            "funclist must be a sequence of scalars",
            dpnp.piecewise,
            ia,
            [dpnp.array([True, False])],
            1,
        )

        # funclist is a string
        assert_raises_regex(
            TypeError,
            "funclist must be a sequence of scalars",
            dpnp.piecewise,
            ia,
            [ia > 0],
            "q",
        )

        assert_raises_regex(
            TypeError,
            "object of type",
            numpy.piecewise,
            ia.asnumpy(),
            [numpy.array([True, False])],
            1,
        )

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_error(self, xp):
        ia = xp.array([0, 0])
        # not enough functions
        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            xp.piecewise,
            ia,
            [xp.array([True, False])],
            [],
        )

        # extra function
        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            xp.piecewise,
            ia,
            [xp.array([True, False])],
            [1, 2, 3],
        )

        # condlist is empty
        assert_raises_regex(
            IndexError,
            "index out of range",
            xp.piecewise,
            ia,
            [],
            [1, 2],
        )
