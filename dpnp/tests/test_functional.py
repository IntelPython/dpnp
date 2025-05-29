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
    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic(self, dtype):
        a = generate_random_numpy_array(10, dtype=dtype)
        ia = dpnp.array(a)

        funclist = [-1, 1]
        expected = numpy.piecewise(a, [a < 0, a >= 0], funclist)
        result = dpnp.piecewise(ia, [ia < 0, ia >= 0], funclist)
        assert a.dtype == result.dtype
        assert_dtype_allclose(result, expected)

    @pytest.mark.parametrize("dtype", get_all_dtypes(no_none=True))
    def test_basic2(self, dtype):
        a = generate_random_numpy_array(10, dtype=dtype)
        ia = dpnp.array(a)

        funclist = [-1.5, 1.5]
        expected = numpy.piecewise(a, [a < 0, a >= 0], funclist)
        result = dpnp.piecewise(ia, [ia < 0, ia >= 0], funclist)
        assert a.dtype == result.dtype
        assert_dtype_allclose(result, expected)

    def test_simple(self):
        ia = dpnp.array([0, 0])
        # Condition is single bool list
        x = dpnp.piecewise(ia, [True, False], [1])
        assert_array_equal(x, [1, 0])

        # List of conditions: int array
        x = dpnp.piecewise(ia, [dpnp.array([1, 0])], [1])
        assert_array_equal(x, [1, 0])

    def test_error(self):
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

        # not enough functions
        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            dpnp.piecewise,
            ia,
            [dpnp.array([True, False])],
            [],
        )

        # extra function
        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            dpnp.piecewise,
            ia,
            [dpnp.array([True, False])],
            [1, 2, 3],
        )

    # cupy.piecewise(x, [False, cupy.array([True, False])], [-1, 1])
    # cupy.piecewise(x, [False, True], [-1, 1])
    # cupy.piecewise(x, cupy.array([False, True]), [-1, 1]) # dpnp error
    # cupy.piecewise(x, [cupy.array(False), cupy.array([True, False])], [-1, 1])
    # cupy.piecewise(x, (True, cupy.array([True, False, False])), [-1, 1]) # error broadcast
    def test_simple2(self):
        ia = dpnp.array([0, 0])
        # Condition is single bool list
        x = dpnp.piecewise(
            ia, [True, False], [1]
        )  # ValueError: with 2 condition(s), either 2 or 3 functions are expected
        assert_array_equal(x, [1, 0])

        # List of conditions: single bool list
        x = dpnp.piecewise(
            ia, [[True, False]], [1]
        )  # TypeError: Unsupported type <class 'list'>
        assert_array_equal(x, [1, 0])

        # Conditions is single bool array
        x = dpnp.piecewise(
            ia, dpnp.array([True, False]), [1]
        )  # ValueError: with 2 condition(s), either 2 or 3 functions are expected
        assert_array_equal(x, [1, 0])

        # Condition is single int array
        x = dpnp.piecewise(
            ia, dpnp.array([1, 0]), [1]
        )  # ValueError: with 2 condition(s), either 2 or 3 functions are expected
        assert_array_equal(x, [1, 0])

        # List of conditions: int array
        x = dpnp.piecewise(ia, [dpnp.array([1, 0])], [1])
        assert_array_equal(x, [1, 0])

        x = dpnp.piecewise(ia, [[False, True]], [lambda x: -1])
        assert_array_equal(x, [0, -1])

        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            dpnp.piecewise,
            ia,
            [[False, True]],
            [],
        )
        assert_raises_regex(
            ValueError,
            "1 or 2 functions are expected",
            dpnp.piecewise,
            ia,
            [[False, True]],
            [1, 2, 3],
        )

    def test_two_conditions(self):
        ia = dpnp.array([1, 2])
        x = dpnp.piecewise(ia, [[True, False], [False, True]], [3, 4])
        assert_array_equal(x, [3, 4])

    def test_scalar_domains_three_conditions(self):
        x = dpnp.piecewise(dpnp.array(3), [True, False, False], [4, 2, 0])
        assert_equal(x, 4)

    def test_default(self):
        # No value specified for x[1], should be 0
        x = dpnp.piecewise(dpnp.array([1, 2]), [True, False], [2])
        assert_array_equal(x, [2, 0])

        # Should set x[1] to 3
        x = dpnp.piecewise([1, 2], [True, False], [2, 3])
        assert_array_equal(x, [2, 3])

    def test_0d(self):
        x = dpnp.array(3)
        y = dpnp.piecewise(x, x > 3, [4, 0])
        assert y.ndim == 0
        assert y == 0

        x = dpnp.array(5)
        y = dpnp.piecewise(x, [True, False], [1, 0])
        assert y.ndim == 0
        assert y == 1

        # With 3 ranges (It was failing, before)
        y = dpnp.piecewise(x, [False, False, True], [1, 2, 3])
        assert_array_equal(y, 3)

    def test_0d_comparison(self):
        x = dpnp.array(3)
        y = dpnp.piecewise(x, [x <= 3, x > 3], [4, 0])  # Should succeed.
        assert_equal(y, 4)

        # With 3 ranges (It was failing, before)
        x = 4
        y = dpnp.piecewise(x, [x <= 3, (x > 3) * (x <= 5), x > 5], [1, 2, 3])
        assert_array_equal(y, 2)

        assert_raises_regex(
            ValueError,
            "2 or 3 functions are expected",
            dpnp.piecewise,
            x,
            [x <= 3, x > 3],
            [1],
        )
        assert_raises_regex(
            ValueError,
            "2 or 3 functions are expected",
            dpnp.piecewise,
            x,
            [x <= 3, x > 3],
            [1, 1, 1, 1],
        )

    def test_0d_0d_condition(self):
        x = dpnp.array(3)
        c = dpnp.array(x > 3)
        y = dpnp.piecewise(x, [c], [1, 2])
        assert_equal(y, 2)

    def test_multidimensional_extrafunc(self):
        x = dpnp.array([[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]])
        y = dpnp.piecewise(x, [x < 0, x >= 2], [-1, 1, 3])
        assert_array_equal(y, dpnp.array([[-1.0, -1.0, -1.0], [3.0, 3.0, 1.0]]))

    def test_subclasses(self):
        class subclass(dpnp.ndarray):
            pass

        x = dpnp.arange(5.0).view(subclass)
        r = dpnp.piecewise(x, [x < 2.0, x >= 4], [-1.0, 1.0, 0.0])
        assert_equal(type(r), subclass)
        assert_equal(r, [-1.0, -1.0, 0.0, 0.0, 1.0])
