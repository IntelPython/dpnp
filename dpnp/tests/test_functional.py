import numpy
import pytest
from numpy.testing import assert_array_equal

import dpnp

from .helper import get_all_dtypes


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
