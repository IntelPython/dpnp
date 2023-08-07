import unittest

import numpy
import pytest

from tests.third_party.cupy import testing


def _calc_out_shape(shape, axis, keepdims):
    if axis is None:
        axis = list(range(len(shape)))
    elif isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)

    shape = numpy.array(shape)

    if keepdims:
        shape[axis] = 1
    else:
        shape[axis] = -1
        shape = filter(lambda x: x != -1, shape)
    return tuple(shape)


@testing.parameterize(
    *testing.product(
        {
            "f": ["all", "any"],
            "x": [
                numpy.arange(24).reshape(2, 3, 4) - 10,
                numpy.zeros((2, 3, 4)),
                numpy.ones((2, 3, 4)),
                numpy.zeros((0, 3, 4)),
                numpy.ones((0, 3, 4)),
            ],
            "axis": [0],
            "keepdims": [False, True],
        }
    )
)
class TestAllAny(unittest.TestCase):
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_without_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        return getattr(xp, self.f)(x, self.axis, None, self.keepdims)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_with_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        out_shape = _calc_out_shape(x.shape, self.axis, self.keepdims)
        out = xp.empty(out_shape, dtype=x.dtype)
        getattr(xp, self.f)(x, self.axis, out, self.keepdims)
        return out


@testing.parameterize(
    *testing.product(
        {
            "f": ["all", "any"],
            "x": [
                numpy.array([[[numpy.nan]]]),
                numpy.array([[[numpy.nan, 0]]]),
                numpy.array([[[numpy.nan, 1]]]),
                numpy.array([[[numpy.nan, 0, 1]]]),
            ],
            "axis": [None, (0, 1, 2), 0, 1, 2, (0, 1)],
            "keepdims": [False, True],
        }
    )
)
class TestAllAnyWithNaN(unittest.TestCase):
    @testing.for_dtypes((*testing.helper._float_dtypes, numpy.bool_))
    @testing.numpy_cupy_array_equal()
    def test_without_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        return getattr(xp, self.f)(x, self.axis, None, self.keepdims)

    @pytest.mark.usefixtures("allow_fall_back_on_numpy")
    @testing.for_dtypes((*testing.helper._float_dtypes, numpy.bool_))
    @testing.numpy_cupy_array_equal()
    def test_with_out(self, xp, dtype):
        x = xp.asarray(self.x).astype(dtype)
        out_shape = _calc_out_shape(x.shape, self.axis, self.keepdims)
        out = xp.empty(out_shape, dtype=x.dtype)
        getattr(xp, self.f)(x, self.axis, out, self.keepdims)
        return out
