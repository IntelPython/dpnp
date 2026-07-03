import numpy
import pytest
from numpy.lib.stride_tricks import as_strided as np_as_strided
from numpy.testing import assert_array_equal, assert_raises

import dpnp
from dpnp.lib.stride_tricks import as_strided

from .helper import generate_random_numpy_array, get_all_dtypes


class TestAsStrided:
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_basic(self, dt):
        a = generate_random_numpy_array((4,), dtype=dt)
        ia = dpnp.array(a)

        result = as_strided(ia, shape=(2,), strides=(2 * ia.itemsize,))
        expected = np_as_strided(a, shape=(2,), strides=(2 * a.itemsize,))
        assert_array_equal(result, expected)

    def test_broadcast_via_zero_stride(self):
        a = numpy.array([1, 2, 3, 4], dtype=numpy.int32)
        ia = dpnp.array(a)

        result = as_strided(ia, shape=(3, 4), strides=(0, ia.itemsize))
        expected = np_as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
        assert_array_equal(result, expected)

    def test_default_shape_and_strides(self):
        a = numpy.arange(6).reshape(2, 3)
        ia = dpnp.array(a)

        result = as_strided(ia)
        assert result.shape == ia.shape
        assert result.strides == ia.strides
        assert_array_equal(result, a)

    def test_self_overlapping_view(self):
        a = numpy.arange(5, dtype=numpy.int32)
        ia = dpnp.array(a)

        result = as_strided(
            ia, shape=(3, 3), strides=(ia.itemsize, ia.itemsize)
        )
        expected = np_as_strided(
            a, shape=(3, 3), strides=(a.itemsize, a.itemsize)
        )
        assert_array_equal(result, expected)

        # element (i, j) maps to base offset (i + j), so (0, 1) and (1, 0)
        # share the same element; writing one changes the other
        result[0, 1] = 100
        expected[0, 1] = 100
        assert_array_equal(result, expected)

    def test_overlapping_bulk_write_rejected(self):
        a = dpnp.arange(5, dtype=dpnp.int32)
        view = as_strided(a, shape=(3, 3), strides=(a.itemsize, a.itemsize))

        with pytest.raises(
            ValueError, match="output array is not sufficiently ample"
        ):
            view[...] = dpnp.full((3, 3), 7, dtype=a.dtype)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_writeable_false(self, xp):
        a = xp.array([1, 2, 3, 4], dtype=xp.int32)
        view = xp.lib.stride_tricks.as_strided(
            a, shape=(2,), strides=(2 * a.itemsize,), writeable=False
        )
        assert view.flags["W"] is False
        assert a.flags["W"] is True

        # writing through a read-only view is rejected
        with pytest.raises(ValueError, match="read-only"):
            view[...] = 5

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_writeable_true_readonly_base(self, xp):
        a = xp.arange(10, dtype=xp.int32)
        a.flags["W"] = False

        view = xp.lib.stride_tricks.as_strided(a, writeable=True)
        assert view.flags["W"] is False

    def test_subok_not_supported(self):
        ia = dpnp.array([1, 2, 3, 4], dtype=dpnp.int32)
        assert_raises(NotImplementedError, as_strided, ia, subok=True)

    @pytest.mark.parametrize(
        "shape, strides",
        [
            ((2000,), (8,)),  # overrun past the highest address
            ((2,), (-8,)),  # underrun before the lowest address
        ],
    )
    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_out_of_bounds(self, shape, strides, xp):
        a = xp.arange(1000, dtype=xp.int64)
        with pytest.raises(ValueError):
            _ = xp.lib.stride_tricks.as_strided(
                a, shape=shape, strides=strides, check_bounds=True
            )

    def test_bounds_use_base_allocation(self):
        # bounds are validated against the whole base allocation, so a view
        # reaching beyond a slice but within its parent is accepted
        a = numpy.arange(1000, dtype=numpy.int64)
        ia = dpnp.array(a)
        b, ib = a[:2], ia[:2]

        result = as_strided(ib, shape=(2,), strides=(400,))
        expected = np_as_strided(b, shape=(2,), strides=(400,))
        assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "start, strides",
        [
            (95, (200,)),  # positive stride overruns the base allocation
            (5, (-48,)),  # negative stride underruns the base allocation
        ],
    )
    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_out_of_bounds_over_slice(self, start, strides, xp):
        a = xp.arange(100, dtype=xp.int64)
        b = a[start : start + 2]

        with pytest.raises(ValueError):
            _ = xp.lib.stride_tricks.as_strided(
                b, shape=(2,), strides=strides, check_bounds=True
            )

    def test_view_with_offset(self):
        a = numpy.arange(1000, dtype=numpy.int64)
        ia = dpnp.array(a)
        b, ib = a[100:102], ia[100:102]

        result = as_strided(ib, shape=(2,), strides=(80,))
        expected = np_as_strided(b, shape=(2,), strides=(80,))
        assert_array_equal(result, expected)

    def test_nested_views(self):
        a = numpy.arange(1000, dtype=numpy.int64)
        ia = dpnp.array(a)
        b, ib = a[10:100], ia[10:100]
        c, ic = b[5:10], ib[5:10]

        result = as_strided(ic, shape=(2,), strides=(160,))
        expected = np_as_strided(c, shape=(2,), strides=(160,))
        assert_array_equal(result, expected)
