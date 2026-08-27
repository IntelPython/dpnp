import numpy as np
import pytest
from numpy.testing import assert_array_equal

import dpnp

from .third_party.cupy import testing


class TestFlatiter:
    @pytest.mark.parametrize(
        "a, index",
        [
            (np.array([1, 0, 2, -3, -1, 2, 21, -9]), 0),
            (np.arange(1, 7).reshape(2, 3), 3),
            (np.arange(1, 7).reshape(2, 3).T, 3),
        ],
        ids=["1D array", "2D array", "2D.T array"],
    )
    def test_flat_getitem(self, a, index):
        ia = dpnp.array(a)
        result = ia.flat[index]
        expected = a.flat[index]
        assert_array_equal(expected, result)

    def test_flat_iteration(self):
        a = np.array([[1, 2], [3, 4]])
        ia = dpnp.array(a)
        for ival, val in zip(ia.flat, a.flat):
            assert ival == val

    def test_init_error(self):
        with pytest.raises(TypeError, match="must be of type dpnp.ndarray"):
            dpnp.flatiter([1, 2, 3])

    @pytest.mark.parametrize("xp", [dpnp, np])
    def test_flat_key_error(self, xp):
        a = xp.array(42)
        with pytest.raises(IndexError):
            _ = a.flat[1]

    @pytest.mark.parametrize("xp", [dpnp, np])
    def test_flat_invalid_key(self, xp):
        flat = xp.array([1, 2, 3]).flat

        # check __getitem__
        with pytest.raises(IndexError):
            _ = flat["invalid"]

        # check __setitem__
        with pytest.raises(IndexError):
            flat["invalid"] = 42

    @pytest.mark.parametrize("xp", [dpnp, np])
    def test_flat_out_of_bounds(self, xp):
        flat = xp.array([1, 2, 3]).flat
        with pytest.raises(IndexError):
            _ = flat[10]

    @pytest.mark.parametrize(
        "key",
        [
            slice(1, 4),
            slice(None),
            slice(None, None, 2),
            [0, 2, 4],
            [-1, -2],
            Ellipsis,
        ],
        ids=["slice", "full_slice", "step_slice", "list", "neg_list", "..."],
    )
    def test_flat_getitem_index_types(self, key):
        a = np.arange(1, 7).reshape(2, 3)
        ia = dpnp.array(a)
        assert_array_equal(ia.flat[key], a.flat[key])

    @pytest.mark.parametrize(
        "key",
        [slice(1, 4), slice(None), [0, 2, 4], [-1, -2], Ellipsis],
        ids=["slice", "full_slice", "list", "neg_list", "..."],
    )
    def test_flat_setitem_index_types(self, key):
        a = np.arange(1, 7).reshape(2, 3)
        ia = dpnp.array(a)
        a.flat[key] = 0
        ia.flat[key] = 0
        assert_array_equal(ia, a)

    def test_flat_index_array(self):
        a = np.arange(1, 7).reshape(2, 3)
        ia = dpnp.array(a)

        # int array index
        assert_array_equal(ia.flat[dpnp.array([0, 3, 5])], a.flat[[0, 3, 5]])

    @testing.with_requires("numpy>=2.4")
    def test_flat_bool_mask(self):
        a = np.arange(1, 7).reshape(2, 3)
        ia = dpnp.array(a)
        mask = np.array([True, False] * 3)

        # getitem via bool array
        assert_array_equal(ia.flat[dpnp.array(mask)], a.flat[mask])

        # setitem via bool array
        a.flat[mask] = -1
        ia.flat[dpnp.array(mask)] = -1
        assert_array_equal(ia, a)

    def test_flat_non_contiguous(self):
        # C-order traversal + write-back for non-contiguous arrays
        a = np.arange(1, 7).reshape(2, 3).T
        ia = dpnp.array(np.arange(1, 7).reshape(2, 3)).T
        assert_array_equal(ia.flat[1:5], a.flat[1:5])
        a.flat[1:5] = 0
        ia.flat[1:5] = 0
        assert_array_equal(ia, a)

    @pytest.mark.parametrize("xp", [dpnp, np])
    def test_flat_getitem_returns_copy(self, xp):
        # flat yields copies, not views
        a = xp.arange(10)
        s = a.flat[1:4]
        s[0] = 999
        assert a[1] != 999

    def test_flat_scalar_getitem_returns_copy(self):
        # dpnp returns a 0-d array copy (NumPy returns an immutable scalar)
        ia = dpnp.arange(10)
        x = ia.flat[3]
        x[...] = 777
        assert ia[3] != 777

    @testing.with_requires("numpy>=2.4")
    @pytest.mark.parametrize("xp", [dpnp, np])
    def test_flat_newaxis(self, xp):
        a = xp.array([1, 2, 3])
        with pytest.raises(IndexError, match="are valid indices"):
            _ = a.flat[None]
        with pytest.raises(IndexError, match="are valid indices"):
            a.flat[None] = 0

    @testing.with_requires("numpy>=2.4")
    @pytest.mark.parametrize("xp", [dpnp, np])
    def test_flat_empty_tuple(self, xp):
        a = xp.arange(1, 7).reshape(2, 3)
        # getitem with () returns the whole flattened array
        assert_array_equal(a.flat[()], xp.arange(1, 7))
        # setitem with a 0-d index is unsupported
        with pytest.raises(IndexError, match="0-D index is not supported"):
            a.flat[()] = 0

    @testing.with_requires("numpy>=2.4")
    @pytest.mark.parametrize("xp", [dpnp, np])
    @pytest.mark.parametrize("key", [[100], [-100]], ids=["oob", "neg_oob"])
    def test_flat_array_out_of_bounds(self, xp, key):
        a = xp.array([1, 2, 3])
        with pytest.raises(IndexError, match="out of bounds for size"):
            _ = a.flat[key]
        with pytest.raises(IndexError, match="out of bounds for size"):
            a.flat[key] = 0
