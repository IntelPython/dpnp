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
            slice(None, None, -1),
            [0, 2, 4],
            [-1, -2],
            Ellipsis,
        ],
        ids=[
            "slice",
            "full_slice",
            "step_slice",
            "neg_step_slice",
            "list",
            "neg_list",
            "...",
        ],
    )
    def test_flat_getitem_index_types(self, key):
        a = np.arange(1, 7).reshape(2, 3)
        ia = dpnp.array(a)
        assert_array_equal(ia.flat[key], a.flat[key])

    @pytest.mark.parametrize(
        "key",
        [
            slice(1, 4),
            slice(None),
            slice(None, None, 2),
            slice(None, None, -1),
            [0, 2, 4],
            [-1, -2],
            Ellipsis,
        ],
        ids=[
            "slice",
            "full_slice",
            "step_slice",
            "neg_step_slice",
            "list",
            "neg_list",
            "...",
        ],
    )
    def test_flat_setitem_index_types(self, key):
        a = np.arange(1, 7).reshape(2, 3)
        ia = dpnp.array(a)
        a.flat[key] = 0
        ia.flat[key] = 0
        assert_array_equal(ia, a)

    @pytest.mark.parametrize("index", [0, 5, -1, -6])
    def test_flat_setitem_scalar(self, index):
        a = np.arange(1, 7)
        ia = dpnp.array(a)
        a.flat[index] = 99
        ia.flat[index] = 99
        assert_array_equal(ia, a)

    @pytest.mark.parametrize("xp", [dpnp, np])
    @pytest.mark.parametrize("index", [6, -7], ids=["oob", "neg_oob"])
    def test_flat_setitem_scalar_out_of_bounds(self, xp, index):
        a = xp.arange(1, 7)
        with pytest.raises(IndexError, match="out of bounds"):
            a.flat[index] = 0

    @testing.with_requires("numpy>=2.4")
    @pytest.mark.parametrize("xp", [dpnp, np])
    def test_flat_setitem_single_item_array_value(self, xp):
        for index in (0, xp.array(0), np.int64(0)):
            a = xp.arange(1, 7)
            with pytest.raises(ValueError, match="single item"):
                a.flat[index] = [1, 2, 3]

    def test_flat_setitem_single_item_scalar_value(self):
        a = np.arange(1, 7)
        ia = dpnp.array(a)

        a.flat[0] = 9
        a.flat[np.array(1)] = np.asarray(8)

        ia.flat[0] = 9
        ia.flat[dpnp.array(1)] = dpnp.asarray(8)
        assert_array_equal(ia, a)

    def test_flat_setitem_length_one_slice_cycles(self):
        a = np.arange(1, 7)
        ia = dpnp.array(a)

        a.flat[0:1] = [10, 20, 30]
        ia.flat[0:1] = [10, 20, 30]
        assert_array_equal(ia, a)

    def test_flat_index_array(self):
        a = np.arange(1, 7).reshape(2, 3)
        ia = dpnp.array(a)

        # int array index
        assert_array_equal(ia.flat[dpnp.array([0, 3, 5])], a.flat[[0, 3, 5]])

    def test_flat_usm_ndarray_index(self):
        a = np.arange(1, 7)
        ia = dpnp.array(a)

        # a usm_ndarray index is validated and used like a dpnp array
        usm_key = dpnp.array([0, 2, 4]).get_array()
        assert_array_equal(ia.flat[usm_key], a.flat[[0, 2, 4]])
        with pytest.raises(IndexError, match="out of bounds"):
            _ = ia.flat[dpnp.array([100]).get_array()]

    @pytest.mark.parametrize("xp", [dpnp, np])
    def test_flat_empty_index(self, xp):
        a = xp.arange(1, 7)
        assert_array_equal(a.flat[xp.array([], dtype=xp.intp)], a.flat[[]])

    def test_flat_single_element_tuple(self):
        a = np.arange(1, 7)
        ia = dpnp.array(a)

        # a 1-element index tuple is equivalent to the bare index
        assert_array_equal(ia.flat[(0,)], a.flat[(0,)])
        assert_array_equal(ia.flat[(slice(1, 4),)], a.flat[(slice(1, 4),)])
        assert_array_equal(
            ia.flat[(dpnp.array([0, 2]),)], a.flat[(np.array([0, 2]),)]
        )

    @pytest.mark.parametrize("xp", [dpnp, np])
    @pytest.mark.parametrize(
        "key",
        [(Ellipsis, 2), (2, Ellipsis), (Ellipsis, slice(1, 3)), (1, 2)],
        ids=["ell_int", "int_ell", "ell_slice", "int_int"],
    )
    def test_flat_multi_element_tuple(self, xp, key):
        a = xp.arange(6)
        with pytest.raises(IndexError):
            _ = a.flat[key]
        with pytest.raises(IndexError):
            a.flat[key] = 0

    @pytest.mark.parametrize("xp", [dpnp, np])
    def test_flat_tuple_array_out_of_bounds(self, xp):
        a = xp.array([1, 2, 3])
        idx = (xp.array([5]),)
        with pytest.raises(IndexError, match="out of bounds"):
            _ = a.flat[idx]
        with pytest.raises(IndexError, match="out of bounds"):
            a.flat[idx] = 0

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

    @pytest.mark.parametrize("xp", [dpnp, np])
    @pytest.mark.parametrize("key", [[100], [-100]], ids=["oob", "neg_oob"])
    def test_flat_array_out_of_bounds(self, xp, key):
        a = xp.array([1, 2, 3])
        with pytest.raises(IndexError, match="out of bounds for size"):
            _ = a.flat[key]
        with pytest.raises(IndexError, match="out of bounds for size"):
            a.flat[key] = 0
