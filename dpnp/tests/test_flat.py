import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_raises

import dpnp


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
        a_dp = dpnp.array(a)
        result = a_dp.flat[index]
        expected = a.flat[index]
        assert_array_equal(expected, result)

    def test_flat_iteration(self):
        a = np.array([[1, 2], [3, 4]])
        a_dp = dpnp.array(a)
        for dp_val, np_val in zip(a_dp.flat, a.flat):
            assert dp_val == np_val

    def test_init_error(self):
        assert_raises(TypeError, dpnp.flatiter, [1, 2, 3])

    def test_flat_key_error(self):
        a_dp = dpnp.array(42)
        with pytest.raises(KeyError):
            _ = a_dp.flat[1]

    def test_flat_invalid_key(self):
        a_dp = dpnp.array([1, 2, 3])
        flat = dpnp.flatiter(a_dp)
        # check __getitem__
        with pytest.raises(TypeError):
            _ = flat["invalid"]
        # check __setitem__
        with pytest.raises(TypeError):
            flat["invalid"] = 42

    def test_flat_out_of_bounds(self):
        a_dp = dpnp.array([1, 2, 3])
        flat = dpnp.flatiter(a_dp)
        with pytest.raises(IndexError):
            _ = flat[10]
