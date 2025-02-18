import copy

import numpy
import pytest
from numpy.testing import assert_allclose, assert_equal

import dpnp


class TestCopyOrder:
    a = dpnp.arange(24).reshape(2, 1, 3, 4)
    b = a.copy(order="F")
    c = dpnp.arange(24).reshape(2, 1, 4, 3).swapaxes(2, 3)

    def check_result(self, x, y, c_contig, f_contig):
        assert not (x is y)
        assert x.flags.c_contiguous == c_contig
        assert x.flags.f_contiguous == f_contig
        assert_equal(x, y)

    @pytest.mark.parametrize("arr", [a, b, c])
    def test_order_c(self, arr):
        res = arr.copy(order="C")
        self.check_result(res, arr, c_contig=True, f_contig=False)

        res = dpnp.copy(arr, order="C")
        self.check_result(res, arr, c_contig=True, f_contig=False)

    @pytest.mark.parametrize("arr", [a, b, c])
    def test_order_f(self, arr):
        res = arr.copy(order="F")
        self.check_result(res, arr, c_contig=False, f_contig=True)

        res = dpnp.copy(arr, order="F")
        self.check_result(res, arr, c_contig=False, f_contig=True)

    @pytest.mark.parametrize("arr", [a, b, c])
    def test_order_k(self, arr):
        res = arr.copy(order="K")
        self.check_result(
            res,
            arr,
            c_contig=arr.flags.c_contiguous,
            f_contig=arr.flags.f_contiguous,
        )

        res = dpnp.copy(arr, order="K")
        self.check_result(
            res,
            arr,
            c_contig=arr.flags.c_contiguous,
            f_contig=arr.flags.f_contiguous,
        )

        res = copy.copy(arr)
        self.check_result(
            res,
            arr,
            c_contig=arr.flags.c_contiguous,
            f_contig=arr.flags.f_contiguous,
        )


@pytest.mark.parametrize(
    "val",
    [3.7, numpy.arange(7), [2, 7, 3.6], (-3, 4), range(4)],
    ids=["scalar", "numpy.array", "list", "tuple", "range"],
)
def test_copy_not_dpnp_array(val):
    a = dpnp.copy(val)
    assert_allclose(a, val)
