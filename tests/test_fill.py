import dpctl
import numpy as np
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_array_equal

import dpnp as dnp


def test_fill_non_scalar():
    a = dnp.ones(5, dtype="i4")
    val = dnp.ones(2, dtype="i4")

    with pytest.raises(ValueError):
        a.fill(val)

    with pytest.raises(TypeError):
        a.fill(dict())


def test_fill_compute_follows_data():
    q1 = dpctl.SyclQueue()
    q2 = dpctl.SyclQueue()

    a = dnp.ones(5, dtype="i4", sycl_queue=q1)
    val = dnp.ones((), dtype=a.dtype, sycl_queue=q2)

    with pytest.raises(ExecutionPlacementError):
        a.fill(val)


def test_fill_strided_array():
    a = dnp.zeros(100, dtype="i4")
    b = a[::-2]

    expected = dnp.tile(dnp.asarray([0, 1], dtype=a.dtype), 50)

    b.fill(1)
    assert_array_equal(b, 1)
    assert_array_equal(a, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_fill_strided_2d_array(order):
    a = dnp.zeros((10, 10), dtype="i4", order=order)
    b = a[::-2, ::2]

    expected = dnp.copy(a)
    expected[::-2, ::2] = 1

    b.fill(1)
    assert_array_equal(b, 1)
    assert_array_equal(a, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_fill_memset(order):
    a = dnp.ones((10, 10), dtype="i4", order=order)
    a.fill(0)

    assert_array_equal(a, 0)
