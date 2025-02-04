import dpctl
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_array_equal

import dpnp


@pytest.mark.parametrize(
    "val, error",
    [
        pytest.param(dpnp.ones(2, dtype="i4"), ValueError, id="array"),
        pytest.param(dict(), TypeError, id="dictionary"),
        pytest.param("0", TypeError, id="string"),
    ],
)
def test_fill_non_scalar(val, error):
    a = dpnp.ones(5, dtype="i4")
    with pytest.raises(error):
        a.fill(val)


def test_fill_compute_follows_data():
    q1 = dpctl.SyclQueue()
    q2 = dpctl.SyclQueue()

    a = dpnp.ones(5, dtype="i4", sycl_queue=q1)
    val = dpnp.ones((), dtype=a.dtype, sycl_queue=q2)

    with pytest.raises(ExecutionPlacementError):
        a.fill(val)


def test_fill_strided_array():
    a = dpnp.zeros(100, dtype="i4")
    b = a[::-2]

    expected = dpnp.tile(dpnp.asarray([0, 1], dtype=a.dtype), 50)

    b.fill(1)
    assert_array_equal(b, 1, strict=False)
    assert_array_equal(a, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_fill_strided_2d_array(order):
    a = dpnp.zeros((10, 10), dtype="i4", order=order)
    b = a[::-2, ::2]

    expected = dpnp.copy(a)
    expected[::-2, ::2] = 1

    b.fill(1)
    assert_array_equal(b, 1, strict=False)
    assert_array_equal(a, expected)


@pytest.mark.parametrize("order", ["C", "F"])
def test_fill_memset(order):
    a = dpnp.ones((10, 10), dtype="i4", order=order)
    a.fill(0)

    assert_array_equal(a, 0, strict=False)


def test_fill_float_complex_to_int():
    a = dpnp.ones((10, 10), dtype="i4")

    a.fill(complex(2, 0))
    assert_array_equal(a, 2, strict=False)

    a.fill(float(3))
    assert_array_equal(a, 3, strict=False)


def test_fill_complex_to_float():
    a = dpnp.ones((10, 10), dtype="f4")

    a.fill(complex(2, 0))
    assert_array_equal(a, 2, strict=False)


def test_fill_bool():
    a = dpnp.full(5, fill_value=7, dtype="i4")
    a.fill(True)
    assert_array_equal(a, 1, strict=False)
