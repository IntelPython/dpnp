import dpnp


def test_byte_bounds():
    a = dpnp.zeros((3, 4), dtype=dpnp.int64, usm_type="shared")
    values = dpnp.arange(12, dtype="int64")
    for i in range(3):
        a[i, :] = values[i * 4 : (i + 1) * 4]
    low, high = dpnp.byte_bounds(a)
    assert (high - low) == (a.size * a.itemsize)


def test_unusual_order_positive_stride():
    a = dpnp.zeros((3, 4), dtype=dpnp.int64, usm_type="shared")
    values = dpnp.arange(12, dtype="int64")
    for i in range(3):
        a[i, :] = values[i * 4 : (i + 1) * 4]
    b = a.T
    low, high = dpnp.byte_bounds(b)
    assert (high - low) == (b.size * b.itemsize)


def test_unusual_order_negative_stride():
    a = dpnp.zeros((3, 4), dtype=dpnp.int64, usm_type="shared")
    values = dpnp.arange(12, dtype="int64")
    for i in range(3):
        a[i, :] = values[i * 4 : (i + 1) * 4]
    b = a.T[::-1]
    low, high = dpnp.byte_bounds(b)
    assert (high - low) == (b.size * b.itemsize)


def test_strided():
    a = dpnp.zeros(12, dtype=dpnp.int64, usm_type="shared")
    a[:] = dpnp.arange(12, dtype="int64")
    b = a[::2]
    low, high = dpnp.byte_bounds(b)
    expected_byte_diff = b.size * 2 * b.itemsize - b.itemsize
    assert (high - low) == expected_byte_diff
