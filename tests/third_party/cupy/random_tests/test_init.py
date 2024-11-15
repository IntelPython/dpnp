import pytest

import dpnp as cupy


@pytest.mark.usefixtures("allow_fall_back_on_numpy")
def test_bytes():
    out = cupy.random.bytes(10)
    assert isinstance(out, bytes)
