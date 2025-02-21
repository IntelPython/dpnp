import numpy
import pytest
from numpy.testing import assert_raises

import dpnp

from .helper import assert_dtype_allclose


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("func", ["hamming"])
@pytest.mark.parametrize(
    "M",
    [
        True,
        False,
        0,
        dpnp.int32(1),
        4,
        5.0,
        dpnp.float32(6),
        dpnp.array(7),
        numpy.array(8),
    ],
)
def test_window(func, M):
    result = getattr(dpnp, func)(M)

    if isinstance(M, dpnp.ndarray):
        M = M.asnumpy()
    expected = getattr(numpy, func)(M)

    assert_dtype_allclose(result, expected)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("func", ["hamming"])
@pytest.mark.parametrize(
    "M",
    [
        5 + 4j,
        numpy.array(5 + 4j),
        dpnp.array([5]),
        numpy.inf,
        numpy.array(-numpy.inf),
        dpnp.array(dpnp.nan),
    ],
)
def test_window_error(func, M):
    assert_raises(TypeError, getattr(dpnp, func), M)
