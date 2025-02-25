import numpy
import pytest
from numpy.testing import assert_raises

import dpnp

from .helper import assert_dtype_allclose


@pytest.mark.parametrize("func", ["hamming"])
@pytest.mark.parametrize(
    "M", [0, 1, 5.0, numpy.int64(0), numpy.int32(1), numpy.float32(5)]
)
def test_window(func, M):
    result = getattr(dpnp, func)(M)
    expected = getattr(numpy, func)(M)

    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("func", ["hamming"])
@pytest.mark.parametrize("M", [5 + 4j, numpy.array(5)])
def test_window_error(func, M):
    assert_raises(TypeError, getattr(dpnp, func), M)
