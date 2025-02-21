import numpy
import pytest
from numpy.testing import assert_allclose, assert_raises

import dpnp


@pytest.mark.parametrize(
    "M", [0, 1, 5.0, numpy.int64(0), numpy.int32(1), numpy.float32(5)]
)
def test_hamming(M):
    assert_allclose(dpnp.hamming(M), numpy.hamming(M))


@pytest.mark.parametrize("M", [5 + 4j, numpy.array(5)])
def test_hamming_error(M):
    assert_raises(TypeError, dpnp.hamming, M)
