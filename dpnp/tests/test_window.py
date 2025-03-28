import numpy
import pytest
from numpy.testing import assert_allclose, assert_raises

import dpnp

from .helper import assert_dtype_allclose


@pytest.mark.parametrize("func", ["bartlett", "blackman", "hamming", "hanning"])
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
        numpy.array(8.0),
    ],
)
def test_window(func, M):
    result = getattr(dpnp, func)(M)

    if isinstance(M, dpnp.ndarray):
        M = M.asnumpy()
    expected = getattr(numpy, func)(M)

    assert_dtype_allclose(result, expected)


@pytest.mark.parametrize("func", ["bartlett", "blackman", "hamming", "hanning"])
@pytest.mark.parametrize(
    "M",
    [
        5 + 4j,
        numpy.array(5 + 4j),
        dpnp.array([5, 3]),
        numpy.inf,
        numpy.array(-numpy.inf),
        dpnp.array(dpnp.nan),
    ],
)
def test_window_error(func, M):
    assert_raises(TypeError, getattr(dpnp, func), M)


class TestKaiser:
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
            numpy.array(8.0),
        ],
    )
    def test_kaiser_M(self, M):
        result = dpnp.kaiser(M, 14)

        if isinstance(M, dpnp.ndarray):
            M = M.asnumpy()
        expected = numpy.kaiser(M, 14)

        assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize(
        "beta",
        [
            True,
            False,
            14,
            dpnp.int32(14),
            14,
            -14.0,
            dpnp.float32(14),
            dpnp.array(-14),
            numpy.array(14.0),
            numpy.inf,
            numpy.array(-numpy.inf),
            dpnp.array(dpnp.nan),
        ],
    )
    def test_kaiser_beta(self, beta):
        result = dpnp.kaiser(4, beta)

        if isinstance(beta, dpnp.ndarray):
            beta = beta.asnumpy()
        expected = numpy.kaiser(4, beta)

        assert_allclose(result, expected, rtol=1e-6)

    @pytest.mark.parametrize(
        "beta",
        [5 + 4j, numpy.array(5 + 4j), dpnp.array([5, 3])],
    )
    def test_kaiser_error(self, beta):
        assert_raises(TypeError, dpnp.kaiser, 4, beta)
