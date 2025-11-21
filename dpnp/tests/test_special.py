import numpy
import pytest
from numpy.testing import assert_allclose, assert_equal

import dpnp

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_complex_dtypes,
)
from .third_party.cupy.testing import installed, with_requires


@with_requires("scipy")
@pytest.mark.parametrize("func", ["erf", "erfc"])
class TestCommon:
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_float16=False, no_complex=True)
    )
    def test_basic(self, func, dt):
        import scipy.special

        a = generate_random_numpy_array((2, 5), dtype=dt)
        ia = dpnp.array(a)

        result = getattr(dpnp.scipy.special, func)(ia)
        expected = getattr(scipy.special, func)(a)

        # scipy >= 0.16.0 returns float64, but dpnp returns float32
        to_float32 = dpnp.result_type(dt, dpnp.float32) == dpnp.float32
        only_type_kind = installed("scipy>=0.16.0") and to_float32
        assert_dtype_allclose(
            result, expected, check_only_type_kind=only_type_kind
        )

    def test_nan_inf(self, func):
        import scipy.special

        a = numpy.array([numpy.nan, -numpy.inf, numpy.inf])
        ia = dpnp.array(a)

        result = getattr(dpnp.scipy.special, func)(ia)
        expected = getattr(scipy.special, func)(a)
        assert_allclose(result, expected)

    def test_zeros(self, func):
        import scipy.special

        a = numpy.array([0.0, -0.0])
        ia = dpnp.array(a)

        result = getattr(dpnp.scipy.special, func)(ia)
        expected = getattr(scipy.special, func)(a)
        assert_allclose(result, expected)
        assert_equal(dpnp.signbit(result), numpy.signbit(expected))

    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_complex(self, func, dt):
        x = dpnp.empty(5, dtype=dt)
        with pytest.raises(ValueError):
            getattr(dpnp.scipy.special, func)(x)


class TestConsistency:

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_erfc(self):
        # TODO: replace with dpnp.random.RandomState, once pareto is added
        rng = numpy.random.RandomState(1234)
        n = 10000
        a = rng.pareto(0.02, n) * (2 * rng.randint(0, 2, n) - 1)
        a = dpnp.array(a)

        res = 1 - dpnp.scipy.special.erf(a)
        mask = dpnp.isfinite(res)
        a = a[mask]

        tol = 8 * dpnp.finfo(a).resolution
        assert dpnp.allclose(
            dpnp.scipy.special.erfc(a), res[mask], rtol=tol, atol=tol
        )
