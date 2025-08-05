import numpy
import pytest
from numpy.testing import assert_allclose

import dpnp

from .helper import (
    assert_dtype_allclose,
    generate_random_numpy_array,
    get_all_dtypes,
    get_complex_dtypes,
)
from .third_party.cupy.testing import installed, with_requires


@with_requires("scipy")
class TestErf:
    @pytest.mark.parametrize(
        "dt", get_all_dtypes(no_none=True, no_float16=False, no_complex=True)
    )
    def test_basic(self, dt):
        import scipy.special

        a = generate_random_numpy_array((2, 5), dtype=dt)
        ia = dpnp.array(a)

        result = dpnp.special.erf(ia)
        expected = scipy.special.erf(a)

        # scipy >= 0.16.0 returns float64, but dpnp returns float32
        to_float32 = dt in (dpnp.bool, dpnp.float16)
        only_type_kind = installed("scipy>=0.16.0") and to_float32
        assert_dtype_allclose(
            result, expected, check_only_type_kind=only_type_kind
        )

    def test_nan_inf(self):
        import scipy.special

        a = numpy.array([numpy.nan, -numpy.inf, numpy.inf])
        ia = dpnp.array(a)

        result = dpnp.special.erf(ia)
        expected = scipy.special.erf(a)
        assert_allclose(result, expected)

    @pytest.mark.parametrize("dt", get_complex_dtypes())
    def test_complex(self, dt):
        x = dpnp.empty(5, dtype=dt)
        with pytest.raises(ValueError):
            dpnp.special.erf(x)
