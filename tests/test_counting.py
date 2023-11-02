import numpy
import pytest
from numpy.testing import (
    assert_allclose,
)

import dpnp

from .helper import (
    get_all_dtypes,
)


@pytest.mark.parametrize("dtype", get_all_dtypes())
@pytest.mark.parametrize("size", [2, 4, 8, 16, 3, 9, 27, 81])
def test_count_nonzero(dtype, size):
    a = numpy.arange(size, dtype=dtype)
    for i in range(int(size / 2)):
        a[(i * (int(size / 3) - 1)) % size] = 0

    ia = dpnp.array(a)

    np_res = numpy.count_nonzero(a)
    dpnp_res = dpnp.count_nonzero(ia)

    assert_allclose(dpnp_res, np_res)
