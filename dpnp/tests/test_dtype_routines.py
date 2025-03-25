import numpy
import pytest
from numpy.testing import assert_raises_regex

import dpnp

from .helper import numpy_version

if numpy_version() >= "2.0.0":
    from numpy._core.numerictypes import sctypes
else:
    from numpy.core.numerictypes import sctypes


class TestIsDType:
    dtype_group = {
        "signed integer": sctypes["int"],
        "unsigned integer": sctypes["uint"],
        "integral": sctypes["int"] + sctypes["uint"],
        "real floating": sctypes["float"],
        "complex floating": sctypes["complex"],
        "numeric": (
            sctypes["int"]
            + sctypes["uint"]
            + sctypes["float"]
            + sctypes["complex"]
        ),
    }

    @pytest.mark.parametrize(
        "dt, close_dt",
        [
            (dpnp.int64, dpnp.int32),
            (dpnp.uint64, dpnp.uint32),
            (dpnp.float64, dpnp.float32),
            (dpnp.complex128, dpnp.complex64),
        ],
    )
    @pytest.mark.parametrize("dt_group", [None] + list(dtype_group.keys()))
    def test_basic(self, dt, close_dt, dt_group):
        # First check if same dtypes return "True" and different ones
        # give "False" (even if they're close in the dtype hierarchy).
        if dt_group is None:
            assert dpnp.isdtype(dt, dt)
            assert not dpnp.isdtype(dt, close_dt)
            assert dpnp.isdtype(dt, (dt, close_dt))

        # Check that dtype and a dtype group that it belongs to return "True",
        # and "False" otherwise.
        elif dt in self.dtype_group[dt_group]:
            assert dpnp.isdtype(dt, dt_group)
            assert dpnp.isdtype(dt, (close_dt, dt_group))
        else:
            assert not dpnp.isdtype(dt, dt_group)

    def test_invalid_args(self):
        with assert_raises_regex(TypeError, r"Expected instance of.*"):
            dpnp.isdtype("int64", dpnp.int64)

        with assert_raises_regex(TypeError, r"Unsupported data type kind:.*"):
            dpnp.isdtype(dpnp.int64, 1)

        with assert_raises_regex(ValueError, r"Unrecognized data type kind:.*"):
            dpnp.isdtype(dpnp.int64, "int64")
