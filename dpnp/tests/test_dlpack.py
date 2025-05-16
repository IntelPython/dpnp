import dpctl
import numpy
import pytest
from numpy.testing import assert_array_equal, assert_raises

import dpnp

from .helper import (
    get_all_dtypes,
)


class TestDLPack:
    @pytest.mark.parametrize("stream", [None, dpctl.SyclQueue()])
    def test_stream(self, stream):
        x = dpnp.arange(5)
        x.__dlpack__(stream=stream)

    @pytest.mark.parametrize(
        "stream",
        [1, dict(), dpctl.SyclDevice()],
        ids=["scalar", "dictionary", "device"],
    )
    def test_invaid_stream(self, stream):
        x = dpnp.arange(5)
        assert_raises(TypeError, x.__dlpack__, stream=stream)

    @pytest.mark.parametrize("copy", [True, None, False])
    def test_copy(self, copy):
        x = dpnp.arange(5)
        x.__dlpack__(copy=copy)

    def test_wrong_copy(self):
        x = dpnp.arange(5)
        x.__dlpack__(copy=dpnp.array([1, 2, 3]))

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    @pytest.mark.parametrize("dt", get_all_dtypes(no_none=True))
    def test_dtype_passthrough(self, xp, dt):
        x = xp.arange(5).astype(dt)
        y = xp.from_dlpack(x)

        assert_array_equal(x, y)

    @pytest.mark.parametrize("xp", [dpnp, numpy])
    def test_non_contiguous(self, xp):
        x = xp.arange(25).reshape((5, 5))

        y1 = x[0]
        assert_array_equal(y1, xp.from_dlpack(y1))

        y2 = x[:, 0]
        assert_array_equal(y2, xp.from_dlpack(y2))

        y3 = x[1, :]
        assert_array_equal(y3, xp.from_dlpack(y3))

        y4 = x[1]
        assert_array_equal(y4, xp.from_dlpack(y4))

        y5 = xp.diagonal(x).copy()
        assert_array_equal(y5, xp.from_dlpack(y5))

    def test_device(self):
        x = dpnp.arange(5)
        assert x.__dlpack_device__()[0] == dpnp.DLDeviceType.kDLOneAPI
        y = dpnp.from_dlpack(x)
        assert y.__dlpack_device__()[0] == dpnp.DLDeviceType.kDLOneAPI
        z = y[::2]
        assert z.__dlpack_device__()[0] == dpnp.DLDeviceType.kDLOneAPI

    def test_ndim0(self):
        x = dpnp.array(1.0)
        y = dpnp.from_dlpack(x)
        assert_array_equal(x, y)

    def test_device(self):
        x = dpnp.arange(5)
        y = dpnp.from_dlpack(x, device=x.__dlpack_device__())
        assert x.device == y.device
        assert x.get_array()._pointer == y.get_array()._pointer

    def test_numpy_input(self):
        x = numpy.arange(10)

        y = dpnp.from_dlpack(x)
        assert isinstance(y, numpy.ndarray)
        assert y.ctypes.data == x.ctypes.data
        assert y.dtype == x.dtype

        z = dpnp.from_dlpack(x, device=(dpnp.DLDeviceType.kDLCPU, 0))
        assert isinstance(z, numpy.ndarray)
        assert z.dtype == y.dtype
