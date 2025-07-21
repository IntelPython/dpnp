import dpctl.tensor as dpt
import numpy
import pytest

import dpnp
import dpnp.memory as dpm


class IntUsmData(dpt.usm_ndarray):
    """Class that overrides `usm_data` property in `dpt.usm_ndarray`."""

    @property
    def usm_data(self):
        return 1


class TestCreateData:
    @pytest.mark.parametrize("x", [numpy.ones(4), dpnp.zeros(2)])
    def test_wrong_input_type(self, x):
        with pytest.raises(TypeError):
            dpm.create_data(x)

    def test_wrong_usm_data(self):
        a = dpt.ones(10)
        d = IntUsmData(a.shape, buffer=a)

        with pytest.raises(TypeError):
            dpm.create_data(d)

    def test_ndarray_from_data(self):
        a = dpnp.empty(5)
        b = dpnp.ndarray(a.shape, buffer=a.data)
        assert b.data.ptr == a.data.ptr
