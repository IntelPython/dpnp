import pytest

import dpnp


class TestDeprecatedSettings:
    @pytest.mark.parametrize(
        "name",
        [
            "__DPNP_QUEUE_GPU__",
            "__DPNP_OUTPUT_NUMPY__",
            "__DPNP_OUTPUT_DPCTL_DEFAULT_SHARED__",
        ],
    )
    def test_warns(self, name):
        with pytest.warns(DeprecationWarning, match="has no effect"):
            value = getattr(dpnp.config, name)
        assert isinstance(value, int)

    def test_unknown_name(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(dpnp.config, "__DPNP_UNKNOWN__")
