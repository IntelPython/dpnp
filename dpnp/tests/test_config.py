import re
import warnings

import pytest

import dpnp


@pytest.fixture(autouse=True)
def reset_deprecated_values(monkeypatch):
    # drop values assigned by other tests
    monkeypatch.setattr(dpnp.config, "_deprecated_values", {})


class TestDeprecatedSettings:
    @pytest.mark.parametrize(
        "name, env",
        [
            ("__DPNP_QUEUE_GPU__", "DPNP_QUEUE_GPU"),
            ("__DPNP_OUTPUT_NUMPY__", "DPNP_OUTPUT_NUMPY"),
            (
                "__DPNP_OUTPUT_DPCTL_DEFAULT_SHARED__",
                "DPNP_OUTPUT_DPCTL_DEFAULT_SHARED",
            ),
        ],
    )
    def test_read_warns(self, monkeypatch, name, env):
        monkeypatch.delenv(env, raising=False)
        with pytest.warns(DeprecationWarning, match=re.escape(name)):
            value = getattr(dpnp.config, name)
        assert value == 0

    @pytest.mark.parametrize(
        "name",
        [
            "__DPNP_QUEUE_GPU__",
            "__DPNP_OUTPUT_NUMPY__",
            "__DPNP_OUTPUT_DPCTL_DEFAULT_SHARED__",
        ],
    )
    def test_write_warns(self, name):
        with pytest.warns(DeprecationWarning, match=re.escape(name)):
            setattr(dpnp.config, name, 1)

        # a later read must still warn
        with pytest.warns(DeprecationWarning, match=re.escape(name)):
            value = getattr(dpnp.config, name)
        assert value == 1

    @pytest.mark.parametrize(
        "name",
        ["__DPNP_ORIGIN__", "__DPNP_RAISE_EXCEPION_ON_NUMPY_FALLBACK__"],
    )
    def test_no_warning(self, monkeypatch, name):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = getattr(dpnp.config, name)
            monkeypatch.setattr(dpnp.config, name, value)

    def test_unknown_name(self):
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(dpnp.config, "__DPNP_UNKNOWN__")
