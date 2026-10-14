import pytest

import dpnp


class TestAll:
    def test_no_duplicates(self):
        assert len(dpnp.__all__) == len(set(dpnp.__all__))

    def test_names_resolve(self):
        missing = [name for name in dpnp.__all__ if not hasattr(dpnp, name)]
        assert not missing

    @pytest.mark.parametrize("name", ["__array_namespace_info__", "ndarray"])
    def test_contains(self, name):
        assert name in dpnp.__all__
