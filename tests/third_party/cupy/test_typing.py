import pytest

import dpnp as cupy


@pytest.mark.skip("dpnp.typing is not implemented yet")
class TestClassGetItem:

    def test_class_getitem(self):
        from typing import Any

        cupy.ndarray[Any, Any]
