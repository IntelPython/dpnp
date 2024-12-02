import unittest

import pytest

# from cupy import _core

pytest.skip("owndata attribute is not supported", allow_module_level=True)


class TestArrayOwndata(unittest.TestCase):

    def setUp(self):
        self.a = _core.ndarray(())

    def test_original_array(self):
        assert self.a.flags.owndata is True

    def test_view_array(self):
        v = self.a.view()
        assert v.flags.owndata is False

    def test_reshaped_array(self):
        r = self.a.reshape(())
        assert r.flags.owndata is False
