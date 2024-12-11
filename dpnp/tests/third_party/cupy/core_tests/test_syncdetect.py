import unittest

import pytest

import dpnp as cupy

# import cupyx

pytest.skip("get() method is not supported", allow_module_level=True)


class TestSyncDetect(unittest.TestCase):

    def test_disallowed(self):
        a = cupy.array([2, 3])
        with cupyx.allow_synchronize(False):
            with pytest.raises(cupyx.DeviceSynchronized):
                a.get()

    def test_allowed(self):
        a = cupy.array([2, 3])
        with cupyx.allow_synchronize(True):
            a.get()

    def test_nested_disallowed(self):
        a = cupy.array([2, 3])
        with cupyx.allow_synchronize(True):
            with cupyx.allow_synchronize(False):
                with pytest.raises(cupyx.DeviceSynchronized):
                    a.get()

    def test_nested_allowed(self):
        a = cupy.array([2, 3])
        with cupyx.allow_synchronize(False):
            with cupyx.allow_synchronize(True):
                a.get()
