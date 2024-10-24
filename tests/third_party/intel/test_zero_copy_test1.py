import importlib
import sys

import pytest


class dummymodule:
    pass


sys.modules["numba_dppy"] = dummymodule

module_not_found = False

reason = ""

try:
    zero_copy_test1 = importlib.import_module("zero-copy-test1")
except ModuleNotFoundError as e:
    module_not_found = True
    reason = str(e)


@pytest.mark.skipif(module_not_found, reason=reason)
def test_dpnp_interaction_with_dpctl_memory():
    return zero_copy_test1.test_dpnp_interaction_with_dpctl_memory()


@pytest.mark.skipif(module_not_found, reason=reason)
def test_dpnp_array_has_iface():
    return zero_copy_test1.test_dpnp_array_has_iface()


@pytest.mark.skipif(module_not_found, reason=reason)
def test_dpctl_dparray_has_iface():
    return zero_copy_test1.test_dpctl_dparray_has_iface()
