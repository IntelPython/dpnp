import pytest

import dpnp as dp

import dpctl.utils as du

list_of_usm_types = [
    "device",
    "shared",
    "host"
]


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_sum(usm_type):
    x = dp.arange(10, usm_type = "device")
    y = dp.arange(10, usm_type = usm_type)

    z = x + y
    
    assert z.usm_type == x.usm_type
    assert z.usm_type == "device"
    assert y.usm_type == usm_type


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_mul(usm_type_x, usm_type_y):
    x = dp.arange(10, usm_type = usm_type_x)
    y = dp.arange(10, usm_type = usm_type_y)

    z = x * y
    
    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_array_creation(usm_type_x, usm_type_y):
    x0 = dp.full(10, 3, usm_type=usm_type_x)

    x = dp.full(10, x0[3])
    y = dp.full(10, x0[3], usm_type=usm_type_y)
    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y

    x = dp.full_like(x0, 4)
    y = dp.full_like(x0, 4, usm_type=usm_type_y)
    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y

    x = dp.zeros_like(x0)
    y = dp.zeros_like(x0, usm_type=usm_type_y)
    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
