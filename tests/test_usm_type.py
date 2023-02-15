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


@pytest.mark.parametrize(
    "func, args",
    [
        pytest.param("full",
                     ['10', 'x0[3]']),
        pytest.param("full_like",
                     ['x0', '4']),
        pytest.param("zeros_like",
                     ['x0']),
        pytest.param("ones_like",
                     ['x0']),
        pytest.param("empty_like",
                     ['x0']),
        pytest.param("linspace",
                     ['x0[0:2]', '4', '4']),
        pytest.param("linspace",
                     ['0', 'x0[3:5]', '4']),
    ])
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_array_creation(func, args, usm_type_x, usm_type_y):
    x0 = dp.full(10, 3, usm_type=usm_type_x)
    new_args = [eval(val, {'x0' : x0}) for val in args]

    x = getattr(dp, func)(*new_args)
    y = getattr(dp, func)(*new_args, usm_type=usm_type_y)

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y


@pytest.mark.parametrize("usm_type_start", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_stop", list_of_usm_types, ids=list_of_usm_types)
def test_linspace_arrays(usm_type_start, usm_type_stop):
    start = dp.asarray([0, 0], usm_type=usm_type_start)
    stop = dp.asarray([2, 4], usm_type=usm_type_stop)
    res = dp.linspace(start, stop, 4)
    assert res.usm_type == du.get_coerced_usm_type([usm_type_start, usm_type_stop])
