import pytest

import dpnp as dp

import dpctl.utils as du

list_of_usm_types = [
    "device",
    "shared",
    "host"
]


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_sum(usm_type_x, usm_type_y):
    x = dp.arange(1000, usm_type = usm_type_x)
    y = dp.arange(1000, usm_type = usm_type_y)

    z = 1.3 + x + y + 2

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_mul(usm_type_x, usm_type_y):
    x = dp.arange(10, usm_type = usm_type_x)
    y = dp.arange(10, usm_type = usm_type_y)

    z = 3 * x * y * 1.5

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

@pytest.mark.parametrize("op",
                         ['equal', 'greater', 'greater_equal', 'less', 'less_equal',
                          'logical_and', 'logical_or', 'logical_xor', 'not_equal'],
                         ids=['equal', 'greater', 'greater_equal', 'less', 'less_equal',
                              'logical_and', 'logical_or', 'logical_xor', 'not_equal'])
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_logic_op(op, usm_type_x, usm_type_y):
    x = dp.arange(100, usm_type = usm_type_x)
    y = dp.arange(100, usm_type = usm_type_y)[::-1]

    z = getattr(dp, op)(x, y)
    zx = getattr(dp, op)(x, 50)
    zy = getattr(dp, op)(30, y)

    assert x.usm_type == zx.usm_type == usm_type_x
    assert y.usm_type == zy.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])
