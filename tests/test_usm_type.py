import pytest

import dpnp as dp

import dpctl.utils as du

list_of_usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_sum(usm_type_x, usm_type_y):
    x = dp.arange(1000, usm_type=usm_type_x)
    y = dp.arange(1000, usm_type=usm_type_y)

    z = 1.3 + x + y + 2

    # inplace add
    z += x
    z += 7.4

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_mul(usm_type_x, usm_type_y):
    x = dp.arange(10, usm_type=usm_type_x)
    y = dp.arange(10, usm_type=usm_type_y)

    z = 3 * x * y * 1.5

    # inplace multiply
    z *= x
    z *= 4.8

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_subtract(usm_type_x, usm_type_y):
    x = dp.arange(50, usm_type=usm_type_x)
    y = dp.arange(50, usm_type=usm_type_y)

    z = 20 - x - y - 7.4

    # inplace subtract
    z -= x
    z -= -3.4

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_divide(usm_type_x, usm_type_y):
    x = dp.arange(120, usm_type=usm_type_x)
    y = dp.arange(120, usm_type=usm_type_y)

    z = 2 / x / y / 1.5

    # inplace divide
    z /= x
    z /= -2.4

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_power(usm_type_x, usm_type_y):
    x = dp.arange(70, usm_type=usm_type_x).reshape((7, 5, 2))
    y = dp.arange(70, usm_type=usm_type_y).reshape((7, 5, 2))

    z = 2**x**y**1.5
    z **= x
    z **= 1.7

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize(
    "func, args",
    [
        pytest.param("full", ["10", "x0[3]"]),
        pytest.param("full_like", ["x0", "4"]),
        pytest.param("zeros_like", ["x0"]),
        pytest.param("ones_like", ["x0"]),
        pytest.param("empty_like", ["x0"]),
        pytest.param("linspace", ["x0[0:2]", "4", "4"]),
        pytest.param("linspace", ["0", "x0[3:5]", "4"]),
    ],
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_array_creation(func, args, usm_type_x, usm_type_y):
    x0 = dp.full(10, 3, usm_type=usm_type_x)
    new_args = [eval(val, {"x0": x0}) for val in args]

    x = getattr(dp, func)(*new_args)
    y = getattr(dp, func)(*new_args, usm_type=usm_type_y)

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y


@pytest.mark.parametrize(
    "usm_type_start", list_of_usm_types, ids=list_of_usm_types
)
@pytest.mark.parametrize(
    "usm_type_stop", list_of_usm_types, ids=list_of_usm_types
)
def test_linspace_arrays(usm_type_start, usm_type_stop):
    start = dp.asarray([0, 0], usm_type=usm_type_start)
    stop = dp.asarray([2, 4], usm_type=usm_type_stop)
    res = dp.linspace(start, stop, 4)
    assert res.usm_type == du.get_coerced_usm_type(
        [usm_type_start, usm_type_stop]
    )


@pytest.mark.parametrize("func", ["tril", "triu"], ids=["tril", "triu"])
@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_tril_triu(func, usm_type):
    x0 = dp.ones((3, 3), usm_type=usm_type)
    x = getattr(dp, func)(x0)
    assert x.usm_type == usm_type


@pytest.mark.parametrize(
    "op",
    [
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "not_equal",
    ],
    ids=[
        "equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "not_equal",
    ],
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_logic_op(op, usm_type_x, usm_type_y):
    x = dp.arange(100, usm_type=usm_type_x)
    y = dp.arange(100, usm_type=usm_type_y)[::-1]

    z = getattr(dp, op)(x, y)
    zx = getattr(dp, op)(x, 50)
    zy = getattr(dp, op)(30, y)

    assert x.usm_type == zx.usm_type == usm_type_x
    assert y.usm_type == zy.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize(
    "op",
    ["bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift"],
    ids=[
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "left_shift",
        "right_shift",
    ],
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_bitwise_op(op, usm_type_x, usm_type_y):
    x = dp.arange(25, usm_type=usm_type_x)
    y = dp.arange(25, usm_type=usm_type_y)[::-1]

    z = getattr(dp, op)(x, y)
    zx = getattr(dp, op)(x, 7)
    zy = getattr(dp, op)(12, y)

    assert x.usm_type == zx.usm_type == usm_type_x
    assert y.usm_type == zy.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_meshgrid(usm_type_x, usm_type_y):
    x = dp.arange(100, usm_type=usm_type_x)
    y = dp.arange(100, usm_type=usm_type_y)
    z = dp.meshgrid(x, y)
    assert z[0].usm_type == usm_type_x
    assert z[1].usm_type == usm_type_y


@pytest.mark.parametrize(
    "func,data",
    [
        pytest.param(
            "sqrt",
            [1.0, 3.0, 9.0],
        ),
    ],
)
@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_1in_1out(func, data, usm_type):
    x = dp.array(data, usm_type=usm_type)
    res = getattr(dp, func)(x)
    assert x.usm_type == usm_type
    assert res.usm_type == usm_type


@pytest.mark.parametrize(
    "func,data1,data2",
    [
        pytest.param(
            "dot",
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]],
        ),
    ],
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_2in_1out(func, data1, data2, usm_type_x, usm_type_y):
    x = dp.array(data1, usm_type=usm_type_x)
    y = dp.array(data2, usm_type=usm_type_y)
    z = getattr(dp, func)(x, y)

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_broadcast_to(usm_type):
    x = dp.ones(7, usm_type=usm_type)
    y = dp.broadcast_to(x, (2, 7))
    assert x.usm_type == y.usm_type
