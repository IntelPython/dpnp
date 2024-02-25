import tempfile
from math import prod

import dpctl.tensor as dpt
import dpctl.utils as du
import numpy
import pytest

import dpnp as dp
from dpnp.dpnp_utils import get_usm_allocations

from .helper import assert_dtype_allclose

list_of_usm_types = ["device", "shared", "host"]


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_add(usm_type_x, usm_type_y):
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
def test_coerced_usm_types_remainder(usm_type_x, usm_type_y):
    x = dp.arange(100, usm_type=usm_type_x).reshape(10, 10)
    y = dp.arange(100, usm_type=usm_type_y).reshape(10, 10)
    y = y.T + 1

    z = 100 % y
    z = y % 7
    z = x % y

    # inplace remainder
    z %= y
    z %= 5

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_floor_divide(usm_type_x, usm_type_y):
    x = dp.arange(100, usm_type=usm_type_x).reshape(10, 10)
    y = dp.arange(100, usm_type=usm_type_y).reshape(10, 10)
    x = x + 1.5
    y = y.T + 0.5

    z = 3.4 // y
    z = y // 2.7
    z = x // y

    # inplace floor_divide
    z //= y
    z //= 2.5

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
        pytest.param("copy", ["x0"]),
        pytest.param("diag", ["x0"]),
        pytest.param("empty_like", ["x0"]),
        pytest.param("full", ["10", "x0[3]"]),
        pytest.param("full_like", ["x0", "4"]),
        pytest.param("geomspace", ["x0[0:3]", "8", "4"]),
        pytest.param("geomspace", ["1", "x0[3:5]", "4"]),
        pytest.param("linspace", ["x0[0:2]", "8", "4"]),
        pytest.param("linspace", ["0", "x0[3:5]", "4"]),
        pytest.param("logspace", ["x0[0:2]", "8", "4"]),
        pytest.param("logspace", ["0", "x0[3:5]", "4"]),
        pytest.param("ones_like", ["x0"]),
        pytest.param("vander", ["x0"]),
        pytest.param("zeros_like", ["x0"]),
    ],
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_array_creation_from_1d_array(func, args, usm_type_x, usm_type_y):
    x0 = dp.full(10, 3, usm_type=usm_type_x)
    new_args = [eval(val, {"x0": x0}) for val in args]

    x = getattr(dp, func)(*new_args)
    y = getattr(dp, func)(*new_args, usm_type=usm_type_y)

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y


@pytest.mark.parametrize(
    "func, args",
    [
        pytest.param("diag", ["x0"]),
        pytest.param("diagflat", ["x0"]),
    ],
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_array_creation_from_2d_array(func, args, usm_type_x, usm_type_y):
    x0 = dp.arange(25, usm_type=usm_type_x).reshape(5, 5)
    new_args = [eval(val, {"x0": x0}) for val in args]

    x = getattr(dp, func)(*new_args)
    y = getattr(dp, func)(*new_args, usm_type=usm_type_y)

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y


@pytest.mark.parametrize(
    "func, arg, kwargs",
    [
        pytest.param("arange", [-25.7], {"stop": 10**8, "step": 15}),
        pytest.param("frombuffer", [b"\x01\x02\x03\x04"], {"dtype": dp.int32}),
        pytest.param("fromstring", ["1, 2"], {"dtype": int, "sep": " "}),
        pytest.param("full", [(2, 2)], {"fill_value": 5}),
        pytest.param("eye", [4, 2], {}),
        pytest.param("geomspace", [1, 4, 8], {}),
        pytest.param("identity", [4], {}),
        pytest.param("linspace", [0, 4, 8], {}),
        pytest.param("logspace", [0, 4, 8], {}),
        pytest.param("ones", [(2, 2)], {}),
        pytest.param("tri", [3, 5, 2], {}),
        pytest.param("zeros", [(2, 2)], {}),
    ],
)
@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_array_creation_from_scratch(func, arg, kwargs, usm_type):
    dpnp_kwargs = dict(kwargs)
    dpnp_kwargs["usm_type"] = usm_type
    dpnp_array = getattr(dp, func)(*arg, **dpnp_kwargs)

    numpy_kwargs = dict(kwargs)
    numpy_kwargs["dtype"] = dpnp_array.dtype
    numpy_array = getattr(numpy, func)(*arg, **numpy_kwargs)

    assert_dtype_allclose(dpnp_array, numpy_array)
    assert dpnp_array.shape == numpy_array.shape
    assert dpnp_array.usm_type == usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_array_creation_from_file(usm_type):
    with tempfile.TemporaryFile() as fh:
        fh.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08")
        fh.flush()

        fh.seek(0)
        numpy_array = numpy.fromfile(fh)

        fh.seek(0)
        dpnp_array = dp.fromfile(fh, usm_type=usm_type)

    assert_dtype_allclose(dpnp_array, numpy_array)
    assert dpnp_array.shape == numpy_array.shape
    assert dpnp_array.usm_type == usm_type


@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_logspace_base(usm_type_x, usm_type_y):
    x0 = dp.full(10, 2, usm_type=usm_type_x)

    x = dp.logspace([2, 2], 8, 4, base=x0[3:5])
    y = dp.logspace([2, 2], 8, 4, base=x0[3:5], usm_type=usm_type_y)

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y


@pytest.mark.parametrize(
    "func",
    [
        "array",
        "asarray",
        "asanyarray",
        "ascontiguousarray",
        "asfarray",
        "asfortranarray",
    ],
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_array_copy(func, usm_type_x, usm_type_y):
    sh = (3, 7, 5)
    x = dp.arange(1, prod(sh) + 1, 1, usm_type=usm_type_x).reshape(sh)

    y = getattr(dp, func)(x, usm_type=usm_type_y)

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y


@pytest.mark.parametrize(
    "copy", [True, False, None], ids=["True", "False", "None"]
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
def test_array_creation_from_dpctl(copy, usm_type_x):
    x = dpt.ones((3, 3), usm_type=usm_type_x)
    y = dp.array(x, copy=copy)

    assert y.usm_type == usm_type_x


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
@pytest.mark.parametrize(
    "shape_pair",
    [
        ((2, 4), (4, 3)),
        ((2, 0), (0, 3)),
        ((2, 4), (4, 0)),
        ((4, 2, 3), (4, 3, 5)),
        ((6, 7, 4, 3), (6, 7, 3, 5)),
    ],
    ids=[
        "((2, 4), (4, 3))",
        "((2, 0), (0, 3))",
        "((2, 4), (4, 0))",
        "((4, 2, 3), (4, 3, 5))",
        "((6, 7, 4, 3), (6, 7, 3, 5))",
    ],
)
def test_matmul(usm_type_x, usm_type_y, shape_pair):
    shape1, shape2 = shape_pair
    x = numpy.arange(numpy.prod(shape1)).reshape(shape1)
    y = numpy.arange(numpy.prod(shape2)).reshape(shape2)

    x = dp.array(x, usm_type=usm_type_x)
    y = dp.array(y, usm_type=usm_type_y)
    z = dp.matmul(x, y)

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
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
        pytest.param("average", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("abs", [-1.2, 1.2]),
        pytest.param("angle", [[1.0 + 1.0j, 2.0 + 3.0j]]),
        pytest.param("arccos", [-0.5, 0.0, 0.5]),
        pytest.param("arccosh", [1.5, 3.5, 5.0]),
        pytest.param("arcsin", [-0.5, 0.0, 0.5]),
        pytest.param("arcsinh", [-5.0, -3.5, 0.0, 3.5, 5.0]),
        pytest.param("arctan", [-1.0, 0.0, 1.0]),
        pytest.param("arctanh", [-0.5, 0.0, 0.5]),
        pytest.param("argmax", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("argmin", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("argsort", [2.0, 1.0, 7.0, 4.0]),
        pytest.param("cbrt", [1, 8, 27]),
        pytest.param("ceil", [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
        pytest.param("conjugate", [[1.0 + 1.0j, 0.0], [0.0, 1.0 + 1.0j]]),
        pytest.param(
            "cos", [-dp.pi / 2, -dp.pi / 4, 0.0, dp.pi / 4, dp.pi / 2]
        ),
        pytest.param("cosh", [-5.0, -3.5, 0.0, 3.5, 5.0]),
        pytest.param("count_nonzero", [0, 1, 7, 0]),
        pytest.param("diff", [1.0, 2.0, 4.0, 7.0, 0.0]),
        pytest.param("exp", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("exp2", [0.0, 1.0, 2.0]),
        pytest.param("expm1", [1.0e-10, 1.0, 2.0, 4.0, 7.0]),
        pytest.param("floor", [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
        pytest.param(
            "imag", [complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)]
        ),
        pytest.param("log", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("log10", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("log1p", [1.0e-10, 1.0, 2.0, 4.0, 7.0]),
        pytest.param("log2", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("logsumexp", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("max", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("mean", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("min", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("nanargmax", [1.0, 2.0, 4.0, dp.nan]),
        pytest.param("nanargmin", [1.0, 2.0, 4.0, dp.nan]),
        pytest.param("nanmax", [1.0, 2.0, 4.0, dp.nan]),
        pytest.param("nanmean", [1.0, 2.0, 4.0, dp.nan]),
        pytest.param("nanmin", [1.0, 2.0, 4.0, dp.nan]),
        pytest.param("nanprod", [1.0, 2.0, dp.nan]),
        pytest.param("nanstd", [1.0, 2.0, 4.0, dp.nan]),
        pytest.param("nansum", [1.0, 2.0, 4.0, dp.nan]),
        pytest.param("nanvar", [1.0, 2.0, 4.0, dp.nan]),
        pytest.param("negative", [1.0, 0.0, -1.0]),
        pytest.param("positive", [1.0, 0.0, -1.0]),
        pytest.param("prod", [1.0, 2.0]),
        pytest.param("proj", [complex(1.0, 2.0), complex(dp.inf, -1.0)]),
        pytest.param("ptp", [1.0, 2.0, 4.0, 7.0]),
        pytest.param(
            "real", [complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)]
        ),
        pytest.param("reciprocal", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("reduce_hypot", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("rsqrt", [1, 8, 27]),
        pytest.param("sign", [-5.0, 0.0, 4.5]),
        pytest.param("signbit", [-5.0, 0.0, 4.5]),
        pytest.param(
            "sin", [-dp.pi / 2, -dp.pi / 4, 0.0, dp.pi / 4, dp.pi / 2]
        ),
        pytest.param("sinh", [-5.0, -3.5, 0.0, 3.5, 5.0]),
        pytest.param("sort", [2.0, 1.0, 7.0, 4.0]),
        pytest.param("sqrt", [1.0, 3.0, 9.0]),
        pytest.param("square", [1.0, 3.0, 9.0]),
        pytest.param("std", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("sum", [1.0, 2.0]),
        pytest.param(
            "tan", [-dp.pi / 2, -dp.pi / 4, 0.0, dp.pi / 4, dp.pi / 2]
        ),
        pytest.param("tanh", [-5.0, -3.5, 0.0, 3.5, 5.0]),
        pytest.param("trunc", [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
        pytest.param("var", [1.0, 2.0, 4.0, 7.0]),
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
            "allclose",
            [[1.2, -0.0], [-7, 2.34567]],
            [[1.2, 0.0], [-7, 2.34567]],
        ),
        pytest.param("arctan2", [[-1, +1, +1, -1]], [[-1, -1, +1, +1]]),
        pytest.param("copysign", [0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]),
        pytest.param("cross", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        # dpnp.dot has 3 different implementations based on input arrays dtype
        # checking all of them
        pytest.param("dot", [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]),
        pytest.param("dot", [3, 4, 5], [1, 2, 3]),
        pytest.param("dot", [3 + 2j, 4 + 1j, 5], [1, 2 + 3j, 3]),
        pytest.param("fmax", [[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]),
        pytest.param("fmin", [[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]),
        pytest.param(
            "hypot", [[1.0, 2.0, 3.0, 4.0]], [[-1.0, -2.0, -4.0, -5.0]]
        ),
        pytest.param("inner", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        pytest.param("kron", [3.0, 4.0, 5.0], [1.0, 2.0]),
        pytest.param("logaddexp", [[-1, 2, 5, 9]], [[4, -3, 2, -8]]),
        pytest.param("maximum", [[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]),
        pytest.param("minimum", [[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]),
        pytest.param(
            "tensordot",
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
        ),
        # dpnp.vdot has 3 different implementations based on input arrays dtype
        # checking all of them
        pytest.param("vdot", [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]),
        pytest.param("vdot", [3, 4, 5], [1, 2, 3]),
        pytest.param("vdot", [3 + 2j, 4 + 1j, 5], [1, 2 + 3j, 3]),
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


@pytest.mark.parametrize(
    "func,data1,data2",
    [
        pytest.param("column_stack", (1, 2, 3), (2, 3, 4)),
        pytest.param("concatenate", [[1, 2], [3, 4]], [[5, 6]]),
        pytest.param("dstack", [[1], [2], [3]], [[2], [3], [4]]),
        pytest.param("hstack", (1, 2, 3), (4, 5, 6)),
        pytest.param("row_stack", [[7], [1], [2], [3]], [[2], [3], [9], [4]]),
        pytest.param("stack", [1, 2, 3], [4, 5, 6]),
        pytest.param("vstack", [0, 1, 2, 3], [4, 5, 6, 7]),
    ],
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("usm_type_y", list_of_usm_types, ids=list_of_usm_types)
def test_concat_stack(func, data1, data2, usm_type_x, usm_type_y):
    x = dp.array(data1, usm_type=usm_type_x)
    y = dp.array(data2, usm_type=usm_type_y)
    z = getattr(dp, func)((x, y))

    assert x.usm_type == usm_type_x
    assert y.usm_type == usm_type_y
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_y])


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_multi_dot(usm_type):
    numpy_array_list = []
    dpnp_array_list = []
    for num_array in [3, 5]:  # number of arrays in multi_dot
        for _ in range(num_array):  # creat arrays one by one
            a = numpy.random.rand(10, 10)
            b = dp.array(a, usm_type=usm_type)

            numpy_array_list.append(a)
            dpnp_array_list.append(b)

        result = dp.linalg.multi_dot(dpnp_array_list)
        expected = numpy.linalg.multi_dot(numpy_array_list)
        assert_dtype_allclose(result, expected)

        input_usm_type, _ = get_usm_allocations(dpnp_array_list)
        assert input_usm_type == usm_type
        assert result.usm_type == usm_type


@pytest.mark.parametrize("func", ["take", "take_along_axis"])
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize(
    "usm_type_ind", list_of_usm_types, ids=list_of_usm_types
)
def test_take(func, usm_type_x, usm_type_ind):
    x = dp.arange(5, usm_type=usm_type_x)
    ind = dp.array([0, 2, 4], usm_type=usm_type_ind)
    z = getattr(dp, func)(x, ind, axis=None)

    assert x.usm_type == usm_type_x
    assert ind.usm_type == usm_type_ind
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_ind])


@pytest.mark.parametrize(
    "data, ind, axis",
    [
        (numpy.arange(6), numpy.array([0, 2, 4]), None),
        (
            numpy.arange(6).reshape((2, 3)),
            numpy.array([0, 1]).reshape((2, 1)),
            1,
        ),
    ],
)
@pytest.mark.parametrize("usm_type_x", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize(
    "usm_type_ind", list_of_usm_types, ids=list_of_usm_types
)
def test_take_along_axis(data, ind, axis, usm_type_x, usm_type_ind):
    x = dp.array(data, usm_type=usm_type_x)
    ind = dp.array(ind, usm_type=usm_type_ind)

    z = dp.take_along_axis(x, ind, axis=axis)

    assert x.usm_type == usm_type_x
    assert ind.usm_type == usm_type_ind
    assert z.usm_type == du.get_coerced_usm_type([usm_type_x, usm_type_ind])


@pytest.mark.parametrize(
    "data, is_empty",
    [
        ([[1, -2], [2, 5]], False),
        ([[[1, -2], [2, 5]], [[1, -2], [2, 5]]], False),
        ((0, 0), True),
        ((3, 0, 0), True),
    ],
    ids=["2D", "3D", "Empty_2D", "Empty_3D"],
)
@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_cholesky(data, is_empty, usm_type):
    if is_empty:
        x = dp.empty(data, dtype=dp.default_float_type(), usm_type=usm_type)
    else:
        x = dp.array(data, dtype=dp.default_float_type(), usm_type=usm_type)

    result = dp.linalg.cholesky(x)

    assert x.usm_type == result.usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_indices(usm_type):
    x = dp.indices((2,), usm_type=usm_type)
    assert x.usm_type == usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("func", ["mgrid", "ogrid"])
def test_grid(usm_type, func):
    assert getattr(dp, func)(usm_type=usm_type)[0:4].usm_type == usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize("sparse", [True, False], ids=["True", "False"])
def test_indices_sparse(usm_type, sparse):
    x = dp.indices((2, 3), sparse=sparse, usm_type=usm_type)
    for i in x:
        assert i.usm_type == usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_clip(usm_type):
    x = dp.arange(10, usm_type=usm_type)
    y = dp.clip(x, 2, 7)
    assert x.usm_type == y.usm_type


@pytest.mark.parametrize(
    "usm_type_matrix", list_of_usm_types, ids=list_of_usm_types
)
@pytest.mark.parametrize(
    "usm_type_vector", list_of_usm_types, ids=list_of_usm_types
)
@pytest.mark.parametrize(
    "matrix, vector",
    [
        ([[1, 2], [3, 5]], dp.empty((2, 0))),
        ([[1, 2], [3, 5]], [1, 2]),
        (
            [
                [[1, 1, 1], [0, 2, 5], [2, 5, -1]],
                [[3, -1, 1], [1, 2, 3], [2, 3, 1]],
                [[1, 4, 1], [1, 2, -2], [4, 1, 2]],
            ],
            [[6, -4, 27], [9, -6, 15], [15, 1, 11]],
        ),
    ],
    ids=[
        "2D_Matrix_Empty_Vector",
        "2D_Matrix_1D_Vector",
        "3D_Matrix_and_Vectors",
    ],
)
def test_solve(matrix, vector, usm_type_matrix, usm_type_vector):
    x = dp.array(matrix, usm_type=usm_type_matrix)
    y = dp.array(vector, usm_type=usm_type_vector)
    z = dp.linalg.solve(x, y)

    assert x.usm_type == usm_type_matrix
    assert y.usm_type == usm_type_vector
    assert z.usm_type == du.get_coerced_usm_type(
        [usm_type_matrix, usm_type_vector]
    )


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize(
    "shape, is_empty",
    [
        ((2, 2), False),
        ((3, 2, 2), False),
        ((0, 0), True),
        ((0, 2, 2), True),
    ],
    ids=[
        "(2, 2)",
        "(3, 2, 2)",
        "(0, 0)",
        "(0, 2, 2)",
    ],
)
def test_slogdet(shape, is_empty, usm_type):
    if is_empty:
        x = dp.empty(shape, dtype=dp.default_float_type(), usm_type=usm_type)
    else:
        count_elem = numpy.prod(shape)
        x = dp.arange(
            1, count_elem + 1, dtype=dp.default_float_type(), usm_type=usm_type
        ).reshape(shape)

    sign, logdet = dp.linalg.slogdet(x)

    assert x.usm_type == sign.usm_type
    assert x.usm_type == logdet.usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize(
    "shape, is_empty",
    [
        ((2, 2), False),
        ((3, 2, 2), False),
        ((0, 0), True),
        ((0, 2, 2), True),
    ],
    ids=[
        "(2, 2)",
        "(3, 2, 2)",
        "(0, 0)",
        "(0, 2, 2)",
    ],
)
def test_det(shape, is_empty, usm_type):
    if is_empty:
        x = dp.empty(shape, dtype=dp.default_float_type(), usm_type=usm_type)
    else:
        count_elem = numpy.prod(shape)
        x = dp.arange(
            1, count_elem + 1, dtype=dp.default_float_type(), usm_type=usm_type
        ).reshape(shape)

    det = dp.linalg.det(x)

    assert x.usm_type == det.usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize(
    "shape, is_empty",
    [
        ((2, 2), False),
        ((3, 2, 2), False),
        ((0, 0), True),
        ((0, 2, 2), True),
    ],
    ids=[
        "(2, 2)",
        "(3, 2, 2)",
        "(0, 0)",
        "(0, 2, 2)",
    ],
)
def test_inv(shape, is_empty, usm_type):
    if is_empty:
        x = dp.empty(shape, dtype=dp.default_float_type(), usm_type=usm_type)
    else:
        count_elem = numpy.prod(shape)
        x = dp.arange(
            1, count_elem + 1, dtype=dp.default_float_type(), usm_type=usm_type
        ).reshape(shape)

    result = dp.linalg.inv(x)

    assert x.usm_type == result.usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize(
    "full_matrices_param", [True, False], ids=["True", "False"]
)
@pytest.mark.parametrize(
    "compute_uv_param", [True, False], ids=["True", "False"]
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 4),
        (3, 2),
        (4, 4),
        (2, 0),
        (0, 2),
        (2, 2, 3),
        (3, 3, 0),
        (0, 2, 3),
        (1, 0, 3),
    ],
    ids=[
        "(1, 4)",
        "(3, 2)",
        "(4, 4)",
        "(2, 0)",
        "(0, 2)",
        "(2, 2, 3)",
        "(3, 3, 0)",
        "(0, 2, 3)",
        "(1, 0, 3)",
    ],
)
def test_svd(usm_type, shape, full_matrices_param, compute_uv_param):
    x = dp.ones(shape, usm_type=usm_type)

    if compute_uv_param:
        u, s, vt = dp.linalg.svd(
            x, full_matrices=full_matrices_param, compute_uv=compute_uv_param
        )

        assert x.usm_type == u.usm_type
        assert x.usm_type == vt.usm_type
    else:
        s = dp.linalg.svd(
            x, full_matrices=full_matrices_param, compute_uv=compute_uv_param
        )

    assert x.usm_type == s.usm_type


@pytest.mark.parametrize(
    "data, tol",
    [
        (numpy.array([1, 2]), None),
        (numpy.array([[1, 2], [3, 4]]), None),
        (numpy.array([[1, 2], [3, 4]]), 1e-06),
    ],
    ids=[
        "1-D array",
        "2-D array no tol",
        "2_d array with tol",
    ],
)
@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_matrix_rank(data, tol, usm_type):
    a = dp.array(data, usm_type=usm_type)

    dp_res = dp.linalg.matrix_rank(a, tol=tol)
    assert a.usm_type == dp_res.usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize(
    "shape, hermitian",
    [
        ((4, 4), False),
        ((2, 0), False),
        ((4, 4), True),
        ((2, 2, 3), False),
        ((0, 2, 3), False),
        ((1, 0, 3), False),
    ],
    ids=[
        "(4, 4)",
        "(2, 0)",
        "(2, 2), hermitian)",
        "(2, 2, 3)",
        "(0, 2, 3)",
        "(1, 0, 3)",
    ],
)
def test_pinv(shape, hermitian, usm_type):
    numpy.random.seed(81)
    if hermitian:
        a = dp.random.randn(*shape) + 1j * dp.random.randn(*shape)
        a = dp.conj(a.T) @ a
    else:
        a = dp.random.randn(*shape)

    a = dp.array(a, usm_type=usm_type)
    B = dp.linalg.pinv(a, hermitian=hermitian)

    assert a.usm_type == B.usm_type


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (2, 0),
        (2, 2, 3),
        (0, 2, 3),
        (1, 0, 3),
    ],
    ids=[
        "(4, 4)",
        "(2, 0)",
        "(2, 2, 3)",
        "(0, 2, 3)",
        "(1, 0, 3)",
    ],
)
@pytest.mark.parametrize(
    "mode",
    ["r", "raw", "complete", "reduced"],
    ids=["r", "raw", "complete", "reduced"],
)
def test_qr(shape, mode, usm_type):
    count_elems = numpy.prod(shape)
    a = dp.arange(count_elems, usm_type=usm_type).reshape(shape)

    if mode == "r":
        dp_r = dp.linalg.qr(a, mode=mode)
        assert a.usm_type == dp_r.usm_type
    else:
        dp_q, dp_r = dp.linalg.qr(a, mode=mode)

        assert a.usm_type == dp_q.usm_type
        assert a.usm_type == dp_r.usm_type
