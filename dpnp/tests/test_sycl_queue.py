import copy
import tempfile

import dpctl
import dpctl.tensor as dpt
import numpy
import pytest
from dpctl.utils import ExecutionPlacementError
from numpy.testing import assert_array_equal, assert_raises

import dpnp
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import get_usm_allocations

from .helper import generate_random_numpy_array, get_all_dtypes, is_win_platform

list_of_backend_str = ["cuda", "host", "level_zero", "opencl"]

list_of_device_type_str = ["host", "gpu", "cpu"]

available_devices = [
    d for d in dpctl.get_devices() if not getattr(d, "has_aspect_host", False)
]

valid_dev = []
for device in available_devices:
    if device.default_selector_score < 0:
        pass
    elif device.backend.name not in list_of_backend_str:
        pass
    elif device.device_type.name not in list_of_device_type_str:
        pass
    elif device.backend.name in "opencl" and device.is_gpu:
        # due to reported crash on Windows: CMPLRLLVM-55640
        pass
    else:
        valid_dev.append(device)

dev_ids = [device.filter_string for device in valid_dev]


def assert_sycl_queue_equal(result, expected):
    assert result.backend == expected.backend
    assert result.sycl_context == expected.sycl_context
    assert result.sycl_device == expected.sycl_device
    assert result.is_in_order == expected.is_in_order
    assert result.has_enable_profiling == expected.has_enable_profiling
    exec_queue = dpctl.utils.get_execution_queue([result, expected])
    assert exec_queue is not None


@pytest.mark.parametrize(
    "func, arg, kwargs",
    [
        pytest.param("arange", [-25.7], {"stop": 10**8, "step": 15}),
        pytest.param("bartlett", [10], {}),
        pytest.param("blackman", [10], {}),
        pytest.param("eye", [4, 2], {}),
        pytest.param("empty", [(2, 2)], {}),
        pytest.param(
            "frombuffer", [b"\x01\x02\x03\x04"], {"dtype": dpnp.int32}
        ),
        pytest.param(
            "fromfunction",
            [(lambda i, j: i + j), (3, 3)],
            {"dtype": dpnp.int32},
        ),
        pytest.param("fromiter", [[1, 2, 3, 4]], {"dtype": dpnp.int64}),
        pytest.param("fromstring", ["1 2"], {"dtype": int, "sep": " "}),
        pytest.param("full", [(2, 2)], {"fill_value": 5}),
        pytest.param("geomspace", [1, 4, 8], {}),
        pytest.param("hamming", [10], {}),
        pytest.param("hanning", [10], {}),
        pytest.param("identity", [4], {}),
        pytest.param("kaiser", [10], {"beta": 14}),
        pytest.param("linspace", [0, 4, 8], {}),
        pytest.param("logspace", [0, 4, 8], {}),
        pytest.param("logspace", [0, 4, 8], {"base": [10]}),
        pytest.param("ones", [(2, 2)], {}),
        pytest.param("tri", [3, 5, 2], {}),
        pytest.param("zeros", [(2, 2)], {}),
    ],
)
@pytest.mark.parametrize("device", valid_dev + [None], ids=dev_ids + [None])
def test_array_creation_from_scratch(func, arg, kwargs, device):
    kwargs = dict(kwargs)
    kwargs["device"] = device
    x = getattr(dpnp, func)(*arg, **kwargs)

    if device is None:
        # assert against default device
        device = dpctl.select_default_device()
    assert x.sycl_device == device


@pytest.mark.parametrize(
    "func, args",
    [
        pytest.param("copy", ["x0"]),
        pytest.param("diag", ["x0"]),
        pytest.param("diag", ["x0.reshape((2,2))"]),
        pytest.param("diagflat", ["x0.reshape((2,2))"]),
        pytest.param("empty_like", ["x0"]),
        pytest.param("full", ["10", "x0[3]"]),
        pytest.param("full_like", ["x0", "5"]),
        pytest.param("geomspace", ["x0[0:3]", "8", "4"]),
        pytest.param("geomspace", ["1", "x0[2:4]", "4"]),
        pytest.param("linspace", ["x0[0:2]", "8", "4"]),
        pytest.param("linspace", ["0", "x0[2:4]", "4"]),
        pytest.param("logspace", ["x0[0:2]", "8", "4"]),
        pytest.param("logspace", ["0", "x0[2:4]", "4"]),
        pytest.param("ones_like", ["x0"]),
        pytest.param("vander", ["x0"]),
        pytest.param("zeros_like", ["x0"]),
    ],
)
@pytest.mark.parametrize("device_x", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("device_y", valid_dev, ids=dev_ids)
def test_array_creation_from_array(func, args, device_x, device_y):
    if func == "linspace" and is_win_platform():
        pytest.skip("CPU driver experiences an instability on Windows.")

    x = dpnp.array([1, 2, 3, 4], device=device_x)
    args = [eval(val, {"x0": x}) for val in args]

    # follow device
    y = getattr(dpnp, func)(*args)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)

    # cross device
    # TODO: include geomspace when issue dpnp#2352 is resolved
    if func != "geomspace":
        y = getattr(dpnp, func)(*args, device=device_y)
        assert_sycl_queue_equal(y.sycl_queue, x.to_device(device_y).sycl_queue)


@pytest.mark.parametrize("device_x", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("device_y", valid_dev, ids=dev_ids)
def test_array_creation_logspace_base(device_x, device_y):
    x = dpnp.array([1, 2, 3, 4], device=device_x)

    # follow device
    y = dpnp.logspace(0, 8, 4, base=x[1:3])
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)

    # TODO: include geomspace when issue dpnp#2353 is resolved
    # cross device
    # y = dpnp.logspace(0, 8, 4, base=x[1:3], device=device_y)
    # assert_sycl_queue_equal(y.sycl_queue, x.to_device(device_y).sycl_queue)


@pytest.mark.parametrize("device", valid_dev + [None], ids=dev_ids + [None])
def test_array_creation_from_file(device):
    with tempfile.TemporaryFile() as fh:
        fh.write(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08")
        fh.flush()

        fh.seek(0)
        x = dpnp.fromfile(fh, device=device)

    if device is None:
        # assert against default device
        device = dpctl.select_default_device()
    assert x.sycl_device == device


@pytest.mark.parametrize("device", valid_dev + [None], ids=dev_ids + [None])
def test_array_creation_load_txt(device):
    with tempfile.TemporaryFile() as fh:
        fh.write(b"1 2 3 4")
        fh.flush()

        fh.seek(0)
        x = dpnp.loadtxt(fh, device=device)

    if device is None:
        # assert against default device
        device = dpctl.select_default_device()
    assert x.sycl_device == device


@pytest.mark.parametrize("device_x", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("device_y", valid_dev, ids=dev_ids)
def test_copy_method(device_x, device_y):
    x = dpnp.array([[1, 2, 3], [4, 5, 6]], device=device_x)
    y = x.copy()
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)

    q = dpctl.SyclQueue(device_y)
    y = x.copy(sycl_queue=q)
    assert_sycl_queue_equal(y.sycl_queue, q)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_copy_operation(device):
    x = dpnp.array([[1, 2, 3], [4, 5, 6]], device=device)
    y = copy.copy(x)
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_extract(device):
    x = dpnp.arange(3, device=device)
    y = dpnp.array([True, False, True], device=device)
    result = dpnp.extract(x, y)

    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, y.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_meshgrid(device):
    x = dpnp.arange(100, device=device)
    y = dpnp.arange(100, device=device)
    z = dpnp.meshgrid(x, y)
    assert_sycl_queue_equal(z[0].sycl_queue, x.sycl_queue)
    assert_sycl_queue_equal(z[1].sycl_queue, y.sycl_queue)


@pytest.mark.parametrize(
    "func,data",
    [
        pytest.param("all", [-1.0, 0.0, 1.0]),
        pytest.param("any", [-1.0, 0.0, 1.0]),
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
        pytest.param("argwhere", [[0, 3], [1, 4], [2, 5]]),
        pytest.param("cbrt", [1.0, 8.0, 27.0]),
        pytest.param("ceil", [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
        pytest.param("conjugate", [[1.0 + 1.0j, 0.0], [0.0, 1.0 + 1.0j]]),
        pytest.param("copy", [1.0, 2.0, 3.0]),
        pytest.param("corrcoef", [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        pytest.param(
            "cos", [-dpnp.pi / 2, -dpnp.pi / 4, 0.0, dpnp.pi / 4, dpnp.pi / 2]
        ),
        pytest.param("cosh", [-5.0, -3.5, 0.0, 3.5, 5.0]),
        pytest.param("cov", [[0, 1, 2], [2, 1, 0]]),
        pytest.param("count_nonzero", [3, 0, 2, -1.2]),
        pytest.param("cumlogsumexp", [[1, 2, 3], [4, 5, 6]]),
        pytest.param("cumprod", [[1, 2, 3], [4, 5, 6]]),
        pytest.param("cumsum", [[1, 2, 3], [4, 5, 6]]),
        pytest.param("cumulative_prod", [1, 2, 3, 4, 5, 6]),
        pytest.param("cumulative_sum", [1, 2, 3, 4, 5, 6]),
        pytest.param("degrees", [dpnp.pi, dpnp.pi / 2, 0]),
        pytest.param("diagonal", [[[1, 2], [3, 4]]]),
        pytest.param("diff", [1.0, 2.0, 4.0, 7.0, 0.0]),
        pytest.param("ediff1d", [1.0, 2.0, 4.0, 7.0, 0.0]),
        pytest.param("exp", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("exp2", [0.0, 1.0, 2.0]),
        pytest.param("expm1", [1.0e-10, 1.0, 2.0, 4.0, 7.0]),
        pytest.param("fabs", [-1.2, 1.2]),
        pytest.param("fix", [2.1, 2.9, -2.1, -2.9]),
        pytest.param("flatnonzero", [-2, -1, 0, 1, 2]),
        pytest.param("floor", [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
        pytest.param("gradient", [1.0, 2.0, 4.0, 7.0, 11.0, 16.0]),
        pytest.param("histogram_bin_edges", [0, 0, 0, 1, 2, 3, 3, 4, 5]),
        pytest.param("i0", [0, 1, 2, 3]),
        pytest.param(
            "imag", [complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)]
        ),
        pytest.param("iscomplex", [1 + 1j, 1 + 0j, 4.5, 3, 2, 2j]),
        pytest.param("isreal", [1 + 1j, 1 + 0j, 4.5, 3, 2, 2j]),
        pytest.param("log", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("log10", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("log1p", [1.0e-10, 1.0, 2.0, 4.0, 7.0]),
        pytest.param("log2", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("logsumexp", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("max", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("mean", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("median", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("min", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("nanargmax", [1.0, 2.0, 4.0, dpnp.nan]),
        pytest.param("nanargmin", [1.0, 2.0, 4.0, dpnp.nan]),
        pytest.param("nancumprod", [1.0, dpnp.nan]),
        pytest.param("nancumsum", [1.0, dpnp.nan]),
        pytest.param("nanmax", [1.0, 2.0, 4.0, dpnp.nan]),
        pytest.param("nanmean", [1.0, 2.0, 4.0, dpnp.nan]),
        pytest.param("nanmedian", [1.0, 2.0, 4.0, dpnp.nan]),
        pytest.param("nanmin", [1.0, 2.0, 4.0, dpnp.nan]),
        pytest.param("nanprod", [1.0, dpnp.nan]),
        pytest.param("nanstd", [1.0, 2.0, 4.0, dpnp.nan]),
        pytest.param("nansum", [1.0, dpnp.nan]),
        pytest.param("nanvar", [1.0, 2.0, 4.0, dpnp.nan]),
        pytest.param("negative", [1.0, 0.0, -1.0]),
        pytest.param("positive", [1.0, 0.0, -1.0]),
        pytest.param("prod", [1.0, 2.0]),
        pytest.param(
            "proj",
            [complex(1, 2), complex(dpnp.inf, -1), complex(0, -dpnp.inf)],
        ),
        pytest.param("ptp", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("radians", [180, 90, 45, 0]),
        pytest.param(
            "real", [complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)]
        ),
        pytest.param("real_if_close", [2.1 + 4e-15j, 5.2 + 3e-16j]),
        pytest.param("reciprocal", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("reduce_hypot", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("rot90", [[1, 2], [3, 4]]),
        pytest.param("rsqrt", [1.0, 8.0, 27.0]),
        pytest.param("sign", [-5.0, 0.0, 4.5]),
        pytest.param("signbit", [-5.0, 0.0, 4.5]),
        pytest.param(
            "sin", [-dpnp.pi / 2, -dpnp.pi / 4, 0.0, dpnp.pi / 4, dpnp.pi / 2]
        ),
        pytest.param("sinc", [-5.0, -3.5, 0.0, 2.5, 4.3]),
        pytest.param("sinh", [-5.0, -3.5, 0.0, 3.5, 5.0]),
        pytest.param("sort", [2.0, 1.0, 7.0, 4.0]),
        pytest.param("sort_complex", [1 + 2j, 2 - 1j, 3 - 2j, 3 - 3j, 3 + 5j]),
        pytest.param("spacing", [1, 2, -3, 0]),
        pytest.param("sqrt", [1.0, 3.0, 9.0]),
        pytest.param("square", [1.0, 3.0, 9.0]),
        pytest.param("std", [1.0, 2.0, 4.0, 7.0]),
        pytest.param("sum", [1.0, 2.0]),
        pytest.param(
            "tan", [-dpnp.pi / 2, -dpnp.pi / 4, 0.0, dpnp.pi / 4, dpnp.pi / 2]
        ),
        pytest.param("tanh", [-5.0, -3.5, 0.0, 3.5, 5.0]),
        pytest.param("trace", numpy.eye(3)),
        pytest.param("tril", numpy.ones((3, 3))),
        pytest.param("triu", numpy.ones((3, 3))),
        pytest.param("trapezoid", [1, 2, 3]),
        pytest.param("trim_zeros", [0, 0, 0, 1, 2, 3, 0, 2, 1, 0]),
        pytest.param("trunc", [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]),
        pytest.param("unwrap", [[0, 1, 2, -1, 0]]),
        pytest.param("var", [1.0, 2.0, 4.0, 7.0]),
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_1in_1out(func, data, device):
    x = dpnp.array(data, device=device)
    result = getattr(dpnp, func)(x)
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    out = dpnp.empty_like(result)
    try:
        # some functions do not support out kwarg
        getattr(dpnp, func)(x, out=out)
        assert_sycl_queue_equal(out.sycl_queue, x.sycl_queue)
    except TypeError:
        pass


@pytest.mark.parametrize(
    "func,data1,data2",
    [
        pytest.param(
            "add",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        ),
        pytest.param(
            "allclose", [1.0, dpnp.inf, -dpnp.inf], [1.0, dpnp.inf, -dpnp.inf]
        ),
        pytest.param("append", [1, 2, 3], [4, 5, 6]),
        pytest.param("arctan2", [-1, +1, +1, -1], [-1, -1, +1, +1]),
        pytest.param("compress", [0, 1, 1, 0], [0, 1, 2, 3]),
        pytest.param("copysign", [0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]),
        pytest.param("convolve", [1, 2, 3], [4, 5, 6]),
        pytest.param(
            "corrcoef",
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
        ),
        pytest.param("correlate", [1, 2, 3], [4, 5, 6]),
        pytest.param("cov", [-2.1, -1, 4.3], [3, 1.1, 0.12]),
        pytest.param("cross", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        pytest.param("digitize", [0.2, 6.4, 3.0], [0.0, 1.0, 2.5, 4.0]),
        pytest.param(
            "divide", [0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 4.0, 4.0, 4.0, 4.0]
        ),
        # dpnp.dot has 3 different implementations based on input arrays dtype
        # checking all of them
        pytest.param("dot", [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]),
        pytest.param("dot", [3, 4, 5], [1, 2, 3]),
        pytest.param("dot", [3 + 2j, 4 + 1j, 5], [1, 2 + 3j, 3]),
        pytest.param("extract", [False, True, True, False], [0, 1, 2, 3]),
        pytest.param(
            "float_power", [0, 1, 2, 3, 4, 5], [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
        ),
        pytest.param(
            "floor_divide", [1.0, 2.0, 3.0, 4.0], [2.5, 2.5, 2.5, 2.5]
        ),
        pytest.param("fmax", [2.0, 3.0, 4.0], [1.0, 5.0, 2.0]),
        pytest.param("fmin", [2.0, 3.0, 4.0], [1.0, 5.0, 2.0]),
        pytest.param(
            "fmod",
            [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ),
        pytest.param("gcd", [0, 1, 2, 3, 4, 5], [20, 20, 20, 20, 20, 20]),
        pytest.param(
            "gradient",
            [1.0, 2.0, 4.0, 7.0, 11.0, 16.0],
            [0.0, 1.0, 1.5, 3.5, 4.0, 6.0],
        ),
        pytest.param("heaviside", [-1.5, 0, 2.0], [0.5]),
        pytest.param(
            "histogram_bin_edges", [0, 0, 0, 1, 2, 3, 3, 4, 5], [1, 2]
        ),
        pytest.param("hypot", [1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -4.0, -5.0]),
        pytest.param("inner", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        pytest.param("kron", [3.0, 4.0, 5.0], [1.0, 2.0]),
        pytest.param("lcm", [0, 1, 2, 3, 4, 5], [20, 20, 20, 20, 20, 20]),
        pytest.param("ldexp", [5, 5, 5, 5, 5], [0, 1, 2, 3, 4]),
        pytest.param("logaddexp", [-1, 2, 5, 9], [4, -3, 2, -8]),
        pytest.param("logaddexp2", [-1, 2, 5, 9], [4, -3, 2, -8]),
        pytest.param(
            "matmul", [[1.0, 0.0], [0.0, 1.0]], [[4.0, 1.0], [1.0, 2.0]]
        ),
        pytest.param("maximum", [2.0, 3.0, 4.0], [1.0, 5.0, 2.0]),
        pytest.param("minimum", [2.0, 3.0, 4.0], [1.0, 5.0, 2.0]),
        pytest.param(
            "multiply",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        ),
        pytest.param("nextafter", [1, 2], [2, 1]),
        pytest.param(
            "outer", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 0.0]
        ),
        pytest.param(
            "power",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
        ),
        pytest.param(
            "remainder",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ),
        pytest.param("round", [1.234, 2.567], 2),
        pytest.param("searchsorted", [11, 12, 13, 14, 15], [-10, 20, 12, 13]),
        pytest.param(
            "subtract",
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        ),
        pytest.param(
            "tensordot",
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
        ),
        pytest.param("trapezoid", [1, 2, 3], [4, 6, 8]),
        # dpnp.vdot has 3 different implementations based on input arrays dtype
        # checking all of them
        pytest.param("vdot", [3.0, 4.0, 5.0], [1.0, 2.0, 3.0]),
        pytest.param("vdot", [3, 4, 5], [1, 2, 3]),
        pytest.param("vdot", [3 + 2j, 4 + 1j, 5], [1, 2 + 3j, 3]),
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_2in_1out(func, data1, data2, device):
    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)(x1, x2)

    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, x2.sycl_queue)

    out = dpnp.empty_like(result)
    try:
        # some function do not support out kwarg
        getattr(dpnp, func)(x1, x2, out=out)
        assert_sycl_queue_equal(out.sycl_queue, x1.sycl_queue)
        assert_sycl_queue_equal(out.sycl_queue, x2.sycl_queue)
    except TypeError:
        pass


@pytest.mark.parametrize(
    "op",
    [
        "all",
        "any",
        "isfinite",
        "isinf",
        "isnan",
        "isneginf",
        "isposinf",
        "logical_not",
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_logic_op_1in(op, device):
    x = dpnp.array(
        [-dpnp.inf, -1.0, 0.0, 1.0, dpnp.inf, dpnp.nan], device=device
    )
    result = getattr(dpnp, op)(x)
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize(
    "op",
    [
        "array_equal",
        "array_equiv",
        "equal",
        "greater",
        "greater_equal",
        "isclose",
        "less",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
        "not_equal",
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_logic_op_2in(op, device):
    x1 = dpnp.array(
        [-dpnp.inf, -1.0, 0.0, 1.0, dpnp.inf, dpnp.nan], device=device
    )
    x2 = dpnp.array(
        [dpnp.inf, 1.0, 0.0, -1.0, -dpnp.inf, dpnp.nan], device=device
    )

    result = getattr(dpnp, op)(x1, x2)
    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, x2.sycl_queue)


@pytest.mark.parametrize(
    "func, data, scalar",
    [
        pytest.param("searchsorted", [11, 12, 13, 14, 15], 13),
        pytest.param("broadcast_to", numpy.ones(7), (2, 7)),
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_2in_with_scalar_1out(func, data, scalar, device):
    x1 = dpnp.array(data, device=device)
    result = getattr(dpnp, func)(x1, scalar)

    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)


@pytest.mark.parametrize(
    "func,data1,data2",
    [
        pytest.param(
            "add",
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [0.0, 1.0, 2.0],
        ),
        pytest.param("divide", [0.0, 1.0, 2.0, 3.0, 4.0], [4.0]),
        pytest.param("floor_divide", [1.0, 2.0, 3.0, 4.0], [2.5]),
        pytest.param("fmod", [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], [2.0]),
        pytest.param(
            "multiply",
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [0.0, 1.0, 2.0],
        ),
        pytest.param("remainder", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [5.0]),
        pytest.param(
            "subtract",
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            [0.0, 1.0, 2.0],
        ),
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_2in_broadcasting(func, data1, data2, device):
    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)(x1, x2)

    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)


@pytest.mark.parametrize(
    "func",
    [
        "add",
        "copysign",
        "divide",
        "floor_divide",
        "fmod",
        "maximum",
        "minimum",
        "multiply",
        "outer",
        "power",
        "remainder",
        "subtract",
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_2in_1out_diff_queue_but_equal_context(func, device):
    x1 = dpnp.arange(10)
    x2 = dpnp.arange(10, sycl_queue=dpctl.SyclQueue(device))[::-1]
    with assert_raises((ValueError, ExecutionPlacementError)):
        getattr(dpnp, func)(x1, x2)


@pytest.mark.parametrize("op", ["bitwise_count", "bitwise_not"])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_bitwise_op_1in(op, device):
    x = dpnp.arange(-10, 10, device=device)
    z = getattr(dpnp, op)(x)

    assert_sycl_queue_equal(x.sycl_queue, z.sycl_queue)


@pytest.mark.parametrize(
    "op",
    ["bitwise_and", "bitwise_or", "bitwise_xor", "left_shift", "right_shift"],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_bitwise_op_2in(op, device):
    x = dpnp.arange(25, device=device)
    y = dpnp.arange(25, device=device)[::-1]

    z = getattr(dpnp, op)(x, y)
    zx = getattr(dpnp, op)(x, 7)
    zy = getattr(dpnp, op)(12, y)

    assert_sycl_queue_equal(z.sycl_queue, x.sycl_queue)
    assert_sycl_queue_equal(z.sycl_queue, y.sycl_queue)
    assert_sycl_queue_equal(zx.sycl_queue, x.sycl_queue)
    assert_sycl_queue_equal(zy.sycl_queue, y.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((2, 4), (4,)),
        ((4,), (4, 3)),
        ((2, 4), (4, 3)),
        ((2, 0), (0, 3)),
        ((2, 4), (4, 0)),
        ((4, 2, 3), (4, 3, 5)),
        ((4, 2, 3), (4, 3, 1)),
        ((4, 1, 3), (4, 3, 5)),
        ((6, 7, 4, 3), (6, 7, 3, 5)),
    ],
    ids=[
        "((2, 4), (4,))",
        "((4,), (4, 3))",
        "((2, 4), (4, 3))",
        "((2, 0), (0, 3))",
        "((2, 4), (4, 0))",
        "((4, 2, 3), (4, 3, 5))",
        "((4, 2, 3), (4, 3, 1))",
        "((4, 1, 3), (4, 3, 5))",
        "((6, 7, 4, 3), (6, 7, 3, 5))",
    ],
)
def test_matmul(device, shape1, shape2):
    a = dpnp.arange(numpy.prod(shape1), device=device).reshape(shape1)
    b = dpnp.arange(numpy.prod(shape2), device=device).reshape(shape2)
    result = dpnp.matmul(a, b)

    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, a.sycl_queue)
    assert_sycl_queue_equal(result_queue, b.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((3, 4), (4,)),
        ((2, 3, 4), (4,)),
        ((3, 4), (2, 4)),
        ((5, 1, 3, 4), (2, 4)),
        ((2, 1, 4), (4,)),
    ],
)
def test_matvec(device, shape1, shape2):
    a = dpnp.arange(numpy.prod(shape1), device=device).reshape(shape1)
    b = dpnp.arange(numpy.prod(shape2), device=device).reshape(shape2)
    result = dpnp.matvec(a, b)

    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, a.sycl_queue)
    assert_sycl_queue_equal(result_queue, b.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((4,), (4,)),  # call_flag: dot
        ((3, 1), (3, 1)),
        ((2, 0), (2, 0)),  # zero-size inputs, 1D output
        ((3, 0, 4), (3, 0, 4)),  # zero-size output
        ((3, 4), (3, 4)),  # call_flag: vecdot
    ],
)
def test_vecdot(device, shape1, shape2):
    a = dpnp.arange(numpy.prod(shape1), device=device).reshape(shape1)
    b = dpnp.arange(numpy.prod(shape2), device=device).reshape(shape2)
    result = dpnp.vecdot(a, b)

    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, a.sycl_queue)
    assert_sycl_queue_equal(result_queue, b.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize(
    "shape1, shape2",
    [
        ((3,), (3, 4)),
        ((3,), (2, 3, 4)),
        ((2, 3), (3, 4)),
        ((2, 3), (5, 1, 3, 4)),
        ((3,), (2, 3, 1)),
    ],
)
def test_vecmat(device, shape1, shape2):
    a = dpnp.arange(numpy.prod(shape1), device=device).reshape(shape1)
    b = dpnp.arange(numpy.prod(shape2), device=device).reshape(shape2)
    result = dpnp.vecmat(a, b)

    result_queue = result.sycl_queue
    assert_sycl_queue_equal(result_queue, a.sycl_queue)
    assert_sycl_queue_equal(result_queue, b.sycl_queue)


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        pytest.param("normal", [], {"loc": 1.0, "scale": 3.4, "size": (5, 12)}),
        pytest.param("rand", [20], {}),
        pytest.param(
            "randint",
            [],
            {"low": 2, "high": 15, "size": (4, 8, 16), "dtype": dpnp.int32},
        ),
        pytest.param("randn", [], {"d0": 20}),
        pytest.param("random", [], {"size": (35, 45)}),
        pytest.param(
            "random_integers", [], {"low": -17, "high": 3, "size": (12, 16)}
        ),
        pytest.param("random_sample", [], {"size": (7, 7)}),
        pytest.param("ranf", [], {"size": (10, 7, 12)}),
        pytest.param("sample", [], {"size": (7, 9)}),
        pytest.param("standard_normal", [], {"size": (4, 4, 8)}),
        pytest.param(
            "uniform", [], {"low": 1.0, "high": 2.0, "size": (4, 2, 5)}
        ),
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("usm_type", ["host", "device", "shared"])
def test_random(func, args, kwargs, device, usm_type):
    kwargs = {**kwargs, "device": device, "usm_type": usm_type}

    # test with default SYCL queue per a device
    res_array = getattr(dpnp.random, func)(*args, **kwargs)
    assert device == res_array.sycl_device
    assert usm_type == res_array.usm_type

    # SAT-7414: w/a to avoid crash on Windows (observing on LNL and ARL)
    # sycl_queue = dpctl.SyclQueue(device, property="in_order")
    # TODO: remove the w/a once resolved
    sycl_queue = dpctl.SyclQueue(device, property="enable_profiling")
    kwargs["device"] = None
    kwargs["sycl_queue"] = sycl_queue

    # test with in-order SYCL queue per a device and passed as argument
    res_array = getattr(dpnp.random, func)(*args, **kwargs)
    assert usm_type == res_array.usm_type
    assert_sycl_queue_equal(res_array.sycl_queue, sycl_queue)


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        pytest.param("normal", [], {"loc": 1.0, "scale": 3.4, "size": (5, 12)}),
        pytest.param("rand", [15, 30, 5], {}),
        pytest.param(
            "randint",
            [],
            {"low": 2, "high": 15, "size": (4, 8, 16), "dtype": dpnp.int32},
        ),
        pytest.param("randn", [20, 5, 40], {}),
        pytest.param("random_sample", [], {"size": (7, 7)}),
        pytest.param("standard_normal", [], {"size": (4, 4, 8)}),
        pytest.param(
            "uniform", [], {"low": 1.0, "high": 2.0, "size": (4, 2, 5)}
        ),
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("usm_type", ["host", "device", "shared"])
def test_random_state(func, args, kwargs, device, usm_type):
    kwargs = {**kwargs, "usm_type": usm_type}

    # test with default SYCL queue per a device
    rs = dpnp.random.RandomState(seed=1234567, device=device)
    res_array = getattr(rs, func)(*args, **kwargs)
    assert device == res_array.sycl_device
    assert usm_type == res_array.usm_type

    # SAT-7414: w/a to avoid crash on Windows (observing on LNL and ARL)
    # sycl_queue = dpctl.SyclQueue(device, property="in_order")
    # TODO: remove the w/a once resolved
    sycl_queue = dpctl.SyclQueue(device, property="enable_profiling")

    # test with in-order SYCL queue per a device and passed as argument
    seed = (147, 56, 896) if device.is_cpu else 987654
    rs = dpnp.random.RandomState(seed, sycl_queue=sycl_queue)
    res_array = getattr(rs, func)(*args, **kwargs)
    assert usm_type == res_array.usm_type
    assert_sycl_queue_equal(res_array.sycl_queue, sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_modf(device):
    x = dpnp.array([0, 3.5], device=device)
    result1, result2 = dpnp.modf(x)

    expected_queue = x.sycl_queue
    assert_sycl_queue_equal(result1.sycl_queue, expected_queue)
    assert_sycl_queue_equal(result2.sycl_queue, expected_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_einsum(device):
    array_list = []
    for _ in range(3):  # create arrays one by one
        a = dpnp.random.rand(10, 10, device=device)
        array_list.append(a)

    result = dpnp.einsum("ij,jk,kl->il", *array_list)
    _, exec_q = get_usm_allocations(array_list)
    assert_sycl_queue_equal(result.sycl_queue, exec_q)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_pad(device):
    all_modes = [
        "constant",
        "edge",
        "linear_ramp",
        "maximum",
        "mean",
        "median",
        "minimum",
        "reflect",
        "symmetric",
        "wrap",
        "empty",
    ]
    x = dpnp.arange(100, device=device)
    for mode in all_modes:
        result = dpnp.pad(x, (25, 20), mode=mode)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_require(device):
    x = dpnp.arange(10, device=device).reshape(2, 5)
    result = dpnp.require(x, dtype="f4", requirements=["F"])
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    # No requirements
    result = dpnp.require(x, dtype="f4")
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_resize(device):
    x = dpnp.arange(10, device=device)
    result = dpnp.resize(x, (2, 5))
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


class TestFft:
    @pytest.mark.parametrize(
        "func", ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"]
    )
    @pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
    def test_fft(self, func, device):
        dtype = dpnp.float32 if func in ["rfft", "ihfft"] else dpnp.complex64
        x = dpnp.arange(20, dtype=dtype, device=device)
        result = getattr(dpnp.fft, func)(x)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
    def test_fftn(self, device):
        x = dpnp.arange(24, dtype=dpnp.complex64, device=device)
        x = x.reshape(2, 3, 4)

        result = dpnp.fft.fftn(x)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

        result = dpnp.fft.ifftn(result)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
    def test_rfftn(self, device):
        x = dpnp.arange(24, dtype=dpnp.float32, device=device)
        x = x.reshape(2, 3, 4)

        result = dpnp.fft.rfftn(x)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

        result = dpnp.fft.irfftn(result)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize("func", ["fftfreq", "rfftfreq"])
    @pytest.mark.parametrize("device", valid_dev + [None], ids=dev_ids + [None])
    def test_fftfreq(self, func, device):
        result = getattr(dpnp.fft, func)(10, 0.5, device=device)

        if device is None:
            # assert against default device
            device = dpctl.select_default_device()
        assert result.sycl_device == device

    @pytest.mark.parametrize("func", ["fftshift", "ifftshift"])
    @pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
    def test_fftshift(self, func, device):
        x = dpnp.fft.fftfreq(10, 0.5, device=device)
        result = getattr(dpnp.fft, func)(x)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


class TestToDevice:
    @pytest.mark.parametrize("device_from", valid_dev, ids=dev_ids)
    @pytest.mark.parametrize("device_to", valid_dev, ids=dev_ids)
    def test_basic(self, device_from, device_to):
        data = [1.0, 1.0, 1.0, 1.0, 1.0]
        x = dpnp.array(data, dtype=dpnp.float32, device=device_from)
        y = x.to_device(device_to)

        assert y.sycl_device == device_to
        assert (x.asnumpy() == y.asnumpy()).all()

    def test_to_queue(self):
        x = dpnp.full(100, 2, dtype=dpnp.int64)
        q_prof = dpctl.SyclQueue(x.sycl_device, property="enable_profiling")
        y = x.to_device(q_prof)

        assert (x.asnumpy() == y.asnumpy()).all()
        assert_sycl_queue_equal(y.sycl_queue, q_prof)

    def test_stream(self):
        x = dpnp.full(100, 2, dtype=dpnp.int64)
        q_prof = dpctl.SyclQueue(x.sycl_device, property="enable_profiling")
        q_exec = dpctl.SyclQueue(x.sycl_device)

        y = x.to_device(q_prof, stream=q_exec)
        assert (x.asnumpy() == y.asnumpy()).all()
        assert_sycl_queue_equal(y.sycl_queue, q_prof)

        q_exec = dpctl.SyclQueue(x.sycl_device)
        _ = dpnp.linspace(0, 20, num=10**5, sycl_queue=q_exec)
        y = x.to_device(q_prof, stream=q_exec)
        assert (x.asnumpy() == y.asnumpy()).all()
        assert_sycl_queue_equal(y.sycl_queue, q_prof)

    def test_stream_no_sync(self):
        x = dpnp.full(100, 2, dtype=dpnp.int64)
        q_prof = dpctl.SyclQueue(x.sycl_device, property="enable_profiling")

        for stream in [None, x.sycl_queue]:
            y = x.to_device(q_prof, stream=stream)
            assert (x.asnumpy() == y.asnumpy()).all()
            assert_sycl_queue_equal(y.sycl_queue, q_prof)

    @pytest.mark.parametrize(
        "stream",
        [1, dict(), dpctl.SyclDevice()],
        ids=["scalar", "dictionary", "device"],
    )
    def test_invalid_stream(self, stream):
        x = dpnp.ones(2, dtype=dpnp.int64)
        q_prof = dpctl.SyclQueue(x.sycl_device, property="enable_profiling")
        assert_raises(TypeError, x.to_device, q_prof, stream=stream)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize(
    "func",
    [
        "array",
        "asarray",
        "asarray_chkfinite",
        "asanyarray",
        "ascontiguousarray",
        "asfarray",
        "asfortranarray",
    ],
)
@pytest.mark.parametrize(
    "device_param", ["", "None", "sycl_device"], ids=["Empty", "None", "device"]
)
@pytest.mark.parametrize(
    "queue_param", ["", "None", "sycl_queue"], ids=["Empty", "None", "queue"]
)
def test_array_copy(device, func, device_param, queue_param):
    data = numpy.ones(100)
    dpnp_data = getattr(dpnp, func)(data, device=device)

    kwargs_items = {"device": device_param, "sycl_queue": queue_param}.items()
    kwargs = {
        k: getattr(dpnp_data, v, None) for k, v in kwargs_items if v != ""
    }

    result = dpnp.array(dpnp_data, **kwargs)
    assert_sycl_queue_equal(result.sycl_queue, dpnp_data.sycl_queue)


@pytest.mark.parametrize("copy", [True, False, None])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_array_creation_from_dpctl(copy, device):
    dpt_data = dpt.ones((3, 3), device=device)
    result = dpnp.array(dpt_data, copy=copy)

    assert_sycl_queue_equal(result.sycl_queue, dpt_data.sycl_queue)
    assert isinstance(result, dpnp_array)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("arr_dtype", get_all_dtypes(no_float16=True))
@pytest.mark.parametrize("shape", [tuple(), (2,), (3, 0, 1), (2, 2, 2)])
def test_from_dlpack(arr_dtype, shape, device):
    X = dpnp.empty(shape=shape, dtype=arr_dtype, device=device)
    Y = dpnp.from_dlpack(X)
    assert_array_equal(X, Y)
    assert X.__dlpack_device__() == Y.__dlpack_device__()
    assert_sycl_queue_equal(X.sycl_queue, Y.sycl_queue)
    assert X.usm_type == Y.usm_type
    if Y.ndim:
        V = Y[::-1]
        W = dpnp.from_dlpack(V)
        assert V.strides == W.strides


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("arr_dtype", get_all_dtypes(no_float16=True))
def test_from_dlpack_with_dpt(arr_dtype, device):
    X = dpctl.tensor.empty((64,), dtype=arr_dtype, device=device)
    Y = dpnp.from_dlpack(X)
    assert_array_equal(X, Y)
    assert isinstance(Y, dpnp.dpnp_array.dpnp_array)
    assert X.__dlpack_device__() == Y.__dlpack_device__()
    assert X.usm_type == Y.usm_type
    assert_sycl_queue_equal(X.sycl_queue, Y.sycl_queue)


@pytest.mark.parametrize(
    "func,data1,data2",
    [
        pytest.param("column_stack", (1, 2, 3), (2, 3, 4)),
        pytest.param("concatenate", [[1, 2], [3, 4]], [[5, 6]]),
        pytest.param("dstack", [[1], [2], [3]], [[2], [3], [4]]),
        pytest.param("hstack", (1, 2, 3), (4, 5, 6)),
        pytest.param("stack", [1, 2, 3], [4, 5, 6]),
        pytest.param("vstack", [0, 1, 2, 3], [4, 5, 6, 7]),
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_concat_stack(func, data1, data2, device):
    x1 = dpnp.array(data1, device=device)
    x2 = dpnp.array(data2, device=device)
    result = getattr(dpnp, func)((x1, x2))

    assert_sycl_queue_equal(result.sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result.sycl_queue, x2.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
class TestDelete:
    @pytest.mark.parametrize(
        "obj",
        [slice(None, None, 2), 3, [2, 3]],
        ids=["slice", "scalar", "list"],
    )
    def test_delete(self, obj, device):
        x = dpnp.arange(5, device=device)
        result = dpnp.delete(x, obj)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    def test_obj_ndarray(self, device):
        x = dpnp.arange(5, device=device)
        y = dpnp.array([1, 4], device=device)
        result = dpnp.delete(x, y)

        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)
        assert_sycl_queue_equal(result.sycl_queue, y.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
class TestInsert:
    @pytest.mark.parametrize(
        "obj",
        [slice(None, None, 2), 3, [2, 3]],
        ids=["slice", "scalar", "list"],
    )
    def test_basic(self, obj, device):
        x = dpnp.arange(5, device=device)
        result = dpnp.insert(x, obj, 3)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize(
        "obj",
        [slice(None, None, 3), 3, [2, 3]],
        ids=["slice", "scalar", "list"],
    )
    def test_values_ndarray(self, obj, device):
        x = dpnp.arange(5, device=device)
        y = dpnp.array([1, 4], device=device)
        result = dpnp.insert(x, obj, y)

        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)
        assert_sycl_queue_equal(result.sycl_queue, y.sycl_queue)

    @pytest.mark.parametrize("values", [-2, [-1, -2]], ids=["scalar", "list"])
    def test_obj_ndarray(self, values, device):
        x = dpnp.arange(5, device=device)
        y = dpnp.array([1, 4], device=device)
        result = dpnp.insert(x, y, values)

        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)
        assert_sycl_queue_equal(result.sycl_queue, y.sycl_queue)

    def test_obj_values_ndarray(self, device):
        x = dpnp.arange(5, device=device)
        y = dpnp.array([1, 4], device=device)
        z = dpnp.array([-1, -3], device=device)
        result = dpnp.insert(x, y, z)

        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)
        assert_sycl_queue_equal(result.sycl_queue, y.sycl_queue)
        assert_sycl_queue_equal(result.sycl_queue, z.sycl_queue)


@pytest.mark.parametrize(
    "func,data1",
    [
        pytest.param("array_split", [1, 2, 3, 4]),
        pytest.param("split", [1, 2, 3, 4]),
        pytest.param("hsplit", [1, 2, 3, 4]),
        pytest.param(
            "dsplit",
            [[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]],
        ),
        pytest.param("vsplit", [[1, 2, 3, 4], [1, 2, 3, 4]]),
    ],
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_split(func, data1, device):
    x1 = dpnp.array(data1, device=device)
    result = getattr(dpnp, func)(x1, 2)
    assert_sycl_queue_equal(result[0].sycl_queue, x1.sycl_queue)
    assert_sycl_queue_equal(result[1].sycl_queue, x1.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_apply_along_axis(device):
    x = dpnp.arange(9, device=device).reshape(3, 3)
    result = dpnp.apply_along_axis(dpnp.sum, 0, x)
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_apply_over_axes(device):
    x = dpnp.arange(18, device=device).reshape(2, 3, 3)
    result = dpnp.apply_over_axes(dpnp.sum, x, [0, 1])
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device_x", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("device_y", valid_dev, ids=dev_ids)
def test_asarray(device_x, device_y):
    x = dpnp.array([1, 2, 3], device=device_x)
    y = dpnp.asarray([x, x, x], device=device_y)
    assert_sycl_queue_equal(y.sycl_queue, x.to_device(device_y).sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"prepend": 7}),
        pytest.param({"append": -2}),
        pytest.param({"prepend": -4, "append": 5}),
    ],
)
def test_diff_scalar_append(device, kwargs):
    x = dpnp.arange(7, device=device)
    result = dpnp.diff(x, **kwargs)

    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_clip(device):
    x = dpnp.arange(10, device=device)
    y = dpnp.clip(x, 3, 7)
    assert_sycl_queue_equal(x.sycl_queue, y.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_take(device):
    x = dpnp.arange(5, device=device)
    ind = dpnp.array([0, 2, 4], device=device)
    result = dpnp.take(x, ind, axis=None)
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


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
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_take_along_axis(data, ind, axis, device):
    x = dpnp.array(data, device=device)
    ind = dpnp.array(ind, device=device)
    result = dpnp.take_along_axis(x, ind, axis=axis)

    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("sparse", [True, False])
def test_indices(device, sparse):
    sycl_queue = dpctl.SyclQueue(device)
    grid = dpnp.indices((2, 3), sparse=sparse, sycl_queue=sycl_queue)
    for dpnp_array in grid:
        assert_sycl_queue_equal(dpnp_array.sycl_queue, sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_nonzero(device):
    a = dpnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
    x = dpnp.nonzero(a)
    for x_el in x:
        assert_sycl_queue_equal(x_el.sycl_queue, a.sycl_queue)


@pytest.mark.parametrize("device", valid_dev + [None], ids=dev_ids + [None])
@pytest.mark.parametrize("func", ["mgrid", "ogrid"])
def test_grid(device, func):
    result = getattr(dpnp, func)(device=device)[0:4]

    if device is None:
        # assert against default device
        device = dpctl.select_default_device()
    assert result.sycl_device == device


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_where(device):
    ia = dpnp.array([[0, 1, 2], [0, 2, 4], [0, 3, 6]], device=device)
    result = dpnp.where(ia < 4, ia, -1)

    assert_sycl_queue_equal(result.sycl_queue, ia.sycl_queue)


@pytest.mark.parametrize("wgt", [None, numpy.arange(7, 12)])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_bincount(wgt, device):
    iv = dpnp.arange(5, device=device)
    iw = None if wgt is None else dpnp.array(wgt, sycl_queue=iv.sycl_queue)
    result_hist = dpnp.bincount(iv, weights=iw)
    assert_sycl_queue_equal(result_hist.sycl_queue, iv.sycl_queue)


@pytest.mark.parametrize("wgt", [None, numpy.arange(7, 12)])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_histogram(wgt, device):
    iv = dpnp.arange(5, device=device)
    iw = None if wgt is None else dpnp.array(wgt, sycl_queue=iv.sycl_queue)
    result_hist, result_edges = dpnp.histogram(iv, weights=iw)

    assert_sycl_queue_equal(result_hist.sycl_queue, iv.sycl_queue)
    assert_sycl_queue_equal(result_edges.sycl_queue, iv.sycl_queue)


@pytest.mark.parametrize("wgt", [None, numpy.arange(7, 12)])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_histogram2d(wgt, device):
    ix = dpnp.arange(5, device=device)
    iy = dpnp.arange(5, device=device)
    iw = None if wgt is None else dpnp.array(wgt, sycl_queue=ix.sycl_queue)
    result_hist, result_edges_x, result_edges_y = dpnp.histogram2d(
        ix, iy, weights=iw
    )

    assert_sycl_queue_equal(result_hist.sycl_queue, ix.sycl_queue)
    assert_sycl_queue_equal(result_edges_x.sycl_queue, ix.sycl_queue)
    assert_sycl_queue_equal(result_edges_y.sycl_queue, ix.sycl_queue)


@pytest.mark.parametrize("wgt", [None, numpy.arange(7, 12)])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_histogramdd(wgt, device):
    iv = dpnp.arange(5, device=device)
    iw = None if wgt is None else dpnp.array(wgt, sycl_queue=iv.sycl_queue)
    result_hist, result_edges = dpnp.histogramdd(iv, weights=iw)

    assert_sycl_queue_equal(result_hist.sycl_queue, iv.sycl_queue)
    for edge in result_edges:
        assert_sycl_queue_equal(edge.sycl_queue, iv.sycl_queue)


@pytest.mark.parametrize(
    "func", ["tril_indices_from", "triu_indices_from", "diag_indices_from"]
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_tri_diag_indices_from(func, device):
    a = dpnp.ones((3, 3), device=device)
    res = getattr(dpnp, func)(a)
    for x in res:
        assert_sycl_queue_equal(x.sycl_queue, a.sycl_queue)


@pytest.mark.parametrize(
    "func", ["tril_indices", "triu_indices", "diag_indices"]
)
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_tri_diag_indices(func, device):
    sycl_queue = dpctl.SyclQueue(device)
    res = getattr(dpnp, func)(4, sycl_queue=sycl_queue)
    for x in res:
        assert_sycl_queue_equal(x.sycl_queue, sycl_queue)


@pytest.mark.parametrize("mask_func", ["tril", "triu"])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_mask_indices(mask_func, device):
    sycl_queue = dpctl.SyclQueue(device)
    res = dpnp.mask_indices(4, getattr(dpnp, mask_func), sycl_queue=sycl_queue)
    for x in res:
        assert_sycl_queue_equal(x.sycl_queue, sycl_queue)


@pytest.mark.parametrize("wgt", [None, numpy.arange(7, 12)])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_histogram_bin_edges(wgt, device):
    iv = dpnp.arange(5, device=device)
    iw = None if wgt is None else dpnp.array(wgt, sycl_queue=iv.sycl_queue)
    result_edges = dpnp.histogram_bin_edges(iv, weights=iw)
    assert_sycl_queue_equal(result_edges.sycl_queue, iv.sycl_queue)


@pytest.mark.parametrize("device_x", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("device_y", valid_dev, ids=dev_ids)
def test_astype(device_x, device_y):
    x = dpnp.array([1, 2, 3], dtype="i4", device=device_x)
    y = dpnp.astype(x, "f4")
    assert_sycl_queue_equal(y.sycl_queue, x.sycl_queue)

    sycl_queue = dpctl.SyclQueue(device_y)
    y = dpnp.astype(x, "f4", device=sycl_queue)
    assert_sycl_queue_equal(y.sycl_queue, sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_select(device):
    condlist = [dpnp.array([True, False], device=device)]
    choicelist = [dpnp.array([1, 2], device=device)]
    res = dpnp.select(condlist, choicelist)
    assert_sycl_queue_equal(res.sycl_queue, condlist[0].sycl_queue)


@pytest.mark.parametrize("axis", [None, 0, -1])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_unique(axis, device):
    ia = dpnp.array([[1, 1], [2, 3]], device=device)
    result = dpnp.unique(ia, True, True, True, axis=axis)
    for iv in result:
        assert_sycl_queue_equal(iv.sycl_queue, ia.sycl_queue)


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_nan_to_num(copy, device):
    a = dpnp.array([-dpnp.nan, -1, 0, 1, dpnp.nan], device=device)
    result = dpnp.nan_to_num(a, copy=copy)

    assert_sycl_queue_equal(result.sycl_queue, a.sycl_queue)
    assert copy == (result is not a)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize(
    ["to_end", "to_begin"],
    [
        (10, None),
        (None, -10),
        (10, -10),
    ],
)
def test_ediff1d(device, to_end, to_begin):
    x = dpnp.array([1, 3, 5, 7], device=device)
    if to_end:
        to_end = dpnp.array(to_end, device=device)

    if to_begin:
        to_begin = dpnp.array(to_begin, device=device)

    res = dpnp.ediff1d(x, to_end=to_end, to_begin=to_begin)
    assert_sycl_queue_equal(res.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_unravel_index(device):
    x = dpnp.array(2, device=device)
    result = dpnp.unravel_index(x, shape=(2, 2))
    for res in result:
        assert_sycl_queue_equal(res.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_ravel_index(device):
    x = dpnp.array([1, 0], device=device)
    result = dpnp.ravel_multi_index(x, (2, 2))
    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device_0", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("device_1", valid_dev, ids=dev_ids)
def test_ix(device_0, device_1):
    x0 = dpnp.array([0, 1], device=device_0)
    x1 = dpnp.array([2, 4], device=device_1)
    ixgrid = dpnp.ix_(x0, x1)
    assert_sycl_queue_equal(ixgrid[0].sycl_queue, x0.sycl_queue)
    assert_sycl_queue_equal(ixgrid[1].sycl_queue, x1.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
def test_choose(device):
    chc = dpnp.arange(5, dtype="i4", device=device)
    inds = dpnp.array([0, 1, 3], dtype="i4", device=device)
    result = dpnp.choose(inds, chc)
    assert_sycl_queue_equal(result.sycl_queue, chc.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
@pytest.mark.parametrize("left", [None, -1.0])
@pytest.mark.parametrize("right", [None, 99.0])
@pytest.mark.parametrize("period", [None, 180.0])
def test_interp(device, left, right, period):
    x = dpnp.linspace(0.1, 9.9, 20, device=device)
    xp = dpnp.linspace(0.0, 10.0, 5, sycl_queue=x.sycl_queue)
    fp = dpnp.array(xp * 2 + 1, sycl_queue=x.sycl_queue)

    l = None if left is None else dpnp.array(left, sycl_queue=x.sycl_queue)
    r = None if right is None else dpnp.array(right, sycl_queue=x.sycl_queue)
    result = dpnp.interp(x, xp, fp, left=l, right=r, period=period)

    assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)


@pytest.mark.parametrize("device", valid_dev, ids=dev_ids)
class TestLinAlgebra:
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
    def test_cholesky(self, data, is_empty, device):
        if is_empty:
            x = dpnp.empty(data, device=device)
        else:
            dtype = dpnp.default_float_type(device)
            x = dpnp.array(data, dtype=dtype, device=device)

        result = dpnp.linalg.cholesky(x)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize(
        "p", [None, -dpnp.inf, -2, -1, 1, 2, dpnp.inf, "fro"]
    )
    def test_cond(self, device, p):
        a = generate_random_numpy_array((2, 4, 4))
        ia = dpnp.array(a, device=device)
        result = dpnp.linalg.cond(ia, p=p)
        assert_sycl_queue_equal(result.sycl_queue, ia.sycl_queue)

    def test_det(self, device):
        data = [[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]
        x = dpnp.array(data, device=device)
        result = dpnp.linalg.det(x)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize("func", ["eig", "eigvals", "eigh", "eigvalsh"])
    @pytest.mark.parametrize(
        "shape",
        [(4, 4), (0, 0), (2, 3, 3), (0, 2, 2), (1, 0, 0)],
        ids=["(4, 4)", "(0, 0)", "(2, 3, 3)", "(0, 2, 2)", "(1, 0, 0)"],
    )
    def test_eigenvalue(self, func, shape, device):
        dtype = dpnp.default_float_type(device)
        # Set a `hermitian` flag for generate_random_numpy_array() to
        # get a symmetric array for eigh() and eigvalsh() or
        # non-symmetric for eig() and eigvals()
        is_hermitian = func in ("eigh, eigvalsh")
        a = generate_random_numpy_array(shape, dtype, hermitian=is_hermitian)
        ia = dpnp.array(a, device=device)
        expected_queue = ia.sycl_queue

        if func in ("eig", "eigh"):
            dp_val, dp_vec = getattr(dpnp.linalg, func)(ia)
            assert_sycl_queue_equal(dp_vec.sycl_queue, expected_queue)
        else:  # eighvals or eigvalsh
            dp_val = getattr(dpnp.linalg, func)(ia)

        assert_sycl_queue_equal(dp_val.sycl_queue, expected_queue)

    @pytest.mark.parametrize(
        "shape, is_empty",
        [
            ((2, 2), False),
            ((3, 2, 2), False),
            ((0, 0), True),
            ((0, 2, 2), True),
        ],
        ids=["(2, 2)", "(3, 2, 2)", "(0, 0)", "(0, 2, 2)"],
    )
    def test_inv(self, shape, is_empty, device):
        if is_empty:
            x = dpnp.empty(shape, device=device)
        else:
            dtype = dpnp.default_float_type(device)
            count_elem = numpy.prod(shape)
            x = dpnp.arange(
                1, count_elem + 1, dtype=dtype, device=device
            ).reshape(shape)

        result = dpnp.linalg.inv(x)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize(
        ["m", "n", "nrhs"],
        [(4, 2, 2), (4, 0, 1), (4, 2, 0), (0, 0, 0)],
    )
    def test_lstsq(self, m, n, nrhs, device):
        dtype = dpnp.default_float_type(device)
        a = dpnp.arange(m * n, dtype=dtype, device=device)
        a = a.reshape(m, n)
        b = dpnp.ones((m, nrhs), device=device)
        result = dpnp.linalg.lstsq(a, b)

        for param in result:
            param_queue = param.sycl_queue
            assert_sycl_queue_equal(param_queue, a.sycl_queue)
            assert_sycl_queue_equal(param_queue, b.sycl_queue)

    @pytest.mark.parametrize("n", [-1, 0, 1, 2, 3])
    def test_matrix_power(self, n, device):
        x = dpnp.array([[1.0, 2.0], [3.0, 5.0]], device=device)
        result = dpnp.linalg.matrix_power(x, n)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize(
        "data, tol",
        [([1, 2], None), ([[1, 2], [3, 4]], None), ([[1, 2], [3, 4]], 1e-06)],
        ids=["1-D array", "2-D array no tol", "2_d array with tol"],
    )
    def test_matrix_rank(self, data, tol, device):
        x = dpnp.array(data, device=device)
        result = dpnp.linalg.matrix_rank(x, tol=tol)
        assert_sycl_queue_equal(result.sycl_queue, x.sycl_queue)

    def test_multi_dot(self, device):
        array_list = []
        for num_array in [3, 5]:  # number of arrays in multi_dot
            for _ in range(num_array):  # create arrays one by one
                a = dpnp.random.rand(10, 10, device=device)
                array_list.append(a)

            result = dpnp.linalg.multi_dot(array_list)
            _, exec_q = get_usm_allocations(array_list)
            assert_sycl_queue_equal(result.sycl_queue, exec_q)

    def test_multi_dot_out(self, device):
        array_list = []
        for num_array in [3, 5]:  # number of arrays in multi_dot
            for _ in range(num_array):  # create arrays one by one
                a = dpnp.random.rand(10, 10, device=device)
                array_list.append(a)

            dp_out = dpnp.empty((10, 10), device=device)
            result = dpnp.linalg.multi_dot(array_list, out=dp_out)
            assert result is dp_out
            _, exec_q = get_usm_allocations(array_list)
            assert_sycl_queue_equal(result.sycl_queue, exec_q)

    @pytest.mark.parametrize(
        "ord", [None, -dpnp.inf, -2, -1, 1, 2, 3, dpnp.inf, "fro", "nuc"]
    )
    @pytest.mark.parametrize(
        "axis",
        [-1, 0, 1, (0, 1), (-2, -1), None],
        ids=["-1", "0", "1", "(0, 1)", "(-2, -1)", "None"],
    )
    def test_norm(self, device, ord, axis):
        ia = dpnp.arange(120, device=device).reshape(2, 3, 4, 5)
        if (axis in [-1, 0, 1] and ord in ["nuc", "fro"]) or (
            isinstance(axis, tuple) and ord == 3
        ):
            pytest.skip("Invalid norm order for vectors.")
        elif axis is None and ord is not None:
            pytest.skip("Improper number of dimensions to norm")
        else:
            result = dpnp.linalg.norm(ia, ord=ord, axis=axis)
            assert_sycl_queue_equal(result.sycl_queue, ia.sycl_queue)

    @pytest.mark.parametrize(
        "shape, hermitian, rcond_as_array",
        [
            ((4, 4), False, False),
            ((4, 4), False, True),
            ((2, 0), False, False),
            ((4, 4), True, False),
            ((4, 4), True, True),
            ((2, 2, 3), False, False),
            ((2, 2, 3), False, True),
            ((0, 2, 3), False, False),
            ((1, 0, 3), False, False),
        ],
        ids=[
            "(4, 4)",
            "(4, 4), rcond_as_array",
            "(2, 0)",
            "(2, 2), hermitian)",
            "(2, 2), hermitian, rcond_as_array)",
            "(2, 2, 3)",
            "(2, 2, 3), rcond_as_array",
            "(0, 2, 3)",
            "(1, 0, 3)",
        ],
    )
    def test_pinv(self, shape, hermitian, rcond_as_array, device):
        dtype = dpnp.default_float_type(device)
        a = generate_random_numpy_array(shape, dtype, hermitian=hermitian)
        ia = dpnp.array(a, device=device)

        if rcond_as_array:
            rcond_dp = dpnp.array(1e-15, device=device)
            result = dpnp.linalg.pinv(ia, rcond=rcond_dp, hermitian=hermitian)
        else:
            # rcond == 1e-15 by default
            result = dpnp.linalg.pinv(ia, hermitian=hermitian)

        assert_sycl_queue_equal(result.sycl_queue, ia.sycl_queue)

    @pytest.mark.parametrize(
        "shape",
        [(4, 4), (2, 0), (2, 2, 3), (0, 2, 3), (1, 0, 3)],
        ids=["(4, 4)", "(2, 0)", "(2, 2, 3)", "(0, 2, 3)", "(1, 0, 3)"],
    )
    @pytest.mark.parametrize("mode", ["r", "raw", "complete", "reduced"])
    def test_qr(self, shape, mode, device):
        dtype = dpnp.default_float_type(device)
        count_elems = numpy.prod(shape)
        a = dpnp.arange(count_elems, dtype=dtype, device=device).reshape(shape)

        expected_queue = a.sycl_queue

        if mode == "r":
            dp_r = dpnp.linalg.qr(a, mode=mode)
            dp_r_queue = dp_r.sycl_queue
            assert_sycl_queue_equal(dp_r_queue, expected_queue)
        else:
            dp_q, dp_r = dpnp.linalg.qr(a, mode=mode)
            assert_sycl_queue_equal(dp_q.sycl_queue, expected_queue)
            assert_sycl_queue_equal(dp_r.sycl_queue, expected_queue)

    @pytest.mark.parametrize(
        "shape, is_empty",
        [
            ((2, 2), False),
            ((3, 2, 2), False),
            ((0, 0), True),
            ((0, 2, 2), True),
        ],
        ids=["(2, 2)", "(3, 2, 2)", "(0, 0)", "(0, 2, 2)"],
    )
    def test_slogdet(self, shape, is_empty, device):
        if is_empty:
            x = dpnp.empty(shape, device=device)
        else:
            dtype = dpnp.default_float_type(device)
            count_elem = numpy.prod(shape)
            x = dpnp.arange(
                1, count_elem + 1, dtype=dtype, device=device
            ).reshape(shape)

        sign_result, logdet_result = dpnp.linalg.slogdet(x)

        assert_sycl_queue_equal(sign_result.sycl_queue, x.sycl_queue)
        assert_sycl_queue_equal(logdet_result.sycl_queue, x.sycl_queue)

    @pytest.mark.parametrize(
        "matrix, rhs",
        [
            ([[1, 2], [3, 5]], numpy.empty((2, 0))),
            ([[1, 2], [3, 5]], [1, 2]),
            (
                [
                    [[1, 1], [0, 2]],
                    [[3, -1], [1, 2]],
                ],
                [
                    [[6, -4], [9, -6]],
                    [[15, 1], [15, 1]],
                ],
            ),
        ],
        ids=["2D_Matrix_Empty_RHS", "2D_Matrix_1D_RHS", "3D_Matrix_and_3D_RHS"],
    )
    def test_solve(self, matrix, rhs, device):
        a = dpnp.array(matrix, device=device)
        b = dpnp.array(rhs, device=device)
        result = dpnp.linalg.solve(a, b)

        assert_sycl_queue_equal(result.sycl_queue, a.sycl_queue)
        assert_sycl_queue_equal(result.sycl_queue, b.sycl_queue)

    @pytest.mark.parametrize("full_matrices", [True, False])
    @pytest.mark.parametrize("compute_uv", [True, False])
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
    def test_svd(self, shape, full_matrices, compute_uv, device):
        dtype = dpnp.default_float_type(device)
        count_elems = numpy.prod(shape)
        x = dpnp.arange(count_elems, dtype=dtype, device=device).reshape(shape)

        expected_queue = x.sycl_queue
        if compute_uv:
            dpnp_u, dpnp_s, dpnp_vt = dpnp.linalg.svd(
                x, full_matrices=full_matrices, compute_uv=compute_uv
            )

            assert_sycl_queue_equal(dpnp_u.sycl_queue, expected_queue)
            assert_sycl_queue_equal(dpnp_vt.sycl_queue, expected_queue)
            assert_sycl_queue_equal(dpnp_s.sycl_queue, expected_queue)

        else:
            dpnp_s = dpnp.linalg.svd(
                x, full_matrices=full_matrices, compute_uv=compute_uv
            )
            assert_sycl_queue_equal(dpnp_s.sycl_queue, expected_queue)

    def test_tensorinv(self, device):
        a = dpnp.eye(12, device=device).reshape(12, 4, 3)
        result = dpnp.linalg.tensorinv(a, ind=1)
        assert_sycl_queue_equal(result.sycl_queue, a.sycl_queue)

    def test_tensorsolve(self, device):
        a = dpnp.random.randn(3, 2, 6, device=device)
        b = dpnp.ones(a.shape[:2], device=device)
        result = dpnp.linalg.tensorsolve(a, b)
        assert_sycl_queue_equal(result.sycl_queue, a.sycl_queue)
